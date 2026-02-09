use ndarray::{Array2, s};

use crate::domain::{
    attention::{
        memory::{with_tls_acc_f64, with_tls_phi, with_tls_qpe},
        position::unified::UnifiedCoPE,
        utils::{smooth_clip_tanh, smooth_saturate_01},
    },
    mixtures::{moh::HeadSelectionConfig, threshold::ThresholdPredictor},
    richards::RichardsGate,
};

/// Context structure containing all data needed for forward computation
#[derive(Debug)]
pub struct ForwardContext<'a> {
    pub input: &'a Array2<f32>,
    pub w_q: &'a Array2<f32>,
    pub w_k: &'a Array2<f32>,
    pub w_v: &'a Array2<f32>,
    pub w_out: &'a Array2<f32>,
    pub w_g: &'a Array2<f32>,
    pub alpha_g: &'a Array2<f32>,
    pub beta_g: &'a Array2<f32>,
    pub gate: &'a RichardsGate,
    pub cope: &'a UnifiedCoPE,
    pub head_selection_config: &'a mut HeadSelectionConfig,
    pub threshold_predictor: &'a mut Option<ThresholdPredictor>,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub p: usize,
    pub a: &'a Array2<f32>,
    pub b: &'a Array2<f32>,
    pub scale: &'a Array2<f32>,
    pub window_size: Option<usize>,
    pub cached_soft_top_p_mask: &'a mut Option<Array2<f32>>,
    pub cached_thresholds_global: &'a mut Option<Array2<f32>>,
    pub token_threshold_scale: &'a Option<Array2<f32>>,
    pub token_latent_features: &'a Option<Array2<f32>>,
    pub eff_skip_threshold: f32,
    pub parallel_batch_size: usize,
    pub parallel_timeout_ms: u64,
    pub training_progress: f64,
}

/// Forward computation result containing output and metrics
#[derive(Debug)]
pub struct ForwardResult {
    pub output: Array2<f32>,
    pub tau_metrics: Option<(f32, f32)>,
    pub pred_norm: Option<f32>,
    pub avg_active_heads: Option<f32>,
    pub head_activity_vec: Option<Vec<f32>>,
    pub token_head_activity_vec: Option<Vec<f32>>,
    pub scores_dump: Option<Vec<ndarray::Array1<f32>>>,
}

/// Workspace for batch forward pass to avoid allocations
#[derive(Debug, Clone, Default)]
pub struct PolyAttentionBatchWorkspace {
    pub q_all: Array2<f32>,
    pub k_all: Array2<f32>,
    pub v_all: Array2<f32>,
    pub xw_all: Array2<f32>,
    pub gate_values: Array2<f32>,
    pub z_col: Array2<f32>,
    pub g_col: Array2<f32>,
    pub y_head: Array2<f32>,
}

/// Compute polynomial attention forward pass into provided output buffer using workspace
pub fn compute_poly_attention_forward_into(
    ctx: &mut ForwardContext,
    causal: bool,
    output: &mut ndarray::Array2<f32>,
    workspace: &mut PolyAttentionBatchWorkspace,
) -> ForwardResult {
    // input: (N, embed_dim)
    let (n, d_model) = (ctx.input.nrows(), ctx.input.ncols());
    assert_eq!(d_model, ctx.embed_dim);
    assert_eq!(output.nrows(), n);
    assert_eq!(output.ncols(), ctx.embed_dim);

    // Resize workspace if needed
    if workspace.q_all.shape() != [n, ctx.num_heads * ctx.head_dim] {
        workspace.q_all = Array2::zeros((n, ctx.num_heads * ctx.head_dim));
    }
    if workspace.k_all.shape() != [n, ctx.num_heads * ctx.head_dim] {
        workspace.k_all = Array2::zeros((n, ctx.num_heads * ctx.head_dim));
    }
    if workspace.v_all.shape() != [n, ctx.num_heads * ctx.head_dim] {
        workspace.v_all = Array2::zeros((n, ctx.num_heads * ctx.head_dim));
    }
    if workspace.xw_all.shape() != [n, ctx.num_heads] {
        workspace.xw_all = Array2::zeros((n, ctx.num_heads));
    }
    if workspace.gate_values.shape() != [n, ctx.num_heads] {
        workspace.gate_values = Array2::zeros((n, ctx.num_heads));
    }
    if workspace.z_col.shape() != [n, 1] {
        workspace.z_col = Array2::zeros((n, 1));
    }
    if workspace.g_col.shape() != [n, 1] {
        workspace.g_col = Array2::zeros((n, 1));
    }
    if workspace.y_head.shape() != [n, ctx.head_dim] {
        workspace.y_head = Array2::zeros((n, ctx.head_dim));
    }

    // Reset cached soft top-p mask for this forward pass
    ctx.cached_soft_top_p_mask.take();
    ctx.cached_thresholds_global.take();

    let dk_scale = 1.0f32 / (ctx.head_dim as f32).sqrt();

    // Ensure output is zeroed if we are accumulating
    output.fill(0.0);

    // Pre-compute threshold predictor or soft top-p selection
    if ctx.head_selection_config.gating.use_learned_predictor {
        if let Some(predictor) = ctx.threshold_predictor {
            // Avoid allocating a scaled copy unless per-token scaling is requested.
            let scaled_input = if let Some(scale) = ctx.token_threshold_scale.as_ref() {
                let mut tmp = ctx.input.to_owned();
                let n = tmp.nrows();
                let d = tmp.ncols();
                for i in 0..n {
                    let s = scale[[i, 0]];
                    for j in 0..d {
                        tmp[[i, j]] *= s;
                    }
                }
                Some(tmp)
            } else {
                None
            };
            let input_view = match scaled_input.as_ref() {
                Some(tmp) => tmp.view(),
                None => ctx.input.view(),
            };

            let mut t = predictor.predict_with_condition(
                &input_view,
                ctx.token_latent_features.as_ref().map(|f| f.view()),
            );
            let m = ctx.head_selection_config.threshold_modulation.value(ctx.training_progress);
            t.mapv_inplace(|v| v * m);
            let k = ctx.head_selection_config.gating.num_active as f32;
            let n = t.nrows();
            let h = t.ncols();
            for i in 0..n {
                let mut sum = 0.0f32;
                for j in 0..h {
                    sum += t[[i, j]];
                }
                if sum > 0.0 {
                    let s = k / sum;
                    for j in 0..h {
                        t[[i, j]] *= s;
                    }
                }
            }
            *ctx.cached_thresholds_global = Some(t);
        }
    } else if ctx.head_selection_config.gating.use_soft_top_p {
        // For SoftTopP, compute gating values for all heads and apply soft top-p selection
        // Build gate_matrix directly to avoid allocating Vec<Array2> + a flattened Vec.
        let mut gate_matrix = ndarray::Array2::<f32>::zeros((n, ctx.num_heads));
        let mut z_col = ndarray::Array2::<f32>::zeros((n, 1));
        let mut g_col = ndarray::Array2::<f32>::zeros((n, 1));

        for h_idx in 0..ctx.num_heads {
            let w_g_col = ctx.w_g.slice(s![.., h_idx..h_idx + 1]);
            let xw_col = ctx.input.dot(&w_g_col);
            let a_h = ctx.alpha_g[[0, h_idx]];
            let b_h = ctx.beta_g[[0, h_idx]];

            let mut max_abs_z = 0.0f32;
            for i in 0..n {
                let z = a_h * xw_col[[i, 0]] + b_h;
                z_col[[i, 0]] = z;
                max_abs_z = max_abs_z.max(z.abs());
            }

            let gate_poly = ctx.gate.update_scaling_from_max_abs(max_abs_z as f64);
            gate_poly.forward_matrix_f32_into(&z_col, &mut g_col);

            for i in 0..n {
                gate_matrix[[i, h_idx]] = g_col[[i, 0]];
            }
        }

        // Apply soft top-p selection using PadeExp and Richards activation
        let mut soft_weights = apply_soft_top_p_with_richards(
            &gate_matrix.view(),
            ctx.head_selection_config.gating.top_p,
            ctx.head_selection_config.gating.soft_top_p_alpha,
        );
        let activation_scale = ctx.head_selection_config.max_heads.max(1) as f32;
        soft_weights.mapv_inplace(|v| smooth_saturate_01(v * activation_scale));

        let m = ctx.head_selection_config.threshold_modulation.value(ctx.training_progress);
        soft_weights.mapv_inplace(|v| v * m);
        if let Some(scale) = ctx.token_threshold_scale.as_ref() {
            let n = soft_weights.nrows();
            let h = soft_weights.ncols();
            for i in 0..n {
                let s = scale[[i, 0]];
                for j in 0..h {
                    soft_weights[[i, j]] *= s;
                }
            }
        }

        // Cache the final per-token per-head selection weights that were actually used.
        // This is consumed by the backward path (PolyAttention::compute_gradients*).
        *ctx.cached_soft_top_p_mask = Some(soft_weights.clone());
    }

    let thresholds_global = ctx.cached_thresholds_global.as_ref();

    // Monolithic Projections
    // q_all: (N, H*D_h)
    ndarray::linalg::general_mat_mul(1.0, ctx.input, ctx.w_q, 0.0, &mut workspace.q_all);
    ndarray::linalg::general_mat_mul(1.0, ctx.input, ctx.w_k, 0.0, &mut workspace.k_all);
    ndarray::linalg::general_mat_mul(1.0, ctx.input, ctx.w_v, 0.0, &mut workspace.v_all);

    // Gating Projections
    // xw_all: (N, H)
    ndarray::linalg::general_mat_mul(1.0, ctx.input, ctx.w_g, 0.0, &mut workspace.xw_all);
    
    workspace.gate_values.fill(0.0);

    let mut tau_min_global = f32::INFINITY;
    let mut tau_max_global = f32::NEG_INFINITY;
    let mut tau_sum_global = 0.0;
    let mut tau_count_global = 0;
    let mut g_sq_sum_global = 0.0;
    let mut g_count_global = 0;

    for h_idx in 0..ctx.num_heads {
        let xw_col_view = workspace.xw_all.slice(s![.., h_idx..h_idx + 1]); // (N, 1)
        let a_h = ctx.alpha_g[[0, h_idx]];
        let b_h = ctx.beta_g[[0, h_idx]];

        // Reuse workspace.z_col
        workspace.z_col.assign(&xw_col_view);
        workspace.z_col.mapv_inplace(|v| a_h * v + b_h);
        
        for i in 0..n {
            let z = workspace.z_col[[i, 0]];
            // Match streaming behavior: use base curve directly without dynamic scaling
            workspace.g_col[[i, 0]] = ctx.gate.curve.forward_scalar_f32(z);
        }

        g_sq_sum_global += xw_col_view.iter().map(|&v| v * v).sum::<f32>();
        g_count_global += n;

        // Apply thresholds
        if let Some(thresholds) = thresholds_global {
            let head_thresholds = thresholds.slice(s![.., h_idx..h_idx + 1]);

            // Update metrics
            let threshold_sum: f32 = head_thresholds.iter().sum();
            let threshold_min = head_thresholds
                .iter()
                .fold(f32::INFINITY, |m, &z| m.min(z));
            let threshold_max = head_thresholds
                .iter()
                .fold(f32::NEG_INFINITY, |m, &z| m.max(z));

            tau_min_global = tau_min_global.min(threshold_min);
            tau_max_global = tau_max_global.max(threshold_max);
            tau_sum_global += threshold_sum;
            tau_count_global += n;

            if ctx.head_selection_config.gating.use_learned_predictor
                || ctx.head_selection_config.gating.use_soft_top_p
            {
                ndarray::Zip::from(&mut workspace.g_col)
                    .and(&head_thresholds)
                    .for_each(|e, &t| *e *= t);
            }
        }

        workspace.gate_values.slice_mut(s![.., h_idx..h_idx + 1]).assign(&workspace.g_col);
    }

    // Reuse a single head-output buffer across heads to reduce allocations.
    // workspace.y_head is used.

    let mut scores_dump = if cfg!(debug_assertions) {
        Some(vec![ndarray::Array1::<f32>::zeros(0); ctx.num_heads])
    } else {
        None
    };

    // Process attention computation for each head
    for h_idx in 0..ctx.num_heads {
        let start = h_idx * ctx.head_dim;
        let end = start + ctx.head_dim;

        // Zero-copy slicing from monolithic arrays in workspace
        let q = workspace.q_all.slice(s![.., start..end]);
        let k = workspace.k_all.slice(s![.., start..end]);
        let v = workspace.v_all.slice(s![.., start..end]);
        // eff_col is (N,) view of column h_idx
        let eff_col = workspace.gate_values.column(h_idx);
        {
            let a = ctx.a[[0, 0]];
            let b = ctx.b[[0, 0]];
            let scale = ctx.scale[[0, 0]];
            let p_i32 = ctx.p as i32;
            let start = h_idx * ctx.head_dim;
            let end = start + ctx.head_dim;
            let w_block = ctx.w_out.slice(s![start..end, ..]);
            workspace.y_head.fill(0.0);
            workspace.y_head
                .axis_iter_mut(ndarray::Axis(0))
                // .into_par_iter()
                .enumerate()
                .for_each(|(i, mut y_row)| {
                    let eff_i = eff_col[i];
                    if eff_i <= ctx.eff_skip_threshold {
                        return;
                    }
                    let j_start = match ctx.window_size {
                        Some(w) => i.saturating_sub(w - 1),
                        None => 0,
                    };
                    let j_end_excl = if causal { i + 1 } else { n };
                    let max_pos = usize::min(ctx.cope.max_pos, i.saturating_sub(j_start));
                    
                    // Slice Q for this head
                    let q_row_i = q.row(i);

                    // Slice K for this head
                    let k_slice = k.slice(s![j_start..j_end_excl, ..]);
                    let k_slice_t = k_slice.t();
                    let scores_row = q_row_i.dot(&k_slice_t) * dk_scale;

                    with_tls_qpe(max_pos + 1, |q_pe| {
                        for (pos, q_pe_val) in q_pe.iter_mut().enumerate() {
                            // Reverted dk_scale (Stream uses unscaled CoPE)
                            *q_pe_val = q_row_i.dot(&ctx.cope.pos_embeddings.row(pos));
                        }

                        let mlen = j_end_excl.saturating_sub(j_start);
                        with_tls_phi(mlen, |phi_row| {
                            for idx in 0..mlen {
                                let j = j_start + idx;
                                let mut s_val = scores_row[idx];

                                let pos = i.saturating_sub(j);
                                if pos < q_pe.len() {
                                    s_val += q_pe[pos];
                                }

                                let s_stable = smooth_clip_tanh(s_val, 8.0);
                                let sp = if p_i32 <= 3 {
                                    match p_i32 {
                                        1 => s_stable,
                                        2 => s_stable * s_stable,
                                        3 => s_stable * s_stable * s_stable,
                                        _ => unreachable!(),
                                    }
                                } else {
                                    // With smooth saturation, `s_stable` is bounded so this is
                                    // safe.
                                    let mut result: f32 = 1.0;
                                    for _ in 0..p_i32 {
                                        result *= s_stable;
                                    }
                                    result
                                };

                                phi_row[idx] = scale * (a * sp + b);
                            }
                            
                            if let Some(dump) = scores_dump.as_mut() {
                                // Capture scores for Last Token (i == n-1) for ALL heads
                                if i == n - 1 {
                                    let mut effective_scores = ndarray::Array1::zeros(mlen);
                                    for idx in 0..mlen {
                                        effective_scores[idx] = phi_row[idx] * eff_i;
                                    }
                                    dump[h_idx] = effective_scores;
                                }
                            }
                            
                            // Slice V for this head
                            let v_slice = v.slice(s![j_start..j_end_excl, ..]);
                            with_tls_acc_f64(ctx.head_dim, |acc| {
                                acc.fill(0.0);
                                let eff = eff_i as f64;
                                for idx in 0..mlen {
                                    let phi = (phi_row[idx] as f64) * eff;
                                    for h in 0..ctx.head_dim {
                                        acc[h] += phi * (v_slice[[idx, h]] as f64);
                                    }
                                }
                                for h in 0..ctx.head_dim {
                                    y_row[h] = acc[h] as f32;
                                }
                            });
                        });
                    });
                });
            // Accumulate directly into `output` to avoid allocating an intermediate block.
            ndarray::linalg::general_mat_mul(1.0, &workspace.y_head, &w_block, 1.0, output);
        }
    }

    // Update gating metrics with collected gate values
    if workspace.gate_values.nrows() > 0 && workspace.gate_values.ncols() > 0 {
        ctx.head_selection_config
            .update_metrics(&workspace.gate_values.view());
    }

    // Update tau metrics from accumulated values
    let tau_metrics = if tau_count_global > 0 {
        ctx.head_selection_config.metrics_tau_min = tau_min_global;
        ctx.head_selection_config.metrics_tau_max = tau_max_global;
        ctx.head_selection_config.metrics_tau_sum = tau_sum_global;
        ctx.head_selection_config.metrics_tau_count = tau_count_global;
        Some((tau_min_global, tau_max_global))
    } else {
        None
    };

    // Update gate metrics from accumulated values
    let pred_norm = if g_count_global > 0 {
        let rms = (g_sq_sum_global / g_count_global as f32).sqrt();
        ctx.head_selection_config.metrics_g_sq_sum = g_sq_sum_global;
        ctx.head_selection_config.metrics_g_count = g_count_global;
        Some(rms)
    } else {
        None
    };

    let avg_active_heads = if workspace.gate_values.nrows() > 0 && workspace.gate_values.ncols() > 0 {
        Some(crate::domain::mixtures::routing::compute_avg_active_components(
            &workspace.gate_values.view(),
        ))
    } else {
        None
    };

    let (head_activity_vec, token_head_activity_vec) =
        if workspace.gate_values.nrows() > 0 && workspace.gate_values.ncols() > 0 {
            let n = workspace.gate_values.nrows();
            let h = workspace.gate_values.ncols();
            let mut head_v = vec![0.0f32; h];
            let inv_n = 1.0 / (n as f32);
            for head in 0..h {
                let mut sum = 0.0f32;
                for tok in 0..n {
                    sum += workspace.gate_values[[tok, head]];
                }
                head_v[head] = (sum * inv_n).clamp(0.0, 1.0);
            }

            let mut tok_v = vec![0.0f32; n];
            let inv_h = 1.0 / (h as f32);
            for tok in 0..n {
                let mut sum = 0.0f32;
                for head in 0..h {
                    sum += workspace.gate_values[[tok, head]];
                }
                tok_v[tok] = (sum * inv_h).clamp(0.0, 1.0);
            }

            (Some(head_v), Some(tok_v))
        } else {
            (None, None)
        };

    ForwardResult {
        output: Array2::zeros((0, 0)),
        tau_metrics,
        pred_norm,
        avg_active_heads,
        head_activity_vec,
        token_head_activity_vec,
        scores_dump,
    }
}

/// Compute polynomial attention forward pass
pub fn compute_poly_attention_forward(ctx: &mut ForwardContext, causal: bool) -> ForwardResult {
    let (n, d_model) = (ctx.input.nrows(), ctx.input.ncols());
    let mut output = Array2::<f32>::zeros((n, d_model));
    let mut workspace = PolyAttentionBatchWorkspace::default();
    let mut res = compute_poly_attention_forward_into(ctx, causal, &mut output, &mut workspace);
    res.output = output;
    res
}

pub fn compute_poly_attention_forward_baseline(
    ctx: &mut ForwardContext,
    causal: bool,
) -> ForwardResult {
    let (n, d_model) = (ctx.input.nrows(), ctx.input.ncols());
    assert_eq!(d_model, ctx.embed_dim);
    ctx.cached_soft_top_p_mask.take();
    ctx.cached_thresholds_global.take();
    let dk_scale = 1.0f32 / (ctx.head_dim as f32).sqrt();
    let mut out = ndarray::Array2::<f32>::zeros((n, ctx.embed_dim));

    let _thresholds_global: Option<ndarray::Array2<f32>> = None;

    let (
        _a_s,
        _t_c,
        (tau_min_local, tau_max_local, tau_sum_local, tau_count_local),
        (g_sq_sum_local, g_count_local),
        projections_acc,
    ) =
        (0..ctx.num_heads)
            .into_iter()
            .map(|h_idx| {
                let start = h_idx * ctx.head_dim;
                let end = start + ctx.head_dim;
                let q: Array2<f32> = ctx.input.dot(&ctx.w_q.slice(s![.., start..end]));
                let k: Array2<f32> = ctx.input.dot(&ctx.w_k.slice(s![.., start..end]));
                let v: Array2<f32> = ctx.input.dot(&ctx.w_v.slice(s![.., start..end]));
                let w_g_col = ctx.w_g.slice(s![.., h_idx..h_idx + 1]);
                let xw_col = ctx.input.dot(&w_g_col);
                let a_h = ctx.alpha_g[[0, h_idx]];
                let b_h = ctx.beta_g[[0, h_idx]];
                let mut g_col = ndarray::Array2::<f32>::zeros(xw_col.raw_dim());
                for (i, &xw) in xw_col.iter().enumerate() {
                    let z = a_h * xw + b_h;
                    // Match streaming behavior: use base curve directly without dynamic scaling
                    g_col[[i, 0]] = ctx.gate.curve.forward_scalar_f32(z);
                }
                let g_sq_sum = xw_col.iter().map(|&v| v * v).sum::<f32>();
                let g_count = n;
                let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                let eff_col = g_col;
                let active_sum = eff_col.sum();
                (
                    (q, k, v, eff_col),
                    (active_sum, n),
                    (g_sq_sum, g_count),
                    tau_metrics,
                )
            })
            .fold(
                (
                    vec![],
                    vec![],
                    (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0),
                    (0.0, 0),
                    vec![],
                ),
                |(mut active_acc, mut token_acc, mut tau_acc, mut g_acc, mut projections_acc),
                 (
                    (q, k, v, eff_col),
                    (active_sum, token_count),
                    (g_sq_sum, g_count),
                    tau_metrics,
                )| {
                    active_acc.push(active_sum);
                    token_acc.push(token_count);
                    tau_acc = (
                        tau_acc.0.min(tau_metrics.0),
                        tau_acc.1.max(tau_metrics.1),
                        tau_acc.2 + tau_metrics.2,
                        tau_acc.3 + tau_metrics.3,
                    );
                    g_acc = (g_acc.0 + g_sq_sum, g_acc.1 + g_count);
                    projections_acc.push((q, k, v, eff_col));
                    (active_acc, token_acc, tau_acc, g_acc, projections_acc)
                },
            );

    let mut gate_values = ndarray::Array2::<f32>::zeros((n, ctx.num_heads));
    for (h_idx, (_q, _k, _v, eff_col)) in projections_acc.iter().enumerate() {
        for t in 0..n {
            gate_values[[t, h_idx]] = eff_col[[t, 0]];
        }
    }

    // Reuse a single head-output buffer across heads (avoids allocating N small row buffers).
    let mut y_head = Array2::<f32>::zeros((n, ctx.head_dim));

    for (h_idx, (q, k, v, eff_col)) in projections_acc.into_iter().enumerate() {
        let a = ctx.a[[0, 0]];
        let b = ctx.b[[0, 0]];
        let scale = ctx.scale[[0, 0]];
        let p_i32 = ctx.p as i32;
        let start = h_idx * ctx.head_dim;
        let end = start + ctx.head_dim;
        let w_block = ctx.w_out.slice(s![start..end, ..]);

        y_head.fill(0.0);
        use rayon::prelude::*;
        y_head
            .axis_iter_mut(ndarray::Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut y_row)| {
                let eff_i = eff_col[[i, 0]];
                if eff_i <= ctx.eff_skip_threshold {
                    return;
                }
                let j_start = match ctx.window_size {
                    Some(w) => i.saturating_sub(w - 1),
                    None => 0,
                };
                let j_end_excl = if causal { i + 1 } else { n };
                let max_pos = usize::min(ctx.cope.max_pos, i.saturating_sub(j_start));
                let q_row_i = q.row(i);
                with_tls_qpe(max_pos + 1, |q_pe| {
                    for (pos, q_pe_val) in q_pe.iter_mut().enumerate() {
                        *q_pe_val = q_row_i.dot(&ctx.cope.pos_embeddings.row(pos));
                    }

                    let k_slice = k.slice(s![j_start..j_end_excl, ..]);
                    let k_slice_t = k_slice.t();
                    let scores_row = q_row_i.dot(&k_slice_t) * dk_scale;
                    let mlen = j_end_excl.saturating_sub(j_start);
                    with_tls_phi(mlen, |phi_row| {
                        for idx in 0..mlen {
                            let j = j_start + idx;
                            let mut s_val = scores_row[idx];
                            let pos = i.saturating_sub(j);
                            if pos < q_pe.len() {
                                s_val += q_pe[pos];
                            }
                            let s_stable = smooth_clip_tanh(s_val, 8.0);
                            let sp = if p_i32 <= 3 {
                                match p_i32 {
                                    1 => s_stable,
                                    2 => s_stable * s_stable,
                                    3 => s_stable * s_stable * s_stable,
                                    _ => unreachable!(),
                                }
                            } else {
                                let mut result: f32 = 1.0;
                                for _ in 0..p_i32 {
                                    result *= s_stable;
                                }
                                result
                            };
                            phi_row[idx] = scale * (a * sp + b);
                        }

                        let v_slice = v.slice(s![j_start..j_end_excl, ..]);
                        with_tls_acc_f64(ctx.head_dim, |acc| {
                            acc.fill(0.0);
                            let eff = eff_i as f64;
                            for idx in 0..mlen {
                                let phi = (phi_row[idx] as f64) * eff;
                                for h in 0..ctx.head_dim {
                                    acc[h] += phi * (v_slice[[idx, h]] as f64);
                                }
                            }
                            for h in 0..ctx.head_dim {
                                y_row[h] = acc[h] as f32;
                            }
                        });
                    });
                });
            });

        ndarray::linalg::general_mat_mul(1.0, &y_head, &w_block, 1.0, &mut out);
    }

    let avg_active_heads = if gate_values.nrows() > 0 && gate_values.ncols() > 0 {
        ctx.head_selection_config
            .update_metrics(&gate_values.view());
        Some(crate::domain::mixtures::routing::compute_avg_active_components(
            &gate_values.view(),
        ))
    } else {
        None
    };
    let tau_metrics = if tau_count_local > 0 {
        ctx.head_selection_config.metrics_tau_min = tau_min_local;
        ctx.head_selection_config.metrics_tau_max = tau_max_local;
        ctx.head_selection_config.metrics_tau_sum = tau_sum_local;
        ctx.head_selection_config.metrics_tau_count = tau_count_local;
        Some((tau_min_local, tau_max_local))
    } else {
        None
    };
    let pred_norm = if g_count_local > 0 {
        let rms = (g_sq_sum_local / g_count_local as f32).sqrt();
        ctx.head_selection_config.metrics_g_sq_sum = g_sq_sum_local;
        ctx.head_selection_config.metrics_g_count = g_count_local;
        Some(rms)
    } else {
        None
    };
    let (head_activity_vec, token_head_activity_vec) =
        if gate_values.nrows() > 0 && gate_values.ncols() > 0 {
            let n = gate_values.nrows();
            let h = gate_values.ncols();
            let mut head_v = vec![0.0f32; h];
            let inv_n = 1.0 / (n as f32);
            for head in 0..h {
                let mut sum = 0.0f32;
                for tok in 0..n {
                    sum += gate_values[[tok, head]];
                }
                head_v[head] = (sum * inv_n).clamp(0.0, 1.0);
            }

            let mut tok_v = vec![0.0f32; n];
            let inv_h = 1.0 / (h as f32);
            for tok in 0..n {
                let mut sum = 0.0f32;
                for head in 0..h {
                    sum += gate_values[[tok, head]];
                }
                tok_v[tok] = (sum * inv_h).clamp(0.0, 1.0);
            }

            (Some(head_v), Some(tok_v))
        } else {
            (None, None)
        };

    ForwardResult {
        output: out,
        tau_metrics,
        pred_norm,
        avg_active_heads,
        head_activity_vec,
        token_head_activity_vec,
        scores_dump: None,
    }
}

/// Apply soft top-p selection using Richards sigmoid for smooth activation
/// Returns differentiable probability distribution for head selection
fn apply_soft_top_p_with_richards(
    gates: &ndarray::ArrayView2<f32>,
    top_p: f32,
    alpha: f32,
) -> ndarray::Array2<f32> {
    let mut result = ndarray::Array2::<f32>::zeros(gates.raw_dim());

    // Use non-learning Richards sigmoid for smooth activation
    let smooth_sigmoid = crate::domain::richards::RichardsCurve::sigmoid(false);

    // Reuse per-token scratch buffers to reduce allocation churn.
    let mut prob_indices: Vec<usize> = Vec::new();
    let mut soft_mask: Vec<f32> = Vec::new();
    let mut unsorted_mask: Vec<f32> = Vec::new();

    // Process each token
    for (token_idx, token_gates) in gates.outer_iter().enumerate() {
        // SoftTopP is defined over probabilities; normalize per token to make `top_p`
        // meaningful even when gate magnitudes drift.
        let mut sum_probs = 0.0f32;
        for &v in token_gates.iter() {
            if v.is_finite() && v > 0.0 {
                sum_probs += v;
            }
        }
        let inv_sum_probs = if sum_probs.is_finite() && sum_probs > 0.0 {
            1.0f32 / sum_probs
        } else {
            0.0f32
        };

        // Sort probabilities and compute cumulative sum (following AutoDeco approach)
        let token_len = token_gates.len();
        prob_indices.clear();
        prob_indices.extend(0..token_len);
        prob_indices.sort_by(|&i, &j| {
            let a = token_gates[i];
            let b = token_gates[j];
            // Treat NaNs as very small so they sink to the end.
            let a = if a.is_finite() { a } else { f32::NEG_INFINITY };
            let b = if b.is_finite() { b } else { f32::NEG_INFINITY };
            b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply soft mask using Richards sigmoid for smooth activation
        // Richards sigmoid is a non-learning activation that provides smooth, well-behaved
        // gradients
        soft_mask.clear();
        soft_mask.reserve(token_len);
        let mut cum = 0.0f32;
        for &idx in &prob_indices {
            let p = if inv_sum_probs > 0.0 {
                let v = token_gates[idx];
                if v.is_finite() && v > 0.0 {
                    v * inv_sum_probs
                } else {
                    0.0
                }
            } else {
                0.0
            };

            cum += p;
            let diff = cum - top_p;
            // Richards sigmoid: smooth activation with better gradient properties than standard
            // sigmoid
            let activation = smooth_sigmoid.forward_scalar_f32(alpha * diff);

            // Apply PadeExp directly for numerical stability
            soft_mask.push(crate::domain::pade::PadeExp::exp(activation as f64) as f32);
        }

        // Unsort the mask
        unsorted_mask.clear();
        unsorted_mask.resize(token_len, 0.0);
        for (i, &idx) in prob_indices.iter().enumerate() {
            unsorted_mask[idx] = soft_mask[i];
        }

        // Apply mask directly into the output row and renormalize.
        let mut sum_masked: f32 = 0.0;
        for (i, &prob_raw) in token_gates.iter().enumerate() {
            let prob = if inv_sum_probs > 0.0 {
                if prob_raw.is_finite() && prob_raw > 0.0 {
                    prob_raw * inv_sum_probs
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let v = prob * unsorted_mask[i];
            result[[token_idx, i]] = v;
            sum_masked += v;
        }
        if sum_masked > 0.0 {
            let inv = 1.0f32 / sum_masked;
            for i in 0..token_len {
                result[[token_idx, i]] *= inv;
            }
        } else {
            // Fallback: use normalized gates (or all zeros if degenerate)
            for (i, &prob_raw) in token_gates.iter().enumerate() {
                let prob = if inv_sum_probs > 0.0 {
                    if prob_raw.is_finite() && prob_raw > 0.0 {
                        prob_raw * inv_sum_probs
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                result[[token_idx, i]] = prob;
            }
        }
    }

    result
}
