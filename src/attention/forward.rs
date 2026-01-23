use ndarray::{Array2, s};

use crate::{
    attention::{
        head::PolyHead,
        memory::{with_tls_acc_f64, with_tls_phi, with_tls_qpe},
        position::cope::CoPE,
        utils::{smooth_clip_tanh, smooth_saturate_01},
    },
    mixtures::{moh::HeadSelectionConfig, threshold::ThresholdPredictor},
    richards::RichardsGate,
};

/// Context structure containing all data needed for forward computation
#[derive(Debug)]
pub struct ForwardContext<'a> {
    pub input: &'a Array2<f32>,
    pub heads: &'a mut [PolyHead],
    pub w_out: &'a Array2<f32>,
    pub w_g: &'a Array2<f32>,
    pub alpha_g: &'a Array2<f32>,
    pub beta_g: &'a Array2<f32>,
    pub gate: &'a mut RichardsGate,
    pub cope: &'a mut CoPE,
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
}

/// Compute polynomial attention forward pass
pub fn compute_poly_attention_forward(ctx: &mut ForwardContext, causal: bool) -> ForwardResult {
    // input: (N, embed_dim)
    let (n, d_model) = (ctx.input.nrows(), ctx.input.ncols());
    assert_eq!(d_model, ctx.embed_dim);

    // Reset cached soft top-p mask for this forward pass
    ctx.cached_soft_top_p_mask.take();
    ctx.cached_thresholds_global.take();

    let dk_scale = 1.0f32 / (ctx.head_dim as f32).sqrt();

    let mut out = ndarray::Array2::<f32>::zeros((n, ctx.embed_dim));

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
            let m = ctx.head_selection_config.threshold_modulation;
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

        let m = ctx.head_selection_config.threshold_modulation;
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

    // Zero-copy iterator-based head processing with accumulation
    let (
        _active_sums_tmp,
        _token_counts_tmp,
        (tau_min_local, tau_max_local, tau_sum_local, tau_count_local),
        (g_sq_sum_local, g_count_local),
        projections_acc,
    ) =
        ctx.heads
            .iter()
            .enumerate()
            .map(|(h_idx, head)| {
                // Project to Q, K, V using zero-copy views
                let q: Array2<f32> = ctx.input.dot(&head.w_q); // (N, d_h)
                let k: Array2<f32> = ctx.input.dot(&head.w_k); // (N, d_h)
                let v: Array2<f32> = ctx.input.dot(&head.w_v); // (N, d_h)

                // Compute per-token gating for this head: g = Richards(alpha * (X·w_g_col) + beta)
                let w_g_col = ctx.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
                let xw_col = ctx.input.dot(&w_g_col); // (N,1)
                let a_h = ctx.alpha_g[[0, h_idx]];
                let b_h = ctx.beta_g[[0, h_idx]];

                // Compute gate values and metrics using iterator chains
                let max_abs_z = xw_col
                    .iter()
                    .fold(0.0_f32, |m, &v| m.max((a_h * v + b_h).abs()));

                let gate_poly = ctx.gate.update_scaling_from_max_abs(max_abs_z as f64);

                let gate_input = xw_col.mapv(|xw| a_h * xw + b_h);
                let mut g_col = ndarray::Array2::<f32>::zeros(gate_input.raw_dim());
                gate_poly.forward_matrix_f32_into(&gate_input, &mut g_col);

                // RMS tracking for gating predictor
                let g_sq_sum = xw_col.iter().map(|&v| v * v).sum::<f32>();
                let g_count = n;

                // Learned threshold predictor or soft top-p selection
                let (tau_metrics, eff_col) = if let Some(thresholds) = thresholds_global {
                    if ctx.head_selection_config.gating.use_learned_predictor {
                        // Use learned thresholds per head (n_tokens x n_heads)
                        let head_thresholds = thresholds.slice(s![.., h_idx..h_idx + 1]);
                        let threshold_sum: f32 = head_thresholds.iter().sum();
                        let threshold_min = head_thresholds
                            .iter()
                            .fold(f32::INFINITY, |m: f32, &z: &f32| m.min(z));
                        let threshold_max = head_thresholds
                            .iter()
                            .fold(f32::NEG_INFINITY, |m: f32, &z: &f32| m.max(z));
                        let tau_metrics = (threshold_min, threshold_max, threshold_sum, n);
                        let mut eff_col = g_col;
                        ndarray::Zip::from(&mut eff_col)
                            .and(&head_thresholds)
                            .for_each(|e, &t| {
                                *e *= t;
                            });
                        (tau_metrics, eff_col)
                    } else if ctx.head_selection_config.gating.use_soft_top_p {
                        // Use soft top-p selection (2D array: n_tokens x n_heads)
                        let head_thresholds = thresholds.slice(s![.., h_idx..h_idx + 1]);
                        let threshold_sum: f32 = head_thresholds.iter().sum();
                        let threshold_min = head_thresholds
                            .iter()
                            .fold(f32::INFINITY, |m: f32, &z: &f32| m.min(z));
                        let threshold_max = head_thresholds
                            .iter()
                            .fold(f32::NEG_INFINITY, |m: f32, &z: &f32| m.max(z));
                        let tau_metrics = (threshold_min, threshold_max, threshold_sum, n);
                        let mut eff_col = g_col;
                        ndarray::Zip::from(&mut eff_col)
                            .and(&head_thresholds)
                            .for_each(|e, &t| {
                                *e *= t;
                            });
                        (tau_metrics, eff_col)
                    } else {
                        // Fallback
                        let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                        (tau_metrics, g_col)
                    }
                } else {
                    // No learned thresholds: m = 1, so eff = g
                    let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                    (tau_metrics, g_col)
                };
                let active_sum = eff_col.sum();
                let token_count = n;

                // Return (projections, gates, metrics) for this head
                (
                    (q, k, v, eff_col),
                    (active_sum, token_count),
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

    // Extract projections for the attention computation loop
    let head_projections = projections_acc;

    // Build gate values directly from the per-head eff columns (avoid storing a second copy).
    let mut gate_values = ndarray::Array2::<f32>::zeros((n, ctx.num_heads));
    for (h_idx, (_q, _k, _v, eff_col)) in head_projections.iter().enumerate() {
        for t in 0..n {
            gate_values[[t, h_idx]] = eff_col[[t, 0]];
        }
    }

    // Reuse a single head-output buffer across heads to reduce allocations.
    let mut y_head = Array2::<f32>::zeros((n, ctx.head_dim));

    // Process attention computation for each head
    for (h_idx, (q, k, v, eff_col)) in head_projections.into_iter().enumerate() {
        {
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
            // Accumulate directly into `out` to avoid allocating an intermediate block.
            ndarray::linalg::general_mat_mul(1.0, &y_head, &w_block, 1.0, &mut out);
        }
    }

    // Update gating metrics with collected gate values
    if gate_values.nrows() > 0 && gate_values.ncols() > 0 {
        ctx.head_selection_config
            .update_metrics(&gate_values.view());
    }

    // Update tau metrics from accumulated values
    let tau_metrics = if tau_count_local > 0 {
        ctx.head_selection_config.metrics_tau_min = tau_min_local;
        ctx.head_selection_config.metrics_tau_max = tau_max_local;
        ctx.head_selection_config.metrics_tau_sum = tau_sum_local;
        ctx.head_selection_config.metrics_tau_count = tau_count_local;
        Some((tau_min_local, tau_max_local))
    } else {
        None
    };

    // Update gate metrics from accumulated values
    let pred_norm = if g_count_local > 0 {
        let rms = (g_sq_sum_local / g_count_local as f32).sqrt();
        ctx.head_selection_config.metrics_g_sq_sum = g_sq_sum_local;
        ctx.head_selection_config.metrics_g_count = g_count_local;
        Some(rms)
    } else {
        None
    };

    let avg_active_heads = if gate_values.nrows() > 0 && gate_values.ncols() > 0 {
        Some(crate::mixtures::routing::compute_avg_active_components(
            &gate_values.view(),
        ))
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
    }
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
        ctx.heads
            .iter()
            .enumerate()
            .map(|(h_idx, head)| {
                let q: Array2<f32> = ctx.input.dot(&head.w_q);
                let k: Array2<f32> = ctx.input.dot(&head.w_k);
                let v: Array2<f32> = ctx.input.dot(&head.w_v);
                let w_g_col = ctx.w_g.slice(s![.., h_idx..h_idx + 1]);
                let xw_col = ctx.input.dot(&w_g_col);
                let a_h = ctx.alpha_g[[0, h_idx]];
                let b_h = ctx.beta_g[[0, h_idx]];
                let max_abs_z = xw_col
                    .iter()
                    .fold(0.0_f32, |m, &v| m.max((a_h * v + b_h).abs()));
                let gate_poly = ctx.gate.update_scaling_from_max_abs(max_abs_z as f64);
                let gate_input = xw_col.mapv(|xw| a_h * xw + b_h);
                let mut g_col = ndarray::Array2::<f32>::zeros(gate_input.raw_dim());
                gate_poly.forward_matrix_f32_into(&gate_input, &mut g_col);
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
        Some(crate::mixtures::routing::compute_avg_active_components(
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
    let smooth_sigmoid = crate::richards::RichardsCurve::sigmoid(false);

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
            soft_mask.push(crate::pade::PadeExp::exp(activation as f64) as f32);
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
