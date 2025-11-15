use ndarray::{Array2, s};
use rayon::prelude::*;

use crate::{
    attention::{head::PolyHead, position::cope::CoPE},
    mixtures::{moh::HeadSelectionConfig, threshold::ThresholdPredictor},
    richards::RichardsCurve,
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
    pub gate_poly: &'a mut RichardsCurve,
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
}

/// Forward computation result containing output and metrics
#[derive(Debug)]
pub struct ForwardResult {
    pub output: Array2<f32>,
    pub tau_metrics: Option<(f32, f32)>,
    pub pred_norm: Option<f32>,
}

/// Compute polynomial attention forward pass
pub fn compute_poly_attention_forward(ctx: &mut ForwardContext, causal: bool) -> ForwardResult {
    // input: (N, embed_dim)
    let (n, d_model) = (ctx.input.nrows(), ctx.input.ncols());
    assert_eq!(d_model, ctx.embed_dim);

    // Reset cached soft top-p mask for this forward pass
    ctx.cached_soft_top_p_mask.take();

    let dk_scale = 1.0f32 / (ctx.head_dim as f32).sqrt();

    // Streamed accumulation: avoid building a large concat buffer
    let mut out = ctx.input.to_owned();

    // Pre-compute threshold predictor or soft top-p selection
    let thresholds_global = if ctx.head_selection_config.gating.use_learned_predictor {
        if let Some(predictor) = ctx.threshold_predictor {
            Some(predictor.predict(&ctx.input.view()))
        } else {
            None
        }
    } else if ctx.head_selection_config.gating.use_soft_top_p {
        // For SoftTopP, compute gating values for all heads and apply soft top-p selection
        let mut all_gates = Vec::new();

        // Compute gating values for all heads
        for h_idx in 0..ctx.num_heads {
            let w_g_col = ctx.w_g.slice(s![.., h_idx..h_idx + 1]);
            let xw_col = ctx.input.dot(&w_g_col);
            let a_h = ctx.alpha_g[[0, h_idx]];
            let b_h = ctx.beta_g[[0, h_idx]];

            let max_abs_z = xw_col
                .iter()
                .map(|&v| (a_h * v + b_h) as f64)
                .fold(0.0_f64, |m, z| m.max(z.abs()));

            let gate_poly = ctx.gate_poly.update_scaling_from_max_abs(max_abs_z);
            let g_col = xw_col.mapv(|xw| gate_poly.forward_scalar((a_h * xw + b_h) as f64) as f32);

            all_gates.push(g_col);
        }

        // Concatenate all gating values: shape (n_tokens, n_heads)
        let gate_matrix = if !all_gates.is_empty() {
            let n_tokens = all_gates[0].nrows();
            let gate_data: Vec<f32> = (0..n_tokens)
                .flat_map(|token_idx| {
                    all_gates
                        .iter()
                        .map(move |gate_col| gate_col[[token_idx, 0]])
                })
                .collect();

            ndarray::Array2::from_shape_vec((n_tokens, ctx.num_heads), gate_data)
                .unwrap_or_else(|_| ndarray::Array2::<f32>::zeros((n, ctx.num_heads)))
        } else {
            ndarray::Array2::<f32>::zeros((n, ctx.num_heads))
        };

        // Apply soft top-p selection using PadeExp and Richards activation
        let mut soft_weights = apply_soft_top_p_with_richards(
            &gate_matrix.view(),
            ctx.head_selection_config.gating.top_p,
            ctx.head_selection_config.gating.soft_top_p_alpha,
        );
        let activation_scale = ctx.head_selection_config.max_heads.max(1) as f32;
        soft_weights.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));
        *ctx.cached_soft_top_p_mask = Some(soft_weights.clone());

        Some(soft_weights)
    } else {
        None
    };

    // Zero-copy iterator-based head processing with accumulation
    let (
        _active_sums_tmp,
        _token_counts_tmp,
        (tau_min_local, tau_max_local, tau_sum_local, tau_count_local),
        (g_sq_sum_local, g_count_local),
        gate_values_acc,
        projections_acc,
    ) = ctx
        .heads
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
                .map(|&v| (a_h * v + b_h) as f64)
                .fold(0.0_f64, |m, z| m.max(z.abs()));

            let gate_poly = ctx.gate_poly.update_scaling_from_max_abs(max_abs_z);

            let g_col = xw_col.mapv(|xw| gate_poly.forward_scalar((a_h * xw + b_h) as f64) as f32);

            // RMS tracking for gating predictor
            let g_sq_sum = xw_col.iter().map(|&v| v * v).sum::<f32>();
            let g_count = n;

            // Learned threshold predictor or soft top-p selection
            let (_m_col, tau_metrics, eff_col) = if let Some(ref thresholds) = thresholds_global {
                if ctx.head_selection_config.gating.use_learned_predictor {
                    // Use learned thresholds (1D array per token)
                    let threshold_sum: f32 = thresholds.iter().sum();
                    let threshold_min = thresholds
                        .iter()
                        .fold(f32::INFINITY, |m: f32, &z: &f32| m.min(z));
                    let threshold_max = thresholds
                        .iter()
                        .fold(f32::NEG_INFINITY, |m: f32, &z: &f32| m.max(z));
                    let tau_metrics = (threshold_min, threshold_max, threshold_sum, n);
                    let eff_col = &g_col * thresholds;
                    (thresholds.clone(), tau_metrics, eff_col)
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
                    let eff_col = &g_col * &head_thresholds;
                    (head_thresholds.to_owned(), tau_metrics, eff_col)
                } else {
                    // Fallback
                    let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                    let eff_col = g_col.clone();
                    (Array2::<f32>::ones((n, 1)), tau_metrics, eff_col)
                }
            } else {
                // No learned thresholds: m = 1, so eff = g
                let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                let eff_col = g_col.clone();
                (Array2::<f32>::ones((n, 1)), tau_metrics, eff_col)
            };
            let active_sum = eff_col.sum();
            let token_count = n;

            // Return (projections, gates, metrics) for this head
            (
                (q, k, v, g_col.clone(), eff_col.clone()),
                (active_sum, token_count),
                (g_sq_sum, g_count),
                tau_metrics,
                eff_col,
            )
        })
        .fold(
            (
                vec![],
                vec![],
                (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0),
                (0.0, 0),
                vec![],
                vec![],
            ),
            |(
                mut active_acc,
                mut token_acc,
                mut tau_acc,
                mut g_acc,
                mut gate_values_acc,
                mut projections_acc,
            ),
             (
                (q, k, v, g_col, eff_col),
                (active_sum, token_count),
                (g_sq_sum, g_count),
                tau_metrics,
                gate_col,
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
                gate_values_acc.push(gate_col);
                projections_acc.push((q, k, v, g_col, eff_col));
                (
                    active_acc,
                    token_acc,
                    tau_acc,
                    g_acc,
                    gate_values_acc,
                    projections_acc,
                )
            },
        );

    // Extract projections for the attention computation loop
    let head_projections = projections_acc;

    // Create gate values array for metrics update
    // gate_values_acc contains effective gating values (g * thresholds) per head, concatenate them
    let gate_values = if !gate_values_acc.is_empty() {
        let n_tokens = gate_values_acc[0].nrows();
        let n_heads = gate_values_acc.len();

        // Use iterator chain to collect all gate values in correct order
        let gate_data: Vec<f32> = (0..n_tokens)
            .flat_map(|token_idx| {
                gate_values_acc
                    .iter()
                    .map(move |gate_col| gate_col[[token_idx, 0]])
            })
            .collect();

        ndarray::Array2::from_shape_vec((n_tokens, n_heads), gate_data)
            .unwrap_or_else(|_| ndarray::Array2::<f32>::zeros((n_tokens, n_heads)))
    } else {
        ndarray::Array2::<f32>::zeros((0, 0))
    };

    // Process attention computation for each head
    for (h_idx, (q, k, v, _g_col, eff_col)) in head_projections.into_iter().enumerate() {
        {
            let a = ctx.a[[0, 0]];
            let b = ctx.b[[0, 0]];
            let scale = ctx.scale[[0, 0]];
            let p_i32 = ctx.p as i32;
            let start = h_idx * ctx.head_dim;
            let end = start + ctx.head_dim;
            let w_block = ctx.w_out.slice(s![start..end, ..]);
            let row_buffers: Vec<(usize, Array2<f32>)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let j_start = match ctx.window_size {
                        Some(w) => i.saturating_sub(w - 1),
                        None => 0,
                    };
                    let j_end_excl = if causal { i + 1 } else { n };

                    let max_pos = usize::min(ctx.cope.max_pos, i.saturating_sub(j_start));
                    let mut q_pe = Vec::with_capacity(max_pos + 1);
                    q_pe.extend(
                        (0..=max_pos).map(|pos| q.row(i).dot(&ctx.cope.pos_embeddings.row(pos))),
                    );

                    let k_slice = k.slice(s![j_start..j_end_excl, ..]);
                    let k_slice_t = k_slice.t();
                    let scores_row = q.row(i).dot(&k_slice_t) * dk_scale;
                    let m = j_end_excl.saturating_sub(j_start);

                    let mut phi_row = ndarray::Array1::<f32>::zeros(m);
                    for idx in 0..m {
                        let j = j_start + idx;
                        let mut s_val = scores_row[idx];
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            s_val += q_pe[pos];
                        }
                        let s_clamped = s_val.clamp(-8.0, 8.0);
                        let sp = if p_i32 <= 3 {
                            match p_i32 {
                                1 => s_clamped,
                                2 => s_clamped * s_clamped,
                                3 => s_clamped * s_clamped * s_clamped,
                                _ => unreachable!(),
                            }
                        } else {
                            let mut result: f32 = 1.0;
                            let current = s_clamped;
                            for _ in 0..p_i32 {
                                result *= current;
                                if !result.is_finite() {
                                    result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                    break;
                                }
                            }
                            result
                        };
                        phi_row[idx] = scale * (a * sp + b);
                    }

                    let v_slice = v.slice(s![j_start..j_end_excl, ..]);
                    let mut row_buf = Array2::<f32>::zeros((1, ctx.head_dim));
                    ndarray::linalg::general_mat_mul(
                        1.0,
                        &phi_row.view().insert_axis(ndarray::Axis(0)),
                        &v_slice,
                        0.0,
                        &mut row_buf,
                    );
                    let eff_i = eff_col[[i, 0]];
                    row_buf.mapv_inplace(|x| x * eff_i);
                    (i, row_buf)
                })
                .collect();

            let mut y_head = Array2::<f32>::zeros((n, ctx.head_dim));
            for (i, buf) in row_buffers {
                let mut y_row = y_head.slice_mut(s![i..i + 1, ..]);
                y_row.assign(&buf);
            }
            let mut out_block = Array2::<f32>::zeros((n, ctx.embed_dim));
            ndarray::linalg::general_mat_mul(1.0, &y_head, &w_block, 0.0, &mut out_block);
            out = out + &out_block;
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

    ForwardResult {
        output: out,
        tau_metrics,
        pred_norm,
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

    // Process each token
    for (token_idx, token_gates) in gates.outer_iter().enumerate() {
        // Convert to 1D array for processing
        let token_gates_1d = token_gates.as_slice().unwrap();

        // Sort probabilities and compute cumulative sum (following AutoDeco approach)
        let mut prob_indices: Vec<usize> = (0..token_gates_1d.len()).collect();
        prob_indices.sort_by(|&i, &j| token_gates_1d[j].partial_cmp(&token_gates_1d[i]).unwrap());

        let mut sorted_probs = Vec::with_capacity(token_gates_1d.len());
        for &idx in &prob_indices {
            sorted_probs.push(token_gates_1d[idx]);
        }

        // Compute cumulative sum
        let mut cumulative = Vec::with_capacity(sorted_probs.len());
        let mut sum = 0.0;
        for &val in &sorted_probs {
            sum += val;
            cumulative.push(sum);
        }

        // Apply soft mask using Richards sigmoid for smooth activation
        // Richards sigmoid is a non-learning activation that provides smooth, well-behaved
        // gradients
        let mut soft_mask = Vec::with_capacity(cumulative.len());
        for &c in &cumulative {
            let diff = c - top_p;
            // Richards sigmoid: smooth activation with better gradient properties than standard
            // sigmoid
            let activation = smooth_sigmoid.forward_scalar((alpha * diff) as f64) as f32;

            // Add numerical stabilization: clamp the activation before exponential
            let clamped_activation = activation.clamp(-5.0, 5.0);

            // Apply PadeExp directly for numerical stability
            soft_mask.push(crate::pade::PadeExp::exp(clamped_activation as f64) as f32);
        }

        // Unsort the mask
        let mut unsorted_mask = vec![0.0; token_gates_1d.len()];
        for (i, &idx) in prob_indices.iter().enumerate() {
            unsorted_mask[idx] = soft_mask[i];
        }

        // Apply mask and renormalize
        let mut masked_probs = Vec::with_capacity(token_gates_1d.len());
        for (i, &prob) in token_gates_1d.iter().enumerate() {
            masked_probs.push(prob * unsorted_mask[i]);
        }

        let sum_masked: f32 = masked_probs.iter().sum();
        if sum_masked > 0.0 {
            for (i, prob) in masked_probs.into_iter().enumerate() {
                result[[token_idx, i]] = prob / sum_masked;
            }
        } else {
            // Fallback to original if all masked
            for (i, &prob) in token_gates_1d.iter().enumerate() {
                result[[token_idx, i]] = prob;
            }
        }
    }

    result
}
