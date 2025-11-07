use std::{cell::RefCell, thread_local};

use ndarray::{Array2, linalg::general_mat_mul, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{MAX_SEQ_LEN, adam::Adam, llm::Layer, richards::{RichardsCurve, Variant}, mixtures::{moh::{HeadSelectionStrategy, HeadSelectionConfig}, threshold::ThresholdPredictor}, attention::position::cope::CoPE};

/// Cached parameter information for PolyAttention
#[derive(Debug, Clone)]
pub struct PolyAttentionParamInfo {
    /// Parameter count per head (w_q, w_k, w_v)
    pub head_params_per_head: usize,
    /// Total head parameters (all heads)
    pub head_params_total: usize,
    /// Output projection parameters
    pub output_projection_params: usize,
    /// Polynomial parameters (a, b, scale)
    pub polynomial_params: usize,
    /// Gating parameters (w_g, alpha_g, beta_g)
    pub gating_params: usize,
    /// Richards curve parameters for gating
    pub gate_poly_params: usize,
    /// Threshold predictor parameters (if present)
    pub threshold_predictor_params: usize,
    /// CoPE parameters
    pub cope_params: usize,
    /// Total parameter count
    pub total_params: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolyHead {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,

    opt_w_q: Adam,
    opt_w_k: Adam,
    opt_w_v: Adam,
}

impl PolyHead {
    fn new(embed_dim: usize, head_dim: usize) -> Self {
        let std_qk = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();
        let std_v = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();

        let mut rng = rand::rng();
        let normal_qk = Normal::new(0.0, std_qk).unwrap();
        let normal_v = Normal::new(0.0, std_v).unwrap();

        let w_q =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_qk.sample(&mut rng));
        let w_k =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_qk.sample(&mut rng));
        let w_v =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_v.sample(&mut rng));

        let opt_w_q = Adam::new((embed_dim, head_dim));
        let opt_w_k = Adam::new((embed_dim, head_dim));
        let opt_w_v = Adam::new((embed_dim, head_dim));

        Self {
            w_q,
            w_k,
            w_v,
            opt_w_q,
            opt_w_k,
            opt_w_v,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolyAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    pub heads: Vec<PolyHead>,

    pub w_out: Array2<f32>,
    opt_w_out: Adam,

    // polynomial parameters (scalars, stored as 1x1 arrays for optimizer compatibility)
    pub p: usize,
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub scale: Array2<f32>,
    opt_a: Adam,
    opt_b: Adam,
    opt_scale: Adam,

    // ===== Adaptive Mixture-of-Heads gating (learned, fully adaptive) =====
    // Per-head gating projection and learned Richards curve gate: g = Richards(alpha * (X·W_g) + beta)
    pub w_g: Array2<f32>,     // (embed_dim, num_heads)
    pub alpha_g: Array2<f32>, // (1, num_heads)
    pub beta_g: Array2<f32>,  // (1, num_heads)
    opt_w_g: Adam,
    opt_alpha_g: Adam,
    opt_beta_g: Adam,

    // Learnable Richards curve for gating
    pub gate_poly: RichardsCurve,

    // ===== Mixture of Heads (MoH) components =====
    /// Head selection configuration and metrics
    pub head_selection_config: HeadSelectionConfig,
    /// Learned head selection predictor for dynamic head selection (AutoDeco-inspired)
    pub threshold_predictor: Option<ThresholdPredictor>,
    /// Optimizer for threshold predictor weights1
    opt_w_tau: Option<Adam>,
    /// Optimizer for threshold predictor bias1
    opt_b_tau: Option<Adam>,
    /// Optimizer for threshold predictor weights2
    opt_w2_tau: Option<Adam>,
    /// Optimizer for threshold predictor bias2
    opt_b2_tau: Option<Adam>,

    // CoPE integration and sliding window
    cope: CoPE,
    window_size: Option<usize>,

    // training cache
    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>, // (N, embed_dim)

    /// Cached parameter information for dynamic tracking
    #[serde(skip)]
    param_info: Option<PolyAttentionParamInfo>,
}

// Thread-local scratch to avoid allocations per call and avoid locking overhead
thread_local! {
    static TLS_SCORES: RefCell<Option<Array2<f32>>> = RefCell::new(None); // (N, N)
    static TLS_WORK:   RefCell<Option<Array2<f32>>> = RefCell::new(None); // (N, N)
    static TLS_YH:     RefCell<Option<Array2<f32>>> = RefCell::new(None); // (N, d_h)
}

#[inline]
fn with_tls_scores<R>(n: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_SCORES.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, n],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, n)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}

#[inline]
fn with_tls_work<R>(n: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_WORK.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, n],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, n)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}

#[inline]
fn with_tls_yh<R>(n: usize, d: usize, f: impl FnOnce(&mut Array2<f32>) -> R) -> R {
    TLS_YH.with(|cell| {
        let mut opt = cell.borrow_mut();
        let need = match &*opt {
            Some(a) => a.shape() != [n, d],
            None => true,
        };
        if need {
            *opt = Some(Array2::<f32>::zeros((n, d)));
        }
        let mat = opt.as_mut().unwrap();
        f(mat)
    })
}


impl PolyAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        p: usize,
        max_pos: usize,
        window_size: Option<usize>,
    ) -> Self {
        assert!(
            num_heads > 0 && embed_dim % num_heads == 0,
            "embed_dim must be divisible by num_heads"
        );
        assert!(p % 2 == 1, "p must be an odd integer for stability");
        let head_dim = embed_dim / num_heads;

        // Initialize heads
        let heads = (0..num_heads)
            .map(|_| PolyHead::new(embed_dim, head_dim))
            .collect::<Vec<_>>();

        // Output projection (concat heads -> embed_dim)
        let mut rng = rand::rng();
        let std_out = (2.0f32 / (embed_dim as f32 + embed_dim as f32)).sqrt();
        let normal_out = Normal::new(0.0, std_out).unwrap();
        let w_out =
            Array2::<f32>::from_shape_fn((embed_dim, embed_dim), |_| normal_out.sample(&mut rng));
        let opt_w_out = Adam::new((embed_dim, embed_dim));

        // Polynomial scalars
        let a = Array2::<f32>::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let b = Array2::<f32>::from_shape_vec((1, 1), vec![0.0]).unwrap();
        let scale =
            Array2::<f32>::from_shape_vec((1, 1), vec![1.0 / (MAX_SEQ_LEN as f32).sqrt()]).unwrap();
        let opt_a = Adam::new((1, 1));
        let opt_b = Adam::new((1, 1));
        let opt_scale = Adam::new((1, 1));

        // Learned gating params: W_g (D,H), alpha_g (1,H), beta_g (1,H)
        let std_g = (2.0f32 / embed_dim as f32).sqrt();
        let normal_g = Normal::new(0.0, std_g).unwrap();
        let w_g =
            Array2::<f32>::from_shape_fn((embed_dim, num_heads), |_| normal_g.sample(&mut rng));
        let alpha_g = Array2::<f32>::ones((1, num_heads));
        let beta_g = Array2::<f32>::zeros((1, num_heads));
        let opt_w_g = Adam::new((embed_dim, num_heads));
        let opt_alpha_g = Adam::new((1, num_heads));
        let opt_beta_g = Adam::new((1, num_heads));

        // CoPE integration (shared pos embeddings across heads)
        let cope = CoPE::new(max_pos, head_dim);

        // Richards curve gate (default sigmoid variant, learnable)
        let gate_poly = RichardsCurve::new_learnable(Variant::Sigmoid);

        // Threshold predictor defaults are handled in HeadSelectionConfig

        Self {
            embed_dim,
            num_heads,
            head_dim,
            heads,
            w_out,
            opt_w_out,
            p,
            a,
            b,
            scale,
            opt_a,
            opt_b,
            opt_scale,
            w_g,
            alpha_g,
            beta_g,
            opt_w_g,
            opt_alpha_g,
            opt_beta_g,
            gate_poly,
            head_selection_config: HeadSelectionConfig {
                gating: crate::mixtures::gating::GatingConfig::default(),
                min_heads: 1,
                max_heads: num_heads,
                metrics_tau_min: f32::INFINITY,
                metrics_tau_max: f32::NEG_INFINITY,
                metrics_tau_sum: 0.0,
                metrics_tau_count: 0,
                metrics_g_sq_sum: 0.0,
                metrics_g_count: 0,
            },
            threshold_predictor: None,
            opt_w_tau: None,
            opt_b_tau: None,
            opt_w2_tau: None,
            opt_b2_tau: None,
            cope,
            window_size,
            cached_input: None,
            param_info: None,
        }
    }

    #[inline]
    fn apply_causal_mask_inplace(mat: &mut Array2<f32>) {
        let n = mat.nrows();
        for i in 0..n {
            for j in (i + 1)..n {
                mat[[i, j]] = 0.0;
            }
        }
    }

    #[inline]
    fn apply_sliding_window_mask_inplace(mat: &mut Array2<f32>, window: Option<usize>) {
        if let Some(w) = window {
            let n = mat.nrows();
            for i in 0..n {
                let j_min = i.saturating_sub(w - 1);
                for j in 0..j_min {
                    mat[[i, j]] = 0.0;
                }
            }
        }
    }


    pub fn forward_impl(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        // input: (N, embed_dim)
        let (n, d_model) = (input.nrows(), input.ncols());
        assert_eq!(d_model, self.embed_dim);

        self.cached_input = Some(input.clone());

        let dk_scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // Streamed accumulation: avoid building a large concat buffer
        let mut out = input.to_owned();

        if self.head_selection_config.gating.use_learned_predictor {
            self.ensure_threshold_predictor();
        }

        // Pre-compute threshold predictor for all heads if needed
        let thresholds_global = if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &mut self.threshold_predictor {
                Some(predictor.predict(&input.view()))
            } else {
                None
            }
        } else {
            None
        };

        // Zero-copy iterator-based head processing with accumulation
        let (active_sums_tmp, token_counts_tmp, (tau_min_local, tau_max_local, tau_sum_local, tau_count_local), (g_sq_sum_local, g_count_local), gate_values_acc, projections_acc) =
            self.heads.iter().enumerate()
            .map(|(h_idx, head)| {
                // Project to Q, K, V using zero-copy views
                let q: Array2<f32> = input.dot(&head.w_q); // (N, d_h)
                let k: Array2<f32> = input.dot(&head.w_k); // (N, d_h)
                let v: Array2<f32> = input.dot(&head.w_v); // (N, d_h)

                // Compute per-token gating for this head: g = Richards(alpha * (X·w_g_col) + beta)
                let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
                let xw_col = input.dot(&w_g_col); // (N,1)
                let a_h = self.alpha_g[[0, h_idx]];
                let b_h = self.beta_g[[0, h_idx]];

                // Compute gate values and metrics using iterator chains
                let max_abs_z = xw_col.iter()
                    .map(|&v| (a_h * v + b_h) as f64)
                    .fold(0.0_f64, |m, z| m.max(z.abs()));

                let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);

                let g_col = xw_col.mapv(|xw| gate_poly.forward_scalar((a_h * xw + b_h) as f64) as f32);

                // RMS tracking for gating predictor
                let g_sq_sum = xw_col.iter().map(|&v| v * v).sum::<f32>();
                let g_count = n;

                // Learned threshold predictor m = sigmoid(alpha_tau * (X·W_tau) + beta_tau)
                let (_m_col, tau_metrics, eff_col) = if let Some(ref thresholds) = thresholds_global {
                    // Use learned thresholds
                    let threshold_sum: f32 = thresholds.iter().sum();
                    let threshold_min = thresholds.iter().fold(f32::INFINITY, |m: f32, &z: &f32| m.min(z));
                    let threshold_max = thresholds.iter().fold(f32::NEG_INFINITY, |m: f32, &z: &f32| m.max(z));
                    let tau_metrics = (threshold_min, threshold_max, threshold_sum, n);
                    let eff_col = &g_col * thresholds;
                    (thresholds.clone(), tau_metrics, eff_col)
                } else {
                    // No learned thresholds: m = 1, so eff = g
                    let tau_metrics = (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0);
                    let eff_col = g_col.clone();
                    (Array2::<f32>::ones((n, 1)), tau_metrics, eff_col)
                };
                let active_sum = eff_col.sum();
                let token_count = n;

                // Return (projections, gates, metrics) for this head
                ((q, k, v, g_col.clone(), eff_col.clone()), (active_sum, token_count), (g_sq_sum, g_count), tau_metrics, eff_col)
            })
            .fold(
                (vec![], vec![], (f32::INFINITY, f32::NEG_INFINITY, 0.0, 0), (0.0, 0), vec![], vec![]),
                |(mut active_acc, mut token_acc, mut tau_acc, mut g_acc, mut gate_values_acc, mut projections_acc),
                 ((q, k, v, g_col, eff_col), (active_sum, token_count), (g_sq_sum, g_count), tau_metrics, gate_col)| {
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
                    (active_acc, token_acc, tau_acc, g_acc, gate_values_acc, projections_acc)
                }
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
                    gate_values_acc.iter().map(move |gate_col| gate_col[[token_idx, 0]])
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
                // True banded computation per row (avoids building N×N scores)
                let a = self.a[[0, 0]];
                let b = self.b[[0, 0]];
                let scale = self.scale[[0, 0]];
                let p_i32 = self.p as i32;
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]); // (d_h, D)

                for i in 0..n {
                    let mut yh_row = Array2::<f32>::zeros((1, self.head_dim));
                    let j_start = match self.window_size { Some(w) => i.saturating_sub(w - 1), None => 0 };
                    let j_end = if causal { i } else { n - 1 };

                    // CoPE q·p_pos caching for row i
                    let max_pos = usize::min(self.cope.max_pos, i.saturating_sub(j_start));
                    let mut q_pe = Vec::with_capacity(max_pos + 1);
                    q_pe.extend((0..=max_pos).map(|pos|
                        q.row(i).dot(&self.cope.pos_embeddings.row(pos))
                    ));

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() { s += q_pe[pos]; }
                        let sp = match p_i32 { 1 => s, 2 => s * s, 3 => s * s * s, _ => s.powi(p_i32) };
                        let phi = scale * (a * sp + b);
                        // yh_row += phi * v[j,:]
                        for h in 0..self.head_dim {
                            yh_row[[0, h]] += phi * v[[j, h]];
                        }
                    }

                    // Apply gating: eff = g * m for token i (precomputed in eff_col)
                    let eff_i = eff_col[[i, 0]];
                    for h in 0..self.head_dim {
                        yh_row[[0, h]] *= eff_i;
                    }

                    // Accumulate into output row i via W_out block
                    let mut out_row = out.slice_mut(s![i..i + 1, ..]);
                    general_mat_mul(1.0, &yh_row, &w_block, 1.0, &mut out_row);
                }
            }
        }

        // Update gating metrics with collected gate values
        if gate_values.nrows() > 0 && gate_values.ncols() > 0 {
            self.head_selection_config.update_metrics(&gate_values.view());
        }

        // Update tau metrics from accumulated values
        if tau_count_local > 0 { // tau_count > 0
            self.head_selection_config.metrics_tau_min = tau_min_local;
            self.head_selection_config.metrics_tau_max = tau_max_local;
            self.head_selection_config.metrics_tau_sum = tau_sum_local;
            self.head_selection_config.metrics_tau_count = tau_count_local;
        }

        // Update gate metrics from accumulated values
        self.head_selection_config.metrics_g_sq_sum = g_sq_sum_local;
        self.head_selection_config.metrics_g_count = g_count_local;

        out
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let (n, _d_model) = (input.nrows(), input.ncols());
        let dk_scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // dL/dX accumulates residual path (+) and projections back from Q,K,V and gating
        let mut grad_input_total = output_grads.clone(); // residual path

        // Scalar grads accumulators for polynomial params
        let mut grad_a_scalar: f32 = 0.0;
        let mut grad_b_scalar: f32 = 0.0;
        let mut grad_scale_scalar: f32 = 0.0;

        // Gating param grads accumulators
        let mut grad_w_g = Array2::<f32>::zeros((self.embed_dim, self.num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, self.num_heads));
        // Gate polynomial coefficient gradient accumulator (shared across heads)
        let n_gate_w = self.gate_poly.weights().len();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

        // Threshold predictor grads - simplified for now
        let (grad_w_tau, grad_b_tau, grad_w2_tau, grad_b2_tau, grad_activation_tau): (Option<Array2<f32>>, Option<Array2<f32>>, Option<Array2<f32>>, Option<Array2<f32>>, Option<Vec<f64>>) = if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &self.threshold_predictor {
                // For now, use zero gradients as placeholder
                // TODO: Implement proper threshold gradient computation with real loss
                let hidden_dim = predictor.weights1.ncols();
                (Some(Array2::<f32>::zeros((self.embed_dim, hidden_dim))),  // weights1 shape
                 Some(Array2::<f32>::zeros((hidden_dim, 1))),              // bias1 shape (matches optimizer)
                 Some(Array2::<f32>::zeros((hidden_dim, 1))),              // weights2 shape
                 Some(Array2::<f32>::zeros((1, 1))),                       // bias2 shape (matches optimizer)
                 Some(vec![0.0_f64; predictor.activation.scalar_weights_len()]))
            } else {
                (None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None)
        };

        // CoPE grads accumulator (shared across heads)
        let mut grad_cope_pos = Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));

        // Per-head param grads (Wq, Wk, Wv) + W_out + scalars + gating params
        let mut all_param_grads: Vec<Array2<f32>> = Vec::new();

        // Build grad for W_out block-wise to avoid materializing H
        let mut grad_w_out = Array2::<f32>::zeros((self.embed_dim, self.embed_dim)); // (D, D)

        let a = self.a[[0, 0]];
        let b = self.b[[0, 0]];
        let scale = self.scale[[0, 0]];
        let p_i32 = self.p as i32;
        let _p_f = self.p as f32;
        for (h_idx, head) in self.heads.iter().enumerate() {
            // Recompute per-head Q, K, V and intermediates
            let q: Array2<f32> = input.dot(&head.w_q); // (N, d_h)
            let k: Array2<f32> = input.dot(&head.w_k); // (N, d_h)
            let v: Array2<f32> = input.dot(&head.w_v); // (N, d_h)

            // Gating forward values for this head (and caches for backward)
            let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
            let xw_col = input.dot(&w_g_col); // (N,1)
            let a_h = self.alpha_g[[0, h_idx]];
            let b_h = self.beta_g[[0, h_idx]];
            // z = a_h * xw + b_h; g = Richards(z)
            let mut z_col = xw_col.clone();
            z_col.mapv_inplace(|v| a_h * v + b_h);
            let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
            let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
            let mut g_col = z_col.clone();
            g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);

            // Threshold path forward
            let mut m_col = Array2::<f32>::ones((n, 1));
            if self.head_selection_config.gating.use_learned_predictor {
                if let Some(predictor) = &self.threshold_predictor {
                    // Use the enhanced AutoDeco-inspired predictor
                    let thresholds = predictor.forward(&input.view());
                    m_col.assign(&thresholds);
                }
            }

            {
                // True banded backward: per-row computations within the window
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]);
                let w_block_t = w_block.t();

                // Allocate per-head grads
                let mut grad_q: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_k: Array2::<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_v: Array2::<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_p_local: Array2<f32> = Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));

                for i in 0..n {
                    // g_yh_gated_row from output_grads and W_out block
                    let out_row = output_grads.slice(s![i..i + 1, ..]);
                    let mut g_yh_gated_row = Array2::<f32>::zeros((1, self.head_dim));
                    general_mat_mul(1.0, &out_row, &w_block_t, 0.0, &mut g_yh_gated_row);

                    // Recompute y_pre_row (pre-gating) via banded phi(S) * V
                    let mut y_pre_row = Array2::<f32>::zeros((1, self.head_dim));
                    let j_start = match self.window_size { Some(w) => i.saturating_sub(w - 1), None => 0 };
                    let j_end = i; // causal always true here

                    // CoPE q·p_pos caching for row i
                    let max_pos = usize::min(self.cope.max_pos, i.saturating_sub(j_start));
                    let mut q_pe = vec![0.0f32; max_pos + 1];
                    for pos in 0..=max_pos {
                        q_pe[pos] = q.row(i).dot(&self.cope.pos_embeddings.row(pos));
                    }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() { s += q_pe[pos]; }
                        let sp = match p_i32 { 1 => s, 2 => s * s, 3 => s * s * s, _ => s.powi(p_i32) };
                        let phi = scale * (a * sp + b);
                        for h in 0..self.head_dim { y_pre_row[[0, h]] += phi * v[[j, h]]; }
                    }

                    // W_out grads: yh_gated_row = y_pre_row * eff_i
                    let eff_i = g_col[[i, 0]] * m_col[[i, 0]];
                    let mut yh_gated_row = y_pre_row.clone();
                    for h in 0..self.head_dim { yh_gated_row[[0, h]] *= eff_i; }
                    {
                        let mut gw_block = grad_w_out.slice_mut(s![start..end, ..]);
                        general_mat_mul(1.0, &yh_gated_row.t(), &out_row, 1.0, &mut gw_block);
                    }

                    // Gradient wrt eff = g*m
                    let mut grad_eff_i = 0.0f32;
                    for h in 0..self.head_dim {
                        grad_eff_i += g_yh_gated_row[[0, h]] * y_pre_row[[0, h]];
                    }
                    let d_g_i = grad_eff_i * m_col[[i, 0]];
                    let _d_m_i = grad_eff_i * g_col[[i, 0]];

                    // Gate Richards path
                    let z_i = a_h * xw_col[[i, 0]] + b_h;
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;
                    // Parameter grads for Richards curve
                    let gws = gate_poly.grad_weights_scalar(z_i as f64, d_g_i as f64);
                    for (wi, &gw) in gws.iter().enumerate() {
                        grad_gate_poly_vec[wi] += gw;
                    }
                    // dW_g_col increment (outer product)
                    {
                        let mut grad_wg_slice = grad_w_g.slice_mut(s![.., h_idx..h_idx + 1]);
                        for d in 0..self.embed_dim { grad_wg_slice[[d, 0]] += a_h * input[[i, d]] * grad_g_i; }
                    }
                    grad_alpha_g[[0, h_idx]] += grad_g_i * xw_col[[i, 0]];
                    grad_beta_g[[0, h_idx]] += grad_g_i;
                    // dX from gating path
                    {
                        let wg_col_owned = self.w_g.slice(s![.., h_idx..h_idx + 1]).to_owned();
                        let wg_scaled_t = wg_col_owned.t();
                        for d in 0..self.embed_dim { grad_input_total[[i, d]] += a_h * wg_scaled_t[[0, d]] * grad_g_i; }
                    }

                    // Threshold sigmoid path - simplified gradients for new predictor
                    if self.head_selection_config.gating.use_learned_predictor {
                        // For now, skip detailed gradient computation for the new predictor
                        // TODO: Implement proper gradient computation for the two-layer network
                    }

                    // Attention path: g_yh_pre_row = g_yh_gated_row * g_i * m_i
                    let mut g_yh_pre_row = g_yh_gated_row.clone();
                    for h in 0..self.head_dim { g_yh_pre_row[[0, h]] *= g_col[[i, 0]] * m_col[[i, 0]]; }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() { s += q_pe[pos]; }
                        let sp = match p_i32 { 1 => s, 2 => s * s, 3 => s * s * s, _ => s.powi(p_i32) };
                        let phi = scale * (a * sp + b);
                        // dV
                        for h in 0..self.head_dim { grad_v[[j, h]] += phi * g_yh_pre_row[[0, h]]; }
                        // dphi
                        let dphi_ij = g_yh_pre_row.row(0).dot(&v.row(j));
                        // accumulate scalar grads
                        grad_scale_scalar += dphi_ij * (a * sp + b);
                        grad_a_scalar += dphi_ij * scale * sp;
                        grad_b_scalar += dphi_ij * scale;
                        // dS
                        let spm1 = match p_i32 { 1 => 1.0, 2 => s, 3 => s * s, _ => s.powi(p_i32 - 1) };
                        let d_s_ij = dphi_ij * scale * a * (self.p as f32) * spm1;
                        // base Q,K grads
                        for h in 0..self.head_dim {
                            grad_q[[i, h]] += d_s_ij * k[[j, h]] * dk_scale;
                            grad_k[[j, h]] += d_s_ij * q[[i, h]] * dk_scale;
                        }
                        // CoPE grads
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            for h in 0..self.head_dim {
                                grad_q[[i, h]] += d_s_ij * self.cope.pos_embeddings[[pos, h]];
                                grad_p_local[[pos, h]] += d_s_ij * q[[i, h]];
                            }
                        }
                    }
                }

                // Backprop through linear projections for this head
                let d_w_q = input.t().dot(&grad_q);
                let d_w_k = input.t().dot(&grad_k);
                let d_w_v = input.t().dot(&grad_v);
                all_param_grads.push(d_w_q);
                all_param_grads.push(d_w_k);
                all_param_grads.push(d_w_v);
                general_mat_mul(1.0, &grad_q, &head.w_q.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_k, &head.w_k.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_v, &head.w_v.t(), 1.0, &mut grad_input_total);

                // Aggregate CoPE position grads
                grad_cope_pos += &grad_p_local;
            }
        }

        // ===== Head-selection regularizers (auxiliary losses) =====
        // TODO: Consider decoupling MoH training like RichardsCurve
        // Option 1: Keep coupled (current) - MoH learns from attention gradients + auxiliary losses
        // Option 2: Independent training - MoH learns from separate head-selection objectives
        // Option 3: Hierarchical training - MoH learns first, then attention layer learns
        if self.head_selection_config.gating.use_learned_predictor && (self.head_selection_config.gating.complexity_loss_weight > 0.0 || self.head_selection_config.gating.load_balance_weight > 0.0 || self.head_selection_config.gating.sparsity_weight > 0.0) {
            // Use the new predictor for threshold computation
            let m_vec = if let Some(predictor) = &self.threshold_predictor {
                predictor.forward(&input.view())
            } else {
                Array2::<f32>::ones((n, 1)) // Fallback
            };

            // Precompute g(z) and eff per head
            let mut g_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut eff_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut z_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut max_abs_vec: Vec<f64> = vec![0.0; self.num_heads];

            for h in 0..self.num_heads {
                let w_g_col = self.w_g.slice(s![.., h..h + 1]);
                let xw_col = input.dot(&w_g_col);
                let a_h = self.alpha_g[[0, h]];
                let b_h = self.beta_g[[0, h]];
                let mut z_col = xw_col.clone();
                z_col.mapv_inplace(|v| a_h * v + b_h);
                let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
                max_abs_vec[h] = max_abs_z;
                let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
                let mut g_col = z_col.clone();
                g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);
                for i in 0..n {
                    z_mat[[i, h]] = z_col[[i, 0]];
                    g_mat[[i, h]] = g_col[[i, 0]];
                    eff_mat[[i, h]] = g_col[[i, 0]] * m_vec[[i, 0]];
                }
            }

            let inv_n = 1.0f32 / (n as f32);
            let inv_h = 1.0f32 / (self.num_heads as f32);
            let target_heads = ((self.head_selection_config.min_heads + self.head_selection_config.max_heads) as f32) * 0.5;
 
            for i in 0..n {
                let m_i = m_vec[[i, 0]];
                // sum over heads
                let mut s = 0.0f32;
                for h in 0..self.num_heads { s += eff_mat[[i, h]]; }
                let mean = s * inv_h;
 
                // base derivative for complexity and sparsity (normalized)
                let mut base_d = 0.0f32;
                if self.head_selection_config.gating.complexity_loss_weight > 0.0 {
                    base_d += self.head_selection_config.gating.complexity_loss_weight * (s - target_heads) * inv_n;
                }
                // sparsity derivative normalized by tokens and heads
                base_d += self.head_selection_config.gating.sparsity_weight * inv_n * inv_h;
 
                // accumulate threshold gradient across heads
                let mut _d_m_total = 0.0f32;
 
                for h in 0..self.num_heads {
                    let eff_h = eff_mat[[i, h]];
                    let mut d_eff_h = base_d;
                    if self.head_selection_config.gating.load_balance_weight > 0.0 {
                        d_eff_h += 2.0 * self.head_selection_config.gating.load_balance_weight * inv_n * inv_h * (eff_h - mean);
                    }
                    // gating path
                    let d_g_i = d_eff_h * m_i;
                    let a_h = self.alpha_g[[0, h]];
                    let z_i = z_mat[[i, h]];
                    let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_vec[h]);
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;

                    // update gating parameter grads
                    for d in 0..self.embed_dim { grad_w_g[[d, h]] += a_h * input[[i, d]] * grad_g_i; }
                    // alpha uses xw; derive xw from z: xw = (z - beta)/alpha when alpha != 0
                    let xw_val = if a_h.abs() > 1e-8 { (z_i - self.beta_g[[0, h]]) / a_h } else { 0.0 };
                    grad_alpha_g[[0, h]] += grad_g_i * xw_val;
                    grad_beta_g[[0, h]] += grad_g_i;
                    for d in 0..self.embed_dim { grad_input_total[[i, d]] += a_h * self.w_g[[d, h]] * grad_g_i; }

                    // threshold accumulation uses g
                    _d_m_total += d_eff_h * g_mat[[i, h]];
                }

                // threshold predictor grads - simplified for new predictor
                // TODO: Implement proper gradient computation for the two-layer network
            }
        }
 
         // Append output projection grads and scalar grads and gating grads
        all_param_grads.push(grad_w_out);
        let grad_a = Array2::<f32>::from_shape_vec((1, 1), vec![grad_a_scalar]).unwrap();
        let grad_b = Array2::<f32>::from_shape_vec((1, 1), vec![grad_b_scalar]).unwrap();
        let grad_scale = Array2::<f32>::from_shape_vec((1, 1), vec![grad_scale_scalar]).unwrap();
        all_param_grads.push(grad_a);
        all_param_grads.push(grad_b);
        all_param_grads.push(grad_scale);
        all_param_grads.push(grad_w_g);
        all_param_grads.push(grad_alpha_g);
        all_param_grads.push(grad_beta_g);
        // gate Richards parameter grads
        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (1, n_gate_w),
            grad_gate_poly_vec.into_iter().map(|v| v as f32).collect(),
        ).unwrap();
        all_param_grads.push(grad_gate_poly);

        // Threshold predictor grads
        if self.head_selection_config.gating.use_learned_predictor {
            all_param_grads.push(grad_w_tau.unwrap());
            all_param_grads.push(grad_b_tau.unwrap());
            all_param_grads.push(grad_w2_tau.unwrap());
            all_param_grads.push(grad_b2_tau.unwrap());
            // Add activation parameter gradients
            let grad_activation_tau_f32 = Array2::<f32>::from_shape_vec(
                (1, grad_activation_tau.as_ref().unwrap().len()),
                grad_activation_tau.unwrap().into_iter().map(|v| v as f32).collect(),
            ).unwrap();
            all_param_grads.push(grad_activation_tau_f32);
        }

        all_param_grads.push(grad_cope_pos);

        (grad_input_total, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        // Expect 3 per head + w_out + a + b + scale + w_g + alpha_g + beta_g + gate_poly_w + threshold_predictor
        let mut expected = self.num_heads * 3 + 1 + 3 + 3 + 1; // + gate_poly_w
        if self.head_selection_config.gating.use_learned_predictor { expected += 5; } // weights1, bias1, weights2, bias2, activation_params
        expected += 1; // CoPE parameters
        if param_grads.len() != expected {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "PolyAttention expected {} grad arrays, got {}",
                    expected,
                    param_grads.len()
                ),
            });
        }
        let mut idx = 0;
        for head in &mut self.heads {
            head.opt_w_q.step(&mut head.w_q, &param_grads[idx], lr);
            head.opt_w_k.step(&mut head.w_k, &param_grads[idx + 1], lr);
            head.opt_w_v.step(&mut head.w_v, &param_grads[idx + 2], lr);
            idx += 3;
        }
        self.opt_w_out.step(&mut self.w_out, &param_grads[idx], lr);
        idx += 1;
        self.opt_a.step(&mut self.a, &param_grads[idx], lr);
        self.opt_b.step(&mut self.b, &param_grads[idx + 1], lr);
        self.opt_scale.step(&mut self.scale, &param_grads[idx + 2], lr);
        idx += 3;
        self.opt_w_g.step(&mut self.w_g, &param_grads[idx], lr);
        self.opt_alpha_g.step(&mut self.alpha_g, &param_grads[idx + 1], lr);
        self.opt_beta_g.step(&mut self.beta_g, &param_grads[idx + 2], lr);
        idx += 3;
        // TODO: Consider decoupling Richards curve training
        // Option 1: Keep coupled (current) - Richards learns from attention gradients
        // Option 2: Independent training - Richards learns from separate objectives
        // Option 3: Meta-learning - Richards learns across multiple attention layers
        {
            let grad_gate_poly = &param_grads[idx];
            let grad_gate_vec: Vec<f64> = grad_gate_poly.iter().map(|&x| x as f64).collect();
            self.gate_poly.step(&grad_gate_vec, lr as f64);
        }
        idx += 1;

        if self.head_selection_config.gating.use_learned_predictor {
            if let (Some(predictor), Some(opt_w1), Some(opt_b1), Some(opt_w2), Some(opt_b2)) =
                (&mut self.threshold_predictor, &mut self.opt_w_tau, &mut self.opt_b_tau,
                 &mut self.opt_w2_tau, &mut self.opt_b2_tau) {
                // Update first layer weights and biases
                opt_w1.step(&mut predictor.weights1, &param_grads[idx], lr);
                // bias1 is (hidden_dim,) but gradient is (hidden_dim, 1), so reshape bias to match optimizer
                let mut bias1_reshaped = predictor.bias1.clone().into_shape((predictor.bias1.len(), 1)).unwrap();
                opt_b1.step(&mut bias1_reshaped, &param_grads[idx + 1], lr);
                predictor.bias1.assign(&bias1_reshaped.view().into_shape(predictor.bias1.len()).unwrap());
                // Update second layer weights and biases
                opt_w2.step(&mut predictor.weights2, &param_grads[idx + 2], lr);
                // bias2 is (1,) but gradient is (1, 1), so reshape bias to match optimizer
                let mut bias2_reshaped = predictor.bias2.clone().into_shape((predictor.bias2.len(), 1)).unwrap();
                opt_b2.step(&mut bias2_reshaped, &param_grads[idx + 3], lr);
                predictor.bias2.assign(&bias2_reshaped.view().into_shape(predictor.bias2.len()).unwrap());
                // Update Richards activation parameters using its own step method
                let grad_activation_vec: Vec<f64> = param_grads[idx + 4].iter().map(|&x| x as f64).collect();
                predictor.activation.step(&grad_activation_vec, lr as f64);
            }
            idx += 5; // Updated parameter count: weights1, bias1, weights2, bias2, activation_params
        }
        self.cope.apply_gradients(&param_grads[idx], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (input_grads, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    /// Get parameter information for this PolyAttention layer
    fn get_param_info(&mut self) -> &PolyAttentionParamInfo {
        if self.param_info.is_none() {
            // Calculate parameter counts for each component
            let head_params_per_head = self.heads.first()
                .map(|h| h.w_q.len() + h.w_k.len() + h.w_v.len())
                .unwrap_or(0);
            let head_params_total = head_params_per_head * self.heads.len();

            let output_projection_params = self.w_out.len();

            let polynomial_params = self.a.len() + self.b.len() + self.scale.len();

            let gating_params = self.w_g.len() + self.alpha_g.len() + self.beta_g.len();

            let gate_poly_params = self.gate_poly.weights().len();

            let threshold_predictor_params = if self.head_selection_config.gating.use_learned_predictor {
                if let Some(predictor) = &self.threshold_predictor {
                    predictor.weights1.len() + predictor.bias1.len() +
                    predictor.weights2.len() + predictor.bias2.len()
                } else {
                    // Fallback to old count for compatibility
                    self.embed_dim + 1 + 1
                }
            } else {
                0
            };

            let cope_params = self.cope.parameters();

            let total_params = head_params_total + output_projection_params + polynomial_params +
                             gating_params + gate_poly_params + threshold_predictor_params + cope_params;

            self.param_info = Some(PolyAttentionParamInfo {
                head_params_per_head,
                head_params_total,
                output_projection_params,
                polynomial_params,
                gating_params,
                gate_poly_params,
                threshold_predictor_params,
                cope_params,
                total_params,
            });
        }

        self.param_info.as_ref().unwrap()
    }

    /// Get detailed parameter breakdown for this PolyAttention layer
    pub fn param_breakdown(&mut self) -> &PolyAttentionParamInfo {
        self.get_param_info()
    }

    fn parameters(&self) -> usize {
        // Use cached value if available, otherwise compute
        if let Some(ref info) = self.param_info {
            info.total_params
        } else {
            // Fallback to original computation (but this won't be cached)
            let head_params = self
                .heads
                .iter()
                .map(|h| h.w_q.len() + h.w_k.len() + h.w_v.len())
                .sum::<usize>();
            let mut total = self.w_out.len()
                + 3
                + head_params
                + self.w_g.len()
                + self.alpha_g.len()
                + self.beta_g.len()
                + self.gate_poly.weights().len();
            total += self.cope.parameters();
            if self.head_selection_config.gating.use_learned_predictor {
                if let Some(predictor) = &self.threshold_predictor {
                    total += predictor.weights1.len() + predictor.bias1.len() +
                            predictor.weights2.len() + predictor.bias2.len();
                } else {
                    total += self.embed_dim + 1 + 1;
                }
            }
            total
        }
    }

    // Initialize or ensure learned threshold predictor parameters
    fn ensure_threshold_predictor(&mut self) {
        if self.head_selection_config.gating.use_learned_predictor && self.threshold_predictor.is_none() {
            // Use smaller hidden dimension like AutoDeco (128 is typical)
            let predictor_hidden_dim = 128.min(self.embed_dim / 2).max(32);
            self.threshold_predictor = Some(ThresholdPredictor::new(self.embed_dim, predictor_hidden_dim, 1));
            // Old optimizer initialization - should not be used with new predictor
            // Keeping for backward compatibility but this path is deprecated
        }
    }

    pub fn set_head_selection_config(&mut self, strategy: &HeadSelectionStrategy) {
        self.head_selection_config = HeadSelectionConfig::from_strategy(strategy, self.num_heads);

        // Initialize threshold predictor if needed (AutoDeco-inspired architecture)
        if self.head_selection_config.gating.use_learned_predictor && self.threshold_predictor.is_none() {
            // Use smaller hidden dimension like AutoDeco (128 is typical)
            let predictor_hidden_dim = 128.min(self.embed_dim / 2).max(32);
            self.threshold_predictor = Some(ThresholdPredictor::new(self.embed_dim, predictor_hidden_dim, 1));
            // Optimizers for the two-layer network
            self.opt_w_tau = Some(Adam::new((self.embed_dim, predictor_hidden_dim)));
            self.opt_b_tau = Some(Adam::new((predictor_hidden_dim, 1)));
            self.opt_w2_tau = Some(Adam::new((predictor_hidden_dim, 1)));
            self.opt_b2_tau = Some(Adam::new((1, 1)));
            // Note: Richards activation uses its own step method, no Adam optimizer needed
            // Note: RichardsNorm doesn't have trainable parameters, so no optimizer needed
        }
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let mut res = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let tokens = self.head_selection_config.gating.metrics.token_count_per_component[h];
            let avg = if tokens > 0 {
                self.head_selection_config.gating.metrics.active_sum_per_component[h] / tokens as f32
            } else { 0.0 };
            res.push((avg, tokens));
            self.head_selection_config.gating.metrics.active_sum_per_component[h] = 0.0;
            self.head_selection_config.gating.metrics.token_count_per_component[h] = 0;
        }
        res
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            let min = self.head_selection_config.metrics_tau_min;
            let max = self.head_selection_config.metrics_tau_max;
            self.head_selection_config.metrics_tau_min = f32::INFINITY;
            self.head_selection_config.metrics_tau_max = f32::NEG_INFINITY;
            self.head_selection_config.metrics_tau_sum = 0.0;
            self.head_selection_config.metrics_tau_count = 0;
            Some((min, max))
        } else {
            None
        }
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        if self.head_selection_config.metrics_g_count > 0 {
            let rms = (self.head_selection_config.metrics_g_sq_sum / self.head_selection_config.metrics_g_count as f32).sqrt();
            self.head_selection_config.metrics_g_sq_sum = 0.0;
            self.head_selection_config.metrics_g_count = 0;
            Some(rms)
        } else { None }
    }
}

impl Layer for PolyAttention {
    fn layer_type(&self) -> &str {
        "PolyAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // default causal
        self.forward_impl(input, true)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        PolyAttention::compute_gradients(self, _input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        PolyAttention::apply_gradients(self, param_grads, lr)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        PolyAttention::backward(self, grads, lr)
    }

    fn parameters(&self) -> usize {
        PolyAttention::parameters(self)
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq: f32 = 0.0;

        // Heads: w_q, w_k, w_v
        for head in &self.heads {
            sumsq += head.w_q.iter().map(|&w| w * w).sum::<f32>();
            sumsq += head.w_k.iter().map(|&w| w * w).sum::<f32>();
            sumsq += head.w_v.iter().map(|&w| w * w).sum::<f32>();
        }

        // Output projection
        sumsq += self.w_out.iter().map(|&w| w * w).sum::<f32>();

        // Polynomial scalars
        sumsq += self.a.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.b.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.scale.iter().map(|&w| w * w).sum::<f32>();

        // Gating parameters
        sumsq += self.w_g.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.alpha_g.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.beta_g.iter().map(|&w| w * w).sum::<f32>();

        // Learnable Richards gate parameters
        sumsq += self
            .gate_poly
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();

        // CoPE positional embeddings
        sumsq += self.cope.weight_norm().powi(2);

        // Threshold predictor weights if present
        if let Some(pred) = &self.threshold_predictor {
            sumsq += pred.weights1.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.weights2.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.bias1.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.bias2.iter().map(|&w| w * w).sum::<f32>();
            // Include RichardsNorm internal weights via its trait method
            sumsq += pred.norm.weight_norm().powi(2);
        }

        sumsq.sqrt()
    }
}
