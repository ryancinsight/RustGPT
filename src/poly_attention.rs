use std::{cell::RefCell, thread_local};

use ndarray::{Array2, Axis, azip, linalg::general_mat_mul, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{MAX_SEQ_LEN, adam::Adam, llm::Layer, richards::{RichardsCurve, Variant}, model_config::HeadSelectionStrategy};

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

    // ===== Learned threshold predictor (optional) =====
    pub use_learned_threshold: bool,
    pub w_tau: Option<Array2<f32>>,     // (embed_dim, 1)
    pub alpha_tau: Option<Array2<f32>>, // (1, 1)
    pub beta_tau: Option<Array2<f32>>,  // (1, 1)
    opt_w_tau: Option<Adam>,
    opt_alpha_tau: Option<Adam>,
    opt_beta_tau: Option<Adam>,

    // Head selection metrics and config
    pub load_balance_weight: f32,
    pub sparsity_weight: f32,
    pub min_heads: usize,
    pub max_heads: usize,
    pub complexity_loss_weight: f32,
    pub metrics_active_sum_per_head: Vec<f32>,
    pub metrics_token_count_per_head: Vec<usize>,
    pub metrics_tau_min: f32,
    pub metrics_tau_max: f32,
    pub metrics_tau_sum: f32,
    pub metrics_tau_count: usize,
    pub metrics_g_sq_sum: f32,
    pub metrics_g_count: usize,

    // CoPE integration and sliding window
    use_cope: bool,
    cope_max_pos: usize,
    cope_pos_embeddings: Option<Array2<f32>>, // (max_pos+1, head_dim)
    opt_cope_pos: Option<Adam>,
    window_size: Option<usize>,

    // training cache
    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>, // (N, embed_dim)
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
        let use_cope = true;
        let cope_max_pos = max_pos;
        let normal_pe = Normal::new(0.0, 0.02).unwrap();
        let pe =
            Array2::<f32>::from_shape_fn((max_pos + 1, head_dim), |_| normal_pe.sample(&mut rng));
        let opt = Adam::new((max_pos + 1, head_dim));
        let cope_pos_embeddings = Some(pe);
        let opt_cope_pos = Some(opt);

        // Richards curve gate (default sigmoid variant, learnable)
        let gate_poly = RichardsCurve::new_learnable(Variant::Sigmoid);

        // Threshold predictor defaults
        let use_learned_threshold = false;

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
            use_learned_threshold,
            w_tau: None,
            alpha_tau: None,
            beta_tau: None,
            opt_w_tau: None,
            opt_alpha_tau: None,
            opt_beta_tau: None,
            load_balance_weight: 0.0,
             sparsity_weight: 0.0,
             min_heads: 1,
             max_heads: num_heads,
             complexity_loss_weight: 0.0,
             metrics_active_sum_per_head: vec![0.0; num_heads],
             metrics_token_count_per_head: vec![0; num_heads],
            metrics_tau_min: f32::INFINITY,
            metrics_tau_max: f32::NEG_INFINITY,
            metrics_tau_sum: 0.0,
            metrics_tau_count: 0,
            metrics_g_sq_sum: 0.0,
            metrics_g_count: 0,
            use_cope,
            cope_max_pos,
            cope_pos_embeddings,
            opt_cope_pos,
            window_size,
            cached_input: None,
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

    fn cope_pos_logits(
        &self,
        q: &Array2<f32>,
        _k: &Array2<f32>,
        window_size: Option<usize>,
    ) -> Array2<f32> {
        let n = q.nrows();
        let mut pos_logits = Array2::<f32>::zeros((n, n));
        if let Some(pe) = &self.cope_pos_embeddings {
            for i in 0..n {
                let j_start = match window_size {
                    Some(w) => i.saturating_sub(w - 1),
                    None => 0,
                };
                for j in j_start..=i {
                    let pos = i - j;
                    if pos <= self.cope_max_pos {
                        pos_logits[[i, j]] = q.row(i).dot(&pe.row(pos));
                    }
                }
            }
        }
        pos_logits
    }

    pub fn forward_impl(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        // input: (N, embed_dim)
        let (n, d_model) = (input.nrows(), input.ncols());
        assert_eq!(d_model, self.embed_dim);

        self.cached_input = Some(input.clone());

        let dk_scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // Streamed accumulation: avoid building a large concat buffer
        let mut out = input.to_owned();

        if self.use_learned_threshold {
            self.ensure_threshold_predictor();
        }
        // Temporary accumulators for head activity and predictor metrics
        let mut active_sums_tmp = vec![0.0f32; self.num_heads];
        let mut token_counts_tmp = vec![0usize; self.num_heads];
        let mut tau_min_local = f32::INFINITY;
        let mut tau_max_local = f32::NEG_INFINITY;
        let mut tau_count_local = 0usize;
        let mut g_sq_sum_local = 0.0f32;
        let mut g_count_local = 0usize;

        for (h_idx, head) in self.heads.iter().enumerate() {
            // Project to Q, K, V
            let q = input.dot(&head.w_q); // (N, d_h)
            let k = input.dot(&head.w_k); // (N, d_h)
            let v = input.dot(&head.w_v); // (N, d_h)

            // Compute per-token gating for this head: g = Richards(alpha * (X·w_g_col) + beta)
            let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
            let mut xw_col = input.dot(&w_g_col); // (N,1)
            let a_h = self.alpha_g[[0, h_idx]];
            let b_h = self.beta_g[[0, h_idx]];
            let max_abs_z = xw_col.iter().fold(0.0_f64, |m, &v| {
                let z = a_h as f64 * v as f64 + b_h as f64;
                m.max(z.abs())
            });
            let mut gate_poly = self.gate_poly.clone();
            gate_poly.update_scaling_from_max_abs(max_abs_z);
            let mut g_col = xw_col.clone();
            g_col.mapv_inplace(|xw| gate_poly.forward_scalar((a_h * xw + b_h) as f64) as f32);
            // Predictor norm RMS tracking (x·W_g)
            g_sq_sum_local += xw_col.iter().map(|&v| v * v).sum::<f32>();
            g_count_local += n;

            // Learned threshold predictor m = sigmoid(alpha_tau * (X·W_tau) + beta_tau)
            let mut m_col = Array2::<f32>::ones((n, 1));
            if self.use_learned_threshold {
                let w_tau = self.w_tau.as_ref().unwrap();
                let alpha_tau = self.alpha_tau.as_ref().unwrap();
                let beta_tau = self.beta_tau.as_ref().unwrap();
                let mut xw_tau = input.dot(w_tau); // (N,1)
                let a_t = alpha_tau[[0, 0]];
                let b_t = beta_tau[[0, 0]];
                // z_tau pre-activation for metrics
                let mut z_tau = xw_tau.clone();
                z_tau.mapv_inplace(|v| a_t * v + b_t);
                let local_min = z_tau.iter().fold(f32::INFINITY, |m, &z| m.min(z));
                let local_max = z_tau.iter().fold(f32::NEG_INFINITY, |m, &z| m.max(z));
                tau_min_local = tau_min_local.min(local_min);
                tau_max_local = tau_max_local.max(local_max);
                tau_count_local += n;
                // m = sigmoid(z_tau)
                m_col.assign(&z_tau);
                m_col.mapv_inplace(|z| 1.0 / (1.0 + (-z).exp()));
            }

            // Effective gate per token: eff = g * m
            let eff_col = &g_col * &m_col;
            // Accumulate active-head metrics (soft routing: sum of eff over tokens)
            active_sums_tmp[h_idx] += eff_col.sum();
            token_counts_tmp[h_idx] += n;

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
                    let q_pe: Option<Vec<f32>> = if self.use_cope {
                        if let Some(pe) = &self.cope_pos_embeddings {
                            let max_pos = usize::min(self.cope_max_pos, i.saturating_sub(j_start));
                            let mut buf = vec![0.0f32; max_pos + 1];
                            for pos in 0..=max_pos {
                                buf[pos] = q.row(i).dot(&pe.row(pos));
                            }
                            Some(buf)
                        } else { None }
                    } else { None };

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        if let Some(ref qpe) = q_pe {
                            let pos = i.saturating_sub(j);
                            if pos < qpe.len() { s += qpe[pos]; }
                        }
                        let sp = match p_i32 { 1 => s, 2 => s * s, 3 => s * s * s, _ => s.powi(p_i32) };
                        let phi = scale * (a * sp + b);
                        // yh_row += phi * v[j,:]
                        for h in 0..self.head_dim {
                            yh_row[[0, h]] += phi * v[[j, h]];
                        }
                    }

                    // Apply gating: eff = g * m for token i
                    let eff_i = g_col[[i, 0]] * m_col[[i, 0]];
                    for h in 0..self.head_dim {
                        yh_row[[0, h]] *= eff_i;
                    }

                    // Accumulate into output row i via W_out block
                    let mut out_row = out.slice_mut(s![i..i + 1, ..]);
                    general_mat_mul(1.0, &yh_row, &w_block, 1.0, &mut out_row);
                }
            }
        }

        // Flush temporary metrics into persistent accumulators
        for h in 0..self.num_heads {
            self.metrics_active_sum_per_head[h] += active_sums_tmp[h];
            self.metrics_token_count_per_head[h] += token_counts_tmp[h];
        }
        if self.use_learned_threshold && tau_count_local > 0 {
            self.metrics_tau_min = self.metrics_tau_min.min(tau_min_local);
            self.metrics_tau_max = self.metrics_tau_max.max(tau_max_local);
            self.metrics_tau_count += tau_count_local;
        }
        self.metrics_g_sq_sum += g_sq_sum_local;
        self.metrics_g_count += g_count_local;

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

        // Threshold predictor grads
        let mut grad_w_tau = if self.use_learned_threshold {
            Some(Array2::<f32>::zeros((self.embed_dim, 1)))
        } else { None };
        let mut grad_alpha_tau = if self.use_learned_threshold {
            Some(Array2::<f32>::zeros((1, 1)))
        } else { None };
        let mut grad_beta_tau = if self.use_learned_threshold {
            Some(Array2::<f32>::zeros((1, 1)))
        } else { None };

        // CoPE grads accumulator (shared across heads)
        let mut grad_cope_pos = if self.use_cope {
            Some(Array2::<f32>::zeros((self.cope_max_pos + 1, self.head_dim)))
        } else {
            None
        };

        // Per-head param grads (Wq, Wk, Wv) + W_out + scalars + gating params
        let mut all_param_grads: Vec<Array2<f32>> = Vec::new();

        // Build grad for W_out block-wise to avoid materializing H
        let mut grad_w_out = Array2::<f32>::zeros((self.embed_dim, self.embed_dim)); // (D, D)

        let a = self.a[[0, 0]];
        let b = self.b[[0, 0]];
        let scale = self.scale[[0, 0]];
        let p_i32 = self.p as i32;
        let p_f = self.p as f32;
        for (h_idx, head) in self.heads.iter().enumerate() {
            // Recompute per-head Q, K, V and intermediates
            let q = input.dot(&head.w_q); // (N, d_h)
            let k = input.dot(&head.w_k); // (N, d_h)
            let v = input.dot(&head.w_v); // (N, d_h)

            // Gating forward values for this head (and caches for backward)
            let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
            let xw_col = input.dot(&w_g_col); // (N,1)
            let a_h = self.alpha_g[[0, h_idx]];
            let b_h = self.beta_g[[0, h_idx]];
            // z = a_h * xw + b_h; g = Richards(z)
            let mut z_col = xw_col.clone();
            z_col.mapv_inplace(|v| a_h * v + b_h);
            let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
            let mut gate_poly = self.gate_poly.clone();
            gate_poly.update_scaling_from_max_abs(max_abs_z);
            let mut g_col = z_col.clone();
            g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);

            // Threshold path forward
            let mut m_col = Array2::<f32>::ones((n, 1));
            let (xw_tau, a_t, b_t) = if self.use_learned_threshold {
                let w_tau = self.w_tau.as_ref().unwrap();
                let alpha_tau = self.alpha_tau.as_ref().unwrap();
                let beta_tau = self.beta_tau.as_ref().unwrap();
                let xw_tau = input.dot(w_tau);
                let a_t = alpha_tau[[0, 0]];
                let b_t = beta_tau[[0, 0]];
                m_col.assign(&xw_tau);
                m_col.mapv_inplace(|v| {
                    let z = a_t * v + b_t;
                    1.0 / (1.0 + (-z).exp())
                });
                (Some(xw_tau), Some(a_t), Some(b_t))
            } else { (None, None, None) };

            {
                // True banded backward: per-row computations within the window
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]);
                let w_block_t = w_block.t();

                // Allocate per-head grads
                let mut grad_q: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_k: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_v: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_p_local: Option<Array2<f32>> = if self.use_cope {
                    Some(Array2::<f32>::zeros((self.cope_max_pos + 1, self.head_dim)))
                } else { None };

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
                    let q_pe: Option<Vec<f32>> = if self.use_cope {
                        if let Some(pe) = &self.cope_pos_embeddings {
                            let max_pos = usize::min(self.cope_max_pos, i.saturating_sub(j_start));
                            let mut buf = vec![0.0f32; max_pos + 1];
                            for pos in 0..=max_pos {
                                buf[pos] = q.row(i).dot(&pe.row(pos));
                            }
                            Some(buf)
                        } else { None }
                    } else { None };

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        if let Some(ref qpe) = q_pe {
                            let pos = i.saturating_sub(j);
                            if pos < qpe.len() { s += qpe[pos]; }
                        }
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
                    let d_m_i = grad_eff_i * g_col[[i, 0]];

                    // Gate Richards path
                    let z_i = a_h * xw_col[[i, 0]] + b_h;
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;
                    // Parameter grads for Richards curve
                    let gws = gate_poly.grad_weights_scalar(z_i as f64, d_g_i as f64);
                    for (wi, gw) in gws.iter().enumerate() { grad_gate_poly_vec[wi] += *gw; }
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

                    // Threshold sigmoid path
                    if self.use_learned_threshold {
                        let xw_tau = xw_tau.as_ref().unwrap();
                        let a_t = a_t.unwrap();
                        let m_i = m_col[[i, 0]];
                        let dm_dz_i = m_i * (1.0 - m_i);
                        let grad_tau_i = d_m_i * dm_dz_i;
                        // dW_tau increment
                        {
                            let grad_wtau = grad_w_tau.as_mut().unwrap();
                            for d in 0..self.embed_dim { grad_wtau[[d, 0]] += a_t * input[[i, d]] * grad_tau_i; }
                        }
                        grad_alpha_tau.as_mut().unwrap()[[0, 0]] += grad_tau_i * xw_tau[[i, 0]];
                        grad_beta_tau.as_mut().unwrap()[[0, 0]] += grad_tau_i;
                        // dX from threshold path
                        {
                            let wt_scaled_t = self.w_tau.as_ref().unwrap().t();
                            for d in 0..self.embed_dim { grad_input_total[[i, d]] += a_t * wt_scaled_t[[0, d]] * grad_tau_i; }
                        }
                    }

                    // Attention path: g_yh_pre_row = g_yh_gated_row * g_i * m_i
                    let mut g_yh_pre_row = g_yh_gated_row.clone();
                    for h in 0..self.head_dim { g_yh_pre_row[[0, h]] *= g_col[[i, 0]] * m_col[[i, 0]]; }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        if let Some(ref qpe) = q_pe {
                            let pos = i.saturating_sub(j);
                            if pos < qpe.len() { s += qpe[pos]; }
                        }
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
                        let dS_ij = dphi_ij * scale * a * (self.p as f32) * spm1;
                        // base Q,K grads
                        for h in 0..self.head_dim {
                            grad_q[[i, h]] += dS_ij * k[[j, h]] * dk_scale;
                            grad_k[[j, h]] += dS_ij * q[[i, h]] * dk_scale;
                        }
                        // CoPE grads
                        if let Some(ref qpe) = q_pe {
                            let pos = i.saturating_sub(j);
                            if pos < qpe.len() {
                                if let Some(pe) = &self.cope_pos_embeddings {
                                    for h in 0..self.head_dim {
                                        grad_q[[i, h]] += dS_ij * pe[[pos, h]];
                                        if let Some(gpl) = grad_p_local.as_mut() {
                                            gpl[[pos, h]] += dS_ij * q[[i, h]];
                                        }
                                    }
                                }
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
                if self.use_cope {
                    if let Some(gpl) = grad_p_local {
                        if let Some(grad_pe_global) = grad_cope_pos.as_mut() {
                            *grad_pe_global += &gpl;
                        }
                    }
                }
            }
        }

        // ===== Head-selection regularizers (auxiliary losses) =====
        if self.use_learned_threshold && (self.complexity_loss_weight > 0.0 || self.load_balance_weight > 0.0 || self.sparsity_weight > 0.0) {
            // m(x) via threshold predictor
            let w_tau = self.w_tau.as_ref().unwrap();
            let alpha_tau = self.alpha_tau.as_ref().unwrap();
            let beta_tau = self.beta_tau.as_ref().unwrap();
            let xw_tau2 = input.dot(w_tau);
            let a_t = alpha_tau[[0, 0]];
            let b_t = beta_tau[[0, 0]];
            let mut m_vec = xw_tau2.clone();
            m_vec.mapv_inplace(|v| {
                let z = a_t * v + b_t;
                1.0 / (1.0 + (-z).exp())
            });

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
                let mut gate_poly = self.gate_poly.clone();
                gate_poly.update_scaling_from_max_abs(max_abs_z);
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
            let target_heads = ((self.min_heads + self.max_heads) as f32) * 0.5;
 
            for i in 0..n {
                let m_i = m_vec[[i, 0]];
                // sum over heads
                let mut s = 0.0f32;
                for h in 0..self.num_heads { s += eff_mat[[i, h]]; }
                let mean = s * inv_h;
 
                // base derivative for complexity and sparsity (normalized)
                let mut base_d = 0.0f32;
                if self.complexity_loss_weight > 0.0 {
                    base_d += self.complexity_loss_weight * (s - target_heads) * inv_n;
                }
                // sparsity derivative normalized by tokens and heads
                base_d += self.sparsity_weight * inv_n * inv_h;
 
                // accumulate threshold gradient across heads
                let mut d_m_total = 0.0f32;
 
                for h in 0..self.num_heads {
                    let eff_h = eff_mat[[i, h]];
                    let mut d_eff_h = base_d;
                    if self.load_balance_weight > 0.0 {
                        d_eff_h += 2.0 * self.load_balance_weight * inv_n * inv_h * (eff_h - mean);
                    }
                    // gating path
                    let d_g_i = d_eff_h * m_i;
                    let a_h = self.alpha_g[[0, h]];
                    let z_i = z_mat[[i, h]];
                    let mut gate_poly = self.gate_poly.clone();
                    gate_poly.update_scaling_from_max_abs(max_abs_vec[h]);
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
                    d_m_total += d_eff_h * g_mat[[i, h]];
                }

                // threshold predictor grads (aggregated across heads)
                let dm_dz_i = m_i * (1.0 - m_i);
                let grad_tau_i = d_m_total * dm_dz_i;
                let a_t = alpha_tau[[0, 0]];
                // dW_tau
                if let Some(gwt) = grad_w_tau.as_mut() {
                    for d in 0..self.embed_dim { gwt[[d, 0]] += a_t * input[[i, d]] * grad_tau_i; }
                }
                if let Some(ga) = grad_alpha_tau.as_mut() {
                    ga[[0, 0]] += grad_tau_i * xw_tau2[[i, 0]];
                }
                if let Some(gb) = grad_beta_tau.as_mut() {
                    gb[[0, 0]] += grad_tau_i;
                }
                // dX from threshold path
                let wt_scaled_t = self.w_tau.as_ref().unwrap().t();
                for d in 0..self.embed_dim { grad_input_total[[i, d]] += a_t * wt_scaled_t[[0, d]] * grad_tau_i; }
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
        if self.use_learned_threshold {
            all_param_grads.push(grad_w_tau.unwrap());
            all_param_grads.push(grad_alpha_tau.unwrap());
            all_param_grads.push(grad_beta_tau.unwrap());
        }

        if let Some(grad_pe) = grad_cope_pos {
            all_param_grads.push(grad_pe);
        }

        (grad_input_total, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        // Expect 3 per head + w_out + a + b + scale + w_g + alpha_g + beta_g + gate_poly_w
        let mut expected = self.num_heads * 3 + 1 + 3 + 3 + 1; // + gate_poly_w
        if self.use_learned_threshold { expected += 3; }
        if self.use_cope { expected += 1; }
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
        // update Richards curve parameters via Adam
        {
            let grad_gate_poly = &param_grads[idx];
            let grad_gate_vec: Vec<f64> = grad_gate_poly.iter().map(|&x| x as f64).collect();
            self.gate_poly.step(&grad_gate_vec, lr as f64);
        }
        idx += 1;

        if self.use_learned_threshold {
            if let (Some(wt), Some(opt)) = (&mut self.w_tau, &mut self.opt_w_tau) {
                opt.step(wt, &param_grads[idx], lr);
            }
            if let (Some(at), Some(opt)) = (&mut self.alpha_tau, &mut self.opt_alpha_tau) {
                opt.step(at, &param_grads[idx + 1], lr);
            }
            if let (Some(bt), Some(opt)) = (&mut self.beta_tau, &mut self.opt_beta_tau) {
                opt.step(bt, &param_grads[idx + 2], lr);
            }
            idx += 3;
        }
        if self.use_cope {
            if let (Some(pe), Some(opt)) = (
                self.cope_pos_embeddings.as_mut(),
                self.opt_cope_pos.as_mut(),
            ) {
                opt.step(pe, &param_grads[idx], lr);
            }
        }
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

    fn parameters(&self) -> usize {
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
        if self.use_cope {
            total += (self.cope_max_pos + 1) * self.head_dim;
        }
        if self.use_learned_threshold {
            total += self.embed_dim + 1 + 1; // w_tau + alpha_tau + beta_tau
        }
        total
    }

    // Initialize or ensure learned threshold predictor parameters
    fn ensure_threshold_predictor(&mut self) {
        if self.w_tau.is_none() {
            let std_tau = (2.0f32 / self.embed_dim as f32).sqrt();
            let normal_tau = Normal::new(0.0, std_tau).unwrap();
            let mut rng = rand::rng();
            let wtau = Array2::<f32>::from_shape_fn((self.embed_dim, 1), |_| normal_tau.sample(&mut rng));
            self.w_tau = Some(wtau);
            self.opt_w_tau = Some(Adam::new((self.embed_dim, 1)));
        }
        if self.alpha_tau.is_none() {
            self.alpha_tau = Some(Array2::<f32>::from_shape_vec((1, 1), vec![1.0]).unwrap());
            self.opt_alpha_tau = Some(Adam::new((1, 1)));
        }
        if self.beta_tau.is_none() {
            self.beta_tau = Some(Array2::<f32>::from_shape_vec((1, 1), vec![0.0]).unwrap());
            self.opt_beta_tau = Some(Adam::new((1, 1)));
        }
    }

    pub fn set_head_selection_config(&mut self, strategy: &HeadSelectionStrategy) {
        match strategy {
            HeadSelectionStrategy::FullyAdaptiveMoH { min_heads, max_heads, complexity_loss_weight, load_balance_weight, sparsity_weight } => {
                self.use_learned_threshold = true;
                self.min_heads = *min_heads as usize;
                self.max_heads = *max_heads as usize;
                self.complexity_loss_weight = *complexity_loss_weight;
                self.load_balance_weight = *load_balance_weight;
                self.sparsity_weight = *sparsity_weight;
                self.ensure_threshold_predictor();
            }
        }
        // reset metrics whenever strategy changes
        for h in 0..self.num_heads {
            self.metrics_active_sum_per_head[h] = 0.0;
            self.metrics_token_count_per_head[h] = 0;
        }
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let mut res = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let tokens = self.metrics_token_count_per_head[h];
            let avg = if tokens > 0 {
                self.metrics_active_sum_per_head[h] / tokens as f32
            } else { 0.0 };
            res.push((avg, tokens));
            self.metrics_active_sum_per_head[h] = 0.0;
            self.metrics_token_count_per_head[h] = 0;
        }
        res
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        if self.metrics_tau_count > 0 {
            let min = self.metrics_tau_min;
            let max = self.metrics_tau_max;
            self.metrics_tau_min = f32::INFINITY;
            self.metrics_tau_max = f32::NEG_INFINITY;
            self.metrics_tau_sum = 0.0;
            self.metrics_tau_count = 0;
            Some((min, max))
        } else {
            None
        }
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        if self.metrics_g_count > 0 {
            let rms = (self.metrics_g_sq_sum / self.metrics_g_count as f32).sqrt();
            self.metrics_g_sq_sum = 0.0;
            self.metrics_g_count = 0;
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
}
