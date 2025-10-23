use std::{cell::RefCell, thread_local};

use ndarray::{Array2, Axis, azip, linalg::general_mat_mul, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use rayon::prelude::*;
use rayon::slice::ParallelSliceMut;

use crate::{MAX_SEQ_LEN, adam::Adam, llm::Layer, sigmoid_poly::SigmoidPoly}; // [MOD] import learnable polynomial gate

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
    // Per-head gating projection and learned polynomial gate: g = phi(alpha * (X·W_g) + beta)
    pub w_g: Array2<f32>,     // (embed_dim, num_heads)
    pub alpha_g: Array2<f32>, // (1, num_heads)
    pub beta_g: Array2<f32>,  // (1, num_heads)
    opt_w_g: Adam,
    opt_alpha_g: Adam,
    opt_beta_g: Adam,

    // [MOD] Learnable polynomial for gating
    pub gate_poly: SigmoidPoly,

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
    static TLS_ROWBUF1: RefCell<Vec<f32>> = RefCell::new(Vec::new()); // (d_h)
    static TLS_ROWBUF2: RefCell<Vec<f32>> = RefCell::new(Vec::new()); // (d_h)
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

#[inline]
fn with_tls_rowbuf1<R>(d: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TLS_ROWBUF1.with(|cell| {
        let mut v = cell.borrow_mut();
        if v.len() != d { v.resize(d, 0.0); } else { v.fill(0.0); }
        f(v.as_mut_slice())
    })
}

#[inline]
fn with_tls_rowbuf2<R>(d: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TLS_ROWBUF2.with(|cell| {
        let mut v = cell.borrow_mut();
        if v.len() != d { v.resize(d, 0.0); } else { v.fill(0.0); }
        f(v.as_mut_slice())
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
            // [MOD]
            gate_poly: SigmoidPoly::new_cubic_default(),
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
                // allow j in [max(0, i+1-w), i]
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
                    let pos = if j == 0 { i } else { i - j };
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

        for (h_idx, head) in self.heads.iter().enumerate() {
            // Project to Q, K, V
            let q = input.dot(&head.w_q); // (N, d_h)
            let k = input.dot(&head.w_k); // (N, d_h)
            let v = input.dot(&head.w_v); // (N, d_h)

            // Compute per-token gating for this head: g = phi(alpha * (X·w_g_col) + beta)
            let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
            let mut g_col = input.dot(&w_g_col); // (N,1)
            let a_h = self.alpha_g[[0, h_idx]];
            let b_h = self.beta_g[[0, h_idx]];
            // polynomial gate with dynamic scaling based on z range
            let max_abs_z = g_col.iter().fold(0.0_f64, |m, &v| {
                let z = a_h as f64 * v as f64 + b_h as f64;
                m.max(z.abs())
            });
            let mut gate_poly = self.gate_poly.clone();
            gate_poly.update_scaling_from_max_abs(max_abs_z);
            g_col.mapv_inplace(|xw| {
                let z = a_h * xw + b_h;
                gate_poly.forward_scalar(z as f64) as f32
            });

            // Common polynomial params
            let a = self.a[[0, 0]];
            let b = self.b[[0, 0]];
            let scale = self.scale[[0, 0]];
            let p_i32 = self.p as i32;

            // If no sliding window, keep the original dense path for full attention
            if self.window_size.is_none() {
                // Dense path (kept as-is)
                with_tls_scores(n, |scores| {
                    scores.fill(0.0);
                    general_mat_mul(1.0, &q, &k.t(), 0.0, scores);
                    *scores *= dk_scale;

                    if causal {
                        Self::apply_causal_mask_inplace(scores);
                    }
                    Self::apply_sliding_window_mask_inplace(scores, self.window_size);

                    if self.use_cope {
                        let pos_logits = self.cope_pos_logits(&q, &k, self.window_size);
                        *scores += &pos_logits;
                        if causal {
                            Self::apply_causal_mask_inplace(scores);
                        }
                        Self::apply_sliding_window_mask_inplace(scores, self.window_size);
                    }

                    scores.mapv_inplace(|x| {
                        let sp = x.powi(p_i32);
                        scale * (a * sp + b)
                    });

                    if causal {
                        Self::apply_causal_mask_inplace(scores);
                    }
                    Self::apply_sliding_window_mask_inplace(scores, self.window_size);

                    with_tls_yh(n, self.head_dim, |yh| {
                        yh.fill(0.0);
                        general_mat_mul(1.0, &*scores, &v, 0.0, yh);
                        *yh *= &g_col;

                        let start = h_idx * self.head_dim;
                        let end = start + self.head_dim;
                        let w_block = self.w_out.slice(s![start..end, ..]); // (d_h, D)
                        general_mat_mul(1.0, &*yh, &w_block, 1.0, &mut out);
                    });
                });
                continue;
            }

            // Windowed path: true banded compute, no NxN scratch
            let w = self.window_size.unwrap();
            let use_cope = self.use_cope;
            let cope_max_pos = self.cope_max_pos;
            let cope_pe_opt = self.cope_pos_embeddings.as_ref();

            with_tls_yh(n, self.head_dim, |yh| {
                yh.fill(0.0);
                // Parallel row compute with zero-copy slices
                let dh = self.head_dim;
                let q_s = q.as_slice_memory_order().expect("Q contiguous");
                let k_s = k.as_slice_memory_order().expect("K contiguous");
                let v_s = v.as_slice_memory_order().expect("V contiguous");
                let g_s = g_col.as_slice_memory_order().expect("g contiguous"); // (N,1)
                let pe_s_opt = cope_pe_opt.and_then(|pe| pe.as_slice_memory_order());
                let yh_s = yh
                    .as_slice_memory_order_mut()
                    .expect("Yh contiguous");
                // Each row has length dh
                yh_s.par_chunks_mut(dh).enumerate().for_each(|(i, row)| {
                    let j_end: usize = i; // lower-tri band in windowed path
                    let j_start = if w > 0 { i.saturating_sub(w - 1) } else { 0 };
                    let qi = &q_s[i * dh..(i + 1) * dh];
                    for j in j_start..=j_end {
                        let kj = &k_s[j * dh..(j + 1) * dh];
                        // dot(q_i, k_j)
                        let mut s = 0.0f32;
                        for u in 0..dh {
                            s += qi[u] * kj[u];
                        }
                        s *= dk_scale;
                        if use_cope {
                            let pos = i - j;
                            if pos <= cope_max_pos {
                                if let Some(pe_s) = pe_s_opt {
                                    let pe_row = &pe_s[pos * dh..(pos + 1) * dh];
                                    let mut add = 0.0f32;
                                    for u in 0..dh {
                                        add += qi[u] * pe_row[u];
                                    }
                                    s += add;
                                }
                            }
                        }
                        let sp = s.powi(p_i32);
                        let s_act = scale * (a * sp + b);
                        let vj = &v_s[j * dh..(j + 1) * dh];
                        for u in 0..dh {
                            row[u] += s_act * vj[u];
                        }
                    }
                    let g_i = g_s[i];
                    for u in 0..dh {
                        row[u] *= g_i;
                    }
                });

                // Accumulate into output via W_out head block
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]); // (d_h, D)
                general_mat_mul(1.0, &*yh, &w_block, 1.0, &mut out);
            });
        }

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
        // [LEARN] Gate polynomial coefficient gradient accumulator (shared across heads)
        let n_gate_w = self.gate_poly.weights.len();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

        // CoPE grads accumulator (shared across heads) — keeping shape for optimizer compatibility
        let grad_cope_pos = if self.use_cope {
            Some(Array2::<f32>::zeros((self.cope_max_pos + 1, self.head_dim)))
        } else {
            None
        };

        // Per-head param grads (Wq, Wk, Wv) + W_out + scalars + gating params
        let mut all_param_grads: Vec<Array2<f32>> =
            Vec::with_capacity(self.num_heads * 3 + 1 + 3 + 3 + if self.use_cope { 1 } else { 0 });

        // Build grad for W_out block-wise to avoid materializing H
        let mut grad_w_out = Array2::<f32>::zeros((self.embed_dim, self.embed_dim)); // (D, D)

        let a = self.a[[0, 0]];
        let b = self.b[[0, 0]];
        let scale = self.scale[[0, 0]];
        let p_i32 = self.p as i32;
        let p_f = self.p as f32;

        // Banded backward path when windowed attention is enabled
        if let Some(w) = self.window_size {
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
                // z = a_h * xw + b_h; g = phi(z)
                let mut z_col = xw_col.clone();
                z_col.mapv_inplace(|v| a_h * v + b_h);
                let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
                let mut gate_poly = self.gate_poly.clone();
                gate_poly.update_scaling_from_max_abs(max_abs_z);
                let mut g_col = z_col.clone();
                g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);

                // grad wrt W_out needs H_h (gated). We avoid storing H by accumulating the block.
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]);
                let w_block_t = w_block.t();

                // g_yh_gated = dL/dY · W_out_block^T
                let mut g_yh_gated = Array2::<f32>::zeros((n, self.head_dim));
                general_mat_mul(1.0, output_grads, &w_block_t, 0.0, &mut g_yh_gated);

                // Per-projection grads
                let mut grad_q = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_k = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_v = Array2::<f32>::zeros((n, self.head_dim));

                // Gating param grads accumulators (per-head)
                let mut d_wg_col_acc = Array2::<f32>::zeros((self.embed_dim, 1)); // (D,1)
                let mut dalpha = 0.0_f32;
                let mut dbeta = 0.0_f32;

                // Gate polynomial coefficient grads (shared accumulator)
                let c_gate = gate_poly.scaling as f64;

                // Row-wise banded backward
                // Use contiguous zero-copy slices and reusable row buffers
                let dh = self.head_dim;
                let q_s = q.as_slice_memory_order().expect("Q contiguous");
                let k_s = k.as_slice_memory_order().expect("K contiguous");
                let v_s = v.as_slice_memory_order().expect("V contiguous");
                let gyg_s = g_yh_gated.as_slice_memory_order().expect("g_yh_gated contiguous");
                let og_s = output_grads.as_slice_memory_order().expect("output grads contiguous");

                for i in 0..n {
                    let g_i = g_col[[i, 0]];
                    let qi = &q_s[i * dh..(i + 1) * dh];

                    // y_pre[i,*] via banded sum; reuse TLS row buffer
                    with_tls_rowbuf1(dh, |y_pre_row| {
                        let j_start = i.saturating_sub(w - 1);
                        for j in j_start..=i {
                            let kj = &k_s[j * dh..(j + 1) * dh];
                            let mut s_ij = 0.0f32;
                            for u in 0..dh { s_ij += qi[u] * kj[u]; }
                            s_ij *= dk_scale;
                            let sp = s_ij.powi(p_i32);
                            let phi_s = scale * (a * sp + b);
                            let vj = &v_s[j * dh..(j + 1) * dh];
                            for u in 0..dh { y_pre_row[u] += phi_s * vj[u]; }
                        }

                        // Accumulate grad_w_out block using H_h[i,*] = g_i * y_pre_row
                        let og_i = &og_s[i * self.embed_dim..(i + 1) * self.embed_dim];
                        for u in 0..dh {
                            let yhu = y_pre_row[u] * g_i;
                            if yhu != 0.0 {
                                let row_idx = start + u;
                                for d in 0..self.embed_dim {
                                    grad_w_out[[row_idx, d]] += yhu * og_i[d];
                                }
                            }
                        }

                        // grad_g_dg[i] = dot(g_yh_gated[i,*], y_pre[i,*])
                        let gyg_i = &gyg_s[i * dh..(i + 1) * dh];
                        let mut grad_g_dg_i = 0.0f32;
                        for u in 0..dh { grad_g_dg_i += gyg_i[u] * y_pre_row[u]; }
                        // dz = grad_g * phi_gate'(z)
                        let dphi_gate = gate_poly.backward_scalar(z_col[[i, 0]] as f64) as f32;
                        let grad_g_col_i = grad_g_dg_i * dphi_gate;

                        // Accumulate gating weight grads: dW_g_col += alpha * X[i]^T * grad_g_col_i
                        for d in 0..self.embed_dim {
                            d_wg_col_acc[[d, 0]] += a_h * grad_g_col_i * input[[i, d]];
                        }
                        // d alpha and d beta
                        dalpha += grad_g_col_i * xw_col[[i, 0]];
                        dbeta += grad_g_col_i;

                        // dX from gating path: grad_input += grad_g_col_i · (alpha * w_g_col^T)
                        for d in 0..self.embed_dim {
                            grad_input_total[[i, d]] += grad_g_col_i * a_h * w_g_col[[d, 0]];
                        }

                        // [LEARN] gate polynomial coefficient gradient: dL/dw_i += grad_g_dg * (c*z)^i
                        let zf = z_col[[i, 0]] as f64;
                        let mut power = 1.0_f64; // (c*z)^0
                        let cz = c_gate * zf;
                        for wi in 0..n_gate_w {
                            grad_gate_poly_vec[wi] += (grad_g_dg_i as f64) * power;
                            power *= cz;
                        }

                        // Now compute grads wrt V, S, then Q and K over the same band
                        with_tls_rowbuf2(dh, |g_yh_pre_row| {
                            for u in 0..dh { g_yh_pre_row[u] = g_i * gyg_i[u]; }
                            let j_start2 = i.saturating_sub(w - 1);
                            for j in j_start2..=i {
                                // Forward recompute s_ij (no CoPE path in backward for parity)
                                let kj2 = &k_s[j * dh..(j + 1) * dh];
                                let mut s_ij = 0.0f32;
                                for u in 0..dh { s_ij += qi[u] * kj2[u]; }
                                s_ij *= dk_scale;
                                let sp = s_ij.powi(p_i32);
                                // phi'(s) factor for dS
                                let dphi_ds = scale * a * p_f * if p_i32 == 1 { 1.0 } else { s_ij.powi(p_i32 - 1) };

                                // dphi_ij = dot(g_yh_pre[i,*], v_j)
                                let vj2 = &v_s[j * dh..(j + 1) * dh];
                                let mut dphi_ij = 0.0f32;
                                for u in 0..dh { dphi_ij += g_yh_pre_row[u] * vj2[u]; }

                                // Scalars grads wrt (a, b, scale)
                                grad_scale_scalar += dphi_ij * (a * sp + b);
                                grad_a_scalar += dphi_ij * scale * sp;
                                grad_b_scalar += dphi_ij * scale;

                                // dS_ij
                                let dS_ij = dphi_ij * dphi_ds;

                                // dV_j += phi(s_ij) * g_yh_pre[i,*]
                                let phi_s = scale * (a * sp + b);
                                for u in 0..dh {
                                    grad_v[[j, u]] += phi_s * g_yh_pre_row[u];
                                }

                                // dQ_i += dS_ij * dk_scale * k_j;  dK_j += dS_ij * dk_scale * q_i
                                for u in 0..dh {
                                    grad_q[[i, u]] += dS_ij * dk_scale * kj2[u];
                                    grad_k[[j, u]] += dS_ij * dk_scale * qi[u];
                                }
                            }
                        });
                    });
                }

                // Backprop through linear projections: Q = X W_q; W grads and input grads
                let d_w_q = input.t().dot(&grad_q);
                let d_w_k = input.t().dot(&grad_k);
                let d_w_v = input.t().dot(&grad_v);

                all_param_grads.push(d_w_q);
                all_param_grads.push(d_w_k);
                all_param_grads.push(d_w_v);

                // Accumulate grad_input from projections
                general_mat_mul(1.0, &grad_q, &head.w_q.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_k, &head.w_k.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_v, &head.w_v.t(), 1.0, &mut grad_input_total);

                // Write gating grads into shared accumulators
                let mut grad_wg_slice = grad_w_g.slice_mut(s![.., h_idx..h_idx + 1]);
                grad_wg_slice += &d_wg_col_acc;
                grad_alpha_g[[0, h_idx]] += dalpha;
                grad_beta_g[[0, h_idx]] += dbeta;
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
            // [LEARN] gate polynomial coefficient grads
            let grad_gate_poly = Array2::<f32>::from_shape_vec(
                (1, n_gate_w),
                grad_gate_poly_vec.into_iter().map(|v| v as f32).collect(),
            )
            .unwrap();
            all_param_grads.push(grad_gate_poly);

            if let Some(grad_pe) = grad_cope_pos {
                all_param_grads.push(grad_pe);
            }

            return (grad_input_total, all_param_grads);
        }

        // ===== Dense backward path (original) when window_size is None =====
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
            // z = a_h * xw + b_h; g = phi(z)
            let mut z_col = xw_col.clone();
            z_col.mapv_inplace(|v| a_h * v + b_h);
            let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
            let mut gate_poly = self.gate_poly.clone();
            gate_poly.update_scaling_from_max_abs(max_abs_z);
            let mut g_col = z_col.clone();
            g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);

            with_tls_scores(n, |scores| {
                scores.fill(0.0);
                general_mat_mul(1.0, &q, &k.t(), 0.0, scores);
                *scores *= dk_scale;
                // causal mask
                Self::apply_causal_mask_inplace(scores);
                Self::apply_sliding_window_mask_inplace(scores, self.window_size);

                with_tls_work(n, |work| {
                    // work <- phi(scores)
                    work.assign(scores);
                    work.mapv_inplace(|x| {
                        let sp = x.powi(p_i32);
                        scale * (a * sp + b)
                    });
                    // apply mask again to keep upper triangle zero
                    Self::apply_causal_mask_inplace(work);
                    Self::apply_sliding_window_mask_inplace(work, self.window_size);

                    with_tls_yh(n, self.head_dim, |yh| {
                        // Y_h (pre-gating)
                        yh.fill(0.0);
                        general_mat_mul(1.0, &*work, &v, 0.0, yh); // yh = phi dot V

                        // Apply gating for W_out gradients
                        *yh *= &g_col; // yh now is H_h (gated)

                        // Block rows in W_out corresponding to this head
                        let start = h_idx * self.head_dim;
                        let end = start + self.head_dim;

                        // dL/dW_out block = H_h^T dL/dY
                        {
                            let mut gw_block = grad_w_out.slice_mut(s![start..end, ..]);
                            let yh_t = yh.t();
                            general_mat_mul(1.0, &yh_t, output_grads, 0.0, &mut gw_block);
                        }

                        // Gradient wrt head output slice (gated): g_yh_gated = dL/dY dot
                        // W_out_block^T
                        let w_block = self.w_out.slice(s![start..end, ..]);
                        let w_block_t = w_block.t();
                        yh.fill(0.0);
                        general_mat_mul(1.0, output_grads, &w_block_t, 0.0, yh); // reuse yh as g_yh_gated

                        // Recompute Y_h (pre-gating) to compute grad wrt gating efficiently
                        let y_pre = work.dot(&v); // (N, d_h)

                        // grad_g_col (N,1) = sum_j g_yh_gated[i,j] * y_pre[i,j]
                        let prod = &y_pre * &*yh; // (N, d_h)
                        let grad_g_dg = prod.sum_axis(Axis(1)).insert_axis(Axis(1)); // (N,1), dL/dg

                        // dz = grad_g * phi'(z)
                        let mut dphi_col = z_col.clone();
                        dphi_col.mapv_inplace(|z| gate_poly.backward_scalar(z as f64) as f32);
                        let mut grad_g_col = grad_g_dg.clone();
                        grad_g_col *= &dphi_col; // chain rule

                        // [LEARN] Accumulate gate polynomial coefficient gradients: dL/dw_i = sum
                        // grad(dL/dg) * (c*z)^i
                        let c_gate = gate_poly.scaling as f64;
                        for i in 0..n {
                            let z = z_col[[i, 0]] as f64;
                            let mut power = 1.0_f64; // (c*z)^0
                            let cz = c_gate * z;
                            for wi in 0..n_gate_w {
                                grad_gate_poly_vec[wi] += (grad_g_dg[[i, 0]] as f64) * power;
                                power *= cz;
                            }
                        }

                        // Accumulate gating param grads
                        // dW_g_col = alpha * X^T dot dz
                        let d_wg_col = input.t().dot(&grad_g_col) * a_h;
                        let mut grad_wg_slice = grad_w_g.slice_mut(s![.., h_idx..h_idx + 1]);
                        grad_wg_slice += &d_wg_col;

                        // d alpha = sum_i dz_i * xw_i
                        let dalpha = (&grad_g_col * &xw_col).sum();
                        grad_alpha_g[[0, h_idx]] += dalpha;

                        // d beta = sum_i dz_i
                        let dbeta = grad_g_col.sum();
                        grad_beta_g[[0, h_idx]] += dbeta;

                        // dX from gating path: grad_input += dz · (alpha * w_g_col^T)
                        let mut wg_scaled_t = w_g_col.t().to_owned(); // (1,D)
                        wg_scaled_t *= a_h;
                        let gx = grad_g_col.dot(&wg_scaled_t); // (N,D)
                        grad_input_total += &gx;

                        // Now continue grads through attention path using g_yh_pre = g_yh_gated .*
                        // g
                        let mut g_yh_pre = yh.to_owned(); // g_yh_gated clone
                        g_yh_pre *= &g_col; // (N, d_h)

                        // dL/dV = phi^T dot g_yh_pre
                        let grad_v = work.t().dot(&g_yh_pre);

                        // dL/dphi = g_yh_pre dot V^T
                        work.fill(0.0);
                        general_mat_mul(1.0, &g_yh_pre, &v.t(), 0.0, work); // work = grad_phi (N,N)

                        // grads wrt scalars (vectorized)
                        let mut acc_scale = 0.0f32;
                        let mut acc_a = 0.0f32;
                        let mut acc_b = 0.0f32;
                        azip!(( &dphi in &*work, &sij in &scores.view()) {
                            let sp = sij.powi(p_i32);
                            acc_scale += dphi * (a * sp + b);
                            acc_a += dphi * scale * sp;
                            acc_b += dphi * scale;
                        });
                        grad_scale_scalar += acc_scale;
                        grad_a_scalar += acc_a;
                        grad_b_scalar += acc_b;

                        // dL/dS = dL/dphi * scale * a * p * S^(p-1) (vectorized)
                        azip!(( w in &mut *work, &sij in &scores.view()) {
                            let dphi = *w;
                            let spm1 = if p_i32 == 1 { 1.0 } else { sij.powi(p_i32 - 1) };
                            *w = dphi * scale * a * p_f * spm1;
                        });
                        // masked positions have zero influence
                        Self::apply_causal_mask_inplace(work);
                        Self::apply_sliding_window_mask_inplace(work, self.window_size);

                        // S = (Q K^T) * dk_scale
                        // Therefore dQ = dS dot K * dk_scale, dK = dS^T dot Q * dk_scale
                        let mut grad_q: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                        general_mat_mul(1.0, &*work, &k, 0.0, &mut grad_q);
                        grad_q *= dk_scale;
                        let mut grad_k: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                        general_mat_mul(1.0, &work.t(), &q, 0.0, &mut grad_k);
                        grad_k *= dk_scale;

                        // Backprop through linear projections: Q = X W_q; so dW_q = X^T dQ, dX +=
                        // dQ W_q^T
                        let d_w_q = input.t().dot(&grad_q);
                        let d_w_k = input.t().dot(&grad_k);
                        let d_w_v = input.t().dot(&grad_v);

                        all_param_grads.push(d_w_q);
                        all_param_grads.push(d_w_k);
                        all_param_grads.push(d_w_v);

                        // Accumulate grad_input from projections
                        general_mat_mul(1.0, &grad_q, &head.w_q.t(), 1.0, &mut grad_input_total);
                        general_mat_mul(1.0, &grad_k, &head.w_k.t(), 1.0, &mut grad_input_total);
                        general_mat_mul(1.0, &grad_v, &head.w_v.t(), 1.0, &mut grad_input_total);
                    });
                });
            });
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
        // [LEARN] gate polynomial coefficient grads
        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (1, n_gate_w),
            grad_gate_poly_vec.into_iter().map(|v| v as f32).collect(),
        )
        .unwrap();
        all_param_grads.push(grad_gate_poly);

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
        // Expect 3 per head + w_out + a + b + scale + w_g + alpha_g + beta_g
        let mut expected = self.num_heads * 3 + 1 + 3 + 3 + 1; // + gate_poly_w
        if self.use_cope {
            expected += 1;
        }
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
        self.opt_scale
            .step(&mut self.scale, &param_grads[idx + 2], lr);
        idx += 3;
        self.opt_w_g.step(&mut self.w_g, &param_grads[idx], lr);
        self.opt_alpha_g
            .step(&mut self.alpha_g, &param_grads[idx + 1], lr);
        self.opt_beta_g
            .step(&mut self.beta_g, &param_grads[idx + 2], lr);
        idx += 3;
        // [LEARN] update gate polynomial weights via SGD
        {
            let grad_gate_poly = &param_grads[idx];
            for i in 0..self.gate_poly.weights.len() {
                self.gate_poly.weights[i] -= (lr as f64) * (grad_gate_poly[[0, i]] as f64);
            }
        }
        idx += 1;
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
            + self.gate_poly.weights.len();
        if self.use_cope {
            total += (self.cope_max_pos + 1) * self.head_dim;
        }
        total
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
        <PolyAttention>::compute_gradients(self, _input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        <PolyAttention>::apply_gradients(self, param_grads, lr)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        <PolyAttention>::backward(self, grads, lr)
    }

    fn parameters(&self) -> usize {
        <PolyAttention>::parameters(self)
    }
}
