use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, richards::RichardsCurve, model_config::HeadSelectionStrategy, metrics::PerHeadMetrics};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MoH {
    pub embed_dim: usize,
    pub num_heads: usize,

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
    #[serde(skip_serializing, skip_deserializing)]
    pub metrics: PerHeadMetrics,
    // Trainable auxiliary weights (learnable scalars)
    pub load_balance_param: Array2<f32>, // (1,1)
    pub sparsity_param: Array2<f32>,     // (1,1)
    opt_load_balance: Adam,
    opt_sparsity: Adam,
    // Trainable complexity scalar
    pub complexity_param: Array2<f32>, // (1,1)
    opt_complexity: Adam,
    #[serde(skip)]
    cached_aux_weights: Option<Array2<f32>>, // (1,2) [load_balance_scalar, sparsity_scalar]

    // Cache
    #[serde(skip)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip)]
    cached_g: Option<Array2<f32>>, // (N, num_heads)
    #[serde(skip)]
    cached_m: Option<Array2<f32>>, // (N, 1)
    #[serde(skip)]
    cached_aux_eff_grads: Option<Array2<f32>>, // (N, num_heads) aux grads for eff (g*m)
}

impl MoH {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let mut rng = rand::rng();
        let std_g = (2.0f32 / embed_dim as f32).sqrt();
        let normal_g = Normal::new(0.0, std_g).unwrap();
        let w_g = Array2::<f32>::from_shape_fn((embed_dim, num_heads), |_| normal_g.sample(&mut rng));
        let alpha_g = Array2::<f32>::ones((1, num_heads));
        let beta_g = Array2::<f32>::zeros((1, num_heads));
        let opt_w_g = Adam::new((embed_dim, num_heads), 0.0, 0.0, 0);
        let opt_alpha_g = Adam::new((1, num_heads), 0.0, 0.0, 0);
        let opt_beta_g = Adam::new((1, num_heads), 0.0, 0.0, 0);

        // Richards curve gate (default sigmoid variant, learnable)
        let gate_poly = RichardsCurve::new_learnable(crate::richards::Variant::Sigmoid);

        // Threshold predictor defaults
        let use_learned_threshold = false;

        Self {
            embed_dim,
            num_heads,
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
            metrics: PerHeadMetrics::new(num_heads),
            // Initialize learned auxiliary scalars with sensible defaults so the aux losses are active from start
            load_balance_param: Array2::from_shape_vec((1,1), vec![0.1]).unwrap(),
            sparsity_param: Array2::from_shape_vec((1,1), vec![0.01]).unwrap(),
            opt_load_balance: Adam::new((1,1), 0.0, 0.0, 0),
            opt_sparsity: Adam::new((1,1), 0.0, 0.0, 0),
            complexity_param: Array2::from_shape_vec((1,1), vec![0.01]).unwrap(),
            opt_complexity: Adam::new((1,1), 0.0, 0.0, 0),
            cached_aux_weights: None,
            cached_input: None,
            cached_g: None,
            cached_m: None,
            cached_aux_eff_grads: None,
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // input: (N, embed_dim)
        let n = input.nrows();
        self.cached_input = Some(input.clone());

        if self.use_learned_threshold {
            self.ensure_threshold_predictor();
        }

        // Compute gating per head: g = Richards(alpha * (X·w_g_col) + beta)
        let mut g_mat = Array2::<f32>::zeros((n, self.num_heads));
        let mut m_vec = Array2::<f32>::ones((n, 1));

        // Temporary accumulators for head activity and predictor metrics
        let mut active_sums_tmp = vec![0.0f32; self.num_heads];
        let mut token_counts_tmp = vec![0usize; self.num_heads];
        let mut tau_min_local = f32::INFINITY;
        let mut tau_max_local = f32::NEG_INFINITY;
        let mut tau_count_local = 0usize;
        let mut g_sq_sum_local = 0.0f32;
        let mut g_count_local = 0usize;

        for h_idx in 0..self.num_heads {
            // Compute per-token gating for this head
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
            let mut g_col = Array2::<f32>::zeros(xw_col.raw_dim());
            for i in 0..n {
                g_col[[i, 0]] = gate_poly.forward_scalar((a_h * xw_col[[i, 0]] + b_h) as f64) as f32;
                g_mat[[i, h_idx]] = g_col[[i, 0]];
            }
            // Predictor norm RMS tracking (x·W_g)
            g_sq_sum_local += xw_col.iter().map(|&v| v * v).sum::<f32>();
            g_count_local += n;
        }

        // Learned threshold predictor m = sigmoid(alpha_tau * (X·W_tau) + beta_tau)
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
            m_vec.assign(&z_tau);
            m_vec.mapv_inplace(|z| 1.0 / (1.0 + (-z).exp()));
        }

        // Effective gate per token per head: eff = g * m, but since m is scalar per token, eff_h = g_h * m
        for h_idx in 0..self.num_heads {
            for i in 0..n {
                let eff = g_mat[[i, h_idx]] * m_vec[[i, 0]];
                active_sums_tmp[h_idx] += eff;
                token_counts_tmp[h_idx] += 1;
            }
        }

        // Flush temporary metrics into PerHeadMetrics
        self.metrics.flush_active(&active_sums_tmp, &token_counts_tmp);
        if self.use_learned_threshold && tau_count_local > 0 {
            self.metrics.update_tau_stats(tau_min_local, tau_max_local, tau_count_local);
        }
        self.metrics.update_pred_norm(g_sq_sum_local, g_count_local);













        // Compute auxiliary loss gradients (per-head scalar) and cache as per-token-per-head
        // dL/d_eff contributions so they can be merged in the backward pass.
        let mut m_h = vec![0.0f32; self.num_heads];
        for h in 0..self.num_heads {
            m_h[h] = active_sums_tmp[h] / n as f32;
        }

        // compute scalar contributions (LB and SP) which we'll also cache for learning their weights
        let mut sum_dm = 0.0f32;
        let mut sum_sign = 0.0f32;
        let h_count = self.num_heads as f32;
        for h in 0..self.num_heads {
            let dm = (m_h[h].max(1e-12) * h_count).ln() + 1.0;
            sum_dm += dm;
            let sign = if m_h[h] > 0.0 { 1.0 } else if m_h[h] < 0.0 { -1.0 } else { 0.0 };
            sum_sign += sign;
        }
        // LB scalar and SP scalar defined per-batch (normalized by n)
        let lb_scalar = sum_dm / n as f32; // corresponds to average dm per token
        let sp_scalar = sum_sign / n as f32; // average sign contribution per token

        // combine trainable and static weights
        let lb_total = self.load_balance_param[[0,0]] + self.load_balance_weight;
        let sp_total = self.sparsity_param[[0,0]] + self.sparsity_weight;
        let comp_total = self.complexity_param[[0,0]] + self.complexity_loss_weight;

        // Base per-head contribution from LB and SP (same across tokens)
        let mut base_aux_per_head = vec![0.0f32; self.num_heads];
        if lb_total.abs() > 0.0 || sp_total.abs() > 0.0 {
            for h in 0..self.num_heads {
                let dm = (m_h[h].max(1e-12) * h_count).ln() + 1.0;
                let sign = if m_h[h] > 0.0 { 1.0 } else if m_h[h] < 0.0 { -1.0 } else { 0.0 };
                base_aux_per_head[h] = lb_total * (dm / n as f32) + sp_total * (sign / n as f32);
            }
        }

        // Build auxiliary eff grads matrix combining base per-head terms and per-token complexity contributions
        let mut aux_eff_grads = Array2::<f32>::zeros((n, self.num_heads));
        // accumulator for complexity param gradient (average NIM)
        let mut grad_comp_param = 0.0_f32;
        for i in 0..n {
            // compute eff vector for this token
            let mut s_sum = 0.0_f32;
            let mut s_sum_sq = 0.0_f32;
            for h in 0..self.num_heads {
                let eff = g_mat[[i, h]] * m_vec[[i, 0]];
                s_sum += eff;
                s_sum_sq += eff * eff;
            }
            let s_sum = s_sum.max(1e-6);
            let q = (s_sum_sq / (s_sum * s_sum)).max(1e-12);
            let nim = 1.0 / q;
            grad_comp_param += nim / (n as f32);

            let q_sq = q * q;
            let denom = (q_sq * s_sum * s_sum * s_sum).max(1e-12);
            for h in 0..self.num_heads {
                let eff_h = g_mat[[i, h]] * m_vec[[i, 0]];
                let numer = eff_h * s_sum - s_sum_sq;
                let dnim_deff = -2.0_f32 * numer / denom;
                let comp_contrib = if comp_total.abs() > 0.0 { comp_total * (dnim_deff / (n as f32)) } else { 0.0 };
                aux_eff_grads[[i, h]] = base_aux_per_head[h] + comp_contrib;
            }
        }

        // Cache gating / threshold and auxiliary grads for backward, and the scalar LB/SP and NIM avg
        self.cached_g = Some(g_mat.clone());
        self.cached_m = Some(m_vec.clone());
        self.cached_aux_eff_grads = Some(aux_eff_grads);
        self.cached_aux_weights = Some(Array2::from_shape_vec((1,3), vec![lb_scalar, sp_scalar, grad_comp_param]).unwrap());

        (g_mat, m_vec)
    }

    pub fn compute_gradients(&self, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        // output_grads: (N, num_heads) for g_mat grads, but actually need to handle the eff grads
        // This is complex; for now, assume output_grads is dL/d_eff_mat where eff_mat = g_mat * m_vec.broadcast
        // But to simplify, since it's integrated, perhaps pass the relevant grads.

        // For now, implement basic gradients for gating params.

        let input = self.cached_input.as_ref().expect("forward must be called before compute_gradients");
        let n = input.nrows();

        let mut grad_input_total = Array2::<f32>::zeros((n, self.embed_dim));

        // Gating param grads
        let mut grad_w_g = Array2::<f32>::zeros((self.embed_dim, self.num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_gate_poly_vec = vec![0.0_f64; self.gate_poly.weights().len()];

        // Threshold grads
        let mut grad_w_tau = if self.use_learned_threshold { Some(Array2::<f32>::zeros((self.embed_dim, 1))) } else { None };
        let mut grad_alpha_tau = if self.use_learned_threshold { Some(Array2::<f32>::zeros((1, 1))) } else { None };
        let mut grad_beta_tau = if self.use_learned_threshold { Some(Array2::<f32>::zeros((1, 1))) } else { None };

        // Merge any auxiliary dL/d_eff contributions computed during forward.
        // Assume output_grads is (N, num_heads) dL/d_eff
        let mut total_output_grads = output_grads.clone();
        if let Some(aux) = &self.cached_aux_eff_grads {
            // add in-place
            total_output_grads += aux;
        }

        for i in 0..n {
            let m_i = if self.use_learned_threshold { self.cached_m.as_ref().unwrap()[[i, 0]] } else { 1.0 };
            for h_idx in 0..self.num_heads {
                let d_eff = total_output_grads[[i, h_idx]];
                let d_g = d_eff * m_i;
                let d_m = d_eff * self.cached_g.as_ref().unwrap()[[i, h_idx]];

                // Gate grads
                let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]);
                let xw = input.row(i).dot(&w_g_col)[[0]];
                let a_h = self.alpha_g[[0, h_idx]];
                let b_h = self.beta_g[[0, h_idx]];
                let z = a_h * xw + b_h;
                let dphi_dz = self.gate_poly.backward_scalar(z as f64) as f32;
                let grad_g = d_g * dphi_dz;

                for d in 0..self.embed_dim {
                    grad_w_g[[d, h_idx]] += a_h * input[[i, d]] * grad_g;
                    grad_input_total[[i, d]] += a_h * w_g_col[[d, 0]] * grad_g;
                }
                grad_alpha_g[[0, h_idx]] += grad_g * xw;
                grad_beta_g[[0, h_idx]] += grad_g;

                let gws = self.gate_poly.grad_weights_scalar(z as f64, grad_g as f64);
                if gws.len() > grad_gate_poly_vec.len() {
                    grad_gate_poly_vec.resize(gws.len(), 0.0_f64);
                }
                for (wi, gw) in gws.iter().enumerate() {
                    grad_gate_poly_vec[wi] += *gw;
                }

                // Threshold
                if self.use_learned_threshold {
                    let dm_dz = m_i * (1.0 - m_i);
                    let grad_tau = d_m * dm_dz;
                    let a_t = self.alpha_tau.as_ref().unwrap()[[0, 0]];
                    let w_tau = self.w_tau.as_ref().unwrap();
                    let xw_tau = input.row(i).dot(w_tau)[[0]];
                    for d in 0..self.embed_dim {
                        grad_w_tau.as_mut().unwrap()[[d, 0]] += a_t * input[[i, d]] * grad_tau;
                        grad_input_total[[i, d]] += a_t * w_tau[[d, 0]] * grad_tau;
                    }
                    grad_alpha_tau.as_mut().unwrap()[[0, 0]] += grad_tau * xw_tau;
                    grad_beta_tau.as_mut().unwrap()[[0, 0]] += grad_tau;
                }
            }
        }

        // For auxiliary losses, similar to PolyAttention, but simplified.
        let mut param_grads = vec![
            grad_w_g,
            grad_alpha_g,
            grad_beta_g,
            Array2::from_shape_vec((1, grad_gate_poly_vec.len()), grad_gate_poly_vec.into_iter().map(|v| v as f32).collect()).unwrap(),
        ];

        // Gradients for trainable auxiliary scalars (LB, SP, Complexity)
        let mut grad_lb = 0.0_f32;
        let mut grad_sp = 0.0_f32;
        let mut grad_comp = 0.0_f32;
        if let Some(ws) = &self.cached_aux_weights {
            // The forward defined L_aux = load_balance_param * lb_scalar + sparsity_param * sp_scalar
            // and complexity_param * nim_scalar
            grad_lb = ws[[0,0]];
            grad_sp = ws[[0,1]];
            grad_comp = ws[[0,2]];
        }
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_lb]).unwrap());
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_sp]).unwrap());
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_comp]).unwrap());

        // push threshold grads if used (kept after aux scalar grads so apply_gradients indexing matches)
        if self.use_learned_threshold {
            param_grads.push(grad_w_tau.unwrap());
            param_grads.push(grad_alpha_tau.unwrap());
            param_grads.push(grad_beta_tau.unwrap());
        }

        (grad_input_total, param_grads)
    }

    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        let mut idx = 0;
        self.opt_w_g.step(&mut self.w_g, &param_grads[idx], lr);
        idx += 1;
        self.opt_alpha_g.step(&mut self.alpha_g, &param_grads[idx], lr);
        idx += 1;
        self.opt_beta_g.step(&mut self.beta_g, &param_grads[idx], lr);
        idx += 1;
        let grad_gate_vec: Vec<f64> = param_grads[idx].iter().map(|&x| x as f64).collect();
        self.gate_poly.step(&grad_gate_vec, lr as f64);
        idx += 1;
        // Step auxiliary scalar optimizers (load-balance, sparsity, complexity)
        if idx < param_grads.len() {
            self.opt_load_balance.step(&mut self.load_balance_param, &param_grads[idx], lr);
            idx += 1;
        }
        if idx < param_grads.len() {
            self.opt_sparsity.step(&mut self.sparsity_param, &param_grads[idx], lr);
            idx += 1;
        }
        if idx < param_grads.len() {
            self.opt_complexity.step(&mut self.complexity_param, &param_grads[idx], lr);
            idx += 1;
        }
        if self.use_learned_threshold {
            if let (Some(wt), Some(opt)) = (&mut self.w_tau, &mut self.opt_w_tau) {
                opt.step(wt, &param_grads[idx], lr);
            }
            idx += 1;
            if let (Some(at), Some(opt)) = (&mut self.alpha_tau, &mut self.opt_alpha_tau) {
                opt.step(at, &param_grads[idx], lr);
            }
            idx += 1;
            if let (Some(bt), Some(opt)) = (&mut self.beta_tau, &mut self.opt_beta_tau) {
                opt.step(bt, &param_grads[idx], lr);
            }
        }
        Ok(())
    }

    // Initialize or ensure learned threshold predictor parameters
    fn ensure_threshold_predictor(&mut self) {
        if self.w_tau.is_none() {
            let std_tau = (2.0f32 / self.embed_dim as f32).sqrt();
            let normal_tau = Normal::new(0.0, std_tau).unwrap();
            let mut rng = rand::rng();
            let wtau = Array2::<f32>::from_shape_fn((self.embed_dim, 1), |_| normal_tau.sample(&mut rng));
            self.w_tau = Some(wtau);
            self.opt_w_tau = Some(Adam::new((self.embed_dim, 1), 0.0, 0.0, 0));
        }
        if self.alpha_tau.is_none() {
            self.alpha_tau = Some(Array2::<f32>::from_shape_vec((1, 1), vec![1.0]).unwrap());
            self.opt_alpha_tau = Some(Adam::new((1, 1), 0.0, 0.0, 0));
        }
        if self.beta_tau.is_none() {
            self.beta_tau = Some(Array2::<f32>::from_shape_vec((1, 1), vec![0.0]).unwrap());
            self.opt_beta_tau = Some(Adam::new((1, 1), 0.0, 0.0, 0));
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
                // Initialize trainable scalar params with provided strategy values so they take effect immediately
                self.load_balance_param[[0,0]] = *load_balance_weight;
                self.sparsity_param[[0,0]] = *sparsity_weight;
            }
            HeadSelectionStrategy::ProgressiveLayerMoH { top_max_heads, min_heads, decay_fraction: _, per_layer: _, load_balance_weight, complexity_loss_weight, sparsity_weight, per_layer_kv: _ } => {
                // Treat progressive strategy as adaptive per-layer: clamp to available heads
                let top = top_max_heads.unwrap_or(self.num_heads);
                self.max_heads = usize::min(top, self.num_heads);
                self.min_heads = usize::min(*min_heads, self.max_heads);
                self.complexity_loss_weight = *complexity_loss_weight;
                self.load_balance_weight = *load_balance_weight;
                self.sparsity_weight = *sparsity_weight;
                self.use_learned_threshold = true;
                self.ensure_threshold_predictor();
                self.load_balance_param[[0,0]] = self.load_balance_weight;
                self.sparsity_param[[0,0]] = self.sparsity_weight;
            }
        }
        // reset metrics whenever strategy changes
        self.metrics.reset_head_metrics();
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        self.metrics.get_head_metrics_and_reset()
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        self.metrics.take_tau_metrics()
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        self.metrics.take_pred_norm()
    }

    pub fn parameters(&self) -> usize {
        let mut total = self.w_g.len() + self.alpha_g.len() + self.beta_g.len() + self.gate_poly.weights().len();
        if self.use_learned_threshold {
            total += self.embed_dim + 1 + 1; // w_tau + alpha_tau + beta_tau
        }
        total
    }
}
