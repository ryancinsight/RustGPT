use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

use crate::{adam::Adam, llm::Layer, richards::RichardsCurve};
use crate::metrics::{select_top_k, compute_nim, NimMetrics};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MoE {
    pub embed_dim: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub load_balance_param: Array2<f32>, // (1,1) trainable scalar
    opt_load_balance: Adam,
    // Trainable complexity scalar
    pub complexity_param: Array2<f32>, // (1,1)
    opt_complexity: Adam,
    // Regularizer weights (not trainable scalars)
    pub load_balance_weight: f32,
    pub complexity_loss_weight: f32,
    pub sparsity_weight: f32,

    // Gating network: scores = Richards(alpha * (X·W) + beta) per expert
    pub gating_w: Array2<f32>,     // (embed_dim, num_experts)
    pub gating_alpha: Array2<f32>, // (1, num_experts)
    pub gating_beta: Array2<f32>,  // (1, num_experts)
    pub gate_curve: RichardsCurve,

    // Experts: each is a small PRiGLU with reduced hidden_dim to maintain param count
    pub experts: Vec<crate::priglu::PRiGLU>,

    // Optimizers
    opt_gating_w: Adam,
    opt_gating_alpha: Adam,
    opt_gating_beta: Adam,

    // Cache
    #[serde(skip)]
    cached_input: Option<Array2<f32>>, 

    // Metrics: average number of important mixture experts per token (NIM)
    #[serde(skip_serializing, skip_deserializing)]
    pub nim_metrics: NimMetrics,
}

impl MoE {
    pub fn new(
        embed_dim: usize,
        num_experts: usize,
        top_k: usize,
        hidden_dim: usize,
        load_balance_weight: f32,
        complexity_loss_weight: f32,
        sparsity_weight: f32,
    ) -> Self {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let gating_w = Array2::from_shape_fn((embed_dim, num_experts), |_| normal.sample(&mut rng));
        let gating_alpha = Array2::zeros((1, num_experts));
        let gating_beta = Array2::zeros((1, num_experts));

        let gate_curve = RichardsCurve::sigmoid(false);

        let mut experts = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            experts.push(crate::priglu::PRiGLU::new(embed_dim, hidden_dim));
        }

        let opt_gating_w = Adam::new((embed_dim, num_experts), 0.0, 0.0, 0);
        let opt_gating_alpha = Adam::new((1, num_experts), 0.0, 0.0, 0);
        let opt_gating_beta = Adam::new((1, num_experts), 0.0, 0.0, 0);

        let mut s = Self {
            embed_dim,
            num_experts,
            top_k,
            // default trainable load_balance scalar initialized to provided weight
            load_balance_param: Array2::from_shape_vec((1,1), vec![0.1]).unwrap(),
            opt_load_balance: Adam::new((1,1), 0.0, 0.0, 0),
            // complexity trainable scalar
            complexity_param: Array2::from_shape_vec((1,1), vec![0.01]).unwrap(),
            opt_complexity: Adam::new((1,1), 0.0, 0.0, 0),
            // set the static weights to defaults for now; builder should update if desired
            load_balance_weight: 0.1,
            complexity_loss_weight: 0.01,
            sparsity_weight: 0.001,
            gating_w,
            gating_alpha,
            gating_beta,
            gate_curve,
            experts,
            opt_gating_w,
            opt_gating_alpha,
            opt_gating_beta,
            cached_input: None,
            nim_metrics: NimMetrics::new(),
        };

        // apply provided weights
        s.load_balance_weight = load_balance_weight;
        s.complexity_loss_weight = complexity_loss_weight;
        s.sparsity_weight = sparsity_weight;
        // initialize trainable scalars
        s.load_balance_param[[0,0]] = load_balance_weight;
        s.complexity_param[[0,0]] = complexity_loss_weight;

        s
    }

    pub fn total_parameters(&self) -> usize {
        let gating_params = self.gating_w.len() + self.gating_alpha.len() + self.gating_beta.len();
        let expert_params: usize = self.experts.iter().map(|e| e.parameters()).sum();
        gating_params + expert_params
    }
}

impl Layer for MoE {
    fn layer_type(&self) -> &str {
        "MoE"
    }

    fn forward(&mut self, input: Rc<Array2<f32>>) -> Array2<f32> {
        let n = input.nrows();
        let d = input.ncols();
        assert_eq!(d, self.embed_dim);

        // Cache input for backward
        self.cached_input = Some((*input).clone());

        // Compute gating scores: (N, num_experts)
        let mut gating_scores = input.dot(&self.gating_w); // (N, num_experts)
        let mut output = Array2::zeros((n, d));

        for e in 0..self.num_experts {
            let alpha_e = self.gating_alpha[[0, e]];
            let beta_e = self.gating_beta[[0, e]];

            // Update curve scaling per expert based on max abs
            let col = gating_scores.slice(s![.., e..e+1]);
            let max_abs = col.iter().fold(0.0_f64, |m, &v| m.max(v.abs() as f64));
            let mut curve = self.gate_curve.clone();
            curve.update_scaling_from_max_abs(max_abs);

            // Apply Richards to gating_scores[:, e]
            for i in 0..n {
                let z = alpha_e * gating_scores[[i, e]] + beta_e;
                gating_scores[[i, e]] = curve.forward_scalar(z as f64) as f32;
            }
        }

        // For each token, select top-k experts and update NIM metrics
        for i in 0..n {
            let row_scores: Vec<f32> = gating_scores.row(i).to_vec();
            let (top_indices, weights) = select_top_k(&row_scores, self.top_k);

            // NIM (soft, learned) over all experts - measures effective diversity
            let nim_count = compute_nim(&row_scores);
            self.nim_metrics.add(nim_count);
            
            // Actual expert count (hard selection) - measures computational cost
            self.nim_metrics.add_actual_count(top_indices.len());

            // Weighted sum of expert outputs
            let mut mixed = Array2::zeros((1, d));
            for (idx, &e) in top_indices.iter().enumerate() {
                let expert_input = input.slice(s![i..i+1, ..]).to_owned();
                let expert_out = self.experts[e].forward(Rc::new(expert_input));
                let w = weights[idx];
                mixed.scaled_add(w, &expert_out);
            }
            output.row_mut(i).assign(&mixed.row(0));
        }

        output
    }

    fn compute_gradients(&self, input: Rc<Array2<f32>>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let n = input.nrows();
        let d = input.ncols();

        let mut grad_load_balance = 0.0_f32;

        // Compute gating scores (same as forward)
        let mut gating_scores = input.dot(&self.gating_w);
        for e in 0..self.num_experts {
            let alpha_e = self.gating_alpha[[0, e]];
            let beta_e = self.gating_beta[[0, e]];
            let max_abs = gating_scores.slice(s![.., e..e+1]).iter().fold(0.0_f64, |m, &v| m.max(v.abs() as f64)).max(1e-6);
            let mut curve = self.gate_curve.clone();
            curve.update_scaling_from_max_abs(max_abs);
            for i in 0..n {
                let z = alpha_e * gating_scores[[i, e]] + beta_e;
                gating_scores[[i, e]] = curve.forward_scalar(z as f64) as f32;
            }
        }

        // Prepare accumulators
        let mut gating_w_grad = Array2::<f32>::zeros((self.embed_dim, self.num_experts));
        let mut gating_alpha_grad = Array2::<f32>::zeros((1, self.num_experts));
        let mut gating_beta_grad = Array2::<f32>::zeros((1, self.num_experts));
        let mut input_grads_accum = Array2::zeros((n, d));

        // Helper: compute expert output for a single token row (as in PRiGLU.forward)
        let compute_expert_output = |e: usize, x_row: &Array2<f32>| -> Array2<f32> {
            // x_row shape (1, d)
            let sw = &*x_row;
            let x1: Array2<f32> = sw.dot(&self.experts[e].w1);
            let x2: Array2<f32> = sw.dot(&self.experts[e].w2);
            let mut swish = Array2::<f32>::zeros(x1.raw_dim());
            let mut gate_sigma = Array2::<f32>::zeros(x2.raw_dim());
            ndarray::Zip::from(&mut swish)
                .and(&x1)
                .for_each(|o, &v| {
                    *o = self.experts[e].swish_activation.forward_scalar(v as f64) as f32;
                });
            ndarray::Zip::from(&mut gate_sigma)
                .and(&x2)
                .for_each(|o, &v| {
                    *o = self.experts[e].gate_curve.forward_scalar(v as f64) as f32;
                });
            let gated = &swish * &gate_sigma;
            let out = gated.dot(&self.experts[e].w_out) + sw;
            out
        };

        // Track expert assignments and weights for batching expert grads
        let mut expert_token_indices: Vec<Vec<usize>> = vec![vec![]; self.num_experts];
        let mut expert_token_weights: Vec<Vec<f32>> = vec![vec![]; self.num_experts];

        // Precompute per-expert curve instances for backward scaling
        let mut curves_per_expert: Vec<RichardsCurve> = Vec::with_capacity(self.num_experts);
        for e in 0..self.num_experts {
            let max_abs = gating_scores
                .slice(s![.., e..e + 1])
                .iter()
                .fold(0.0_f64, |m, &v| m.max(v.abs() as f64))
                .max(1e-6);
            let mut curve = self.gate_curve.clone();
            curve.update_scaling_from_max_abs(max_abs);
            curves_per_expert.push(curve);
        }

        // Mixture and gating gradients per token (top-k normalized weights)
        for i in 0..n {
            // Top-k selection
            let row_scores: Vec<f32> = gating_scores.row(i).to_vec();
            let (top_indices, weights_top) = select_top_k(&row_scores, self.top_k);

            // Expert outputs for selected experts
            let x_row = input.slice(s![i..i + 1, ..]).to_owned();
            let mut expert_outs: Vec<Array2<f32>> = Vec::with_capacity(self.top_k);
            for &e in &top_indices {
                let out_e = compute_expert_output(e, &x_row);
                expert_outs.push(out_e);
            }

            // dL/dw_e = output_grads_i ⋅ expert_out_e
            let og_i = output_grads.slice(s![i..i + 1, ..]).to_owned();
            let mut dL_d_w: Vec<f32> = Vec::with_capacity(self.top_k);
            for out_e in &expert_outs {
                // dot over embedding dims
                let mut dot: f32 = 0.0;
                ndarray::Zip::from(&og_i).and(out_e).for_each(|&g, &y| {
                    dot += g * y;
                });
                dL_d_w.push(dot);
            }

            // Convert to dL/d s via normalized weights identity over top-k
            let sum_pw = weights_top
                .iter()
                .zip(dL_d_w.iter())
                .map(|(p, g)| p * g)
                .sum::<f32>();
            for (idx, &e) in top_indices.iter().enumerate() {
                let dL_d_w_e = dL_d_w[idx];
                // Robust normalization: use larger epsilon to prevent division by near-zero
                let sum_top = weights_top.iter().sum::<f32>().max(1e-3);
                let dL_d_s_e = (dL_d_w_e - sum_pw) / sum_top; // 1/S * (dL/dw_e - Σ p_j dL/dw_j)
                // Additional safety: clamp gradient to prevent explosion
                let dL_d_s_e = dL_d_s_e.clamp(-100.0, 100.0);

                // Chain to z = alpha * (x·W) + beta
                let pre = input.row(i).dot(&self.gating_w.column(e));
                let z = self.gating_alpha[[0, e]] * pre + self.gating_beta[[0, e]];
                let deriv = curves_per_expert[e].backward_scalar(z as f64) as f32;

                // Param grads
                let alpha_e = self.gating_alpha[[0, e]];
                // d s / d W[:,e] = deriv * alpha_e * x
                for k in 0..d {
                    gating_w_grad[[k, e]] += dL_d_s_e * deriv * alpha_e * input[[i, k]];
                }
                // d s / d alpha_e = deriv * pre
                gating_alpha_grad[[0, e]] += dL_d_s_e * deriv * pre;
                // d s / d beta_e = deriv
                gating_beta_grad[[0, e]] += dL_d_s_e * deriv;
                // d s / d x = deriv * alpha_e * W[:,e]
                for k in 0..d {
                    input_grads_accum[[i, k]] += dL_d_s_e * deriv * alpha_e * self.gating_w[[k, e]];
                }

                // Record expert assignment and weight for expert backprop
                expert_token_indices[e].push(i);
                expert_token_weights[e].push(weights_top[idx]);
            }
        }

        // Load-balance loss (KL to uniform over mean gating probabilities across batch)
        if self.load_balance_param[[0,0]].abs() > 0.0 {
            // Compute p_all per token (normalize over all experts)
            // First compute mean per expert m_e
            let mut m_e = vec![0.0f32; self.num_experts];
            for i in 0..n {
                let s_sum = (0..self.num_experts)
                    .map(|e| gating_scores[[i, e]])
                    .sum::<f32>()
                    .max(1e-6);
                for e in 0..self.num_experts {
                    let p = gating_scores[[i, e]] / s_sum;
                    m_e[e] += p / (n as f32);
                }
            }
            // ∂ KL / ∂ m_e = log(m_e * E) + 1
            let e_count = self.num_experts as f32;
            let dm = m_e
                .iter()
                .map(|&me| (me.max(1e-12) * e_count).ln() + 1.0)
                .collect::<Vec<f32>>();

            // store scalar gradient for load_balance param: average dm per token
            let sum_dm: f32 = dm.iter().sum();
            grad_load_balance = sum_dm / n as f32;

            // Backprop to s via p normalization per token
            for i in 0..n {
                let s_sum = (0..self.num_experts)
                    .map(|e| gating_scores[[i, e]])
                    .sum::<f32>()
                    .max(1e-6);
                // grad wrt p_i,e: λ * dm_e / N
                let mut sum_p_g: f32 = 0.0;
                for e in 0..self.num_experts {
                    let p = gating_scores[[i, e]] / s_sum;
                    let g_pe = self.load_balance_param[[0,0]] * dm[e] / (n as f32);
                    sum_p_g += p * g_pe;
                }
                for e in 0..self.num_experts {
                    let pre = input.row(i).dot(&self.gating_w.column(e));
                    let z = self.gating_alpha[[0, e]] * pre + self.gating_beta[[0, e]];
                    let deriv = curves_per_expert[e].backward_scalar(z as f64) as f32;
                    let g_pe = self.load_balance_param[[0,0]] * dm[e] / (n as f32);
                    let dL_d_s_e = (g_pe - sum_p_g) / s_sum; // 1/S * (∂L/∂p_e - Σ p_j ∂L/∂p_j)

                    let alpha_e = self.gating_alpha[[0, e]];
                    for k in 0..d {
                        gating_w_grad[[k, e]] += dL_d_s_e * deriv * alpha_e * input[[i, k]];
                    }
                    gating_alpha_grad[[0, e]] += dL_d_s_e * deriv * pre;
                    gating_beta_grad[[0, e]] += dL_d_s_e * deriv;
                    for k in 0..d {
                        input_grads_accum[[i, k]] += dL_d_s_e * deriv * alpha_e * self.gating_w[[k, e]];
                    }
                }
            }
        }

        // Additional regularizers: complexity (NIM-based) and sparsity (L1 proxy)
        // Use trainable scalars combined with static weights:
        // total_comp = complexity_param + complexity_loss_weight
        // total_sp = sparsity_param + sparsity_weight
        let total_comp = self.complexity_param[[0,0]] + self.complexity_loss_weight;
        let total_sp = self.sparsity_weight; // no trainable sparsity param in MoE; static only

        // Accumulators for parameter gradients of the trainable scalars
        let mut grad_complexity_param = 0.0_f32;
        let grad_sparsity_param = 0.0_f32; // unused unless later added as trainable

        if total_comp.abs() > 0.0 || total_sp.abs() > 0.0 {
            // For each token compute NIM and its derivative wrt gating scores
            for i in 0..n {
                // gather s vector for token i
                let mut s_sum = 0.0_f32;
                let mut s_sum_sq = 0.0_f32;
                for e in 0..self.num_experts {
                    let s = gating_scores[[i, e]];
                    s_sum += s;
                    s_sum_sq += s * s;
                }
                let s_sum = s_sum.max(1e-6);
                let q = (s_sum_sq / (s_sum * s_sum)).max(1e-12);
                let nim = 1.0 / q;
                // Accumulate gradient for complexity scalar (L = comp_param * mean(nim))
                grad_complexity_param += nim / (n as f32);

                // Derivative d NIM / d s_j = -2 / (Q^2 * S^3) * (s_j * S - sum_sq)
                let q_sq = q * q;
                let s_sum_f = s_sum;
                let denom = (q_sq * s_sum_f * s_sum_f * s_sum_f).max(1e-12);
                for e in 0..self.num_experts {
                    let s_j = gating_scores[[i, e]];
                    let numer = s_j * s_sum_f - s_sum_sq;
                    let dnim_ds = -2.0_f32 * numer / denom;

                    // total dL/d s contribution for this token/expert
                    let mut dL_d_s = 0.0_f32;
                    if total_comp.abs() > 0.0 {
                        dL_d_s += total_comp * (dnim_ds / (n as f32));
                    }
                    if total_sp.abs() > 0.0 {
                        let sign = if s_j > 0.0 { 1.0 } else if s_j < 0.0 { -1.0 } else { 0.0 };
                        dL_d_s += total_sp * (sign / (n as f32));
                    }

                    if dL_d_s.abs() > 0.0 {
                        let pre = input.row(i).dot(&self.gating_w.column(e));
                        let z = self.gating_alpha[[0, e]] * pre + self.gating_beta[[0, e]];
                        let deriv = curves_per_expert[e].backward_scalar(z as f64) as f32;
                        let alpha_e = self.gating_alpha[[0, e]];
                        for k in 0..d {
                            gating_w_grad[[k, e]] += dL_d_s * deriv * alpha_e * input[[i, k]];
                        }
                        gating_alpha_grad[[0, e]] += dL_d_s * deriv * pre;
                        gating_beta_grad[[0, e]] += dL_d_s * deriv;
                        for k in 0..d {
                            input_grads_accum[[i, k]] += dL_d_s * deriv * alpha_e * self.gating_w[[k, e]];
                        }
                    }
                }
            }
        }

        // Compute gradients for experts using weighted output_grads
        let mut expert_param_grads = Vec::new();
        for e in 0..self.num_experts {
            let indices = &expert_token_indices[e];
            if indices.is_empty() {
                expert_param_grads.push(Vec::new());
                continue;
            }
            let weights = &expert_token_weights[e];
            let mut expert_input = Array2::zeros((indices.len(), d));
            let mut expert_output_grads = Array2::zeros((indices.len(), d));
            for (j, &i) in indices.iter().enumerate() {
                expert_input.row_mut(j).assign(&input.row(i));
                // scale OG by mixture weight
                expert_output_grads
                    .row_mut(j)
                    .assign(&output_grads.row(i));
                expert_output_grads.row_mut(j).mapv_inplace(|v| v * weights[j]);
            }
            let (expert_input_grads, params) =
                self.experts[e].compute_gradients(Rc::new(expert_input), &expert_output_grads);
            expert_param_grads.push(params);
            // Accumulate input grads from experts
            for (j, &i) in indices.iter().enumerate() {
                input_grads_accum
                    .row_mut(i)
                    .scaled_add(1.0, &expert_input_grads.row(j));
            }
        }

        // Package gradients
        let gating_w_grad = gating_w_grad;
        let gating_alpha_grad = gating_alpha_grad;
        let gating_beta_grad = gating_beta_grad;

        let mut param_grads = vec![
            gating_w_grad,
            gating_alpha_grad,
            gating_beta_grad,
        ];
        // push grad for trainable load-balance scalar
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_load_balance]).unwrap());
        // push grad for trainable complexity scalar
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_complexity_param]).unwrap());
        // push grad for sparsity param placeholder (not trainable here) as 0
        param_grads.push(Array2::from_shape_vec((1,1), vec![grad_sparsity_param]).unwrap());
        param_grads.extend(expert_param_grads.into_iter().flatten());

        (input_grads_accum, param_grads)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = Rc::new(self.cached_input.as_ref().expect("forward must be called first").clone());
        let (input_grads, param_grads) = self.compute_gradients(input, grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.total_parameters()
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        if param_grads.len() < 3 {
            return Err(crate::errors::ModelError::GradientError { message: "Insufficient param grads".to_string() });
        }

        // Update gating
        self.opt_gating_w.step(&mut self.gating_w, &param_grads[0], lr);
        self.opt_gating_alpha.step(&mut self.gating_alpha, &param_grads[1], lr);
        self.opt_gating_beta.step(&mut self.gating_beta, &param_grads[2], lr);

        // Step trainable load-balance param if provided (it's at index 3)
        let mut idx = 3;
        if param_grads.len() > idx {
            self.opt_load_balance.step(&mut self.load_balance_param, &param_grads[idx], lr);
            idx += 1;
        }
        // Step trainable complexity param (it's at index 4)
        if param_grads.len() > idx {
            self.opt_complexity.step(&mut self.complexity_param, &param_grads[idx], lr);
            idx += 1;
        }

        // Optionally handle sparsity param if made trainable in future (param at current idx)
        if param_grads.len() > idx {
            // currently we do not have a sparsity optimizer for MoE; skip stepping but consume slot
            idx += 1;
        }

        // Update experts: each SwiGLU returns 5 gradient blocks
        for expert in &mut self.experts {
            if idx + 4 < param_grads.len() {
                expert.apply_gradients(&param_grads[idx..idx+5], lr)?;
                idx += 5;
            } else {
                break;
            }
        }

        Ok(())
    }
}

// Expose NIM metrics averaged per token since last reset
// Returns (avg_nim_experts_per_token, tokens_count) if any tokens were recorded
impl MoE {
    pub fn get_nim_metrics_and_reset(&mut self) -> Option<(f32, usize)> {
        self.nim_metrics.get_and_reset()
    }
    
    pub fn get_actual_experts_and_reset(&mut self) -> Option<f32> {
        self.nim_metrics.get_actual_and_reset()
    }
}
