use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    adam::Adam,
    mixtures::{routing::{apply_selection_algorithm, RoutingConfig, SelectionAlgorithm}, moh::{HeadSelectionConfig, HeadSelectionStrategy}},
    richards::RichardsGate,
    rng::get_rng,
    mixtures::threshold::ThresholdPredictor,
};

/// Shared Mixture-of-Heads (MoH) gating module.
///
/// This owns the gating parameters and metrics used to produce per-token per-head
/// activation weights. It is intended to be reusable across attention and SSM mixers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoHGating {
    /// Per-head gating projection: X·W_g
    pub w_g: Array2<f32>,     // (embed_dim, num_heads)
    pub alpha_g: Array2<f32>, // (1, num_heads)
    pub beta_g: Array2<f32>,  // (1, num_heads)

    pub opt_w_g: Adam,
    pub opt_alpha_g: Adam,
    pub opt_beta_g: Adam,

    /// Learnable Richards gate used to map z -> g in (0,1)
    pub gate: RichardsGate,

    /// Head selection configuration and metrics
    pub head_selection_config: HeadSelectionConfig,

    /// Optional learned threshold predictor (AutoDeco-inspired)
    pub threshold_predictor: Option<ThresholdPredictor>,

    pub opt_w_tau: Option<Adam>,
    pub opt_b_tau: Option<Adam>,
    pub opt_w2_tau: Option<Adam>,
    pub opt_b2_tau: Option<Adam>,
    pub opt_cond_w_tau: Option<Adam>,

    /// Cached SoftTopP mask (tokens x heads) from last forward pass.
    #[serde(skip_serializing, skip_deserializing)]
    pub cached_soft_top_p_mask: Option<Array2<f32>>,
}

impl MoHGating {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let mut rng = get_rng();
        let std_g = (2.0f32 / embed_dim.max(1) as f32).sqrt();
        let normal_g = Normal::new(0.0, std_g as f64).unwrap();

        let w_g = Array2::<f32>::from_shape_fn((embed_dim, num_heads), |_| normal_g.sample(&mut rng) as f32);
        let alpha_g = Array2::<f32>::ones((1, num_heads));
        let beta_g = Array2::<f32>::zeros((1, num_heads));

        let mut opt_w_g = Adam::new((embed_dim, num_heads));
        let mut opt_alpha_g = Adam::new((1, num_heads));
        let mut opt_beta_g = Adam::new((1, num_heads));
        opt_w_g.set_amsgrad(true);
        opt_alpha_g.set_amsgrad(true);
        opt_beta_g.set_amsgrad(true);

        Self {
            w_g,
            alpha_g,
            beta_g,
            opt_w_g,
            opt_alpha_g,
            opt_beta_g,
            gate: RichardsGate::new(),
            head_selection_config: HeadSelectionConfig::default(),
            threshold_predictor: None,
            opt_w_tau: None,
            opt_b_tau: None,
            opt_w2_tau: None,
            opt_b2_tau: None,
            opt_cond_w_tau: None,
            cached_soft_top_p_mask: None,
        }
    }

    /// Configure the gating strategy (and initialize predictor/optimizers if required).
    pub fn set_head_selection_config(&mut self, strategy: &HeadSelectionStrategy) {
        let num_heads = self.w_g.ncols();
        let embed_dim = self.w_g.nrows();
        self.head_selection_config = HeadSelectionConfig::from_strategy(strategy, num_heads);

        if self.head_selection_config.gating.use_learned_predictor && self.threshold_predictor.is_none() {
            let predictor_hidden_dim = 128.min(embed_dim / 2).max(32);
            self.threshold_predictor = Some(ThresholdPredictor::new_with_cond(
                embed_dim,
                predictor_hidden_dim,
                num_heads,
                embed_dim,
            ));

            self.opt_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
            self.opt_b_tau = Some(Adam::new((predictor_hidden_dim, 1)));
            self.opt_w2_tau = Some(Adam::new((predictor_hidden_dim, num_heads)));
            self.opt_b2_tau = Some(Adam::new((num_heads, 1)));
            self.opt_cond_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
        }
    }

    /// Compute per-token per-head weights (tokens x heads) and update MoH metrics.
    ///
    /// Returns weights in [0,1] (not necessarily summing to 1).
    pub fn forward_weights(
        &mut self,
        input: &Array2<f32>,
        token_threshold_scale: Option<&Array2<f32>>,
        token_latent_features: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let n = input.nrows();
        let num_heads = self.w_g.ncols();
        if n == 0 || num_heads == 0 {
            return Array2::<f32>::zeros((n, num_heads));
        }

        self.cached_soft_top_p_mask = None;

        // Compute X·W_g once: shape (n, num_heads)
        let xw = input.dot(&self.w_g);

        // Compute raw gate values g (tokens x heads) using Richards gate.
        let mut g_mat = Array2::<f32>::zeros((n, num_heads));
        let mut g_sq_sum = 0.0f32;
        let mut g_count = 0usize;

        for h in 0..num_heads {
            let a_h = self.alpha_g[[0, h]];
            let b_h = self.beta_g[[0, h]];

            // Track predictor RMS based on xw pre-activation.
            for i in 0..n {
                let v = xw[[i, h]];
                g_sq_sum += v * v;
            }
            g_count += n;

            // Update Richards gate scaling for this head based on z-range.
            let mut max_abs_z = 0.0_f64;
            for i in 0..n {
                let z = (a_h * xw[[i, h]] + b_h) as f64;
                max_abs_z = max_abs_z.max(z.abs());
            }
            let _ = self.gate.update_scaling_from_max_abs(max_abs_z);

            // g = Richards(z)
            for i in 0..n {
                let z = (a_h * xw[[i, h]] + b_h) as f64;
                g_mat[[i, h]] = self.gate.curve.forward_scalar(z) as f32;
            }
        }

        // Compute head selection mask m (tokens x heads).
        let mut m_mat = Array2::<f32>::ones((n, num_heads));
        if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &mut self.threshold_predictor {
                let mut cond_input = input.to_owned();
                if let Some(scale) = token_threshold_scale {
                    let d = cond_input.ncols();
                    for i in 0..n {
                        let s0 = scale[[i, 0]];
                        for j in 0..d {
                            cond_input[[i, j]] *= s0;
                        }
                    }
                }
                let mut t = predictor.predict_with_condition(
                    &cond_input.view(),
                    token_latent_features.map(|f| f.view()),
                );

                let m = self.head_selection_config.threshold_modulation;
                t.mapv_inplace(|v| v * m);

                // Normalize each row to sum=k (like the attention implementation).
                let k = self.head_selection_config.gating.num_active as f32;
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for h in 0..num_heads {
                        sum += t[[i, h]];
                    }
                    if sum > 0.0 {
                        let s = k / sum;
                        for h in 0..num_heads {
                            t[[i, h]] *= s;
                        }
                    }
                }

                m_mat.assign(&t);
            }

            // Update tau metrics based on mask.
            self.head_selection_config.metrics_tau_count += n;
            for v in m_mat.iter() {
                let vv = if v.is_finite() { *v } else { 0.0 };
                if vv < self.head_selection_config.metrics_tau_min {
                    self.head_selection_config.metrics_tau_min = vv;
                }
                if vv > self.head_selection_config.metrics_tau_max {
                    self.head_selection_config.metrics_tau_max = vv;
                }
                self.head_selection_config.metrics_tau_sum += vv;
            }
        } else if self.head_selection_config.gating.use_soft_top_p {
            // Use shared routing SoftTopP on g_mat.
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP {
                    top_p: self.head_selection_config.gating.top_p,
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let mut weights = apply_selection_algorithm(&g_mat.view(), &cfg);

            // Scale and clamp to mimic "active heads" semantics.
            let activation_scale = self.head_selection_config.max_heads.max(1) as f32;
            weights.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));

            let m = self.head_selection_config.threshold_modulation;
            weights.mapv_inplace(|v| (v * m).clamp(0.0, 1.0));

            if let Some(scale) = token_threshold_scale {
                for i in 0..n {
                    let s0 = scale[[i, 0]];
                    for h in 0..num_heads {
                        weights[[i, h]] = (weights[[i, h]] * s0).clamp(0.0, 1.0);
                    }
                }
            }

            self.cached_soft_top_p_mask = Some(weights.clone());
            m_mat.assign(&weights);

            // Update tau metrics based on mask.
            self.head_selection_config.metrics_tau_count += n;
            for v in m_mat.iter() {
                let vv = if v.is_finite() { *v } else { 0.0 };
                if vv < self.head_selection_config.metrics_tau_min {
                    self.head_selection_config.metrics_tau_min = vv;
                }
                if vv > self.head_selection_config.metrics_tau_max {
                    self.head_selection_config.metrics_tau_max = vv;
                }
                self.head_selection_config.metrics_tau_sum += vv;
            }
        }

        // Effective weights.
        let mut eff = &g_mat * &m_mat;
        eff.mapv_inplace(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });

        // Update gating metrics.
        self.head_selection_config.metrics_g_sq_sum += g_sq_sum;
        self.head_selection_config.metrics_g_count += g_count;
        self.head_selection_config.update_metrics(&eff.view());

        eff
    }

    pub fn moh_num_active(&self) -> usize {
        self.head_selection_config.gating.num_active
    }

    pub fn compute_moh_aux_losses(&self, target_avg_components: f32) -> (f32, f32, f32) {
        let lb = self.head_selection_config.compute_load_balance_loss();
        let cx = self.head_selection_config.compute_complexity_loss(target_avg_components);
        let sp = self.head_selection_config.compute_sparsity_loss();
        (lb, cx, sp)
    }

    pub fn compute_moh_aux_weighted_total(&self, target_avg_components: f32) -> f32 {
        let (lb, cx, sp) = self.compute_moh_aux_losses(target_avg_components);
        let g = &self.head_selection_config.gating;
        (lb * g.load_balance_weight) + (cx * g.complexity_loss_weight) + (sp * g.sparsity_weight)
    }

    pub fn peek_tau_metrics(&self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            Some((
                self.head_selection_config.metrics_tau_min,
                self.head_selection_config.metrics_tau_max,
            ))
        } else {
            None
        }
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
            let rms = (self.head_selection_config.metrics_g_sq_sum
                / self.head_selection_config.metrics_g_count as f32)
                .sqrt();
            self.head_selection_config.metrics_g_sq_sum = 0.0;
            self.head_selection_config.metrics_g_count = 0;
            Some(rms)
        } else {
            None
        }
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let num_heads = self.w_g.ncols();
        let mut res = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let tokens = self
                .head_selection_config
                .gating
                .metrics
                .token_count_per_component[h];
            let avg = if tokens > 0 {
                self.head_selection_config.gating.metrics.active_sum_per_component[h] / tokens as f32
            } else {
                0.0
            };
            res.push((avg, tokens));
            self.head_selection_config.gating.metrics.active_sum_per_component[h] = 0.0;
            self.head_selection_config.gating.metrics.token_count_per_component[h] = 0;
        }
        res
    }

    /// Compute gradients for MoH gating parameters given upstream gradients w.r.t. effective weights.
    ///
    /// Returns (grad_input, grad_params) where grad_params matches the ordering:
    /// w_g, alpha_g, beta_g, gate_poly, (optional predictor grads: w1,b1,w2,b2,cond_w,activation)
    pub fn compute_gradients_from_eff(
        &mut self,
        input: &Array2<f32>,
        eff_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let n = input.nrows();
        let embed_dim = self.w_g.nrows();
        let num_heads = self.w_g.ncols();
        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());

        let mut grad_w_g = Array2::<f32>::zeros((embed_dim, num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, num_heads));

        let n_gate_w = self.gate.parameters();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

        // Compute X·W_g once.
        let xw = input.dot(&self.w_g);

        // Recompute raw gate values g_mat (needed for learned-predictor gradients) and
        // compute m_mat consistently with forward.
        let mut g_mat = Array2::<f32>::zeros((n, num_heads));
        for h in 0..num_heads {
            let a_h = self.alpha_g[[0, h]];
            let b_h = self.beta_g[[0, h]];

            // Ensure RichardsGate scaling matches the forward path.
            let mut max_abs_z = 0.0_f64;
            for i in 0..n {
                let z = (a_h * xw[[i, h]] + b_h) as f64;
                max_abs_z = max_abs_z.max(z.abs());
            }
            let _ = self.gate.update_scaling_from_max_abs(max_abs_z);

            for i in 0..n {
                let z = (a_h * xw[[i, h]] + b_h) as f64;
                g_mat[[i, h]] = self.gate.curve.forward_scalar(z) as f32;
            }
        }

        // Mask matrix m_mat for backward.
        let mut m_mat = Array2::<f32>::ones((n, num_heads));

        // For learned predictor: recompute predictor output and apply the same per-row normalization.
        // For SoftTopP: recompute the SoftTopP weights from g_mat (more reliable than relying on cache).
        let mut pred_output: Option<Array2<f32>> = None;
        let mut pred_pre_norm: Option<Array2<f32>> = None;
        if self.head_selection_config.gating.use_learned_predictor {
            if let Some(pred) = &mut self.threshold_predictor {
                let mut p = pred.predict_with_condition(&input.view(), None);
                let mod_f = self.head_selection_config.threshold_modulation;
                p.mapv_inplace(|v| {
                    let v = if v.is_finite() { v } else { 0.0 };
                    v * mod_f
                });

                // Save pre-normalized output for correct normalization backward.
                pred_pre_norm = Some(p.clone());

                // Normalize each row to sum=k.
                let k = self.head_selection_config.gating.num_active.max(1) as f32;
                for i in 0..n {
                    let mut sum = 0.0f32;
                    for h in 0..num_heads {
                        sum += p[[i, h]].max(0.0);
                    }
                    if sum > 0.0 {
                        let s = k / sum;
                        for h in 0..num_heads {
                            p[[i, h]] *= s;
                        }
                    } else {
                        for h in 0..num_heads {
                            p[[i, h]] = 0.0;
                        }
                    }
                }

                pred_output = Some(p.clone());
                m_mat.assign(&p);
            }
        } else if self.head_selection_config.gating.use_soft_top_p {
            let cfg = RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP {
                    top_p: self.head_selection_config.gating.top_p,
                },
                use_learned_predictor: false,
                num_active: self.head_selection_config.gating.num_active.max(1),
                temperature: 1.0,
                soft_top_p_alpha: self.head_selection_config.gating.soft_top_p_alpha,
            };
            let mut weights = apply_selection_algorithm(&g_mat.view(), &cfg);
            let activation_scale = self.head_selection_config.max_heads.max(1) as f32;
            weights.mapv_inplace(|v| (v * activation_scale).clamp(0.0, 1.0));
            let m = self.head_selection_config.threshold_modulation;
            weights.mapv_inplace(|v| (v * m).clamp(0.0, 1.0));
            m_mat.assign(&weights);
        }

        for h in 0..num_heads {
            let w_g_col = self.w_g.slice(s![.., h..h + 1]);
            let a_h = self.alpha_g[[0, h]];
            let b_h = self.beta_g[[0, h]];

            for i in 0..n {
                let xw_ih = xw[[i, h]];
                let z = (a_h * xw_ih + b_h) as f64;
                let m = m_mat[[i, h]];

                let d_eff = eff_grads[[i, h]];
                let d_eff = if d_eff.is_finite() { d_eff } else { 0.0 };
                let d_g = d_eff * m;

                let dphi_dz = self.gate.backward_scalar(z) as f32;
                let grad_z = d_g * dphi_dz;

                // Richards curve parameter grads (uses upstream d_g).
                let gws = self.gate.grad_weights_scalar(z, d_g as f64);
                for (wi, gw) in gws.iter().enumerate() {
                    grad_gate_poly_vec[wi] += *gw;
                }

                // W_g slice grad
                {
                    let mut gw_slice = grad_w_g.slice_mut(s![.., h..h + 1]);
                    for d in 0..embed_dim {
                        gw_slice[[d, 0]] += a_h * input[[i, d]] * grad_z;
                    }
                }
                grad_alpha_g[[0, h]] += grad_z * xw_ih;
                grad_beta_g[[0, h]] += grad_z;

                // Input grad contribution (g-path)
                for d in 0..embed_dim {
                    grad_input[[i, d]] += a_h * w_g_col[[d, 0]] * grad_z;
                }
            }
        }

        // Predictor grads (and predictor->input gradients)
        let mut extra: Vec<Array2<f32>> = Vec::new();
        if self.head_selection_config.gating.use_learned_predictor {
            if let (Some(pred), Some(_)) = (&self.threshold_predictor, pred_output.as_ref()) {
                // dL/dm from eff = g*m
                let mut d_m = Array2::<f32>::zeros((n, num_heads));
                for i in 0..n {
                    for h in 0..num_heads {
                        let d_eff = eff_grads[[i, h]];
                        let d_eff = if d_eff.is_finite() { d_eff } else { 0.0 };
                        let g = g_mat[[i, h]];
                        let g = if g.is_finite() { g } else { 0.0 };
                        d_m[[i, h]] = d_eff * g;
                    }
                }

                // Backprop through row-normalization: m = k * u / sum(u), where u is the pre-normalized predictor output.
                // Use the saved pre-normalized values from this function's predictor forward.
                let u = pred_pre_norm
                    .clone()
                    .unwrap_or_else(|| Array2::<f32>::zeros((n, num_heads)));

                let k = self.head_selection_config.gating.num_active.max(1) as f32;
                let mut d_u = Array2::<f32>::zeros((n, num_heads));
                for i in 0..n {
                    let mut sum_u = 0.0f32;
                    for h in 0..num_heads {
                        sum_u += u[[i, h]];
                    }
                    if sum_u <= 0.0 || !sum_u.is_finite() {
                        continue;
                    }
                    let c = k / sum_u;
                    let mut dot = 0.0f32;
                    for h in 0..num_heads {
                        dot += d_m[[i, h]] * u[[i, h]];
                    }
                    let common = -(k * dot) / (sum_u * sum_u);
                    for h in 0..num_heads {
                        d_u[[i, h]] = c * d_m[[i, h]] + common;
                    }
                }

                // u = modulation * predictor_output (modulation is a scalar).
                // Therefore dL/d(predictor_output) = modulation * dL/du.
                let mod_f = self.head_selection_config.threshold_modulation;
                let mut d_p = d_u;
                d_p.mapv_inplace(|v| v * mod_f);

                // Important: use the predictor instance with cached activations.
                let (dx_pred, gw1, gb1_1d, gw2, gb2_1d, gcond, gact) = {
                    let pred_mut = self
                        .threshold_predictor
                        .as_ref()
                        .expect("predictor must exist");
                    pred_mut.compute_gradients_with_input(&d_p)
                };

                // Predictor->input gradient
                grad_input += &dx_pred;

                let gb1 = gb1_1d.clone().to_shape((gb1_1d.len(), 1)).unwrap().to_owned();
                let gb2 = gb2_1d.clone().to_shape((gb2_1d.len(), 1)).unwrap().to_owned();
                extra.push(gw1);
                extra.push(gb1);
                extra.push(gw2);
                extra.push(gb2);
                if let Some(gcond) = gcond {
                    extra.push(gcond);
                } else {
                    extra.push(Array2::<f32>::zeros((embed_dim, pred.weights1.ncols())));
                }
                // Pack activation params into a 2D array like PolyAttention does.
                let act_arr = Array2::<f32>::from_shape_vec(
                    (gact.len(), 1),
                    gact.iter().map(|&x| x as f32).collect(),
                )
                .unwrap();
                extra.push(act_arr);
            } else if let Some(pred) = &self.threshold_predictor {
                // Keep shape compatibility even if forward cache is missing.
                let hidden_dim = pred.weights1.ncols();
                let act_len = pred.activation.scalar_weights_len();
                extra.push(Array2::<f32>::zeros((embed_dim, hidden_dim))); // w1
                extra.push(Array2::<f32>::zeros((hidden_dim, 1))); // b1
                extra.push(Array2::<f32>::zeros((hidden_dim, num_heads))); // w2
                extra.push(Array2::<f32>::zeros((num_heads, 1))); // b2
                extra.push(Array2::<f32>::zeros((embed_dim, hidden_dim))); // cond_w
                extra.push(Array2::<f32>::zeros((act_len, 1))); // activation
            } else {
                // No predictor available; fall back to minimal shapes.
                extra.push(Array2::<f32>::zeros((embed_dim, 1)));
                extra.push(Array2::<f32>::zeros((1, 1)));
                extra.push(Array2::<f32>::zeros((1, num_heads)));
                extra.push(Array2::<f32>::zeros((num_heads, 1)));
                extra.push(Array2::<f32>::zeros((embed_dim, 1)));
                extra.push(Array2::<f32>::zeros((1, 1)));
            }
        }

        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (grad_gate_poly_vec.len(), 1),
            grad_gate_poly_vec.into_iter().map(|x| x as f32).collect(),
        )
        .unwrap();

        let mut grads = vec![grad_w_g, grad_alpha_g, grad_beta_g, grad_gate_poly];
        grads.extend(extra);

        (grad_input, grads)
    }

    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        // grads ordering described in compute_gradients_from_eff.
        if grads.len() < 4 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!("MoHGating expected at least 4 grad arrays, got {}", grads.len()),
            });
        }
        let mut idx = 0usize;
        self.opt_w_g.step(&mut self.w_g, &grads[idx], lr);
        self.opt_alpha_g.step(&mut self.alpha_g, &grads[idx + 1], lr);
        self.opt_beta_g.step(&mut self.beta_g, &grads[idx + 2], lr);
        idx += 3;
        let grad_gate_poly = &grads[idx];
        let _ = self.gate.apply_gradients(&[grad_gate_poly.clone()], lr);
        idx += 1;

        if self.head_selection_config.gating.use_learned_predictor {
            if let (Some(pred), Some(opt_w1), Some(opt_b1), Some(opt_w2), Some(opt_b2)) = (
                &mut self.threshold_predictor,
                &mut self.opt_w_tau,
                &mut self.opt_b_tau,
                &mut self.opt_w2_tau,
                &mut self.opt_b2_tau,
            ) {
                if grads.len() < idx + 6 {
                    return Err(crate::errors::ModelError::GradientError {
                        message: format!("MoHGating expected predictor grads, got {}", grads.len()),
                    });
                }
                opt_w1.step(&mut pred.weights1, &grads[idx], lr);
                let mut bias1_reshaped = pred.bias1.clone().to_shape((pred.bias1.len(), 1)).unwrap().to_owned();
                opt_b1.step(&mut bias1_reshaped, &grads[idx + 1], lr);
                pred.bias1.assign(&bias1_reshaped.view().to_shape(pred.bias1.len()).unwrap());
                opt_w2.step(&mut pred.weights2, &grads[idx + 2], lr);
                let mut bias2_reshaped = pred.bias2.clone().to_shape((pred.bias2.len(), 1)).unwrap().to_owned();
                opt_b2.step(&mut bias2_reshaped, &grads[idx + 3], lr);
                pred.bias2.assign(&bias2_reshaped.view().to_shape(pred.bias2.len()).unwrap());
                if let Some(opt_cond) = &mut self.opt_cond_w_tau {
                    opt_cond.step(&mut pred.cond_w, &grads[idx + 4], lr);
                }
                let grad_activation_vec: Vec<f64> = grads[idx + 5].iter().map(|&x| x as f64).collect();
                pred.activation.step(&grad_activation_vec, lr as f64);
            }
        }

        Ok(())
    }

    pub fn grad_arrays_len(&self) -> usize {
        let mut n = 4; // w_g, alpha_g, beta_g, gate_poly
        if self.head_selection_config.gating.use_learned_predictor {
            n += 6;
        }
        n
    }
}
