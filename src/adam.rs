use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: usize,
    pub m: Array2<f32>,
    pub v: Array2<f32>,
    /// AMSGrad variant: tracks maximum of past squared gradients
    pub v_hat_max: Option<Array2<f32>>,
    /// Enable AMSGrad variant for better convergence guarantees
    pub use_amsgrad: bool,
    /// Weight decay coefficient (AdamW)
    pub weight_decay: f32,
    /// Use decoupled weight decay (AdamW style)
    pub use_decoupled_wd: bool,
}

impl Adam {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            v_hat_max: None,
            use_amsgrad: false, // Default to standard Adam for backward compatibility
            weight_decay: 0.0,  // No weight decay by default
            use_decoupled_wd: false, // Use L2 regularization style by default
        }
    }

    /// Enable or disable AMSGrad variant
    pub fn set_amsgrad(&mut self, enable: bool) {
        self.use_amsgrad = enable;
        if enable && self.v_hat_max.is_none() {
            self.v_hat_max = Some(Array2::zeros(self.m.dim()));
        } else if !enable {
            self.v_hat_max = None;
        }
    }

    /// Create Adam optimizer with AMSGrad variant enabled
    pub fn new_amsgrad(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            v_hat_max: Some(Array2::zeros(shape)),
            use_amsgrad: true,
            weight_decay: 0.0,
            use_decoupled_wd: false,
        }
    }

    /// Create AdamW optimizer (Adam with decoupled weight decay)
    pub fn new_adamw(shape: (usize, usize), weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            v_hat_max: Some(Array2::zeros(shape)), // AdamW typically uses AMSGrad
            use_amsgrad: true,
            weight_decay,
            use_decoupled_wd: true,
        }
    }

    /// Set weight decay parameters
    pub fn set_weight_decay(&mut self, weight_decay: f32, decoupled: bool) {
        self.weight_decay = weight_decay;
        self.use_decoupled_wd = decoupled;
    }

    /// Reset optimizer state (useful for restarting training)
    pub fn reset(&mut self) {
        self.timestep = 0;
        self.m.fill(0.0);
        self.v.fill(0.0);
        if let Some(ref mut v_hat_max) = self.v_hat_max {
            v_hat_max.fill(0.0);
        }
    }

    #[inline]
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        // Validate shapes to avoid runtime panics
        if params.dim() != grads.dim() {
            tracing::warn!(
                "Adam::step shape mismatch: params={:?}, grads={:?} — skipping update",
                params.dim(),
                grads.dim()
            );
            return;
        }
        if self.m.dim() != grads.dim() || self.v.dim() != grads.dim() {
            self.m = Array2::zeros(grads.dim());
            self.v = Array2::zeros(grads.dim());
            if self.use_amsgrad {
                self.v_hat_max = Some(Array2::zeros(grads.dim()));
            }
        }
        self.timestep += 1;

        if lr == 0.0 {
            return;
        }

        // Bias-correction scalars.
        let inv_bias1 = 1.0 / (1.0 - self.beta1.powi(self.timestep as i32));
        let inv_bias2 = 1.0 / (1.0 - self.beta2.powi(self.timestep as i32));

        // Apply weight decay (AdamW style: decoupled from gradients).
        if self.use_decoupled_wd && self.weight_decay > 0.0 {
            // AdamW: Apply weight decay directly to parameters, not gradients
            *params *= 1.0 - self.weight_decay * lr;
        }

        let use_l2_wd = (!self.use_decoupled_wd) && self.weight_decay > 0.0;

        // Ensure AMSGrad buffer exists and has correct shape.
        if self.use_amsgrad {
            let need_init = self
                .v_hat_max
                .as_ref()
                .is_none_or(|a| a.dim() != grads.dim());
            if need_init {
                self.v_hat_max = Some(Array2::zeros(grads.dim()));
            }
        }

        // Update moments in-place (no intermediate allocations).
        if self.use_amsgrad {
            let v_hat_max = self
                .v_hat_max
                .as_mut()
                .expect("AMSGrad buffer must exist");
            Zip::from(&mut self.m)
                .and(&mut self.v)
                .and(&mut *v_hat_max)
                .and(params.view())
                .and(grads)
                .for_each(|m, v, v_max, &p, &g_in| {
                    let mut g = if g_in.is_finite() { g_in } else { 0.0 };
                    if use_l2_wd {
                        let wd_term = p * self.weight_decay;
                        g += if wd_term.is_finite() { wd_term } else { 0.0 };
                    }
                    *m = *m * self.beta1 + g * (1.0 - self.beta1);
                    *v = *v * self.beta2 + (g * g) * (1.0 - self.beta2);

                    // Track max of bias-corrected v_hat (AMSGrad).
                    let v_hat = (*v) * inv_bias2;
                    if v_hat.is_finite() {
                        *v_max = v_max.max(v_hat);
                    }
                });

            // Apply update.
            Zip::from(params)
                .and(self.m.view())
                .and(v_hat_max.view())
                .for_each(|p, &m, &v_hat_max| {
                    let m_hat = m * inv_bias1;
                    let denom = v_hat_max.sqrt() + self.epsilon;
                    if denom.is_finite() && denom > 0.0 && m_hat.is_finite() {
                        *p -= lr * (m_hat / denom);
                    }
                });
        } else {
            Zip::from(&mut self.m)
                .and(&mut self.v)
                .and(params.view())
                .and(grads)
                .for_each(|m, v, &p, &g_in| {
                    let mut g = if g_in.is_finite() { g_in } else { 0.0 };
                    if use_l2_wd {
                        let wd_term = p * self.weight_decay;
                        g += if wd_term.is_finite() { wd_term } else { 0.0 };
                    }
                    *m = *m * self.beta1 + g * (1.0 - self.beta1);
                    *v = *v * self.beta2 + (g * g) * (1.0 - self.beta2);
                });

            Zip::from(params)
                .and(self.m.view())
                .and(self.v.view())
                .for_each(|p, &m, &v| {
                    let m_hat = m * inv_bias1;
                    let v_hat = v * inv_bias2;
                    let denom = v_hat.sqrt() + self.epsilon;
                    if denom.is_finite() && denom > 0.0 && m_hat.is_finite() {
                        *p -= lr * (m_hat / denom);
                    }
                });
        }
    }
}
