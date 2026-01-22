//! Adam optimizer with AMSGrad and AdamW variants
//!
//! Provides efficient, numerically stable implementations of:
//! - Standard Adam optimizer
//! - AMSGrad variant with maximum tracking
//! - AdamW with decoupled weight decay

use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

/// Adam optimizer with optional AMSGrad and AdamW variants
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u32, // Changed from usize to avoid casting issues
    pub m: Array2<f32>,
    pub v: Array2<f32>,
    /// `AMSGrad` variant: tracks maximum of past squared gradients
    pub v_hat_max: Option<Array2<f32>>,
    /// Enable `AMSGrad` variant for better convergence guarantees
    pub use_amsgrad: bool,
    /// Weight decay coefficient (`AdamW`)
    pub weight_decay: f32,
    /// Use decoupled weight decay (`AdamW` style)
    pub use_decoupled_wd: bool,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters
    #[must_use]
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            v_hat_max: None,
            use_amsgrad: false,
            weight_decay: 0.0,
            use_decoupled_wd: false,
        }
    }

    /// Enable or disable `AMSGrad` variant
    pub fn set_amsgrad(&mut self, enable: bool) {
        self.use_amsgrad = enable;
        if enable && self.v_hat_max.is_none() {
            self.v_hat_max = Some(Array2::zeros(self.m.dim()));
        } else if !enable {
            self.v_hat_max = None;
        }
    }

    /// Create Adam optimizer with `AMSGrad` variant enabled
    #[must_use]
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

    /// Create `AdamW` optimizer (Adam with decoupled weight decay)
    #[must_use]
    pub fn new_adamw(shape: (usize, usize), weight_decay: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            m: Array2::zeros(shape),
            v: Array2::zeros(shape),
            v_hat_max: Some(Array2::zeros(shape)),
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

    /// Perform optimization step
    ///
    /// # Panics
    /// This method validates shapes and will resize buffers if needed, so it won't panic
    #[inline]
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        // Early exit for zero learning rate
        if lr == 0.0 {
            return;
        }

        // Validate and resize buffers if needed
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

        // Bias-correction factors (using u32 to avoid casting issues)
        let inv_bias1 = 1.0 / (1.0 - self.beta1.powi(self.timestep as i32));
        let inv_bias2 = 1.0 / (1.0 - self.beta2.powi(self.timestep as i32));

        // Apply decoupled weight decay (AdamW style)
        if self.use_decoupled_wd && self.weight_decay > 0.0 {
            params.mapv_inplace(|p| p * (1.0 - self.weight_decay * lr));
        }

        let use_l2_wd = !self.use_decoupled_wd && self.weight_decay > 0.0;

        // Ensure AMSGrad buffer exists with correct shape
        if self.use_amsgrad {
            let need_init = self
                .v_hat_max
                .as_ref()
                .is_none_or(|a| a.dim() != grads.dim());
            if need_init {
                self.v_hat_max = Some(Array2::zeros(grads.dim()));
            }
        }

        // Update moments and parameters in-place
        if self.use_amsgrad {
            let v_hat_max = self.v_hat_max.as_mut().expect("AMSGrad buffer must exist");

            Zip::from(&mut self.m)
                .and(&mut self.v)
                .and(&mut *v_hat_max)
                .and(params.view())
                .and(grads)
                .for_each(|m, v, v_max, &p, &g_in| {
                    // Sanitize gradient
                    let mut g = if g_in.is_finite() { g_in } else { 0.0 };

                    // Add L2 weight decay to gradient if enabled
                    if use_l2_wd {
                        let wd_term = p * self.weight_decay;
                        g += if wd_term.is_finite() { wd_term } else { 0.0 };
                    }

                    // Update first moment (momentum)
                    *m = *m * self.beta1 + g * (1.0 - self.beta1);

                    // Update second moment (variance)
                    *v = *v * self.beta2 + (g * g) * (1.0 - self.beta2);

                    // Track maximum of bias-corrected second moment
                    let v_hat = *v * inv_bias2;
                    if v_hat.is_finite() {
                        *v_max = v_max.max(v_hat);
                    }
                });

            // Apply parameter update
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
            // Standard Adam
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

impl Default for Adam {
    fn default() -> Self {
        Self::new((1, 1))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_adam_basic_update() {
        let mut adam = Adam::new((2, 2));
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let initial = params.clone();
        adam.step(&mut params, &grads, 0.01);

        // Parameters should have changed
        assert!((params[[0, 0]] - initial[[0, 0]]).abs() > 1e-6);
    }

    #[test]
    fn test_adam_zero_lr() {
        let mut adam = Adam::new((2, 2));
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        let initial = params.clone();
        adam.step(&mut params, &grads, 0.0);

        // Parameters should not change with zero learning rate
        assert_abs_diff_eq!(params, initial, epsilon = 1e-9);
    }

    #[test]
    fn test_amsgrad_tracking() {
        let mut adam = Adam::new_amsgrad((2, 2));
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        adam.step(&mut params, &grads, 0.01);

        // v_hat_max should be populated
        assert!(adam.v_hat_max.is_some());
        let v_max = adam.v_hat_max.as_ref().unwrap();
        assert!(v_max.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut adam = Adam::new_adamw((2, 2), 0.01);
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grads = Array2::zeros((2, 2));

        let initial = params.clone();
        adam.step(&mut params, &grads, 0.1);

        // With zero gradients, AdamW should still decay weights
        assert!(params.iter().zip(initial.iter()).all(|(p, i)| p < i));
    }
}
