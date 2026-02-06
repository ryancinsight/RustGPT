//! Adam optimizer with AMSGrad and AdamW variants
//!
//! # Mathematical Formulation
//!
//! ## Adam Algorithm
//!
//! **Theorem 1 (Adam Update)**: The Adam optimizer updates parameters $\theta$ using first and second
//! moment estimates of the gradient $g_t$:
//!
//! $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
//! $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
//!
//! Bias correction:
//! $$\hat{m}_t = m_t / (1 - \beta_1^t)$$
//! $$\hat{v}_t = v_t / (1 - \beta_2^t)$$
//!
//! Parameter update:
//! $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
//!
//! ## AMSGrad Variant
//!
//! **Theorem 2 (AMSGrad Convergence)**: To ensure convergence, AMSGrad maintains the maximum of
//! all past squared gradients (or bias-corrected squared gradients) to ensure the effective learning
//! rate is non-increasing.
//!
//! $$\hat{v}_{max, t} = \max(\hat{v}_{max, t-1}, \hat{v}_t)$$
//! $$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_{max, t}} + \epsilon}$$
//!
//! ## AdamW (Decoupled Weight Decay)
//!
//! **Theorem 3 (Decoupled Weight Decay)**: Standard L2 regularization adds $\lambda \theta$ to the gradient,
//! which interacts with the adaptive learning rate. AdamW decouples this:
//!
//! $$\theta_t = \theta_{t-1}(1 - \lambda \eta) - \eta \dots$$
//!
//! This implementation strictly adheres to these definitions, ensuring numerical stability
//! via explicit finite checks and `epsilon` terms.

use ndarray::{Array2, Zip};
use serde::{Deserialize, Serialize};

/// Configuration constants for Adam step to reduce argument passing
#[derive(Clone, Copy)]
struct AdamStepParams {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    inv_bias1: f32,
    inv_bias2: f32,
    weight_decay: f32,
    use_decoupled_wd: bool,
    use_l2_wd: bool,
}

/// Adam optimizer with optional AMSGrad and AdamW variants
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Adam {
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    timestep: u32,
    m: Array2<f32>,
    v: Array2<f32>,
    /// `AMSGrad` variant: tracks maximum of past squared gradients
    v_hat_max: Option<Array2<f32>>,
    /// Enable `AMSGrad` variant for better convergence guarantees
    use_amsgrad: bool,
    /// Weight decay coefficient (`AdamW`)
    weight_decay: f32,
    /// Use decoupled weight decay (`AdamW` style)
    use_decoupled_wd: bool,
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

    // Accessors
    pub fn m(&self) -> &Array2<f32> {
        &self.m
    }
    pub fn v(&self) -> &Array2<f32> {
        &self.v
    }
    pub fn v_hat_max(&self) -> Option<&Array2<f32>> {
        self.v_hat_max.as_ref()
    }
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
    pub fn is_amsgrad(&self) -> bool {
        self.use_amsgrad
    }
    pub fn is_decoupled_wd(&self) -> bool {
        self.use_decoupled_wd
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
        let inv_bias1 = 1.0 / (1.0 - self.beta1.powf(self.timestep as f32));
        let inv_bias2 = 1.0 / (1.0 - self.beta2.powf(self.timestep as f32));

        let step_params = AdamStepParams {
            lr,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            inv_bias1,
            inv_bias2,
            weight_decay: self.weight_decay,
            use_decoupled_wd: self.use_decoupled_wd && self.weight_decay > 0.0,
            use_l2_wd: !self.use_decoupled_wd && self.weight_decay > 0.0,
        };

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

        if self.use_amsgrad {
            let v_hat_max = self.v_hat_max.as_mut().expect("AMSGrad buffer must exist");

            Zip::from(&mut self.m)
                .and(&mut self.v)
                .and(&mut *v_hat_max)
                .and(params)
                .and(grads)
                .par_for_each(|m, v, v_max, p, &g_in| {
                    Self::adam_kernel(m, v, Some(v_max), p, g_in, step_params);
                });
        } else {
            Zip::from(&mut self.m)
                .and(&mut self.v)
                .and(params)
                .and(grads)
                .par_for_each(|m, v, p, &g_in| {
                    Self::adam_kernel(m, v, None, p, g_in, step_params);
                });
        }
    }

    /// Unified kernel for Adam update on a single element
    #[inline(always)]
    fn adam_kernel(
        m: &mut f32,
        v: &mut f32,
        v_max: Option<&mut f32>,
        p: &mut f32,
        g_in: f32,
        params: AdamStepParams,
    ) {
        let mut g = if g_in.is_finite() { g_in } else { 0.0 };

        // Decoupled Weight Decay (AdamW)
        if params.use_decoupled_wd {
            *p *= 1.0 - params.weight_decay * params.lr;
        } else if params.use_l2_wd {
            let wd_term = *p * params.weight_decay;
            g += if wd_term.is_finite() { wd_term } else { 0.0 };
        }

        // Update moments
        *m = *m * params.beta1 + g * (1.0 - params.beta1);
        *v = *v * params.beta2 + (g * g) * (1.0 - params.beta2);

        // Bias Correction
        let m_hat = *m * params.inv_bias1;
        let v_hat = *v * params.inv_bias2;

        // Denominator selection (AMSGrad vs Standard)
        let denom = if let Some(v_max_val) = v_max {
            // AMSGrad: track max of bias-corrected second moment
            if v_hat.is_finite() {
                *v_max_val = v_max_val.max(v_hat);
            }
            v_max_val.sqrt() + params.epsilon
        } else {
            v_hat.sqrt() + params.epsilon
        };

        // Parameter Update
        if denom.is_finite() && denom > 0.0 && m_hat.is_finite() {
            *p -= params.lr * (m_hat / denom);
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
        let v_max = adam.v_hat_max().unwrap();
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
