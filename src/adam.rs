use ndarray::Array2;
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
        self.timestep += 1;

        // Apply weight decay (AdamW style: decoupled from gradients)
        let effective_grads = if self.use_decoupled_wd && self.weight_decay > 0.0 {
            // AdamW: Apply weight decay directly to parameters, not gradients
            *params *= 1.0 - self.weight_decay * lr;
            grads.clone()
        } else if self.weight_decay > 0.0 {
            // Traditional L2 regularization: add weight decay to gradients
            grads + &(params.clone() * self.weight_decay)
        } else {
            grads.clone()
        };

        // Update m first
        self.m = &self.m * self.beta1 + &(effective_grads.clone() * (1.0 - self.beta1));

        // Then update v using the same effective_grads
        self.v = &self.v * self.beta2 + &(effective_grads.mapv(|x| x * x) * (1.0 - self.beta2));

        let m_hat = &self.m / (1.0 - self.beta1.powi(self.timestep as i32));
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.timestep as i32));

        // AMSGrad variant: use maximum of past v_hat values for better convergence
        let v_hat_used = if self.use_amsgrad {
            if let Some(ref mut v_hat_max) = self.v_hat_max {
                // Update maximum: v_hat_max = max(v_hat_max, v_hat)
                v_hat_max.zip_mut_with(&v_hat, |max_val, &curr_val| {
                    *max_val = max_val.max(curr_val);
                });
                v_hat_max
            } else {
                // Fallback to regular v_hat if v_hat_max not initialized
                &v_hat
            }
        } else {
            &v_hat
        };

        let update = &m_hat / (v_hat_used.mapv(|x| x.sqrt()) + self.epsilon);
        *params -= &(update * lr);
    }
}
