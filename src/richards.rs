use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::adam::Adam;
use rayon::prelude::*;

/// Scalar Richards sigmoid: σ(x; ν, k, m) = 1 / (1 + exp(-k(x - m))^(1/ν))
fn richards_sigmoid_scalar(x: f64, nu: f64, k: f64, m: f64) -> f64 {
    if nu <= 0.0 {
        // Fallback: standard logistic when ν→0
        1.0 / (1.0 + (-k * (x - m)).exp())
    } else {
        let exponent = -k * (x - m);
        let u = exponent.exp().powf(1.0 / nu);
        1.0 / (1.0 + u)
    }
}

/// Vectorized Richards sigmoid over Array1.
fn richards_sigmoid(x: &Array1<f64>, nu: f64, k: f64, m: f64) -> Array1<f64> {
    x.mapv(|xi| richards_sigmoid_scalar(xi, nu, k, m))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum Variant {
    Sigmoid,  // Direct σ(x), a=1, b=0 fixed
    Tanh,     // 2σ(2x) - 1, a=1, b=0 fixed
    Gompertz, // ν clamped low (e.g., 0.01), a=1, b=0 fixed
    None,     // No constraints, all parameters learnable including a,b
}

/// Unified Richards curve with variant-based initialization and full parameter learning
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsCurve {
    // Core Richards parameter values (Some for fixed, None for learnable)
    pub nu: Option<f64>,    // Shape (asymmetry)
    pub k: Option<f64>,     // Growth rate
    pub m: Option<f64>,     // Midpoint

    // Gating and affine parameter values (Some for fixed, None for learnable)
    pub beta: Option<f64>,  // Self-gate scaling
    pub a: Option<f64>,     // Affine scale
    pub b: Option<f64>,     // Affine shift

    // Input scaling parameter values (Some for fixed, None for learnable)
    pub scale: Option<f64>, // Input scaling
    pub shift: Option<f64>, // Input shift

    // Learned values for learnable parameters
    pub learned_nu: Option<f64>,
    pub learned_k: Option<f64>,
    pub learned_m: Option<f64>,
    pub learned_beta: Option<f64>,
    pub learned_a: Option<f64>,
    pub learned_b: Option<f64>,
    pub learned_scale: Option<f64>,
    pub learned_shift: Option<f64>,

    // Learnability flags (fixed at initialization)
    pub nu_learnable: bool,
    pub k_learnable: bool,
    pub m_learnable: bool,
    pub beta_learnable: bool,
    pub a_learnable: bool,
    pub b_learnable: bool,
    pub scale_learnable: bool,
    pub shift_learnable: bool,

    // Variant configuration
    pub variant: Variant,   // Sigmoid, Tanh, or Gompertz mode

    // Optimization
    #[serde(skip_serializing, skip_deserializing)]
    optimizer: Option<Adam>,
    pub l2_reg: f64,
    pub adaptive_lr_scale: f64,
    pub grad_norm_history: Vec<f64>,
}

impl RichardsCurve {
    /// Constructor with learnable params based on variant.
    pub fn new_learnable(variant: Variant) -> Self {
        // Set a,b coefficients based on variant (Some for fixed, None for learnable)
        let (a_val, b_val) = match variant {
            Variant::Sigmoid | Variant::Gompertz => (Some(1.0), Some(0.0)), // [0, 1] range, fixed
            Variant::Tanh => (Some(1.0), Some(0.0)), // [-1, 1] via 2σ(2x) - 1 transform, fixed
            Variant::None => (None, None), // Fully learnable including a,b
        };

        // Determine parameter count based on whether a,b are learnable
        let param_count = 6 + if a_val.is_none() { 1 } else { 0 } + if b_val.is_none() { 1 } else { 0 };

        Self {
            // Parameter values (Some for fixed, None for learnable)
            nu: None,
            k: None,
            m: None,
            beta: None,
            a: a_val,
            b: b_val,
            scale: None,
            shift: None,

            // Learned values (None initially)
            learned_nu: None,
            learned_k: None,
            learned_m: None,
            learned_beta: None,
            learned_a: None,
            learned_b: None,
            learned_scale: None,
            learned_shift: None,

            // Learnability flags
            nu_learnable: true,
            k_learnable: true,
            m_learnable: true,
            beta_learnable: true,
            a_learnable: a_val.is_none(),
            b_learnable: b_val.is_none(),
            scale_learnable: true,
            shift_learnable: true,

            variant,
            optimizer: Some(Adam::new_amsgrad((param_count, 1))),
            l2_reg: 1e-4,
            adaptive_lr_scale: 0.01,
            grad_norm_history: Vec::with_capacity(10),
        }
    }

    /// Default Richards parameters approximating logistic: nu=1, k=1, m=0
    pub fn new_default() -> Self {
        Self {
            nu: Some(1.0),
            k: Some(1.0),
            m: Some(0.0),
            beta: Some(1.0),
            a: Some(1.0),
            b: Some(0.0),
            scale: Some(1.0),
            shift: Some(0.0),
            learned_nu: None,
            learned_k: None,
            learned_m: None,
            learned_beta: None,
            learned_a: None,
            learned_b: None,
            learned_scale: None,
            learned_shift: None,
            nu_learnable: false,
            k_learnable: false,
            m_learnable: false,
            beta_learnable: false,
            a_learnable: false,
            b_learnable: false,
            scale_learnable: false,
            shift_learnable: false,
            variant: Variant::Sigmoid,
            optimizer: Some(Adam::new_amsgrad((6, 1))),
            l2_reg: 1e-4,
            adaptive_lr_scale: 0.01,
            grad_norm_history: Vec::with_capacity(10),
        }
    }

    /// Sigmoid builder: fixed params, or learnable.
    pub fn sigmoid(learnable: bool) -> Self {
        if learnable {
            Self::new_learnable(Variant::Sigmoid)
        } else {
            Self {
                nu: Some(1.0),
                k: Some(1.0),
                m: Some(0.0),
                beta: Some(1.0),
                a: Some(1.0),
                b: Some(0.0),
                scale: Some(1.0),
                shift: Some(0.0),
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_a: None,
                learned_b: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                a_learnable: false,
                b_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Sigmoid,
                optimizer: Some(Adam::new_amsgrad((6, 1))),
                l2_reg: 1e-4,
                adaptive_lr_scale: 0.01,
                grad_norm_history: Vec::with_capacity(10),
            }
        }
    }

    /// Tanh builder: fixed (ν=1, k=2, m=0 for exact match), or learnable.
    pub fn tanh(learnable: bool) -> Self {
        if learnable {
            Self::new_learnable(Variant::Tanh)
        } else {
            Self {
                nu: Some(1.0),
                k: Some(1.0),  // Fixed: Changed from 2.0 to 1.0 for accurate tanh approximation
                m: Some(0.0),
                beta: Some(1.0),
                a: Some(1.0),
                b: Some(0.0),
                scale: Some(1.0),  // Fixed for specific variant
                shift: Some(0.0),  // Fixed for specific variant
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_a: None,
                learned_b: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                a_learnable: false,
                b_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Tanh,
                optimizer: Some(Adam::new_amsgrad((6, 1))),
                l2_reg: 1e-4,
                adaptive_lr_scale: 0.01,
                grad_norm_history: Vec::with_capacity(10),
            }
        }
    }

    /// Gompertz builder: low ν fixed (0.01 approx), or learnable.
    pub fn gompertz(learnable: bool) -> Self {
        if learnable {
            Self::new_learnable(Variant::Gompertz)
        } else {
            Self {
                nu: Some(0.01),
                k: Some(1.0),
                m: Some(0.0),
                beta: Some(1.0),
                a: Some(1.0),
                b: Some(0.0),
                scale: Some(1.0),  // Fixed for specific variant
                shift: Some(0.0),  // Fixed for specific variant
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_a: None,
                learned_b: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                a_learnable: false,
                b_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Gompertz,
                optimizer: Some(Adam::new_amsgrad((6, 1))),
                l2_reg: 1e-4,
                adaptive_lr_scale: 0.01,
                grad_norm_history: Vec::with_capacity(10),
            }
        }
    }

    /// Create fully learnable Richards curve without variant constraints
    /// All parameters are learnable and no input/output transformations are applied
    /// This is equivalent to new_learnable(Variant::None)
    pub fn new_fully_learnable() -> Self {
        Self::new_learnable(Variant::None)
    }

    /// Simple scaling based on max absolute value (for numerical stability)
    /// Only updates scale and shift if they are fixed (Some), not learnable (None)
    pub fn update_scaling_from_max_abs(&mut self, max_abs_x: f64) {
        // Only update if scale and shift are fixed (not learnable)
        if self.scale.is_some() && self.shift.is_some() {
            if max_abs_x > 0.0 {
                self.scale = Some((1.0 / max_abs_x).min(0.5));
                self.shift = Some(0.0);
            } else {
                self.scale = Some(1.0);
                self.shift = Some(0.0);
            }
        }
    }

    /// Helper: get parameter value (learnable or fixed).
    fn get_param(&self, param: Option<f64>, learned: Option<f64>, default: f64) -> f64 {
        if param.is_some() {
            param.unwrap()
        } else {
            learned.unwrap_or(default)
        }
    }

    /// Vectorized forward pass: f(x) = a * σ(x) + b (elementwise).
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let a = self.get_param(self.a, self.learned_a, 1.0);
        let b = self.get_param(self.b, self.learned_b, 0.0);
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);

        let (richards_output, _) = self.compute_gate_and_dgate_dx(x, nu, k, m, scale, shift);
        a * &richards_output + b
    }

    /// Forward for a single scalar x (backward compatibility)
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let a = self.get_param(self.a, self.learned_a, 1.0);
        let b = self.get_param(self.b, self.learned_b, 0.0);
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        let input_scale = match self.variant { Variant::Tanh => 2.0, _ => 1.0 };
        let input = input_scale * (scale * x + shift);

        let exponent = -k * (input - m);
        let sigma = if nu <= 0.0 {
            1.0 / (1.0 + exponent.exp())
        } else {
            let u = exponent.exp().powf(1.0 / nu);
            1.0 / (1.0 + u)
        };

        let gate = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };
        a * gate + b
    }

    /// Vectorized backward pass: df/dx at x (analytical elementwise gradient).
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        let nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let a = self.get_param(self.a, self.learned_a, 1.0);

        let (_, dgate_dx) = self.compute_gate_and_dgate_dx(x, nu, k, m, scale, shift);
        a * &dgate_dx
    }

    /// Helper: compute gate vector and its dgate/dx vector (handles variant scaling).
    fn compute_gate_and_dgate_dx(&self, x: &Array1<f64>, nu: f64, k: f64, m: f64, scale: f64, shift: f64) -> (Array1<f64>, Array1<f64>) {
        let input_scale = match self.variant { Variant::Tanh => 2.0, _ => 1.0 };
        let outer_scale = match self.variant { Variant::Tanh => 2.0, _ => 1.0 };
    
        let n = x.len();
        let mut gate = vec![0.0f64; n];
        let mut dgate = vec![0.0f64; n];
    
        if let Some(xs) = x.as_slice() {
            xs.par_iter()
                .zip(gate.par_iter_mut())
                .zip(dgate.par_iter_mut())
                .for_each(|((xi_ref, g_out), d_out)| {
                    let xi = *xi_ref;
                    let input = input_scale * (scale * xi + shift);
                    let exponent = -k * (input - m);
                    let sigma = if nu <= 0.0 {
                        1.0 / (1.0 + exponent.exp())
                    } else {
                        let u = exponent.exp().powf(1.0 / nu);
                        1.0 / (1.0 + u)
                    };
            
                    let g_val = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };
                    *g_out = g_val;
            
                    let dsig_dinput = if nu <= 0.0 { k * sigma * (1.0 - sigma) } else { (k / nu) * sigma * (1.0 - sigma) };
                    *d_out = dsig_dinput * input_scale * outer_scale * scale;
                });
        } else {
            for (i, &xi) in x.iter().enumerate() {
                let input = input_scale * (scale * xi + shift);
                let exponent = -k * (input - m);
                let sigma = if nu <= 0.0 {
                    1.0 / (1.0 + exponent.exp())
                } else {
                    let u = exponent.exp().powf(1.0 / nu);
                    1.0 / (1.0 + u)
                };
            
                gate[i] = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };
                let dsig_dinput = if nu <= 0.0 { k * sigma * (1.0 - sigma) } else { (k / nu) * sigma * (1.0 - sigma) };
                dgate[i] = dsig_dinput * input_scale * outer_scale * scale;
            }
        }
    
        (Array1::from_vec(gate), Array1::from_vec(dgate))
    }

    /// Compute gradients w.r.t. learnable parameters for a single scalar input
    pub fn grad_weights_scalar(&self, x: f64, grad_output: f64) -> Vec<f64> {
        // Forward: f(x) = a * gate(x) + b, where gate(x) is Richards sigmoid
        // Variant-specific scaling:
        // - Tanh: input_scale = 2, outer_scale = 2, gate = 2*sigma - 1
        // - Sigmoid/None/Gompertz: input_scale = 1, outer_scale = 1, gate = sigma
        let nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let a = self.get_param(self.a, self.learned_a, 1.0);
        let b = self.get_param(self.b, self.learned_b, 0.0);
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);

        let input_scale = match self.variant { Variant::Tanh => 2.0, _ => 1.0 };
        let outer_scale = match self.variant { Variant::Tanh => 2.0, _ => 1.0 };

        let input = input_scale * (scale * x + shift);
        let exponent = -k * (input - m);

        // Richards sigmoid: sigma = 1/(1 + exp(exponent)^(1/nu))
        let sigma = if nu <= 0.0 {
            // Fallback to standard logistic with t = exponent
            1.0 / (1.0 + exponent.exp())
        } else {
            let u = exponent.exp().powf(1.0 / nu);
            1.0 / (1.0 + u)
        };

        let gate = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };

        // Common factor for parameter grads that affect sigma directly
        // dσ/dt for σ=1/(1+e^{t}) is -σ(1-σ)
        // t = exponent/ν when ν>0, and t=exponent when ν<=0 (fallback)
        let denom = if nu <= 0.0 { 1.0 } else { nu.max(1e-6) };
        let dsigma_dinput = (k / denom) * sigma * (1.0 - sigma); // sign becomes positive via chain (-σ(1-σ)) * (-k/ν)

        let mut grads = Vec::new();
        let pref = grad_output * a * outer_scale; // since gate = outer_scale * sigma + offset

        // ∂f/∂ν
        if self.nu_learnable {
            if nu <= 0.0 {
                // In fallback logistic, output does not depend on ν
                grads.push(0.0);
            } else {
                // t = exponent/ν, dt/dν = -(exponent)/ν^2
                // dσ/dν = dσ/dt * dt/dν = -σ(1-σ) * (-(exponent)/ν^2) = σ(1-σ) * exponent / ν^2
                let d_sigma_d_nu = sigma * (1.0 - sigma) * (exponent / (denom * denom));
                grads.push(pref * d_sigma_d_nu);
            }
        }

        // ∂f/∂k
        if self.k_learnable {
            // t = exponent/ν, dt/dk = -(input - m)/ν
            // dσ/dk = -σ(1-σ) * (-(input - m)/ν) = σ(1-σ) * (input - m)/ν
            let d_sigma_d_k = sigma * (1.0 - sigma) * ((input - m) / denom);
            grads.push(pref * d_sigma_d_k);
        }

        // ∂f/∂m
        if self.m_learnable {
            // t = exponent/ν, dt/dm = +k/ν
            // dσ/dm = -σ(1-σ) * (k/ν) = -(k/ν) * σ(1-σ)
            let d_sigma_d_m = -(k / denom) * sigma * (1.0 - sigma);
            grads.push(pref * d_sigma_d_m);
        }

        // ∂f/∂beta (not used in this forward; keep zero for layout consistency)
        if self.beta_learnable {
            grads.push(0.0);
        }

        // ∂f/∂a = gate(x)
        if self.a_learnable {
            grads.push(grad_output * gate);
        }

        // ∂f/∂b = 1
        if self.b_learnable {
            grads.push(grad_output);
        }

        // ∂f/∂scale via input: input = input_scale * (scale*x + shift)
        if self.scale_learnable {
            let d_input_d_scale = input_scale * x;
            let d_gate_d_scale = outer_scale * dsigma_dinput * d_input_d_scale;
            grads.push(grad_output * a * d_gate_d_scale);
        }

        // ∂f/∂shift via input
        if self.shift_learnable {
            let d_input_d_shift = input_scale;
            let d_gate_d_shift = outer_scale * dsigma_dinput * d_input_d_shift;
            grads.push(grad_output * a * d_gate_d_shift);
        }

        grads
    }

    /// Derivative for a single scalar x (backward compatibility)
    pub fn backward_scalar(&self, x: f64) -> f64 {
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        let cx = scale * x + shift;
        let nu = self.get_param(self.nu, self.learned_nu, 1.0).max(1e-6);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let a = self.get_param(self.a, self.learned_a, 1.0);

        // Apply variant-specific input scaling
        let input_scale = match self.variant {
            Variant::Tanh => 2.0,
            _ => 1.0,
        };
        let input = input_scale * cx;

        let exponent = -k * (input - m);
        let u = (exponent).exp().powf(1.0 / nu);
        let sigma = 1.0 / (1.0 + u);

        // Derivative of Richards sigmoid w.r.t. input
        let dsig_dinput = if nu <= 0.0 {
            k * sigma * (1.0 - sigma)
        } else {
            (k / nu) * sigma * (1.0 - sigma)
        };

        // Apply variant-specific output scaling
        let outer_scale = match self.variant {
            Variant::Tanh => 2.0,
            _ => 1.0,
        };

        // Chain rule: d/dx [variant_transform(richards(input_scale * (scale*x + shift)))]
        let dgate_dx = dsig_dinput * input_scale * outer_scale;

        // Full derivative: d/dx [a * gate + b] = a * scale * dgate_dx
        a * scale * dgate_dx
    }

    /// Update parameters using Adam optimizer
    pub fn step(&mut self, gradients: &[f64], learning_rate: f64) {
        // Count learnable parameters
        let mut param_count = 0;
        if self.nu_learnable { param_count += 1; }
        if self.k_learnable { param_count += 1; }
        if self.m_learnable { param_count += 1; }
        if self.beta_learnable { param_count += 1; }
        if self.a_learnable { param_count += 1; }
        if self.b_learnable { param_count += 1; }
        if self.scale_learnable { param_count += 1; }
        if self.shift_learnable { param_count += 1; }
        
        // Ensure optimizer is properly initialized for the correct number of parameters
        if self.optimizer.is_none() || 
           (self.optimizer.as_ref().unwrap().m.shape() != &[param_count, 1]) {
            self.optimizer = Some(Adam::new_amsgrad((param_count, 1)));
        }
        
        // Extract current parameter values for learnable parameters
        let mut param_values = Vec::new();
        if self.nu_learnable { param_values.push(self.get_param(self.nu, self.learned_nu, 1.0) as f32); }
        if self.k_learnable { param_values.push(self.get_param(self.k, self.learned_k, 1.0) as f32); }
        if self.m_learnable { param_values.push(self.get_param(self.m, self.learned_m, 0.0) as f32); }
        if self.beta_learnable { param_values.push(self.get_param(self.beta, self.learned_beta, 1.0) as f32); }
        if self.a_learnable { param_values.push(self.get_param(self.a, self.learned_a, 1.0) as f32); }
        if self.b_learnable { param_values.push(self.get_param(self.b, self.learned_b, 0.0) as f32); }
        if self.scale_learnable { param_values.push(self.get_param(self.scale, self.learned_scale, 1.0) as f32); }
        if self.shift_learnable { param_values.push(self.get_param(self.shift, self.learned_shift, 0.0) as f32); }
        
        if let Some(ref mut optimizer) = self.optimizer {
            // Create 2D arrays for Adam optimizer interface
            let mut params = Array2::from_shape_vec((param_count, 1), param_values)
                .expect("Failed to create params array");
            let grads = Array2::from_shape_vec((param_count, 1), gradients.iter().map(|&g| g as f32).collect())
                .expect("Failed to create grads array");
            
            optimizer.step(&mut params, &grads, learning_rate as f32);
            
            // Apply updates back to learned parameters
            let mut idx = 0;
            if self.nu_learnable {
                self.learned_nu = Some((params[[idx, 0]] as f64).max(1e-6)); // Ensure nu > 0
                idx += 1;
            }
            if self.k_learnable {
                self.learned_k = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.m_learnable {
                self.learned_m = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.beta_learnable {
                self.learned_beta = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.a_learnable {
                self.learned_a = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.b_learnable {
                self.learned_b = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.scale_learnable {
                self.learned_scale = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.shift_learnable {
                self.learned_shift = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
        }
    }

    /// Reset the optimizer state
    pub fn reset_optimizer(&mut self) {
        if let Some(ref mut optimizer) = self.optimizer {
            optimizer.reset();
        }
        self.grad_norm_history.clear();
    }

    /// Return current learnable parameter values as a vector (only learnable parameters)
    pub fn weights(&self) -> Vec<f64> {
        let mut weights = Vec::new();

        // Only include learnable parameters in the same order as grad_weights_scalar
        if self.nu_learnable {
            weights.push(self.get_param(self.nu, self.learned_nu, 1.0));
        }
        if self.k_learnable {
            weights.push(self.get_param(self.k, self.learned_k, 1.0));
        }
        if self.m_learnable {
            weights.push(self.get_param(self.m, self.learned_m, 0.0));
        }
        if self.beta_learnable {
            weights.push(self.get_param(self.beta, self.learned_beta, 1.0));
        }
        if self.a_learnable {
            weights.push(self.get_param(self.a, self.learned_a, 1.0));
        }
        if self.b_learnable {
            weights.push(self.get_param(self.b, self.learned_b, 0.0));
        }
        if self.scale_learnable {
            weights.push(self.get_param(self.scale, self.learned_scale, 1.0));
        }
        if self.shift_learnable {
            weights.push(self.get_param(self.shift, self.learned_shift, 0.0));
        }

        weights
    }

    /// Get current scaling parameters
    pub fn get_scaling(&self) -> (f64, f64) {
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        (scale, shift)
    }

    /// Setter for learning updates (e.g., from optimizer).
    pub fn set_param(&mut self, nu: Option<f64>, k: Option<f64>, m: Option<f64>, beta: Option<f64>, a: Option<f64>, b: Option<f64>) {
        if let Some(nu_val) = nu {
            self.nu = Some(nu_val);
        }
        if let Some(k_val) = k {
            self.k = Some(k_val);
        }
        if let Some(m_val) = m {
            self.m = Some(m_val);
        }
        if let Some(beta_val) = beta {
            self.beta = Some(beta_val);
        }
        if let Some(a_val) = a {
            self.a = Some(a_val);
        }
        if let Some(b_val) = b {
            self.b = Some(b_val);
        }
    }
}

/// RichardsActivation: Multiplies input by Richards curve output (x * Richards(x))
/// This creates swish-like activations and other gated activations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsActivation {
    pub richards_curve: RichardsCurve,
}

/// Backward compatibility alias: RichardsAttention is the same as RichardsActivation
pub type RichardsAttention = RichardsActivation;

impl RichardsActivation {
    /// Create learnable Richards activation with specified variant
    pub fn new_learnable(variant: Variant) -> Self {
        Self {
            richards_curve: RichardsCurve::new_learnable(variant),
        }
    }

    /// Create fully learnable Richards activation without variant constraints
    pub fn new_fully_learnable() -> Self {
        Self {
            richards_curve: RichardsCurve::new_learnable(Variant::None),
        }
    }

    /// Create a new RichardsActivation with default Richards curve (sigmoid-like)
    pub fn new_default() -> Self {
        Self {
            richards_curve: RichardsCurve::new_default(),
        }
    }

    /// Create a sigmoid-based activation (similar to swish activation)
    pub fn sigmoid(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::sigmoid(learnable),
        }
    }

    /// Create a tanh-based activation
    pub fn tanh(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::tanh(learnable),
        }
    }

    /// Create a Gompertz-based activation
    pub fn gompertz(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::gompertz(learnable),
        }
    }

    /// Forward pass: x * Richards(x) (elementwise multiplication)
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let richards_output = self.richards_curve.forward(x);
        x * &richards_output
    }

    /// Forward pass for a single scalar
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let richards_output = self.richards_curve.forward_scalar(x);
        x * richards_output
    }

    /// Backward pass: derivative of x * Richards(x)
    /// d/dx[x * Richards(x)] = Richards(x) + x * Richards'(x)
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let richards_output = self.richards_curve.forward(x);
        let richards_derivative = self.richards_curve.derivative(x);
        &richards_output + x * &richards_derivative
    }

    /// Backward pass for a single scalar
    pub fn backward_scalar(&self, x: f64) -> f64 {
        let richards_output = self.richards_curve.forward_scalar(x);
        let richards_derivative = self.richards_curve.backward_scalar(x);
        richards_output + x * richards_derivative
    }

    /// Get the weights from the underlying Richards curve
    pub fn weights(&self) -> Vec<f64> {
        self.richards_curve.weights()
    }

    /// Compute gradients with respect to the Richards curve parameters
    pub fn grad_weights_scalar(&self, x: f64, grad_output: f64) -> Vec<f64> {
        // For f(x) = x * Richards(x), we need:
        // df/dθ = x * dRichards/dθ where θ are the Richards parameters
        let richards_grads = self.richards_curve.grad_weights_scalar(x, x * grad_output);
        richards_grads
    }

    /// Update parameters using gradients
    pub fn step(&mut self, gradients: &[f64], learning_rate: f64) {
        self.richards_curve.step(gradients, learning_rate);
    }

    /// Reset the optimizer state
    pub fn reset_optimizer(&mut self) {
        self.richards_curve.reset_optimizer();
    }

    /// Update scaling based on input statistics
    pub fn update_scaling_from_max_abs(&mut self, max_abs_x: f64) {
        self.richards_curve.update_scaling_from_max_abs(max_abs_x);
    }

    /// Get scaling parameters
    pub fn get_scaling(&self) -> (f64, f64) {
        self.richards_curve.get_scaling()
    }

    /// Set parameters directly
    pub fn set_param(&mut self, nu: Option<f64>, k: Option<f64>, m: Option<f64>, beta: Option<f64>, a: Option<f64>, b: Option<f64>) {
        self.richards_curve.set_param(nu, k, m, beta, a, b);
    }
}