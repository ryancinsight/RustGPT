pub mod richards_act;
pub mod richards_norm;

pub use self::richards_act::*;
pub use self::richards_norm::*;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use crate::adam::Adam;
use rayon::prelude::*;



#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum Variant {
    Sigmoid,     // Direct σ(x), output_gain=1, output_bias=0 fixed
    Tanh,        // 2σ(2x) - 1, output_gain=1, output_bias=0 fixed
    Gompertz,    // ν clamped low (e.g., 0.01), output_gain=1, output_bias=0 fixed
    Adaptive,    // Adaptive normalization with running statistics tracking
    Polynomial,  // Polynomial input transformation before Richards activation
    None,        // No constraints, all parameters learnable including output_gain,output_bias
}

/// Unified Richards curve with variant-based initialization and full parameter learning
/// Extended with beta parameter for asymmetric control and temperature for sharpness
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsCurve {
    // Core Richards parameter values (Some for fixed, None for learnable)
    pub nu: Option<f64>,    // Shape (asymmetry)
    pub k: Option<f64>,     // Growth rate
    pub m: Option<f64>,     // Midpoint
    pub beta: Option<f64>,  // Asymmetry factor for extended Richards

    // Temperature parameter (controls curve sharpness/softness)
    pub temperature: Option<f64>,  // Temperature scaling factor

    // Affine parameter values (Some for fixed, None for learnable)
    #[serde(rename = "a")]
    pub output_gain: Option<f64>,     // Affine output gain (scale)
    #[serde(rename = "b")]
    pub output_bias: Option<f64>,     // Affine output bias (shift)

    // Input scaling parameter values (Some for fixed, None for learnable)
    pub scale: Option<f64>, // Input scaling
    pub shift: Option<f64>, // Input shift

    // Polynomial input transformation (used by Polynomial variant)
    #[serde(skip_serializing, skip_deserializing)]
    pub poly_power: Option<usize>,         // Polynomial degree (1-5, 1=identity)
    #[serde(skip_serializing, skip_deserializing)]
    pub poly_coeffs: Option<Vec<f64>>,     // Polynomial coefficients [ coeff_0, coeff_1, ..., coeff_power]

    // Learned values for learnable parameters
    pub learned_nu: Option<f64>,
    pub learned_k: Option<f64>,
    pub learned_m: Option<f64>,
    pub learned_beta: Option<f64>,
    pub learned_temperature: Option<f64>,
    #[serde(rename = "learned_a")]
    pub learned_output_gain: Option<f64>,
    #[serde(rename = "learned_b")]
    pub learned_output_bias: Option<f64>,
    pub learned_scale: Option<f64>,
    pub learned_shift: Option<f64>,

    // Learnability flags (fixed at initialization)
    pub nu_learnable: bool,
    pub k_learnable: bool,
    pub m_learnable: bool,
    pub beta_learnable: bool,
    pub temperature_learnable: bool,
    #[serde(rename = "a_learnable")]
    pub output_gain_learnable: bool,
    #[serde(rename = "b_learnable")]
    pub output_bias_learnable: bool,
    pub scale_learnable: bool,
    pub shift_learnable: bool,

    // Variant configuration
    pub variant: Variant,   // Sigmoid, Tanh, or Gompertz mode

    // Adaptive normalization (used by Adaptive variant)
    #[serde(skip_serializing, skip_deserializing)]
    running_sum: Option<f64>,       // Running sum for mean estimation
    #[serde(skip_serializing, skip_deserializing)]
    running_sq_sum: Option<f64>,    // Running sum of squares for variance estimation
    #[serde(skip_serializing, skip_deserializing)]
    count: Option<u64>,             // Number of samples seen
    pub momentum: f64,              // Momentum for running statistics (0.01 typical)
    #[serde(skip_serializing, skip_deserializing)]
    adaptive_scale: Option<f64>,    // Automatically computed scale factor
    #[serde(skip_serializing, skip_deserializing)]
    adaptive_shift: Option<f64>,    // Automatically computed shift factor

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
        // Set output_gain/output_bias coefficients based on variant (Some for fixed, None for learnable)
        let (output_gain_val, output_bias_val) = match variant {
            Variant::Sigmoid | Variant::Gompertz => (Some(1.0), Some(0.0)), // [0, 1] range, fixed
            Variant::Tanh => (Some(1.0), Some(0.0)), // [-1, 1] via 2σ(2x) - 1 transform, fixed
            Variant::Adaptive | Variant::None | Variant::Polynomial => (None, None), // Fully learnable including output_gain/output_bias
        };

        // Determine parameter count based on whether output_gain/output_bias are learnable
        // nu, k, m, beta, temp, scale, shift + optionally output_gain, output_bias
        let param_count = 7 + if output_gain_val.is_none() { 1 } else { 0 } + if output_bias_val.is_none() { 1 } else { 0 };

        let (adaptive_initialized, momentum) = match variant {
            Variant::Adaptive => (true, 0.01), // Enable adaptive normalization with default momentum
            _ => (false, 0.0), // Disable adaptive for other variants
        };

        let polynomial_initialized = match variant {
            Variant::Polynomial => true, // Enable polynomial transformation
            _ => false, // Disable polynomial for other variants
        };

        Self {
            // Parameter values (Some for fixed, None for learnable)
            nu: None,
            k: None,
            m: None,
            beta: None,
            temperature: None,
            output_gain: output_gain_val,
            output_bias: output_bias_val,
            scale: None,
            shift: None,

            // Polynomial transformation
            poly_power: if polynomial_initialized { Some(1) } else { None },  // Default to degree 1 (identity)
            poly_coeffs: if polynomial_initialized { Some(vec![0.0, 1.0]) } else { None }, // [0, 1] = identity

            // Learned values (None initially)
            learned_nu: None,
            learned_k: None,
            learned_m: None,
            learned_beta: None,
            learned_temperature: None,
            learned_output_gain: None,
            learned_output_bias: None,
            learned_scale: None,
            learned_shift: None,

            // Learnability flags
            nu_learnable: true,
            k_learnable: true,
            m_learnable: true,
            beta_learnable: true,
            temperature_learnable: true,
            output_gain_learnable: output_gain_val.is_none(),
            output_bias_learnable: output_bias_val.is_none(),
            scale_learnable: true,
            shift_learnable: true,

            // Adaptive normalization
            running_sum: if adaptive_initialized { Some(0.0) } else { None },
            running_sq_sum: if adaptive_initialized { Some(0.0) } else { None },
            count: if adaptive_initialized { Some(0) } else { None },
            momentum,
            adaptive_scale: if adaptive_initialized { Some(1.0) } else { None },
            adaptive_shift: if adaptive_initialized { Some(0.0) } else { None },

            variant,
            optimizer: Some(Adam::new((param_count, 1))),
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
            temperature: Some(1.0),
            output_gain: Some(1.0),
            output_bias: Some(0.0),
            scale: Some(1.0),
            shift: Some(0.0),
            learned_nu: None,
            learned_k: None,
            learned_m: None,
            learned_beta: None,
            learned_temperature: None,
            learned_output_gain: None,
            learned_output_bias: None,
            learned_scale: None,
            learned_shift: None,
            nu_learnable: false,
            k_learnable: false,
            m_learnable: false,
            beta_learnable: false,
            temperature_learnable: false,
            output_gain_learnable: false,
            output_bias_learnable: false,
            scale_learnable: false,
            shift_learnable: false,
            variant: Variant::Sigmoid,
            poly_power: None,         // Not polynomial variant
            poly_coeffs: None,
            running_sum: None,       // Not adaptive variant
            running_sq_sum: None,
            count: None,
            momentum: 0.0,
            adaptive_scale: None,
            adaptive_shift: None,
            optimizer: Some(Adam::new((6, 1))),
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
                temperature: Some(1.0),
                output_gain: Some(1.0),
                output_bias: Some(0.0),
                scale: Some(1.0),
                shift: Some(0.0),
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_temperature: None,
                learned_output_gain: None,
                learned_output_bias: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                temperature_learnable: false,
                output_gain_learnable: false,
                output_bias_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Sigmoid,
                poly_power: None,         // Not polynomial variant
                poly_coeffs: None,
                running_sum: None,
                running_sq_sum: None,
                count: None,
                momentum: 0.0,
                adaptive_scale: None,
                adaptive_shift: None,
                optimizer: Some(Adam::new((6, 1))),
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
                temperature: Some(1.0),
                 output_gain: Some(1.0),
                 output_bias: Some(0.0),
                 scale: Some(1.0),  // Fixed for specific variant
                 shift: Some(0.0),  // Fixed for specific variant
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_temperature: None,
                learned_output_gain: None,
                learned_output_bias: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                temperature_learnable: false,
                output_gain_learnable: false,
                output_bias_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Tanh,
                poly_power: None,         // Not polynomial variant
                poly_coeffs: None,
                running_sum: None,
                running_sq_sum: None,
                count: None,
                momentum: 0.0,
                adaptive_scale: None,
                adaptive_shift: None,
                optimizer: Some(Adam::new((6, 1))),
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
                temperature: Some(1.0),
                 output_gain: Some(1.0),
                 output_bias: Some(0.0),
                 scale: Some(1.0),  // Fixed for specific variant
                 shift: Some(0.0),  // Fixed for specific variant
                learned_nu: None,
                learned_k: None,
                learned_m: None,
                learned_beta: None,
                learned_temperature: None,
                learned_output_gain: None,
                learned_output_bias: None,
                learned_scale: None,
                learned_shift: None,
                nu_learnable: false,
                k_learnable: false,
                m_learnable: false,
                beta_learnable: false,
                temperature_learnable: false,
                output_gain_learnable: false,
                output_bias_learnable: false,
                scale_learnable: false,
                shift_learnable: false,
                variant: Variant::Gompertz,
                poly_power: None,         // Not polynomial variant
                poly_coeffs: None,
                running_sum: None,
                running_sq_sum: None,
                count: None,
                momentum: 0.0,
                adaptive_scale: None,
                adaptive_shift: None,
                optimizer: Some(Adam::new((6, 1))),
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

    /// Helper: get all parameters at once to reduce redundancy.
    fn get_all_params(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let beta = self.get_param(self.beta, self.learned_beta, 1.0);
        let temp = self.get_param(self.temperature, self.learned_temperature, 1.0);
        let output_gain = self.get_param(self.output_gain, self.learned_output_gain, 1.0);
        let output_bias = self.get_param(self.output_bias, self.learned_output_bias, 0.0);
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        (nu, k, m, beta, temp, output_gain, output_bias, scale, shift)
    }

    /// Helper: get variant-specific input and output scales.
    fn get_variant_scales(&self) -> (f64, f64) {
        match self.variant {
            Variant::Tanh => (2.0, 2.0),
            _ => (1.0, 1.0),
        }
    }

    /// Vectorized forward pass: f(x) = output_gain * gate(x) + output_bias (elementwise), single-pass.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let (nu, k, m, beta, temp, output_gain, output_bias, scale, shift) = self.get_all_params();
        let (input_scale, _) = self.get_variant_scales();
        let (adaptive_scale, adaptive_shift) = self.get_adaptive_scaling();

        let mut out = Array1::zeros(x.len());
        let xs = x.as_slice().unwrap();
        let os = out.as_slice_mut().unwrap();

        xs.par_iter()
            .zip(os.par_iter_mut())
            .for_each(|(&xi, o)| {
                // Apply adaptive normalization first (x - mean) / std for Adaptive variant
                let adaptive_normalized = adaptive_scale * xi + adaptive_shift;
                // Apply temperature scaling: sharper when temp < 1, softer when temp > 1
                let temp_scaled = adaptive_normalized / temp;
                let input = input_scale * (scale * temp_scaled + shift);
        let exponent: f64 = -k * (input - m);
        // Clamp exponent to prevent overflow: exp(exponent) should not exceed 1e10
        let clamped_exponent = exponent.clamp(-23.0, 23.0); // exp(±23) ≈ 1e±10
        // Extended Richards with beta asymmetry factor
        // y = (beta + (1-beta) * exp(-k*(x-m))) ^ (-1/ν)
        let beta_term = beta + (1.0 - beta) * clamped_exponent.exp();
                let extended_richards = if nu <= 0.0 {
                    1.0 / beta_term
                } else {
                    beta_term.powf(-1.0 / nu)
                };
                let gate = match self.variant { Variant::Tanh => 2.0 * extended_richards - 1.0, _ => extended_richards };
                let output = output_gain * gate + output_bias;
                // Numerical stability: clamp extreme values to prevent NaN/inf propagation
                *o = output.clamp(-1e6, 1e6);
            });

        out
    }

    /// Forward for a single scalar x (backward compatibility)
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let (nu, k, m, _, _, output_gain, output_bias, scale, shift) = self.get_all_params();
        let (input_scale, _) = self.get_variant_scales();
        let temp_scaled = x;  // Backward compatibility: no temperature scaling
        let input = input_scale * (scale * temp_scaled + shift);

        let exponent = -k * (input - m);
        // Clamp exponent to prevent overflow
        let clamped_exponent = exponent.clamp(-23.0, 23.0);
        let sigma = if nu <= 0.0 {
            1.0 / (1.0 + clamped_exponent.exp())
        } else {
            let u = clamped_exponent.exp().powf(1.0 / nu);
            1.0 / (1.0 + u)
        };

        let gate = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };
        let output = output_gain * gate + output_bias;
        // Numerical stability: clamp extreme values to prevent NaN/inf propagation
        output.clamp(-1e6, 1e6)
    }

    /// Vectorized backward pass: df/dx at x (analytical gradient), single-pass.
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let (nu, k, m, _, _, output_gain, _, scale, shift) = self.get_all_params();
        let (input_scale, outer_scale) = self.get_variant_scales();

        let mut out = Array1::zeros(x.len());
        let xs = x.as_slice().unwrap();
        let os = out.as_slice_mut().unwrap();

        xs.par_iter()
            .zip(os.par_iter_mut())
            .for_each(|(&xi, o)| {
                let input = input_scale * (scale * xi + shift);
                let exponent: f64 = -k * (input - m);
                let sigma = if nu <= 0.0 {
                    1.0 / (1.0 + exponent.exp())
                } else {
                    let u = exponent.exp().powf(1.0 / nu);
                    1.0 / (1.0 + u)
                };

                let dsig_dinput = if nu <= 0.0 { k * sigma * (1.0 - sigma) } else { (k / nu) * sigma * (1.0 - sigma) };
                *o = output_gain * dsig_dinput * input_scale * outer_scale * scale;
            });

        out
    }



    /// Compute gradients w.r.t. learnable parameters for a single scalar input into a preallocated slice
    pub fn grad_weights_scalar_into(&self, x: f64, grad_output: f64, out: &mut [f64]) {
        // Forward: f(x) = output_gain * gate(x) + output_bias, where gate(x) is Richards sigmoid
        // Variant-specific scaling:
        // - Tanh: input_scale = 2, outer_scale = 2, gate = 2*sigma - 1
        // - Sigmoid/None/Gompertz: input_scale = 1, outer_scale = 1, gate = sigma
        let (nu, k, m, beta, temp, output_gain, _, scale, shift) = self.get_all_params();
        let (input_scale, outer_scale) = self.get_variant_scales();
        let (adaptive_scale, adaptive_shift) = self.get_adaptive_scaling();

        let adaptive_normalized = adaptive_scale * x + adaptive_shift;
        let temp_scaled = adaptive_normalized / temp;
        let input = input_scale * (scale * temp_scaled + shift);

        let exponent = -k * (input - m);
        let sigma = if nu <= 0.0 {
            1.0 / (1.0 + exponent.exp())
        } else {
            let u = exponent.exp().powf(1.0 / nu);
            1.0 / (1.0 + u)
        };
        let gate = match self.variant { Variant::Tanh => 2.0 * sigma - 1.0, _ => sigma };

        let denom = if nu <= 0.0 { 1.0 } else { nu.max(1e-6) };
        let dsigma_dinput = (k / denom) * sigma * (1.0 - sigma);
        let pref = grad_output * output_gain * outer_scale;

        let mut pos = 0usize;
        if self.nu_learnable {
            if nu <= 0.0 {
                out[pos] = 0.0;
            } else {
                let d_sigma_d_nu = sigma * (1.0 - sigma) * (exponent / (denom * denom));
                out[pos] = pref * d_sigma_d_nu;
            }
            pos += 1;
        }
        if self.k_learnable {
            let d_sigma_d_k = sigma * (1.0 - sigma) * ((input - m) / denom);
            out[pos] = pref * d_sigma_d_k;
            pos += 1;
        }
        if self.m_learnable {
            let d_sigma_d_m = -(k / denom) * sigma * (1.0 - sigma);
            out[pos] = pref * d_sigma_d_m;
            pos += 1;
        }
        if self.beta_learnable {
            // Richards(y) = [β + (1-β) * exp(-k*(y-m))] ^ (-1/ν)
            // Let D = β + (1-β) * exp(-k*(y-m))
            // Let Richards(y) = D^(-1/ν)
            // dRichards/dβ = dRichards/dD * dD/dβ
            // dD/dβ = 1 - exp(-k*(y-m))
            let exp_term = exponent.exp();
            let d = beta + (1.0 - beta) * exp_term;
            let d_d_beta = 1.0 - exp_term;

            let d_richards_d_d = if nu <= 0.0 {
                // Richards = 1/D, so dRichards/dD = -1/D² = -Richards²
                -gate * gate
            } else {
                // Richards = D^(-1/ν), dRichards/dD = (-1/ν) * D^(-1/ν - 1)
                (-1.0 / nu) * d.powf(-1.0 / nu - 1.0)
            };

            out[pos] = pref * d_richards_d_d * d_d_beta;
            pos += 1;
        }

        if self.temperature_learnable {
            // Temperature affects input scaling: temp_scaled = adaptive_normalized / temp
            // input = input_scale * (scale * temp_scaled + shift)
            // dinput/dtemp = input_scale * scale * d(temp_scaled)/dtemp
            // d(temp_scaled)/dtemp = d(adaptive_normalized/temp)/dtemp = -adaptive_normalized/temp²
            // = -temp_scaled / temp
            let d_input_d_temp = input_scale * scale * (-temp_scaled / temp);

            let d_richards_d_input = if nu <= 0.0 {
                // Richards = 1/D, dRichards/dinput = d(1/D)/dinput = -1/D² * dD/dinput
                let exp_term = exponent.exp();
                let d = beta + (1.0 - beta) * exp_term;
                - (1.0 / (d * d)) * (-k * exp_term * (1.0 - beta))
            } else {
                // Richards = D^(-1/ν), dRichards/dinput = (-1/ν) * D^(-1/ν-1) * dD/dinput
                let exp_term = exponent.exp();
                let d = beta + (1.0 - beta) * exp_term;
                (-1.0 / nu) * d.powf(-1.0 / nu - 1.0) * (-k * exp_term * (1.0 - beta))
            };

            out[pos] = pref * d_richards_d_input * d_input_d_temp;
            pos += 1;
        }
        if self.output_gain_learnable {
             out[pos] = grad_output * gate;
             pos += 1;
         }
         if self.output_bias_learnable {
             out[pos] = grad_output;
             pos += 1;
         }
        if self.scale_learnable {
            let d_input_d_scale = input_scale * temp_scaled;
            let d_gate_d_scale = outer_scale * dsigma_dinput * d_input_d_scale;
            out[pos] = grad_output * output_gain * d_gate_d_scale;
            pos += 1;
        }
        if self.shift_learnable {
            let d_input_d_shift = input_scale;
            let d_gate_d_shift = outer_scale * dsigma_dinput * d_input_d_shift;
            out[pos] = grad_output * output_gain * d_gate_d_shift;
            pos += 1;
        }

        debug_assert_eq!(pos, out.len(), "grad_weights_scalar_into: slice length mismatch");
    }

    /// Compute gradients w.r.t. learnable parameters for a single scalar input
    pub fn grad_weights_scalar(&self, x: f64, grad_output: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.weights_len()];
        self.grad_weights_scalar_into(x, grad_output, &mut out);
        // Check for NaN/inf values and replace with safe defaults
        for val in &mut out {
            if !val.is_finite() {
                *val = 0.0; // Replace NaN/inf with zero gradient
            }
        }
        out
    }

    /// Derivative for a single scalar x (backward compatibility)
    pub fn backward_scalar(&self, x: f64) -> f64 {
        let (nu_raw, k, m, _, _, output_gain, _, scale, shift) = self.get_all_params();
        let nu = nu_raw.max(1e-6);
        let (input_scale, outer_scale) = self.get_variant_scales();
        let cx = scale * x + shift;
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

        // Chain rule: d/dx [variant_transform(richards(input_scale * (scale*x + shift)))]
        let dgate_dx = dsig_dinput * input_scale * outer_scale;

        // Full derivative: d/dx [output_gain * gate + output_bias] = output_gain * scale * dgate_dx
        output_gain * scale * dgate_dx
    }

    /// Update parameters using Adam optimizer
    pub fn step(&mut self, gradients: &[f64], learning_rate: f64) {
        // Count learnable parameters
        let param_count = [self.nu_learnable, self.k_learnable, self.m_learnable, self.beta_learnable, self.temperature_learnable, self.output_gain_learnable, self.output_bias_learnable, self.scale_learnable, self.shift_learnable].iter().filter(|&&b| b).count();
        
        // Ensure optimizer is properly initialized for the correct number of parameters
        if self.optimizer.is_none() || 
           (self.optimizer.as_ref().unwrap().m.shape() != &[param_count, 1]) {
            self.optimizer = Some(Adam::new((param_count, 1)));
        }
        
        // Extract current parameter values for learnable parameters
        let param_values: Vec<f32> = std::iter::empty()
            .chain(self.nu_learnable.then(|| self.get_param(self.nu, self.learned_nu, 1.0) as f32))
            .chain(self.k_learnable.then(|| self.get_param(self.k, self.learned_k, 1.0) as f32))
            .chain(self.m_learnable.then(|| self.get_param(self.m, self.learned_m, 0.0) as f32))
            .chain(self.beta_learnable.then(|| self.get_param(self.beta, self.learned_beta, 1.0) as f32))
            .chain(self.temperature_learnable.then(|| self.get_param(self.temperature, self.learned_temperature, 1.0) as f32))
            .chain(self.output_gain_learnable.then(|| self.get_param(self.output_gain, self.learned_output_gain, 1.0) as f32))
            .chain(self.output_bias_learnable.then(|| self.get_param(self.output_bias, self.learned_output_bias, 0.0) as f32))
            .chain(self.scale_learnable.then(|| self.get_param(self.scale, self.learned_scale, 1.0) as f32))
            .chain(self.shift_learnable.then(|| self.get_param(self.shift, self.learned_shift, 0.0) as f32))
            .collect();
        
        if let Some(ref mut optimizer) = self.optimizer {
            // Create 2D arrays for Adam optimizer interface
            let mut params = Array2::from_shape_vec((param_count, 1), param_values)
                .expect("Failed to create params array");
            let grads = Array2::from_shape_vec((param_count, 1), gradients.iter().map(|&g| g as f32).collect())
                .expect("Failed to create grads array");
            
            optimizer.step(&mut params, &grads, learning_rate as f32);
            
            // Apply updates back to learned parameters with numerical stability constraints
            let mut idx = 0;
            if self.nu_learnable {
                self.learned_nu = Some((params[[idx, 0]] as f64).clamp(1e-6, 10.0)); // Constrain nu to prevent instability
                idx += 1;
            }
            if self.k_learnable {
                self.learned_k = Some((params[[idx, 0]] as f64).clamp(1e-6, 100.0)); // Constrain k to prevent overflow
                idx += 1;
            }
            if self.m_learnable {
                self.learned_m = Some((params[[idx, 0]] as f64).clamp(-10.0, 10.0)); // Constrain m
                idx += 1;
            }
            if self.beta_learnable {
                self.learned_beta = Some((params[[idx, 0]] as f64).clamp(1e-6, 10.0)); // Constrain beta
                idx += 1;
            }
            if self.temperature_learnable {
                self.learned_temperature = Some((params[[idx, 0]] as f64).clamp(0.1, 10.0)); // Constrain temperature
                idx += 1;
            }
            if self.output_gain_learnable {
                self.learned_output_gain = Some((params[[idx, 0]] as f64).clamp(-10.0, 10.0)); // Constrain gain
                idx += 1;
            }
            if self.output_bias_learnable {
                self.learned_output_bias = Some((params[[idx, 0]] as f64).clamp(-10.0, 10.0)); // Constrain bias
                idx += 1;
            }
            if self.scale_learnable {
                self.learned_scale = Some((params[[idx, 0]] as f64).clamp(-10.0, 10.0)); // Constrain scale
                idx += 1;
            }
            if self.shift_learnable {
                self.learned_shift = Some((params[[idx, 0]] as f64).clamp(-5.0, 5.0)); // Constrain shift
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
        std::iter::empty()
            .chain(self.nu_learnable.then(|| self.get_param(self.nu, self.learned_nu, 1.0)))
            .chain(self.k_learnable.then(|| self.get_param(self.k, self.learned_k, 1.0)))
            .chain(self.m_learnable.then(|| self.get_param(self.m, self.learned_m, 0.0)))
            .chain(self.beta_learnable.then(|| self.get_param(self.beta, self.learned_beta, 1.0)))
            .chain(self.temperature_learnable.then(|| self.get_param(self.temperature, self.learned_temperature, 1.0)))
            .chain(self.output_gain_learnable.then(|| self.get_param(self.output_gain, self.learned_output_gain, 1.0)))
            .chain(self.output_bias_learnable.then(|| self.get_param(self.output_bias, self.learned_output_bias, 0.0)))
            .chain(self.scale_learnable.then(|| self.get_param(self.scale, self.learned_scale, 1.0)))
            .chain(self.shift_learnable.then(|| self.get_param(self.shift, self.learned_shift, 0.0)))
            .collect()
    }

    /// Number of learnable parameters in the internal order
    pub fn weights_len(&self) -> usize {
        [self.nu_learnable, self.k_learnable, self.m_learnable, self.beta_learnable, self.temperature_learnable, self.output_gain_learnable, self.output_bias_learnable, self.scale_learnable, self.shift_learnable].iter().filter(|&&b| b).count()
    }

    /// Iterator over current learnable parameter values (zero-allocation)
    pub fn weights_iter(&self) -> WeightsIter<'_> {
        WeightsIter { curve: self, idx: 0 }
    }

    /// Get current scaling parameters
    pub fn get_scaling(&self) -> (f64, f64) {
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        (scale, shift)
    }

    /// Setter for learning updates (e.g., from optimizer).
    pub fn set_param(&mut self, nu: Option<f64>, k: Option<f64>, m: Option<f64>, beta: Option<f64>, output_gain: Option<f64>, output_bias: Option<f64>) {
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
        if let Some(output_gain_val) = output_gain {
            self.output_gain = Some(output_gain_val);
        }
        if let Some(output_bias_val) = output_bias {
            self.output_bias = Some(output_bias_val);
        }
    }

    /// Update running statistics from input batch (for Adaptive variant)
    /// This tracks mean and variance to automatically adapt scale/shift parameters
    pub fn update_running_stats(&mut self, x: &Array1<f64>) {
        if self.variant != Variant::Adaptive {
            return; // Only Adaptive variant uses running statistics
        }

        if self.running_sum.is_none() || self.running_sq_sum.is_none() || self.count.is_none() {
            // Initialize if not already done
            self.running_sum = Some(0.0);
            self.running_sq_sum = Some(0.0);
            self.count = Some(0);
        }

        let current_count = self.count.unwrap();
        let new_count = current_count + x.len() as u64;
        let batch_mean = x.mean().unwrap_or(0.0);
        let batch_var_sum: f64 = x.iter().map(|&xi| (xi - batch_mean).powi(2)).sum();

        // Update running statistics with momentum
        let momentum = self.momentum.max(1e-7); // Ensure minimum momentum for stability
        let new_running_sum = (self.running_sum.unwrap() * momentum + x.sum() * (1.0 - momentum)) as f64;
        let new_running_sq_sum = (self.running_sq_sum.unwrap() * momentum + batch_var_sum * (1.0 - momentum)) as f64;

        self.running_sum = Some(new_running_sum);
        self.running_sq_sum = Some(new_running_sq_sum);
        self.count = Some(new_count);

        self.update_adaptive_scaling();
    }

    /// Update adaptive scale and shift from running statistics
    fn update_adaptive_scaling(&mut self) {
        if let (Some(running_sum), Some(running_sq_sum), Some(count)) = (self.running_sum, self.running_sq_sum, self.count) {
            if count > 1 {
                let mean = running_sum / count as f64;
                let variance = (running_sq_sum / (count - 1) as f64) - (running_sum.powi(2) / count as f64) / (count - 1) as f64;
                let std = variance.sqrt().max(1e-6); // Minimum std for numerical stability

                // Adaptive normalization: center at mean, scale to unit variance
                self.adaptive_scale = Some(1.0 / std);
                self.adaptive_shift = Some(-mean / std);
            }
        }
    }

    /// Get adaptive scaling parameters (or default to (1.0, 0.0) if not adaptive)
    fn get_adaptive_scaling(&self) -> (f64, f64) {
        if self.variant == Variant::Adaptive {
            (self.adaptive_scale.unwrap_or(1.0), self.adaptive_shift.unwrap_or(0.0))
        } else {
            (1.0, 0.0) // Identity transformation for non-adaptive variants
        }
    }

    /// Reset running statistics (useful for new training epochs)
    pub fn reset_running_stats(&mut self) {
        if self.variant == Variant::Adaptive {
            self.running_sum = Some(0.0);
            self.running_sq_sum = Some(0.0);
            self.count = Some(0);
            self.adaptive_scale = Some(1.0);
            self.adaptive_shift = Some(0.0);
        }
    }

    /// Set polynomial coefficients for Polynomial variant
    /// Coefficients are [coeff_0, coeff_1, coeff_2, ..., coeff_power]
    /// defining polynomial: coeff_0 + coeff_1*x + coeff_2*x^2 + ... + coeff_power*x^power
    pub fn set_polynomial(&mut self, power: usize, coeffs: Vec<f64>) -> Result<(), String> {
        if self.variant != Variant::Polynomial {
            return Err("Can only set polynomial coefficients for Polynomial variant".to_string());
        }
        if power < 1 || power > 5 {
            return Err("Polynomial degree must be between 1 and 5".to_string());
        }
        if coeffs.len() != power + 1 {
            return Err(format!("Expected {} coefficients for degree {}, got {}", power + 1, power, coeffs.len()));
        }

        self.poly_power = Some(power);
        self.poly_coeffs = Some(coeffs);
        Ok(())
    }

    /// Get polynomial degree (or 1 for identity if not polynomial variant)
    fn get_polynomial_power(&self) -> usize {
        self.poly_power.unwrap_or(1)
    }

    /// Evaluate polynomial at a given point
    fn evaluate_polynomial(&self, x: f64) -> f64 {
        if let Some(coeffs) = &self.poly_coeffs {
            coeffs.iter().enumerate().fold(0.0, |sum, (i, &coeff)| {
                sum + coeff * x.powi(i as i32)
            })
        } else {
            // Identity if no coefficients set
            x
        }
    }

    /// Get polynomial-input scaling (applied before Richards activation)
    fn get_polynomial_scaling(&self) -> f64 {
        if self.variant == Variant::Polynomial {
            self.evaluate_polynomial(1.0) // Evaluate at x=1 for scaling check
        } else {
            1.0 // Identity scaling for non-polynomial variants
        }
    }
}

// Zero-allocation iterator over RichardsCurve learnable weights in internal order
pub struct WeightsIter<'a> {
    curve: &'a RichardsCurve,
    idx: usize,
}

impl<'a> Iterator for WeightsIter<'a> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.idx {
                0 => {
                    self.idx += 1;
                    if self.curve.nu_learnable {
                        return Some(self.curve.get_param(self.curve.nu, self.curve.learned_nu, 1.0));
                    }
                }
                1 => {
                    self.idx += 1;
                    if self.curve.k_learnable {
                        return Some(self.curve.get_param(self.curve.k, self.curve.learned_k, 1.0));
                    }
                }
                2 => {
                    self.idx += 1;
                    if self.curve.m_learnable {
                        return Some(self.curve.get_param(self.curve.m, self.curve.learned_m, 0.0));
                    }
                }
                3 => {
                    self.idx += 1;
                    if self.curve.beta_learnable {
                        return Some(self.curve.get_param(self.curve.beta, self.curve.learned_beta, 1.0));
                    }
                }
                4 => {
                    self.idx += 1;
                    if self.curve.temperature_learnable {
                        return Some(self.curve.get_param(self.curve.temperature, self.curve.learned_temperature, 1.0));
                    }
                }
                5 => {
                    self.idx += 1;
                    if self.curve.output_gain_learnable {
                        return Some(self.curve.get_param(self.curve.output_gain, self.curve.learned_output_gain, 1.0));
                    }
                }
                6 => {
                    self.idx += 1;
                    if self.curve.output_bias_learnable {
                        return Some(self.curve.get_param(self.curve.output_bias, self.curve.learned_output_bias, 0.0));
                    }
                }
                7 => {
                    self.idx += 1;
                    if self.curve.scale_learnable {
                        return Some(self.curve.get_param(self.curve.scale, self.curve.learned_scale, 1.0));
                    }
                }
                8 => {
                    self.idx += 1;
                    if self.curve.shift_learnable {
                        return Some(self.curve.get_param(self.curve.shift, self.curve.learned_shift, 0.0));
                    }
                }
                _ => return None,
            }
        }
    }
}
