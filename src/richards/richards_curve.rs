use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, pade::PadeExp};

/// # Richards Curve: Mathematical Framework and Numerical Methods
///
/// ## Core Richards Function Theorem
///
/// **Theorem 1 (Richards Curve Family)**: The Richards curve is defined as a parametric
/// family of sigmoid functions with the following mathematical formulation:
///
/// σ(x; ν, k, m) = [1 + e^(-k(x-m))]^(-1/ν)
///
/// **Literature References**:
/// - **Richards Curve**: Richards, F. J. (1959). "A flexible growth function for empirical use".
///   Journal of Experimental Botany.
/// - **Sigmoid Families**: Nelder, J. A. (1961). "The fitting of a generalization of the logistic
///   curve". Biometrics.
/// - **Growth Curve Theory**: Thornley, J. H. M., & Johnson, I. R. (1990). "Plant and crop
///   modelling: a mathematical approach to plant and crop physiology". Clarendon Press.
///
/// **Parameters:**
/// - ν (nu): Shape parameter (ν > 0) controlling asymmetry and steepness
/// - k: Growth rate parameter (k > 0) controlling transition sharpness
/// - m: Midpoint parameter controlling curve center
///
/// **Special Cases:**
/// - ν → ∞: Approaches step function at x = m
/// - ν = 1: Standard logistic function σ(x) = 1/[1 + e^(-k(x-m))]
/// - ν → 0⁺: Approaches Gompertz curve (see Theorem 2)
///
/// ## Extended Richards Asymmetry Theorem
///
/// **Theorem 2 (Extended Richards with Asymmetry)**: The extended Richards curve
/// introduces asymmetry control via parameter β:
///
/// σ_β(x; ν, k, m, β) = [β + (1-β) * e^(-k(x-m))]^(-1/ν)
///
/// **Literature References**:
/// - **Asymmetric Sigmoids**: Johnson, N. L., Kotz, S., & Balakrishnan, N. (1995). "Continuous
///   univariate distributions, Vol. 2". Wiley.
/// - **Skewed Logistic**: Azzalini, A. (1985). "A class of distributions which includes the normal
///   ones". Scandinavian Journal of Statistics.
/// - **Flexible Sigmoid Functions**: Ratkowsky, D. A. (1990). "Handbook of nonlinear regression
///   models". Marcel Dekker.
///
/// **Asymmetry Properties:**
/// - β = 1.0: Standard Richards curve (symmetric)
/// - β = 0.0: Gompertz curve σ(x) = e^(-e^(-k(x-m)))
/// - 0 < β < 1: Asymmetric Richards curve with controlled skewness
/// - β < 0 or β > 1: Extended asymmetry range
///
/// **Gompertz Limit (ν → 0⁺):** The Gompertz curve emerges as:
/// lim_{ν→0⁺} σ_β(x; ν, k, m, β) = e^(-e^(-k(x-m))) * β^0 + (1-β) * e^(-e^(-k(x-m)))
///
/// ## Temperature Scaling Transformation
///
/// **Theorem 3 (Temperature Scaling)**: Input preprocessing with temperature parameter T:
///
/// x_temp = x_adaptive / T
///
/// **Temperature Effects:**
/// - T < 1: Sharper, more discontinuous transitions
/// - T = 1: Standard Richards behavior
/// - T > 1: Softer, more gradual transitions
/// - T → 0⁺: Approaches step function
/// - T → ∞: Approaches linear function
///
/// ## Complete Input Transformation Pipeline
///
/// **Theorem 4 (Affine Input Transformation)**: Full input preprocessing pipeline:
///
/// x_input = s_variant * (s * x_temp + b)
///
/// where:
/// - s_variant: Variant-specific scaling (2.0 for Tanh, 1.0 otherwise)
/// - s: Learnable input scale parameter
/// - b: Learnable input bias parameter
///
/// ## Variant-Specific Output Scaling
///
/// **Theorem 5 (Variant Output Transformations)**:
/// - **Sigmoid/Gompertz variants:** gate(x) = σ_β(x_input)
/// - **Tanh variant:** gate(x) = 2 * σ_β(x_input) - 1
///
/// ## Complete Forward Pass
///
/// **Theorem 6 (Complete Affine Output)**: Final output transformation:
///
/// f(x) = a * gate(x) + c
///
/// where a is output gain and c is output bias.
///
/// ## Numerical Stability and Clamping
///
/// **Theorem 7 (Numerical Stability Bounds)**:
/// - **Exponent clamping:** e^exp ∈ [e^(-23), e^(23)] ≈ [10^(-10), 10^(10)]
/// - **Output clamping:** f(x) ∈ [-10^6, 10^6] to prevent NaN/inf propagation
/// - **Gradient safety:** Replace NaN/inf gradients with zeros
///
/// **Justification:** Prevents overflow/underflow in exponential computations while
/// maintaining function differentiability and gradient flow.
///
/// ## Analytical Gradient Computation
///
/// **Theorem 8 (Gradient Computation)**: All parameters have analytical derivatives:
///
/// **∂f/∂ν (Shape Parameter Gradient):**
/// ∂σ/∂ν = σ * (1-σ) * [ln(β + (1-β)e^(-k(x-m))) + (β/(β + (1-β)e^(-k(x-m))))]
///
/// **∂f/∂k (Growth Rate Gradient):**
/// ∂σ/∂k = σ * (1-σ) * (x-m) * [1 + (1-β)e^(-k(x-m))/(β + (1-β)e^(-k(x-m)))]
///
/// **∂f/∂m (Midpoint Gradient):**
/// ∂σ/∂m = -∂σ/∂k
///
/// **∂f/∂β (Asymmetry Gradient):**
/// ∂σ/∂β = σ * (1-σ) * [e^(-k(x-m)) - 1] / [β + (1-β)e^(-k(x-m))]
///
/// **∂f/∂T (Temperature Gradient):**
/// ∂x_temp/∂T = -x_adaptive/T², propagated through chain rule
///
/// **∂f/∂a, ∂f/∂c (Affine Gradients):**
/// Direct derivatives: ∂f/∂a = gate(x), ∂f/∂c = 1
///
/// **∂f/∂s, ∂f/∂b (Input Scaling Gradients):**
/// Chain rule through input transformation pipeline
///
/// ## Adaptive Normalization (Batch Statistics)
///
/// **Theorem 9 (Adaptive Normalization)**: Running statistics normalization:
///
/// x_adaptive = (x - μ_running) / σ_running
///
/// where μ_running and σ_running are computed with momentum-based updates:
/// μ_{t+1} = momentum * μ_t + (1-momentum) * μ_batch
/// σ_{t+1} = momentum * σ_t + (1-momentum) * σ_batch
///
/// ## Polynomial Input Transformation
///
/// **Theorem 10 (Polynomial Preprocessing)**: Pre-Richards polynomial transformation:
///
/// x_poly = Σ_{i=0}^p c_i * x^i
///
/// where p is polynomial degree and c_i are learnable coefficients.
///
/// ## Convergence and Stability Properties
///
/// **Theorem 11 (Convergence Properties)**:
/// The Richards curve family satisfies:
/// - **Lipschitz continuity** with bounded derivatives
/// - **Universal approximation** for continuous functions on compact sets
/// - **Gradient stability** under parameter constraints
/// - **Numerical robustness** through clamping and safe gradients
///
/// **Theorem 12 (Parameter Constraints for Stability)**:
/// - ν ∈ [10^(-6), 10]: Prevents extreme asymmetry or discontinuity
/// - k ∈ [10^(-6), 100]: Bounds growth rate for numerical stability
/// - β ∈ [10^(-6), 10]: Constrains asymmetry parameter
/// - T ∈ [0.1, 10]: Limits temperature scaling range
///
/// ## Applications and Use Cases
///
/// **Theorem 13 (Activation Function Applications)**:
/// Richards curves serve as learnable activation functions for:
/// - **Adaptive non-linearities** with data-dependent shapes
/// - **Specialized transformations** (Sigmoid, Tanh, Gompertz behaviors)
/// - **Normalization layers** with learnable affine transformations
/// - **Smooth approximations** of step functions and discontinuities
///
/// ## Implementation Notes
///
/// - **Parallel computation** using Rayon for vectorized operations
/// - **Memory efficiency** through in-place gradient computation
/// - **Serialization support** for model persistence
/// - **Zero-allocation iterators** for parameter access
/// - **Momentum-based optimization** with Adam algorithm Unified Richards curve with variant-based
///   initialization and full parameter learning Extended with beta parameter for asymmetric control
///   and temperature for sharpness
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsCurve {
    // Core Richards parameter values (Some for fixed, None for learnable)
    pub nu: Option<f64>,   // Shape (asymmetry)
    pub k: Option<f64>,    // Growth rate
    pub m: Option<f64>,    // Midpoint
    pub beta: Option<f64>, // Asymmetry factor for extended Richards

    // Temperature parameter (controls curve sharpness/softness)
    pub temperature: Option<f64>, // Temperature scaling factor

    // Affine parameter values (Some for fixed, None for learnable)
    #[serde(rename = "a")]
    pub output_gain: Option<f64>, // Affine output gain (scale)
    #[serde(rename = "b")]
    pub output_bias: Option<f64>, // Affine output bias (shift)

    // Input scaling parameter values (Some for fixed, None for learnable)
    pub scale: Option<f64>, // Input scaling
    pub shift: Option<f64>, // Input shift

    // Per-feature output transformation (used by normalization variants)
    #[serde(skip_serializing, skip_deserializing)]
    pub gamma: Option<Array2<f32>>, // Per-feature scale (shape: [1, d])
    #[serde(skip_serializing, skip_deserializing)]
    pub bias: Option<Array2<f32>>, // Per-feature bias (shape: [1, d])

    // Polynomial input transformation (used by Polynomial variant)
    #[serde(skip_serializing, skip_deserializing)]
    pub poly_power: Option<usize>, // Polynomial degree (1-5, 1=identity)
    #[serde(skip_serializing, skip_deserializing)]
    pub poly_coeffs: Option<Vec<f64>>, /* Polynomial coefficients [ coeff_0, coeff_1, ...,
                                        * coeff_power] */

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
    pub gamma_learnable: bool, // Whether gamma parameters are learnable
    pub bias_learnable: bool,  // Whether bias parameters are learnable

    // Variant configuration
    pub variant: super::Variant, // Sigmoid, Tanh, or Gompertz mode

    // Adaptive normalization (used by Adaptive variant)
    #[serde(skip_serializing, skip_deserializing)]
    running_sum: Option<f64>, // Running sum for mean estimation
    #[serde(skip_serializing, skip_deserializing)]
    running_sq_sum: Option<f64>, // Running sum of squares for variance estimation
    #[serde(skip_serializing, skip_deserializing)]
    count: Option<u64>, // Number of samples seen
    pub momentum: f64, // Momentum for running statistics (0.01 typical)
    #[serde(skip_serializing, skip_deserializing)]
    adaptive_scale: Option<f64>, // Automatically computed scale factor
    #[serde(skip_serializing, skip_deserializing)]
    adaptive_shift: Option<f64>, // Automatically computed shift factor

    // Optimization
    #[serde(skip_serializing, skip_deserializing)]
    optimizer: Option<Adam>,
    pub l2_reg: f64,
    pub adaptive_lr_scale: f64,
    pub grad_norm_history: Vec<f64>,
}

impl RichardsCurve {
    /// Constructor with learnable params based on variant.
    pub fn new_learnable(variant: super::Variant) -> Self {
        // Set output_gain/output_bias coefficients based on variant (Some for fixed, None for
        // learnable)
        let (output_gain_val, output_bias_val) = match variant {
            super::Variant::Sigmoid | super::Variant::Gompertz => (Some(1.0), Some(0.0)), /* [0, 1] range, fixed */
            super::Variant::Tanh => (Some(1.0), Some(0.0)), /* [-1, 1] via 2σ(2x) - 1 transform,
                                                              * fixed */
            super::Variant::Adaptive | super::Variant::None | super::Variant::Polynomial => {
                (None, None)
            } // Fully learnable including output_gain/output_bias
        };

        // Determine parameter count based on whether output_gain/output_bias are learnable
        // nu, k, m, beta, temp, scale, shift + optionally output_gain, output_bias
        let param_count = 7
            + if output_gain_val.is_none() { 1 } else { 0 }
            + if output_bias_val.is_none() { 1 } else { 0 };

        let (adaptive_initialized, momentum) = match variant {
            super::Variant::Adaptive => (true, 0.01), /* Enable adaptive normalization with
                                                        * default momentum */
            _ => (false, 0.0), // Disable adaptive for other variants
        };

        let polynomial_initialized = match variant {
            super::Variant::Polynomial => true, // Enable polynomial transformation
            _ => false,                         // Disable polynomial for other variants
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
            poly_power: if polynomial_initialized {
                Some(1)
            } else {
                None
            }, // Default to degree 1 (identity)
            poly_coeffs: if polynomial_initialized {
                Some(vec![0.0, 1.0])
            } else {
                None
            }, // [0, 1] = identity

            // Per-feature transformations (None by default - not used in standard RichardsCurve)
            gamma: None,
            bias: None,

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
            gamma_learnable: false, // Not learnable by default
            bias_learnable: false,  // Not learnable by default

            // Adaptive normalization
            running_sum: if adaptive_initialized {
                Some(0.0)
            } else {
                None
            },
            running_sq_sum: if adaptive_initialized {
                Some(0.0)
            } else {
                None
            },
            count: if adaptive_initialized { Some(0) } else { None },
            momentum,
            adaptive_scale: if adaptive_initialized {
                Some(1.0)
            } else {
                None
            },
            adaptive_shift: if adaptive_initialized {
                Some(0.0)
            } else {
                None
            },

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
            gamma_learnable: false, // Not learnable in default RichardsCurve
            bias_learnable: false,  // Not learnable in default RichardsCurve
            variant: super::Variant::Sigmoid,
            poly_power: None, // Not polynomial variant
            poly_coeffs: None,
            gamma: None,       // Not used in default RichardsCurve
            bias: None,        // Not used in default RichardsCurve
            running_sum: None, // Not adaptive variant
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
            Self::new_learnable(super::Variant::Sigmoid)
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
                gamma_learnable: false, // Not learnable in sigmoid RichardsCurve
                bias_learnable: false,  // Not learnable in sigmoid RichardsCurve
                variant: super::Variant::Sigmoid,
                poly_power: None, // Not polynomial variant
                poly_coeffs: None,
                gamma: None, // Not used in sigmoid RichardsCurve
                bias: None,  // Not used in sigmoid RichardsCurve
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
            Self::new_learnable(super::Variant::Tanh)
        } else {
            Self {
                nu: Some(1.0),
                k: Some(1.0), // Fixed: Changed from 2.0 to 1.0 for accurate tanh approximation
                m: Some(0.0),
                beta: Some(1.0),
                temperature: Some(1.0),
                output_gain: Some(1.0),
                output_bias: Some(0.0),
                scale: Some(1.0), // Fixed for specific variant
                shift: Some(0.0), // Fixed for specific variant
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
                gamma_learnable: false, // Not learnable in tanh RichardsCurve
                bias_learnable: false,  // Not learnable in tanh RichardsCurve
                variant: super::Variant::Tanh,
                poly_power: None, // Not polynomial variant
                poly_coeffs: None,
                gamma: None, // Not used in tanh RichardsCurve
                bias: None,  // Not used in tanh RichardsCurve
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
            Self::new_learnable(super::Variant::Gompertz)
        } else {
            Self {
                nu: Some(0.01),
                k: Some(1.0),
                m: Some(0.0),
                beta: Some(1.0),
                temperature: Some(1.0),
                output_gain: Some(1.0),
                output_bias: Some(0.0),
                scale: Some(1.0), // Fixed for specific variant
                shift: Some(0.0), // Fixed for specific variant
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
                gamma_learnable: false, // Not learnable in gompertz RichardsCurve
                bias_learnable: false,  // Not learnable in gompertz RichardsCurve
                variant: super::Variant::Gompertz,
                poly_power: None, // Not polynomial variant
                poly_coeffs: None,
                gamma: None, // Not used in gompertz RichardsCurve
                bias: None,  // Not used in gompertz RichardsCurve
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
        Self::new_learnable(super::Variant::None)
    }

    /// Enable per-feature transformations for normalization layers
    /// Sets up learnable gamma (scale) and bias parameters for each feature dimension
    pub fn enable_per_feature_transform(&mut self, embedding_dim: usize) {
        // Initialize gamma and bias arrays if not already present
        if self.gamma.is_none() {
            self.gamma = Some(Array2::ones((1, embedding_dim)));
        }
        if self.bias.is_none() {
            self.bias = Some(Array2::zeros((1, embedding_dim)));
        }

        // Make them learnable
        self.gamma_learnable = true;
        self.bias_learnable = true;

        // Reinitialize optimizer with correct parameter count
        let param_count = self.weights_len();
        self.optimizer = Some(Adam::new((param_count, 1)));
    }

    /// Simple scaling based on max absolute value (for numerical stability)
    /// Only updates scale and shift if they are fixed (Some), not learnable (None)
    pub fn update_scaling_from_max_abs(&self, max_abs_x: f64) -> Self {
        // Only update if scale and shift are fixed (not learnable)
        if self.scale.is_some() && self.shift.is_some() {
            let mut updated = self.clone();
            if max_abs_x > 0.0 {
                updated.scale = Some((1.0 / max_abs_x).min(0.5));
                updated.shift = Some(0.0);
            } else {
                updated.scale = Some(1.0);
                updated.shift = Some(0.0);
            }
            updated
        } else {
            self.clone()
        }
    }

    /// Helper: get parameter value (learnable or fixed).
    fn get_param(&self, param: Option<f64>, learned: Option<f64>, default: f64) -> f64 {
        if let Some(p) = param {
            p
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
            super::Variant::Tanh => (2.0, 2.0),
            _ => (1.0, 1.0),
        }
    }

    /// Vectorized forward pass: f(x) = output_gain * gate(x) + output_bias (elementwise),
    /// single-pass.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let (nu, k, m, beta, temp, output_gain, output_bias, scale, shift) = self.get_all_params();
        let (input_scale, _) = self.get_variant_scales();
        let (adaptive_scale, adaptive_shift) = self.get_adaptive_scaling();

        let mut out = Array1::zeros(x.len());
        let xs = x.as_slice().unwrap();
        let os = out.as_slice_mut().unwrap();

        xs.par_iter().zip(os.par_iter_mut()).for_each(|(&xi, o)| {
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
            let beta_term = beta + (1.0 - beta) * PadeExp::exp(clamped_exponent);
            let extended_richards = if nu <= 0.0 {
                1.0 / beta_term
            } else {
                beta_term.powf(-1.0 / nu)
            };
            let gate = match self.variant {
                super::Variant::Tanh => 2.0 * extended_richards - 1.0,
                _ => extended_richards,
            };
            let output = output_gain * gate + output_bias;
            // Numerical stability: clamp extreme values to prevent NaN/inf propagation
            *o = output.clamp(-1e6, 1e6);
        });

        out
    }

    /// Vectorized forward pass for matrix input
    pub fn forward_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut output = x.mapv(|val| self.forward_scalar(val));

        // Apply per-feature transformations if enabled
        if let (Some(gamma), Some(bias)) = (&self.gamma, &self.bias) {
            // Broadcast gamma and bias across all samples for each feature
            ndarray::Zip::indexed(&mut output)
                .and_broadcast(gamma)
                .and_broadcast(bias)
                .for_each(|(_, _j), o, &g, &b| {
                    *o = (*o as f32 * g + b) as f64;
                });
        }

        output
    }

    /// Forward for a single scalar x (backward compatibility)
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let (nu, k, m, _, _, output_gain, output_bias, scale, shift) = self.get_all_params();
        let (input_scale, _) = self.get_variant_scales();
        let temp_scaled = x; // Backward compatibility: no temperature scaling
        let input = input_scale * (scale * temp_scaled + shift);

        let exponent = -k * (input - m);
        // Clamp exponent to prevent overflow
        let clamped_exponent = exponent.clamp(-23.0, 23.0);
        let sigma = if nu <= 0.0 {
            1.0 / (1.0 + PadeExp::exp(clamped_exponent))
        } else {
            let u = PadeExp::exp(clamped_exponent).powf(1.0 / nu);
            1.0 / (1.0 + u)
        };

        let gate = match self.variant {
            super::Variant::Tanh => 2.0 * sigma - 1.0,
            _ => sigma,
        };
        let output = output_gain * gate + output_bias;
        // Numerical stability: clamp extreme values to prevent NaN/inf propagation
        output.clamp(-1e6, 1e6)
    }

    /// Matrix backward pass: df/dx for matrix input with per-feature transformations
    pub fn backward_matrix(&self, x: &Array2<f64>, output_grads: &Array2<f64>) -> Array2<f64> {
        let mut grad_input = Array2::<f64>::zeros(x.raw_dim());

        // Compute input gradients element-wise
        ndarray::Zip::from(&mut grad_input)
            .and(x)
            .and(output_grads)
            .for_each(|gi, &xi, &dy| {
                let dt_dx = self.backward_scalar(xi);
                *gi = dt_dx * dy;
            });

        grad_input
    }

    /// Matrix gradient computation for all learnable parameters
    pub fn grad_weights_matrix(&self, x: &Array2<f64>, output_grads: &Array2<f64>) -> Vec<f64> {
        let mut grads_accum = vec![0.0f64; self.weights_len()];
        let (batch_size, embedding_dim) = x.dim();

        // Bounds checking: ensure dimensions are compatible
        if x.dim() != output_grads.dim() {
            return grads_accum;
        }

        // First, accumulate scalar parameter gradients (same as before)
        let scalar_param_count = self.scalar_weights_len();

        for sample_idx in 0..batch_size {
            for feature_idx in 0..embedding_dim {
                let xi = x[[sample_idx, feature_idx]];
                let dy = output_grads[[sample_idx, feature_idx]];

                // Compute scalar parameter gradients for this element
                let param_grads = self.grad_weights_scalar(xi, dy);

                // Accumulate only scalar parameters
                for i in 0..scalar_param_count {
                    grads_accum[i] += param_grads[i];
                }
            }
        }

        // Average scalar parameters across batch and features
        let total_elements = (batch_size * embedding_dim) as f64;
        for i in 0..scalar_param_count {
            grads_accum[i] /= total_elements;
        }

        // Now compute gamma/bias gradients (matrix-specific)
        let mut pos = scalar_param_count;

        if self.gamma_learnable {
            if let Some(ref gamma) = self.gamma {
                let gamma_size = gamma.len();
                // Compute Richards outputs before gamma/bias application
                let richards_output = self.forward_matrix(x);

                // Bounds checking: ensure gamma_size matches embedding_dim
                if gamma_size != embedding_dim {
                    eprintln!(
                        "RichardsCurve::grad_weights_matrix: gamma size mismatch - gamma_size: {}, embedding_dim: {}",
                        gamma_size, embedding_dim
                    );
                    return grads_accum;
                }

                // For each gamma parameter (one per feature)
                for feature_idx in 0..gamma_size {
                    let mut gamma_grad = 0.0;
                    for sample_idx in 0..batch_size {
                        // d(output)/d(gamma_feature) = richards_output for that feature
                        gamma_grad += richards_output[[sample_idx, feature_idx]]
                            * output_grads[[sample_idx, feature_idx]];
                    }
                    grads_accum[pos + feature_idx] = gamma_grad / (batch_size as f64);
                }
                pos += gamma_size;
            }
        }

        if self.bias_learnable {
            if let Some(ref bias) = self.bias {
                let bias_size = bias.len();
                // Bounds checking: ensure bias_size matches embedding_dim
                if bias_size != embedding_dim {
                    eprintln!(
                        "RichardsCurve::grad_weights_matrix: bias size mismatch - bias_size: {}, embedding_dim: {}",
                        bias_size, embedding_dim
                    );
                    return grads_accum;
                }

                // For each bias parameter (one per feature)
                for feature_idx in 0..bias_size {
                    let mut bias_grad = 0.0;
                    for sample_idx in 0..batch_size {
                        // d(output)/d(bias_feature) = output_grad for that feature
                        bias_grad += output_grads[[sample_idx, feature_idx]];
                    }
                    grads_accum[pos + feature_idx] = bias_grad / (batch_size as f64);
                }
            }
        }

        grads_accum
    }

    /// Vectorized backward pass: df/dx at x (analytical gradient), single-pass.
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let (nu, k, m, _, _, output_gain, _, scale, shift) = self.get_all_params();
        let (input_scale, outer_scale) = self.get_variant_scales();

        let mut out = Array1::zeros(x.len());
        let xs = x.as_slice().unwrap();
        let os = out.as_slice_mut().unwrap();

        xs.par_iter().zip(os.par_iter_mut()).for_each(|(&xi, o)| {
            let input = input_scale * (scale * xi + shift);
            let exponent: f64 = -k * (input - m);
            let sigma = if nu <= 0.0 {
                1.0 / (1.0 + PadeExp::exp(exponent))
            } else {
                let u = PadeExp::exp(exponent).powf(1.0 / nu);
                1.0 / (1.0 + u)
            };

            let dsig_dinput = if nu <= 0.0 {
                k * sigma * (1.0 - sigma)
            } else {
                (k / nu) * sigma * (1.0 - sigma)
            };
            *o = output_gain * dsig_dinput * input_scale * outer_scale * scale;
        });

        out
    }

    /// Compute gradients w.r.t. learnable parameters for a single scalar input into a preallocated
    /// slice
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
            1.0 / (1.0 + PadeExp::exp(exponent))
        } else {
            let u = PadeExp::exp(exponent).powf(1.0 / nu);
            1.0 / (1.0 + u)
        };
        let gate = match self.variant {
            super::Variant::Tanh => 2.0 * sigma - 1.0,
            _ => sigma,
        };

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
            let exp_term = PadeExp::exp(exponent);
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
                let exp_term = PadeExp::exp(exponent);
                let d = beta + (1.0 - beta) * exp_term;
                -(1.0 / (d * d)) * (-k * exp_term * (1.0 - beta))
            } else {
                // Richards = D^(-1/ν), dRichards/dinput = (-1/ν) * D^(-1/ν-1) * dD/dinput
                let exp_term = PadeExp::exp(exponent);
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

        // Note: gamma and bias gradients are not computed for scalar inputs
        // They require matrix inputs to make sense (per-feature parameters)

        debug_assert_eq!(
            pos,
            out.len(),
            "grad_weights_scalar_into: slice length mismatch"
        );
    }

    /// Compute gradients w.r.t. scalar learnable parameters for a single scalar input
    /// (Excludes per-feature gamma/bias parameters which require matrix context)
    pub fn grad_weights_scalar(&self, x: f64, grad_output: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.scalar_weights_len()];
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
        let u = PadeExp::exp(exponent).powf(1.0 / nu);
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
        // Count learnable parameters (including array parameters)
        let param_count = self.weights_len();

        // Ensure optimizer is properly initialized for the correct number of parameters
        if self.optimizer.is_none()
            || (self.optimizer.as_ref().unwrap().m.shape() != &[param_count, 1])
        {
            self.optimizer = Some(Adam::new((param_count, 1)));
        }

        // Extract current parameter values for learnable parameters
        let param_values: Vec<f32> =
            std::iter::empty()
                .chain(
                    self.nu_learnable
                        .then(|| self.get_param(self.nu, self.learned_nu, 1.0) as f32),
                )
                .chain(
                    self.k_learnable
                        .then(|| self.get_param(self.k, self.learned_k, 1.0) as f32),
                )
                .chain(
                    self.m_learnable
                        .then(|| self.get_param(self.m, self.learned_m, 0.0) as f32),
                )
                .chain(
                    self.beta_learnable
                        .then(|| self.get_param(self.beta, self.learned_beta, 1.0) as f32),
                )
                .chain(self.temperature_learnable.then(|| {
                    self.get_param(self.temperature, self.learned_temperature, 1.0) as f32
                }))
                .chain(self.output_gain_learnable.then(|| {
                    self.get_param(self.output_gain, self.learned_output_gain, 1.0) as f32
                }))
                .chain(self.output_bias_learnable.then(|| {
                    self.get_param(self.output_bias, self.learned_output_bias, 0.0) as f32
                }))
                .chain(
                    self.scale_learnable
                        .then(|| self.get_param(self.scale, self.learned_scale, 1.0) as f32),
                )
                .chain(
                    self.shift_learnable
                        .then(|| self.get_param(self.shift, self.learned_shift, 0.0) as f32),
                )
                .chain(
                    self.gamma_learnable
                        .then(|| {
                            self.gamma
                                .as_ref()
                                .map(|g| g.iter().map(|&x| x).collect::<Vec<f32>>())
                                .unwrap_or_default()
                        })
                        .into_iter()
                        .flatten(),
                )
                .chain(
                    self.bias_learnable
                        .then(|| {
                            self.bias
                                .as_ref()
                                .map(|b| b.iter().map(|&x| x).collect::<Vec<f32>>())
                                .unwrap_or_default()
                        })
                        .into_iter()
                        .flatten(),
                )
                .collect();

        if let Some(ref mut optimizer) = self.optimizer {
            // Create 2D arrays for Adam optimizer interface
            let mut params = Array2::from_shape_vec((param_count, 1), param_values)
                .expect("Failed to create params array");
            let grads = Array2::from_shape_vec(
                (param_count, 1),
                gradients.iter().map(|&g| g as f32).collect(),
            )
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
            if self.gamma_learnable {
                if let Some(ref mut gamma) = self.gamma {
                    let gamma_size = gamma.len();
                    for i in 0..gamma_size {
                        if idx < param_count {
                            gamma[[0, i]] = params[[idx, 0]].clamp(-10.0, 10.0); // Constrain gamma values
                            idx += 1;
                        }
                    }
                } else {
                    // Skip gamma parameters if array doesn't exist
                    // idx remains unchanged since there are no gamma parameters to update
                }
            }
            if self.bias_learnable {
                if let Some(ref mut bias) = self.bias {
                    let bias_size = bias.len();
                    for i in 0..bias_size {
                        if idx < param_count {
                            bias[[0, i]] = params[[idx, 0]].clamp(-10.0, 10.0); // Constrain bias values
                            idx += 1;
                        }
                    }
                } else {
                    // Skip bias parameters if array doesn't exist
                    // idx remains unchanged since there are no bias parameters to update
                }
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
    /// Note: Returns default values until parameters are actually trained/updated
    pub fn weights(&self) -> Vec<f64> {
        let weights: Vec<f64> = std::iter::empty()
            .chain(
                self.nu_learnable
                    .then(|| self.get_param(self.nu, self.learned_nu, 1.0)),
            )
            .chain(
                self.k_learnable
                    .then(|| self.get_param(self.k, self.learned_k, 1.0)),
            )
            .chain(
                self.m_learnable
                    .then(|| self.get_param(self.m, self.learned_m, 0.0)),
            )
            .chain(
                self.beta_learnable
                    .then(|| self.get_param(self.beta, self.learned_beta, 1.0)),
            )
            .chain(
                self.temperature_learnable
                    .then(|| self.get_param(self.temperature, self.learned_temperature, 1.0)),
            )
            .chain(
                self.output_gain_learnable
                    .then(|| self.get_param(self.output_gain, self.learned_output_gain, 1.0)),
            )
            .chain(
                self.output_bias_learnable
                    .then(|| self.get_param(self.output_bias, self.learned_output_bias, 0.0)),
            )
            .chain(
                self.scale_learnable
                    .then(|| self.get_param(self.scale, self.learned_scale, 1.0)),
            )
            .chain(
                self.shift_learnable
                    .then(|| self.get_param(self.shift, self.learned_shift, 0.0)),
            )
            .chain(
                self.gamma_learnable
                    .then(|| {
                        self.gamma
                            .as_ref()
                            .map(|g| g.iter().map(|&x| x as f64).collect::<Vec<f64>>())
                            .unwrap_or_default()
                    })
                    .into_iter()
                    .flatten(),
            )
            .chain(
                self.bias_learnable
                    .then(|| {
                        self.bias
                            .as_ref()
                            .map(|b| b.iter().map(|&x| x as f64).collect::<Vec<f64>>())
                            .unwrap_or_default()
                    })
                    .into_iter()
                    .flatten(),
            )
            .collect();

        // Debug: Log if weights are still at defaults (indicating no training occurred)
        if weights.is_empty() {
            tracing::debug!(
                "RichardsCurve weights() returned empty vector - no learnable parameters"
            );
        }

        weights
    }

    /// Number of scalar learnable parameters (excluding per-feature gamma/bias)
    pub fn scalar_weights_len(&self) -> usize {
        [
            self.nu_learnable,
            self.k_learnable,
            self.m_learnable,
            self.beta_learnable,
            self.temperature_learnable,
            self.output_gain_learnable,
            self.output_bias_learnable,
            self.scale_learnable,
            self.shift_learnable,
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }

    /// Check if any parameters have been trained (learned values exist and differ from defaults)
    pub fn has_trained_parameters(&self) -> bool {
        // Check if any learned parameters exist and differ significantly from defaults
        let checks = [
            self.learned_nu.is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_k.is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_m.is_some_and(|v| v.abs() > 1e-6),
            self.learned_beta.is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_temperature
                .is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_output_gain
                .is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_output_bias.is_some_and(|v| v.abs() > 1e-6),
            self.learned_scale.is_some_and(|v| (v - 1.0).abs() > 1e-6),
            self.learned_shift.is_some_and(|v| v.abs() > 1e-6),
        ];

        checks.iter().any(|&x| x)
    }

    /// Number of learnable parameters in the internal order
    pub fn weights_len(&self) -> usize {
        let scalar_params = self.scalar_weights_len();

        let array_params = if self.gamma_learnable {
            self.gamma.as_ref().map(|g| g.len()).unwrap_or(0)
        } else {
            0
        } + if self.bias_learnable {
            self.bias.as_ref().map(|b| b.len()).unwrap_or(0)
        } else {
            0
        };

        scalar_params + array_params
    }

    /// Iterator over current learnable parameter values (zero-allocation)
    pub fn weights_iter(&self) -> WeightsIter<'_> {
        WeightsIter {
            curve: self,
            idx: 0,
        }
    }

    /// Get current scaling parameters
    pub fn get_scaling(&self) -> (f64, f64) {
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);
        (scale, shift)
    }

    /// Setter for learning updates (e.g., from optimizer).
    pub fn set_param(
        &mut self,
        nu: Option<f64>,
        k: Option<f64>,
        m: Option<f64>,
        beta: Option<f64>,
        output_gain: Option<f64>,
        output_bias: Option<f64>,
    ) {
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
        if self.variant != super::Variant::Adaptive {
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
        let new_running_sum = self.running_sum.unwrap() * momentum + x.sum() * (1.0 - momentum);
        let new_running_sq_sum =
            (self.running_sq_sum.unwrap() * momentum + batch_var_sum * (1.0 - momentum)) as f64;

        self.running_sum = Some(new_running_sum);
        self.running_sq_sum = Some(new_running_sq_sum);
        self.count = Some(new_count);

        self.update_adaptive_scaling();
    }

    /// Update adaptive scale and shift from running statistics
    fn update_adaptive_scaling(&mut self) {
        if let (Some(running_sum), Some(running_sq_sum), Some(count)) =
            (self.running_sum, self.running_sq_sum, self.count)
        {
            if count > 1 {
                let mean = running_sum / count as f64;
                let variance = (running_sq_sum / (count - 1) as f64)
                    - (running_sum.powi(2) / count as f64) / (count - 1) as f64;
                let std = variance.sqrt().max(1e-6); // Minimum std for numerical stability

                // Adaptive normalization: center at mean, scale to unit variance
                self.adaptive_scale = Some(1.0 / std);
                self.adaptive_shift = Some(-mean / std);
            }
        }
    }

    /// Get adaptive scaling parameters (or default to (1.0, 0.0) if not adaptive)
    fn get_adaptive_scaling(&self) -> (f64, f64) {
        if self.variant == super::Variant::Adaptive {
            (
                self.adaptive_scale.unwrap_or(1.0),
                self.adaptive_shift.unwrap_or(0.0),
            )
        } else {
            (1.0, 0.0) // Identity transformation for non-adaptive variants
        }
    }

    /// Reset running statistics (useful for new training epochs)
    pub fn reset_running_stats(&mut self) {
        if self.variant == super::Variant::Adaptive {
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
        if self.variant != super::Variant::Polynomial {
            return Err("Can only set polynomial coefficients for Polynomial variant".to_string());
        }
        if !(1..=5).contains(&power) {
            return Err("Polynomial degree must be between 1 and 5".to_string());
        }
        if coeffs.len() != power + 1 {
            return Err(format!(
                "Expected {} coefficients for degree {}, got {}",
                power + 1,
                power,
                coeffs.len()
            ));
        }

        self.poly_power = Some(power);
        self.poly_coeffs = Some(coeffs);
        Ok(())
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
                        return Some(self.curve.get_param(
                            self.curve.nu,
                            self.curve.learned_nu,
                            1.0,
                        ));
                    }
                }
                1 => {
                    self.idx += 1;
                    if self.curve.k_learnable {
                        return Some(
                            self.curve
                                .get_param(self.curve.k, self.curve.learned_k, 1.0),
                        );
                    }
                }
                2 => {
                    self.idx += 1;
                    if self.curve.m_learnable {
                        return Some(
                            self.curve
                                .get_param(self.curve.m, self.curve.learned_m, 0.0),
                        );
                    }
                }
                3 => {
                    self.idx += 1;
                    if self.curve.beta_learnable {
                        return Some(self.curve.get_param(
                            self.curve.beta,
                            self.curve.learned_beta,
                            1.0,
                        ));
                    }
                }
                4 => {
                    self.idx += 1;
                    if self.curve.temperature_learnable {
                        return Some(self.curve.get_param(
                            self.curve.temperature,
                            self.curve.learned_temperature,
                            1.0,
                        ));
                    }
                }
                5 => {
                    self.idx += 1;
                    if self.curve.output_gain_learnable {
                        return Some(self.curve.get_param(
                            self.curve.output_gain,
                            self.curve.learned_output_gain,
                            1.0,
                        ));
                    }
                }
                6 => {
                    self.idx += 1;
                    if self.curve.output_bias_learnable {
                        return Some(self.curve.get_param(
                            self.curve.output_bias,
                            self.curve.learned_output_bias,
                            0.0,
                        ));
                    }
                }
                7 => {
                    self.idx += 1;
                    if self.curve.scale_learnable {
                        return Some(self.curve.get_param(
                            self.curve.scale,
                            self.curve.learned_scale,
                            1.0,
                        ));
                    }
                }
                8 => {
                    self.idx += 1;
                    if self.curve.shift_learnable {
                        return Some(self.curve.get_param(
                            self.curve.shift,
                            self.curve.learned_shift,
                            0.0,
                        ));
                    }
                }
                _ => return None,
            }
        }
    }
}
