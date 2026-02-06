use std::marker::PhantomData;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::infrastructure::optimizer::adam::Adam;

// Shared internal numerics for the richards module.
// Kept non-public to prevent namespace bleeding into the rest of the codebase.

#[inline]
pub(super) fn exp_f64_richards(x: f64) -> f64 {
    crate::domain::pade::exp(x)
}

#[inline]
pub fn exp_f32_richards(x: f32) -> f32 {
    crate::domain::pade::exp(x as f64) as f32
}

#[inline]
pub(super) fn softplus_f64_richards(x: f64) -> f64 {
    crate::domain::soft::softplus(x)
}

#[inline]
pub fn softplus_f32_richards(x: f32) -> f32 {
    crate::domain::soft::softplus(x)
}

#[inline]
pub(super) fn inv_softplus_f64_richards(t: f64) -> f64 {
    if !t.is_finite() {
        return t;
    }
    if t > 20.0 {
        t
    } else {
        (crate::domain::pade::exp(t) - 1.0).ln()
    }
}

#[inline]
pub(super) fn unit_from_softplus_f64_richards(t: f64) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return 1.0;
    }
    if t == f64::NEG_INFINITY {
        return 0.0;
    }
    1.0 - crate::domain::pade::exp(-t)
}

#[inline]
pub(super) fn unit_from_softplus_f32_richards(t: f32) -> f32 {
    if t.is_nan() {
        return f32::NAN;
    }
    if t == f32::INFINITY {
        return 1.0;
    }
    if t == f32::NEG_INFINITY {
        return 0.0;
    }
    1.0 - exp_f32_richards(-t)
}

// Rayon parallelism has overhead for small slices; avoid it on tiny tensors.
const PAR_THRESHOLD: usize = 1024;

// Max number of scalar weights supported by RichardsCurve.
// Order: nu, k, m, beta, temperature, output_gain, output_bias, scale, shift.
const MAX_SCALAR_PARAMS: usize = 9;

// --- Zero-cost (compile-time) variant specialization ---

trait VariantMarker: Sync + Send {
    const INPUT_SCALE: f64;
    const OUTER_SCALE: f64;
    fn gate(sigma: f64) -> f64;
}

trait VariantMarkerF32: Sync + Send {
    const INPUT_SCALE: f32;
    const OUTER_SCALE: f32;
    fn gate(sigma: f32) -> f32;
}

struct SigmoidLike;
struct TanhLike;

impl VariantMarker for SigmoidLike {
    const INPUT_SCALE: f64 = 1.0;
    const OUTER_SCALE: f64 = 1.0;

    #[inline]
    fn gate(sigma: f64) -> f64 {
        sigma
    }
}

impl VariantMarkerF32 for SigmoidLike {
    const INPUT_SCALE: f32 = 1.0;
    const OUTER_SCALE: f32 = 1.0;

    #[inline]
    fn gate(sigma: f32) -> f32 {
        sigma
    }
}

impl VariantMarker for TanhLike {
    const INPUT_SCALE: f64 = 2.0;
    const OUTER_SCALE: f64 = 2.0;

    #[inline]
    fn gate(sigma: f64) -> f64 {
        2.0 * sigma - 1.0
    }
}

impl VariantMarkerF32 for TanhLike {
    const INPUT_SCALE: f32 = 2.0;
    const OUTER_SCALE: f32 = 2.0;

    #[inline]
    fn gate(sigma: f32) -> f32 {
        2.0 * sigma - 1.0
    }
}

#[derive(Clone, Copy)]
struct RichardsKernel<V: VariantMarker> {
    nu_eff: f64,
    k_eff: f64,
    m: f64,
    beta: f64,
    temp_reciprocal: f64,
    output_gain: f64,
    output_bias: f64,
    scale: f64,
    shift: f64,
    adaptive_scale: f64,
    adaptive_shift: f64,
    inv_nu: f64,
    _variant: PhantomData<V>,
}

impl<V: VariantMarker> RichardsKernel<V> {
    #[inline]
    fn from_curve(curve: &RichardsCurve) -> Self {
        let (nu, k, m, beta, temp, output_gain, output_bias, scale, shift) = curve.get_all_params();
        let (adaptive_scale, adaptive_shift) = curve.get_adaptive_scaling();
        // `get_all_params` enforces nu>0, beta>0, temp>0.
        let nu_eff = nu;
        let k_eff = if curve.birch_exponential_tail {
            k * nu_eff
        } else {
            k
        };
        Self {
            nu_eff,
            k_eff,
            m,
            beta,
            temp_reciprocal: 1.0 / temp,
            output_gain,
            output_bias,
            scale,
            shift,
            adaptive_scale,
            adaptive_shift,
            inv_nu: -1.0 / nu,
            _variant: PhantomData,
        }
    }

    #[inline]
    fn forward_one_f64(&self, xi: f64) -> f64 {
        let (sigma, _r, _ln_base, _nu_eff, _dinput_dx) = self.common_terms(xi);
        let gate = V::gate(sigma);
        self.output_gain * gate + self.output_bias
    }

    #[inline]
    fn derivative_one_f64(&self, xi: f64) -> f64 {
        let (sigma, r, _ln_base, nu_eff, dinput_dx) = self.common_terms(xi);
        let dsig_dinput = (sigma * self.k_eff * r) / nu_eff;
        self.output_gain * V::OUTER_SCALE * dsig_dinput * dinput_dx
    }

    #[inline]
    fn eval_one_f64(&self, xi: f64) -> (f64, f64) {
        // Returns: (f(x), df/dx)
        // df/dx = output_gain * gate'(sigma) * dsigma/dinput * dinput/dx
        // where dinput/dx = INPUT_SCALE * scale * adaptive_scale / temp

        let (sigma, r, _ln_base, nu_eff, dinput_dx) = self.common_terms(xi);
        let gate = V::gate(sigma);
        let y = self.output_gain * gate + self.output_bias;

        let dsig_dinput = (sigma * self.k_eff * r) / nu_eff;
        let dy_dx = self.output_gain * V::OUTER_SCALE * dsig_dinput * dinput_dx;
        (y, dy_dx)
    }

    #[inline]
    fn common_terms(&self, xi: f64) -> (f64, f64, f64, f64, f64) {
        // Returns: (sigma, r, ln_base, nu_eff, dinput_dx)
        let adaptive_normalized = self.adaptive_scale * xi + self.adaptive_shift;
        let temp_scaled = adaptive_normalized * self.temp_reciprocal;
        let input = V::INPUT_SCALE * (self.scale * temp_scaled + self.shift);

        let exponent: f64 = -self.k_eff * (input - self.m);

        // base = 1 + beta * exp(exponent)
        // Use log1p-space to avoid overflow for large positive exponent.
        // ln_base = log(base) = softplus(ln(beta) + exponent)
        // r = beta*exp(exponent)/base = sigmoid(ln(beta) + exponent)
        let t = self.beta.ln() + exponent;
        let ln_base = softplus_f64_richards(t);
        let r = unit_from_softplus_f64_richards(ln_base);

        let nu_eff = self.nu_eff;
        let sigma = exp_f64_richards(self.inv_nu * ln_base);
        let dinput_dx = V::INPUT_SCALE * self.scale * self.adaptive_scale * self.temp_reciprocal;
        (sigma, r, ln_base, nu_eff, dinput_dx)
    }
}

#[derive(Clone, Copy)]
struct RichardsKernelF32<V: VariantMarkerF32> {
    nu_eff: f32,
    k_eff: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    inv_nu: f32,
    _variant: PhantomData<V>,
}

impl<V: VariantMarkerF32> RichardsKernelF32<V> {
    #[inline]
    fn from_curve(curve: &RichardsCurve) -> Self {
        Self::from_curve_with_overrides(curve, None, None, None)
    }

    #[inline]
    fn from_curve_with_overrides(
        curve: &RichardsCurve,
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) -> Self {
        let (nu, k, mut m, mut beta, mut temp, output_gain, output_bias, scale, shift) = curve.get_all_params();
        let (adaptive_scale, adaptive_shift) = curve.get_adaptive_scaling();

        if let Some(t) = temp_override {
            temp = t;
            if !temp.is_finite() || temp <= 0.0 {
                temp = RichardsCurve::MIN_POS_PARAM;
            }
        }
        if let Some(mv) = m_override {
            m = mv;
        }
        if let Some(b) = beta_override {
            beta = b;
            if !beta.is_finite() || beta <= 0.0 {
                beta = RichardsCurve::MIN_POS_PARAM;
            }
        }

        let nu_eff = nu as f32;
        let k = k as f32;
        let k_eff = if curve.birch_exponential_tail {
            k * nu_eff
        } else {
            k
        };

        Self {
            nu_eff,
            k_eff,
            m: m as f32,
            beta: beta as f32,
            temp_reciprocal: 1.0f32 / (temp as f32),
            output_gain: output_gain as f32,
            output_bias: output_bias as f32,
            scale: scale as f32,
            shift: shift as f32,
            adaptive_scale: adaptive_scale as f32,
            adaptive_shift: adaptive_shift as f32,
            inv_nu: -(1.0f32 / (nu as f32)),
            _variant: PhantomData,
        }
    }

    #[inline]
    fn forward_one_f32(&self, xi: f32) -> f32 {
        let (sigma, _r, _ln_base, _nu_eff, _dinput_dx) = self.common_terms(xi);
        let gate = V::gate(sigma);
        self.output_gain * gate + self.output_bias
    }

    #[inline]
    fn derivative_one_f32(&self, xi: f32) -> f32 {
        let (sigma, r, _ln_base, nu_eff, dinput_dx) = self.common_terms(xi);
        let dsig_dinput = (sigma * self.k_eff * r) / nu_eff;
        self.output_gain * V::OUTER_SCALE * dsig_dinput * dinput_dx
    }

    #[inline]
    fn eval_one_f32(&self, xi: f32) -> (f32, f32) {
        let (sigma, r, _ln_base, nu_eff, dinput_dx) = self.common_terms(xi);
        let gate = V::gate(sigma);
        let y = self.output_gain * gate + self.output_bias;

        let dsig_dinput = (sigma * self.k_eff * r) / nu_eff;
        let dy_dx = self.output_gain * V::OUTER_SCALE * dsig_dinput * dinput_dx;
        (y, dy_dx)
    }

    #[inline]
    fn common_terms(&self, xi: f32) -> (f32, f32, f32, f32, f32) {
        let adaptive_normalized = self.adaptive_scale * xi + self.adaptive_shift;
        let temp_scaled = adaptive_normalized * self.temp_reciprocal;
        let input = V::INPUT_SCALE * (self.scale * temp_scaled + self.shift);

        let exponent: f32 = -self.k_eff * (input - self.m);

        let t = self.beta.ln() + exponent;
        let ln_base = softplus_f32_richards(t);
        let r = unit_from_softplus_f32_richards(ln_base);

        let nu_eff = self.nu_eff;
        let sigma = exp_f32_richards(self.inv_nu * ln_base);
        let dinput_dx = V::INPUT_SCALE * self.scale * self.adaptive_scale * self.temp_reciprocal;
        (sigma, r, ln_base, nu_eff, dinput_dx)
    }
}

/// # Richards Curve: Mathematical Framework and Numerical Methods
///
/// ## Core Richards Function Theorem
///
/// **Theorem 1 (Richards Curve Family)**: The Richards curve is defined as a parametric
/// family of sigmoid functions with the following mathematical formulation:
///
/// σ(x; ν, k, m) = [1 + e^(-k(x-m))]^(-1/ν)
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
/// σ_β(x; ν, k, m, β) = [1 + β * e^(-k(x-m))]^(-1/ν)
///
/// **Asymmetry Properties:**
/// - β = 1.0: Standard Richards curve σ(x) = [1 + e^(-k(x-m))]^(-1/ν)
/// - 0 < β < 1: Softer sigmoid transitions
/// - β > 1: Sharper sigmoid transitions
///
/// **Implementation Note:** This codebase enforces β > 0 for global numerical stability
/// (see `get_all_params`). Negative or zero β would make `log(1 + β·exp(…))` undefined
/// on parts of ℝ, so it is treated as invalid configuration.
///
/// **Mathematical Interpretation:**
/// The β parameter scales the exponential term, controlling the steepness and asymmetry
/// of the sigmoid transition without degenerating into a constant.
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
    #[serde(default)]
    pub temperature: Option<f64>, // Temperature scaling factor

    // Affine parameter values (Some for fixed, None for learnable)
    #[serde(rename = "a")]
    pub output_gain: Option<f64>, // Affine output gain (scale)
    #[serde(rename = "b")]
    pub output_bias: Option<f64>, // Affine output bias (shift)

    // Input scaling parameter values (Some for fixed, None for learnable)
    pub scale: Option<f64>, // Input scaling
    pub shift: Option<f64>, // Input shift

    /// Birch-inspired exponential-tail mode.
    ///
    /// When enabled, the exponent uses an effective growth rate `k_eff = k * nu`.
    /// This keeps the left-tail exponential rate in input-space approximately
    /// independent of `nu` (i.e. $\sigma(x) \approx C\,e^{k x}$ as $x\to-\infty$),
    /// while still allowing `nu` to shape the overall sigmoid asymmetry.
    #[serde(default)]
    pub birch_exponential_tail: bool,

    // Per-feature output transformation (used by normalization variants)
    //
    // These are learnable parameters (RichardsNorm uses them), so they must be persisted.
    // `default` keeps backward compatibility with older checkpoints where these fields
    // were absent.
    #[serde(default)]
    pub gamma: Option<Arc<Array2<f32>>>, // Per-feature scale (shape: [1, d])
    #[serde(default)]
    pub bias: Option<Arc<Array2<f32>>>, // Per-feature bias (shape: [1, d])

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
    #[serde(default)]
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
    #[serde(default)]
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
    pub variant: crate::domain::richards::Variant, // Sigmoid, Tanh, or Gompertz mode

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

#[allow(dead_code)]
impl RichardsCurve {
    const MIN_POS_PARAM: f64 = 1e-6;

    // NOTE: internal numerics are Pad 0e9-backed and kept private to this module.

    /// Enable/disable Birch-inspired exponential-tail behavior.
    pub fn set_birch_exponential_tail(&mut self, enabled: bool) {
        self.birch_exponential_tail = enabled;
    }

    /// Builder-style toggle for Birch-inspired exponential-tail behavior.
    pub fn with_birch_exponential_tail(mut self, enabled: bool) -> Self {
        self.birch_exponential_tail = enabled;
        self
    }

    #[inline]
    fn eval_kernel_into_f64<V: VariantMarker>(&self, x: &[f64], y: &mut [f64], dy: &mut [f64]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), dy.len());
        let k = RichardsKernel::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for i in 0..x.len() {
                let (yi, dyi) = k.eval_one_f64(x[i]);
                y[i] = yi;
                dy[i] = dyi;
            }
        } else {
            y.par_iter_mut()
                .zip(dy.par_iter_mut())
                .zip(x.par_iter())
                .for_each(|((yo, dyo), &xi)| {
                    let (yi, dyi) = k.eval_one_f64(xi);
                    *yo = yi;
                    *dyo = dyi;
                });
        }
    }

    #[inline]
    fn eval_kernel_into_f32<V: VariantMarkerF32>(&self, x: &[f32], y: &mut [f32], dy: &mut [f32]) {
        debug_assert_eq!(x.len(), y.len());
        debug_assert_eq!(x.len(), dy.len());
        let k = RichardsKernelF32::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for i in 0..x.len() {
                let (yi, dyi) = k.eval_one_f32(x[i]);
                y[i] = yi;
                dy[i] = dyi;
            }
        } else {
            y.par_iter_mut()
                .zip(dy.par_iter_mut())
                .zip(x.par_iter())
                .for_each(|((yo, dyo), &xi)| {
                    let (yi, dyi) = k.eval_one_f32(xi);
                    *yo = yi;
                    *dyo = dyi;
                });
        }
    }

    /// Fused evaluation: computes both forward and derivative into caller-provided buffers.
    pub fn eval_into_f32(&self, x: &[f32], y: &mut [f32], dy: &mut [f32]) {
        assert_eq!(x.len(), y.len(), "Input and output lengths must match");
        assert_eq!(x.len(), dy.len(), "Input and derivative lengths must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.eval_kernel_into_f32::<TanhLike>(x, y, dy),
            _ => self.eval_kernel_into_f32::<SigmoidLike>(x, y, dy),
        }
    }

    /// Fused evaluation for scalars: returns (f(x), df/dx).
    #[inline]
    pub fn eval_scalar(&self, x: f64) -> (f64, f64) {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                RichardsKernel::<TanhLike>::from_curve(self).eval_one_f64(x)
            }
            _ => RichardsKernel::<SigmoidLike>::from_curve(self).eval_one_f64(x),
        }
    }

    #[inline]
    fn forward_kernel_into_f64<V: VariantMarker>(&self, x: &[f64], out: &mut [f64]) {
        let k = RichardsKernel::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for (xi, o) in x.iter().copied().zip(out.iter_mut()) {
                *o = k.forward_one_f64(xi);
            }
        } else {
            x.par_iter().zip(out.par_iter_mut()).for_each(|(&xi, o)| {
                *o = k.forward_one_f64(xi);
            });
        }
    }

    /// Fused evaluation (f64 slices): computes both forward and derivative into caller-provided
    /// buffers.
    pub fn eval_into(&self, x: &[f64], y: &mut [f64], dy: &mut [f64]) {
        assert_eq!(x.len(), y.len(), "Input and output lengths must match");
        assert_eq!(x.len(), dy.len(), "Input and derivative lengths must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.eval_kernel_into_f64::<TanhLike>(x, y, dy),
            _ => self.eval_kernel_into_f64::<SigmoidLike>(x, y, dy),
        }
    }

    #[inline]
    fn forward_kernel_into_f32<V: VariantMarkerF32>(&self, x: &[f32], out: &mut [f32]) {
        let k = RichardsKernelF32::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for (xi, o) in x.iter().copied().zip(out.iter_mut()) {
                *o = k.forward_one_f32(xi);
            }
        } else {
            x.par_iter().zip(out.par_iter_mut()).for_each(|(&xi, o)| {
                *o = k.forward_one_f32(xi);
            });
        }
    }

    #[inline]
    fn derivative_kernel_into_f64<V: VariantMarker>(&self, x: &[f64], out: &mut [f64]) {
        let k = RichardsKernel::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for (xi, o) in x.iter().copied().zip(out.iter_mut()) {
                *o = k.derivative_one_f64(xi);
            }
        } else {
            x.par_iter().zip(out.par_iter_mut()).for_each(|(&xi, o)| {
                *o = k.derivative_one_f64(xi);
            });
        }
    }

    #[inline]
    fn derivative_kernel_into_f32<V: VariantMarkerF32>(&self, x: &[f32], out: &mut [f32]) {
        let k = RichardsKernelF32::<V>::from_curve(self);
        if x.len() < PAR_THRESHOLD {
            for (xi, o) in x.iter().copied().zip(out.iter_mut()) {
                *o = k.derivative_one_f32(xi);
            }
        } else {
            x.par_iter().zip(out.par_iter_mut()).for_each(|(&xi, o)| {
                *o = k.derivative_one_f32(xi);
            });
        }
    }

    /// Constructor with learnable params based on variant.
    pub fn new_learnable(variant: crate::domain::richards::Variant) -> Self {
        // Set output_gain/output_bias coefficients based on variant (Some for fixed, None for
        // learnable)
        let (output_gain_val, output_bias_val) = match variant {
            crate::domain::richards::Variant::Sigmoid | crate::domain::richards::Variant::Gompertz => {
                (Some(1.0), Some(0.0))
            } // [0, 1] range, fixed
            crate::domain::richards::Variant::Tanh => (Some(1.0), Some(0.0)), /* [-1, 1] via 2σ(2x) - 1
                                                                        * transform, */
            // fixed
            crate::domain::richards::Variant::Adaptive
            | crate::domain::richards::Variant::None
            | crate::domain::richards::Variant::Polynomial => (None, None), /* Fully learnable including
                                                                     * output_gain/output_bias */
        };

        // Determine parameter count based on whether output_gain/output_bias are learnable
        // nu, k, m, beta, temp, scale, shift + optionally output_gain, output_bias
        let param_count = 7
            + if output_gain_val.is_none() { 1 } else { 0 }
            + if output_bias_val.is_none() { 1 } else { 0 };

        let (adaptive_initialized, momentum) = match variant {
            crate::domain::richards::Variant::Adaptive => (true, 0.01), /* Enable adaptive normalization
                                                                  * with */
            // default momentum
            _ => (false, 0.0), // Disable adaptive for other variants
        };

        let polynomial_initialized = match variant {
            crate::domain::richards::Variant::Polynomial => true, // Enable polynomial transformation
            _ => false,                                   // Disable polynomial for other variants
        };

        // Enable Birch-inspired exponential tail by default for sigmoid-like generalized logistic
        // usage (helps ensure exponential behavior in the left tail across nu values).
        // Keep it disabled for Tanh, where the notion of “small size” growth is less aligned.
        let birch_exponential_tail = !matches!(variant, crate::domain::richards::Variant::Tanh);

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

            birch_exponential_tail,

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

            birch_exponential_tail: true,
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
            variant: crate::domain::richards::Variant::Sigmoid,
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
            Self::new_learnable(crate::domain::richards::Variant::Sigmoid)
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

                birch_exponential_tail: true,
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
                variant: crate::domain::richards::Variant::Sigmoid,
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
            Self::new_learnable(crate::domain::richards::Variant::Tanh)
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

                birch_exponential_tail: false,
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
                variant: crate::domain::richards::Variant::Tanh,
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
            Self::new_learnable(crate::domain::richards::Variant::Gompertz)
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

                birch_exponential_tail: true,
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
                variant: crate::domain::richards::Variant::Gompertz,
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
        Self::new_learnable(crate::domain::richards::Variant::None)
    }

    /// Enable per-feature transformations for normalization layers
    /// Sets up learnable gamma (scale) and bias parameters for each feature dimension
    pub fn enable_per_feature_transform(&mut self, embedding_dim: usize) {
        // Initialize gamma and bias arrays if not already present
        if self.gamma.is_none() {
            self.gamma = Some(Arc::new(Array2::ones((1, embedding_dim))));
        }
        if self.bias.is_none() {
            self.bias = Some(Arc::new(Array2::zeros((1, embedding_dim))));
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
            let (scale, shift) = if max_abs_x > 0.0 {
                (Some((1.0 / max_abs_x).min(0.5)), Some(0.0))
            } else {
                (Some(1.0), Some(0.0))
            };

            // Lightweight clone: Copy scalars, skip heavy heap allocations (optimizer, history, etc.)
            // This is safe because the returned instance is only used for temporary scalar evaluation
            // in MoHGating, not for training or matrix operations that would need gamma/bias.
            Self {
                nu: self.nu,
                k: self.k,
                m: self.m,
                beta: self.beta,
                temperature: self.temperature,
                output_gain: self.output_gain,
                output_bias: self.output_bias,
                scale, // Updated
                shift, // Updated
                birch_exponential_tail: self.birch_exponential_tail,
                gamma: None,      // Heavy, unused for scalar forward
                bias: None,       // Heavy, unused for scalar forward
                poly_power: self.poly_power,
                poly_coeffs: None, // Heavy, unused for scalar forward (mostly)
                learned_nu: self.learned_nu,
                learned_k: self.learned_k,
                learned_m: self.learned_m,
                learned_beta: self.learned_beta,
                learned_temperature: self.learned_temperature,
                learned_output_gain: self.learned_output_gain,
                learned_output_bias: self.learned_output_bias,
                learned_scale: self.learned_scale,
                learned_shift: self.learned_shift,
                nu_learnable: self.nu_learnable,
                k_learnable: self.k_learnable,
                m_learnable: self.m_learnable,
                beta_learnable: self.beta_learnable,
                temperature_learnable: self.temperature_learnable,
                output_gain_learnable: self.output_gain_learnable,
                output_bias_learnable: self.output_bias_learnable,
                scale_learnable: self.scale_learnable,
                shift_learnable: self.shift_learnable,
                gamma_learnable: self.gamma_learnable,
                bias_learnable: self.bias_learnable,
                variant: self.variant,
                running_sum: None,    // Unused for scalar forward
                running_sq_sum: None, // Unused for scalar forward
                count: None,          // Unused for scalar forward
                momentum: self.momentum,
                adaptive_scale: self.adaptive_scale,
                adaptive_shift: self.adaptive_shift,
                optimizer: None, // Heavy
                l2_reg: self.l2_reg,
                adaptive_lr_scale: self.adaptive_lr_scale,
                grad_norm_history: Vec::new(), // Heavy
            }
        } else {
            self.clone()
        }
    }

    /// In-place version of `update_scaling_from_max_abs`.
    /// Only updates if scale/shift are fixed (`Some`) so it won't override learnable params.
    pub fn update_scaling_from_max_abs_inplace(&mut self, max_abs_x: f64) {
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
        if let Some(param) = param {
            param
        } else {
            learned.unwrap_or(default)
        }
    }

    /// Helper: get all parameters at once to reduce redundancy.
    fn get_all_params(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        let mut nu = self.get_param(self.nu, self.learned_nu, 1.0);
        let mut k = self.get_param(self.k, self.learned_k, 1.0);
        let m = self.get_param(self.m, self.learned_m, 0.0);
        let mut beta = self.get_param(self.beta, self.learned_beta, 1.0);
        let mut temp = self.get_param(self.temperature, self.learned_temperature, 1.0);
        let output_gain = self.get_param(self.output_gain, self.learned_output_gain, 1.0);
        let output_bias = self.get_param(self.output_bias, self.learned_output_bias, 0.0);
        let scale = self.get_param(self.scale, self.learned_scale, 1.0);
        let shift = self.get_param(self.shift, self.learned_shift, 0.0);

        // --- SOTA safety constraints ---
        // These keep the generalized logistic family well-defined for all call sites.
        // Learnable paths already enforce positivity via softplus, but fixed values in
        // configs/checkpoints can still be invalid.
        if !nu.is_finite() || nu <= 0.0 {
            nu = Self::MIN_POS_PARAM;
        }
        if !k.is_finite() || k == 0.0 {
            // Preserve sign if caller provided it; otherwise default positive.
            k = Self::MIN_POS_PARAM.copysign(if k == 0.0 { 1.0 } else { k });
        }
        if !beta.is_finite() || beta <= 0.0 {
            beta = Self::MIN_POS_PARAM;
        }
        if !temp.is_finite() || temp <= 0.0 {
            temp = Self::MIN_POS_PARAM;
        }

        (nu, k, m, beta, temp, output_gain, output_bias, scale, shift)
    }

    /// Returns the effective (clamped) parameter tuple used for forward/derivative.
    ///
    /// This is the safest way for other modules to read the “current” parameters because it:
    /// - prefers fixed params (`Some`) over learned params (`learned_*`)
    /// - applies the same positivity / finiteness constraints as the compute kernels
    #[inline]
    pub fn effective_params(&self) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
        self.get_all_params()
    }

    #[inline]
    pub fn effective_nu(&self) -> f64 {
        let (nu, _, _, _, _, _, _, _, _) = self.get_all_params();
        nu
    }

    #[inline]
    pub fn effective_k(&self) -> f64 {
        let (_, k, _, _, _, _, _, _, _) = self.get_all_params();
        k
    }

    #[inline]
    pub fn effective_m(&self) -> f64 {
        let (_, _, m, _, _, _, _, _, _) = self.get_all_params();
        m
    }

    #[inline]
    pub fn effective_beta(&self) -> f64 {
        let (_, _, _, beta, _, _, _, _, _) = self.get_all_params();
        beta
    }

    #[inline]
    pub fn effective_temperature(&self) -> f64 {
        let (_, _, _, _, temp, _, _, _, _) = self.get_all_params();
        temp
    }

    #[inline]
    pub fn effective_output_gain(&self) -> f64 {
        let (_, _, _, _, _, a, _, _, _) = self.get_all_params();
        a
    }

    #[inline]
    pub fn effective_output_bias(&self) -> f64 {
        let (_, _, _, _, _, _, b, _, _) = self.get_all_params();
        b
    }

    #[inline]
    pub fn effective_scale(&self) -> f64 {
        let (_, _, _, _, _, _, _, s, _) = self.get_all_params();
        s
    }

    #[inline]
    pub fn effective_shift(&self) -> f64 {
        let (_, _, _, _, _, _, _, _, sh) = self.get_all_params();
        sh
    }

    /// Effective input multiplier combining `scale` and `temperature`.
    ///
    /// In the current parameterization, the pre-activation uses `scale * (x / temperature)`.
    /// This means `scale` and `temperature` are partially non-identifiable if both are learnable;
    /// most call sites should prefer learning only one of them.
    #[inline]
    pub fn effective_scale_over_temperature(&self) -> f64 {
        let (_, _, _, _, temp, _, _, scale, _) = self.get_all_params();
        scale / temp
    }

    /// Vectorized forward pass: f(x) = output_gain * gate(x) + output_bias (elementwise), writing
    /// to output slice. Optimized for zero-copy usage. Uses extended Richards with beta and
    /// temperature parameters.
    pub fn forward_into(&self, x: &[f64], out: &mut [f64]) {
        // Ensure output size matches input
        assert_eq!(x.len(), out.len(), "Input and output lengths must match");

        match self.variant {
            crate::domain::richards::Variant::Tanh => self.forward_kernel_into_f64::<TanhLike>(x, out),
            _ => self.forward_kernel_into_f64::<SigmoidLike>(x, out),
        }
    }

    /// f32-friendly forward pass: computes in f64 internally and writes f32 output.
    #[inline]
    pub fn forward_into_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), out.len(), "Input and output lengths must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.forward_kernel_into_f32::<TanhLike>(x, out),
            _ => self.forward_kernel_into_f32::<SigmoidLike>(x, out),
        }
    }

    pub fn forward_into_f32_with_overrides(
        &self,
        x: &[f32],
        out: &mut [f32],
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) {
        assert_eq!(x.len(), out.len(), "Input and output lengths must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.forward_kernel_with_overrides_into_f32::<TanhLike>(
                x,
                out,
                temp_override,
                m_override,
                beta_override,
            ),
            _ => self.forward_kernel_with_overrides_into_f32::<SigmoidLike>(
                x,
                out,
                temp_override,
                m_override,
                beta_override,
            ),
        }
    }

    fn forward_kernel_with_overrides_into_f32<V: VariantMarkerF32>(
        &self,
        x: &[f32],
        out: &mut [f32],
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) {
        let k = RichardsKernelF32::<V>::from_curve_with_overrides(
            self,
            temp_override,
            m_override,
            beta_override,
        );

        match (&self.gamma, &self.bias) {
            (Some(gamma), Some(bias)) => {
                let g_slice = gamma.as_slice().unwrap();
                let b_slice = bias.as_slice().unwrap();
                if x.len() < PAR_THRESHOLD {
                    for i in 0..x.len() {
                        out[i] = k.forward_one_f32(x[i]) * g_slice[i] + b_slice[i];
                    }
                } else {
                    out.par_iter_mut()
                        .zip(x.par_iter())
                        .zip(g_slice.par_iter())
                        .zip(b_slice.par_iter())
                        .for_each(|(((o, &xi), &g), &b)| {
                            *o = k.forward_one_f32(xi) * g + b;
                        });
                }
            }
            (Some(gamma), None) => {
                let g_slice = gamma.as_slice().unwrap();
                if x.len() < PAR_THRESHOLD {
                    for i in 0..x.len() {
                        out[i] = k.forward_one_f32(x[i]) * g_slice[i];
                    }
                } else {
                    out.par_iter_mut()
                        .zip(x.par_iter())
                        .zip(g_slice.par_iter())
                        .for_each(|((o, &xi), &g)| {
                            *o = k.forward_one_f32(xi) * g;
                        });
                }
            }
            (None, Some(bias)) => {
                let b_slice = bias.as_slice().unwrap();
                if x.len() < PAR_THRESHOLD {
                    for i in 0..x.len() {
                        out[i] = k.forward_one_f32(x[i]) + b_slice[i];
                    }
                } else {
                    out.par_iter_mut()
                        .zip(x.par_iter())
                        .zip(b_slice.par_iter())
                        .for_each(|((o, &xi), &b)| {
                            *o = k.forward_one_f32(xi) + b;
                        });
                }
            }
            (None, None) => {
                if x.len() < PAR_THRESHOLD {
                    for i in 0..x.len() {
                        out[i] = k.forward_one_f32(x[i]);
                    }
                } else {
                    out.par_iter_mut()
                        .zip(x.par_iter())
                        .for_each(|(o, &xi)| {
                            *o = k.forward_one_f32(xi);
                        });
                }
            }
        }
    }

    /// Vectorized forward pass: f(x) = output_gain * gate(x) + output_bias (elementwise),
    /// single-pass.
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(x.len());
        self.forward_into(x.as_slice().unwrap(), out.as_slice_mut().unwrap());
        out
    }

    /// Vectorized forward pass for matrix input, writing to output array
    pub fn forward_matrix_into(&self, x: &Array2<f64>, out: &mut Array2<f64>) {
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.forward_matrix_kernel_into::<TanhLike>(x, out),
            _ => self.forward_matrix_kernel_into::<SigmoidLike>(x, out),
        }
    }

    fn forward_matrix_kernel_into<V: VariantMarker>(&self, x: &Array2<f64>, out: &mut Array2<f64>) {
        let k = RichardsKernel::<V>::from_curve(self);

        match (&self.gamma, &self.bias) {
            (Some(gamma), Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &g, &b| {
                        *o = k.forward_one_f64(xi) * (g as f64) + (b as f64);
                    });
            }
            (Some(gamma), None) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .par_for_each(|o, &xi, &g| {
                        *o = k.forward_one_f64(xi) * (g as f64);
                    });
            }
            (None, Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &b| {
                        *o = k.forward_one_f64(xi) + (b as f64);
                    });
            }
            (None, None) => {
                if let (Some(x_slice), Some(out_slice)) = (x.as_slice(), out.as_slice_mut()) {
                    self.forward_kernel_into_f64::<V>(x_slice, out_slice);
                } else {
                    ndarray::Zip::from(out).and(x).par_for_each(|o, &xi| {
                        *o = k.forward_one_f64(xi);
                    });
                }
            }
        }
    }

    /// f32-friendly forward for matrices (avoids f64 materialization of input/output).
    /// Computes elementwise Richards into `out`, then applies per-feature gamma/bias if enabled.
    pub fn forward_matrix_f32_into(&self, x: &Array2<f32>, out: &mut Array2<f32>) {
        assert_eq!(x.dim(), out.dim(), "Input/output dims must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                self.forward_matrix_kernel_into_f32::<TanhLike>(x, out)
            }
            _ => self.forward_matrix_kernel_into_f32::<SigmoidLike>(x, out),
        }
    }

    pub fn forward_matrix_f32_with_overrides_into(
        &self,
        x: &Array2<f32>,
        out: &mut Array2<f32>,
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) {
        assert_eq!(x.dim(), out.dim(), "Input/output dims must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.forward_matrix_kernel_with_overrides_into::<TanhLike>(
                x,
                out,
                temp_override,
                m_override,
                beta_override,
            ),
            _ => self.forward_matrix_kernel_with_overrides_into::<SigmoidLike>(
                x,
                out,
                temp_override,
                m_override,
                beta_override,
            ),
        }
    }

    fn forward_matrix_kernel_with_overrides_into<V: VariantMarkerF32>(
        &self,
        x: &Array2<f32>,
        out: &mut Array2<f32>,
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) {
        let k = RichardsKernelF32::<V>::from_curve_with_overrides(
            self,
            temp_override,
            m_override,
            beta_override,
        );

        match (&self.gamma, &self.bias) {
            (Some(gamma), Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &g, &b| {
                        *o = k.forward_one_f32(xi) * g + b;
                    });
            }
            (Some(gamma), None) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .par_for_each(|o, &xi, &g| {
                        *o = k.forward_one_f32(xi) * g;
                    });
            }
            (None, Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &b| {
                        *o = k.forward_one_f32(xi) + b;
                    });
            }
            (None, None) => {
                if let (Some(x_slice), Some(out_slice)) = (x.as_slice(), out.as_slice_mut()) {
                    // Manual parallel iteration if slices are available
                    if x_slice.len() < PAR_THRESHOLD {
                        for i in 0..x_slice.len() {
                            out_slice[i] = k.forward_one_f32(x_slice[i]);
                        }
                    } else {
                        out_slice
                            .par_iter_mut()
                            .zip(x_slice.par_iter())
                            .for_each(|(o, &xi)| {
                                *o = k.forward_one_f32(xi);
                            });
                    }
                } else {
                    ndarray::Zip::from(out).and(x).par_for_each(|o, &xi| {
                        *o = k.forward_one_f32(xi);
                    });
                }
            }
        }
    }

    fn forward_matrix_kernel_into_f32<V: VariantMarkerF32>(
        &self,
        x: &Array2<f32>,
        out: &mut Array2<f32>,
    ) {
        let k = RichardsKernelF32::<V>::from_curve(self);

        match (&self.gamma, &self.bias) {
            (Some(gamma), Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &g, &b| {
                        *o = k.forward_one_f32(xi) * (g as f32) + (b as f32);
                    });
            }
            (Some(gamma), None) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(gamma.as_ref())
                    .par_for_each(|o, &xi, &g| {
                        *o = k.forward_one_f32(xi) * (g as f32);
                    });
            }
            (None, Some(bias)) => {
                ndarray::Zip::from(out)
                    .and(x)
                    .and_broadcast(bias.as_ref())
                    .par_for_each(|o, &xi, &b| {
                        *o = k.forward_one_f32(xi) + (b as f32);
                    });
            }
            (None, None) => {
                if let (Some(x_slice), Some(out_slice)) = (x.as_slice(), out.as_slice_mut()) {
                    self.forward_kernel_into_f32::<V>(x_slice, out_slice);
                } else {
                    ndarray::Zip::from(out).and(x).par_for_each(|o, &xi| {
                        *o = k.forward_one_f32(xi);
                    });
                }
            }
        }
    }

    /// Vectorized forward pass for matrix input
    pub fn forward_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(x.dim());
        self.forward_matrix_into(x, &mut output);
        output
    }

    /// Forward for a single scalar x
    pub fn forward_scalar(&self, x: f64) -> f64 {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                RichardsKernel::<TanhLike>::from_curve(self).forward_one_f64(x)
            }
            _ => RichardsKernel::<SigmoidLike>::from_curve(self).forward_one_f64(x),
        }
    }

    /// Allocation-free scalar forward for f32 inputs (avoids f32->f64 conversion).
    #[inline]
    pub fn forward_scalar_f32(&self, x: f32) -> f32 {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                RichardsKernelF32::<TanhLike>::from_curve(self).forward_one_f32(x)
            }
            _ => RichardsKernelF32::<SigmoidLike>::from_curve(self).forward_one_f32(x),
        }
    }

    /// Matrix backward pass: df/dx for matrix input with per-feature transformations
    /// Writes into `grad_input` to avoid allocation.
    pub fn backward_matrix_into(
        &self,
        x: &Array2<f64>,
        output_grads: &Array2<f64>,
        grad_input: &mut Array2<f64>,
    ) {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                self.backward_matrix_kernel_into::<TanhLike>(x, output_grads, grad_input)
            }
            _ => self.backward_matrix_kernel_into::<SigmoidLike>(x, output_grads, grad_input),
        }
    }

    fn backward_matrix_kernel_into<V: VariantMarker>(
        &self,
        x: &Array2<f64>,
        output_grads: &Array2<f64>,
        grad_input: &mut Array2<f64>,
    ) {
        let k = RichardsKernel::<V>::from_curve(self);
        if let Some(gamma) = &self.gamma {
            ndarray::Zip::from(grad_input)
                .and(x)
                .and(output_grads)
                .and_broadcast(gamma.as_ref())
                .par_for_each(|gi, &xi, &dy, &g| {
                    *gi = k.derivative_one_f64(xi) * dy * (g as f64);
                });
        } else {
            ndarray::Zip::from(grad_input)
                .and(x)
                .and(output_grads)
                .par_for_each(|gi, &xi, &dy| {
                    *gi = k.derivative_one_f64(xi) * dy;
                });
        }
    }

    /// Matrix backward pass: df/dx for matrix input with per-feature transformations
    pub fn backward_matrix(&self, x: &Array2<f64>, output_grads: &Array2<f64>) -> Array2<f64> {
        let mut grad_input = Array2::<f64>::zeros(x.raw_dim());
        self.backward_matrix_into(x, output_grads, &mut grad_input);
        grad_input
    }

    /// Matrix gradient computation for all learnable parameters
    /// Optimized with parallel reduction to avoid O(N*D) sequential accumulation
    pub fn grad_weights_matrix(&self, x: &Array2<f64>, output_grads: &Array2<f64>) -> Vec<f64> {
        let (batch_size, embedding_dim) = x.dim();

        // Bounds checking: ensure dimensions are compatible
        if x.dim() != output_grads.dim() {
            return vec![0.0f64; self.weights_len()];
        }

        let scalar_param_count = self.scalar_weights_len();
        let total_elements = (batch_size * embedding_dim) as f64;

        debug_assert!(scalar_param_count <= MAX_SCALAR_PARAMS);

        // Pre-fetch gamma row if present
        let gamma_row = self.gamma.as_ref().map(|g| g.row(0));
        let needs_gamma_grad = self.gamma_learnable && self.gamma.is_some();
        let needs_bias_grad = self.bias_learnable && self.bias.is_some();

        // Initial accumulators
        let init_scalar = vec![0.0f64; scalar_param_count];
        let init_gamma = if needs_gamma_grad { Some(vec![0.0f64; embedding_dim]) } else { None };
        let init_bias = if needs_bias_grad { Some(vec![0.0f64; embedding_dim]) } else { None };

        let (mut scalar_acc, gamma_acc, bias_acc) = if let (Some(x_slice), Some(grad_slice)) =
            (x.as_slice(), output_grads.as_slice())
        {
            let gamma_slice = gamma_row.as_ref().map(|r| r.as_slice().unwrap());

            x_slice
                .par_chunks_exact(embedding_dim)
                .zip(grad_slice.par_chunks_exact(embedding_dim))
                .fold(
                    || (init_scalar.clone(), init_gamma.clone(), init_bias.clone()),
                    |mut acc, (x_row, grad_row)| {
                        let mut buf = [0.0f64; MAX_SCALAR_PARAMS];
                        let (s_acc, g_acc, b_acc) = &mut acc;

                        for j in 0..embedding_dim {
                            let dy = grad_row[j];
                            let eff_dy = if let Some(g) = gamma_slice {
                                dy * (g[j] as f64)
                            } else {
                                dy
                            };

                            let forward_val = self.grad_weights_scalar_into(
                                x_row[j],
                                eff_dy,
                                &mut buf[..scalar_param_count],
                            );

                            for i in 0..scalar_param_count {
                                s_acc[i] += buf[i];
                            }

                            if let Some(ga) = g_acc {
                                ga[j] += forward_val * dy;
                            }
                            if let Some(ba) = b_acc {
                                ba[j] += dy;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || (init_scalar.clone(), init_gamma.clone(), init_bias.clone()),
                    |mut a, b| {
                        for (dst, src) in a.0.iter_mut().zip(b.0.iter()) {
                            *dst += src;
                        }
                        if let (Some(ga), Some(gb)) = (&mut a.1, &b.1) {
                            for (dst, src) in ga.iter_mut().zip(gb.iter()) {
                                *dst += src;
                            }
                        }
                        if let (Some(ba), Some(bb)) = (&mut a.2, &b.2) {
                            for (dst, src) in ba.iter_mut().zip(bb.iter()) {
                                *dst += src;
                            }
                        }
                        a
                    },
                )
        } else {
            // Fallback for non-contiguous arrays
            x.outer_iter().zip(output_grads.outer_iter()).fold(
                (init_scalar.clone(), init_gamma.clone(), init_bias.clone()),
                |mut acc, (x_row, grad_row)| {
                    let mut buf = [0.0f64; MAX_SCALAR_PARAMS];
                    let (s_acc, g_acc, b_acc) = &mut acc;

                    for j in 0..embedding_dim {
                        let dy = grad_row[j];
                        let eff_dy = if let Some(g) = &gamma_row {
                            dy * (g[j] as f64)
                        } else {
                            dy
                        };

                        let forward_val = self.grad_weights_scalar_into(
                            x_row[j],
                            eff_dy,
                            &mut buf[..scalar_param_count],
                        );

                        for i in 0..scalar_param_count {
                            s_acc[i] += buf[i];
                        }

                        if let Some(ga) = g_acc {
                            ga[j] += forward_val * dy;
                        }
                        if let Some(ba) = b_acc {
                            ba[j] += dy;
                        }
                    }
                    acc
                },
            )
        };

        // Average scalar parameters across batch and features
        for g in scalar_acc.iter_mut() {
            *g /= total_elements;
            if !g.is_finite() {
                *g = 0.0;
            }
        }

        let mut final_grads = scalar_acc;

        // Append gamma/bias gradients (averaged over batch)
        if let Some(mut ga) = gamma_acc {
            let scale = 1.0 / (batch_size as f64);
            for g in ga.iter_mut() {
                *g *= scale;
            }
            final_grads.extend(ga);
        }
        if let Some(mut ba) = bias_acc {
            let scale = 1.0 / (batch_size as f64);
            for b in ba.iter_mut() {
                *b *= scale;
            }
            final_grads.extend(ba);
        }

        final_grads
    }

    /// Matrix backward pass for f32 inputs without materializing f64 matrices.
    /// Writes df/dx * dy into `grad_input`.
    pub fn backward_matrix_f32_into(
        &self,
        x: &Array2<f32>,
        output_grads: &Array2<f32>,
        grad_input: &mut Array2<f32>,
    ) {
        if x.dim() != output_grads.dim() || x.dim() != grad_input.dim() {
            grad_input.fill(0.0);
            return;
        }

        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                self.backward_matrix_kernel_into_f32::<TanhLike>(x, output_grads, grad_input)
            }
            _ => {
                self.backward_matrix_kernel_into_f32::<SigmoidLike>(x, output_grads, grad_input)
            }
        }
    }

    fn backward_matrix_kernel_into_f32<V: VariantMarkerF32>(
        &self,
        x: &Array2<f32>,
        output_grads: &Array2<f32>,
        grad_input: &mut Array2<f32>,
    ) {
        let k = RichardsKernelF32::<V>::from_curve(self);

        if let Some(gamma) = &self.gamma {
            ndarray::Zip::from(grad_input)
                .and(x)
                .and(output_grads)
                .and_broadcast(gamma.as_ref())
                .par_for_each(|gi, &xi, &dy, &g| {
                    *gi = k.derivative_one_f32(xi) * dy * (g as f32);
                });
        } else {
            ndarray::Zip::from(grad_input)
                .and(x)
                .and(output_grads)
                .par_for_each(|gi, &xi, &dy| {
                    *gi = k.derivative_one_f32(xi) * dy;
                });
        }
    }

    /// Matrix gradient computation for all learnable parameters from f32 inputs.
    /// Avoids allocating intermediate f64 matrices by iterating and casting per element.
    /// Matrix gradient computation for all learnable parameters from f32 inputs.
    /// Avoids allocating intermediate f64 matrices by iterating and casting per element.
    /// Fused single-pass implementation for cache efficiency.
    pub fn grad_weights_matrix_f32(&self, x: &Array2<f32>, output_grads: &Array2<f32>) -> Vec<f64> {
        let (batch_size, embedding_dim) = x.dim();

        if x.dim() != output_grads.dim() {
            return vec![0.0f64; self.weights_len()];
        }

        let scalar_param_count = self.scalar_weights_len();
        let total_elements = (batch_size * embedding_dim) as f64;
        let batch_denom = batch_size as f32;

        debug_assert!(scalar_param_count <= MAX_SCALAR_PARAMS);
        
        let gamma_row = if let Some(gamma) = &self.gamma {
            Some(gamma.row(0))
        } else {
            None
        };
        
        // Accumulator: (scalar_grads, gamma_grads, bias_grads)
        // We use Option for gamma/bias buffers to avoid allocation if not needed
        let init_scalar = vec![0.0f32; scalar_param_count];
        let init_gamma = if self.gamma_learnable { Some(vec![0.0f32; embedding_dim]) } else { None };
        let init_bias = if self.bias_learnable { Some(vec![0.0f32; embedding_dim]) } else { None };

        let (scalar_acc, gamma_acc, bias_acc) = if let (Some(x_slice), Some(grad_slice)) = (x.as_slice(), output_grads.as_slice()) {
             x_slice.par_chunks_exact(embedding_dim)
                .zip(grad_slice.par_chunks_exact(embedding_dim))
                .fold(
                    || (init_scalar.clone(), init_gamma.clone(), init_bias.clone()),
                    |mut acc, (x_row, grad_row)| {
                        let (ref mut s_acc, ref mut g_acc, ref mut b_acc) = acc;
                        let mut buf = [0.0f32; MAX_SCALAR_PARAMS];
                        
                        let g_row = gamma_row;
                        
                        match self.variant {
                             crate::domain::richards::Variant::Tanh => {
                                 for j in 0..embedding_dim {
                                    let dy = grad_row[j];
                                    let eff_dy = if let Some(g) = g_row { dy * (g[j] as f32) } else { dy };
                                    
                                    let forward_val = self.grad_weights_scalar_into_kernel_f32::<TanhLike>(
                                        x_row[j], eff_dy, &mut buf[..scalar_param_count]
                                    );
                                    
                                    for i in 0..scalar_param_count {
                                        s_acc[i] += buf[i];
                                    }
                                    
                                    if let Some(ga) = g_acc {
                                        ga[j] += forward_val * dy;
                                    }
                                    if let Some(ba) = b_acc {
                                        ba[j] += dy;
                                    }
                                 }
                             },
                             _ => { // SigmoidLike
                                 for j in 0..embedding_dim {
                                    let dy = grad_row[j];
                                    let eff_dy = if let Some(g) = g_row { dy * (g[j] as f32) } else { dy };
                                    
                                    let forward_val = self.grad_weights_scalar_into_kernel_f32::<SigmoidLike>(
                                        x_row[j], eff_dy, &mut buf[..scalar_param_count]
                                    );
                                    
                                    for i in 0..scalar_param_count {
                                        s_acc[i] += buf[i];
                                    }
                                    
                                    if let Some(ga) = g_acc {
                                        ga[j] += forward_val * dy;
                                    }
                                    if let Some(ba) = b_acc {
                                        ba[j] += dy;
                                    }
                                 }
                             }
                        }
                        acc
                    }
                )
                .reduce(
                    || (init_scalar.clone(), init_gamma.clone(), init_bias.clone()),
                    |mut a, b| {
                        // Merge scalars
                        for i in 0..scalar_param_count {
                            a.0[i] += b.0[i];
                        }
                        // Merge gamma
                        if let (Some(ga), Some(gb)) = (&mut a.1, &b.1) {
                             for i in 0..embedding_dim {
                                 ga[i] += gb[i];
                             }
                        }
                        // Merge bias
                        if let (Some(ba), Some(bb)) = (&mut a.2, &b.2) {
                             for i in 0..embedding_dim {
                                 ba[i] += bb[i];
                             }
                        }
                        a
                    }
                )
        } else {
             // Fallback for non-contiguous arrays (rare but possible)
             let mut s_acc = init_scalar;
             let mut g_acc = init_gamma;
             let mut b_acc = init_bias;
             let mut buf = [0.0f32; MAX_SCALAR_PARAMS];
             
             let g_row = gamma_row;
             
             match self.variant {
                 crate::domain::richards::Variant::Tanh => {
                     for (x_row, grad_row) in x.outer_iter().zip(output_grads.outer_iter()) {
                         for j in 0..embedding_dim {
                            let dy = grad_row[j];
                            let eff_dy = if let Some(g) = g_row { dy * (g[j] as f32) } else { dy };
                            
                            let forward_val = self.grad_weights_scalar_into_kernel_f32::<TanhLike>(
                                x_row[j], eff_dy, &mut buf[..scalar_param_count]
                            );
                            
                            for i in 0..scalar_param_count {
                                s_acc[i] += buf[i];
                            }
                            
                            if let Some(ga) = &mut g_acc {
                                ga[j] += forward_val * dy;
                            }
                            if let Some(ba) = &mut b_acc {
                                ba[j] += dy;
                            }
                         }
                     }
                 },
                 _ => {
                     for (x_row, grad_row) in x.outer_iter().zip(output_grads.outer_iter()) {
                         for j in 0..embedding_dim {
                            let dy = grad_row[j];
                            let eff_dy = if let Some(g) = g_row { dy * (g[j] as f32) } else { dy };
                            
                            let forward_val = self.grad_weights_scalar_into_kernel_f32::<SigmoidLike>(
                                x_row[j], eff_dy, &mut buf[..scalar_param_count]
                            );
                            
                            for i in 0..scalar_param_count {
                                s_acc[i] += buf[i];
                            }
                            
                            if let Some(ga) = &mut g_acc {
                                ga[j] += forward_val * dy;
                            }
                            if let Some(ba) = &mut b_acc {
                                ba[j] += dy;
                            }
                         }
                     }
                 }
             }
             (s_acc, g_acc, b_acc)
        };

        // Finalize results
        let mut grads_accum_f64: Vec<f64> = Vec::with_capacity(self.weights_len());
        
        for &gi in scalar_acc.iter() {
            let mut g = (gi as f64) / total_elements;
            if !g.is_finite() { g = 0.0; }
            grads_accum_f64.push(g);
        }
        
        if let Some(ga) = gamma_acc {
             grads_accum_f64.extend(ga.into_iter().map(|v| {
                 let g = (v / batch_denom) as f64;
                 if g.is_finite() { g } else { 0.0 }
             }));
        }
        
        if let Some(ba) = bias_acc {
             grads_accum_f64.extend(ba.into_iter().map(|v| {
                 let g = (v / batch_denom) as f64;
                 if g.is_finite() { g } else { 0.0 }
             }));
        }
        
        grads_accum_f64
    }

    /// Vectorized backward pass: df/dx at x (analytical gradient), writing to output slice.
    pub fn derivative_into(&self, x: &[f64], out: &mut [f64]) {
        // Ensure output size matches input
        assert_eq!(x.len(), out.len(), "Input and output lengths must match");

        match self.variant {
            crate::domain::richards::Variant::Tanh => self.derivative_kernel_into_f64::<TanhLike>(x, out),
            _ => self.derivative_kernel_into_f64::<SigmoidLike>(x, out),
        }
    }

    /// Allocation-free scalar derivative.
    #[inline]
    pub fn derivative_scalar(&self, x: f64) -> f64 {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                RichardsKernel::<TanhLike>::from_curve(self).derivative_one_f64(x)
            }
            _ => RichardsKernel::<SigmoidLike>::from_curve(self).derivative_one_f64(x),
        }
    }

    /// Allocation-free scalar derivative for f32 inputs (avoids f32->f64 conversion).
    #[inline]
    pub fn derivative_scalar_f32(&self, x: f32) -> f32 {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                RichardsKernelF32::<TanhLike>::from_curve(self).derivative_one_f32(x)
            }
            _ => RichardsKernelF32::<SigmoidLike>::from_curve(self).derivative_one_f32(x),
        }
    }

    /// f32-friendly derivative into a caller-provided slice.
    pub fn derivative_into_f32(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), out.len(), "Input and output lengths must match");
        match self.variant {
            crate::domain::richards::Variant::Tanh => self.derivative_kernel_into_f32::<TanhLike>(x, out),
            _ => self.derivative_kernel_into_f32::<SigmoidLike>(x, out),
        }
    }

    /// Vectorized backward pass: df/dx at x (analytical gradient), single-pass.
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let mut out = Array1::zeros(x.len());
        self.derivative_into(x.as_slice().unwrap(), out.as_slice_mut().unwrap());
        out
    }

    fn grad_weights_scalar_into_kernel<V: VariantMarker>(
        &self,
        x: f64,
        grad_output: f64,
        out: &mut [f64],
    ) -> f64 {
        // Forward: f(x) = output_gain * gate(x) + output_bias
        let (nu, k, m, beta, temp, output_gain, output_bias, scale, shift) = self.get_all_params();
        let birch_tail = self.birch_exponential_tail;
        let input_scale = V::INPUT_SCALE;
        let outer_scale = V::OUTER_SCALE;
        let (adaptive_scale, adaptive_shift) = self.get_adaptive_scaling();

        let adaptive_normalized = adaptive_scale * x + adaptive_shift;
        let temp_scaled = adaptive_normalized / temp;
        let input = input_scale * (scale * temp_scaled + shift);

        // `get_all_params` enforces nu>0, beta>0, temp>0.
        let nu_eff = nu;
        let k_eff = if birch_tail { k * nu_eff } else { k };

        let exponent = -k_eff * (input - m);

        // base = 1 + beta * exp(exponent)
        // ln_base = log(base) = softplus(ln(beta) + exponent)
        // r = beta*exp(exponent)/base = sigmoid(ln(beta) + exponent)
        let t = beta.ln() + exponent;
        let ln_base = softplus_f64_richards(t);
        let r = unit_from_softplus_f64_richards(ln_base);

        let sigma = exp_f64_richards(-(ln_base) / nu);
        let gate = V::gate(sigma);
        let forward_val = output_gain * gate + output_bias;

        // dsigma/dinput = sigma * k * (beta*exp_term/base) / nu_eff = sigma * k * r / nu_eff
        let dsigma_dinput = (sigma * k_eff * r) / nu_eff;

        let pref = grad_output * output_gain * outer_scale;

        let mut pos = 0usize;
        if self.nu_learnable {
            // Birch-tail mode: nu also affects exponent via k_eff = k * nu.
            // d ln(sigma)/dnu = ln_base/nu^2 + (k * (input-m) * r)/nu
            let d_ln_sigma_d_nu = if birch_tail {
                (ln_base / (nu * nu)) + (k * (input - m) * r) / nu
            } else {
                ln_base / (nu * nu)
            };
            let d_sigma_d_nu = sigma * d_ln_sigma_d_nu;
            out[pos] = pref * d_sigma_d_nu;
            pos += 1;
        }
        if self.k_learnable {
            let d_sigma_d_k = if birch_tail {
                sigma * (input - m) * r
            } else {
                (sigma / nu_eff) * (input - m) * r
            };
            out[pos] = pref * d_sigma_d_k;
            pos += 1;
        }
        if self.m_learnable {
            let d_sigma_d_m = if birch_tail {
                -(sigma) * k * r
            } else {
                -(sigma / nu_eff) * k * r
            };
            out[pos] = pref * d_sigma_d_m;
            pos += 1;
        }
        if self.beta_learnable {
            // d ln(base)/d beta = exp(exponent)/base = r/beta
            let d_sigma_d_beta = -(sigma / nu_eff) * (r / beta);
            out[pos] = pref * d_sigma_d_beta;
            pos += 1;
        }

        if self.temperature_learnable {
            let d_temp_scaled_d_temp = -temp_scaled / temp;
            let d_input_d_temp = input_scale * scale * d_temp_scaled_d_temp;
            out[pos] = pref * dsigma_dinput * d_input_d_temp;
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

        debug_assert_eq!(
            pos,
            out.len(),
            "grad_weights_scalar_into: slice length mismatch"
        );
        forward_val
    }

    fn grad_weights_scalar_into_kernel_f32<V: VariantMarkerF32>(
        &self,
        x: f32,
        grad_output: f32,
        out: &mut [f32],
    ) -> f32 {
        // Forward: f(x) = output_gain * gate(x) + output_bias
        let (nu, k, m, beta, temp, output_gain, output_bias, scale, shift) = self.get_all_params();
        let birch_tail = self.birch_exponential_tail;
        let input_scale = V::INPUT_SCALE;
        let outer_scale = V::OUTER_SCALE;
        let (adaptive_scale, adaptive_shift) = self.get_adaptive_scaling();

        let nu = nu as f32;
        let k = k as f32;
        let m = m as f32;
        let beta = beta as f32;
        let temp = temp as f32;
        let output_gain = output_gain as f32;
        let output_bias = output_bias as f32;
        let scale = scale as f32;
        let shift = shift as f32;
        let adaptive_scale = adaptive_scale as f32;
        let adaptive_shift = adaptive_shift as f32;

        let adaptive_normalized = adaptive_scale * x + adaptive_shift;
        let temp_scaled = adaptive_normalized / temp;
        let input = input_scale * (scale * temp_scaled + shift);

        // `get_all_params` enforces nu>0, beta>0, temp>0.
        let nu_eff = nu;
        let k_eff = if birch_tail { k * nu_eff } else { k };

        let exponent = -k_eff * (input - m);

        let t = beta.ln() + exponent;
        let ln_base = softplus_f32_richards(t);
        let r = unit_from_softplus_f32_richards(ln_base);

        let sigma = exp_f32_richards(-(ln_base) / nu);
        let gate = V::gate(sigma);
        let forward_val = output_gain * gate + output_bias;

        let dsigma_dinput = (sigma * k_eff * r) / nu_eff;
        let pref = grad_output * output_gain * outer_scale;

        let mut pos = 0usize;
        if self.nu_learnable {
            let d_ln_sigma_d_nu = if birch_tail {
                (ln_base / (nu * nu)) + (k * (input - m) * r) / nu
            } else {
                ln_base / (nu * nu)
            };
            let d_sigma_d_nu = sigma * d_ln_sigma_d_nu;
            out[pos] = pref * d_sigma_d_nu;
            pos += 1;
        }
        if self.k_learnable {
            let d_sigma_d_k = if birch_tail {
                sigma * (input - m) * r
            } else {
                (sigma / nu_eff) * (input - m) * r
            };
            out[pos] = pref * d_sigma_d_k;
            pos += 1;
        }
        if self.m_learnable {
            let d_sigma_d_m = if birch_tail {
                -(sigma) * k * r
            } else {
                -(sigma / nu_eff) * k * r
            };
            out[pos] = pref * d_sigma_d_m;
            pos += 1;
        }
        if self.beta_learnable {
            let d_sigma_d_beta = -(sigma / nu_eff) * (r / beta);
            out[pos] = pref * d_sigma_d_beta;
            pos += 1;
        }
        if self.temperature_learnable {
            let d_temp_scaled_d_temp = -temp_scaled / temp;
            let d_input_d_temp = input_scale * scale * d_temp_scaled_d_temp;
            out[pos] = pref * dsigma_dinput * d_input_d_temp;
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
            out[pos] = pref * dsigma_dinput * d_input_d_scale;
            pos += 1;
        }
        if self.shift_learnable {
            let d_input_d_shift = input_scale;
            out[pos] = pref * dsigma_dinput * d_input_d_shift;
            pos += 1;
        }

        debug_assert_eq!(
            pos,
            out.len(),
            "grad_weights_scalar_into_kernel_f32: slice length mismatch"
        );
        forward_val
    }

    /// Compute gradients w.r.t. learnable parameters for a single scalar input into a preallocated
    /// slice. Returns the forward pass value.
    pub fn grad_weights_scalar_into(&self, x: f64, grad_output: f64, out: &mut [f64]) -> f64 {
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                self.grad_weights_scalar_into_kernel::<TanhLike>(x, grad_output, out)
            }
            _ => self.grad_weights_scalar_into_kernel::<SigmoidLike>(x, grad_output, out),
        }
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
        self.derivative_scalar(x)
    }

    /// Derivative for a single scalar x (f32-friendly, avoids f32->f64 conversion).
    #[inline]
    pub fn backward_scalar_f32(&self, x: f32) -> f32 {
        self.derivative_scalar_f32(x)
    }

    /// Compute scalar parameter gradients for a single f32 input.
    ///
    /// This returns gradients in the same internal order as `weights()` (scalar portion only).
    pub fn grad_weights_scalar_f32(&self, x: f32, grad_output: f32) -> Vec<f64> {
        let n = self.scalar_weights_len();
        debug_assert!(n <= MAX_SCALAR_PARAMS);

        let mut buf = vec![0.0f32; n];
        match self.variant {
            crate::domain::richards::Variant::Tanh => {
                self.grad_weights_scalar_into_kernel_f32::<TanhLike>(x, grad_output, &mut buf);
            }
            _ => {
                self.grad_weights_scalar_into_kernel_f32::<SigmoidLike>(x, grad_output, &mut buf);
            }
        };

        buf.into_iter()
            .map(|g| {
                let g = g as f64;
                if g.is_finite() { g } else { 0.0 }
            })
            .collect()
    }

    /// Update parameters using Adam optimizer
    pub fn step(&mut self, gradients: &[f64], learning_rate: f64) {
        // Count learnable parameters (including array parameters)
        let param_count = self.weights_len();

        // Ensure optimizer is properly initialized for the correct number of parameters
        let needs_optimizer_init = match self.optimizer.as_ref() {
            None => true,
            Some(opt) => opt.m().shape() != [param_count, 1],
        };
        if needs_optimizer_init {
            self.optimizer = Some(Adam::new((param_count, 1)));
        }

        // Extract current parameter values for learnable parameters without intermediate
        // allocations. For positive-constrained parameters we optimize u where p =
        // softplus(u) to keep p > 0
        let mut param_values: Vec<f32> = Vec::with_capacity(param_count);
        let mut grad_values: Vec<f32> = Vec::with_capacity(param_count);
        let mut grad_idx: usize = 0;
        if self.nu_learnable {
            let nu = self.get_param(self.nu, self.learned_nu, 1.0);
            let nu_pos = if nu > 0.0 { nu } else { 1e-6 };
            let u = inv_softplus_f64_richards(nu_pos);
            let d_nu_d_u = unit_from_softplus_f64_richards(nu_pos);
            param_values.push(u as f32);
            grad_values.push((gradients[grad_idx] * d_nu_d_u) as f32);
            grad_idx += 1;
        }
        if self.k_learnable {
            let k = self.get_param(self.k, self.learned_k, 1.0);
            let k_pos = if k > 0.0 { k } else { 1e-6 };
            let u = inv_softplus_f64_richards(k_pos);
            let d_k_d_u = unit_from_softplus_f64_richards(k_pos);
            param_values.push(u as f32);
            grad_values.push((gradients[grad_idx] * d_k_d_u) as f32);
            grad_idx += 1;
        }
        if self.m_learnable {
            param_values.push(self.get_param(self.m, self.learned_m, 0.0) as f32);
            grad_values.push(gradients[grad_idx] as f32);
            grad_idx += 1;
        }
        if self.beta_learnable {
            let beta = self.get_param(self.beta, self.learned_beta, 1.0);
            let beta_pos = if beta > 0.0 { beta } else { 1e-6 };
            let u = inv_softplus_f64_richards(beta_pos);
            let d_beta_d_u = unit_from_softplus_f64_richards(beta_pos);
            param_values.push(u as f32);
            grad_values.push((gradients[grad_idx] * d_beta_d_u) as f32);
            grad_idx += 1;
        }
        if self.temperature_learnable {
            let t = self.get_param(self.temperature, self.learned_temperature, 1.0);
            let t_pos = if t > 0.0 { t } else { 1e-6 };
            let u = inv_softplus_f64_richards(t_pos);
            let d_t_d_u = unit_from_softplus_f64_richards(t_pos);
            param_values.push(u as f32);
            grad_values.push((gradients[grad_idx] * d_t_d_u) as f32);
            grad_idx += 1;
        }
        if self.output_gain_learnable {
            param_values
                .push(self.get_param(self.output_gain, self.learned_output_gain, 1.0) as f32);
            grad_values.push(gradients[grad_idx] as f32);
            grad_idx += 1;
        }
        if self.output_bias_learnable {
            param_values
                .push(self.get_param(self.output_bias, self.learned_output_bias, 0.0) as f32);
            grad_values.push(gradients[grad_idx] as f32);
            grad_idx += 1;
        }
        if self.scale_learnable {
            param_values.push(self.get_param(self.scale, self.learned_scale, 1.0) as f32);
            grad_values.push(gradients[grad_idx] as f32);
            grad_idx += 1;
        }
        if self.shift_learnable {
            param_values.push(self.get_param(self.shift, self.learned_shift, 0.0) as f32);
            grad_values.push(gradients[grad_idx] as f32);
            grad_idx += 1;
        }
        if self.gamma_learnable
            && let Some(g) = self.gamma.as_ref()
        {
            param_values.extend(g.iter().copied());
            for _ in 0..g.len() {
                grad_values.push(gradients[grad_idx] as f32);
                grad_idx += 1;
            }
        }
        if self.bias_learnable
            && let Some(b) = self.bias.as_ref()
        {
            param_values.extend(b.iter().copied());
            for _ in 0..b.len() {
                grad_values.push(gradients[grad_idx] as f32);
                grad_idx += 1;
            }
        }

        if let Some(ref mut optimizer) = self.optimizer {
            // Create 2D arrays for Adam optimizer interface
            let mut params = Array2::from_shape_vec((param_count, 1), param_values)
                .expect("Failed to create params array");
            let grads = Array2::from_shape_vec((param_count, 1), grad_values)
                .expect("Failed to create grads array");

            optimizer.step(&mut params, &grads, learning_rate as f32);

            // Apply updates back to learned parameters (no hard clipping)
            let mut idx = 0;
            if self.nu_learnable {
                self.learned_nu = Some(softplus_f64_richards(params[[idx, 0]] as f64));
                idx += 1;
            }
            if self.k_learnable {
                self.learned_k = Some(softplus_f64_richards(params[[idx, 0]] as f64));
                idx += 1;
            }
            if self.m_learnable {
                self.learned_m = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.beta_learnable {
                self.learned_beta = Some(softplus_f64_richards(params[[idx, 0]] as f64));
                idx += 1;
            }
            if self.temperature_learnable {
                self.learned_temperature = Some(softplus_f64_richards(params[[idx, 0]] as f64));
                idx += 1;
            }
            if self.output_gain_learnable {
                self.learned_output_gain = Some(params[[idx, 0]] as f64);
                idx += 1;
            }
            if self.output_bias_learnable {
                self.learned_output_bias = Some(params[[idx, 0]] as f64);
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
            if self.gamma_learnable {
                if let Some(ref mut gamma_arc) = self.gamma {
                    let gamma = Arc::make_mut(gamma_arc);
                    let gamma_size = gamma.len();
                    for i in 0..gamma_size {
                        if idx < param_count {
                            gamma[[0, i]] = params[[idx, 0]];
                            idx += 1;
                        }
                    }
                } else {
                    // Skip gamma parameters if array doesn't exist
                    // idx remains unchanged since there are no gamma parameters to update
                }
            }
            if self.bias_learnable {
                if let Some(ref mut bias_arc) = self.bias {
                    let bias = Arc::make_mut(bias_arc);
                    let bias_size = bias.len();
                    for i in 0..bias_size {
                        if idx < param_count {
                            bias[[0, i]] = params[[idx, 0]];
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

    /// Set learnable parameter values from a vector (for testing)
    pub fn set_weights_from_vec(&mut self, weights: &[f64]) {
        let mut _idx = 0;

        if self.nu_learnable && _idx < weights.len() {
            let v = weights[_idx];
            self.learned_nu = Some(if v > 0.0 { v } else { 1e-6 });
            _idx += 1;
        }
        if self.k_learnable && _idx < weights.len() {
            let v = weights[_idx];
            self.learned_k = Some(if v > 0.0 { v } else { 1e-6 });
            _idx += 1;
        }
        if self.m_learnable && _idx < weights.len() {
            self.learned_m = Some(weights[_idx]);
            _idx += 1;
        }
        if self.beta_learnable && _idx < weights.len() {
            let v = weights[_idx];
            self.learned_beta = Some(if v > 0.0 { v } else { 1e-6 });
            _idx += 1;
        }
        if self.temperature_learnable && _idx < weights.len() {
            let v = weights[_idx];
            self.learned_temperature = Some(if v > 0.0 { v } else { 1e-6 });
            _idx += 1;
        }
        if self.output_gain_learnable && _idx < weights.len() {
            self.learned_output_gain = Some(weights[_idx]);
            _idx += 1;
        }
        if self.output_bias_learnable && _idx < weights.len() {
            self.learned_output_bias = Some(weights[_idx]);
            _idx += 1;
        }
        if self.scale_learnable && _idx < weights.len() {
            self.learned_scale = Some(weights[_idx]);
            _idx += 1;
        }
        if self.shift_learnable && _idx < weights.len() {
            self.learned_shift = Some(weights[_idx]);
            _idx += 1;
        }
        // Note: gamma and bias not supported in set_weights_from_vec (would need matrix dims)
    }

    /// Return current learnable parameter values as a vector (only learnable parameters)
    /// Note: Returns default values until parameters are actually trained/updated
    pub fn weights(&self) -> Vec<f64> {
        let mut weights: Vec<f64> = Vec::with_capacity(self.weights_len());
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
        if self.temperature_learnable {
            weights.push(self.get_param(self.temperature, self.learned_temperature, 1.0));
        }
        if self.output_gain_learnable {
            weights.push(self.get_param(self.output_gain, self.learned_output_gain, 1.0));
        }
        if self.output_bias_learnable {
            weights.push(self.get_param(self.output_bias, self.learned_output_bias, 0.0));
        }
        if self.scale_learnable {
            weights.push(self.get_param(self.scale, self.learned_scale, 1.0));
        }
        if self.shift_learnable {
            weights.push(self.get_param(self.shift, self.learned_shift, 0.0));
        }
        if self.gamma_learnable
            && let Some(g) = self.gamma.as_ref()
        {
            weights.extend(g.iter().map(|&x| x as f64));
        }
        if self.bias_learnable
            && let Some(b) = self.bias.as_ref()
        {
            weights.extend(b.iter().map(|&x| x as f64));
        }

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
        if self.variant != crate::domain::richards::Variant::Adaptive {
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
            self.running_sq_sum.unwrap() * momentum + batch_var_sum * (1.0 - momentum);

        self.running_sum = Some(new_running_sum);
        self.running_sq_sum = Some(new_running_sq_sum);
        self.count = Some(new_count);

        self.update_adaptive_scaling();
    }

    /// Update adaptive scale and shift from running statistics
    fn update_adaptive_scaling(&mut self) {
        if let (Some(running_sum), Some(running_sq_sum), Some(count)) =
            (self.running_sum, self.running_sq_sum, self.count)
            && count > 1
        {
            let mean = running_sum / count as f64;
            let variance = (running_sq_sum / (count - 1) as f64)
                - (running_sum.powi(2) / count as f64) / (count - 1) as f64;
            let std = variance.sqrt().max(1e-6); // Minimum std for numerical stability

            // Adaptive normalization: center at mean, scale to unit variance
            self.adaptive_scale = Some(1.0 / std);
            self.adaptive_shift = Some(-mean / std);
        }
    }

    /// Get adaptive scaling parameters (or default to (1.0, 0.0) if not adaptive)
    fn get_adaptive_scaling(&self) -> (f64, f64) {
        if self.variant == crate::domain::richards::Variant::Adaptive {
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
        if self.variant == crate::domain::richards::Variant::Adaptive {
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
        if self.variant != crate::domain::richards::Variant::Polynomial {
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

    /// Get polynomial degree (or 1 for identity if not polynomial variant)
    fn get_polynomial_power(&self) -> usize {
        self.poly_power.unwrap_or(1)
    }

    /// Evaluate polynomial at a given point
    fn evaluate_polynomial(&self, x: f64) -> f64 {
        if let Some(coeffs) = &self.poly_coeffs {
            coeffs
                .iter()
                .enumerate()
                .fold(0.0, |sum, (i, &coeff)| sum + coeff * x.powi(i as i32))
        } else {
            // Identity if no coefficients set
            x
        }
    }

    /// Get polynomial-input scaling (applied before Richards activation)
    fn get_polynomial_scaling(&self) -> f64 {
        if self.variant == crate::domain::richards::Variant::Polynomial {
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

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn test_richards_scalar_vector_consistency() {
        // Test that forward_scalar and forward_into produce consistent results
        // Note: forward() uses extended Richards with beta/temperature, so we test the simpler
        // methods
        let curve = RichardsCurve::new_default();
        let x_vals = vec![0.5, -0.5, 1.0, -1.0, 0.0];

        for &x_val in &x_vals {
            let scalar_out = curve.forward_scalar(x_val);

            // Test forward_into
            let mut vector_out = vec![0.0];
            curve.forward_into(&[x_val], &mut vector_out);

            // forward_scalar should match forward_into since both use extended Richards with same
            // params
            assert!(
                (scalar_out - vector_out[0]).abs() < 1e-10,
                "Mismatch at x={}: scalar={}, vector={}",
                x_val,
                scalar_out,
                vector_out[0]
            );
        }
    }

    #[test]
    fn test_richards_zero_copy() {
        let curve = RichardsCurve::new_default();
        let x_val = vec![0.5, -0.5, 1.0];
        let mut out = vec![0.0; 3];

        curve.forward_into(&x_val, &mut out);

        for (i, &val) in x_val.iter().enumerate() {
            let scalar = curve.forward_scalar(val);
            assert!(
                (out[i] - scalar).abs() < 1e-6,
                "Zero-copy output mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_gradient_numerical_check() {
        // Numerical gradient checking using finite differences
        let mut curve = RichardsCurve::new_learnable(crate::domain::richards::Variant::Sigmoid);

        // Initialize with standard Richards parameters (beta=1, temp=1)
        curve.learned_nu = Some(1.5);
        curve.learned_k = Some(2.0);
        curve.learned_m = Some(0.5);
        curve.learned_beta = Some(1.0); // Standard Richards
        curve.learned_temperature = Some(1.0); // No temperature scaling

        let x = 0.3;
        let grad_output = 1.0;
        let epsilon = 1e-5;

        // Compute analytical gradients
        let analytical_grads = curve.grad_weights_scalar(x, grad_output);

        // Compute numerical gradients for each parameter
        let params = curve.weights();
        let mut numerical_grads = vec![0.0; params.len()];

        for i in 0..params.len() {
            // Perturb parameter +epsilon
            let mut params_plus = params.clone();
            params_plus[i] += epsilon;
            let mut curve_plus = curve.clone();
            curve_plus.set_weights_from_vec(&params_plus);
            let f_plus = curve_plus.forward_scalar(x);

            // Perturb parameter -epsilon
            let mut params_minus = params.clone();
            params_minus[i] -= epsilon;
            let mut curve_minus = curve.clone();
            curve_minus.set_weights_from_vec(&params_minus);
            let f_minus = curve_minus.forward_scalar(x);

            // Numerical gradient
            numerical_grads[i] = (f_plus - f_minus) / (2.0 * epsilon) * grad_output;
        }

        // Compare analytical vs numerical
        let param_names = [
            "nu",
            "k",
            "m",
            "beta",
            "temp",
            "output_gain",
            "output_bias",
            "scale",
            "shift",
        ];

        println!("\\nGradient comparison:");
        println!(
            "Params: nu={}, k={}, m={}, beta={}, temp={}",
            curve.get_param(curve.nu, curve.learned_nu, 1.0),
            curve.get_param(curve.k, curve.learned_k, 1.0),
            curve.get_param(curve.m, curve.learned_m, 0.0),
            curve.get_param(curve.beta, curve.learned_beta, 1.0),
            curve.get_param(curve.temperature, curve.learned_temperature, 1.0)
        );

        let mut max_rel_error: f64 = 0.0;
        for i in 0..analytical_grads.len() {
            let diff = (analytical_grads[i] - numerical_grads[i]).abs();
            let rel_error = if numerical_grads[i].abs() > 1e-8 {
                diff / numerical_grads[i].abs()
            } else {
                diff
            };

            let param_name = param_names.get(i).unwrap_or(&"unknown");

            println!(
                "{}[{}]: analytical={:.6}, numerical={:.6}, diff={:.6}, rel_err={:.6}",
                param_name, i, analytical_grads[i], numerical_grads[i], diff, rel_error
            );

            max_rel_error = max_rel_error.max(rel_error);
        }

        // Assert that all gradients are accurate within 1% relative error
        assert!(
            max_rel_error < 0.01,
            "Maximum relative error {:.6} exceeds 1% threshold",
            max_rel_error
        );
    }

    #[test]
    fn test_beta_parameter_behavior() {
        // Test that beta=1.0 gives standard Richards
        let mut curve = RichardsCurve::new_default();
        curve.learned_beta = Some(1.0);
        curve.learned_nu = Some(1.0);
        curve.learned_k = Some(1.0);
        curve.learned_m = Some(0.0);
        curve.learned_temperature = Some(1.0);

        let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        for &x in &x_vals {
            let output = curve.forward_scalar(x);
            // Standard logistic: σ(x) = 1 / (1 + e^(-x))
            let expected = 1.0 / (1.0 + (-x).exp());
            assert!(
                (output - expected).abs() < 1e-6,
                "Beta=1.0 should give standard logistic at x={}: got {}, expected {}",
                x,
                output,
                expected
            );
        }
    }

    #[test]
    fn test_temperature_scaling() {
        // Create non-adaptive curve to test temperature without interference
        let mut curve = RichardsCurve::sigmoid(true); // Learnable sigmoid
        curve.learned_nu = Some(1.0);
        curve.learned_k = Some(1.0);
        curve.learned_m = Some(0.0);
        curve.learned_beta = Some(1.0);
        curve.learned_scale = Some(1.0);
        curve.learned_shift = Some(0.0);
        curve.learned_output_gain = Some(1.0);
        curve.learned_output_bias = Some(0.0);

        // Test at a point well above the midpoint
        let test_x = 1.0;

        curve.learned_temperature = Some(0.5); // Sharper (lower temp scales input up)
        let sharp_output = curve.forward_scalar(test_x);

        curve.learned_temperature = Some(2.0); // Softer (higher temp scales input down)
        let soft_output = curve.forward_scalar(test_x);

        // At x=1.0 (positive), lower temperature (0.5) amplifies input: x/0.5=2.0
        // Higher temperature (2.0) reduces input: x/2.0=0.5
        // So sharp should have higher sigmoid output than soft
        assert!(
            sharp_output > soft_output,
            "Lower temperature should amplify transitions: sharp={}, soft={}",
            sharp_output,
            soft_output
        );
    }

    #[test]
    fn test_birch_exponential_tail_decouples_nu_in_left_tail() {
        // In Birch-tail mode we scale the exponent by nu so the left-tail behaves like:
        // sigma(x) ~= C * exp(k * x), independent of nu.
        let k = 1.7;

        let mut c1 = RichardsCurve::sigmoid(false).with_birch_exponential_tail(true);
        c1.k = Some(k);
        c1.nu = Some(0.5);
        c1.m = Some(0.0);
        c1.beta = Some(1.0);
        c1.temperature = Some(1.0);
        c1.scale = Some(1.0);
        c1.shift = Some(0.0);

        let mut c2 = c1.clone();
        c2.nu = Some(2.0);

        let x1 = -20.0;
        let x2 = -21.0;
        let ratio1 = c1.forward_scalar(x2) / c1.forward_scalar(x1);
        let ratio2 = c2.forward_scalar(x2) / c2.forward_scalar(x1);
        let expected = (k * (x2 - x1)).exp();

        assert!(
            (ratio1 - expected).abs() < 1e-3,
            "ratio1={} expected={}",
            ratio1,
            expected
        );
        assert!(
            (ratio2 - expected).abs() < 1e-3,
            "ratio2={} expected={}",
            ratio2,
            expected
        );
        assert!(
            (ratio1 - ratio2).abs() < 1e-4,
            "ratios should match across nu: {} vs {}",
            ratio1,
            ratio2
        );

        // Sanity check: default Richards behavior depends on nu (ratio ~= exp(k*(x2-x1)/nu)).
        let mut r1 = c1.clone();
        r1.set_birch_exponential_tail(false);
        let mut r2 = c2.clone();
        r2.set_birch_exponential_tail(false);
        let rr1 = r1.forward_scalar(x2) / r1.forward_scalar(x1);
        let rr2 = r2.forward_scalar(x2) / r2.forward_scalar(x1);
        assert!(
            (rr1 - rr2).abs() > 1e-4,
            "default Richards ratios should differ across nu: {} vs {}",
            rr1,
            rr2
        );
    }

    #[test]
    fn test_no_nan_inf_in_gradients() {
        let curve = RichardsCurve::new_learnable(crate::domain::richards::Variant::Sigmoid);
        // Test with extreme inputs
        let extreme_inputs = vec![-100.0, -10.0, 0.0, 10.0, 100.0];

        for &x in &extreme_inputs {
            let grads = curve.grad_weights_scalar(x, 1.0);

            for (i, &g) in grads.iter().enumerate() {
                assert!(
                    g.is_finite(),
                    "Gradient {} is not finite for input x={}: grad={}",
                    i,
                    x,
                    g
                );
            }
        }
    }

    #[test]
    fn test_richards_optimizations_integration() {
        // Test that all optimizations work together correctly
        let curve = RichardsCurve::new_default();

        // Test input data
        let x_vals = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let x_array = Array1::from_vec(x_vals.clone());

        // Test RichardsCurve optimizations
        let curve_output = curve.forward(&x_array);
        assert_eq!(curve_output.len(), x_vals.len());

        // Verify outputs are reasonable (no NaN/inf)
        for val in curve_output.iter() {
            assert!(
                val.is_finite(),
                "RichardsCurve output contains non-finite value: {}",
                val
            );
        }
    }
}
