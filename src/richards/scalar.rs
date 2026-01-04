use std::sync::OnceLock;

use super::{act::RichardsActivation, curve::RichardsCurve};

/// Minimal trait to make Richards scalar helpers generic without pulling in a numeric crate.
pub trait RichardsScalar: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

impl RichardsScalar for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl RichardsScalar for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

static SIGMOID_CURVE: OnceLock<RichardsCurve> = OnceLock::new();
static TANH_CURVE: OnceLock<RichardsCurve> = OnceLock::new();
static SILU_ACT: OnceLock<RichardsActivation> = OnceLock::new();

#[inline]
pub fn sigmoid<T: RichardsScalar>(x: T) -> T {
    let curve = SIGMOID_CURVE.get_or_init(|| RichardsCurve::sigmoid(false));
    T::from_f64(curve.forward_scalar(x.to_f64()))
}

#[inline]
pub fn dsigmoid<T: RichardsScalar>(x: T) -> T {
    let curve = SIGMOID_CURVE.get_or_init(|| RichardsCurve::sigmoid(false));
    T::from_f64(curve.derivative_scalar(x.to_f64()))
}

#[inline]
pub fn tanh<T: RichardsScalar>(x: T) -> T {
    let curve = TANH_CURVE.get_or_init(|| RichardsCurve::tanh(false));
    T::from_f64(curve.forward_scalar(x.to_f64()))
}

#[inline]
pub fn dtanh<T: RichardsScalar>(x: T) -> T {
    let curve = TANH_CURVE.get_or_init(|| RichardsCurve::tanh(false));
    T::from_f64(curve.derivative_scalar(x.to_f64()))
}

#[inline]
pub fn silu<T: RichardsScalar>(x: T) -> T {
    // SiLU(x) = x * sigmoid(x). Use RichardsActivation to keep this on the shared path.
    let act = SILU_ACT.get_or_init(|| RichardsActivation::sigmoid(false));
    T::from_f64(act.forward_scalar(x.to_f64()))
}

#[inline]
pub fn dsilu<T: RichardsScalar>(x: T) -> T {
    // d/dx (x * sigmoid(x)) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    let s = sigmoid(x);
    let xf = x.to_f64();
    let sf = s.to_f64();
    T::from_f64(sf + xf * sf * (1.0 - sf))
}

// --- Back-compat wrappers (kept at module level, but no f32 submodule) ---

#[inline]
pub fn sigmoid_f32(x: f32) -> f32 {
    let curve = SIGMOID_CURVE.get_or_init(|| RichardsCurve::sigmoid(false));
    curve.forward_scalar_f32(x)
}

#[inline]
pub fn dsigmoid_f32(x: f32) -> f32 {
    let curve = SIGMOID_CURVE.get_or_init(|| RichardsCurve::sigmoid(false));
    curve.derivative_scalar_f32(x)
}

#[inline]
pub fn tanh_f32(x: f32) -> f32 {
    let curve = TANH_CURVE.get_or_init(|| RichardsCurve::tanh(false));
    curve.forward_scalar_f32(x)
}

#[inline]
pub fn dtanh_f32(x: f32) -> f32 {
    let curve = TANH_CURVE.get_or_init(|| RichardsCurve::tanh(false));
    curve.derivative_scalar_f32(x)
}

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    // SiLU(x) = x * sigmoid(x)
    x * sigmoid_f32(x)
}

#[inline]
pub fn dsilu_f32(x: f32) -> f32 {
    dsilu(x)
}
