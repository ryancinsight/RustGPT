use std::sync::OnceLock;

use super::{act::RichardsActivation, curve::RichardsCurve};

static SIGMOID_CURVE_F32: OnceLock<RichardsCurve> = OnceLock::new();
static TANH_CURVE_F32: OnceLock<RichardsCurve> = OnceLock::new();
static SILU_ACT_F32: OnceLock<RichardsActivation> = OnceLock::new();

#[inline]
pub fn sigmoid_f32(x: f32) -> f32 {
    let curve = SIGMOID_CURVE_F32.get_or_init(|| RichardsCurve::sigmoid(false));
    curve.forward_scalar(x as f64) as f32
}

#[inline]
pub fn dsigmoid_f32(x: f32) -> f32 {
    let curve = SIGMOID_CURVE_F32.get_or_init(|| RichardsCurve::sigmoid(false));
    curve.derivative_scalar(x as f64) as f32
}

#[inline]
pub fn tanh_f32(x: f32) -> f32 {
    let curve = TANH_CURVE_F32.get_or_init(|| RichardsCurve::tanh(false));
    curve.forward_scalar(x as f64) as f32
}

#[inline]
pub fn dtanh_f32(x: f32) -> f32 {
    let curve = TANH_CURVE_F32.get_or_init(|| RichardsCurve::tanh(false));
    curve.derivative_scalar(x as f64) as f32
}

#[inline]
pub fn silu_f32(x: f32) -> f32 {
    // SiLU(x) = x * sigmoid(x). Use RichardsActivation to keep this on the shared path.
    let act = SILU_ACT_F32.get_or_init(|| RichardsActivation::sigmoid(false));
    act.forward_scalar(x as f64) as f32
}

#[inline]
pub fn dsilu_f32(x: f32) -> f32 {
    // d/dx (x * sigmoid(x)) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    let s = sigmoid_f32(x);
    s + x * s * (1.0 - s)
}
