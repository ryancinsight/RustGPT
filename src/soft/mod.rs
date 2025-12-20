//! "Soft" numeric algorithms (softmax, softplus, etc.)
//!
//! This module centralizes numerically-stable "soft" transforms so they don't
//! get duplicated in domain-specific modules (e.g. `richards`).

pub mod softmax;

pub use softmax::Softmax;

use crate::pade;

#[inline]
pub fn softplus_f64(x: f64) -> f64 {
    // log(1 + exp(x)) computed stably, using Padé exp.
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }

    // Standard stable softplus split.
    if x > 0.0 {
        x + pade::exp_f64(-x).ln_1p()
    } else {
        pade::exp_f64(x).ln_1p()
    }
}

#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    softplus_f64(x as f64) as f32
}

#[inline]
pub fn logsumexp_f64(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mut max_val = f64::NEG_INFINITY;
    for &v in xs {
        if v.is_finite() {
            max_val = max_val.max(v);
        }
    }
    if !max_val.is_finite() {
        return max_val; // all -inf => -inf; any NaN => NaN
    }

    let mut sum = 0.0;
    for &v in xs {
        if v.is_finite() {
            sum += pade::exp_f64(v - max_val);
        }
    }
    max_val + sum.ln()
}

#[inline]
pub fn logsumexp_f32(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NEG_INFINITY;
    }

    let mut max_val = f32::NEG_INFINITY;
    for &v in xs {
        if v.is_finite() {
            max_val = max_val.max(v);
        }
    }
    if !max_val.is_finite() {
        return max_val;
    }

    let mut sum: f64 = 0.0;
    for &v in xs {
        if v.is_finite() {
            sum += pade::exp_f64((v - max_val) as f64);
        }
    }
    (max_val as f64 + sum.ln()) as f32
}
