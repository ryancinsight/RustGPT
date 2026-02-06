//! "Soft" numeric algorithms (softmax, softplus, etc.)
//!
//! This module centralizes numerically-stable "soft" transforms so they don't
//! get duplicated in domain-specific modules (e.g. `richards`).

pub mod softmax;

pub use softmax::Softmax;

use crate::domain::pade;

/// Scalar types supported by the `soft` helpers.
///
/// This avoids a dependency on `num-traits` while allowing ergonomic generic call sites.
pub trait SoftScalar: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

impl SoftScalar for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl SoftScalar for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

/// Numerically-stable softplus.
///
/// Uses $\log(1+\exp(x))$ with the usual stable split, computing exp via Padé.
#[inline]
pub fn softplus<T: SoftScalar>(x: T) -> T {
    let x64 = x.to_f64();
    if x64.is_nan() {
        return T::from_f64(f64::NAN);
    }
    if x64 == f64::INFINITY {
        return T::from_f64(f64::INFINITY);
    }
    if x64 == f64::NEG_INFINITY {
        return T::from_f64(0.0);
    }

    let out = if x64 > 0.0 {
        x64 + pade::exp(-x64).ln_1p()
    } else {
        pade::exp(x64).ln_1p()
    };

    T::from_f64(out)
}

/// Numerically-stable log-sum-exp for a slice.
#[inline]
pub fn logsumexp<T: SoftScalar>(xs: &[T]) -> T {
    if xs.is_empty() {
        return T::from_f64(f64::NEG_INFINITY);
    }

    let mut any_pos_inf = false;
    let mut any_nan = false;
    for &v in xs {
        let v64 = v.to_f64();
        if v64 == f64::INFINITY {
            any_pos_inf = true;
        } else if v64.is_nan() {
            any_nan = true;
        }
    }
    if any_pos_inf {
        return T::from_f64(f64::INFINITY);
    }
    if any_nan {
        return T::from_f64(f64::NAN);
    }

    let mut max_val = f64::NEG_INFINITY;
    for &v in xs {
        let v64 = v.to_f64();
        if v64.is_finite() {
            max_val = max_val.max(v64);
        }
    }
    if !max_val.is_finite() {
        return T::from_f64(f64::NEG_INFINITY);
    }

    let mut sum = 0.0f64;
    for &v in xs {
        let v64 = v.to_f64();
        if v64.is_finite() {
            sum += pade::exp(v64 - max_val);
        }
    }

    T::from_f64(max_val + sum.ln())
}

#[deprecated(note = "use crate::domain::soft::softplus(x) (generic) instead")]
#[inline]
pub fn softplus_f64(x: f64) -> f64 {
    softplus(x)
}

#[deprecated(note = "use crate::domain::soft::softplus(x) (generic) instead")]
#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    softplus(x)
}

#[inline]
#[deprecated(note = "use crate::domain::soft::logsumexp(xs) (generic) instead")]
pub fn logsumexp_f64(xs: &[f64]) -> f64 {
    logsumexp(xs)
}

#[inline]
#[deprecated(note = "use crate::domain::soft::logsumexp(xs) (generic) instead")]
pub fn logsumexp_f32(xs: &[f32]) -> f32 {
    logsumexp(xs)
}
