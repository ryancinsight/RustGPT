use super::exp::PadeExp;

/// Scalar types supported by Padé exp helpers.
///
/// This keeps the crate dependency-free (no `num-traits`) while still allowing
/// ergonomic generic call sites: `pade::exp(x)` for both `f32` and `f64`.
pub trait ExpScalar: Copy {
    fn to_f64(self) -> f64;
    fn from_f64(x: f64) -> Self;
}

impl ExpScalar for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

impl ExpScalar for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

/// Generic, stable exponential approximation.
///
/// Prefer this over `exp_f32`/`exp_f64`.
#[inline]
pub fn exp<T: ExpScalar>(x: T) -> T {
    T::from_f64(PadeExp::exp(x.to_f64()))
}

#[deprecated(note = "use crate::domain::pade::exp(x) (generic) instead")]
#[inline]
pub fn exp_f64(x: f64) -> f64 {
    exp(x)
}

#[deprecated(note = "use crate::domain::pade::exp(x) (generic) instead")]
#[inline]
pub fn exp_f32(x: f32) -> f32 {
    exp(x)
}
