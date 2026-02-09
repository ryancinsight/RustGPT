//! Safe numeric conversion utilities
//!
//! Provides helper functions for common numeric conversions that avoid
//! precision loss warnings and handle edge cases properly.
//!
//! # Note on Activation Functions
//!
//! This codebase uses learnable adaptive Richards activations from the
//! `domain::richards` module instead of fixed activation functions.
//! See [`domain::richards::RichardsCurve`] for the primary activation API.

/// Convert `usize` to `f32` with precision loss acknowledgment
#[inline]
#[allow(clippy::cast_precision_loss)]
pub const fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

/// Convert `usize` to `f64` with precision loss acknowledgment
#[inline]
#[allow(clippy::cast_precision_loss)]
pub const fn usize_to_f64(value: usize) -> f64 {
    value as f64
}

/// Convert `i32` to `f32` with precision loss acknowledgment
#[inline]
#[allow(clippy::cast_precision_loss)]
pub const fn i32_to_f32(value: i32) -> f32 {
    value as f32
}

/// Convert `f32` to `f64` losslessly
#[inline]
pub fn f32_to_f64(value: f32) -> f64 {
    f64::from(value)
}

/// Convert `f64` to `f32` with truncation acknowledgment
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn f64_to_f32(value: f64) -> f32 {
    value as f32
}

/// Convert `f32` to `usize` with truncation and sign loss acknowledgment
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn f32_to_usize(value: f32) -> usize {
    value.max(0.0) as usize
}

/// Convert `f32` to `usize` with rounding
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn f32_to_usize_round(value: f32) -> usize {
    value.round().max(0.0) as usize
}

/// Convert `usize` to `i32` with truncation acknowledgment
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
pub fn usize_to_i32(value: usize) -> i32 {
    value.min(i32::MAX as usize) as i32
}

/// Compute reciprocal of `usize` as `f32`
#[inline]
#[allow(clippy::cast_precision_loss)]
pub fn reciprocal_usize_f32(value: usize) -> f32 {
    1.0 / (value.max(1) as f32)
}

/// Compute reciprocal of `usize` as `f64`
#[inline]
#[allow(clippy::cast_precision_loss)]
pub fn reciprocal_usize_f64(value: usize) -> f64 {
    1.0 / (value.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usize_to_f32() {
        assert_eq!(usize_to_f32(100), 100.0);
        assert_eq!(usize_to_f32(0), 0.0);
    }

    #[test]
    fn test_f32_to_usize() {
        assert_eq!(f32_to_usize(10.5), 10);
        assert_eq!(f32_to_usize(-5.0), 0); // Negative clamped to 0
        assert_eq!(f32_to_usize(0.0), 0);
    }

    #[test]
    fn test_f32_to_usize_round() {
        assert_eq!(f32_to_usize_round(10.5), 11);
        assert_eq!(f32_to_usize_round(10.4), 10);
        assert_eq!(f32_to_usize_round(-5.0), 0);
    }

    #[test]
    fn test_reciprocal() {
        assert_eq!(reciprocal_usize_f32(2), 0.5);
        assert_eq!(reciprocal_usize_f32(0), 1.0); // Handles zero safely
    }
}
