use serde::{Deserialize, Serialize};

use crate::richards::RichardsCurve;

/// A scalar value that can adapt over time (or other input) using a Richards curve.
///
/// This allows hyperparameters like loss weights, thresholds, or mixing coefficients
/// to be learned or scheduled dynamically rather than being fixed constants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveScalar {
    /// Fixed constant value
    Fixed(f32),
    /// Value modulated by a Richards curve based on input signal (e.g., progress t)
    /// val(t) = curve(t)
    Richards {
        curve: Box<RichardsCurve>,
        /// Optional scale factor to apply to curve output (default 1.0)
        output_scale: f32,
    },
}

impl Default for AdaptiveScalar {
    fn default() -> Self {
        Self::Fixed(1.0)
    }
}

impl From<f32> for AdaptiveScalar {
    fn from(v: f32) -> Self {
        Self::Fixed(v)
    }
}

impl AdaptiveScalar {
    /// Create a fixed value
    pub fn fixed(val: f32) -> Self {
        Self::Fixed(val)
    }

    /// Create a learnable adaptive scalar initialized with Richards curve defaults
    pub fn learned_curve() -> Self {
        Self::Richards {
            curve: Box::new(RichardsCurve::new_learnable(
                crate::richards::Variant::Sigmoid,
            )),
            output_scale: 1.0,
        }
    }

    /// Get the current effective value for a given input signal `x`
    pub fn value(&self, x: f64) -> f32 {
        match self {
            Self::Fixed(v) => *v,
            Self::Richards { curve, output_scale } => {
                let (val, _) = curve.eval_scalar(x);
                (val as f32) * output_scale
            }
        }
    }

    /// Get learnable parameters (if any)
    pub fn parameters(&self) -> Vec<f64> {
        match self {
            Self::Fixed(_) => Vec::new(),
            Self::Richards { curve, .. } => curve.weights(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_adaptive_scalar_fixed(val in -100.0f32..100.0f32, progress in 0.0f64..1.0f64) {
            let scalar = AdaptiveScalar::fixed(val);
            let v = scalar.value(progress);
            prop_assert_eq!(v, val);
            prop_assert!(scalar.parameters().is_empty());
        }

        #[test]
        fn test_adaptive_scalar_richards_finite(progress in 0.0f64..1.0f64) {
            let scalar = AdaptiveScalar::learned_curve();
            let v = scalar.value(progress);
            prop_assert!(v.is_finite());
            
            // Default richards curve params
            let params = scalar.parameters();
            prop_assert!(!params.is_empty());
        }
    }

    #[test]
    fn test_adaptive_scalar_default() {
        let scalar = AdaptiveScalar::default();
        assert!(matches!(scalar, AdaptiveScalar::Fixed(1.0)));
        assert_eq!(scalar.value(0.5), 1.0);
    }

    #[test]
    fn test_adaptive_scalar_from_f32() {
        let scalar: AdaptiveScalar = 2.5.into();
        assert!(matches!(scalar, AdaptiveScalar::Fixed(2.5)));
        assert_eq!(scalar.value(0.9), 2.5);
    }
}
