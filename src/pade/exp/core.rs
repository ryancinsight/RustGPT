use super::{PadeExp, PrecisionLevel};

impl PadeExp {
    /// Lookup table for common exponential values to reduce computation.
    /// These values are exactly representable in IEEE 754 double precision.
    const COMMON_VALUES: [(f64, f64); 9] = [
        (0.0, 1.0),                    // exp(0) = 1
        (1.0, std::f64::consts::E),    // exp(1) = e
        (-1.0, 0.36787944117144233),   // exp(-1) = 1/e
        (2.0, 7.38905609893065),       // exp(2)
        (-2.0, 0.1353352832366127),    // exp(-2)
        (0.5, 1.648721271049738),      // exp(0.5)
        (-0.5, 0.6065306597126334),    // exp(-0.5)
        (std::f64::consts::LN_2, 2.0), // exp(ln(2)) = 2
        (-std::f64::consts::LN_2, 0.5),
    ];

    /// Optimized lookup for common exponential values.
    ///
    /// Note: these are exact IEEE-754 representable inputs/outputs, so we intentionally use
    /// exact equality (no tolerance) to avoid introducing discontinuities near the listed values.
    #[inline]
    fn lookup_common_exp(x: f64) -> Option<f64> {
        Self::COMMON_VALUES
            .iter()
            .find(|&&(val, _)| x == val)
            .map(|&(_, exp_val)| exp_val)
    }

    /// Compute stable exponential using Padé approximation with range reduction.
    #[inline]
    pub fn exp(x: f64) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }

        if x.is_infinite() {
            return if x.is_sign_positive() { f64::INFINITY } else { 0.0 };
        }

        // Underflow to 0 only below the smallest positive subnormal.
        if x < -745.133_219_101_941_1 {
            return 0.0;
        }

        // For very large positive values, return infinity to avoid overflow.
        if x > 709.782_712_893_384 {
            return f64::INFINITY;
        }

        if let Some(result) = Self::lookup_common_exp(x) {
            return result;
        }

        // Prefer a single accurate direct approximant in the non-reduced region.
        let abs_x = x.abs();
        if abs_x <= 1.2 {
            Self::chebyshev_pade_5_5(x)
        } else {
            Self::exp_range_reduction(x)
        }
    }

    /// Adaptive precision exponential computation with user-specified accuracy.
    #[inline]
    pub fn exp_with_precision(x: f64, precision: PrecisionLevel) -> f64 {
        if x.is_nan() {
            return f64::NAN;
        }

        if x.is_infinite() {
            return if x.is_sign_positive() { f64::INFINITY } else { 0.0 };
        }

        if x < -745.133_219_101_941_1 {
            return 0.0;
        }
        if x > 709.782_712_893_384 {
            return f64::INFINITY;
        }

        if let Some(result) = Self::lookup_common_exp(x) {
            return result;
        }

        let abs_x = x.abs();

        match precision {
            PrecisionLevel::QUANTUM => {
                if abs_x <= 1.2 {
                    Self::chebyshev_pade_7_7(x)
                } else {
                    Self::exp_range_reduction(x)
                }
            }
            PrecisionLevel::SUBATOMIC | PrecisionLevel::ATOMIC => {
                if abs_x <= 0.4 {
                    Self::chebyshev_pade_7_7(x)
                } else if abs_x <= 1.2 {
                    Self::chebyshev_pade_5_5(x)
                } else {
                    Self::exp_range_reduction(x)
                }
            }
            PrecisionLevel::MOLECULAR => {
                if abs_x <= 1.2 {
                    Self::chebyshev_pade_5_5(x)
                } else {
                    Self::exp_range_reduction(x)
                }
            }
            PrecisionLevel::MACROSCOPIC => {
                if abs_x <= 1.2 {
                    Self::chebyshev_pade_3_3(x)
                } else {
                    Self::exp_range_reduction(x)
                }
            }
        }
    }

    /// Modern Chebyshev-Padé approximation entry point (currently unified with `exp`).
    #[inline]
    pub fn exp_chebyshev_pade(x: f64) -> f64 {
        Self::exp(x)
    }

    /// Compute stable exp(-x).
    #[inline]
    pub fn exp_neg(x: f64) -> f64 {
        Self::exp(-x)
    }

    /// Stable gradient for exp(x).
    #[inline]
    pub fn exp_grad(x: f64) -> f64 {
        Self::exp(x)
    }

    /// Compute both value and gradient for exp(x).
    #[inline]
    pub fn exp_with_grad(x: f64) -> (f64, f64) {
        let value = Self::exp(x);
        let grad = Self::exp_grad(x);
        (value, grad)
    }
}
