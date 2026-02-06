use super::super::PadeExp;

impl PadeExp {
    /// Range reduction using binary exponent decomposition.
    #[inline]
    pub(crate) fn exp_range_reduction(x: f64) -> f64 {
        const LN2: f64 = std::f64::consts::LN_2;

        let k = (x / LN2).round() as i32;
        let r = (-(k as f64)).mul_add(LN2, x);

        let ln2_half = LN2 * 0.5;
        let (adjusted_k, adjusted_r) = if r >= ln2_half {
            (k + 1, r - LN2)
        } else if r < -ln2_half {
            (k - 1, r + LN2)
        } else {
            (k, r)
        };

        let abs_r = adjusted_r.abs();
        let exp_r = if abs_r <= 0.3 {
            Self::chebyshev_pade_7_7(adjusted_r)
        } else if abs_r <= 0.7 {
            Self::chebyshev_pade_5_5(adjusted_r)
        } else {
            Self::chebyshev_pade_3_3(adjusted_r)
        };

        Self::ldexp(exp_r, adjusted_k)
    }
}
