use super::super::{utils::horner_iter, PadeExp};

impl PadeExp {
    #[inline]
    pub(crate) fn chebyshev_pade_3_3(x: f64) -> f64 {
        const P_COEFFS: [f64; 4] = [120.0, 60.0, 12.0, 1.0];
        const Q_COEFFS: [f64; 4] = [120.0, -60.0, 12.0, -1.0];

        let p = horner_iter(&P_COEFFS, x);
        let q = horner_iter(&Q_COEFFS, x);
        p / q
    }
}
