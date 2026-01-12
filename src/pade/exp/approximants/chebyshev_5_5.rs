use super::super::{PadeExp, utils::horner_iter};

impl PadeExp {
    #[inline]
    pub(crate) fn chebyshev_pade_5_5(x: f64) -> f64 {
        const P_COEFFS: [f64; 6] = [30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0];
        const Q_COEFFS: [f64; 6] = [30240.0, -15120.0, 3360.0, -420.0, 30.0, -1.0];

        let p = horner_iter(&P_COEFFS, x);
        let q = horner_iter(&Q_COEFFS, x);
        p / q
    }
}
