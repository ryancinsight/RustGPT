use super::super::{utils::horner_iter, PadeExp};

impl PadeExp {
    #[inline]
    pub(crate) fn chebyshev_pade_7_7(x: f64) -> f64 {
        const P_COEFFS: [f64; 8] = [
            17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0,
        ];
        const Q_COEFFS: [f64; 8] = [
            17297280.0,
            -8648640.0,
            1995840.0,
            -277200.0,
            25200.0,
            -1512.0,
            56.0,
            -1.0,
        ];

        let p = horner_iter(&P_COEFFS, x);
        let q = horner_iter(&Q_COEFFS, x);
        p / q
    }
}
