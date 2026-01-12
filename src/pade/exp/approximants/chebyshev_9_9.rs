use super::super::{PadeExp, utils::horner_iter};

impl PadeExp {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn chebyshev_pade_9_9(x: f64) -> f64 {
        const P_COEFFS: [f64; 10] = [
            17643225600.0,
            8821612800.0,
            2205403200.0,
            330810240.0,
            31000704.0,
            1835008.0,
            69888.0,
            1584.0,
            20.0,
            1.0,
        ];
        const Q_COEFFS: [f64; 10] = [
            17643225600.0,
            -8821612800.0,
            2205403200.0,
            -330810240.0,
            31000704.0,
            -1835008.0,
            69888.0,
            -1584.0,
            20.0,
            -1.0,
        ];

        let p = horner_iter(&P_COEFFS, x);
        let q = horner_iter(&Q_COEFFS, x);
        p / q
    }
}
