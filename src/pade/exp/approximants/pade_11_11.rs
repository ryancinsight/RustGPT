use super::super::{PadeExp, utils::horner_iter};

impl PadeExp {
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn pade_exp_11_11(x: f64) -> f64 {
        const P_COEFFS: [f64; 12] = [
            1330243200.0,
            665121600.0,
            166280400.0,
            25004800.0,
            2333760.0,
            139776.0,
            5376.0,
            132.0,
            2.0,
            0.0,
            0.0,
            0.0,
        ];
        const Q_COEFFS: [f64; 12] = [
            1330243200.0,
            -665121600.0,
            166280400.0,
            -25004800.0,
            2333760.0,
            -139776.0,
            5376.0,
            -132.0,
            2.0,
            0.0,
            0.0,
            0.0,
        ];

        let p = horner_iter(&P_COEFFS, x);
        let q = horner_iter(&Q_COEFFS, x);
        p / q
    }
}
