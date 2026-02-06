#[inline]
pub(super) fn horner_iter(coeffs: &[f64], x: f64) -> f64 {
    // Reverse coefficients -> accumulate via Horner with FMA when available
    coeffs.iter().rev().fold(0.0, |acc, &c| acc.mul_add(x, c))
}
