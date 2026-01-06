use super::super::PadeExp;

impl PadeExp {
    /// Efficient scaling by powers of 2 using bit manipulation.
    #[inline]
    pub(crate) fn ldexp(x: f64, exp: i32) -> f64 {
        if x == 0.0 || exp == 0 {
            return x;
        }

        let bits = x.to_bits();
        let exponent = ((bits >> 52) & 0x7FF) as i32;

        // Subnormal inputs fall back to exp2-based scaling because they lack an implicit leading 1
        if exponent == 0 {
            return Self::ldexp_fallback(x, exp);
        }

        let new_exp = exponent + exp;
        if !(1..0x7FF).contains(&new_exp) {
            return Self::ldexp_fallback(x, exp);
        }

        let cleared = bits & 0x800F_FFFF_FFFF_FFFF; // Preserve sign/mantissa, clear exponent bits
        let new_bits = cleared | ((new_exp as u64) << 52);
        f64::from_bits(new_bits)
    }

    #[inline]
    fn ldexp_fallback(x: f64, exp: i32) -> f64 {
        let scaled = x * f64::exp2(exp as f64);
        if scaled == 0.0 {
            0.0f64.copysign(x)
        } else {
            scaled
        }
    }
}
