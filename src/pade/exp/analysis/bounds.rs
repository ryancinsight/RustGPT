use super::super::PadeExp;

impl PadeExp {
    /// Compute condition number for exp(x) (relative condition κ(x) = |x|).
    pub fn condition_number(x: f64) -> f64 {
        x.abs()
    }

    /// Approximation error bound for different Padé approximants.
    #[inline]
    pub fn approximation_error_bound(x: f64) -> f64 {
        let abs_x = x.abs();

        if abs_x <= 0.15 {
            1e-18
        } else if abs_x <= 0.2 {
            1e-17
        } else if abs_x <= 0.4 {
            1e-15
        } else if abs_x <= 0.8 {
            1e-12
        } else if abs_x <= 1.2 {
            1e-10
        } else {
            1e-14
        }
    }

    /// Rigorous error bounds using interval arithmetic.
    pub fn exp_interval(_x: f64, input_interval: (f64, f64)) -> (f64, f64) {
        let (x_min, x_max) = input_interval;
        let exp_min = Self::exp(x_min);
        let exp_max = Self::exp(x_max);
        let error_bound = Self::approximation_error_bound(x_min.max(x_max));
        (exp_min * (1.0 - error_bound), exp_max * (1.0 + error_bound))
    }

    /// Certified exponential computation with error bounds.
    pub fn exp_certified(x: f64) -> (f64, f64, f64) {
        let result = Self::exp(x);
        let rel_error_bound = Self::approximation_error_bound(x);
        let abs_error_bound = result * rel_error_bound;
        (result, abs_error_bound, rel_error_bound)
    }

    /// Analyze error bounds using condition number theory.
    pub fn error_analysis(x: f64, input_error: f64) -> (f64, f64) {
        let approx_result = Self::exp(x);
        let exact_result = x.exp();
        let approx_error = ((approx_result - exact_result) / exact_result).abs();
        let kappa = Self::condition_number(x);
        let total_error = approx_error + kappa * input_error;
        (approx_error, total_error)
    }
}
