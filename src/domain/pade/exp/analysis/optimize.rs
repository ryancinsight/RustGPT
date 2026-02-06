use super::super::PadeExp;

impl PadeExp {
    /// Comprehensive coefficient optimization using systematic testing.
    pub fn optimize_coefficients() -> String {
        let mut results = String::new();

        let current_7_7_error = Self::benchmark_accuracy(10000, (-0.4, 0.4));
        results.push_str(&format!(
            "[7/7] Current coefficients max error: {:.2e}\n",
            current_7_7_error
        ));

        let current_5_5_error = Self::benchmark_accuracy(10000, (-0.8, 0.8));
        results.push_str(&format!(
            "[5/5] Current coefficients max error: {:.2e}\n",
            current_5_5_error
        ));

        let current_3_3_error = Self::benchmark_accuracy(10000, (-1.2, 1.2));
        results.push_str(&format!(
            "[3/3] Current coefficients max error: {:.2e}\n",
            current_3_3_error
        ));

        let grad_test_points = [-0.3, -0.1, 0.0, 0.1, 0.3];
        let mut max_grad_error: f64 = 0.0;
        for &x in &grad_test_points {
            let pade_grad = Self::exp_grad(x);
            let true_grad = x.exp();
            let error = ((pade_grad - true_grad) / true_grad).abs();
            max_grad_error = max_grad_error.max(error);
        }
        results.push_str(&format!(
            "Gradient max relative error: {:.2e}\n",
            max_grad_error
        ));

        let perf_results = Self::performance_benchmark();
        results.push_str(&format!("\n{}", perf_results));

        results
    }

    /// Test optimal approximant selection for given precision requirements.
    pub fn test_optimal_selection(required_accuracy: f64) -> String {
        let test_ranges = [
            (-0.15, 0.15, "[11/11]"),
            (-0.2, 0.2, "[9/9]"),
            (-0.4, 0.4, "[7/7]"),
            (-0.8, 0.8, "[5/5]"),
            (-1.2, 1.2, "[3/3]"),
        ];

        let mut results = format!(
            "Optimal approximant selection for {:.0e} accuracy:\n",
            required_accuracy
        );

        for (min_x, max_x, name) in &test_ranges {
            let max_error = Self::benchmark_accuracy(1000, (*min_x, *max_x));
            let meets_requirement = max_error <= required_accuracy;

            results.push_str(&format!(
                "{}: error={:.2e}, meets_req={}\n",
                name, max_error, meets_requirement
            ));
        }

        results
    }
}
