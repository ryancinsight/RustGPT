use std::time::Instant;

use super::super::PadeExp;

impl PadeExp {
    /// Compare Padé approximation accuracy against std::exp.
    pub fn benchmark_accuracy(num_points: usize, range: (f64, f64)) -> f64 {
        let (min_val, max_val) = range;
        let step = (max_val - min_val) / (num_points as f64 - 1.0);

        let mut max_error: f64 = 0.0;

        for i in 0..num_points {
            let x = min_val + (i as f64) * step;
            let pade_result = Self::exp(x);
            let std_result = x.exp();

            if std_result.is_finite() && pade_result.is_finite() {
                let rel_error = ((pade_result - std_result) / std_result).abs();
                max_error = max_error.max(rel_error);
            }
        }

        max_error
    }

    /// Test numerical stability at critical points.
    pub fn test_critical_points() -> (f64, f64) {
        let critical_values = [
            -0.5,
            0.0,
            0.5,
            -std::f64::consts::LN_2,
            std::f64::consts::LN_2,
            -1.0,
            1.0,
            -2.0,
            2.0,
        ];

        let mut max_error = 0.0;
        let mut worst_x = 0.0;

        for &x in &critical_values {
            let pade_result = Self::exp(x);
            let std_result = x.exp();

            if std_result.is_finite() && pade_result.is_finite() {
                let rel_error = ((pade_result - std_result) / std_result).abs();
                if rel_error > max_error {
                    max_error = rel_error;
                    worst_x = x;
                }
            }
        }

        (max_error, worst_x)
    }

    /// Performance benchmark comparing different Padé orders.
    pub fn performance_benchmark() -> String {
        let test_values: Vec<f64> = (-50..50).map(|x| x as f64 * 0.02).collect();
        let iterations = 1000;

        let start = Instant::now();
        for _ in 0..iterations {
            for &x in &test_values {
                if x.abs() <= 0.3 {
                    let _ = Self::chebyshev_pade_7_7(x);
                }
            }
        }
        let time_7_7 = start.elapsed().as_nanos();

        let start = Instant::now();
        for _ in 0..iterations {
            for &x in &test_values {
                if x.abs() <= 0.7 {
                    let _ = Self::chebyshev_pade_5_5(x);
                }
            }
        }
        let time_5_5 = start.elapsed().as_nanos();

        let start = Instant::now();
        for _ in 0..iterations {
            for &x in &test_values {
                if x.abs() <= 1.0 {
                    let _ = Self::chebyshev_pade_3_3(x);
                }
            }
        }
        let time_3_3 = start.elapsed().as_nanos();

        let acc_7_7 = Self::benchmark_accuracy(1000, (-0.3, 0.3));
        let acc_5_5 = Self::benchmark_accuracy(1000, (-0.7, 0.7));
        let acc_3_3 = Self::benchmark_accuracy(1000, (-1.0, 1.0));

        format!(
            "Performance Benchmark Results:\n\
             [7/7] Pade: {:.2} ns/op, accuracy: {:.2e}\n\
             [5/5] Pade: {:.2} ns/op, accuracy: {:.2e}\n\
             [3/3] Pade: {:.2} ns/op, accuracy: {:.2e}",
            time_7_7 as f64 / (test_values.len() * iterations) as f64,
            acc_7_7,
            time_5_5 as f64 / (test_values.len() * iterations) as f64,
            acc_5_5,
            time_3_3 as f64 / (test_values.len() * iterations) as f64,
            acc_3_3
        )
    }
}
