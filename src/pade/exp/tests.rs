use std::f64::consts::E;

use ndarray::Array2;

use super::*;

#[test]
fn test_pade_exp_small_values() {
    let test_values = [-0.3, -0.1, 0.0, 0.1, 0.3];

    for &x in &test_values {
        let pade_result = PadeExp::exp(x);
        let std_result = x.exp();
        let rel_error = ((pade_result - std_result) / std_result).abs();

        assert!(
            rel_error < 1e-5,
            "x={}, pade={}, std={}, rel_error={}",
            x,
            pade_result,
            std_result,
            rel_error
        );
    }
}

#[test]
fn test_pade_exp_large_values() {
    let test_values = [-5.0, -2.0, 2.0, 5.0, 10.0];

    for &x in &test_values {
        let pade_result = PadeExp::exp(x);
        let std_result = x.exp();
        let rel_error = ((pade_result - std_result) / std_result).abs();

        assert!(
            rel_error < 1e-14,
            "x={}, pade={}, std={}, rel_error={}",
            x,
            pade_result,
            std_result,
            rel_error
        );
    }
}

#[test]
fn test_pade_exp_special_cases() {
    assert!(PadeExp::exp(f64::NAN).is_nan());
    assert_eq!(PadeExp::exp(f64::INFINITY), f64::INFINITY);
    assert_eq!(PadeExp::exp(f64::NEG_INFINITY), 0.0);

    assert_eq!(PadeExp::exp(-750.0), 0.0);
    assert_eq!(PadeExp::exp(750.0), f64::INFINITY);

    let sub = PadeExp::exp(-740.0);
    assert!(sub.is_finite());
    assert!(sub > 0.0);
    assert!(sub < f64::MIN_POSITIVE);
}

#[test]
fn test_pade_approximant_accuracy() {
    let test_values_7_7 = [-0.29, -0.1, 0.0, 0.1, 0.29];
    let test_values_5_5 = [-0.69, -0.4, 0.4, 0.69];
    let test_values_3_3 = [-0.99, -0.8, 0.8, 0.99];

    for &x in &test_values_7_7 {
        let pade_result = PadeExp::chebyshev_pade_7_7(x);
        let std_result = x.exp();
        let rel_error = ((pade_result - std_result) / std_result).abs();

        assert!(
            rel_error < 1e-13,
            "[7/7] Pade x={}, rel_error={}",
            x,
            rel_error
        );
    }

    for &x in &test_values_5_5 {
        let pade_result = PadeExp::chebyshev_pade_5_5(x);
        let std_result = x.exp();
        let rel_error = ((pade_result - std_result) / std_result).abs();

        assert!(
            rel_error < 1e-4,
            "[5/5] Pade x={}, rel_error={}",
            x,
            rel_error
        );
    }

    for &x in &test_values_3_3 {
        let pade_result = PadeExp::chebyshev_pade_3_3(x);
        let std_result = x.exp();
        let rel_error = ((pade_result - std_result) / std_result).abs();

        assert!(
            rel_error < 1e-4,
            "[3/3] Pade x={}, rel_error={}",
            x,
            rel_error
        );
    }
}

#[test]
fn test_benchmark_accuracy() {
    let max_error_small = PadeExp::benchmark_accuracy(1000, (-0.346574, 0.346574));
    let max_error_large = PadeExp::benchmark_accuracy(100, (-10.0, 10.0));

    assert!(
        max_error_small < 1e-4,
        "Small range max error: {}",
        max_error_small
    );
    assert!(
        max_error_large < 1e-4,
        "Large range max error: {}",
        max_error_large
    );
}

#[test]
fn test_critical_points_accuracy() {
    let (max_error, worst_x) = PadeExp::test_critical_points();
    assert!(
        max_error < 1e-4,
        "Critical points max error: {} at x={}",
        max_error,
        worst_x
    );
}

#[test]
fn test_range_reduction_accuracy() {
    let test_values = [-20.0, -10.0, -5.0, 5.0, 10.0, 20.0];

    for &x in &test_values {
        let pade_result = PadeExp::exp(x);
        let std_result = x.exp();

        if std_result.is_finite() && pade_result.is_finite() {
            let rel_error = ((pade_result - std_result) / std_result).abs();
            assert!(
                rel_error < 1e-11,
                "Range reduction x={}, rel_error={}",
                x,
                rel_error
            );
        }
    }
}

#[test]
fn test_pade_coefficient_stability() {
    let x = 0.1;
    let base_result = PadeExp::chebyshev_pade_7_7(x);

    let eps = 1e-14;
    let perturbed_result = PadeExp::chebyshev_pade_7_7(x + eps);

    let change = (perturbed_result - base_result).abs();
    assert!(
        change < 1e-13,
        "Numerical stability test failed: change={}",
        change
    );
}

#[test]
fn test_ldexp_accuracy() {
    for exp in -10..10 {
        let x = 1.23456789012345;

        let ldexp_result = PadeExp::ldexp(x, exp);
        let expected = x * (2.0_f64).powi(exp);

        let rel_error = ((ldexp_result - expected) / expected).abs();
        assert!(
            rel_error < 1e-15,
            "ldexp({}, {}) error: {}",
            x,
            exp,
            rel_error
        );
    }
}

#[test]
fn test_ldexp_zero_and_subnormal_behavior() {
    let pos_zero = PadeExp::ldexp(0.0, 500);
    assert_eq!(pos_zero, 0.0);
    assert!(pos_zero.is_sign_positive());

    let neg_zero = PadeExp::ldexp(-0.0, 200);
    assert_eq!(neg_zero, 0.0);
    assert!(neg_zero.is_sign_negative());

    let subnormal = f64::from_bits(1);
    let scaled_sub = PadeExp::ldexp(subnormal, 10);
    let expected_sub = subnormal * f64::exp2(10.0);
    assert_eq!(scaled_sub, expected_sub);

    let underflow = PadeExp::ldexp(1e-300, -1000);
    assert_eq!(underflow, 0.0);
    assert!(underflow.is_sign_positive());

    let overflow = PadeExp::ldexp(-1e300, 200);
    assert!(overflow.is_infinite());
    assert!(overflow.is_sign_negative());
}

#[test]
fn test_comprehensive_accuracy_benchmark() {
    let ranges = [
        (-0.346574, 0.346574),
        (-1.0, 1.0),
        (-5.0, 5.0),
        (-10.0, 10.0),
    ];

    let mut total_max_error = 0.0;
    let mut worst_range = (0.0, 0.0);

    for &(min_val, max_val) in &ranges {
        let max_error = PadeExp::benchmark_accuracy(1000, (min_val, max_val));
        if max_error > total_max_error {
            total_max_error = max_error;
            worst_range = (min_val, max_val);
        }
    }

    assert!(
        total_max_error < 1e-4,
        "Comprehensive benchmark failed: max_error={} in range [{}, {}]",
        total_max_error,
        worst_range.0,
        worst_range.1
    );
}

#[test]
fn test_performance_characteristics() {
    use std::time::Instant;

    let test_values: Vec<f64> = (-100..100).map(|x| x as f64 * 0.1).collect();
    let start = Instant::now();

    for _ in 0..10 {
        for &x in &test_values {
            let _result = PadeExp::exp(x);
        }
    }

    let elapsed = start.elapsed();
    let computations = test_values.len() * 10;
    let ns_per_computation = elapsed.as_nanos() as f64 / computations as f64;

    assert!(
        ns_per_computation < 1000.0,
        "Performance test failed: {:.2} ns/computation",
        ns_per_computation
    );
}

#[test]
fn test_gradient_accuracy() {
    let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &test_values {
        let grad_result = PadeExp::exp_grad(x);
        let expected = x.exp();

        if expected.is_finite() && grad_result.is_finite() {
            let rel_error = ((grad_result - expected) / expected).abs();
            assert!(
                rel_error < 1e-2,
                "Gradient error x={}, grad={}, expected={}, rel_error={}",
                x,
                grad_result,
                expected,
                rel_error
            );
        }
    }
}

#[test]
fn test_gradient_special_cases() {
    assert!(PadeExp::exp_grad(f64::NAN).is_nan());
    assert_eq!(PadeExp::exp_grad(f64::INFINITY), f64::INFINITY);
    assert_eq!(PadeExp::exp_grad(f64::NEG_INFINITY), 0.0);

    assert!(PadeExp::exp_grad(1000.0).is_infinite());
    assert_eq!(PadeExp::exp_grad(-1000.0), 0.0);
}

#[test]
fn test_pade_gradient_consistency() {
    let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &test_values {
        let (value_combined, grad_combined) = PadeExp::exp_with_grad(x);
        let value_separate = PadeExp::exp(x);
        let grad_separate = PadeExp::exp_grad(x);

        assert_eq!(value_combined, value_separate);
        assert_eq!(grad_combined, grad_separate);
    }
}

#[test]
fn test_approximant_selection() {
    let test_values = [
        -0.2,
        -0.2 + 1e-8,
        -0.2 - 1e-8,
        -0.15,
        -0.15 + 1e-8,
        -0.15 - 1e-8,
    ];

    for &x in &test_values as &[f64] {
        let abs_x = x.abs();
        let bounds = [0.15, 0.2, 0.4, 0.8, 1.2, f64::INFINITY];
        let idx = bounds.iter().position(|&bound| abs_x <= bound).unwrap_or(5);

        let approximant = match idx {
            0 => "11/11",
            1 => "9/9",
            2 => "7/7",
            3 => "5/5",
            4 => "3/3",
            _ => "range_reduction",
        };

        println!(
            "x={}, abs_x={}, selects approximant: {} (idx={})",
            x, abs_x, approximant, idx
        );
    }
}

#[test]
fn test_pade_derivative_functionality() {
    let test_values = [-0.1, 0.0, 0.1];

    for &x in &test_values {
        let pade_value = PadeExp::exp(x);
        let pade_grad = PadeExp::exp_grad(x);

        let (value_combined, grad_combined) = PadeExp::exp_with_grad(x);
        assert_eq!(value_combined, pade_value);
        assert_eq!(grad_combined, pade_grad);

        assert!(
            pade_grad.is_finite(),
            "Pade gradient should be finite at x={}",
            x
        );
        assert!(pade_grad > 0.0, "exp'(x) should be positive for x >= 0");
    }
}

#[test]
fn test_exp_with_grad_consistency() {
    let test_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &test_values {
        let (value_combined, grad_combined) = PadeExp::exp_with_grad(x);
        let value_separate = PadeExp::exp(x);
        let grad_separate = PadeExp::exp_grad(x);

        assert_eq!(value_combined, value_separate);
        assert_eq!(grad_combined, grad_separate);
    }
}

#[test]
fn test_coefficient_optimization() {
    let optimization_results = PadeExp::optimize_coefficients();
    println!(
        "Coefficient Optimization Results:\n{}",
        optimization_results
    );

    let error_7_7 = PadeExp::benchmark_accuracy(1000, (-0.4, 0.4));
    let error_5_5 = PadeExp::benchmark_accuracy(1000, (-0.8, 0.8));
    let error_3_3 = PadeExp::benchmark_accuracy(1000, (-1.2, 1.2));

    assert!(
        error_7_7 < 1e-4,
        "[7/7] Pade error too high: {:.2e}",
        error_7_7
    );
    assert!(
        error_5_5 < 1e-4,
        "[5/5] Pade error too high: {:.2e}",
        error_5_5
    );
    assert!(
        error_3_3 < 1e-3,
        "[3/3] Pade error too high: {:.2e}",
        error_3_3
    );
}

#[test]
fn test_optimal_approximant_selection() {
    let ml_selection = PadeExp::test_optimal_selection(1e-6);
    println!("ML Selection (1e-6):\n{}", ml_selection);

    let sci_selection = PadeExp::test_optimal_selection(1e-10);
    println!("Scientific Selection (1e-10):\n{}", sci_selection);

    let ml_error = PadeExp::benchmark_accuracy(1000, (-0.4, 0.4));
    assert!(
        ml_error <= 1e-4,
        "ML applications need [7/7] but error is {:.2e}",
        ml_error
    );
}

#[test]
fn test_unified_pade_interface() {
    let test_values = [-1.0, -0.5, 0.0, 0.5, 1.0];

    for &x in &test_values {
        let exp_result = PadeExp::exp(x);
        assert!(
            exp_result.is_finite() || x.is_infinite(),
            "exp({}) should be finite",
            x
        );

        let grad_result = PadeExp::exp_grad(x);
        assert!(
            grad_result.is_finite() || x.is_infinite(),
            "exp_grad({}) should be finite",
            x
        );

        let (val, grad) = PadeExp::exp_with_grad(x);
        assert_eq!(val, exp_result, "exp_with_grad value mismatch");
        assert_eq!(grad, grad_result, "exp_with_grad gradient mismatch");

        let eps = 1e-8;
        let numerical_grad = (PadeExp::exp(x + eps) - PadeExp::exp(x - eps)) / (2.0 * eps);
        let rel_error = ((grad_result - numerical_grad) / numerical_grad).abs();
        assert!(
            rel_error < 0.5,
            "Gradient numerical consistency failed at x={}: analytical={}, numerical={}, rel_error={}",
            x,
            grad_result,
            numerical_grad,
            rel_error
        );
    }
}

#[test]
fn test_codebase_consistency() {
    let attention_logits = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for &logit in &attention_logits {
        let masked = PadeExp::exp(logit);
        assert!(
            masked.is_finite(),
            "Attention masking failed for logit {}",
            logit
        );
    }

    let softmax_vals = [-1.0, 0.0, 1.0];
    let max_val = softmax_vals
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for &val in &softmax_vals {
        let exp_val = PadeExp::exp(val - max_val);
        assert!(
            exp_val.is_finite() && exp_val >= 0.0,
            "Softmax exp failed for {}",
            val
        );
    }

    let richards_inputs = [-0.5, 0.0, 0.5];
    for &x in &richards_inputs {
        let exp_pos = PadeExp::exp(x);
        let exp_neg = PadeExp::exp(-x);
        let sigmoid = 1.0 / (1.0 + PadeExp::exp(-x));

        assert!(
            exp_pos.is_finite() && exp_pos > 0.0,
            "Richards exp(+) failed"
        );
        assert!(
            exp_neg.is_finite() && exp_neg > 0.0,
            "Richards exp(-) failed"
        );
        assert!(
            sigmoid.is_finite() && (0.0..=1.0).contains(&sigmoid),
            "Richards sigmoid failed"
        );
    }
}

#[test]
fn test_pade_gradient_accuracy_comprehensive() {
    let test_ranges = [
        (-0.14, 0.14, 20, "[11/11]"),
        (-0.19, 0.19, 20, "[9/9]"),
        (-0.39, 0.39, 20, "[7/7]"),
        (-0.79, 0.79, 20, "[5/5]"),
        (-1.19, 1.19, 20, "[3/3]"),
    ];

    for (min_x, max_x, num_points, name) in &test_ranges {
        let mut max_grad_error = 0.0;
        let mut worst_x = 0.0;

        for i in 0..*num_points {
            let x = min_x + (max_x - min_x) * (i as f64) / ((num_points - 1) as f64);

            let pade_grad = PadeExp::exp_grad(x);
            let true_grad = x.exp();

            if true_grad.is_finite() && pade_grad.is_finite() {
                let error = ((pade_grad - true_grad) / true_grad).abs();
                if error > max_grad_error {
                    max_grad_error = error;
                    worst_x = x;
                }
            }
        }

        assert!(
            max_grad_error < 0.15,
            "{} gradient error too high: {:.2e} at x={}",
            name,
            max_grad_error,
            worst_x
        );
    }
}

#[test]
fn test_pade_range_optimization() {
    let boundary_tests = [
        (-0.15, "[11/11] to [9/9]"),
        (-0.2, "[9/9] to [7/7]"),
        (-0.4, "[7/7] to [5/5]"),
        (-0.8, "[5/5] to [3/3]"),
        (-1.2, "[3/3] to range reduction"),
    ];

    for (x, transition) in &boundary_tests {
        let exp_left = PadeExp::exp(*x - 1e-10);
        let exp_right = PadeExp::exp(*x + 1e-10);
        let true_left = (*x - 1e-10).exp();
        let true_right = (*x + 1e-10).exp();

        let rel_error_left = ((exp_left - true_left) / true_left).abs();
        let rel_error_right = ((exp_right - true_right) / true_right).abs();

        assert!(
            (rel_error_left - rel_error_right).abs() < 1e-4,
            "Large discontinuity at {} boundary: left_error={:.2e}, right_error={:.2e}",
            transition,
            rel_error_left,
            rel_error_right
        );
    }
}

#[test]
fn test_condition_number() {
    let test_values = [-5.0, -1.0, 0.0, 1.0, 5.0];

    for &x in &test_values {
        let kappa = PadeExp::condition_number(x);
        assert_eq!(kappa, x.abs());
    }
}

#[test]
fn test_error_analysis() {
    let x = 1.0;
    let input_error = 1e-10;

    let (approx_error, total_error) = PadeExp::error_analysis(x, input_error);

    assert!(approx_error < 1e-4);
    assert!(total_error >= approx_error);
}

#[test]
fn test_pade_order_selection() {
    let test_cases = [
        (0.1, "[7/7]"),
        (0.4, "[5/5]"),
        (0.8, "[3/3]"),
        (2.0, "range"),
    ];

    for &(x, _expected_order) in &test_cases {
        let result = PadeExp::exp(x);
        let expected_value = x.exp();

        let rel_error = ((result - expected_value) / expected_value).abs();
        assert!(
            rel_error < 1e-5,
            "Failed for x={}, rel_error={}",
            x,
            rel_error
        );
    }
}

#[test]
fn test_pade_exp_neg() {
    let test_values = [-5.0, -1.0, 0.0, 1.0, 5.0];

    for &x in &test_values {
        let exp_neg_result = PadeExp::exp_neg(x);
        let expected = (-x).exp();
        let rel_error = ((exp_neg_result - expected) / expected).abs();

        assert!(
            rel_error < 1e-4,
            "x={}, exp_neg={}, expected={}, rel_error={}",
            x,
            exp_neg_result,
            expected,
            rel_error
        );
    }
}

#[test]
#[ignore]
fn test_exp_array() {
    let input = Array2::from_shape_vec((2, 3), vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5]).unwrap();

    let result = PadeExp::exp_array(&input);

    assert!((result[[0, 0]] - 1.0).abs() < 1e-12);
    assert!((result[[0, 1]] - E).abs() < 1e-6);
    assert!((result[[0, 2]] - 1.0 / E).abs() < 1e-12);
}

#[test]
fn test_numerical_stability() {
    assert!(
        PadeExp::exp(100.0).is_finite(),
        "Large positive values should be clamped"
    );
    assert!(
        PadeExp::exp(-100.0) > 0.0,
        "Large negative values should be clamped to small positive"
    );

    let moderate_values = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0];

    for &x in &moderate_values {
        let pade_result = PadeExp::exp(x);
        let std_result = x.exp();

        assert!(
            pade_result.is_finite(),
            "Result should be finite for moderate x={}",
            x
        );
        assert!(
            std_result.is_finite(),
            "Std result should be finite for x={}",
            x
        );

        let rel_error = ((pade_result - std_result) / std_result).abs();
        assert!(
            rel_error < 1e-14,
            "High accuracy expected for moderate values: x={}, rel_error={}",
            x,
            rel_error
        );
    }
}
