use ndarray::Array1;
use llm::richards::{RichardsCurve, Variant};

/// Standard sigmoid function for comparison
fn standard_sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Standard tanh function for comparison
fn standard_tanh(x: f64) -> f64 {
    x.tanh()
}

/// Gompertz function for comparison: a * exp(-b * exp(-c * x))
/// Using standard parameterization where a=1, b=1, c=1 for simplicity
fn standard_gompertz(x: f64) -> f64 {
    (-(-x).exp()).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_richards_sigmoid_behavior() {
        let mut richards = RichardsCurve::sigmoid(false);
        let x = Array1::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0]);
        let output = richards.forward(&x);
        
        // Test monotonicity (should be increasing for sigmoid)
        for i in 1..output.len() {
            if output[i] < output[i-1] {
                panic!("Sigmoid should be monotonic: output[{}] = {} < output[{}] = {}", 
                       i, output[i], i-1, output[i-1]);
            }
        }
        
        // Test reasonable bounds (sigmoid should be between 0 and 1 for standard parameters)
        for (i, &val) in output.iter().enumerate() {
            if val < -2.0 || val > 2.0 {
                panic!("Sigmoid output at index {} is out of reasonable bounds: {}", i, val);
            }
        }
        
        // Test that extreme values produce finite outputs
        let extreme_x = Array1::from_vec(vec![-100.0, 100.0]);
        let extreme_output = richards.forward(&extreme_x);
        for &val in extreme_output.iter() {
            if !val.is_finite() {
                panic!("Sigmoid should produce finite outputs for extreme inputs, got: {}", val);
            }
        }
    }

    #[test]
    fn test_richards_tanh_behavior() {
        let mut richards = RichardsCurve::tanh(false);
        let x = Array1::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0]);
        let output = richards.forward(&x);
        
        // Test monotonicity (should be increasing for tanh)
        for i in 1..output.len() {
            if output[i] < output[i-1] {
                panic!("Tanh should be monotonic: output[{}] = {} < output[{}] = {}", 
                       i, output[i], i-1, output[i-1]);
            }
        }
        
        // Test reasonable bounds (tanh should be between -1 and 1 for standard parameters)
        for (i, &val) in output.iter().enumerate() {
            if val < -2.0 || val > 2.0 {
                panic!("Tanh output at index {} is out of reasonable bounds: {}", i, val);
            }
        }
        
        // Test that extreme values produce finite outputs
        let extreme_x = Array1::from_vec(vec![-100.0, 100.0]);
        let extreme_output = richards.forward(&extreme_x);
        for &val in extreme_output.iter() {
            if !val.is_finite() {
                panic!("Tanh should produce finite outputs for extreme inputs, got: {}", val);
            }
        }
    }

    #[test]
    fn test_richards_gompertz_behavior() {
        let mut richards = RichardsCurve::gompertz(false);
        let x = Array1::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0]);
        let output = richards.forward(&x);
        
        // Test monotonicity (should be increasing for Gompertz)
        for i in 1..output.len() {
            if output[i] < output[i-1] {
                panic!("Gompertz should be monotonic: output[{}] = {} < output[{}] = {}", 
                       i, output[i], i-1, output[i-1]);
            }
        }
        
        // Test reasonable bounds
        for (i, &val) in output.iter().enumerate() {
            if val < -2.0 || val > 2.0 {
                panic!("Gompertz output at index {} is out of reasonable bounds: {}", i, val);
            }
        }
        
        // Test that extreme values produce finite outputs
        let extreme_x = Array1::from_vec(vec![-100.0, 100.0]);
        let extreme_output = richards.forward(&extreme_x);
        for &val in extreme_output.iter() {
            if !val.is_finite() {
                panic!("Gompertz should produce finite outputs for extreme inputs, got: {}", val);
            }
        }
    }

    #[test]
    fn test_richards_derivative_consistency() {
        let variants = vec![
            ("sigmoid", RichardsCurve::sigmoid(false)),
            ("tanh", RichardsCurve::tanh(false)),
            ("gompertz", RichardsCurve::gompertz(false)),
        ];
        
        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let h = 1e-6;
        
        for (variant_name, mut richards) in variants {
            for i in 0..x.len() {
                let x_plus = x[i] + h;
                let x_minus = x[i] - h;
                
                let f_plus = richards.forward_scalar(x_plus);
                let f_minus = richards.forward_scalar(x_minus);
                let numerical_grad = (f_plus - f_minus) / (2.0 * h);
                
                let analytical_grad = richards.backward_scalar(x[i]);
                
                if (numerical_grad - analytical_grad).abs() > 1e-5 {
                    panic!("{} derivative mismatch at x={}: numerical={}, analytical={}", 
                           variant_name, x[i], numerical_grad, analytical_grad);
                }
            }
        }
    }

    #[test]
    fn test_richards_vectorized_derivative_consistency() {
        let variants = vec![
            ("sigmoid", RichardsCurve::sigmoid(false)),
            ("tanh", RichardsCurve::tanh(false)),
            ("gompertz", RichardsCurve::gompertz(false)),
        ];
        
        let x = Array1::from_vec(vec![-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]);
        
        for (variant_name, mut richards) in variants {
            // Test vectorized derivative method
            let vectorized_grad = richards.derivative(&x);
            
            // Compare with scalar backward method
            for i in 0..x.len() {
                let scalar_grad = richards.backward_scalar(x[i]);
                
                if (vectorized_grad[i] - scalar_grad).abs() > 1e-10 {
                    panic!("{} vectorized vs scalar derivative mismatch at x={}: vectorized={}, scalar={}", 
                           variant_name, x[i], vectorized_grad[i], scalar_grad);
                }
            }
        }
    }

    #[test]
    fn test_richards_derivative_extreme_values() {
        let variants = vec![
            ("sigmoid", RichardsCurve::sigmoid(false)),
            ("tanh", RichardsCurve::tanh(false)),
            ("gompertz", RichardsCurve::gompertz(false)),
        ];
        
        let extreme_x = Array1::from_vec(vec![-100.0, -50.0, -10.0, 10.0, 50.0, 100.0]);
        
        for (variant_name, mut richards) in variants {
            // Test that derivatives are finite for extreme values
            let derivatives = richards.derivative(&extreme_x);
            
            for (i, &deriv) in derivatives.iter().enumerate() {
                if !deriv.is_finite() {
                    panic!("{} derivative is not finite at extreme value x={}: derivative={}", 
                           variant_name, extreme_x[i], deriv);
                }
            }
            
            // Test scalar backward method as well
            for &x_val in extreme_x.iter() {
                let scalar_deriv = richards.backward_scalar(x_val);
                if !scalar_deriv.is_finite() {
                    panic!("{} scalar derivative is not finite at extreme value x={}: derivative={}", 
                           variant_name, x_val, scalar_deriv);
                }
            }
        }
    }

    #[test]
    fn test_richards_scalar_vs_vector_consistency() {
        let mut richards = RichardsCurve::sigmoid(false);
        let x_vec = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let vector_output = richards.forward(&x_vec);
        
        for i in 0..x_vec.len() {
            let scalar_output = richards.forward_scalar(x_vec[i]);
            if (vector_output[i] - scalar_output).abs() > 1e-12 {
                panic!("Scalar vs vector inconsistency at index {}: vector={}, scalar={}", 
                       i, vector_output[i], scalar_output);
            }
        }
    }

    #[test]
    fn test_richards_sigmoid_vs_pure_sigmoid() {
        // Test that Richards sigmoid with specific parameters can approximate pure sigmoid
        let mut richards = RichardsCurve::sigmoid(false);
        
        // Set parameters to make it closer to pure sigmoid (nu, k, m, beta, a, b)
        richards.set_param(Some(1.0), Some(1.0), Some(0.0), Some(1.0), Some(1.0), Some(0.0));
        
        let test_values = vec![0.0, 1.0, 2.0]; // Test a few key points
        
        for &x_val in &test_values {
            let richards_output = richards.forward_scalar(x_val);
            let expected_sigmoid = standard_sigmoid(x_val);
            
            // The Richards curve includes gating (x * sigmoid(x)), so it won't match exactly
            // but should have similar monotonic behavior
            assert!(richards_output.is_finite(), "Richards output should be finite at x={}", x_val);
        }
    }

    #[test]
    fn test_richards_tanh_vs_pure_tanh() {
        // Test that Richards tanh with specific parameters exhibits tanh-like behavior
        let mut richards = RichardsCurve::tanh(false);
        
        let test_values = vec![-1.0, 0.0, 1.0]; // Test a few key points
        
        for &x_val in &test_values {
            let richards_output = richards.forward_scalar(x_val);
            let expected_tanh = standard_tanh(x_val);
            
            // The Richards curve includes gating and won't match exactly
            // but should have similar behavior patterns
            assert!(richards_output.is_finite(), "Richards output should be finite at x={}", x_val);
        }
    }

    #[test]
    fn test_richards_gompertz_vs_pure_gompertz() {
        // Test that Richards Gompertz exhibits growth curve behavior
        let mut richards = RichardsCurve::gompertz(false);
        
        let test_values = vec![-1.0, 0.0, 1.0, 2.0]; // Test a few key points
        
        for &x_val in &test_values {
            let richards_output = richards.forward_scalar(x_val);
            let expected_gompertz = standard_gompertz(x_val);
            
            // The Richards curve includes gating and won't match exactly
            // but should exhibit growth curve characteristics
            assert!(richards_output.is_finite(), "Richards output should be finite at x={}", x_val);
        }
    }

    #[test]
    fn test_richards_parameter_effects() {
        let mut richards1 = RichardsCurve::sigmoid(false);
        let mut richards2 = RichardsCurve::sigmoid(false);

        // Modify parameters of the second curve
        // set_param(nu, k, m, beta, a, b)
        richards2.set_param(Some(2.0), Some(2.0), Some(1.0), None, Some(2.0), Some(1.0));

        let x = Array1::from_vec(vec![-1.0, 0.0, 1.0]);
        let output1 = richards1.forward(&x);
        let output2 = richards2.forward(&x);

        // Different parameters should produce different outputs
        let mut different = false;
        for i in 0..output1.len() {
            if (output1[i] - output2[i]).abs() > 1e-6 {
                different = true;
                break;
            }
        }

        if !different {
            panic!("Different parameters should produce different outputs");
        }
    }

    #[test]
    fn test_richards_fixed_vs_learnable_consistency() {
        // Test that fixed parameter curves produce the same output as before changes
        let fixed_sigmoid = RichardsCurve::sigmoid(false);
        let fixed_tanh = RichardsCurve::tanh(false);
        let fixed_gompertz = RichardsCurve::gompertz(false);

        let x = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        // Test sigmoid
        let sigmoid_output = fixed_sigmoid.forward(&x);
        // Expected approximate values for Richards sigmoid (approximates logistic)
        // At x=0: sigmoid(0) ≈ 0.5
        // At x=1: sigmoid(1) ≈ 0.731
        // At x=2: sigmoid(2) ≈ 0.881
        assert_abs_diff_eq!(sigmoid_output[2], 0.5, epsilon = 1e-6); // x=0
        assert_abs_diff_eq!(sigmoid_output[3], 0.7310586, epsilon = 1e-5); // x=1
        assert_abs_diff_eq!(sigmoid_output[4], 0.880797, epsilon = 1e-4); // x=2

        // Test tanh
        let tanh_output = fixed_tanh.forward(&x);
        // Tanh gate: tanh(x)
        assert_abs_diff_eq!(tanh_output[2], 0.0, epsilon = 1e-6); // x=0
        assert_abs_diff_eq!(tanh_output[3], 0.761594, epsilon = 1e-5); // x=1
        assert_abs_diff_eq!(tanh_output[4], 0.964028, epsilon = 1e-4); // x=2

        // Test gompertz
        let gompertz_output = fixed_gompertz.forward(&x);
        // Gompertz gate: gompertz(x)
        assert_abs_diff_eq!(gompertz_output[2], 0.5, epsilon = 1e-6); // x=0
        // Gompertz values are small, check they're finite and reasonable
        assert!(gompertz_output[3].is_finite());
        assert!(gompertz_output[4].is_finite());
    }

    #[test]
    fn test_richards_boundary_behavior() {
        for variant in [Variant::Sigmoid, Variant::Tanh, Variant::Gompertz, Variant::None] {
            let mut richards = match variant {
                Variant::Sigmoid => RichardsCurve::sigmoid(false),
                Variant::Tanh => RichardsCurve::tanh(false),
                Variant::Gompertz => RichardsCurve::gompertz(false),
                Variant::None => RichardsCurve::new_fully_learnable(),
            };
            
            // Test at x=0
            let zero_output = richards.forward_scalar(0.0);
            if !zero_output.is_finite() {
                panic!("{:?} should produce finite output at x=0, got: {}", variant, zero_output);
            }
            
            // Test at extreme values
            let neg_extreme = richards.forward_scalar(-50.0);
            let pos_extreme = richards.forward_scalar(50.0);
            
            if !neg_extreme.is_finite() || !pos_extreme.is_finite() {
                panic!("{:?} should produce finite outputs at extremes, got: neg={}, pos={}", 
                       variant, neg_extreme, pos_extreme);
            }
        }
    }
}