#[cfg(test)]
mod tests {
    use llm::domain::richards::RichardsCurve;
    use ndarray::Array2;

    fn get_curve() -> RichardsCurve {
        // Create a curve with gamma/bias enabled
        let mut curve = RichardsCurve::sigmoid(true);
        curve.gamma = Some(std::sync::Arc::new(Array2::from_elem((1, 128), 2.0f32))); // Gamma = 2.0
        curve.bias = Some(std::sync::Arc::new(Array2::from_elem((1, 128), 0.5f32))); // Bias = 0.5
        curve.gamma_learnable = true;
        curve.bias_learnable = true;
        curve
    }

    #[test]
    fn test_forward_matrix_correctness() {
        let curve = get_curve();
        let x = Array2::from_elem((10, 128), 0.1);
        let mut out = Array2::zeros((10, 128));

        curve.forward_matrix_into(&x, &mut out);

        let raw = curve.forward_scalar(0.1);
        let expected = raw * 2.0 + 0.5;

        assert!(
            (out[[0, 0]] - expected).abs() < 1e-6,
            "Forward mismatch: got {}, expected {}",
            out[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_forward_matrix_f32_correctness() {
        let curve = get_curve();
        let x = Array2::from_elem((10, 128), 0.1f32);
        let mut out = Array2::zeros((10, 128));

        curve.forward_matrix_f32_into(&x, &mut out);

        let raw = curve.forward_scalar_f32(0.1);
        let expected = raw * 2.0 + 0.5;

        assert!(
            (out[[0, 0]] - expected).abs() < 1e-6,
            "Forward f32 mismatch: got {}, expected {}",
            out[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_backward_matrix_correctness() {
        let curve = get_curve();
        let x = Array2::from_elem((2, 128), 0.1);
        let dy = Array2::from_elem((2, 128), 1.0);
        let mut dx = Array2::zeros((2, 128));

        curve.backward_matrix_into(&x, &dy, &mut dx);

        // Analytical: dL/dx = dL/dy * gamma * f'(x)
        // gamma = 2.0
        let raw_deriv = curve.derivative_scalar(0.1);
        let expected = 1.0 * 2.0 * raw_deriv;

        assert!(
            (dx[[0, 0]] - expected).abs() < 1e-6,
            "Backward mismatch: got {}, expected {}",
            dx[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_backward_matrix_f32_correctness() {
        let curve = get_curve();
        let x = Array2::from_elem((2, 128), 0.1f32);
        let dy = Array2::from_elem((2, 128), 1.0f32);
        let mut dx = Array2::zeros((2, 128));

        curve.backward_matrix_f32_into(&x, &dy, &mut dx);

        // Analytical: dL/dx = dL/dy * gamma * f'(x)
        // gamma = 2.0
        let raw_deriv = curve.derivative_scalar_f32(0.1);
        let expected = 1.0 * 2.0 * raw_deriv;

        assert!(
            (dx[[0, 0]] - expected).abs() < 1e-6,
            "Backward f32 mismatch: got {}, expected {}",
            dx[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_grad_weights_matrix_correctness() {
        let curve = get_curve();
        let x = Array2::from_elem((1, 128), 0.1);
        let dy = Array2::from_elem((1, 128), 1.0);

        let grads = curve.grad_weights_matrix(&x, &dy);

        // Scalar gradients should be computed using effective_dy = dy * gamma = 1.0 * 2.0 = 2.0.
        // We can verify this by computing gradients with gamma=1.0 and dy=2.0 and comparing.

        let curve_identity = RichardsCurve::sigmoid(true);
        // gamma is None by default (effectively 1.0)

        let grads_identity = curve_identity.grad_weights_matrix(&x, &(&dy * 2.0));

        // Compare the first few scalar parameters (nu, k, m, etc.)
        let len = curve.scalar_weights_len();
        for i in 0..len {
            assert!(
                (grads[i] - grads_identity[i]).abs() < 1e-6,
                "Scalar gradient {} mismatch",
                i
            );
        }
    }

    #[test]
    fn test_numerical_stability() {
        let mut stability = RichardsCurve::sigmoid(true);
        stability.gamma = Some(std::sync::Arc::new(Array2::from_elem((1, 128), 1.0f32)));
        stability.bias = Some(std::sync::Arc::new(Array2::from_elem((1, 128), 0.0f32)));
        stability.gamma_learnable = true;
        stability.bias_learnable = true;
    }
}
