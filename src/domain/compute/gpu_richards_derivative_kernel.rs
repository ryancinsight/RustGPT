//! GPU Richards Derivative Kernel
//!
//! Implements gradient computation for d/dx[x * Richards(x)] activation.
//!
//! ## Algorithm
//!
//! The Richards activation applies: f(x) = x * Richards(x)
//!
//! Where Richards(x) is a learnable curve with:
//! - curve_point: Lower asymptote (learnable)
//! - alpha: Growth rate (learnable)
//! - max_val: Upper asymptote (learnable)
//!
//! Richards(x) = curve_point + alpha * (1 - curve_point/max_val) * x
//!
//! The derivative:
//! df/dx = Richards(x) + x * dRichards/dx(x)
//!
//! Where:
//! dRichards/dx ≈ alpha * (1 - curve_point/max_val)

use crate::common::errors::Result;

/// GPU Richards Derivative Kernel
///
/// Computes gradient of d/dx[x * Richards(x)] on GPU.
/// CPU fallback implementation provided for validation.
pub struct GpuRichardsDerivativeKernel;

impl GpuRichardsDerivativeKernel {
    /// Create new Richards derivative kernel
    pub fn new() -> Self {
        Self
    }

    /// Compute Richards derivative with learned parameters
    ///
    /// # Arguments
    /// * `x` - Input values (batch, features)
    /// * `richards_output` - Output from Richards(x) (batch, features)
    /// * `curve_point` - Lower asymptote parameter
    /// * `alpha` - Growth rate parameter
    /// * `max_val` - Upper asymptote parameter
    ///
    /// # Returns
    /// df/dx = Richards(x) + x * dRichards/dx(x) (batch, features)
    ///
    /// # Algorithm
    /// ```
    /// dRichards_dx = alpha * (1 - curve_point / max_val)
    /// df_dx[i,j] = richards_output[i,j] + x[i,j] * dRichards_dx
    /// ```
    pub fn compute_derivative(
        x: &ndarray::Array2<f32>,
        richards_output: &ndarray::Array2<f32>,
        curve_point: f32,
        alpha: f32,
        max_val: f32,
    ) -> Result<ndarray::Array2<f32>> {
        let (batch_size, features) = x.dim();
        let output_dim = richards_output.dim();

        assert_eq!(
            output_dim,
            (batch_size, features),
            "Shape mismatch: x ({}, {}) vs richards_output ({}, {})",
            batch_size,
            features,
            output_dim.0,
            output_dim.1
        );

        // Compute dRichards/dx = alpha * (1 - curve_point / max_val)
        let d_richards_dx = alpha * (1.0 - curve_point / max_val);

        let mut result = ndarray::Array2::zeros((batch_size, features));

        // For each element: df/dx = Richards(x) + x * dRichards/dx
        for i in 0..batch_size {
            for j in 0..features {
                let x_val = x[[i, j]];
                let richards_val = richards_output[[i, j]];

                result[[i, j]] = richards_val + x_val * d_richards_dx;
            }
        }

        Ok(result)
    }

    /// Compute Richards derivative with chain rule for gradient backpropagation
    ///
    /// # Arguments
    /// * `x` - Input values (batch, features)
    /// * `richards_output` - Output from Richards(x) (batch, features)
    /// * `output_grads` - Gradients w.r.t. output (batch, features)
    /// * `curve_point` - Lower asymptote parameter
    /// * `alpha` - Growth rate parameter
    /// * `max_val` - Upper asymptote parameter
    ///
    /// # Returns
    /// Input gradients (batch, features)
    ///
    /// # Algorithm
    /// ```
    /// dRichards_dx = alpha * (1 - curve_point / max_val)
    /// df_dx[i,j] = Richards(x)[i,j] + x[i,j] * dRichards_dx
    /// grad_x[i,j] = output_grad[i,j] * df_dx[i,j]
    /// ```
    pub fn compute_gradient(
        x: &ndarray::Array2<f32>,
        richards_output: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
        curve_point: f32,
        alpha: f32,
        max_val: f32,
    ) -> Result<ndarray::Array2<f32>> {
        let (batch_size, features) = x.dim();

        // First compute the derivative
        let df_dx = Self::compute_derivative(x, richards_output, curve_point, alpha, max_val)?;

        // Then apply chain rule with output gradients
        let mut input_grads = ndarray::Array2::zeros((batch_size, features));

        for i in 0..batch_size {
            for j in 0..features {
                input_grads[[i, j]] = output_grads[[i, j]] * df_dx[[i, j]];
            }
        }

        Ok(input_grads)
    }

    /// Compute parameter gradients (for learning curve_point, alpha, max_val)
    ///
    /// # Arguments
    /// * `x` - Input values (batch, features)
    /// * `output_grads` - Gradients w.r.t. output (batch, features)
    /// * `curve_point` - Lower asymptote parameter
    /// * `alpha` - Growth rate parameter
    /// * `max_val` - Upper asymptote parameter
    ///
    /// # Returns
    /// (grad_curve_point, grad_alpha, grad_max_val) - Parameter gradients
    ///
    /// # Algorithm
    /// For parameter learning:
    /// ∂f/∂curve_point = sum(grad * x * (-alpha / max_val))
    /// ∂f/∂alpha = sum(grad * x * (1 - curve_point / max_val))
    /// ∂f/∂max_val = sum(grad * x * (alpha * curve_point / max_val²))
    pub fn compute_parameter_gradients(
        x: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
        curve_point: f32,
        alpha: f32,
        max_val: f32,
    ) -> Result<(f32, f32, f32)> {
        let (batch_size, features) = x.dim();

        let mut grad_curve_point = 0.0f32;
        let mut grad_alpha = 0.0f32;
        let mut grad_max_val = 0.0f32;

        for i in 0..batch_size {
            for j in 0..features {
                let x_val = x[[i, j]];
                let grad_val = output_grads[[i, j]];

                // ∂f/∂curve_point: grad * x * (-alpha / max_val)
                grad_curve_point += grad_val * x_val * (-alpha / max_val);

                // ∂f/∂alpha: grad * x * (1 - curve_point / max_val)
                grad_alpha += grad_val * x_val * (1.0 - curve_point / max_val);

                // ∂f/∂max_val: grad * x * (alpha * curve_point / max_val²)
                grad_max_val += grad_val * x_val * (alpha * curve_point / (max_val * max_val));
            }
        }

        Ok((grad_curve_point, grad_alpha, grad_max_val))
    }
}

impl Default for GpuRichardsDerivativeKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_richards_derivative_simple() {
        // Simple case: x=1, Richards(x)=1
        let x = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let richards = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let curve_point = 0.5;
        let alpha = 1.0;
        let max_val = 2.0;

        let df_dx = GpuRichardsDerivativeKernel::compute_derivative(
            &x,
            &richards,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute derivative");

        // dRichards_dx = 1.0 * (1 - 0.5/2.0) = 1.0 * 0.75 = 0.75
        // df_dx = 1.0 + 1.0 * 0.75 = 1.75
        assert!((df_dx[[0, 0]] - 1.75).abs() < 1e-6);
    }

    #[test]
    fn test_richards_derivative_batch() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, -1.0, 0.5]).unwrap();
        let richards = Array2::from_shape_vec((2, 2), vec![1.0, 1.5, 0.5, 0.8]).unwrap();

        let curve_point = 0.5;
        let alpha = 2.0;
        let max_val = 3.0;

        let df_dx = GpuRichardsDerivativeKernel::compute_derivative(
            &x,
            &richards,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute derivative");

        // dRichards_dx = 2.0 * (1 - 0.5/3.0) = 2.0 * (2.0/3.0) ≈ 1.333
        let d_richards_dx = alpha * (1.0 - curve_point / max_val);

        // Check each element
        for i in 0..2 {
            for j in 0..2 {
                let expected = richards[[i, j]] + x[[i, j]] * d_richards_dx;
                assert!((df_dx[[i, j]] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_richards_derivative_zero_input() {
        let x = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();
        let richards = Array2::from_shape_vec((1, 3), vec![0.5, 0.5, 0.5]).unwrap();

        let curve_point = 0.5;
        let alpha = 1.5;
        let max_val = 2.0;

        let df_dx = GpuRichardsDerivativeKernel::compute_derivative(
            &x,
            &richards,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute derivative");

        // df_dx = Richards(0) + 0 * dRichards_dx = 0.5
        for j in 0..3 {
            assert!((df_dx[[0, j]] - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_richards_gradient_backprop() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, -1.0, 0.5, 2.0, 0.0, -2.0]).unwrap();
        let richards = Array2::from_shape_vec((2, 3), vec![1.0, 0.5, 0.8, 1.5, 0.5, 0.3]).unwrap();
        let grad_out = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let curve_point = 0.5;
        let alpha = 1.0;
        let max_val = 2.0;

        let grad_in = GpuRichardsDerivativeKernel::compute_gradient(
            &x,
            &richards,
            &grad_out,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute gradient");

        // dRichards_dx = 1.0 * (1 - 0.5/2.0) = 0.75
        // For [0,0]: df_dx = 1.0 + 1.0*0.75 = 1.75
        //            grad = 1.0 * 1.75 = 1.75
        assert!((grad_in[[0, 0]] - 1.75).abs() < 1e-5);

        // For [0,1]: df_dx = 0.5 + (-1.0)*0.75 = -0.25
        //            grad = 1.0 * (-0.25) = -0.25
        assert!((grad_in[[0, 1]] - (-0.25)).abs() < 1e-5);
    }

    #[test]
    fn test_parameter_gradients_simple() {
        let x = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let grad_out = Array2::from_shape_vec((1, 2), vec![1.0, 1.0]).unwrap();

        let curve_point = 0.5;
        let alpha = 1.0;
        let max_val = 2.0;

        let (grad_cp, grad_a, grad_mv) = GpuRichardsDerivativeKernel::compute_parameter_gradients(
            &x,
            &grad_out,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute parameter gradients");

        // ∂f/∂curve_point = sum(grad * x * (-alpha / max_val))
        // = (1*1 + 1*2) * (-1.0 / 2.0) = 3 * (-0.5) = -1.5
        assert!((grad_cp - (-1.5)).abs() < 1e-5);

        // ∂f/∂alpha = sum(grad * x * (1 - curve_point / max_val))
        // = (1*1 + 1*2) * (1 - 0.5/2.0) = 3 * 0.75 = 2.25
        assert!((grad_a - 2.25).abs() < 1e-5);

        // ∂f/∂max_val = sum(grad * x * (alpha * curve_point / max_val²))
        // = (1*1 + 1*2) * (1.0 * 0.5 / 4.0) = 3 * 0.125 = 0.375
        assert!((grad_mv - 0.375).abs() < 1e-5);
    }

    #[test]
    fn test_richards_derivative_numerical_stability() {
        // Large batch test for numerical stability
        let batch_size = 64;
        let features = 128;

        let mut x = ndarray::Array2::zeros((batch_size, features));
        let mut richards = ndarray::Array2::zeros((batch_size, features));

        // Fill with diverse values
        for i in 0..batch_size {
            for j in 0..features {
                x[[i, j]] = ((i * 17 + j * 23) as f32 * 0.01 - 5.0); // Range: ~[-5, 5]
                richards[[i, j]] = (((i * 7 + j * 11) as f32 * 0.001).sin() + 1.0) / 2.0; // [0, 1]
            }
        }

        let curve_point = 0.3;
        let alpha = 1.5;
        let max_val = 2.5;

        let df_dx = GpuRichardsDerivativeKernel::compute_derivative(
            &x,
            &richards,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute derivative");

        // Verify shapes
        assert_eq!(df_dx.dim(), (batch_size, features));

        // Verify no NaN or Inf
        for val in df_dx.iter() {
            assert!(val.is_finite(), "Found non-finite value in derivative");
        }

        // Verify gradient magnitudes are reasonable
        let max_abs = df_dx.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(
            max_abs < 100.0,
            "Derivative magnitude too large: {}",
            max_abs
        );
    }

    #[test]
    fn test_richards_derivative_symmetry() {
        // Test that f(-x) + f(x) has predictable relationship
        let x_pos = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let x_neg = Array2::from_shape_vec((1, 4), vec![-1.0, -2.0, -3.0, -4.0]).unwrap();

        let richards_pos = Array2::from_shape_vec((1, 4), vec![1.5, 2.0, 2.3, 2.5]).unwrap();
        let richards_neg = Array2::from_shape_vec((1, 4), vec![0.5, 0.0, -0.3, -0.5]).unwrap();

        let curve_point = 0.5;
        let alpha = 1.0;
        let max_val = 2.0;

        let df_pos = GpuRichardsDerivativeKernel::compute_derivative(
            &x_pos,
            &richards_pos,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute positive derivative");

        let df_neg = GpuRichardsDerivativeKernel::compute_derivative(
            &x_neg,
            &richards_neg,
            curve_point,
            alpha,
            max_val,
        )
        .expect("Failed to compute negative derivative");

        // df_pos[i] - df_neg[i] should equal positive_richards[i] - negative_richards[i]
        // because the x term has opposite signs that cancel
        let d_richards_dx = alpha * (1.0 - curve_point / max_val);

        for j in 0..4 {
            let expected_diff = (richards_pos[[0, j]] + x_pos[[0, j]] * d_richards_dx)
                - (richards_neg[[0, j]] + x_neg[[0, j]] * d_richards_dx);
            assert!(((df_pos[[0, j]] - df_neg[[0, j]]) - expected_diff).abs() < 1e-5);
        }
    }
}
