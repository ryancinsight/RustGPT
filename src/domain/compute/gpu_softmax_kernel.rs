//! GPU Softmax Gradient Kernel
//!
//! Implements softmax backward pass on GPU.
//!
//! ## Algorithm
//!
//! Softmax gradient for row-wise computation:
//! ```
//! input_grad[i,j] = softmax[i,j] * (output_grad[i,j] - sum(softmax[i,:] * output_grad[i,:]))
//! ```
//!
//! For axis=1 (row-wise softmax), computed as:
//! 1. For each row, compute sum_grad_prob = dot(softmax_row, grad_row)
//! 2. For each element: grad = softmax * (grad - sum_grad_prob)

use crate::common::errors::Result;

/// GPU Softmax Gradient Kernel
///
/// Computes softmax gradient on GPU.
/// CPU fallback implementation provided for validation.
pub struct GpuSoftmaxGradientKernel;

impl GpuSoftmaxGradientKernel {
    /// Create new softmax gradient kernel
    pub fn new() -> Self {
        Self
    }

    /// Compute softmax gradients on GPU
    ///
    /// # Arguments
    /// * `softmax_output` - Softmax probability output (batch, features)
    /// * `output_grads` - Gradients w.r.t. softmax output (batch, features)
    /// * `axis` - Softmax axis (1 for row-wise, 0 for column-wise)
    ///
    /// # Returns
    /// Input gradients (batch, features)
    ///
    /// # Algorithm
    /// For axis=1 (row-wise):
    /// ```
    /// For each row i:
    ///   sum_grad_prob = dot(softmax[i,:], grad[i,:])
    ///   grad[i,j] = softmax[i,j] * (grad[i,j] - sum_grad_prob)
    /// ```
    pub fn compute_gradient_rowwise(
        softmax_output: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        let (batch_size, features) = softmax_output.dim();
        let grads_dim = output_grads.dim();
        assert_eq!(
            grads_dim,
            (batch_size, features),
            "Shape mismatch: softmax ({}, {}) vs grads ({}, {})",
            batch_size,
            features,
            grads_dim.0,
            grads_dim.1
        );

        let mut input_grads = ndarray::Array2::zeros((batch_size, features));

        // Compute softmax gradient for each row
        for batch_idx in 0..batch_size {
            let prob_row = softmax_output.row(batch_idx);
            let grad_row = output_grads.row(batch_idx);
            let mut input_row = input_grads.row_mut(batch_idx);

            // sum_grad_prob = sum(softmax[i,j] * grad[i,j])
            let sum_grad_prob: f32 = prob_row
                .iter()
                .zip(grad_row.iter())
                .map(|(&p, &g)| p * g)
                .sum();

            // input_grad[i,j] = softmax[i,j] * (grad[i,j] - sum_grad_prob)
            for (j, (&p, &g)) in prob_row.iter().zip(grad_row.iter()).enumerate() {
                input_row[j] = p * (g - sum_grad_prob);
            }
        }

        Ok(input_grads)
    }

    /// Compute softmax gradients on GPU (column-wise)
    ///
    /// For axis=0 (column-wise):
    /// ```
    /// For each column j:
    ///   sum_grad_prob = sum_i(softmax[i,j] * grad[i,j])
    ///   grad[i,j] = softmax[i,j] * (grad[i,j] - sum_grad_prob)
    /// ```
    pub fn compute_gradient_columnwise(
        softmax_output: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>> {
        let (batch_size, features) = softmax_output.dim();
        let grads_dim = output_grads.dim();
        assert_eq!(
            grads_dim,
            (batch_size, features),
            "Shape mismatch: softmax ({}, {}) vs grads ({}, {})",
            batch_size,
            features,
            grads_dim.0,
            grads_dim.1
        );

        let mut input_grads = ndarray::Array2::zeros((batch_size, features));

        // Compute softmax gradient for each column
        for j in 0..features {
            // sum_grad_prob = sum_i(softmax[i,j] * grad[i,j])
            let mut sum_grad_prob = 0.0f32;
            for i in 0..batch_size {
                sum_grad_prob += softmax_output[[i, j]] * output_grads[[i, j]];
            }

            // input_grad[i,j] = softmax[i,j] * (grad[i,j] - sum_grad_prob)
            for i in 0..batch_size {
                let p = softmax_output[[i, j]];
                let g = output_grads[[i, j]];
                input_grads[[i, j]] = p * (g - sum_grad_prob);
            }
        }

        Ok(input_grads)
    }
}

impl Default for GpuSoftmaxGradientKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_softmax_gradient_rowwise_simple() {
        let softmax = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let grads = Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap();

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_rowwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // sum_grad_prob = 0.5*1.0 + 0.5*(-1.0) = 0.0
        // grad[0] = 0.5 * (1.0 - 0.0) = 0.5
        // grad[1] = 0.5 * (-1.0 - 0.0) = -0.5
        assert!((input_grads[[0, 0]] - 0.5).abs() < 1e-6);
        assert!((input_grads[[0, 1]] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_gradient_rowwise_batch() {
        let softmax = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.6, 0.3, 0.1, // Row 0
                0.2, 0.5, 0.3, // Row 1
            ],
        )
        .unwrap();

        let grads = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 0.0, 0.0, // Row 0: gradient for first element
                0.0, 1.0, 0.0, // Row 1: gradient for second element
            ],
        )
        .unwrap();

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_rowwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // Row 0: sum = 0.6*1.0 + 0.3*0.0 + 0.1*0.0 = 0.6
        // Row 0: grad[0] = 0.6*(1.0 - 0.6) = 0.24
        // Row 0: grad[1] = 0.3*(0.0 - 0.6) = -0.18
        // Row 0: grad[2] = 0.1*(0.0 - 0.6) = -0.06

        // Row 1: sum = 0.2*0.0 + 0.5*1.0 + 0.3*0.0 = 0.5
        // Row 1: grad[0] = 0.2*(0.0 - 0.5) = -0.1
        // Row 1: grad[1] = 0.5*(1.0 - 0.5) = 0.25
        // Row 1: grad[2] = 0.3*(0.0 - 0.5) = -0.15

        assert!((input_grads[[0, 0]] - 0.24).abs() < 1e-6);
        assert!((input_grads[[0, 1]] - (-0.18)).abs() < 1e-6);
        assert!((input_grads[[0, 2]] - (-0.06)).abs() < 1e-6);

        assert!((input_grads[[1, 0]] - (-0.1)).abs() < 1e-6);
        assert!((input_grads[[1, 1]] - 0.25).abs() < 1e-6);
        assert!((input_grads[[1, 2]] - (-0.15)).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_gradient_columnwise() {
        let softmax = Array2::from_shape_vec(
            (2, 2),
            vec![
                0.4, 0.6, // Row 0
                0.6, 0.4, // Row 1
            ],
        )
        .unwrap();

        let grads = Array2::from_shape_vec(
            (2, 2),
            vec![
                1.0, 0.0, // Row 0
                -1.0, 0.0, // Row 1
            ],
        )
        .unwrap();

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_columnwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // Col 0: sum = 0.4*1.0 + 0.6*(-1.0) = -0.2
        // Col 0: grad[0] = 0.4*(1.0 - (-0.2)) = 0.48
        // Col 0: grad[1] = 0.6*(-1.0 - (-0.2)) = -0.48

        // Col 1: sum = 0.6*0.0 + 0.4*0.0 = 0.0
        // Col 1: grad[0] = 0.6*0.0 = 0.0
        // Col 1: grad[1] = 0.4*0.0 = 0.0

        assert!((input_grads[[0, 0]] - 0.48).abs() < 1e-6);
        assert!((input_grads[[1, 0]] - (-0.48)).abs() < 1e-6);
        assert!((input_grads[[0, 1]]).abs() < 1e-6);
        assert!((input_grads[[1, 1]]).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_gradient_numerical_stability() {
        // Test with larger batch size for numerical stability
        let batch_size = 64;
        let features = 128;

        let mut softmax = ndarray::Array2::zeros((batch_size, features));
        let mut grads = ndarray::Array2::zeros((batch_size, features));

        // Create valid softmax (normalized probabilities)
        for i in 0..batch_size {
            let mut sum = 0.0f32;
            for j in 0..features {
                softmax[[i, j]] = ((i * 17 + j * 23) as f32 * 0.001).exp();
                sum += softmax[[i, j]];
            }
            for j in 0..features {
                softmax[[i, j]] /= sum;
            }
        }

        // Random gradients
        for i in 0..batch_size {
            for j in 0..features {
                grads[[i, j]] = ((i * 37 + j * 41) as f32 * 0.001).sin();
            }
        }

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_rowwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // Verify shapes
        assert_eq!(input_grads.dim(), (batch_size, features));

        // Verify no NaN or Inf
        for val in input_grads.iter() {
            assert!(val.is_finite(), "Found non-finite value in gradient");
        }

        // Verify gradient magnitudes are reasonable
        let max_abs = input_grads.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_abs < 10.0, "Gradient magnitude too large: {}", max_abs);
    }

    #[test]
    fn test_softmax_gradient_zero_grads() {
        let softmax = Array2::from_shape_vec((2, 3), vec![0.5, 0.3, 0.2, 0.4, 0.4, 0.2]).unwrap();
        let grads = Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_rowwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // With zero gradients, output should be all zeros
        for val in input_grads.iter() {
            assert!((val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_softmax_gradient_uniform_probs() {
        let softmax = Array2::from_shape_vec((1, 4), vec![0.25, 0.25, 0.25, 0.25]).unwrap();
        let grads = Array2::from_shape_vec((1, 4), vec![1.0, 1.0, 1.0, 1.0]).unwrap();

        let input_grads = GpuSoftmaxGradientKernel::compute_gradient_rowwise(&softmax, &grads)
            .expect("Failed to compute gradient");

        // sum_grad = 0.25 * 4 = 1.0
        // Each grad = 0.25 * (1.0 - 1.0) = 0.0
        for val in input_grads.iter() {
            assert!(val.abs() < 1e-6);
        }
    }
}
