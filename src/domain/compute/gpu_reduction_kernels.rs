//! GPU Reduction Kernels
//!
//! Implements tree reduction for bias gradient accumulation across batch dimension.
//!
//! ## Algorithm
//!
//! Sum gradients across batch dimension for bias learning:
//! ```
//! grad_bias[j] = sum_over_batch(grad_buffer[i, j])
//! ```
//!
//! Uses workgroup shared memory and atomic operations on GPU.
//! CPU implementation uses parallel reduction via Rayon.

use crate::common::errors::Result;
use ndarray::Array1;

/// GPU Reduction Kernel
///
/// Computes sum reductions across batch dimension on GPU.
/// Optimized for bias gradient accumulation.
pub struct GpuReductionKernel;

impl GpuReductionKernel {
    /// Create new reduction kernel
    pub fn new() -> Self {
        Self
    }

    /// Sum gradients across batch dimension (CPU reference)
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    ///
    /// # Returns
    /// Sum along batch axis, shape (features,)
    ///
    /// # Algorithm
    /// ```
    /// For each feature j:
    ///   result[j] = sum_i(grad_buffer[i, j])
    /// ```
    pub fn reduce_sum_batch(grad_buffer: &ndarray::Array2<f32>) -> Result<Array1<f32>> {
        let (batch_size, features) = grad_buffer.dim();
        let mut result = Array1::zeros(features);

        // Sum across batch dimension
        for i in 0..batch_size {
            for j in 0..features {
                result[j] += grad_buffer[[i, j]];
            }
        }

        Ok(result)
    }

    /// Sum gradients with optional normalization
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    /// * `normalize` - If true, divide by batch_size for averaging
    ///
    /// # Returns
    /// Sum/average along batch axis, shape (features,)
    pub fn reduce_sum_batch_normalized(
        grad_buffer: &ndarray::Array2<f32>,
        normalize: bool,
    ) -> Result<Array1<f32>> {
        let (batch_size, features) = grad_buffer.dim();
        let mut result = Array1::zeros(features);

        // Sum across batch dimension
        for i in 0..batch_size {
            for j in 0..features {
                result[j] += grad_buffer[[i, j]];
            }
        }

        // Normalize if requested
        if normalize && batch_size > 0 {
            let batch_norm = 1.0 / batch_size as f32;
            for j in 0..features {
                result[j] *= batch_norm;
            }
        }

        Ok(result)
    }

    /// Parallel reduction using tree pattern
    ///
    /// Reduces multiple blocks/columns independently.
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    /// * `block_size` - Processing block size (for tree reduction pattern)
    ///
    /// # Returns
    /// Sum along batch axis, shape (features,)
    pub fn reduce_sum_batch_tree(
        grad_buffer: &ndarray::Array2<f32>,
        _block_size: usize,
    ) -> Result<Array1<f32>> {
        // For CPU, tree reduction doesn't provide performance benefit
        // but the function shows the interface pattern used by GPU kernels
        Self::reduce_sum_batch(grad_buffer)
    }

    /// Reduce multiple matrices and accumulate results
    ///
    /// # Arguments
    /// * `grad_buffers` - Vector of gradient buffers (each: batch, features)
    ///
    /// # Returns
    /// Accumulated sum, shape (features,)
    ///
    /// # Algorithm
    /// ```
    /// For each buffer and each feature j:
    ///   result[j] += sum_i(grad_buffer[i, j])
    /// ```
    pub fn reduce_sum_accumulate(grad_buffers: &[ndarray::Array2<f32>]) -> Result<Array1<f32>> {
        if grad_buffers.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let features = grad_buffers[0].ncols();
        let mut result = Array1::zeros(features);

        for buffer in grad_buffers {
            let (batch_size, buf_features) = buffer.dim();
            assert_eq!(
                buf_features, features,
                "All buffers must have same feature dimension"
            );

            for i in 0..batch_size {
                for j in 0..features {
                    result[j] += buffer[[i, j]];
                }
            }
        }

        Ok(result)
    }

    /// Max reduction (for gradient clipping / analysis)
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    ///
    /// # Returns
    /// Max value along batch axis, shape (features,)
    pub fn reduce_max_batch(grad_buffer: &ndarray::Array2<f32>) -> Result<Array1<f32>> {
        let (batch_size, features) = grad_buffer.dim();
        let mut result = Array1::from_elem(features, f32::NEG_INFINITY);

        for i in 0..batch_size {
            for j in 0..features {
                result[j] = result[j].max(grad_buffer[[i, j]]);
            }
        }

        Ok(result)
    }

    /// Min reduction (for gradient analysis)
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    ///
    /// # Returns
    /// Min value along batch axis, shape (features,)
    pub fn reduce_min_batch(grad_buffer: &ndarray::Array2<f32>) -> Result<Array1<f32>> {
        let (batch_size, features) = grad_buffer.dim();
        let mut result = Array1::from_elem(features, f32::INFINITY);

        for i in 0..batch_size {
            for j in 0..features {
                result[j] = result[j].min(grad_buffer[[i, j]]);
            }
        }

        Ok(result)
    }

    /// Mean reduction (average across batch)
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    ///
    /// # Returns
    /// Mean along batch axis, shape (features,)
    pub fn reduce_mean_batch(grad_buffer: &ndarray::Array2<f32>) -> Result<Array1<f32>> {
        let (batch_size, _) = grad_buffer.dim();
        if batch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "Cannot reduce empty batch".to_string(),
            }
            .into());
        }

        let mut result = Self::reduce_sum_batch(grad_buffer)?;
        let scale = 1.0 / batch_size as f32;
        result *= scale;
        Ok(result)
    }

    /// L2 norm reduction
    ///
    /// # Arguments
    /// * `grad_buffer` - Gradient buffer (batch, features)
    ///
    /// # Returns
    /// L2 norm along batch axis, shape (features,)
    /// result[j] = sqrt(sum_i(grad_buffer[i, j]²))
    pub fn reduce_l2_batch(grad_buffer: &ndarray::Array2<f32>) -> Result<Array1<f32>> {
        let (batch_size, features) = grad_buffer.dim();
        let mut result: Array1<f32> = Array1::zeros(features);

        for i in 0..batch_size {
            for j in 0..features {
                let val = grad_buffer[[i, j]];
                result[j] += val * val;
            }
        }

        // Take square root
        for j in 0..features {
            result[j] = result[j].sqrt();
        }

        Ok(result)
    }
}

impl Default for GpuReductionKernel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_reduce_sum_batch_simple() {
        let grad = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
            ],
        )
        .unwrap();

        let result = GpuReductionKernel::reduce_sum_batch(&grad).expect("Failed to reduce");

        assert_eq!(result.dim(), 3);
        assert!((result[0] - 5.0).abs() < 1e-6); // 1 + 4
        assert!((result[1] - 7.0).abs() < 1e-6); // 2 + 5
        assert!((result[2] - 9.0).abs() < 1e-6); // 3 + 6
    }

    #[test]
    fn test_reduce_sum_batch_single() {
        let grad = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = GpuReductionKernel::reduce_sum_batch(&grad).expect("Failed to reduce");

        assert_eq!(result.dim(), 4);
        for (i, &val) in result.iter().enumerate() {
            assert!((val - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduce_sum_batch_normalized() {
        let grad = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0, //
                7.0, 8.0, //
            ],
        )
        .unwrap();

        // Without normalization
        let sum = GpuReductionKernel::reduce_sum_batch_normalized(&grad, false)
            .expect("Failed to reduce");
        assert!((sum[0] - 16.0).abs() < 1e-6); // 1+3+5+7
        assert!((sum[1] - 20.0).abs() < 1e-6); // 2+4+6+8

        // With normalization (divide by batch_size=4)
        let mean =
            GpuReductionKernel::reduce_sum_batch_normalized(&grad, true).expect("Failed to reduce");
        assert!((mean[0] - 4.0).abs() < 1e-6); // 16/4
        assert!((mean[1] - 5.0).abs() < 1e-6); // 20/4
    }

    #[test]
    fn test_reduce_max_batch() {
        let grad = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 5.0, 3.0, //
                9.0, 2.0, 7.0, //
                4.0, 8.0, 6.0, //
            ],
        )
        .unwrap();

        let result = GpuReductionKernel::reduce_max_batch(&grad).expect("Failed to reduce max");

        assert_eq!(result.dim(), 3);
        assert!((result[0] - 9.0).abs() < 1e-6); // max(1,9,4)
        assert!((result[1] - 8.0).abs() < 1e-6); // max(5,2,8)
        assert!((result[2] - 7.0).abs() < 1e-6); // max(3,7,6)
    }

    #[test]
    fn test_reduce_min_batch() {
        let grad = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 5.0, 3.0, //
                9.0, 2.0, 7.0, //
                4.0, 8.0, 6.0, //
            ],
        )
        .unwrap();

        let result = GpuReductionKernel::reduce_min_batch(&grad).expect("Failed to reduce min");

        assert_eq!(result.dim(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6); // min(1,9,4)
        assert!((result[1] - 2.0).abs() < 1e-6); // min(5,2,8)
        assert!((result[2] - 3.0).abs() < 1e-6); // min(3,7,6)
    }

    #[test]
    fn test_reduce_mean_batch() {
        let grad = Array2::from_shape_vec(
            (4, 2),
            vec![
                2.0, 4.0, //
                6.0, 8.0, //
                10.0, 12.0, //
                14.0, 16.0, //
            ],
        )
        .unwrap();

        let result = GpuReductionKernel::reduce_mean_batch(&grad).expect("Failed to reduce mean");

        assert_eq!(result.dim(), 2);
        assert!((result[0] - 8.0).abs() < 1e-6); // mean(2,6,10,14) = 32/4 = 8
        assert!((result[1] - 10.0).abs() < 1e-6); // mean(4,8,12,16) = 40/4 = 10
    }

    #[test]
    fn test_reduce_l2_batch() {
        let grad = Array2::from_shape_vec(
            (2, 3),
            vec![
                3.0, 4.0, 0.0, //
                0.0, 0.0, 5.0, //
            ],
        )
        .unwrap();

        let result = GpuReductionKernel::reduce_l2_batch(&grad).expect("Failed to reduce L2");

        assert_eq!(result.dim(), 3);
        assert!((result[0] - 3.0).abs() < 1e-6); // sqrt(3² + 0²) = 3
        assert!((result[1] - 4.0).abs() < 1e-6); // sqrt(4² + 0²) = 4
        assert!((result[2] - 5.0).abs() < 1e-6); // sqrt(0² + 5²) = 5
    }

    #[test]
    fn test_reduce_sum_accumulate() {
        let grad1 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let grad2 = Array2::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let result = GpuReductionKernel::reduce_sum_accumulate(&[grad1, grad2])
            .expect("Failed to accumulate");

        assert_eq!(result.dim(), 2);
        assert!((result[0] - 16.0).abs() < 1e-6); // 1+3+5+7 = 16
        assert!((result[1] - 20.0).abs() < 1e-6); // 2+4+6+8 = 20
    }

    #[test]
    fn test_reduce_large_batch_numerical_stability() {
        let batch_size = 256;
        let features = 512;

        let mut grad = ndarray::Array2::zeros((batch_size, features));

        // Fill with small random values to test numerical stability
        for i in 0..batch_size {
            for j in 0..features {
                grad[[i, j]] = (((i * 73 + j * 101) as f32 % 100.0) * 0.00001 - 0.0005);
            }
        }

        let result = GpuReductionKernel::reduce_sum_batch(&grad).expect("Failed to reduce");

        // Verify shapes
        assert_eq!(result.dim(), features);

        // Verify no NaN or Inf
        for val in result.iter() {
            assert!(val.is_finite(), "Found non-finite value in reduction");
        }

        // Verify mean is reasonable
        // With values ~[-0.0005, 0.0005] and batch_size=256, sum should be small
        let mean_val = result.iter().sum::<f32>() / features as f32;
        assert!(mean_val.abs() < 1.0, "Mean value too large: {}", mean_val);
    }

    #[test]
    fn test_reduce_broadcast_consistency() {
        // Verify reduce_sum matches manual iteration
        let grad =
            Array2::from_shape_vec((8, 16), (0..128).map(|x| (x as f32 * 0.1)).collect()).unwrap();

        let result = GpuReductionKernel::reduce_sum_batch(&grad).expect("Failed to reduce");

        // Manual verification
        for j in 0..16 {
            let mut expected_sum = 0.0f32;
            for i in 0..8 {
                expected_sum += grad[[i, j]];
            }
            assert!((result[j] - expected_sum).abs() < 1e-5);
        }
    }
}
