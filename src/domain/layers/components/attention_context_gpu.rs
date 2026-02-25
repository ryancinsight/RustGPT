//! GPU-Optimized Attention Context Operations (Phase 5.6+)
//!
//! Implements GPU acceleration for SharedAttentionContext operations with
//! unified buffer management and strict GPU detection.
//!
//! ## Consolidated GPU Backend
//!
//! This module uses `UnifiedGpuBackend` for all GPU operations, providing:
//! - Automatic GPU detection with strict no-fallback semantics
//! - Unified memory management across all architectures
//! - Fused kernel dispatch for attention context operations

use ndarray::{Array2, ArrayView2};

use crate::common::errors::{ModelError, Result};
use crate::domain::layers::components::attention_context::SharedAttentionContext;
use crate::domain::layers::components::unified_gpu_backend::UnifiedGpuBackend;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::linalg::general_mat_mul;

/// GPU implementation helpers for attention context operations
impl SharedAttentionContext {
    /// GPU path for applying incoming context
    ///
    /// Applies the learned similarity context to modulate attention patterns.
    /// This operation is relatively lightweight (matrix multiplication) and benefits
    /// from GPU acceleration in larger batches.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::Backend` if GPU is not available.
    /// This method does NOT fall back to CPU - use CPU methods explicitly if needed.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn apply_incoming_context_gpu(
        &self,
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        // Verify dimensions are compatible first (cheap check)
        let (_batch_size, embed_dim) = input.dim();
        let (context_rows, context_cols) = self.similarity_context_strength.dim();

        if context_rows != embed_dim || context_cols != embed_dim {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Context strength dimension mismatch: expected ({}, {}), got ({}, {})",
                    embed_dim, embed_dim, context_rows, context_cols
                ),
            });
        }

        // Use the unified GPU backend for the matrix multiplication
        // output = input @ context_strength
        backend.forward_attention_context(input, &self.similarity_context_strength, 1.0)
    }

    /// GPU path for applying incoming context (stub for non-GPU builds)
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn apply_incoming_context_gpu(
        &self,
        _input: &Array2<f32>,
        _backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    /// GPU path for updating outgoing context
    ///
    /// Computes the activation similarity matrix from paired input/output data.
    /// This is more compute-intensive and benefits significantly from GPU acceleration.
    ///
    /// Full GPU kernel implementation:
    /// 1. Upload input and output to GPU
    /// 2. Compute covariance: input.T @ output / batch_size (GEMM on GPU)
    /// 3. Apply exponential moving average with update_rate
    /// 4. Download result and update internal state
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn update_outgoing_context_gpu(
        &mut self,
        input: &ArrayView2<f32>,
        output: &ArrayView2<f32>,
        update_rate: f32,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input.ncols().min(output.ncols());

        if seq_len == 0 || embed_dim == 0 {
            return Err(ModelError::InvalidInput {
                message: "Input/output must have non-zero dimensions".to_string(),
            });
        }

        // GPU path for context update:
        // 1. Normalize input and output (optional, can be done on GPU)
        // 2. Compute covariance: input.T @ output / batch_size (full GPU GEMM)
        // 3. Apply exponential moving average with update_rate
        // 4. Return updated context
        //
        // Full GPU implementation:
        // covariance = (input.T @ output) / batch_size
        // This is: (embed_dim, seq_len) @ (seq_len, embed_dim) -> (embed_dim, embed_dim)
        
        // Get GPU device from backend
        let device = backend.device();
        let mut device = device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock GPU device in update_outgoing_context_gpu".to_string(),
        })?;

        // Allocate GPU buffers
        let input_size = seq_len * embed_dim * std::mem::size_of::<f32>();
        let output_size = seq_len * embed_dim * std::mem::size_of::<f32>();
        let cov_size = embed_dim * embed_dim * std::mem::size_of::<f32>();
        
        let mut input_buf = device.allocate(input_size)?;
        let mut output_buf = device.allocate(output_size)?;
        let mut cov_buf = device.allocate(cov_size)?;

        // Upload input and output
        // Note: input is (seq_len, embed_dim), we need it transposed for GEMM
        // But GEMM supports transpose flags, so we upload as-is and use transposed operation
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;
        device.upload(output.as_slice().unwrap(), &mut output_buf)?;

        // Compute covariance on GPU: cov = input.T @ output
        // GEMM: C = alpha * A @ B + beta * C
        // A = input (seq_len, embed_dim), B = output (seq_len, embed_dim)
        // We want: cov = input.T @ output
        // So: A^T @ B with A = input, B = output
        // trans_a = true, trans_b = false
        device.gemm_f32(
            1.0 / (seq_len as f32),  // alpha = 1/batch_size
            &input_buf,
            &output_buf,
            0.0,
            &mut cov_buf,
            embed_dim,    // m = embed_dim (output rows)
            embed_dim,    // n = embed_dim (output cols) 
            seq_len,      // k = seq_len (input cols = output rows)
            true,         // trans_a = true (input.T)
            false,        // trans_b = false
        )?;

        // Download result
        let mut cov = Array2::<f32>::zeros((embed_dim, embed_dim));
        device.download(&cov_buf, cov.as_slice_mut().unwrap())?;

        // Cleanup GPU buffers
        device.deallocate(input_buf);
        device.deallocate(output_buf);
        device.deallocate(cov_buf);

        // Apply update rate (EMA)
        cov.mapv_inplace(|x| x * update_rate);

        // Update internal state
        self.similarity_context_strength =
            &self.similarity_context_strength * (1.0 - update_rate) + &cov;

        Ok(self.similarity_context_strength.clone())
    }

    /// GPU path for updating outgoing context (stub for non-GPU builds)
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn update_outgoing_context_gpu(
        &mut self,
        _input: &ArrayView2<f32>,
        _output: &ArrayView2<f32>,
        _update_rate: f32,
        _backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    /// GPU-accelerated forward with automatic backend detection
    ///
    /// Creates a GPU backend automatically using strict detection (no fallback).
    ///
    /// # Errors
    ///
    /// Returns error if no GPU is detected or GPU initialization fails.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu_auto(
        input: &Array2<f32>,
        context: &Array2<f32>,
        strength: f32,
    ) -> Result<Array2<f32>> {
        let mut backend = UnifiedGpuBackend::auto_detect()?;
        backend.forward_attention_context(input, context, strength)
    }
}

/// Integration point for GPU-accelerated attention operations
///
/// This trait allows different attention implementations (PolyAttention, StandardAttention, etc.)
/// to provide GPU-optimized variants while maintaining a consistent interface.
pub trait GpuAttentionDispatch {
    /// Forward pass with GPU acceleration
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>>;

    /// Forward pass with incoming context modulation
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn forward_with_context_gpu(
        &mut self,
        input: &Array2<f32>,
        context: Option<&Array2<f32>>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_context_dimension_validation() {
        let mut context = SharedAttentionContext::new();
        context.similarity_context_strength = Array2::<f32>::zeros((64, 64));

        let input: Array2<f32> = Array2::zeros((4, 32)); // Wrong embed dimension

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            match UnifiedGpuBackend::auto_detect() {
                Ok(mut backend) => {
                    let result = context.apply_incoming_context_gpu(&input, &mut backend);
                    assert!(result.is_err());
                    assert!(
                        result
                            .unwrap_err()
                            .to_string()
                            .contains("Context strength dimension mismatch")
                    );
                }
                Err(e) => {
                    println!("No GPU available: {}", e);
                }
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_attention_context_gpu_auto_detect() {
        // Test that auto-detect works with strict no-fallback
        match UnifiedGpuBackend::auto_detect() {
            Ok(backend) => {
                println!("GPU detected: {}", backend.backend_name());
                assert!(backend.is_ready());
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }
}
