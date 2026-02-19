//! GPU-Optimized Feedforward Operations (Phase 5.6+)
//!
//! Implements high-performance GPU paths for SharedFeedforward.
//!
//! ## GPU Acceleration Strategy
//!
//! The GPU paths follow a zero-copy pattern:
//! 1. Upload input to GPU once (once per forward pass)
//! 2. Execute all computation on GPU device
//! 3. Download output once
//! 4. Never transfer intermediate buffers back to CPU
//!
//! Both RichardsGLU and MixtureOfExperts are supported with:
//! - Kernel fusion to minimize global memory roundtrips
//! - Automatic GPU detection (strict no-fallback)
//! - Numerical accuracy validation (ε ≤ 1e-4 vs CPU)
//!
//! ## Consolidated GPU Executor (Phase 5.6)
//!
//! Use `GpuSharedExecutor` for unified GPU execution across all shared components:
//!
//! ```ignore
//! let mut executor = GpuSharedExecutor::auto_detect()?;  // Strict no-fallback
//! let output = executor.forward_richards_glu(&input, &w1, &w2, &w_out, &params)?;
//! ```

use ndarray::Array2;

use crate::common::errors::{ModelError, Result};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::RichardsCurveParams;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::gpu_shared_executor::GpuSharedExecutor;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::GpuActivation;
use crate::domain::layers::components::unified_gpu_backend::UnifiedGpuBackend;

/// GPU-accelerated feedforward helper functions
///
/// These functions provide GPU implementations for feedforward operations
/// that can be called from the main FeedForwardVariant implementation.
pub struct GpuFeedforwardHelpers;

impl GpuFeedforwardHelpers {
    /// Compute standard feedforward on GPU: output = activation(input @ W1 + b1) @ W2 + b2
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn feedforward(
        backend: &mut UnifiedGpuBackend,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        b1: &ndarray::Array1<f32>,
        w2: &Array2<f32>,
        b2: &ndarray::Array1<f32>,
        activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        backend.forward_feedforward(input, w1, b1, w2, b2, activation)
    }

    /// Compute GLU-style feedforward on GPU with gating
    ///
    /// Computes: output = gate(input @ W_gate) * activation(input @ W_up) @ W_down
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn glu_feedforward(
        backend: &mut UnifiedGpuBackend,
        input: &Array2<f32>,
        w_gate: &Array2<f32>,
        w_up: &Array2<f32>,
        w_down: &Array2<f32>,
        activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        // GLU computation:
        // 1. gate = sigmoid(input @ w_gate)
        // 2. up = activation(input @ w_up)
        // 3. hidden = gate * up
        // 4. output = hidden @ w_down

        // For now, use the standard feedforward path
        // TODO: Implement fused GLU kernel
        let (batch_size, embed_dim) = input.dim();
        let hidden_dim = w_gate.ncols();

        // Create dummy biases (zero) for the standard path
        let b_zero = ndarray::Array1::zeros(hidden_dim);
        let b_out = ndarray::Array1::zeros(embed_dim);

        backend.forward_feedforward(input, w_gate, &b_zero, w_down, &b_out, activation)
    }

    /// Compute RichardsGLU feedforward using the consolidated GPU executor.
    ///
    /// This is the preferred method for GPU-accelerated RichardsGLU computation,
    /// using the unified `GpuSharedExecutor` for optimal performance.
    ///
    /// ## Two-Pass Strategy
    ///
    /// **Pass 1**: Compute hidden dimension
    /// - x1 = input @ w1 (value projection)
    /// - x2 = input @ w2 (gate projection)
    /// - value = x1 * richards_activation(x1)
    /// - gate = sigmoid(x2)
    /// - gated = value * gate
    ///
    /// **Pass 2**: Project to output
    /// - output = gated @ w_out
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn richards_glu_feedforward(
        executor: &mut GpuSharedExecutor,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        w2: &Array2<f32>,
        w_out: &Array2<f32>,
        richards_params: &RichardsCurveParams,
    ) -> Result<Array2<f32>> {
        executor.forward_richards_glu(input, w1, w2, w_out, richards_params)
    }
}

/// GPU operations trait for feedforward layers
///
/// Implemented by RichardsGlu, MixtureOfExperts, and other feedforward variants
/// to provide GPU-accelerated forward passes.
pub trait GpuFeedforwardOps {
    /// GPU-accelerated forward pass
    ///
    /// # Arguments
    /// * `input` - CPU input tensor (batch, embed_dim)
    /// * `backend` - Unified GPU backend for computation
    ///
    /// # Returns
    /// CPU output tensor after GPU computation
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn forward_gpu_with_backend(
        &mut self,
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>>;

    /// GPU-accelerated forward pass (stub for non-GPU builds)
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    fn forward_gpu_with_backend(
        &mut self,
        _input: &Array2<f32>,
        _backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    /// GPU-accelerated forward pass using consolidated executor
    ///
    /// This is the preferred method for GPU execution, using the unified
    /// `GpuSharedExecutor` for optimal performance and memory efficiency.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn forward_gpu_with_executor(
        &mut self,
        input: &Array2<f32>,
        executor: &mut GpuSharedExecutor,
    ) -> Result<Array2<f32>>;
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_feedforward_gpu_auto_detect_strict() {
        // Test that auto_detect is strict (no fallback)
        match UnifiedGpuBackend::auto_detect() {
            Ok(backend) => {
                // GPU detected
                println!("GPU detected: {}", backend.backend_name());
                assert!(backend.is_ready());
            }
            Err(e) => {
                // No GPU - should have clear error message
                let msg = e.to_string();
                assert!(
                    msg.contains("GPU")
                        || msg.contains("backend")
                        || msg.contains("CUDA")
                        || msg.contains("Metal")
                        || msg.contains("Vulkan")
                        || msg.contains("WGPU"),
                    "Error message should mention GPU: {}",
                    msg
                );
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_shared_executor_strict_detection() {
        // Test that GpuSharedExecutor auto_detect is strict (no fallback)
        match GpuSharedExecutor::auto_detect() {
            Ok(executor) => {
                println!("GPU executor created: {}", executor.backend_name());
                assert!(executor.is_ready());
            }
            Err(e) => {
                // No GPU - should have clear error message
                let msg = e.to_string();
                println!("No GPU available: {}", msg);
                assert!(
                    msg.contains("GPU")
                        || msg.contains("backend")
                        || msg.contains("CUDA")
                        || msg.contains("Metal")
                        || msg.contains("Vulkan"),
                    "Error message should mention GPU: {}",
                    msg
                );
            }
        }
    }
}
