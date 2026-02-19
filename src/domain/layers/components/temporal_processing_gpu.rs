//! GPU-Optimized Temporal Processing Operations (Phase 5.6+)
//!
//! Implements high-performance GPU paths for SharedTemporalProcessing covering
//! Transformer attention, Mamba SSM, and RG-LRU variants with unified buffer management.
//!
//! ## Consolidated GPU Backend
//!
//! This module uses `UnifiedGpuBackend` for all GPU operations, providing:
//! - Automatic GPU detection with strict no-fallback semantics
//! - Unified memory management across all architectures
//! - Fused kernel dispatch for attention, SSM, and recurrent operations
//!
//! ## Consolidated GPU Executor (Phase 5.6)
//!
//! Use `GpuSharedExecutor` for unified GPU execution across all shared components:
//!
//! ```ignore
//! let mut executor = GpuSharedExecutor::auto_detect()?;  // Strict no-fallback
//! let output = executor.forward_attention(&query, &key, &value, num_heads, true)?;
//! ```

use ndarray::Array2;

use crate::common::errors::{ModelError, Result};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::gpu_shared_executor::GpuSharedExecutor;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::GpuTemporalType;
use crate::domain::layers::components::unified_gpu_backend::UnifiedGpuBackend;

/// GPU-accelerated temporal processing helper functions
///
/// These functions provide GPU implementations for temporal mixing operations
/// that can be called from the main TemporalMixingLayer implementation.
pub struct GpuTemporalHelpers;

impl GpuTemporalHelpers {
    /// GPU-accelerated forward pass using UnifiedGpuBackend
    ///
    /// Automatically dispatches to the appropriate GPU kernel based on mixing type.
    /// All kernels maintain the same input/output interface for consistency.
    ///
    /// # Errors
    ///
    /// Returns `ModelError::Backend` if GPU is not available or computation fails.
    /// This method does NOT fall back to CPU - use CPU methods explicitly if needed.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_attention(
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        backend.forward_temporal(input, GpuTemporalType::Attention)
    }

    /// GPU kernel for Mamba forward pass
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_mamba(
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        backend.forward_temporal(input, GpuTemporalType::Mamba)
    }

    /// GPU kernel for RG-LRU forward pass
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_rg_lru(
        input: &Array2<f32>,
        backend: &mut UnifiedGpuBackend,
    ) -> Result<Array2<f32>> {
        backend.forward_temporal(input, GpuTemporalType::RgLru)
    }

    /// GPU-accelerated forward with automatic backend detection
    ///
    /// Creates a GPU backend automatically using strict detection (no fallback).
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_auto_detect(input: &Array2<f32>, layer_type: &str) -> Result<Array2<f32>> {
        let mut backend = UnifiedGpuBackend::auto_detect()?;

        match layer_type {
            "attention" => Self::forward_attention(input, &mut backend),
            "mamba" => Self::forward_mamba(input, &mut backend),
            "rg_lru" => Self::forward_rg_lru(input, &mut backend),
            _ => Err(ModelError::Backend {
                message: format!("Unknown layer type for GPU: {}", layer_type),
            }),
        }
    }

    /// GPU-accelerated attention using consolidated executor.
    ///
    /// This is the preferred method for GPU-accelerated attention computation,
    /// using the unified `GpuSharedExecutor` for optimal performance.
    ///
    /// Computes scaled dot-product attention:
    /// output = softmax(Q @ K^T / sqrt(d_k)) @ V
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_attention_with_executor(
        executor: &mut GpuSharedExecutor,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        num_heads: usize,
        causal: bool,
    ) -> Result<Array2<f32>> {
        executor.forward_attention(query, key, value, num_heads, causal)
    }
}

/// GPU operations trait for temporal processing
///
/// Implemented by temporal mixing layers to provide GPU-accelerated forward passes.
pub trait GpuTemporalOps {
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
    use super::*;

    #[test]
    fn test_temporal_gpu_requires_gpu_feature() {
        #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
        {
            // Without GPU features, forward_auto_detect is not available
            // This test just verifies the module compiles
            let _input: Array2<f32> = Array2::zeros((4, 64));
        }

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            // With GPU features, test auto-detection
            let input: Array2<f32> = Array2::zeros((4, 64));
            match GpuTemporalHelpers::forward_auto_detect(&input, "attention") {
                Ok(_) => println!("GPU forward succeeded"),
                Err(e) => println!("No GPU available: {}", e),
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_temporal_gpu_auto_detect_strict() {
        // Test that auto_detect is strict (no fallback)
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

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_shared_executor_temporal_strict() {
        // Test that GpuSharedExecutor auto_detect is strict (no fallback)
        match GpuSharedExecutor::auto_detect() {
            Ok(executor) => {
                println!("GPU executor created: {}", executor.backend_name());
                assert!(executor.is_ready());
            }
            Err(e) => {
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
