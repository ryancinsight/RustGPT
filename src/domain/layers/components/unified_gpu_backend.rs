//! Unified GPU Backend for Shared Components (Phase 5.6)
//!
//! Consolidates GPU operations across Diffusion, SSM, and Transformer architectures.
//! Implements automatic GPU detection with strict no-fallback semantics for troubleshooting.
//!
//! ## Phase 5.6: Fused Kernels & Consolidation
//!
//! This phase focuses on:
//! 1. **Fused Kernels**: Combine multiple operations into single GPU passes
//!    - RichardsGLU: W1 → Richards → W2 → Gate → W_out (1 kernel instead of 5)
//!    - Poly Attention: Q @ K → softmax → V (1 kernel instead of 3+)
//!    - Mamba Scan: Selective scan + projection (1 kernel)
//!
//! 2. **Memory Efficiency**: Reduce global memory traffic
//!    - Power-of-2 buffer sizing for alignment
//!    - Zero-allocation reuse across forward passes
//!    - Streaming data (keep on GPU between operations)
//!
//! 3. **Performance Targets**:
//!    - RichardsGLU: 25x speedup (50ms → 2ms on 1K batch)
//!    - PolyAttention: 30x speedup (30ms → 1ms on 512 batch)
//!    - Mamba Scan: 20x speedup (40ms → 2ms on 512 batch)
//!
//! ## Architecture
//!
//! This module provides:
//! - `UnifiedGpuBackend`: Single entry point for GPU operations across all shared components
//! - `GpuKernelDispatcher`: Type-safe kernel dispatch for attention, feedforward, and temporal ops
//! - `GpuMemoryPool`: Unified memory management with power-of-2 sizing
//!
//! ## Strict No-Fallback Design
//!
//! GPU operations will NOT fall back to CPU. If a GPU operation fails,
//! an error is returned. This ensures predictable performance characteristics
//! and facilitates troubleshooting.
//!
//! **Detection Priority**: CUDA > Metal > Vulkan > WGPU (errors if none available)
//!
//! ## Usage
//!
//! ```ignore
//! // Automatic GPU detection (strict - errors if no GPU)
//! let backend = UnifiedGpuBackend::auto_detect()?;
//!
//! // Or specify a backend explicitly
//! let backend = UnifiedGpuBackend::new(ComputeBackend::Vulkan)?;
//!
//! // Execute operations with strict GPU-first guarantee
//! let output = backend.forward_attention(&input, &context)?;  // Errors if GPU unavailable
//! ```

use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::Array1;
use ndarray::Array2;

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::{GpuDevice, gpu_memory::MemoryStats};
use crate::domain::compute_backend::{
    ComputeBackend, ComputeBackendPreference, resolve_compute_backend,
};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuMatrixOps;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_kernels::{SsmParams, UnifiedGpuKernels};

/// Unified GPU backend for all shared components.
///
/// Provides a single entry point for GPU operations across:
/// - SharedAttentionContext
/// - SharedFeedforward  
/// - SharedTemporalProcessing
///
/// # Thread Safety
///
/// The backend is thread-safe through internal locking. Multiple components
/// can share the same backend instance.
#[derive(Debug)]
pub struct UnifiedGpuBackend {
    /// GPU device context
    device: Arc<Mutex<GpuDevice>>,
    /// Backend type
    backend_type: ComputeBackend,
    /// Memory pool statistics
    stats: GpuBackendStats,
}

/// Statistics for GPU backend usage monitoring.
#[derive(Debug, Clone, Default)]
pub struct GpuBackendStats {
    /// Total kernel launches
    pub kernel_launches: usize,
    /// Total bytes uploaded to GPU
    pub bytes_uploaded: usize,
    /// Total bytes downloaded from GPU
    pub bytes_downloaded: usize,
    /// Buffer reuse count (no reallocation needed)
    pub buffer_reuse_count: usize,
    /// Buffer reallocation count
    pub buffer_realloc_count: usize,
}

impl UnifiedGpuBackend {
    /// Create a new GPU backend for the specified compute backend.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The specified backend is `ComputeBackend::Cpu` (use `GpuDevice` directly for CPU)
    /// - The backend is not available on this system
    /// - GPU initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let backend = UnifiedGpuBackend::new(ComputeBackend::Vulkan)?;
    /// ```
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        if backend == ComputeBackend::Cpu {
            return Err(ModelError::Backend {
                message: "UnifiedGpuBackend requires a GPU backend. Use CPU computation directly for CPU execution.".to_string(),
            });
        }

        let device = GpuDevice::new(backend)?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            backend_type: backend,
            stats: GpuBackendStats::default(),
        })
    }

    /// Create a GPU backend with automatic detection (strict no-fallback).
    ///
    /// Uses the priority order: CUDA > Metal > Vulkan
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU is detected on the system
    /// - GPU feature flags are not enabled (compile with `--features gpu-wgpu` or similar)
    /// - GPU initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// match UnifiedGpuBackend::auto_detect() {
    ///     Ok(backend) => println!("GPU backend: {}", backend.backend_name()),
    ///     Err(e) => println!("No GPU available: {}", e),
    /// }
    /// ```
    pub fn auto_detect() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        let backend_type = device.backend();
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            backend_type,
            stats: GpuBackendStats::default(),
        })
    }

    /// Create a GPU backend from a preference (strict no-fallback).
    ///
    /// Resolves `AutoGpu` preference using runtime detection.
    /// Returns error if GPU is detected but feature flags don't match.
    pub fn from_preference(preference: ComputeBackendPreference) -> Result<Self> {
        let backend = resolve_compute_backend(preference)?;
        if backend == ComputeBackend::Cpu {
            return Err(ModelError::Backend {
                message: "AutoGpu preference resolved to CPU, but UnifiedGpuBackend requires GPU. \
                         Use CPU computation directly or ensure GPU is available."
                    .to_string(),
            });
        }
        Self::new(backend)
    }

    /// Get the backend type.
    #[inline]
    pub fn backend_type(&self) -> ComputeBackend {
        self.backend_type
    }

    /// Get the backend name as a string.
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        self.backend_type.as_str()
    }

    /// Check if GPU is ready for operations.
    pub fn is_ready(&self) -> bool {
        self.device
            .lock()
            .map(|d| d.backend().is_gpu())
            .unwrap_or(false)
    }

    /// Get the underlying GPU device.
    pub fn device(&self) -> Arc<Mutex<GpuDevice>> {
        self.device.clone()
    }

    /// Get backend statistics.
    pub fn stats(&self) -> &GpuBackendStats {
        &self.stats
    }

    /// Reset backend statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GpuBackendStats::default();
    }

    /// Get memory statistics from the GPU device.
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        let device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuBackend::memory_stats".to_string(),
        })?;
        Ok(device.memory_stats())
    }

    // ========================================================================
    // Attention Operations
    // ========================================================================

    /// GPU-accelerated attention context forward pass.
    ///
    /// Computes similarity-based context modulation for attention layers.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, embed_dim)
    /// * `context` - Context tensor (embed_dim, embed_dim)
    /// * `strength` - Context strength multiplier
    ///
    /// # Returns
    /// * Output tensor (batch_size, embed_dim)
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_attention_context(
        &mut self,
        input: &Array2<f32>,
        context: &Array2<f32>,
        strength: f32,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuBackend::forward_attention_context"
                .to_string(),
        })?;

        let (batch_size, embed_dim) = input.dim();
        let (ctx_rows, ctx_cols) = context.dim();

        if ctx_rows != embed_dim || ctx_cols != embed_dim {
            return Err(ModelError::ShapeMismatch {
                expected: vec![embed_dim, embed_dim],
                actual: vec![ctx_rows, ctx_cols],
                message: "Context matrix dimensions must match embed_dim".to_string(),
            });
        }

        // Allocate GPU buffers
        let input_size = batch_size * embed_dim * std::mem::size_of::<f32>();
        let context_size = embed_dim * embed_dim * std::mem::size_of::<f32>();
        let output_size = batch_size * embed_dim * std::mem::size_of::<f32>();

        let mut input_buf = device.allocate(input_size)?;
        let mut context_buf = device.allocate(context_size)?;
        let mut output_buf = device.allocate(output_size)?;

        // Upload data using GpuDevice methods
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;
        device.upload(context.as_slice().unwrap(), &mut context_buf)?;

        // Compute: output = input @ context * strength
        // GEMM: C = alpha * A @ B + beta * C
        device.gemm_f32(
            1.0,
            &input_buf,
            &context_buf,
            0.0,
            &mut output_buf,
            batch_size,
            embed_dim,
            embed_dim,
            false,
            false,
        )?;

        // Download result
        let mut output = Array2::zeros((batch_size, embed_dim));
        device.download(&output_buf, output.as_slice_mut().unwrap())?;

        // Update stats
        self.stats.kernel_launches += 1;
        self.stats.bytes_uploaded += input_size + context_size;
        self.stats.bytes_downloaded += output_size;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(context_buf);
        device.deallocate(output_buf);

        Ok(output)
    }

    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_attention_context(
        &mut self,
        _input: &Array2<f32>,
        _context: &Array2<f32>,
        _strength: f32,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    // ========================================================================
    // Feedforward Operations
    // ========================================================================

    /// GPU-accelerated feedforward forward pass.
    ///
    /// Computes: output = activation(input @ W1 + b1) @ W2 + b2
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, embed_dim)
    /// * `w1` - First weight matrix (embed_dim, hidden_dim)
    /// * `b1` - First bias (hidden_dim,)
    /// * `w2` - Second weight matrix (hidden_dim, embed_dim)
    /// * `b2` - Second bias (embed_dim,)
    /// * `activation` - Activation function type
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_feedforward(
        &mut self,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        use ndarray::Array1;

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuBackend::forward_feedforward".to_string(),
        })?;

        let (batch_size, embed_dim) = input.dim();
        let (_, hidden_dim) = w1.dim();

        // Allocate buffers
        let input_size = batch_size * embed_dim * std::mem::size_of::<f32>();
        let hidden_size = batch_size * hidden_dim * std::mem::size_of::<f32>();
        let output_size = batch_size * embed_dim * std::mem::size_of::<f32>();
        let w1_size = embed_dim * hidden_dim * std::mem::size_of::<f32>();
        let w2_size = hidden_dim * embed_dim * std::mem::size_of::<f32>();

        let mut input_buf = device.allocate(input_size)?;
        let mut hidden_buf = device.allocate(hidden_size)?;
        let mut output_buf = device.allocate(output_size)?;
        let mut w1_buf = device.allocate(w1_size)?;
        let mut w2_buf = device.allocate(w2_size)?;

        // Upload weights and input using GpuDevice methods
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;
        device.upload(w1.as_slice().unwrap(), &mut w1_buf)?;
        device.upload(w2.as_slice().unwrap(), &mut w2_buf)?;

        // First projection: hidden = input @ W1
        device.gemm_f32(
            1.0,
            &input_buf,
            &w1_buf,
            0.0,
            &mut hidden_buf,
            batch_size,
            hidden_dim,
            embed_dim,
            false,
            false,
        )?;

        // Add bias and apply activation
        // Note: For now, we do this on CPU after download
        // TODO: Implement GPU bias add and activation kernels

        let mut hidden = Array2::zeros((batch_size, hidden_dim));
        device.download(&hidden_buf, hidden.as_slice_mut().unwrap())?;

        // Add bias
        for i in 0..batch_size {
            for j in 0..hidden_dim {
                hidden[[i, j]] += b1[j];
            }
        }

        // Apply activation
        match activation {
            GpuActivation::Relu => hidden.mapv_inplace(|x| x.max(0.0)),
            GpuActivation::Gelu => hidden.mapv_inplace(|x| {
                // Approximate GELU
                0.5 * x * (1.0 + (x * 0.7978845608).tanh())
            }),
            GpuActivation::Silu => hidden.mapv_inplace(|x| x / (1.0 + (-x).exp())),
            GpuActivation::Identity => {}
        }

        // Upload activated hidden for second projection
        device.upload(hidden.as_slice().unwrap(), &mut hidden_buf)?;

        // Second projection: output = hidden @ W2
        device.gemm_f32(
            1.0,
            &hidden_buf,
            &w2_buf,
            0.0,
            &mut output_buf,
            batch_size,
            embed_dim,
            hidden_dim,
            false,
            false,
        )?;

        // Download and add final bias
        let mut output = Array2::zeros((batch_size, embed_dim));
        device.download(&output_buf, output.as_slice_mut().unwrap())?;

        for i in 0..batch_size {
            for j in 0..embed_dim {
                output[[i, j]] += b2[j];
            }
        }

        // Update stats
        self.stats.kernel_launches += 2; // Two GEMM operations
        self.stats.bytes_uploaded += input_size + w1_size + w2_size + hidden_size;
        self.stats.bytes_downloaded += hidden_size + output_size;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(hidden_buf);
        device.deallocate(output_buf);
        device.deallocate(w1_buf);
        device.deallocate(w2_buf);

        Ok(output)
    }

    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_feedforward(
        &mut self,
        _input: &Array2<f32>,
        _w1: &Array2<f32>,
        _b1: &ndarray::Array1<f32>,
        _w2: &Array2<f32>,
        _b2: &ndarray::Array1<f32>,
        _activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    // ========================================================================
    // Temporal Processing Operations
    // ========================================================================

    /// GPU-accelerated temporal processing forward pass.
    ///
    /// Dispatches to the appropriate kernel based on the temporal mixing type.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_temporal(
        &mut self,
        input: &Array2<f32>,
        temporal_type: GpuTemporalType,
    ) -> Result<Array2<f32>> {
        let (seq_len, embed_dim) = input.dim();
        if seq_len == 0 || embed_dim == 0 {
            return Ok(Array2::zeros((seq_len, embed_dim)));
        }
        match temporal_type {
            GpuTemporalType::Attention => {
                Err(ModelError::Backend {
                    message: "UnifiedGpuBackend::forward_temporal(Attention) requires Q/K/V layer weights and is not wired through this generic path yet.".to_string(),
                })
            }
            GpuTemporalType::Mamba => {
                let mut kernels = UnifiedGpuKernels::new(self.backend_type)?;
                let params = SsmParams::new(embed_dim.clamp(1, 32), embed_dim, seq_len, 1);
                let out = kernels.ssm_forward(input, &params, GpuTemporalType::Mamba)?;
                self.stats.kernel_launches += 1;
                let bytes = seq_len * embed_dim * std::mem::size_of::<f32>();
                self.stats.bytes_uploaded += bytes;
                self.stats.bytes_downloaded += bytes;
                Ok(out)
            }
            GpuTemporalType::RgLru => {
                let mut kernels = UnifiedGpuKernels::new(self.backend_type)?;
                let params = SsmParams::new(embed_dim.max(1), embed_dim, seq_len, 1);
                let out = kernels.ssm_forward(input, &params, GpuTemporalType::RgLru)?;
                self.stats.kernel_launches += 1;
                let bytes = seq_len * embed_dim * std::mem::size_of::<f32>();
                self.stats.bytes_uploaded += bytes;
                self.stats.bytes_downloaded += bytes;
                Ok(out)
            }
        }
    }

    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_temporal(
        &mut self,
        _input: &Array2<f32>,
        _temporal_type: GpuTemporalType,
    ) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }
}

/// GPU activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuActivation {
    /// No activation (identity)
    Identity,
    /// ReLU: max(0, x)
    Relu,
    /// GELU: x * Φ(x) where Φ is the standard Gaussian CDF
    Gelu,
    /// SiLU/Swish: x * sigmoid(x)
    Silu,
}

/// GPU temporal processing types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuTemporalType {
    /// Standard attention (Q @ K^T @ V)
    Attention,
    /// Mamba SSM (selective scan)
    Mamba,
    /// RG-LRU (recurrent)
    RgLru,
}

impl Default for GpuActivation {
    fn default() -> Self {
        Self::Gelu
    }
}

// ============================================================================
// GpuComponent Implementation for Shared Components
// ============================================================================

/// Implement `GpuComponent` trait for any component that can use GPU.
///
/// This macro implements the unified GPU interface for shared components.
#[macro_export]
macro_rules! impl_gpu_component {
    ($ty:ty) => {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        impl $crate::domain::compute::GpuComponent for $ty {
            fn set_gpu_device(
                &mut self,
                device: std::sync::Arc<std::sync::Mutex<$crate::domain::compute::GpuDevice>>,
            ) {
                self.gpu_device = Some(device);
            }

            fn enable_gpu_auto_detect(&mut self) -> $crate::common::errors::Result<()> {
                let device = $crate::domain::compute::GpuDevice::auto_detect()?;
                self.gpu_device = Some(std::sync::Arc::new(std::sync::Mutex::new(device)));
                Ok(())
            }

            fn is_gpu_ready(&self) -> bool {
                self.gpu_device.is_some()
            }

            fn gpu_backend_name(&self) -> Option<&'static str> {
                self.gpu_device
                    .as_ref()
                    .and_then(|d| d.lock().ok().map(|guard| guard.backend().as_str()))
            }

            fn gpu_device(
                &self,
            ) -> Option<std::sync::Arc<std::sync::Mutex<$crate::domain::compute::GpuDevice>>> {
                self.gpu_device.clone()
            }

            fn ensure_capacity(
                &mut self,
                _batch_size: usize,
                _embed_dim: usize,
                _seq_len: usize,
            ) -> $crate::common::errors::Result<()> {
                // Default implementation - override in specific components as needed
                Ok(())
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_detect_no_fallback() {
        // This test verifies strict no-fallback behavior
        match UnifiedGpuBackend::auto_detect() {
            Ok(backend) => {
                println!("GPU detected: {}", backend.backend_name());
                assert!(backend.backend_type().is_gpu());
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
                // This is correct behavior - strict no-fallback
            }
        }
    }

    #[test]
    fn test_cpu_backend_rejected() {
        let result = UnifiedGpuBackend::new(ComputeBackend::Cpu);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("requires a GPU backend")
        );
    }

    #[test]
    fn test_stats_tracking() {
        // Stats should start at zero
        let stats = GpuBackendStats::default();
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.bytes_uploaded, 0);
        assert_eq!(stats.bytes_downloaded, 0);
    }
}
