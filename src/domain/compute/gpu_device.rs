//! GPU Device Abstraction
//!
//! Provides a unified device context for GPU computation across different backends.
//! Coordinates memory management, operation dispatch, and synchronization.
//!
//! ## Architecture (Phase 5.3)
//!
//! The GpuDevice integrates:
//! - Memory pool for buffer allocation/deallocation
//! - Matrix operations dispatcher
//! - Shader pipeline management
//!
//! ## Strict No-Fallback Mode
//!
//! GPU operations will NOT fall back to CPU. If a GPU operation fails,
//! an error is returned. This ensures predictable performance characteristics.

use super::gpu_memory::{GpuBuffer, GpuMemoryPool, MemoryStats};
use super::gpu_ops::{GpuMatrixOps, RichardsCurveParams};
use crate::common::errors::{ModelError, Result};
use crate::domain::compute_backend::{
    ComputeBackend, detect_available_gpu_backends, detect_available_gpu_backends_runtime,
};
use std::fmt;

/// GPU device context
///
/// Manages memory pools and operation dispatchers for the selected backend.
/// Single-threaded per device (use multiple instances for multi-GPU scenarios).
pub struct GpuDevice {
    backend: ComputeBackend,
    memory: Box<dyn GpuMemoryPool>,
    ops: Box<dyn GpuMatrixOps>,
    name: String,
}

impl fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuDevice")
            .field("backend", &self.backend)
            .field("name", &self.name)
            .field("memory", &self.memory.memory_stats())
            .finish()
    }
}

impl GpuDevice {
    /// Create a new GPU device for the specified backend
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        let name = match backend {
            ComputeBackend::Cuda => "CUDA".to_string(),
            ComputeBackend::Metal => "Metal".to_string(),
            ComputeBackend::Vulkan => "Vulkan".to_string(),
            ComputeBackend::Cpu => {
                return Err(ModelError::Backend {
                    message:
                        "GpuDevice cannot be used with CPU backend. Use CPU computation directly."
                            .to_string(),
                });
            }
        };

        // Try to create backend-specific implementations
        match backend {
            ComputeBackend::Cuda => {
                #[cfg(feature = "gpu-cuda")]
                {
                    let memory = super::cuda::CudaMemoryPool::new(0)?;
                    let ops = super::cuda::CudaMatrixOps::new(memory.device_handle());
                    return Ok(Self {
                        backend,
                        memory: Box::new(memory),
                        ops: Box::new(ops),
                        name,
                    });
                }
                #[cfg(not(feature = "gpu-cuda"))]
                {
                    return Err(ModelError::Backend {
                        message:
                            "CUDA backend requires cudarc feature. Compile with --features gpu-cuda"
                                .to_string(),
                    });
                }
            }
            ComputeBackend::Metal => {
                #[cfg(all(feature = "gpu-metal", target_os = "macos"))]
                {
                    let memory = super::metal::MetalMemoryPool::new()?;
                    let ops = super::metal::MetalMatrixOps::new(memory.device().clone());
                    return Ok(Self {
                        backend,
                        memory: Box::new(memory),
                        ops: Box::new(ops),
                        name,
                    });
                }
                #[cfg(not(all(feature = "gpu-metal", target_os = "macos")))]
                {
                    return Err(ModelError::Backend {
                        message: "Metal backend requires macOS + gpu-metal feature. \
                             Compile on macOS with --features gpu-metal"
                            .to_string(),
                    });
                }
            }
            ComputeBackend::Vulkan => {
                #[cfg(feature = "wgpu")]
                {
                    let memory =
                        futures::executor::block_on(super::wgpu_ops::WgpuMemoryPool::new())
                            .map_err(|e| ModelError::Backend {
                                message: format!(
                                    "Failed to create WGPU memory pool for {}: {:?}",
                                    name, e
                                ),
                            })?;

                    let device = memory.device().clone();
                    let queue = memory.queue().clone();
                    let ops = super::wgpu_ops::WgpuMatrixOps::new(device, queue);

                    return Ok(Self {
                        backend,
                        memory: Box::new(memory),
                        ops: Box::new(ops),
                        name,
                    });
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    return Err(ModelError::Backend {
                        message: format!(
                            "{} backend requires wgpu feature. Compile with --features gpu-wgpu",
                            name
                        ),
                    });
                }
            }
            ComputeBackend::Cpu => unreachable!("CPU handled above"),
        }
    }

    /// Create a GPU device with automatic backend detection.
    ///
    /// Implements unified, strict no-fallback GPU detection with priority order:
    /// **CUDA > Metal > Vulkan > WGPU**
    ///
    /// # Detection Algorithm
    ///
    /// 1. Check compile-time feature flags (which backends are built)
    /// 2. Query runtime GPU detection (which backends are available on system)
    /// 3. Try backends in priority order until one initializes successfully
    /// 4. Return error if no GPU available (strict no-fallback)
    ///
    /// # Errors
    ///
    /// Returns `ModelError::Backend` if:
    /// - No GPU is detected on the system (all backends failed)
    /// - Runtime GPU exists but binary was not compiled with matching feature flags
    /// - GPU backend initialization fails (e.g., device not ready, insufficient memory)
    ///
    /// # Example
    ///
    /// ```ignore
    /// match GpuDevice::auto_detect() {
    ///     Ok(device) => println!("Using GPU backend: {}", device.backend_name()),
    ///     Err(e) => eprintln!("No GPU available: {}", e),
    /// }
    /// ```
    pub fn auto_detect() -> Result<Self> {
        // Query both compile-time and runtime GPU availability
        let compile_time_backends = detect_available_gpu_backends();
        let runtime_backends = detect_available_gpu_backends_runtime();

        // Priority order (CUDA > Metal > Vulkan > WGPU)
        let priority_order = vec![
            ComputeBackend::Cuda,
            ComputeBackend::Metal,
            ComputeBackend::Vulkan,
            // Note: WGPU/Vulkan may be same depending on platform
        ];

        // Try each backend in priority order
        for &backend in &priority_order {
            // Skip if not compiled with this backend
            if !compile_time_backends.contains(&backend) {
                continue;
            }

            // Try to initialize this backend
            match Self::new(backend) {
                Ok(device) => {
                    // Successfully initialized this backend
                    return Ok(device);
                }
                Err(_) => {
                    // This backend is compiled but not available at runtime
                    // Continue to next priority
                    continue;
                }
            }
        }

        // No backend succeeded - provide helpful error message
        if !runtime_backends.is_empty() {
            // GPU exists at runtime but binary wasn't compiled with matching features
            let runtime_names = runtime_backends
                .iter()
                .map(|b| b.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            let available_features = vec!["gpu-cuda", "gpu-metal", "gpu-wgpu"].join(", ");
            return Err(ModelError::Backend {
                message: format!(
                    "GPU detected at runtime ({}) but binary was compiled without matching features. \n\
                     Recompile with one of: --features {}",
                    runtime_names, available_features
                ),
            });
        }

        // No GPU detected at runtime or compile time
        let compiled_features = if compile_time_backends.is_empty() {
            "no GPU features enabled".to_string()
        } else {
            format!(
                "features: {}",
                compile_time_backends
                    .iter()
                    .map(|b| b.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        Err(ModelError::Backend {
            message: format!(
                "No GPU backend available. {} \
                 To use GPU, compile with: --features gpu-cuda (NVIDIA), \
                 --features gpu-metal (Apple), or --features gpu-wgpu (cross-platform)",
                compiled_features
            ),
        })
    }

    /// Get the backend type
    #[inline]
    pub fn backend(&self) -> ComputeBackend {
        self.backend
    }

    /// Get device name
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get backend name as a string (for kernel dispatch)
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        self.backend.as_str()
    }

    /// Allocate a buffer on this device
    pub fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        self.memory.allocate(size_bytes)
    }

    /// Allocate buffer for N f32 elements
    pub fn allocate_f32(&mut self, num_elements: usize) -> Result<GpuBuffer> {
        let size_bytes = num_elements * std::mem::size_of::<f32>();
        self.allocate(size_bytes)
    }

    /// Deallocate a buffer
    pub fn deallocate(&mut self, buffer: GpuBuffer) {
        self.memory.deallocate(buffer);
    }

    /// Free all allocated buffers
    pub fn clear(&mut self) {
        self.memory.clear();
    }

    /// Get memory statistics
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.memory_stats()
    }

    /// Format device info for display
    pub fn format_info(&self) -> String {
        let stats = self.memory_stats();
        format!(
            "{} device '{}': {}",
            self.backend.as_str(),
            self.name,
            stats.format_human()
        )
    }

    //
    // Matrix Operations
    //

    /// Get execution context (pool and ops) for low-level kernel execution
    pub fn execution_context(&mut self) -> (&mut dyn GpuMemoryPool, &mut dyn GpuMatrixOps) {
        (self.memory.as_mut(), self.ops.as_mut())
    }

    /// GEMM: output = alpha * A @ B + beta * output
    ///
    /// # Arguments
    /// * `alpha` - Scaling factor for A @ B
    /// * `a` - First input matrix buffer (M x K)
    /// * `b` - Second input matrix buffer (K x N)
    /// * `beta` - Scaling factor for existing output
    /// * `output` - Output matrix buffer (M x N)
    /// * `m` - Rows of A
    /// * `n` - Columns of B
    /// * `k` - Columns of A / Rows of B
    /// * `trans_a` - Whether to transpose A
    /// * `trans_b` - Whether to transpose B
    pub fn gemm_f32(
        &mut self,
        alpha: f32,
        a: &GpuBuffer,
        b: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        self.ops.gemm_f32(
            self.memory.as_mut(),
            alpha,
            a,
            b,
            beta,
            output,
            m,
            n,
            k,
            trans_a,
            trans_b,
        )
    }

    /// Batched GEMM: output[i] = alpha * A[i] @ B[i] + beta * output[i]
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched_f32(
        &mut self,
        alpha: f32,
        a: &GpuBuffer,
        b: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        strides: [usize; 3],
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        self.ops.gemm_batched_f32(
            self.memory.as_mut(),
            alpha,
            a,
            b,
            beta,
            output,
            m,
            n,
            k,
            batch_count,
            strides,
            trans_a,
            trans_b,
        )
    }

    /// Permute 4D tensor
    pub fn permute_4d(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        output_dims: [usize; 4],
        permuted_input_strides: [usize; 4],
    ) -> Result<()> {
        self.ops.permute_4d(
            self.memory.as_mut(),
            input,
            output,
            output_dims,
            permuted_input_strides,
        )
    }

    /// Compute CoPE Scores
    #[allow(clippy::too_many_arguments)]
    pub fn compute_cope_scores(
        &mut self,
        q: &GpuBuffer,
        pos_emb: &GpuBuffer,
        scores: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        max_pos: usize,
    ) -> Result<()> {
        self.ops.compute_cope_scores(
            self.memory.as_mut(),
            q,
            pos_emb,
            scores,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            max_pos,
        )
    }

    /// Richards Curve
    pub fn richards_curve(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
    ) -> Result<()> {
        self.ops
            .richards_curve(self.memory.as_mut(), input, output, params, size)
    }

    /// MoH Gate Activation
    #[allow(clippy::too_many_arguments)]
    pub fn moh_gate_activation(
        &mut self,
        logits: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        gate_params: &RichardsCurveParams,
        output: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
    ) -> Result<()> {
        self.ops.moh_gate_activation(
            self.memory.as_mut(),
            logits,
            alpha,
            beta,
            gate_params,
            output,
            batch_size,
            num_heads,
        )
    }

    /// Fused Poly Attention
    #[allow(clippy::too_many_arguments)]
    pub fn poly_attention_fused(
        &mut self,
        content_scores: &GpuBuffer,
        pos_scores: &GpuBuffer,
        q_h: &GpuBuffer,
        k_comp: &GpuBuffer,
        poly_a: &GpuBuffer,
        poly_b: &GpuBuffer,
        poly_scale: &GpuBuffer,
        gate: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        max_pos: usize,
        p: usize,
        blr_rank: usize,
    ) -> Result<()> {
        self.ops.poly_attention_fused(
            self.memory.as_mut(),
            content_scores,
            pos_scores,
            q_h,
            k_comp,
            poly_a,
            poly_b,
            poly_scale,
            gate,
            output,
            batch_size,
            num_heads,
            seq_len,
            max_pos,
            p,
            blr_rank,
        )
    }

    /// BLR Projection
    #[allow(clippy::too_many_arguments)]
    pub fn blr_projection(
        &mut self,
        q: &GpuBuffer,
        k: &GpuBuffer,
        q_h: &mut GpuBuffer,
        k_comp: &mut GpuBuffer,
        richards_params: &RichardsCurveParams,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        rank: usize,
    ) -> Result<()> {
        self.ops.blr_projection(
            self.memory.as_mut(),
            q,
            k,
            q_h,
            k_comp,
            richards_params,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            rank,
        )
    }

    /// Element-wise ReLU
    pub fn relu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize) -> Result<()> {
        self.ops.relu(self.memory.as_mut(), input, output, size)
    }

    /// Element-wise GELU
    pub fn gelu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize) -> Result<()> {
        self.ops.gelu(self.memory.as_mut(), input, output, size)
    }

    /// Element-wise SiLU (Swish)
    pub fn silu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize) -> Result<()> {
        self.ops.silu(self.memory.as_mut(), input, output, size)
    }

    /// Element-wise Sigmoid
    pub fn sigmoid(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops.sigmoid(self.memory.as_mut(), input, output, size)
    }

    /// Element-wise multiplication
    pub fn mul(
        &mut self,
        input1: &GpuBuffer,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops
            .mul(self.memory.as_mut(), input1, input2, output, size)
    }

    /// Element-wise scaled addition: output += scale * input
    pub fn add_scaled(
        &mut self,
        scale: f32,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops
            .add_scaled(self.memory.as_mut(), scale, input, output, size)
    }

    /// Element-wise scale: output *= scale
    pub fn scale(&mut self, scale: f32, output: &mut GpuBuffer, size: usize) -> Result<()> {
        self.ops.scale(self.memory.as_mut(), scale, output, size)
    }

    /// AXPY: output = a * input1 + b * input2
    pub fn axpy(
        &mut self,
        a: f32,
        input1: &GpuBuffer,
        b: f32,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops
            .axpy(self.memory.as_mut(), a, input1, b, input2, output, size)
    }

    /// Layer Normalization
    pub fn layer_norm(
        &mut self,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
        feature_size: usize,
        eps: f32,
    ) -> Result<()> {
        self.ops.layer_norm(
            self.memory.as_mut(),
            input,
            gamma,
            beta,
            output,
            batch_size,
            feature_size,
            eps,
        )
    }

    /// Softmax normalization
    ///
    /// Applies softmax row-wise to the input matrix.
    /// Uses numerically stable log-sum-exp trick.
    pub fn softmax(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        self.ops
            .softmax(self.memory.as_mut(), input, output, rows, cols)
    }

    /// Sum reduction
    pub fn sum(&mut self, buffer: &GpuBuffer, size: usize) -> Result<f32> {
        self.ops.sum(self.memory.as_mut(), buffer, size)
    }

    /// Mean reduction
    pub fn mean(&mut self, buffer: &GpuBuffer, size: usize) -> Result<f32> {
        self.ops.mean(self.memory.as_mut(), buffer, size)
    }

    //
    // Data Transfer
    //

    /// Download buffer from device to CPU
    pub fn download(&mut self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()> {
        self.ops
            .download(self.memory.as_mut(), gpu_buffer, cpu_data)
    }

    /// Upload buffer from CPU to device
    pub fn upload(&mut self, cpu_data: &[f32], gpu_buffer: &mut GpuBuffer) -> Result<()> {
        self.ops.upload(self.memory.as_mut(), cpu_data, gpu_buffer)
    }

    /// Copy within device
    pub fn copy_within_device(
        &mut self,
        src: &GpuBuffer,
        dst: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        self.ops
            .copy_within_device(self.memory.as_mut(), src, dst, size)
    }

    // ========================================================================
    // High-Level Operations for Shared Components
    // ========================================================================

    /// Apply attention context modulation on GPU
    ///
    /// Computes: output = input + (strength / embed_dim) * (input @ context)
    ///
    /// This is a high-level operation used by SharedAttentionContext.
    ///
    /// # Arguments
    /// * `input` - Input activation (batch_size, embed_dim)
    /// * `context` - Context matrix (embed_dim, embed_dim)
    /// * `output` - Output buffer (batch_size, embed_dim)
    /// * `strength` - Context modulation strength
    /// * `batch_size` - Number of samples
    /// * `embed_dim` - Embedding dimension
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn apply_attention_context(
        &mut self,
        input: &GpuBuffer,
        context: &GpuBuffer,
        output: &mut GpuBuffer,
        strength: f32,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        // Step 1: Compute input @ context -> output
        // GEMM: output = 1.0 * input @ context + 0.0 * output
        self.gemm_f32(
            1.0, input, context, 0.0, output, batch_size, embed_dim, embed_dim, false, false,
        )?;

        // Step 2: Scale by strength / embed_dim
        let scale = strength / (embed_dim as f32).max(1.0);
        self.scale(scale, output, batch_size * embed_dim)?;

        // Step 3: Add input (output = output + input)
        self.add_scaled(1.0, input, output, batch_size * embed_dim)?;

        Ok(())
    }

    /// Compute similarity matrix for attention context on GPU
    ///
    /// Computes: similarity = activation @ activation^T
    /// Then applies row-wise softmax.
    ///
    /// # Arguments
    /// * `activation` - Input activation (batch_size, embed_dim)
    /// * `similarity_out` - Output similarity matrix (batch_size, batch_size)
    /// * `softmax_out` - Output after softmax (batch_size, batch_size)
    /// * `batch_size` - Number of samples
    /// * `embed_dim` - Embedding dimension
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_similarity_softmax(
        &mut self,
        activation: &GpuBuffer,
        similarity_out: &mut GpuBuffer,
        softmax_out: &mut GpuBuffer,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        // Step 1: Compute similarity = activation @ activation^T
        // For A @ A^T, we use the same buffer for both operands
        // GEMM: similarity = 1.0 * activation @ activation^T + 0.0 * similarity
        self.gemm_f32(
            1.0,
            activation,
            activation,
            0.0,
            similarity_out,
            batch_size,
            batch_size,
            embed_dim,
            false,
            true,
        )?;

        // Step 2: Apply softmax row-wise
        self.softmax(similarity_out, softmax_out, batch_size, batch_size)?;

        Ok(())
    }

    /// Get the underlying wgpu device (if applicable)
    #[cfg(feature = "wgpu")]
    pub fn wgpu_device(&self) -> Option<&wgpu::Device> {
        self.memory
            .as_any()
            .downcast_ref::<super::wgpu_ops::WgpuMemoryPool>()
            .map(|p| p.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_device_memory_tracking() {
        // This test will only work if a GPU is available
        if let Ok(mut device) = GpuDevice::auto_detect() {
            let stats = device.memory_stats();
            println!("Initial: {}", stats.format_human());

            if let Ok(buf) = device.allocate_f32(1024) {
                let stats = device.memory_stats();
                assert_eq!(stats.allocation_count, 1);
                println!("After alloc: {}", stats.format_human());

                device.deallocate(buf);
                let stats = device.memory_stats();
                println!("After dealloc: {}", stats.format_human());
            }
        } else {
            println!("No GPU available, skipping GPU device test");
        }
    }

    #[test]
    fn gpu_device_format_info() {
        if let Ok(device) = GpuDevice::auto_detect() {
            let info = device.format_info();
            println!("{}", info);
            assert!(info.contains(&device.name));
        } else {
            println!("No GPU available, skipping GPU device format test");
        }
    }

    #[test]
    fn gpu_device_strict_no_fallback() {
        // GpuDevice is GPU-only: auto_detect must return an error when no GPU is available.
        match GpuDevice::auto_detect() {
            Ok(device) => {
                println!(
                    "GPU detected: {} ({})",
                    device.name(),
                    device.backend().as_str()
                );
                assert!(device.backend().is_gpu());
            }
            Err(e) => {
                println!("No GPU available (expected on systems without GPU): {}", e);
                // This is the expected behavior - no silent fallback to CPU
            }
        }
    }
}
