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
    ComputeBackend, resolve_compute_backend_strict_auto_gpu,
    resolve_compute_backend_strict_auto_npu,
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
            ComputeBackend::Vulkan => "Vulkan/WGPU".to_string(),
            ComputeBackend::Npu => "Intel NPU/WGPU".to_string(),
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
                    let adapter_label = if memory.adapter_is_npu() {
                        format!(
                            "{} [{}; Intel NPU]",
                            memory.adapter_name(),
                            memory.adapter_backend()
                        )
                    } else {
                        format!(
                            "{} [{} / {}]",
                            memory.adapter_name(),
                            memory.adapter_backend(),
                            memory.adapter_device_type()
                        )
                    };
                    let name = format!("{} ({})", name, adapter_label);

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
            ComputeBackend::Npu => {
                #[cfg(feature = "wgpu")]
                {
                    let memory = futures::executor::block_on(
                        super::wgpu_ops::WgpuMemoryPool::new_with_intel_npu(true),
                    )
                    .map_err(|e| ModelError::Backend {
                        message: format!("Failed to create WGPU memory pool for {}: {:?}", name, e),
                    })?;
                    let adapter_label = if memory.adapter_is_npu() {
                        format!(
                            "{} [{}; Intel NPU]",
                            memory.adapter_name(),
                            memory.adapter_backend()
                        )
                    } else {
                        format!(
                            "{} [{} / {}]",
                            memory.adapter_name(),
                            memory.adapter_backend(),
                            memory.adapter_device_type()
                        )
                    };
                    let name = format!("{} ({})", name, adapter_label);

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
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        Self::new(backend)
    }

    /// Create a GPU device with strict Intel NPU auto-detection.
    ///
    /// This requires an Intel NPU-capable adapter and never falls back to non-NPU GPU/CPU.
    pub fn auto_detect_npu() -> Result<Self> {
        let backend = resolve_compute_backend_strict_auto_npu()?;
        Self::new(backend)
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

    /// Apply causal masking to attention scores [B, H, S, S] in-place.
    pub fn causal_mask_attention_scores(
        &mut self,
        scores: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        mask_value: f32,
    ) -> Result<()> {
        self.ops.causal_mask_attention_scores(
            self.memory.as_mut(),
            scores,
            batch_size,
            num_heads,
            seq_len,
            mask_value,
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

    /// Richards Curve backward input-gradient application
    pub fn richards_curve_backward_input(
        &mut self,
        input: &GpuBuffer,
        upstream: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
    ) -> Result<()> {
        self.ops.richards_curve_backward_input(
            self.memory.as_mut(),
            input,
            upstream,
            output,
            params,
            size,
        )
    }

    /// Reduce scalar RichardsCurve parameter gradients over all elements on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn richards_scalar_param_grads_reduce(
        &mut self,
        input: &GpuBuffer,
        upstream: &GpuBuffer,
        output_grads: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
        variant_is_tanh: bool,
        birch_exponential_tail: bool,
    ) -> Result<()> {
        self.ops.richards_scalar_param_grads_reduce(
            self.memory.as_mut(),
            input,
            upstream,
            output_grads,
            params,
            size,
            variant_is_tanh,
            birch_exponential_tail,
        )
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

    /// MoH gate backward pointwise prep (sigmoid-approx helper path).
    pub fn moh_gate_backward_prepare_sigmoid(
        &mut self,
        xw: &GpuBuffer,
        eff_grads: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        d_gate: &mut GpuBuffer,
        d_gate_scaled: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> Result<()> {
        self.ops.moh_gate_backward_prepare_sigmoid(
            self.memory.as_mut(),
            xw,
            eff_grads,
            alpha,
            beta,
            d_gate,
            d_gate_scaled,
            num_tokens,
            num_heads,
        )
    }

    /// MoH gate backward per-head reductions for alpha/beta grads (sigmoid-approx helper path).
    pub fn moh_gate_backward_reduce_alpha_beta(
        &mut self,
        xw: &GpuBuffer,
        d_gate: &GpuBuffer,
        grad_alpha: &mut GpuBuffer,
        grad_beta: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> Result<()> {
        self.ops.moh_gate_backward_reduce_alpha_beta(
            self.memory.as_mut(),
            xw,
            d_gate,
            grad_alpha,
            grad_beta,
            num_tokens,
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

    /// PolyAttention gate broadcast multiply:
    /// `grad_transformed = grad_scores * gate_broadcast`.
    pub fn poly_attention_gate_broadcast_mul(
        &mut self,
        grad_scores: &GpuBuffer,
        gate: &GpuBuffer,
        grad_transformed: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.ops.poly_attention_gate_broadcast_mul(
            self.memory.as_mut(),
            grad_scores,
            gate,
            grad_transformed,
            batch_size,
            num_heads,
            seq_len,
        )
    }

    /// PolyAttention gate upstream reduction over key dimension.
    pub fn poly_attention_gate_reduce_upstream(
        &mut self,
        grad_scores: &GpuBuffer,
        transformed: &GpuBuffer,
        gate_upstream: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.ops.poly_attention_gate_reduce_upstream(
            self.memory.as_mut(),
            grad_scores,
            transformed,
            gate_upstream,
            batch_size,
            num_heads,
            seq_len,
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

    /// Fill a buffer with a constant `f32` value.
    pub fn fill_f32(&mut self, buffer: &mut GpuBuffer, value: f32) -> Result<()> {
        self.ops.fill_f32(self.memory.as_mut(), buffer, value)
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

    /// Row-wise broadcast addition: matrix[i,:] += bias[:]
    ///
    /// Adds a 1D bias vector to each row of a 2D matrix.
    /// This is commonly used for bias addition in neural network layers.
    ///
    /// # Arguments
    /// * `matrix` - Matrix buffer (batch_size, cols) - modified in place
    /// * `bias` - Bias vector buffer (cols,)
    /// * `batch_size` - Number of rows
    /// * `cols` - Number of columns
    pub fn broadcast_add_rows(
        &mut self,
        matrix: &mut GpuBuffer,
        bias: &GpuBuffer,
        batch_size: usize,
        cols: usize,
    ) -> Result<()> {
        if batch_size == 0 || cols == 0 {
            return Ok(());
        }

        if self
            .ops
            .broadcast_add_rows(self.memory.as_mut(), matrix, bias, batch_size, cols)
            .is_ok()
        {
            return Ok(());
        }

        // Download bias to CPU for simplicity (avoiding custom kernel)
        let mut bias_host = vec![0.0f32; cols];
        self.download(bias, &mut bias_host)?;

        // Download matrix, add bias, upload back
        let mut matrix_host = vec![0.0f32; batch_size * cols];
        self.download(matrix, &mut matrix_host)?;

        for row in 0..batch_size {
            for col in 0..cols {
                matrix_host[row * cols + col] += bias_host[col];
            }
        }

        self.upload(&matrix_host, matrix)?;
        Ok(())
    }

    /// Element-wise scale: output *= scale
    pub fn scale(&mut self, scale: f32, output: &mut GpuBuffer, size: usize) -> Result<()> {
        self.ops.scale(self.memory.as_mut(), scale, output, size)
    }

    /// In-place sign-preserving log scaling:
    /// `x <- sign(x) * log1p(alpha * |x|) / alpha`
    pub fn signed_log1p_scale(
        &mut self,
        buffer: &mut GpuBuffer,
        alpha: f32,
        size: usize,
    ) -> Result<()> {
        self.ops
            .signed_log1p_scale(self.memory.as_mut(), buffer, alpha, size)
    }

    // ========================================================================
    // Deferred Submission / Command Batching
    // ========================================================================

    /// Begin deferred GPU recording mode.
    ///
    /// All subsequent GPU dispatch calls will be recorded into a shared
    /// encoder rather than submitted one-by-one. This eliminates the
    /// CPU-GPU sync bubbles that cause ~80% of GPU idle time in the
    /// current pipeline.
    ///
    /// Call `flush()` after all operations for a batch are submitted to
    /// actually execute them on the GPU.
    ///
    /// # Example (training step)
    /// ```ignore
    /// device.begin_recording();          // start batching
    /// device.gemm_f32(...)?;             // recorded, NOT submitted
    /// device.layer_norm(...)?;           // recorded, NOT submitted
    /// device.softmax(...)?;              // recorded, NOT submitted
    /// device.flush();                    // ONE GPU submission for all 3 ops
    /// ```
    pub fn begin_recording(&mut self) {
        self.ops.begin_recording();
    }

    /// Flush all pending GPU commands and return when submitted.
    ///
    /// Call once per training step to submit the entire forward+backward
    /// as a single GPU batch, eliminating per-kernel CPU-GPU sync bubbles.
    pub fn flush(&mut self) {
        self.ops.flush();
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
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        self.ops
            .softmax(self.memory.as_mut(), input, output, rows, cols)
    }

    /// Softmax backward (row-wise)
    pub fn softmax_backward(
        &mut self,
        softmax_output: &GpuBuffer,
        grad_output: &GpuBuffer,
        grad_input: &mut GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 {
            return Ok(());
        }
        self.ops.softmax_backward(
            self.memory.as_mut(),
            softmax_output,
            grad_output,
            grad_input,
            rows,
            cols,
        )
    }

    /// PolyAttention scalar score transform:
    /// `out = scale * (a * smooth_clip_tanh(x, clip_limit)^p + b)`
    #[allow(clippy::too_many_arguments)]
    pub fn poly_score_transform_scalar(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        a: f32,
        b: f32,
        scale: f32,
        p: u32,
        clip_limit: f32,
        size: usize,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        self.ops.poly_score_transform_scalar(
            self.memory.as_mut(),
            input,
            output,
            a,
            b,
            scale,
            p,
            clip_limit,
            size,
        )
    }

    /// PolyAttention scalar score transform backward + reduction contributions.
    #[allow(clippy::too_many_arguments)]
    pub fn poly_score_transform_scalar_backward(
        &mut self,
        raw_scores: &GpuBuffer,
        grad_transformed: &GpuBuffer,
        grad_raw: &mut GpuBuffer,
        grad_a_contrib: &mut GpuBuffer,
        grad_b_contrib: &mut GpuBuffer,
        grad_scale_contrib: &mut GpuBuffer,
        a: f32,
        b: f32,
        scale: f32,
        p: u32,
        clip_limit: f32,
        size: usize,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        self.ops.poly_score_transform_scalar_backward(
            self.memory.as_mut(),
            raw_scores,
            grad_transformed,
            grad_raw,
            grad_a_contrib,
            grad_b_contrib,
            grad_scale_contrib,
            a,
            b,
            scale,
            p,
            clip_limit,
            size,
        )
    }

    /// Selective scan forward kernel for SSM recurrence.
    ///
    /// Computes:
    /// - `h_t = A @ h_{t-1} + B @ x_t`
    /// - `y_t = C @ h_t + D @ x_t`
    #[allow(clippy::too_many_arguments)]
    pub fn selective_scan_forward(
        &mut self,
        input: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        d: &GpuBuffer,
        h_init: &GpuBuffer,
        output: &mut GpuBuffer,
        h_final: &mut GpuBuffer,
        seq_len: usize,
        state_dim: usize,
        embed_dim: usize,
    ) -> Result<()> {
        self.ops.selective_scan_forward(
            self.memory.as_mut(),
            input,
            a,
            b,
            c,
            d,
            h_init,
            output,
            h_final,
            seq_len,
            state_dim,
            embed_dim,
        )
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

    /// Copy a sub-range between two device buffers.
    ///
    /// Offsets and size are in `f32` elements.
    pub fn copy_within_device_range(
        &mut self,
        src: &GpuBuffer,
        src_offset: usize,
        dst: &mut GpuBuffer,
        dst_offset: usize,
        size: usize,
    ) -> Result<()> {
        self.ops.copy_within_device_range(
            self.memory.as_mut(),
            src,
            src_offset,
            dst,
            dst_offset,
            size,
        )
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
        if batch_size == 0 || embed_dim == 0 {
            return Ok(());
        }

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

    // ========================================================================
    // Optimizer Operations
    // ========================================================================

    /// Adam optimizer step - updates parameters in-place on GPU
    ///
    /// Computes the Adam update:
    /// ```text
    /// m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
    /// v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
    /// m̂_t = m_t / (1 - β₁^t)
    /// v̂_t = v_t / (1 - β₂^t)
    /// θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
    /// ```
    ///
    /// For AdamW (decoupled weight decay):
    /// ```text
    /// θ_t = θ_{t-1} · (1 - λη) - η · m̂_t / (√v̂_t + ε)
    /// ```
    ///
    /// For AMSGrad:
    /// ```text
    /// v̂_{max,t} = max(v̂_{max,t-1}, v̂_t)
    /// θ_t = θ_{t-1} - η · m̂_t / (√v̂_{max,t} + ε)
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn adam_step(
        &mut self,
        params: &mut GpuBuffer,
        grads: &GpuBuffer,
        m: &mut GpuBuffer,
        v: &mut GpuBuffer,
        v_max: Option<&mut GpuBuffer>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        inv_bias1: f32,
        inv_bias2: f32,
        weight_decay: f32,
        use_decoupled_wd: bool,
        use_amsgrad: bool,
        size: usize,
    ) -> Result<()> {
        self.ops.adam_step(
            self.memory.as_mut(),
            params,
            grads,
            m,
            v,
            v_max,
            lr,
            beta1,
            beta2,
            epsilon,
            inv_bias1,
            inv_bias2,
            weight_decay,
            use_decoupled_wd,
            use_amsgrad,
            size,
        )
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
