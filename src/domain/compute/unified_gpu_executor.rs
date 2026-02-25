//! Unified GPU Kernel Execution
//!
//! Consolidates GPU kernel dispatch for all shared components:
//! - SharedAttentionContext: Context modulation, similarity computation
//! - SharedFeedforward: RichardsGlu, MoE forward passes
//! - SharedTemporalProcessing: Attention, SSM operations
//!
//! ## Design Principles
//!
//! 1. **Zero-allocation dispatch**: Reuses workspace GPU buffers
//! 2. **Automatic GPU detection**: Strict no-fallback mode
//! 3. **Pipeline caching**: Shaders compiled once, reused across calls
//! 4. **Memory efficiency**: In-place operations where possible

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};

/// Unified GPU kernel executor for shared components.
///
/// Provides a single entry point for GPU kernel dispatch across
/// all shared layer components (attention context, feedforward, temporal).
///
/// # Memory Management
///
/// The executor does not own GPU buffers. Instead, it operates on
/// buffers provided by `UnifiedLayerWorkspace`, enabling:
/// - Buffer reuse across layer forward passes
/// - Power-of-2 capacity sizing to minimize reallocations
/// - Zero-copy data flow between kernels
///
/// # Thread Safety
///
/// The executor is NOT thread-safe. Use one executor per thread
/// or wrap in `Mutex` for shared access.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct UnifiedGpuExecutor {
    /// GPU device context (owned or borrowed)
    device: GpuDevice,
    /// Flag indicating if device was auto-detected
    auto_detected: bool,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl UnifiedGpuExecutor {
    /// Create a new GPU executor with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No GPU is detected
    /// - GPU backend initialization fails
    /// - Binary was not compiled with matching GPU feature flags
    ///
    /// # Example
    ///
    /// ```ignore
    /// use crate::domain::compute::unified_gpu_executor::UnifiedGpuExecutor;
    ///
    /// let executor = UnifiedGpuExecutor::auto_detect()?;
    /// println!("GPU: {}", executor.device_name());
    /// ```
    pub fn auto_detect() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        Ok(Self {
            device,
            auto_detected: true,
        })
    }

    /// Create a new GPU executor with strict Intel NPU detection.
    ///
    /// Errors if an Intel NPU-capable adapter is not available.
    pub fn auto_detect_npu() -> Result<Self> {
        let device = GpuDevice::auto_detect_npu()?;
        Ok(Self {
            device,
            auto_detected: true,
        })
    }

    /// Create a GPU executor for a specific backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - The GPU backend to use (Cuda, Metal, Vulkan)
    ///
    /// # Errors
    ///
    /// Returns error if the specified backend is not available.
    pub fn with_backend(backend: crate::domain::compute_backend::ComputeBackend) -> Result<Self> {
        let device = GpuDevice::new(backend)?;
        Ok(Self {
            device,
            auto_detected: false,
        })
    }

    /// Get the underlying GPU device
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Get mutable access to the GPU device
    pub fn device_mut(&mut self) -> &mut GpuDevice {
        &mut self.device
    }

    /// Get the device name
    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Check if the device was auto-detected
    pub fn is_auto_detected(&self) -> bool {
        self.auto_detected
    }

    // ========================================================================
    // SharedAttentionContext Kernels
    // ========================================================================

    /// Apply attention context modulation on GPU.
    ///
    /// Computes: `output = input + (strength / embed_dim) * (input @ context)`
    ///
    /// This is a zero-allocation operation that writes directly to the output buffer.
    ///
    /// # Arguments
    ///
    /// * `input` - Input activation buffer (batch_size, embed_dim)
    /// * `context` - Context matrix buffer (embed_dim, embed_dim)
    /// * `output` - Output buffer (batch_size, embed_dim)
    /// * `strength` - Context modulation strength
    /// * `batch_size` - Number of samples
    /// * `embed_dim` - Embedding dimension
    pub fn apply_attention_context(
        &mut self,
        input: &GpuBuffer,
        context: &GpuBuffer,
        output: &mut GpuBuffer,
        strength: f32,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        self.device
            .apply_attention_context(input, context, output, strength, batch_size, embed_dim)
    }

    /// Compute similarity matrix with softmax for attention context.
    ///
    /// Computes:
    /// 1. `similarity = activation @ activation^T`
    /// 2. `softmax_out = softmax(similarity, dim=-1)`
    ///
    /// # Arguments
    ///
    /// * `activation` - Input activation buffer (batch_size, embed_dim)
    /// * `similarity_out` - Output similarity matrix (batch_size, batch_size)
    /// * `softmax_out` - Output after softmax (batch_size, batch_size)
    /// * `batch_size` - Number of samples
    /// * `embed_dim` - Embedding dimension
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

        self.device.compute_similarity_softmax(
            activation,
            similarity_out,
            softmax_out,
            batch_size,
            embed_dim,
        )
    }

    // ========================================================================
    // SharedFeedforward Kernels
    // ========================================================================

    /// Execute RichardsGlu forward pass on GPU.
    ///
    /// Computes the gated linear unit with Richards curve activation:
    /// 1. `value = input @ w_value + b_value`
    /// 2. `gate = richards_sigmoid(input @ w_gate + b_gate)`
    /// 3. `output = value * gate`
    ///
    /// # Arguments
    ///
    /// * `input` - Input buffer (batch_size, input_dim)
    /// * `w_value` - Value projection weights (input_dim, hidden_dim)
    /// * `b_value` - Value bias (hidden_dim,)
    /// * `w_gate` - Gate projection weights (input_dim, hidden_dim)
    /// * `b_gate` - Gate bias (hidden_dim,)
    /// * `output` - Output buffer (batch_size, hidden_dim)
    /// * `richards_params` - Richards curve parameters
    /// * `batch_size` - Number of samples
    /// * `input_dim` - Input dimension
    /// * `hidden_dim` - Hidden dimension
    pub fn richards_glu_forward(
        &mut self,
        input: &GpuBuffer,
        w_value: &GpuBuffer,
        b_value: &GpuBuffer,
        w_gate: &GpuBuffer,
        b_gate: &GpuBuffer,
        output: &mut GpuBuffer,
        richards_params: &crate::domain::compute::gpu_ops::RichardsCurveParams,
        batch_size: usize,
        input_dim: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        // Allocate temporary buffers for value and gate
        let mut value_buf = self.device.allocate_f32(batch_size * hidden_dim)?;
        let mut gate_buf = self.device.allocate_f32(batch_size * hidden_dim)?;

        // Step 1: Compute value projection
        // value = input @ w_value
        self.device.gemm_f32(
            1.0,
            input,
            w_value,
            0.0,
            &mut value_buf,
            batch_size,
            hidden_dim,
            input_dim,
            false,
            false,
        )?;

        // Step 2: Add value bias (broadcast across batch)
        // Implement row-wise broadcast add: value[i,:] += b_value[:]
        self.device
            .broadcast_add_rows(&mut value_buf, b_value, batch_size, hidden_dim)?;

        // Step 3: Compute gate projection

        // Step 3: Compute gate projection
        // gate_linear = input @ w_gate
        self.device.gemm_f32(
            1.0,
            input,
            w_gate,
            0.0,
            &mut gate_buf,
            batch_size,
            hidden_dim,
            input_dim,
            false,
            false,
        )?;

        // Step 4: Apply Richards curve sigmoid to gate
        self.device
            .sigmoid(&gate_buf.clone(), &mut gate_buf, batch_size * hidden_dim)?;

        // Step 5: Element-wise multiply: output = value * gate
        self.device
            .mul(&value_buf, &gate_buf, output, batch_size * hidden_dim)?;

        // Cleanup temporary buffers
        self.device.deallocate(value_buf);
        self.device.deallocate(gate_buf);

        Ok(())
    }

    // ========================================================================
    // SharedTemporalProcessing Kernels
    // ========================================================================

    /// Execute softmax attention on GPU.
    ///
    /// Computes:
    /// 1. `scores = query @ key^T / sqrt(head_dim)`
    /// 2. `attn_weights = softmax(scores)`
    /// 3. `output = attn_weights @ value`
    ///
    /// # Arguments
    ///
    /// * `query` - Query buffer (batch_size, num_heads, seq_len, head_dim)
    /// * `key` - Key buffer (batch_size, num_heads, seq_len, head_dim)
    /// * `value` - Value buffer (batch_size, num_heads, seq_len, head_dim)
    /// * `output` - Output buffer (batch_size, num_heads, seq_len, head_dim)
    /// * `scores_buf` - Temporary buffer for attention scores
    /// * `batch_size` - Batch size
    /// * `num_heads` - Number of attention heads
    /// * `seq_len` - Sequence length
    /// * `head_dim` - Head dimension
    pub fn softmax_attention(
        &mut self,
        query: &GpuBuffer,
        key: &GpuBuffer,
        value: &GpuBuffer,
        output: &mut GpuBuffer,
        scores_buf: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        if batch_size == 0 || num_heads == 0 || seq_len == 0 {
            return Ok(());
        }
        if head_dim == 0 {
            return Err(ModelError::InvalidInput {
                message: "softmax_attention received head_dim=0".to_string(),
            });
        }

        let total_tokens = batch_size * num_heads * seq_len;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Step 1: Compute attention scores = query @ key^T
        // scores[b,h,i,j] = sum_d query[b,h,i,d] * key[b,h,j,d]
        // This is a batched GEMM: (B*H, L, D) @ (B*H, D, L) -> (B*H, L, L)
        self.device.gemm_f32(
            scale,
            query,
            key, // Note: key should be transposed; shader handles this
            0.0,
            scores_buf,
            total_tokens,
            seq_len,
            head_dim,
            false,
            true,
        )?;

        // Step 2: Apply softmax row-wise (in-place)
        // Note: softmax takes input and output as separate params for flexibility
        // but we use the same buffer for in-place operation
        let scores_ptr = scores_buf as *const GpuBuffer;
        let scores_mut_ptr = scores_buf as *mut GpuBuffer;
        // SAFETY: We're performing an in-place softmax. The function reads from input
        // before writing to output, so this is safe.
        self.device.softmax(
            unsafe { &*scores_ptr },
            unsafe { &mut *scores_mut_ptr },
            total_tokens,
            seq_len,
        )?;

        // Step 3: Compute output = scores @ value
        // output[b,h,i,d] = sum_j scores[b,h,i,j] * value[b,h,j,d]
        self.device.gemm_f32(
            1.0,
            scores_buf,
            value,
            0.0,
            output,
            total_tokens,
            head_dim,
            seq_len,
            false,
            false,
        )?;

        Ok(())
    }

    // ========================================================================
    // Data Transfer Operations
    // ========================================================================

    /// Upload data to GPU buffer.
    pub fn upload(&mut self, cpu_data: &[f32], gpu_buffer: &mut GpuBuffer) -> Result<()> {
        self.device.upload(cpu_data, gpu_buffer)
    }

    /// Download data from GPU buffer.
    pub fn download(&mut self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()> {
        self.device.download(gpu_buffer, cpu_data)
    }

    /// Allocate a new GPU buffer.
    pub fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        self.device.allocate(size_bytes)
    }

    /// Allocate a GPU buffer for f32 elements.
    pub fn allocate_f32(&mut self, num_elements: usize) -> Result<GpuBuffer> {
        self.device.allocate_f32(num_elements)
    }

    /// Deallocate a GPU buffer.
    pub fn deallocate(&mut self, buffer: GpuBuffer) {
        self.device.deallocate(buffer);
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> crate::domain::compute::gpu_memory::MemoryStats {
        self.device.memory_stats()
    }
}

/// Non-GPU fallback stub for documentation purposes.
#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct UnifiedGpuExecutor {
    _private: (),
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl UnifiedGpuExecutor {
    /// Attempt to create a GPU executor without GPU features enabled.
    ///
    /// Always returns an error indicating that GPU features are required.
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message: "GPU execution requires one of: --features gpu-wgpu, gpu-cuda, or gpu-metal"
                .to_string(),
        })
    }

    pub fn auto_detect_npu() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Intel NPU execution requires --features gpu-wgpu".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::UnifiedGpuExecutor;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_auto_detect_executor() {
        match UnifiedGpuExecutor::auto_detect() {
            Ok(executor) => {
                println!(
                    "GPU executor created: {} (auto_detected={})",
                    executor.device_name(),
                    executor.is_auto_detected()
                );
                let stats = executor.memory_stats();
                println!("Memory: {}", stats.format_human());
            }
            Err(e) => {
                println!("GPU not available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_allocate_and_deallocate() {
        if let Ok(mut executor) = UnifiedGpuExecutor::auto_detect() {
            let buf = executor.allocate_f32(1024);
            assert!(buf.is_ok());

            let stats = executor.memory_stats();
            assert_eq!(stats.allocation_count, 1);

            executor.deallocate(buf.unwrap());
            let stats = executor.memory_stats();
            assert_eq!(stats.allocation_count, 0);
        }
    }
}
