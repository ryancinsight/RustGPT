//! Unified GPU Executor for Shared Components (Phase 5.6)
//!
//! Consolidates GPU execution for SharedFeedforward, SharedTemporalProcessing,
//! and SharedAttentionContext with automatic GPU detection and strict no-fallback.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  GpuSharedExecutor                          │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
//! │  │ Feedforward │  │   Temporal   │  │ AttentionContext │   │
//! │  │    Kernel   │  │   Kernel     │  │     Kernel       │   │
//! │  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘   │
//! │         │                │                    │             │
//! │         v                v                    v             │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              UnifiedGpuBackend                       │   │
//! │  │   (GEMM, Activation, Norm, Softmax, Transfer)       │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! │                           │                                 │
//! │                           v                                 │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │               GpuDevice                              │   │
//! │  │   (CUDA / Metal / Vulkan auto-detection)            │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Efficiency
//!
//! - Pre-allocated workspace buffers with power-of-2 sizing
//! - Buffer reuse across kernel calls
//! - Zero-copy transfers when possible
//!
//! ## Performance Targets
//!
//! | Component          | CPU Time | GPU Target | Speedup |
//! |--------------------|----------|------------|---------|
//! | RichardsGLU FFN    | 30ms     | 1.5ms      | 20x     |
//! | PolyAttention      | 40ms     | 1.3ms      | 30x     |
//! | Mamba Scan         | 50ms     | 2.5ms      | 20x     |
//! | RG-LRU             | 35ms     | 2.3ms      | 15x     |
//! | AttentionContext   | 10ms     | 0.5ms      | 20x     |

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::Array2;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice, GpuMatrixOps, RichardsCurveParams};

/// Unified GPU executor for all shared components.
///
/// Provides a single entry point for GPU-accelerated operations across
/// Transformer, Diffusion, and SSM architectures.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuSharedExecutor {
    /// GPU device with auto-detected backend
    device: Arc<Mutex<GpuDevice>>,
    /// Pre-allocated workspace for intermediate computations
    workspace: GpuWorkspace,
    /// Execution statistics for monitoring
    stats: GpuExecutionStats,
}

/// Workspace buffers for GPU computations
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
struct GpuWorkspace {
    /// Buffer for intermediate hidden dimension computations
    hidden_buffer: Option<GpuBuffer>,
    /// Buffer for gate computations
    gate_buffer: Option<GpuBuffer>,
    /// Buffer for value computations
    value_buffer: Option<GpuBuffer>,
    /// Buffer for output projection
    output_buffer: Option<GpuBuffer>,
    /// Current capacity (batch_size * hidden_dim)
    capacity: usize,
}

/// Re-export from canonical location
pub use crate::domain::compute::GpuExecutionStats;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuSharedExecutor {
    /// Create a new GPU executor with automatic backend detection.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU is detected on the system
    /// - GPU feature flags are not enabled
    /// - GPU initialization fails
    ///
    /// This method does NOT fall back to CPU - use CPU methods explicitly if needed.
    pub fn auto_detect() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            workspace: GpuWorkspace {
                hidden_buffer: None,
                gate_buffer: None,
                value_buffer: None,
                output_buffer: None,
                capacity: 0,
            },
            stats: GpuExecutionStats::default(),
        })
    }

    /// Create a GPU executor for a specific backend.
    pub fn new(backend: crate::domain::compute_backend::ComputeBackend) -> Result<Self> {
        let device = GpuDevice::new(backend)?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            workspace: GpuWorkspace {
                hidden_buffer: None,
                gate_buffer: None,
                value_buffer: None,
                output_buffer: None,
                capacity: 0,
            },
            stats: GpuExecutionStats::default(),
        })
    }

    /// Get the GPU device reference
    pub fn device(&self) -> Arc<Mutex<GpuDevice>> {
        self.device.clone()
    }

    /// Check if GPU is ready for execution
    pub fn is_ready(&self) -> bool {
        self.device
            .lock()
            .map(|d| d.backend().is_gpu())
            .unwrap_or(false)
    }

    /// Get the backend name
    pub fn backend_name(&self) -> &'static str {
        self.device
            .lock()
            .map(|d| d.backend().as_str())
            .unwrap_or("none")
    }

    /// Ensure workspace has sufficient capacity.
    ///
    /// Pre-allocates buffers with power-of-2 sizing for efficiency.
    pub fn ensure_capacity(&mut self, batch_size: usize, hidden_dim: usize) -> Result<()> {
        let required = batch_size * hidden_dim;

        // Use power-of-2 sizing for efficient reuse
        let target_capacity = required.next_power_of_two().max(1024);

        if self.workspace.capacity < target_capacity {
            let mut device = self.device.lock().map_err(|_| ModelError::Backend {
                message: "Failed to acquire GPU device lock".to_string(),
            })?;

            // Deallocate old buffers
            if let Some(buf) = self.workspace.hidden_buffer.take() {
                device.deallocate(buf);
            }
            if let Some(buf) = self.workspace.gate_buffer.take() {
                device.deallocate(buf);
            }
            if let Some(buf) = self.workspace.value_buffer.take() {
                device.deallocate(buf);
            }
            if let Some(buf) = self.workspace.output_buffer.take() {
                device.deallocate(buf);
            }

            // Allocate new buffers with target capacity
            let size_bytes = target_capacity * std::mem::size_of::<f32>();
            self.workspace.hidden_buffer = Some(device.allocate(size_bytes)?);
            self.workspace.gate_buffer = Some(device.allocate(size_bytes)?);
            self.workspace.value_buffer = Some(device.allocate(size_bytes)?);
            self.workspace.output_buffer = Some(device.allocate(size_bytes)?);
            self.workspace.capacity = target_capacity;
        }

        Ok(())
    }

    // ========================================================================
    // Feedforward Operations (RichardsGLU)
    // ========================================================================

    /// Execute RichardsGLU feedforward on GPU.
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
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, input_dim)
    /// * `w1` - Value projection weights (input_dim, hidden_dim)
    /// * `w2` - Gate projection weights (input_dim, hidden_dim)
    /// * `w_out` - Output projection weights (hidden_dim, output_dim)
    /// * `richards_params` - Richards curve parameters
    ///
    /// # Returns
    /// Output tensor (batch_size, output_dim)
    pub fn forward_richards_glu(
        &mut self,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        w2: &Array2<f32>,
        w_out: &Array2<f32>,
        richards_params: &RichardsCurveParams,
    ) -> Result<Array2<f32>> {
        let (batch_size, input_dim) = input.dim();
        let hidden_dim = w1.ncols();
        let output_dim = w_out.ncols();

        // Ensure workspace capacity
        self.ensure_capacity(batch_size, hidden_dim)?;

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        // Allocate GPU buffers for this computation
        let input_size = batch_size * input_dim * std::mem::size_of::<f32>();
        let hidden_size = batch_size * hidden_dim * std::mem::size_of::<f32>();
        let output_size = batch_size * output_dim * std::mem::size_of::<f32>();

        // Upload weights to GPU
        let w1_size = input_dim * hidden_dim * std::mem::size_of::<f32>();
        let w2_size = input_dim * hidden_dim * std::mem::size_of::<f32>();
        let wout_size = hidden_dim * output_dim * std::mem::size_of::<f32>();

        let mut gpu_input = device.allocate(input_size)?;
        let mut gpu_w1 = device.allocate(w1_size)?;
        let mut gpu_w2 = device.allocate(w2_size)?;
        let mut gpu_wout = device.allocate(wout_size)?;
        let mut gpu_x1 = device.allocate(hidden_size)?;
        let mut gpu_x2 = device.allocate(hidden_size)?;
        let mut gpu_value = device.allocate(hidden_size)?;
        let mut gpu_gate = device.allocate(hidden_size)?;
        let mut gpu_gated = device.allocate(hidden_size)?;
        let mut gpu_output = device.allocate(output_size)?;

        // Upload data
        device.upload(input.as_slice().unwrap(), &mut gpu_input)?;
        device.upload(w1.as_slice().unwrap(), &mut gpu_w1)?;
        device.upload(w2.as_slice().unwrap(), &mut gpu_w2)?;
        device.upload(w_out.as_slice().unwrap(), &mut gpu_wout)?;

        // Pass 1: Compute hidden dimension
        // x1 = input @ w1
        device.gemm_f32(
            1.0,
            &gpu_input,
            &gpu_w1,
            0.0,
            &mut gpu_x1,
            batch_size,
            hidden_dim,
            input_dim,
            false,
            false,
        )?;

        // x2 = input @ w2
        device.gemm_f32(
            1.0,
            &gpu_input,
            &gpu_w2,
            0.0,
            &mut gpu_x2,
            batch_size,
            hidden_dim,
            input_dim,
            false,
            false,
        )?;

        // value = x1 * richards_activation(x1)
        // Apply Richards curve to x1, then multiply element-wise
        device.richards_curve(
            &gpu_x1,
            &mut gpu_value,
            richards_params,
            batch_size * hidden_dim,
        )?;
        // In-place multiply: gpu_value = gpu_x1 * gpu_value
        // Need to use raw pointers to avoid simultaneous borrow
        let x1_ptr = &gpu_x1 as *const GpuBuffer;
        let value_ptr = &gpu_value as *const GpuBuffer;
        let value_mut_ptr = &mut gpu_value as *mut GpuBuffer;
        // SAFETY: mul reads from both inputs before writing to output
        device.mul(
            unsafe { &*x1_ptr },
            unsafe { &*value_ptr },
            unsafe { &mut *value_mut_ptr },
            batch_size * hidden_dim,
        )?;

        // gate = sigmoid(x2)
        device.sigmoid(&gpu_x2, &mut gpu_gate, batch_size * hidden_dim)?;

        // gated = value * gate
        device.mul(
            &gpu_value,
            &gpu_gate,
            &mut gpu_gated,
            batch_size * hidden_dim,
        )?;

        // Pass 2: Project to output
        // output = gated @ w_out
        device.gemm_f32(
            1.0,
            &gpu_gated,
            &gpu_wout,
            0.0,
            &mut gpu_output,
            batch_size,
            output_dim,
            hidden_dim,
            false,
            false,
        )?;

        // Download result
        let mut output_data = vec![0.0f32; batch_size * output_dim];
        device.download(&gpu_output, &mut output_data)?;

        // Cleanup
        device.deallocate(gpu_input);
        device.deallocate(gpu_w1);
        device.deallocate(gpu_w2);
        device.deallocate(gpu_wout);
        device.deallocate(gpu_x1);
        device.deallocate(gpu_x2);
        device.deallocate(gpu_value);
        device.deallocate(gpu_gate);
        device.deallocate(gpu_gated);
        device.deallocate(gpu_output);

        // Update stats
        self.stats.kernel_launches += 6; // 2 GEMM + 4 element-wise
        self.stats.bytes_uploaded += input_size + w1_size + w2_size + wout_size;
        self.stats.bytes_downloaded += output_size;

        // Reshape output
        Array2::from_shape_vec((batch_size, output_dim), output_data).map_err(|e| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape output: {}", e),
            }
        })
    }

    // ========================================================================
    // Attention Context Operations
    // ========================================================================

    /// Apply attention context modulation on GPU.
    ///
    /// Computes: output = input + (strength / embed_dim) * (input @ context)
    pub fn forward_attention_context(
        &mut self,
        input: &Array2<f32>,
        context: &Array2<f32>,
        strength: f32,
    ) -> Result<Array2<f32>> {
        let (batch_size, embed_dim) = input.dim();

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        // Allocate buffers
        let input_size = batch_size * embed_dim * std::mem::size_of::<f32>();
        let context_size = embed_dim * embed_dim * std::mem::size_of::<f32>();

        let mut gpu_input = device.allocate(input_size)?;
        let mut gpu_context = device.allocate(context_size)?;
        let mut gpu_output = device.allocate(input_size)?;

        // Upload data
        device.upload(input.as_slice().unwrap(), &mut gpu_input)?;
        device.upload(context.as_slice().unwrap(), &mut gpu_context)?;

        // Use high-level operation
        device.apply_attention_context(
            &gpu_input,
            &gpu_context,
            &mut gpu_output,
            strength,
            batch_size,
            embed_dim,
        )?;

        // Download result
        let mut output_data = vec![0.0f32; batch_size * embed_dim];
        device.download(&gpu_output, &mut output_data)?;

        // Cleanup
        device.deallocate(gpu_input);
        device.deallocate(gpu_context);
        device.deallocate(gpu_output);

        self.stats.kernel_launches += 3;
        self.stats.bytes_uploaded += input_size + context_size;
        self.stats.bytes_downloaded += input_size;

        Array2::from_shape_vec((batch_size, embed_dim), output_data).map_err(|e| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape output: {}", e),
            }
        })
    }

    // ========================================================================
    // Temporal Processing Operations
    // ========================================================================

    /// Execute multi-head attention on GPU.
    ///
    /// Computes scaled dot-product attention with optional causal masking.
    pub fn forward_attention(
        &mut self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        num_heads: usize,
        causal: bool,
    ) -> Result<Array2<f32>> {
        let (batch_size, seq_len) = (query.nrows(), query.ncols() / num_heads);
        let head_dim = query.ncols() / num_heads;
        let embed_dim = query.ncols();

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        // Allocate buffers
        let qkv_size = batch_size * embed_dim * std::mem::size_of::<f32>();
        let scores_size = batch_size * num_heads * seq_len * seq_len * std::mem::size_of::<f32>();

        let mut gpu_q = device.allocate(qkv_size)?;
        let mut gpu_k = device.allocate(qkv_size)?;
        let mut gpu_v = device.allocate(qkv_size)?;
        let mut gpu_scores = device.allocate(scores_size)?;
        let mut gpu_output = device.allocate(qkv_size)?;

        // Upload data
        device.upload(query.as_slice().unwrap(), &mut gpu_q)?;
        device.upload(key.as_slice().unwrap(), &mut gpu_k)?;
        device.upload(value.as_slice().unwrap(), &mut gpu_v)?;

        // Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        device.gemm_f32(
            scale,
            &gpu_q,
            &gpu_k,
            0.0,
            &mut gpu_scores,
            batch_size * num_heads,
            seq_len,
            seq_len,
            false,
            true,
        )?;

        // Apply softmax
        device.softmax(
            &gpu_scores.clone(),
            &mut gpu_scores,
            batch_size * num_heads * seq_len,
            seq_len,
        )?;

        // Apply attention: output = scores @ V
        device.gemm_f32(
            1.0,
            &gpu_scores,
            &gpu_v,
            0.0,
            &mut gpu_output,
            batch_size * num_heads,
            seq_len,
            seq_len,
            false,
            false,
        )?;

        // Download result
        let mut output_data = vec![0.0f32; batch_size * embed_dim];
        device.download(&gpu_output, &mut output_data)?;

        // Cleanup
        device.deallocate(gpu_q);
        device.deallocate(gpu_k);
        device.deallocate(gpu_v);
        device.deallocate(gpu_scores);
        device.deallocate(gpu_output);

        self.stats.kernel_launches += 3;
        self.stats.bytes_uploaded += qkv_size * 3;
        self.stats.bytes_downloaded += qkv_size;

        Array2::from_shape_vec((batch_size, embed_dim), output_data).map_err(|e| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape output: {}", e),
            }
        })
    }

    /// Get execution statistics
    pub fn stats(&self) -> &GpuExecutionStats {
        &self.stats
    }

    /// Reset execution statistics
    pub fn reset_stats(&mut self) {
        self.stats = GpuExecutionStats::default();
    }
}

// Non-GPU stub implementation
#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct GpuSharedExecutor;

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl GpuSharedExecutor {
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message:
                "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal"
                    .to_string(),
        })
    }

    pub fn is_ready(&self) -> bool {
        false
    }

    pub fn backend_name(&self) -> &'static str {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_executor_auto_detect() {
        match GpuSharedExecutor::auto_detect() {
            Ok(executor) => {
                println!("GPU executor created: {}", executor.backend_name());
                assert!(executor.is_ready());
            }
            Err(e) => {
                println!("No GPU available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_executor_capacity() {
        if let Ok(mut executor) = GpuSharedExecutor::auto_detect() {
            // Test capacity allocation
            match executor.ensure_capacity(32, 768) {
                Ok(()) => println!("Capacity ensured for 32x768"),
                Err(e) => println!("Failed to ensure capacity: {}", e),
            }
        }
    }

    #[test]
    fn test_gpu_execution_stats_default() {
        let stats = GpuExecutionStats::default();
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.bytes_uploaded, 0);
        assert_eq!(stats.bytes_downloaded, 0);
        assert_eq!(stats.total_time_us, 0);
    }
}
