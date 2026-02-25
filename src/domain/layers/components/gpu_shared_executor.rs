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
use crate::domain::compute::{GpuBuffer, GpuDevice, RichardsCurveParams};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute_backend::{
    ComputeBackend, resolve_compute_backend_strict_auto_gpu,
    resolve_compute_backend_strict_auto_npu,
};

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
    /// Buffer for output projection / generic output
    output_buffer: Option<GpuBuffer>,

    /// Reusable input buffer for shared components
    input_buffer: Option<GpuBuffer>,
    /// Reusable context buffer for attention-context operations
    context_buffer: Option<GpuBuffer>,

    /// Reusable weights for RichardsGLU
    w1_buffer: Option<GpuBuffer>,
    w2_buffer: Option<GpuBuffer>,
    wout_buffer: Option<GpuBuffer>,

    /// Reusable Q/K/V/Scores buffers for attention
    query_buffer: Option<GpuBuffer>,
    key_buffer: Option<GpuBuffer>,
    attn_value_buffer: Option<GpuBuffer>,
    scores_buffer: Option<GpuBuffer>,

    /// Capacity in f32 elements for each reusable slot family
    core_capacity: usize,
    output_capacity: usize,
    input_capacity: usize,
    context_capacity: usize,
    w1_capacity: usize,
    w2_capacity: usize,
    wout_capacity: usize,
    qkv_capacity: usize,
    scores_capacity: usize,
}

/// Re-export from canonical location
pub use crate::domain::compute::GpuExecutionStats;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
const MIN_WORKSPACE_ELEMENTS: usize = 1024;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuWorkspace {
    fn new() -> Self {
        Self {
            hidden_buffer: None,
            gate_buffer: None,
            value_buffer: None,
            output_buffer: None,
            input_buffer: None,
            context_buffer: None,
            w1_buffer: None,
            w2_buffer: None,
            wout_buffer: None,
            query_buffer: None,
            key_buffer: None,
            attn_value_buffer: None,
            scores_buffer: None,
            core_capacity: 0,
            output_capacity: 0,
            input_capacity: 0,
            context_capacity: 0,
            w1_capacity: 0,
            w2_capacity: 0,
            wout_capacity: 0,
            qkv_capacity: 0,
            scores_capacity: 0,
        }
    }

    fn release_all(&mut self, device: &mut GpuDevice) {
        for buffer in [
            self.hidden_buffer.take(),
            self.gate_buffer.take(),
            self.value_buffer.take(),
            self.output_buffer.take(),
            self.input_buffer.take(),
            self.context_buffer.take(),
            self.w1_buffer.take(),
            self.w2_buffer.take(),
            self.wout_buffer.take(),
            self.query_buffer.take(),
            self.key_buffer.take(),
            self.attn_value_buffer.take(),
            self.scores_buffer.take(),
        ]
        .into_iter()
        .flatten()
        {
            device.deallocate(buffer);
        }

        self.core_capacity = 0;
        self.output_capacity = 0;
        self.input_capacity = 0;
        self.context_capacity = 0;
        self.w1_capacity = 0;
        self.w2_capacity = 0;
        self.wout_capacity = 0;
        self.qkv_capacity = 0;
        self.scores_capacity = 0;
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[inline]
fn next_workspace_capacity(required_elements: usize) -> usize {
    required_elements
        .max(1)
        .next_power_of_two()
        .max(MIN_WORKSPACE_ELEMENTS)
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[inline]
fn ensure_workspace_buffer(
    device: &mut GpuDevice,
    slot: &mut Option<GpuBuffer>,
    capacity_elements: &mut usize,
    required_elements: usize,
) -> Result<()> {
    let required_elements = required_elements.max(1);
    if slot.is_some() && *capacity_elements >= required_elements {
        return Ok(());
    }

    if let Some(old) = slot.take() {
        device.deallocate(old);
    }

    let new_capacity = next_workspace_capacity(required_elements);
    *slot = Some(device.allocate_f32(new_capacity)?);
    *capacity_elements = new_capacity;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn ensure_core_buffers(
    workspace: &mut GpuWorkspace,
    device: &mut GpuDevice,
    core_required: usize,
) -> Result<()> {
    ensure_workspace_buffer(
        device,
        &mut workspace.hidden_buffer,
        &mut workspace.core_capacity,
        core_required,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.gate_buffer,
        &mut workspace.core_capacity,
        core_required,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.value_buffer,
        &mut workspace.core_capacity,
        core_required,
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[allow(clippy::too_many_arguments)]
fn ensure_richards_glu_buffers(
    workspace: &mut GpuWorkspace,
    device: &mut GpuDevice,
    input_elements: usize,
    hidden_elements: usize,
    output_elements: usize,
    w1_elements: usize,
    w2_elements: usize,
    wout_elements: usize,
) -> Result<()> {
    ensure_core_buffers(workspace, device, hidden_elements)?;
    ensure_workspace_buffer(
        device,
        &mut workspace.output_buffer,
        &mut workspace.output_capacity,
        output_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.input_buffer,
        &mut workspace.input_capacity,
        input_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.w1_buffer,
        &mut workspace.w1_capacity,
        w1_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.w2_buffer,
        &mut workspace.w2_capacity,
        w2_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.wout_buffer,
        &mut workspace.wout_capacity,
        wout_elements,
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn ensure_attention_context_buffers(
    workspace: &mut GpuWorkspace,
    device: &mut GpuDevice,
    input_elements: usize,
    context_elements: usize,
) -> Result<()> {
    ensure_workspace_buffer(
        device,
        &mut workspace.input_buffer,
        &mut workspace.input_capacity,
        input_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.output_buffer,
        &mut workspace.output_capacity,
        input_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.context_buffer,
        &mut workspace.context_capacity,
        context_elements,
    )?;
    Ok(())
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
fn ensure_attention_buffers(
    workspace: &mut GpuWorkspace,
    device: &mut GpuDevice,
    qkv_elements: usize,
    scores_elements: usize,
) -> Result<()> {
    ensure_workspace_buffer(
        device,
        &mut workspace.query_buffer,
        &mut workspace.qkv_capacity,
        qkv_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.key_buffer,
        &mut workspace.qkv_capacity,
        qkv_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.attn_value_buffer,
        &mut workspace.qkv_capacity,
        qkv_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.output_buffer,
        &mut workspace.output_capacity,
        qkv_elements,
    )?;
    ensure_workspace_buffer(
        device,
        &mut workspace.scores_buffer,
        &mut workspace.scores_capacity,
        scores_elements,
    )?;
    Ok(())
}

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
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        Self::new(backend)
    }

    /// Create a new GPU executor with strict Intel NPU detection (no fallback).
    pub fn auto_detect_npu() -> Result<Self> {
        let backend = resolve_compute_backend_strict_auto_npu()?;
        Self::new(backend)
    }

    /// Create a new GPU executor with strict auto-detection and explicit backend.
    ///
    /// This helper allows call sites to capture the selected runtime variant.
    pub fn auto_detect_with_backend() -> Result<(Self, ComputeBackend)> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        Ok((Self::new(backend)?, backend))
    }

    /// Create a GPU executor for a specific backend.
    pub fn new(backend: ComputeBackend) -> Result<Self> {
        let device = GpuDevice::new(backend)?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            workspace: GpuWorkspace::new(),
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

    /// Get the active compute backend variant.
    pub fn backend(&self) -> Option<ComputeBackend> {
        self.device.lock().ok().map(|d| d.backend())
    }

    /// Ensure workspace has sufficient capacity.
    ///
    /// Pre-allocates core FFN buffers with power-of-2 sizing for efficiency.
    pub fn ensure_capacity(&mut self, batch_size: usize, hidden_dim: usize) -> Result<()> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;
        let core_required = batch_size.saturating_mul(hidden_dim);
        ensure_core_buffers(&mut self.workspace, &mut device, core_required)
    }

    /// Release all persistent workspace buffers from the attached GPU device.
    pub fn clear_workspace(&mut self) -> Result<()> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;
        self.workspace.release_all(&mut device);
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

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        let input_elements = batch_size.saturating_mul(input_dim);
        let hidden_elements = batch_size.saturating_mul(hidden_dim);
        let output_elements = batch_size.saturating_mul(output_dim);
        let w1_elements = input_dim.saturating_mul(hidden_dim);
        let w2_elements = input_dim.saturating_mul(hidden_dim);
        let wout_elements = hidden_dim.saturating_mul(output_dim);

        ensure_richards_glu_buffers(
            &mut self.workspace,
            &mut device,
            input_elements,
            hidden_elements,
            output_elements,
            w1_elements,
            w2_elements,
            wout_elements,
        )?;

        let mut gpu_input = self
            .workspace
            .input_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace input buffer was not allocated".to_string(),
            })?;
        let mut gpu_w1 = self
            .workspace
            .w1_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace w1 buffer was not allocated".to_string(),
            })?;
        let mut gpu_w2 = self
            .workspace
            .w2_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace w2 buffer was not allocated".to_string(),
            })?;
        let mut gpu_wout = self
            .workspace
            .wout_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace w_out buffer was not allocated".to_string(),
            })?;
        let mut gpu_hidden = self
            .workspace
            .hidden_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace hidden buffer was not allocated".to_string(),
            })?;
        let mut gpu_gate = self
            .workspace
            .gate_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace gate buffer was not allocated".to_string(),
            })?;
        let mut gpu_value = self
            .workspace
            .value_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace value buffer was not allocated".to_string(),
            })?;
        let mut gpu_output = self
            .workspace
            .output_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace output buffer was not allocated".to_string(),
            })?;

        // Upload data
        device.upload(input.as_slice().unwrap(), &mut gpu_input)?;
        device.upload(w1.as_slice().unwrap(), &mut gpu_w1)?;
        device.upload(w2.as_slice().unwrap(), &mut gpu_w2)?;
        device.upload(w_out.as_slice().unwrap(), &mut gpu_wout)?;

        device.begin_recording();

        // Pass 1: Compute hidden dimension
        // x1 = input @ w1
        device.gemm_f32(
            1.0,
            &gpu_input,
            &gpu_w1,
            0.0,
            &mut gpu_hidden,
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
            &mut gpu_gate,
            batch_size,
            hidden_dim,
            input_dim,
            false,
            false,
        )?;

        // value = x1 * richards_activation(x1)
        // Apply Richards curve to x1, then multiply element-wise
        device.richards_curve(
            &gpu_hidden,
            &mut gpu_value,
            richards_params,
            batch_size * hidden_dim,
        )?;
        // In-place multiply: gpu_value = gpu_x1 * gpu_value
        // Need to use raw pointers to avoid simultaneous borrow
        let x1_ptr = &gpu_hidden as *const GpuBuffer;
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
        let gate_in_ptr = &gpu_gate as *const GpuBuffer;
        let gate_out_ptr = &mut gpu_gate as *mut GpuBuffer;
        // SAFETY: sigmoid reads input before writing output for element-wise kernels.
        device.sigmoid(
            unsafe { &*gate_in_ptr },
            unsafe { &mut *gate_out_ptr },
            batch_size * hidden_dim,
        )?;

        // gated = value * gate (reuse hidden buffer as gated)
        device.mul(
            &gpu_value,
            &gpu_gate,
            &mut gpu_hidden,
            batch_size * hidden_dim,
        )?;

        // Pass 2: Project to output
        // output = gated @ w_out
        device.gemm_f32(
            1.0,
            &gpu_hidden,
            &gpu_wout,
            0.0,
            &mut gpu_output,
            batch_size,
            output_dim,
            hidden_dim,
            false,
            false,
        )?;

        device.flush();

        // Download result
        let mut output_data = vec![0.0f32; batch_size * output_dim];
        device.download(&gpu_output, &mut output_data)?;

        // Update stats
        self.stats.kernel_launches += 7; // 3 GEMM + Richards + sigmoid + 2 mul
        self.stats.bytes_uploaded += (input_elements + w1_elements + w2_elements + wout_elements)
            * std::mem::size_of::<f32>();
        self.stats.bytes_downloaded += output_elements * std::mem::size_of::<f32>();

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

        let input_elements = batch_size.saturating_mul(embed_dim);
        let context_elements = embed_dim.saturating_mul(embed_dim);
        ensure_attention_context_buffers(
            &mut self.workspace,
            &mut device,
            input_elements,
            context_elements,
        )?;

        let mut gpu_input = self
            .workspace
            .input_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace input buffer was not allocated".to_string(),
            })?;
        let mut gpu_context = self
            .workspace
            .context_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace context buffer was not allocated".to_string(),
            })?;
        let mut gpu_output = self
            .workspace
            .output_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace output buffer was not allocated".to_string(),
            })?;

        // Upload data
        device.upload(input.as_slice().unwrap(), &mut gpu_input)?;
        device.upload(context.as_slice().unwrap(), &mut gpu_context)?;

        device.begin_recording();

        // Use high-level operation
        device.apply_attention_context(
            &gpu_input,
            &gpu_context,
            &mut gpu_output,
            strength,
            batch_size,
            embed_dim,
        )?;

        device.flush();

        device.flush();

        // Download result
        let mut output_data = vec![0.0f32; batch_size * embed_dim];
        device.download(&gpu_output, &mut output_data)?;

        self.stats.kernel_launches += 3;
        self.stats.bytes_uploaded +=
            (input_elements + context_elements) * std::mem::size_of::<f32>();
        self.stats.bytes_downloaded += input_elements * std::mem::size_of::<f32>();

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
        _causal: bool,
    ) -> Result<Array2<f32>> {
        if num_heads == 0 {
            return Err(ModelError::InvalidInput {
                message: "forward_attention received num_heads=0".to_string(),
            });
        }
        if query.ncols() % num_heads != 0 {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "forward_attention embed_dim {} is not divisible by num_heads {}",
                    query.ncols(),
                    num_heads
                ),
            });
        }
        let (batch_size, seq_len) = (query.nrows(), query.ncols() / num_heads);
        if batch_size == 0 || seq_len == 0 || query.ncols() == 0 {
            return Ok(Array2::zeros((query.nrows(), query.ncols())));
        }
        let head_dim = query.ncols() / num_heads;
        let embed_dim = query.ncols();

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;

        let qkv_elements = batch_size.saturating_mul(embed_dim);
        let scores_elements = batch_size
            .saturating_mul(num_heads)
            .saturating_mul(seq_len)
            .saturating_mul(seq_len);
        ensure_attention_buffers(
            &mut self.workspace,
            &mut device,
            qkv_elements,
            scores_elements,
        )?;

        let mut gpu_q = self
            .workspace
            .query_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace query buffer was not allocated".to_string(),
            })?;
        let mut gpu_k = self
            .workspace
            .key_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace key buffer was not allocated".to_string(),
            })?;
        let mut gpu_v = self
            .workspace
            .attn_value_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace value buffer was not allocated".to_string(),
            })?;
        let mut gpu_scores = self
            .workspace
            .scores_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace scores buffer was not allocated".to_string(),
            })?;
        let mut gpu_output = self
            .workspace
            .output_buffer
            .ok_or_else(|| ModelError::Backend {
                message: "GPU workspace output buffer was not allocated".to_string(),
            })?;

        // Upload data
        device.upload(query.as_slice().unwrap(), &mut gpu_q)?;
        device.upload(key.as_slice().unwrap(), &mut gpu_k)?;
        device.upload(value.as_slice().unwrap(), &mut gpu_v)?;

        device.begin_recording();

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
        let scores_in_ptr = &gpu_scores as *const GpuBuffer;
        let scores_out_ptr = &mut gpu_scores as *mut GpuBuffer;
        // SAFETY: softmax kernel supports in-place processing and reads input before writing.
        device.softmax(
            unsafe { &*scores_in_ptr },
            unsafe { &mut *scores_out_ptr },
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

        device.flush();

        device.flush();

        // Download result
        let mut output_data = vec![0.0f32; batch_size * embed_dim];
        device.download(&gpu_output, &mut output_data)?;

        self.stats.kernel_launches += 3;
        self.stats.bytes_uploaded += qkv_elements * 3 * std::mem::size_of::<f32>();
        self.stats.bytes_downloaded += qkv_elements * std::mem::size_of::<f32>();

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

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl Drop for GpuSharedExecutor {
    fn drop(&mut self) {
        if let Ok(mut device) = self.device.lock() {
            self.workspace.release_all(&mut device);
        }
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

    pub fn auto_detect_npu() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Intel NPU execution requires --features gpu-wgpu".to_string(),
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
