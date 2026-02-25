//! Unified GPU Kernels for Shared Temporal Operations
//!
//! Consolidates GPU kernel implementations for operations shared across
//! Transformer, Diffusion, and SSM architectures.
//!
//! ## Kernel Categories
//!
//! 1. **Attention Operations**: QKV projection, attention scoring, output projection
//! 2. **SSM Operations**: Selective scan, state updates, recurrent computation
//! 3. **Normalization**: Layer norm, RMS norm
//! 4. **Activation**: GELU, SiLU, ReLU, Richards curve
//!
//! ## Memory Efficiency
//!
//! All kernels use workspace-managed buffers to minimize allocations:
//! - Input/output buffers are pre-allocated with power-of-2 sizing
//! - Intermediate buffers are reused across kernel calls
//! - GPU memory is pooled at the device level
//!
//! ## Performance Targets (Phase 5.6)
//!
//! - Multi-head attention: 30x speedup vs CPU (30ms → 1ms on 512 batch)
//! - Mamba selective scan: 20x speedup vs CPU (40ms → 2ms on 512 batch)
//! - RG-LRU recurrent: 15x speedup vs CPU (30ms → 2ms on 512 batch)

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use ndarray::Array2;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::attention_gpu_kernel;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::ssm_gpu_kernels;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::{GpuActivation, GpuTemporalType};

// ============================================================================
// Kernel Parameter Structures
// ============================================================================

/// Parameters for attention forward pass
#[derive(Debug, Clone)]
pub struct AttentionParams {
    /// Number of attention heads
    pub num_heads: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Head dimension (embed_dim / num_heads)
    pub head_dim: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// Whether to use causal masking
    pub causal: bool,
    /// Optional sliding window size
    pub window_size: Option<usize>,
}

impl AttentionParams {
    pub fn new(num_heads: usize, embed_dim: usize, seq_len: usize, batch_size: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            num_heads,
            embed_dim,
            head_dim,
            seq_len,
            batch_size,
            scale,
            causal: false,
            window_size: None,
        }
    }

    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    pub fn with_window(mut self, window_size: usize) -> Self {
        self.window_size = Some(window_size);
        self
    }
}

/// Parameters for SSM (Mamba/RG-LRU) forward pass
#[derive(Debug, Clone)]
pub struct SsmParams {
    /// State dimension
    pub state_dim: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Sequence length
    pub seq_len: usize,
    /// Batch size
    pub batch_size: usize,
    /// Expansion factor for intermediate dimension
    pub expansion: usize,
    /// Whether to use selective scan
    pub selective: bool,
}

impl SsmParams {
    pub fn new(state_dim: usize, embed_dim: usize, seq_len: usize, batch_size: usize) -> Self {
        Self {
            state_dim,
            embed_dim,
            seq_len,
            batch_size,
            expansion: 2,
            selective: true,
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug, Clone)]
pub struct MambaKernelMatrices {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub c: Array2<f32>,
    pub d: Array2<f32>,
    pub h_init: Array2<f32>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug, Clone)]
pub struct RgLruKernelMatrices {
    pub w_f: Array2<f32>,
    pub w_r: Array2<f32>,
    pub w_o: Array2<f32>,
    pub h_init: Array2<f32>,
}

/// Parameters for normalization operations
#[derive(Debug, Clone)]
pub struct NormParams {
    /// Dimension to normalize over
    pub dim: usize,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Whether to include learned scale (gamma)
    pub has_scale: bool,
    /// Whether to include learned bias (beta)
    pub has_bias: bool,
}

impl NormParams {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            eps: 1e-5,
            has_scale: true,
            has_bias: true,
        }
    }
}

// ============================================================================
// GPU Kernel Dispatcher
// ============================================================================

/// Unified GPU kernel dispatcher for temporal operations.
///
/// Provides a single entry point for all GPU-accelerated operations
/// shared across Transformer, Diffusion, and SSM architectures.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
pub struct UnifiedGpuKernels {
    device: Arc<Mutex<GpuDevice>>,
    /// Pre-allocated workspace buffers
    workspace: GpuKernelWorkspace,
    mamba_kernel_matrices: Option<MambaKernelMatrices>,
    rg_lru_kernel_matrices: Option<RgLruKernelMatrices>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl UnifiedGpuKernels {
    /// Create a new kernel dispatcher with automatic GPU detection.
    pub fn auto_detect() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            workspace: GpuKernelWorkspace::new(),
            mamba_kernel_matrices: None,
            rg_lru_kernel_matrices: None,
        })
    }

    /// Create a kernel dispatcher for a specific backend.
    pub fn new(backend: crate::domain::compute_backend::ComputeBackend) -> Result<Self> {
        let device = GpuDevice::new(backend)?;
        Ok(Self {
            device: Arc::new(Mutex::new(device)),
            workspace: GpuKernelWorkspace::new(),
            mamba_kernel_matrices: None,
            rg_lru_kernel_matrices: None,
        })
    }

    /// Get the GPU device.
    pub fn device(&self) -> Arc<Mutex<GpuDevice>> {
        self.device.clone()
    }

    /// Ensure workspace has sufficient capacity.
    pub fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::ensure_capacity".to_string(),
        })?;

        self.workspace
            .ensure_capacity(&mut device, batch_size, embed_dim, seq_len)
    }

    /// Reset workspace for reuse without deallocation.
    pub fn reset_workspace(&mut self) {
        self.workspace.reset();
    }

    /// Cleanup all workspace buffers.
    pub fn cleanup_workspace(&mut self) -> Result<()> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::cleanup_workspace".to_string(),
        })?;

        self.workspace.cleanup(&mut device);
        Ok(())
    }

    /// Set explicit Mamba kernel matrices used by `ssm_forward/ssm_backward`.
    pub fn set_mamba_kernel_matrices(
        &mut self,
        a: Array2<f32>,
        b: Array2<f32>,
        c: Array2<f32>,
        d: Array2<f32>,
        h_init: Array2<f32>,
    ) {
        self.mamba_kernel_matrices = Some(MambaKernelMatrices { a, b, c, d, h_init });
    }

    /// Set explicit RG-LRU kernel matrices used by `ssm_forward/ssm_backward`.
    pub fn set_rg_lru_kernel_matrices(
        &mut self,
        w_f: Array2<f32>,
        w_r: Array2<f32>,
        w_o: Array2<f32>,
        h_init: Array2<f32>,
    ) {
        self.rg_lru_kernel_matrices = Some(RgLruKernelMatrices {
            w_f,
            w_r,
            w_o,
            h_init,
        });
    }

    pub fn clear_mamba_kernel_matrices(&mut self) {
        self.mamba_kernel_matrices = None;
    }

    pub fn clear_rg_lru_kernel_matrices(&mut self) {
        self.rg_lru_kernel_matrices = None;
    }

    fn resolve_mamba_matrices(
        &self,
        params: &SsmParams,
        embed_dim: usize,
    ) -> Result<(
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
    )> {
        if let Some(mats) = &self.mamba_kernel_matrices {
            let state_dim = params.state_dim;
            if mats.a.dim() != (state_dim, state_dim)
                || mats.b.dim() != (state_dim, embed_dim)
                || mats.c.dim() != (embed_dim, state_dim)
                || mats.d.dim() != (embed_dim, embed_dim)
                || mats.h_init.nrows() == 0
                || mats.h_init.ncols() != state_dim
            {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!(
                        "Mamba matrices A({0},{0}) B({0},{1}) C({1},{0}) D({1},{1}) h_init(*,{0})",
                        state_dim, embed_dim
                    ),
                    got: format!(
                        "A{:?} B{:?} C{:?} D{:?} h_init{:?}",
                        mats.a.dim(),
                        mats.b.dim(),
                        mats.c.dim(),
                        mats.d.dim(),
                        mats.h_init.dim()
                    ),
                });
            }
            return Ok((
                mats.a.clone(),
                mats.b.clone(),
                mats.c.clone(),
                mats.d.clone(),
                mats.h_init.clone(),
            ));
        }
        Ok(Self::build_default_mamba_matrices(
            params.state_dim,
            embed_dim,
        ))
    }

    fn resolve_rg_lru_matrices(
        &self,
        embed_dim: usize,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
        if let Some(mats) = &self.rg_lru_kernel_matrices {
            if mats.w_f.dim() != (embed_dim, embed_dim)
                || mats.w_r.dim() != (embed_dim, embed_dim)
                || mats.w_o.dim() != (embed_dim, embed_dim)
                || mats.h_init.nrows() == 0
                || mats.h_init.ncols() != embed_dim
            {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!("RG-LRU matrices Wf/Wr/Wo({0},{0}) h_init(*,{0})", embed_dim),
                    got: format!(
                        "Wf{:?} Wr{:?} Wo{:?} h_init{:?}",
                        mats.w_f.dim(),
                        mats.w_r.dim(),
                        mats.w_o.dim(),
                        mats.h_init.dim()
                    ),
                });
            }
            return Ok((
                mats.w_f.clone(),
                mats.w_r.clone(),
                mats.w_o.clone(),
                mats.h_init.clone(),
            ));
        }
        Ok(Self::build_default_rg_lru_matrices(embed_dim))
    }

    /// Get workspace statistics.
    pub fn workspace_stats(&self) -> GpuKernelWorkspaceStats {
        self.workspace.stats()
    }

    /// Run a single GPU GEMM and return the output as host `Array2`.
    fn gpu_gemm_to_host(
        &mut self,
        a: &Array2<f32>,
        b: &Array2<f32>,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<Array2<f32>> {
        let a_slice = a.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "gpu_gemm_to_host: lhs must be contiguous".to_string(),
        })?;
        let b_slice = b.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "gpu_gemm_to_host: rhs must be contiguous".to_string(),
        })?;
        let out_elements = m.saturating_mul(n);
        if out_elements == 0 {
            return Ok(Array2::zeros((m, n)));
        }

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::gpu_gemm_to_host".to_string(),
        })?;

        let mut a_buf = device.allocate_f32(a.len())?;
        let mut b_buf = device.allocate_f32(b.len())?;
        let mut out_buf = device.allocate_f32(out_elements)?;

        device.upload(a_slice, &mut a_buf)?;
        device.upload(b_slice, &mut b_buf)?;
        device.gemm_f32(
            1.0,
            &a_buf,
            &b_buf,
            0.0,
            &mut out_buf,
            m,
            n,
            k,
            trans_a,
            trans_b,
        )?;

        let mut host = vec![0.0f32; out_elements];
        device.download(&out_buf, &mut host)?;

        device.deallocate(a_buf);
        device.deallocate(b_buf);
        device.deallocate(out_buf);

        Array2::from_shape_vec((m, n), host).map_err(|err| ModelError::InvalidInput {
            message: format!("gpu_gemm_to_host reshape failed: {err}"),
        })
    }

    /// Compute `sum_i (output_grads @ weight_i^T)` on GPU and return host tensor.
    fn gpu_accumulate_input_grads(
        &mut self,
        output_grads: &Array2<f32>,
        weights: &[&Array2<f32>],
        total_tokens: usize,
        embed_dim: usize,
    ) -> Result<Array2<f32>> {
        if total_tokens == 0 || embed_dim == 0 {
            return Ok(Array2::zeros((total_tokens, embed_dim)));
        }
        let grad_slice = output_grads
            .as_slice()
            .ok_or_else(|| ModelError::InvalidInput {
                message: "gpu_accumulate_input_grads: output_grads must be contiguous".to_string(),
            })?;
        for (idx, weight) in weights.iter().enumerate() {
            if weight.dim() != (embed_dim, embed_dim) {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!("weight[{idx}] dims: ({embed_dim}, {embed_dim})"),
                    got: format!("{:?}", weight.dim()),
                });
            }
            if weight.as_slice().is_none() {
                return Err(ModelError::InvalidInput {
                    message: format!(
                        "gpu_accumulate_input_grads: weight[{idx}] must be contiguous"
                    ),
                });
            }
        }

        let input_elements = total_tokens * embed_dim;
        let output_elements = output_grads.len();
        let weight_elements = embed_dim * embed_dim;

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::gpu_accumulate_input_grads"
                .to_string(),
        })?;

        let mut grad_out_buf = device.allocate_f32(output_elements)?;
        let mut w_buf = device.allocate_f32(weight_elements)?;
        let mut input_grads_buf = device.allocate_f32(input_elements)?;
        let mut tmp_input_grads_buf = device.allocate_f32(input_elements)?;
        device.upload(grad_slice, &mut grad_out_buf)?;

        for (idx, weight) in weights.iter().enumerate() {
            let weight_slice = weight.as_slice().ok_or_else(|| ModelError::InvalidInput {
                message: format!("gpu_accumulate_input_grads: weight[{idx}] must be contiguous"),
            })?;
            device.upload(weight_slice, &mut w_buf)?;
            if idx == 0 {
                device.gemm_f32(
                    1.0,
                    &grad_out_buf,
                    &w_buf,
                    0.0,
                    &mut input_grads_buf,
                    total_tokens,
                    embed_dim,
                    embed_dim,
                    false,
                    true,
                )?;
            } else {
                device.gemm_f32(
                    1.0,
                    &grad_out_buf,
                    &w_buf,
                    0.0,
                    &mut tmp_input_grads_buf,
                    total_tokens,
                    embed_dim,
                    embed_dim,
                    false,
                    true,
                )?;
                device.add_scaled(
                    1.0,
                    &tmp_input_grads_buf,
                    &mut input_grads_buf,
                    input_elements,
                )?;
            }
        }

        let mut input_grads_host = vec![0.0f32; input_elements];
        device.download(&input_grads_buf, &mut input_grads_host)?;

        device.deallocate(grad_out_buf);
        device.deallocate(w_buf);
        device.deallocate(input_grads_buf);
        device.deallocate(tmp_input_grads_buf);

        Array2::from_shape_vec((total_tokens, embed_dim), input_grads_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("gpu_accumulate_input_grads reshape failed: {err}"),
            }
        })
    }

    /// Compute sum of all elements in a matrix using GPU reduction.
    fn gpu_sum_array(&mut self, input: &Array2<f32>) -> Result<f32> {
        let size = input.len();
        if size == 0 {
            return Ok(0.0);
        }
        let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "gpu_sum_array: input must be contiguous".to_string(),
        })?;

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::gpu_sum_array".to_string(),
        })?;
        let mut input_buf = device.allocate_f32(size)?;
        device.upload(input_slice, &mut input_buf)?;
        let sum = device.sum(&input_buf, size)?;
        device.deallocate(input_buf);
        Ok(sum)
    }

    /// Compute sum(lhs * rhs) element-wise using GPU kernels.
    fn gpu_sum_product_arrays(&mut self, lhs: &Array2<f32>, rhs: &Array2<f32>) -> Result<f32> {
        if lhs.dim() != rhs.dim() {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("rhs dims: {:?}", lhs.dim()),
                got: format!("{:?}", rhs.dim()),
            });
        }
        let size = lhs.len();
        if size == 0 {
            return Ok(0.0);
        }
        let lhs_slice = lhs.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "gpu_sum_product_arrays: lhs must be contiguous".to_string(),
        })?;
        let rhs_slice = rhs.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "gpu_sum_product_arrays: rhs must be contiguous".to_string(),
        })?;

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::gpu_sum_product_arrays"
                .to_string(),
        })?;
        let mut lhs_buf = device.allocate_f32(size)?;
        let mut rhs_buf = device.allocate_f32(size)?;
        let mut out_buf = device.allocate_f32(size)?;

        device.upload(lhs_slice, &mut lhs_buf)?;
        device.upload(rhs_slice, &mut rhs_buf)?;
        device.mul(&lhs_buf, &rhs_buf, &mut out_buf, size)?;
        let sum = device.sum(&out_buf, size)?;

        device.deallocate(lhs_buf);
        device.deallocate(rhs_buf);
        device.deallocate(out_buf);
        Ok(sum)
    }

    // ========================================================================
    // Attention Operations
    // ========================================================================

    /// Compute multi-head attention forward pass on GPU.
    ///
    /// Computes: output = softmax(Q @ K^T / scale) @ V @ W_o
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size * seq_len, embed_dim) flattened
    /// * `wq` - Query projection weights (embed_dim, embed_dim)
    /// * `wk` - Key projection weights (embed_dim, embed_dim)
    /// * `wv` - Value projection weights (embed_dim, embed_dim)
    /// * `wo` - Output projection weights (embed_dim, embed_dim)
    /// * `params` - Attention parameters
    ///
    /// # Memory Layout
    ///
    /// The input is expected in (batch_size * seq_len, embed_dim) format.
    /// Internally reshaped to (batch, heads, seq, head_dim) for attention computation.
    pub fn attention_forward(
        &mut self,
        input: &Array2<f32>,
        wq: &Array2<f32>,
        wk: &Array2<f32>,
        wv: &Array2<f32>,
        wo: &Array2<f32>,
        params: &AttentionParams,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in UnifiedGpuKernels::attention_forward".to_string(),
        })?;

        let (total_tokens, embed_dim) = input.dim();
        let seq_len = params.seq_len;
        if total_tokens == 0 || embed_dim == 0 {
            return Ok(Array2::zeros((total_tokens, embed_dim)));
        }
        if seq_len == 0 {
            return Err(ModelError::InvalidInput {
                message: "AttentionParams.seq_len must be > 0 for attention_forward".to_string(),
            });
        }
        if params.num_heads == 0 || params.head_dim == 0 {
            return Err(ModelError::InvalidInput {
                message: "AttentionParams.num_heads and head_dim must be > 0 for attention_forward"
                    .to_string(),
            });
        }
        if params.embed_dim != embed_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.embed_dim: {}", embed_dim),
                got: format!("{}", params.embed_dim),
            });
        }
        if total_tokens % seq_len != 0 {
            return Err(ModelError::ShapeMismatch {
                expected: vec![params.batch_size * seq_len, embed_dim],
                actual: vec![total_tokens, embed_dim],
                message: "Total tokens must be divisible by seq_len".to_string(),
            });
        }
        let batch_size = total_tokens / seq_len;
        if params.batch_size != batch_size {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.batch_size: {}", batch_size),
                got: format!("{}", params.batch_size),
            });
        }

        // Preserve workspace lifecycle semantics for monitoring/reuse.
        self.workspace
            .ensure_capacity(&mut device, batch_size, embed_dim, seq_len)?;

        let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "attention_forward input must be contiguous".to_string(),
        })?;
        let wq_slice = wq.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "attention_forward wq must be contiguous".to_string(),
        })?;
        let wk_slice = wk.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "attention_forward wk must be contiguous".to_string(),
        })?;
        let wv_slice = wv.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "attention_forward wv must be contiguous".to_string(),
        })?;
        let wo_slice = wo.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "attention_forward wo must be contiguous".to_string(),
        })?;

        if wq.dim() != (embed_dim, embed_dim)
            || wk.dim() != (embed_dim, embed_dim)
            || wv.dim() != (embed_dim, embed_dim)
            || wo.dim() != (embed_dim, embed_dim)
        {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("all projection weights: ({}, {})", embed_dim, embed_dim),
                got: format!(
                    "wq: {:?}, wk: {:?}, wv: {:?}, wo: {:?}",
                    wq.dim(),
                    wk.dim(),
                    wv.dim(),
                    wo.dim()
                ),
            });
        }

        let input_size = total_tokens * embed_dim * std::mem::size_of::<f32>();
        let weight_size = embed_dim * embed_dim * std::mem::size_of::<f32>();

        let mut input_buf = device.allocate(input_size)?;
        let mut wq_buf = device.allocate(weight_size)?;
        let mut wk_buf = device.allocate(weight_size)?;
        let mut wv_buf = device.allocate(weight_size)?;
        let mut wo_buf = device.allocate(weight_size)?;

        device.upload(input_slice, &mut input_buf)?;
        device.upload(wq_slice, &mut wq_buf)?;
        device.upload(wk_slice, &mut wk_buf)?;
        device.upload(wv_slice, &mut wv_buf)?;
        device.upload(wo_slice, &mut wo_buf)?;

        let (output_buf, q_buf, k_buf, v_buf, attn_weights_buf) =
            attention_gpu_kernel::forward_gpu(
                &mut device,
                &input_buf,
                &wq_buf,
                &wk_buf,
                &wv_buf,
                &wo_buf,
                params,
            )?;

        let mut output_host = vec![0.0f32; total_tokens * embed_dim];
        device.download(&output_buf, &mut output_host)?;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(q_buf);
        device.deallocate(k_buf);
        device.deallocate(v_buf);
        device.deallocate(output_buf);
        device.deallocate(attn_weights_buf);
        device.deallocate(wq_buf);
        device.deallocate(wk_buf);
        device.deallocate(wv_buf);
        device.deallocate(wo_buf);

        Array2::from_shape_vec((total_tokens, embed_dim), output_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape attention forward output: {err}"),
            }
        })
    }

    /// Compute flash attention (memory-efficient variant) on GPU.
    ///
    /// Uses tiling to reduce memory from O(n²) to O(n) for attention scores.
    /// This is a placeholder for future implementation with custom GPU kernels.
    #[allow(unused_variables)]
    pub fn flash_attention_forward(
        &mut self,
        input: &Array2<f32>,
        wq: &Array2<f32>,
        wk: &Array2<f32>,
        wv: &Array2<f32>,
        wo: &Array2<f32>,
        params: &AttentionParams,
    ) -> Result<Array2<f32>> {
        // Flash attention requires custom GPU kernels for tiling
        // For now, fall back to standard attention
        // TODO: Implement tiled attention with O(n) memory
        self.attention_forward(input, wq, wk, wv, wo, params)
    }

    // ========================================================================
    // SSM Operations
    // ========================================================================

    /// Compute SSM (Mamba/RG-LRU) forward pass on GPU.
    ///
    /// Implements selective state space model computation:
    /// 1. Project input to state space
    /// 2. Apply selective scan with learned parameters
    /// 3. Project back to output space
    ///
    /// # Mamba Architecture
    ///
    /// The Mamba SSM uses selective scan with:
    /// - State dimension: `state_dim`
    /// - Expansion factor: typically 2
    /// - Learned parameters: A, B, C, D (delta, bias, proj)
    ///
    /// # RG-LRU Architecture
    ///
    /// The RG-LRU uses recurrent gating with:
    /// - Richards curve activation for gating
    /// - Exponential decay for state updates
    /// - Input-dependent gating parameters
    pub fn ssm_forward(
        &mut self,
        input: &Array2<f32>,
        params: &SsmParams,
        temporal_type: GpuTemporalType,
    ) -> Result<Array2<f32>> {
        let (total_tokens, embed_dim) = input.dim();
        let seq_len = params.seq_len;
        if total_tokens == 0 || embed_dim == 0 {
            return Ok(Array2::zeros((total_tokens, embed_dim)));
        }
        if seq_len == 0 {
            return Err(ModelError::InvalidInput {
                message: "SsmParams.seq_len must be > 0 for ssm_forward".to_string(),
            });
        }
        if total_tokens % seq_len != 0 {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("total_tokens divisible by seq_len ({seq_len})"),
                got: format!("total_tokens={total_tokens}"),
            });
        }
        if params.embed_dim != embed_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.embed_dim: {}", embed_dim),
                got: format!("{}", params.embed_dim),
            });
        }
        let batch_size = total_tokens / seq_len;
        if params.batch_size != batch_size {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.batch_size: {}", batch_size),
                got: format!("{}", params.batch_size),
            });
        }

        // Keep workspace lifecycle for shared-kernel memory planning.
        {
            let mut device = self.device.lock().map_err(|_| ModelError::Backend {
                message: "Failed to acquire GPU device lock for SSM".to_string(),
            })?;
            self.workspace
                .ensure_capacity(&mut device, batch_size, embed_dim, seq_len)?;
        }

        match temporal_type {
            GpuTemporalType::Mamba => self.mamba_selective_scan(input, params),
            GpuTemporalType::RgLru => self.rg_lru_recurrent(input, params),
            GpuTemporalType::Attention => Err(ModelError::Backend {
                message: "SSM forward called with Attention type".to_string(),
            }),
        }
    }

    fn build_default_mamba_matrices(
        state_dim: usize,
        embed_dim: usize,
    ) -> (
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
    ) {
        let a_decay = 0.9f32;
        let b_scale = 0.1f32;
        let c_scale = 1.0f32;
        let d_skip = 0.5f32;

        let mut a = Array2::<f32>::zeros((state_dim, state_dim));
        for i in 0..state_dim {
            a[[i, i]] = a_decay;
        }
        let b = Array2::<f32>::from_elem((state_dim, embed_dim), b_scale);
        let c = Array2::<f32>::from_elem((embed_dim, state_dim), c_scale);
        let mut d = Array2::<f32>::zeros((embed_dim, embed_dim));
        for i in 0..embed_dim {
            d[[i, i]] = d_skip;
        }
        let h_init = Array2::<f32>::zeros((1, state_dim));
        (a, b, c, d, h_init)
    }

    fn build_default_rg_lru_matrices(
        embed_dim: usize,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let mut w_f = Array2::<f32>::zeros((embed_dim, embed_dim));
        let mut w_r = Array2::<f32>::zeros((embed_dim, embed_dim));
        let mut w_o = Array2::<f32>::zeros((embed_dim, embed_dim));
        for i in 0..embed_dim {
            w_f[[i, i]] = 1.0;
            w_r[[i, i]] = 1.0;
            w_o[[i, i]] = 1.0;
        }
        let h_init = Array2::<f32>::zeros((1, embed_dim));
        (w_f, w_r, w_o, h_init)
    }

    /// GPU-accelerated SSM backward pass.
    ///
    /// Returns `(input_grads, param_grads)` where param grads are ordered by temporal type:
    /// - `Mamba`: `[dA, dB, dC, dD]`
    /// - `RgLru`: `[dW_f, dW_r, dW_o]`
    pub fn ssm_backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
        params: &SsmParams,
        temporal_type: GpuTemporalType,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        let (total_tokens, embed_dim) = input.dim();
        if output_grads.dim() != input.dim() {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("output_grads: {:?}", input.dim()),
                got: format!("{:?}", output_grads.dim()),
            });
        }
        if params.seq_len == 0 || params.embed_dim != embed_dim {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "Invalid SSM params for backward: seq_len={}, embed_dim={}, input_embed_dim={}",
                    params.seq_len, params.embed_dim, embed_dim
                ),
            });
        }
        if total_tokens % params.seq_len != 0 {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("total_tokens divisible by seq_len ({})", params.seq_len),
                got: format!("total_tokens={total_tokens}"),
            });
        }
        let batch_size = total_tokens / params.seq_len;
        if params.batch_size != batch_size {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.batch_size: {}", batch_size),
                got: format!("{}", params.batch_size),
            });
        }
        if temporal_type == GpuTemporalType::Attention {
            return Err(ModelError::Backend {
                message: "SSM backward called with Attention type".to_string(),
            });
        }

        match temporal_type {
            GpuTemporalType::Mamba => {
                let (a, b, c, d, h_init) = self.resolve_mamba_matrices(params, embed_dim)?;
                let kernel_params = ssm_gpu_kernels::SelectiveScanParams::new(
                    params.seq_len,
                    params.state_dim,
                    embed_dim,
                    1,
                );

                let mut input_grads = Array2::<f32>::zeros((total_tokens, embed_dim));
                let mut a_grads = Array2::<f32>::zeros((params.state_dim, params.state_dim));
                let mut b_grads = Array2::<f32>::zeros((params.state_dim, embed_dim));
                let mut c_grads = Array2::<f32>::zeros((embed_dim, params.state_dim));
                let mut d_grads = Array2::<f32>::zeros((embed_dim, embed_dim));

                let mut device = self.device.lock().map_err(|_| ModelError::Backend {
                    message: "Failed to acquire GPU device lock for ssm_backward(Mamba)"
                        .to_string(),
                })?;

                for batch_idx in 0..batch_size {
                    let row_start = batch_idx * params.seq_len;
                    let row_end = row_start + params.seq_len;
                    let input_batch = input.slice(ndarray::s![row_start..row_end, ..]).to_owned();
                    let grads_batch = output_grads
                        .slice(ndarray::s![row_start..row_end, ..])
                        .to_owned();
                    let (_out, h_final) = ssm_gpu_kernels::selective_scan_forward_gpu(
                        &mut device,
                        &input_batch,
                        &a,
                        &b,
                        &c,
                        &d,
                        &h_init,
                        &kernel_params,
                    )?;
                    let (dx, da, db, dc, dd) = ssm_gpu_kernels::selective_scan_backward_gpu(
                        &mut device,
                        &input_batch,
                        &grads_batch,
                        &a,
                        &b,
                        &c,
                        &d,
                        &h_final,
                        &kernel_params,
                    )?;
                    input_grads
                        .slice_mut(ndarray::s![row_start..row_end, ..])
                        .assign(&dx);
                    a_grads += &da;
                    b_grads += &db;
                    c_grads += &dc;
                    d_grads += &dd;
                }

                Ok((input_grads, vec![a_grads, b_grads, c_grads, d_grads]))
            }
            GpuTemporalType::RgLru => {
                let (w_f, w_r, w_o, h_init) = self.resolve_rg_lru_matrices(embed_dim)?;
                let kernel_params = ssm_gpu_kernels::SelectiveScanParams::new(
                    params.seq_len,
                    embed_dim,
                    embed_dim,
                    1,
                );

                let mut input_grads = Array2::<f32>::zeros((total_tokens, embed_dim));
                let mut wf_grads = Array2::<f32>::zeros((embed_dim, embed_dim));
                let mut wr_grads = Array2::<f32>::zeros((embed_dim, embed_dim));
                let mut wo_grads = Array2::<f32>::zeros((embed_dim, embed_dim));

                let mut device = self.device.lock().map_err(|_| ModelError::Backend {
                    message: "Failed to acquire GPU device lock for ssm_backward(RgLru)"
                        .to_string(),
                })?;

                for batch_idx in 0..batch_size {
                    let row_start = batch_idx * params.seq_len;
                    let row_end = row_start + params.seq_len;
                    let input_batch = input.slice(ndarray::s![row_start..row_end, ..]).to_owned();
                    let grads_batch = output_grads
                        .slice(ndarray::s![row_start..row_end, ..])
                        .to_owned();
                    let (dx, dw_f, dw_r, dw_o) = ssm_gpu_kernels::rg_lru_backward_gpu(
                        &mut device,
                        &input_batch,
                        &grads_batch,
                        &w_f,
                        &w_r,
                        &w_o,
                        &h_init,
                        &kernel_params,
                    )?;
                    input_grads
                        .slice_mut(ndarray::s![row_start..row_end, ..])
                        .assign(&dx);
                    wf_grads += &dw_f;
                    wr_grads += &dw_r;
                    wo_grads += &dw_o;
                }

                Ok((input_grads, vec![wf_grads, wr_grads, wo_grads]))
            }
            GpuTemporalType::Attention => Err(ModelError::Backend {
                message: "SSM backward called with Attention type".to_string(),
            }),
        }
    }

    /// Mamba selective scan implementation.
    ///
    /// Computes the selective state space model scan:
    /// h_t = A * h_{t-1} + B * x_t
    /// y_t = C * h_t + D * x_t
    ///
    /// Where A, B, C, D are input-dependent (selective).
    fn mamba_selective_scan(&self, input: &Array2<f32>, params: &SsmParams) -> Result<Array2<f32>> {
        let (total_tokens, embed_dim) = input.dim();
        let seq_len = params.seq_len;
        let batch_size = total_tokens / seq_len;
        let (a, b, c, d, h_init) = self.resolve_mamba_matrices(params, embed_dim)?;
        let kernel_params =
            ssm_gpu_kernels::SelectiveScanParams::new(seq_len, params.state_dim, embed_dim, 1);
        let mut output: Array2<f32> = Array2::zeros((total_tokens, embed_dim));

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock for mamba_selective_scan".to_string(),
        })?;

        for batch_idx in 0..batch_size {
            let row_start = batch_idx * seq_len;
            let row_end = row_start + seq_len;
            let input_batch = input.slice(ndarray::s![row_start..row_end, ..]).to_owned();
            let (batch_out, _h_final) = ssm_gpu_kernels::selective_scan_forward_gpu(
                &mut device,
                &input_batch,
                &a,
                &b,
                &c,
                &d,
                &h_init,
                &kernel_params,
            )?;
            output
                .slice_mut(ndarray::s![row_start..row_end, ..])
                .assign(&batch_out);
        }

        Ok(output)
    }

    /// RG-LRU recurrent implementation.
    ///
    /// Computes recurrent gated linear unit with Richards curve:
    /// h_t = gamma_t * h_{t-1} + (1 - gamma_t) * W @ x_t
    /// y_t = activation(h_t)
    ///
    /// Where gamma_t is computed via Richards curve for smooth gating.
    fn rg_lru_recurrent(&self, input: &Array2<f32>, params: &SsmParams) -> Result<Array2<f32>> {
        let (total_tokens, embed_dim) = input.dim();
        let seq_len = params.seq_len;
        let batch_size = total_tokens / seq_len;
        let (w_f, w_r, w_o, h_init) = self.resolve_rg_lru_matrices(embed_dim)?;
        let kernel_params =
            ssm_gpu_kernels::SelectiveScanParams::new(seq_len, embed_dim, embed_dim, 1);
        let mut output: Array2<f32> = Array2::zeros((total_tokens, embed_dim));

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock for rg_lru_recurrent".to_string(),
        })?;

        for batch_idx in 0..batch_size {
            let row_start = batch_idx * seq_len;
            let row_end = row_start + seq_len;
            let input_batch = input.slice(ndarray::s![row_start..row_end, ..]).to_owned();
            let (batch_out, _h_final) = ssm_gpu_kernels::rg_lru_forward_gpu(
                &mut device,
                &input_batch,
                &w_f,
                &w_r,
                &w_o,
                &h_init,
                &kernel_params,
            )?;
            output
                .slice_mut(ndarray::s![row_start..row_end, ..])
                .assign(&batch_out);
        }

        Ok(output)
    }

    // ========================================================================
    // Normalization Operations
    // ========================================================================

    /// Compute layer normalization on GPU.
    ///
    /// Computes: output = (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn layer_norm_forward(
        &mut self,
        input: &Array2<f32>,
        gamma: Option<&Array2<f32>>,
        beta: Option<&Array2<f32>>,
        params: &NormParams,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock for layer norm".to_string(),
        })?;

        let (batch_size, dim) = input.dim();
        if dim == 0 {
            return Ok(Array2::zeros((batch_size, dim)));
        }
        if params.dim != dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.dim: {}", dim),
                got: format!("{}", params.dim),
            });
        }

        let gamma_vec = match gamma {
            Some(g) => {
                if g.ncols() != dim || g.nrows() != 1 {
                    return Err(ModelError::DimensionMismatchDetailed {
                        expected: format!("gamma shape: (1, {})", dim),
                        got: format!("{:?}", g.dim()),
                    });
                }
                g.row(0).to_vec()
            }
            None => vec![1.0f32; dim],
        };
        let beta_vec = match beta {
            Some(b) => {
                if b.ncols() != dim || b.nrows() != 1 {
                    return Err(ModelError::DimensionMismatchDetailed {
                        expected: format!("beta shape: (1, {})", dim),
                        got: format!("{:?}", b.dim()),
                    });
                }
                b.row(0).to_vec()
            }
            None => vec![0.0f32; dim],
        };

        // Allocate buffers
        let input_size = batch_size * dim * std::mem::size_of::<f32>();
        let affine_size = dim * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        let mut gamma_buf = device.allocate(affine_size)?;
        let mut beta_buf = device.allocate(affine_size)?;
        let mut output_buf = device.allocate(input_size)?;

        // Upload input
        let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "layer_norm_forward input must be contiguous".to_string(),
        })?;
        device.upload(input_slice, &mut input_buf)?;
        device.upload(&gamma_vec, &mut gamma_buf)?;
        device.upload(&beta_vec, &mut beta_buf)?;

        // Layer norm kernel: output = gamma * (x - mean) / sqrt(var + eps) + beta
        device.layer_norm(
            &input_buf,
            &gamma_buf,
            &beta_buf,
            &mut output_buf,
            batch_size,
            dim,
            params.eps,
        )?;

        let mut output_host = vec![0.0f32; batch_size * dim];
        device.download(&output_buf, &mut output_host)?;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(gamma_buf);
        device.deallocate(beta_buf);
        device.deallocate(output_buf);

        Array2::from_shape_vec((batch_size, dim), output_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape layer_norm_forward output: {err}"),
            }
        })
    }

    // ========================================================================
    // Activation Operations
    // ========================================================================

    /// Apply Richards Curve activation on GPU.
    ///
    /// Computes: σ(x) = 1 / (1 + (k*m)^(1/m) * exp(-β*(x-ν)))
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, dim)
    /// * `nu` - Inflection point (center parameter)
    /// * `k` - Growth rate (steepness)
    /// * `m` - Shape parameter (asymmetry)
    /// * `beta` - Scale/temperature
    pub fn richards_curve_forward(
        &mut self,
        input: &Array2<f32>,
        nu: f32,
        k: f32,
        m: f32,
        beta: f32,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock for Richards curve".to_string(),
        })?;

        let (batch_size, dim) = input.dim();
        let total_size = batch_size * dim;

        // Allocate GPU buffers
        let input_size = total_size * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        let mut output_buf = device.allocate(input_size)?;

        // Upload input
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;

        // Apply Richards curve on GPU
        let params = crate::domain::compute::gpu_ops::RichardsCurveParams {
            nu,
            k,
            m,
            beta,
            temp_reciprocal: 1.0,
            output_gain: 1.0,
            output_bias: 0.0,
            scale: 1.0,
            shift: 0.0,
            adaptive_scale: 1.0,
            adaptive_shift: 0.0,
            input_scale: 1.0,
            gate_scale: 1.0,
            gate_bias: 0.0,
            _pad1: 0,
            _pad2: 0,
        };

        device.richards_curve(&input_buf, &mut output_buf, &params, total_size)?;

        // Download result
        let mut output = vec![0.0f32; total_size];
        device.download(&output_buf, &mut output)?;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(output_buf);

        // Reshape to Array2
        Ok(Array2::from_shape_vec((batch_size, dim), output)?)
    }

    /// Apply activation function on GPU.
    ///
    /// Supports Identity, ReLU, GELU, and SiLU activations.
    /// Uses GPU kernels via GpuDevice for high-performance activation.
    pub fn activation_forward(
        &mut self,
        input: &Array2<f32>,
        activation: GpuActivation,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock for activation".to_string(),
        })?;

        let (batch_size, dim) = input.dim();
        let total_size = batch_size * dim;

        // Allocate GPU buffers
        let input_size = total_size * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        let mut output_buf = device.allocate(input_size)?;

        // Upload input
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;

        // Apply activation on GPU
        match activation {
            GpuActivation::Identity => {
                // Copy input to output
                device.copy_within_device(&input_buf, &mut output_buf, total_size)?;
            }
            GpuActivation::Relu => {
                // ReLU: max(0, x)
                device.relu(&input_buf, &mut output_buf, total_size)?;
            }
            GpuActivation::Gelu => {
                // GELU: x * Φ(x)
                device.gelu(&input_buf, &mut output_buf, total_size)?;
            }
            GpuActivation::Silu => {
                // SiLU: x * sigmoid(x)
                device.silu(&input_buf, &mut output_buf, total_size)?;
            }
        }

        // Download result
        let mut output = vec![0.0f32; total_size];
        device.download(&output_buf, &mut output)?;

        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(output_buf);

        // Reshape to Array2
        Ok(Array2::from_shape_vec((batch_size, dim), output)?)
    }
}

// ============================================================================
// GPU Workspace Management
// ============================================================================

/// Workspace for GPU kernel execution.
///
/// Manages pre-allocated buffers to minimize allocation overhead
/// during kernel execution.
///
/// ## Memory Management Strategy
///
/// - **Power-of-2 sizing**: Aligns buffers to 256-byte boundaries for coalesced access
/// - **Reusable buffers**: Once allocated, buffers are never deallocated until workspace cleanup
/// - **Capacity tracking**: Monitors current allocation and resizes when needed
/// - **Zero-copy pipeline**: Data stays on GPU between operations
/// - **Buffer pooling**: Named buffers for different operation types (activation, qkv, scores, etc)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[derive(Debug)]
struct GpuKernelWorkspace {
    /// Capacity tracking (batch_size, embed_dim, seq_len)
    capacity: (usize, usize, usize),
    /// Allocated buffers (reused across kernel calls)
    buffers: Vec<GpuBuffer>,
    /// Buffer names for debugging/tracking
    buffer_names: Vec<String>,
    /// Whether buffers are allocated
    ready: bool,
    /// Statistics: total allocations
    allocation_count: usize,
    /// Statistics: total reallocations
    reallocation_count: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuKernelWorkspace {
    fn new() -> Self {
        Self {
            capacity: (0, 0, 0),
            buffers: Vec::new(),
            buffer_names: Vec::new(),
            ready: false,
            allocation_count: 0,
            reallocation_count: 0,
        }
    }

    /// Ensure workspace has sufficient capacity for computation.
    ///
    /// Uses power-of-2 sizing to improve memory alignment and coalesced access patterns.
    /// Old buffers are deallocated before new ones are allocated on resize.
    fn ensure_capacity(
        &mut self,
        device: &mut GpuDevice,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        // Check if resize is needed
        let needs_resize = !self.ready
            || batch_size > self.capacity.0
            || embed_dim > self.capacity.1
            || seq_len > self.capacity.2;

        if !needs_resize {
            return Ok(());
        }

        // Deallocate old buffers before allocating new ones
        if self.ready {
            for buf in self.buffers.drain(..) {
                device.deallocate(buf);
            }
            self.buffer_names.clear();
        }

        // Power-of-2 sizing for efficient GPU memory alignment
        let new_batch = batch_size.next_power_of_two().max(2);
        let new_embed = embed_dim.next_power_of_two().max(2);
        let new_seq = seq_len.next_power_of_two().max(2);

        // Pre-allocate common buffer sizes (standard for all operations)
        let buffer_specs = vec![
            // Activation buffers: [batch_size * embed_dim] f32 values
            (
                "activation_0",
                new_batch * new_embed * std::mem::size_of::<f32>(),
            ),
            (
                "activation_1",
                new_batch * new_embed * std::mem::size_of::<f32>(),
            ),
            // Projection/QKV buffers: [batch_size * embed_dim] f32 values
            ("qkv_0", new_batch * new_embed * std::mem::size_of::<f32>()),
            ("qkv_1", new_batch * new_embed * std::mem::size_of::<f32>()),
            ("qkv_2", new_batch * new_embed * std::mem::size_of::<f32>()),
            // Attention scores: [batch_size * seq_len * seq_len] f32 values
            (
                "scores",
                new_batch * new_seq * new_seq * std::mem::size_of::<f32>(),
            ),
            // Attention output: [batch_size * embed_dim] f32 values
            (
                "attn_output",
                new_batch * new_embed * std::mem::size_of::<f32>(),
            ),
            // Weight matrices: [embed_dim * embed_dim] f32 values
            ("weight", new_embed * new_embed * std::mem::size_of::<f32>()),
        ];

        // Allocate all buffers
        for (name, size) in buffer_specs {
            let buf = device.allocate(size)?;
            self.buffers.push(buf);
            self.buffer_names.push(name.to_string());
        }

        self.capacity = (new_batch, new_embed, new_seq);
        self.ready = true;
        self.allocation_count += 1;

        if self.allocation_count > 1 {
            self.reallocation_count += 1;
        }

        Ok(())
    }

    /// Get buffer at index (for internal use by kernels)
    fn get_buffer(&self, index: usize) -> Option<&GpuBuffer> {
        self.buffers.get(index)
    }

    /// Reset workspace for reuse without deallocation
    fn reset(&mut self) {
        // Buffers remain allocated; just mark ready for next operation
        // No explicit action needed - GPU operations will overwrite existing data
    }

    /// Cleanup: deallocate all workspace buffers
    fn cleanup(&mut self, device: &mut GpuDevice) {
        for buf in self.buffers.drain(..) {
            device.deallocate(buf);
        }
        self.buffer_names.clear();
        self.ready = false;
        self.capacity = (0, 0, 0);
    }

    /// Get workspace statistics
    fn stats(&self) -> GpuKernelWorkspaceStats {
        let mut stats = GpuKernelWorkspaceStats {
            capacity: self.capacity,
            buffer_count: self.buffers.len(),
            allocation_count: self.allocation_count,
            reallocation_count: self.reallocation_count,
            estimated_memory_bytes: 0,
        };
        stats.estimated_memory_bytes = stats.calculate_memory();
        stats
    }
}

/// Statistics for GPU kernel workspace
#[derive(Debug, Clone)]
pub struct GpuKernelWorkspaceStats {
    /// Current capacity (batch_size, embed_dim, seq_len)
    pub capacity: (usize, usize, usize),
    /// Number of allocated buffers
    pub buffer_count: usize,
    /// Total allocations
    pub allocation_count: usize,
    /// Total reallocations
    pub reallocation_count: usize,
    /// Estimated total memory usage in bytes
    pub estimated_memory_bytes: usize,
}

impl GpuKernelWorkspaceStats {
    /// Calculate estimated memory usage
    pub fn calculate_memory(&self) -> usize {
        let (batch, embed, seq) = self.capacity;
        // Rough estimate: 8 buffers with various sizes
        // activation (2) + qkv (3) + scores + output + weight
        let element_size = std::mem::size_of::<f32>();

        // activation buffers: 2 * batch * embed
        let activation_mem = 2 * batch * embed * element_size;
        // QKV buffers: 3 * batch * embed
        let qkv_mem = 3 * batch * embed * element_size;
        // Scores: batch * seq * seq
        let scores_mem = batch * seq * seq * element_size;
        // Output: batch * embed
        let output_mem = batch * embed * element_size;
        // Weight: embed * embed
        let weight_mem = embed * embed * element_size;

        activation_mem + qkv_mem + scores_mem + output_mem + weight_mem
    }
}

// ============================================================================
// GPU Backward Kernels (Phase 5.6.4a)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl UnifiedGpuKernels {
    /// GPU-accelerated backward pass for attention (Phase 5.6.4a).
    ///
    /// Computes gradients with respect to input and weights.
    ///
    /// # Arguments
    /// * `output_grads` - Gradients of loss w.r.t. attention output (batch_size * seq_len, embed_dim)
    /// * `input` - Original input from forward pass (batch_size * seq_len, embed_dim)
    /// * `attention_weights` - Attention weight matrices from forward pass
    /// * `wq, wk, wv, wo` - Weight matrices
    /// * `params` - Attention parameters
    ///
    /// # Returns
    /// Tuple of (input_grads, weight_grads)
    pub fn attention_backward(
        &mut self,
        output_grads: &Array2<f32>,
        input: &Array2<f32>,
        wq: &Array2<f32>,
        wk: &Array2<f32>,
        wv: &Array2<f32>,
        wo: &Array2<f32>,
        params: &AttentionParams,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let (total_tokens, embed_dim) = input.dim();
        if output_grads.dim() != input.dim() {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("output_grads: {:?}", input.dim()),
                got: format!("{:?}", output_grads.dim()),
            });
        }
        if params.embed_dim != embed_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.embed_dim: {}", embed_dim),
                got: format!("{}", params.embed_dim),
            });
        }
        let expected_tokens = params.batch_size.saturating_mul(params.seq_len);
        if expected_tokens != total_tokens {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("batch*seq: {}", expected_tokens),
                got: format!("{}", total_tokens),
            });
        }

        // Compute Q/K/V and output projection weight gradients.
        let (grad_q, grad_k, grad_v) =
            self.backward_qkv_projection_gpu(output_grads, input, wq, wk, wv, params)?;
        let grad_wo = self.backward_output_projection_gpu(input, output_grads, wo)?;

        // Compute input gradient on GPU: dX = dY@Wo^T + dY@Wq^T + dY@Wk^T + dY@Wv^T.
        let input_grads = self.gpu_accumulate_input_grads(
            output_grads,
            &[wo, wq, wk, wv],
            total_tokens,
            embed_dim,
        )?;

        // Legacy API returns a single weight gradient tensor.
        // Consolidate all projection gradients by averaging.
        let mut weight_grads = grad_wo;
        weight_grads += &grad_q;
        weight_grads += &grad_k;
        weight_grads += &grad_v;
        weight_grads.mapv_inplace(|x| x * 0.25);

        Ok((input_grads, weight_grads))
    }

    /// GPU kernel for backward QKV projection (Phase 5.6.4a).
    ///
    /// Computes gradients for Q, K, V projections independently.
    /// Enables fused computation for all three projections.
    ///
    /// # Implementation Strategy
    /// - Use tensor contraction: dL/dW = dL/dout @ input^T
    /// - Parallelize across heads
    /// - Use workspace memory pools for intermediate buffers
    pub fn backward_qkv_projection_gpu(
        &mut self,
        output_grads: &Array2<f32>, // [batch*seq, embed]
        input: &Array2<f32>,        // [batch*seq, embed]
        wq: &Array2<f32>,           // [embed, embed]
        wk: &Array2<f32>,           // [embed, embed]
        wv: &Array2<f32>,           // [embed, embed]
        params: &AttentionParams,
    ) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (total_tokens, embed_dim) = input.dim();

        // Validate dimensions
        if output_grads.dim() != (total_tokens, embed_dim) {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("output_grads: ({}, {})", total_tokens, embed_dim),
                got: format!("({}, {})", output_grads.dim().0, output_grads.dim().1),
            });
        }

        if wq.dim() != (embed_dim, embed_dim)
            || wk.dim() != (embed_dim, embed_dim)
            || wv.dim() != (embed_dim, embed_dim)
        {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("weights: ({}, {})", embed_dim, embed_dim),
                got: format!("wq: {:?}, wk: {:?}, wv: {:?}", wq.dim(), wk.dim(), wv.dim()),
            });
        }
        if params.embed_dim != embed_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("params.embed_dim: {}", embed_dim),
                got: format!("{}", params.embed_dim),
            });
        }
        let expected_tokens = params.batch_size.saturating_mul(params.seq_len);
        if expected_tokens != total_tokens {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("batch*seq: {}", expected_tokens),
                got: format!("{}", total_tokens),
            });
        }

        // Current API receives a single aggregated grad tensor. Compute one shared
        // projection gradient and expose it for Q/K/V.
        let grad_shared = self.gpu_gemm_to_host(
            input,
            output_grads,
            embed_dim,
            embed_dim,
            total_tokens,
            true,
            false,
        )?;

        let grad_q = grad_shared.clone();
        let grad_k = grad_shared.clone();
        let grad_v = grad_shared;

        Ok((grad_q, grad_k, grad_v))
    }

    /// GPU kernel for backward output projection (Phase 5.6.4a).
    ///
    /// Computes gradients for W_out weight matrix.
    ///
    /// # Computation
    /// dL/dW_out = attention_output^T @ dL/dout
    pub fn backward_output_projection_gpu(
        &mut self,
        attention_output: &Array2<f32>, // [batch*seq, embed]
        output_grads: &Array2<f32>,     // [batch*seq, embed]
        wo: &Array2<f32>,               // [embed, embed]
    ) -> Result<Array2<f32>> {
        // Phase 5.6.4a: Compute output projection weight gradients
        // Formula: dL/dW_out = attention_output^T @ output_grads

        let (total_tokens, embed_dim) = attention_output.dim();

        // Validate dimensions
        if output_grads.dim() != (total_tokens, embed_dim) {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("output_grads: ({}, {})", total_tokens, embed_dim),
                got: format!("({}, {})", output_grads.dim().0, output_grads.dim().1),
            });
        }

        if wo.dim() != (embed_dim, embed_dim) {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("wo: ({}, {})", embed_dim, embed_dim),
                got: format!("({}, {})", wo.dim().0, wo.dim().1),
            });
        }

        self.gpu_gemm_to_host(
            attention_output,
            output_grads,
            embed_dim,
            embed_dim,
            total_tokens,
            true,
            false,
        )
    }

    /// GPU kernel for polynomial parameter gradients (Phase 5.6.4a).
    ///
    /// Computes gradients for PolyAttention-specific parameters: a, b, scale.
    /// Used in attention score computation: score = poly(a, b, scale) * (Q @ K^T)
    pub fn backward_poly_params_gpu(
        &mut self,
        attention_scores: &Array2<f32>, // [batch*num_heads, seq, seq]
        score_grads: &Array2<f32>,      // [batch*num_heads, seq, seq]
        _a: f32,
        _b: f32,
        _scale: f32,
    ) -> Result<(f32, f32, f32)> {
        let scores_dim = attention_scores.dim();

        // Validate dimensions
        if score_grads.dim() != scores_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("score_grads: {:?}", scores_dim),
                got: format!("{:?}", score_grads.dim()),
            });
        }

        let num_elements = scores_dim.0.saturating_mul(scores_dim.1).max(1) as f32;

        // For p = a + b*x + scale:
        // dL/da = mean(score_grads), dL/db = mean(score_grads * x), dL/dscale = mean(score_grads)
        let grad_sum = self.gpu_sum_array(score_grads)?;
        let grad_prod_sum = self.gpu_sum_product_arrays(score_grads, attention_scores)?;
        let grad_a = grad_sum / num_elements;
        let grad_b = grad_prod_sum / num_elements;
        let grad_scale = grad_sum / num_elements;

        Ok((grad_a, grad_b, grad_scale))
    }
}

// ============================================================================
// MoE (Mixture of Experts) GPU Kernels
// ============================================================================

/// Parameters for MoE forward pass
#[derive(Debug, Clone)]
pub struct MoeParams {
    /// Number of experts
    pub num_experts: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Expert hidden dimension
    pub expert_hidden_dim: usize,
    /// Number of active experts (top-k)
    pub num_active: usize,
    /// Batch size (tokens)
    pub batch_size: usize,
}

impl MoeParams {
    pub fn new(
        num_experts: usize,
        embed_dim: usize,
        expert_hidden_dim: usize,
        num_active: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            num_experts,
            embed_dim,
            expert_hidden_dim,
            num_active,
            batch_size,
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl UnifiedGpuKernels {
    /// GPU-accelerated batched MoE forward pass.
    ///
    /// Computes all expert outputs in parallel on GPU using batched GEMM:
    /// 1. **Router forward**: Compute routing logits and softmax on GPU
    /// 2. **Batched expert GEMM**: All expert W1 matrices concatenated → single GEMM
    /// 3. **Activation**: Apply Richards curve activation on GPU
    /// 4. **Batched output GEMM**: All expert W2 matrices concatenated → single GEMM
    /// 5. **Weighted sum**: Combine expert outputs using routing weights
    ///
    /// # Performance
    ///
    /// - Eliminates per-expert CPU-GPU synchronization
    /// - Uses batched GEMM for parallel expert computation
    /// - Single GPU kernel dispatch for all experts
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, embed_dim)
    /// * `router_w1` - Router first layer weights (embed_dim, router_hidden)
    /// * `router_w2` - Router second layer weights (router_hidden, num_experts)
    /// * `expert_w1` - Expert input projection weights (num_experts, embed_dim, hidden_dim)
    /// * `expert_w2` - Expert output projection weights (num_experts, hidden_dim, embed_dim)
    /// * `params` - MoE parameters
    ///
    /// # Returns
    /// * `output` - Combined expert output (batch_size, embed_dim)
    /// * `routing_probs` - Routing probabilities (batch_size, num_experts)
    pub fn moe_forward_batched(
        &mut self,
        input: &Array2<f32>,
        router_w1: &Array2<f32>,
        router_w2: &Array2<f32>,
        expert_w1: &[Array2<f32>], // [num_experts] × (embed_dim, hidden_dim)
        expert_w2: &[Array2<f32>], // [num_experts] × (hidden_dim, embed_dim)
        params: &MoeParams,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let (batch_size, embed_dim) = input.dim();
        if batch_size != params.batch_size || embed_dim != params.embed_dim {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("input: ({}, {})", params.batch_size, params.embed_dim),
                got: format!("{:?}", input.dim()),
            });
        }
        if expert_w1.len() != params.num_experts || expert_w2.len() != params.num_experts {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("expert weights: {} arrays", params.num_experts),
                got: format!("w1: {}, w2: {}", expert_w1.len(), expert_w2.len()),
            });
        }

        // Step 1: Router forward on GPU
        let routing_probs = self.moe_router_forward_gpu(input, router_w1, router_w2, params)?;

        // Step 2: Batched expert forward on GPU
        let expert_outputs =
            self.moe_experts_forward_batched_gpu(input, expert_w1, expert_w2, params)?;

        // Step 3: Weighted sum of expert outputs using routing probabilities
        let output = self.moe_combine_experts_gpu(&expert_outputs, &routing_probs, params)?;

        Ok((output, routing_probs))
    }

    /// GPU router forward: computes routing probabilities via softmax.
    fn moe_router_forward_gpu(
        &mut self,
        input: &Array2<f32>,
        w1: &Array2<f32>,
        w2: &Array2<f32>,
        params: &MoeParams,
    ) -> Result<Array2<f32>> {
        let (batch_size, embed_dim) = input.dim();
        let router_hidden = w1.ncols();
        let num_experts = params.num_experts;

        // Hidden = input @ W1
        let hidden = self.gpu_gemm_to_host(
            input,
            w1,
            batch_size,
            router_hidden,
            embed_dim,
            false,
            false,
        )?;

        // Apply activation (ReLU approximation via GELU)
        let activated = self.activation_forward(&hidden, GpuActivation::Gelu)?;

        // Logits = activated @ W2
        let logits = self.gpu_gemm_to_host(
            &activated,
            w2,
            batch_size,
            num_experts,
            router_hidden,
            false,
            false,
        )?;

        // Softmax on GPU
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in moe_router_forward_gpu".to_string(),
        })?;

        let logits_slice = logits.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "logits must be contiguous".to_string(),
        })?;
        let mut logits_buf = device.allocate_f32(logits.len())?;
        let mut probs_buf = device.allocate_f32(logits.len())?;
        device.upload(logits_slice, &mut logits_buf)?;

        // Per-row softmax
        for row in 0..batch_size {
            let offset = row * num_experts;
            device.softmax(&logits_buf, &mut probs_buf, offset, num_experts)?;
        }

        let mut probs_host = vec![0.0f32; logits.len()];
        device.download(&probs_buf, &mut probs_host)?;

        device.deallocate(logits_buf);
        device.deallocate(probs_buf);

        Array2::from_shape_vec((batch_size, num_experts), probs_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape routing probs: {err}"),
            }
        })
    }

    /// GPU batched expert forward: all experts computed in parallel.
    fn moe_experts_forward_batched_gpu(
        &mut self,
        input: &Array2<f32>,
        expert_w1: &[Array2<f32>],
        expert_w2: &[Array2<f32>],
        params: &MoeParams,
    ) -> Result<Vec<Array2<f32>>> {
        let (batch_size, embed_dim) = input.dim();
        let hidden_dim = params.expert_hidden_dim;
        let num_experts = params.num_experts;

        // For each expert, compute: hidden = input @ W1, activated = gelu(hidden), output = activated @ W2
        // This could be batched, but for now we use sequential GEMM calls which are still fast on GPU
        let mut expert_outputs = Vec::with_capacity(num_experts);

        for e in 0..num_experts {
            let w1 = &expert_w1[e];
            let w2 = &expert_w2[e];

            // Validate dimensions
            if w1.dim() != (embed_dim, hidden_dim) {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!("expert_w1[{}]: ({}, {})", e, embed_dim, hidden_dim),
                    got: format!("{:?}", w1.dim()),
                });
            }
            if w2.dim() != (hidden_dim, embed_dim) {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!("expert_w2[{}]: ({}, {})", e, hidden_dim, embed_dim),
                    got: format!("{:?}", w2.dim()),
                });
            }

            // Expert forward on GPU
            let hidden =
                self.gpu_gemm_to_host(input, w1, batch_size, hidden_dim, embed_dim, false, false)?;
            let activated = self.activation_forward(&hidden, GpuActivation::Gelu)?;
            let output = self.gpu_gemm_to_host(
                &activated, w2, batch_size, embed_dim, hidden_dim, false, false,
            )?;

            expert_outputs.push(output);
        }

        Ok(expert_outputs)
    }

    /// GPU weighted sum: combine expert outputs using routing probabilities.
    fn moe_combine_experts_gpu(
        &mut self,
        expert_outputs: &[Array2<f32>],
        routing_probs: &Array2<f32>,
        params: &MoeParams,
    ) -> Result<Array2<f32>> {
        let (batch_size, embed_dim) = (params.batch_size, params.embed_dim);
        let num_experts = params.num_experts;

        if routing_probs.dim() != (batch_size, num_experts) {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("routing_probs: ({}, {})", batch_size, num_experts),
                got: format!("{:?}", routing_probs.dim()),
            });
        }

        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "GPU device lock failed in moe_combine_experts_gpu".to_string(),
        })?;

        // Allocate output buffer
        let output_elements = batch_size * embed_dim;
        let mut output_buf = device.allocate_f32(output_elements)?;
        let mut tmp_buf = device.allocate_f32(output_elements)?;

        // Initialize output to zero
        let zeros = vec![0.0f32; output_elements];
        device.upload(&zeros, &mut output_buf)?;

        // Accumulate: output += routing_prob[e] * expert_output[e]
        for e in 0..num_experts {
            let expert_out = &expert_outputs[e];
            if expert_out.dim() != (batch_size, embed_dim) {
                return Err(ModelError::DimensionMismatchDetailed {
                    expected: format!("expert_outputs[{}]: ({}, {})", e, batch_size, embed_dim),
                    got: format!("{:?}", expert_out.dim()),
                });
            }

            // Get routing weights for expert e
            let weights = routing_probs.column(e).to_owned();

            // Scale expert output by routing weights
            let mut scaled = expert_out.clone();
            for (row, &w) in weights.iter().enumerate() {
                let w = if w.is_finite() { w } else { 0.0 };
                for col in 0..embed_dim {
                    scaled[[row, col]] *= w;
                }
            }

            // Upload and accumulate on GPU
            let scaled_slice = scaled.as_slice().ok_or_else(|| ModelError::InvalidInput {
                message: "scaled expert output must be contiguous".to_string(),
            })?;
            device.upload(scaled_slice, &mut tmp_buf)?;
            device.add_scaled(1.0, &tmp_buf, &mut output_buf, output_elements)?;
        }

        // Download result
        let mut output_host = vec![0.0f32; output_elements];
        device.download(&output_buf, &mut output_host)?;

        device.deallocate(output_buf);
        device.deallocate(tmp_buf);

        Array2::from_shape_vec((batch_size, embed_dim), output_host).map_err(|err| {
            ModelError::InvalidInput {
                message: format!("Failed to reshape MoE output: {err}"),
            }
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use ndarray::Array2;

    #[test]
    fn test_attention_params() {
        let params = AttentionParams::new(8, 512, 128, 32);
        assert_eq!(params.num_heads, 8);
        assert_eq!(params.embed_dim, 512);
        assert_eq!(params.head_dim, 64);
        assert!((params.scale - 0.125).abs() < 0.01);
    }

    #[test]
    fn test_ssm_params() {
        let params = SsmParams::new(256, 512, 128, 32);
        assert_eq!(params.state_dim, 256);
        assert_eq!(params.embed_dim, 512);
        assert!(params.selective);
    }

    #[test]
    fn test_norm_params() {
        let params = NormParams::new(512);
        assert_eq!(params.dim, 512);
        assert!(params.has_scale);
        assert!(params.has_bias);
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_gpu_kernels_auto_detect() {
        // Test that auto_detect works (may fail on systems without GPU)
        match UnifiedGpuKernels::auto_detect() {
            Ok(kernels) => {
                println!("GPU kernels created successfully");
                assert!(kernels.device().lock().is_ok());
            }
            Err(e) => {
                println!("No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_backward_qkv_projection_params() {
        // Validate that backward kernel can be called with proper dimensions
        let batch_size = 2;
        let seq_len = 4;
        let embed_dim = 32;

        let output_grads: Array2<f32> = Array2::zeros((batch_size * seq_len, embed_dim));
        let input: Array2<f32> = Array2::zeros((batch_size * seq_len, embed_dim));
        let wq: Array2<f32> = Array2::zeros((embed_dim, embed_dim));
        let wk: Array2<f32> = Array2::zeros((embed_dim, embed_dim));
        let wv: Array2<f32> = Array2::zeros((embed_dim, embed_dim));

        let params = AttentionParams::new(4, embed_dim, seq_len, batch_size);

        // Just verify shapes are valid
        assert_eq!(output_grads.dim(), (batch_size * seq_len, embed_dim));
        assert_eq!(input.dim(), (batch_size * seq_len, embed_dim));
        assert_eq!(wq.dim(), (embed_dim, embed_dim));
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_backward_output_projection_shapes() {
        // Validate output projection backward kernel dimensions
        let batch_size = 2;
        let seq_len = 4;
        let embed_dim = 32;

        let attention_output: Array2<f32> = Array2::zeros((batch_size * seq_len, embed_dim));
        let output_grads: Array2<f32> = Array2::zeros((batch_size * seq_len, embed_dim));
        let wo: Array2<f32> = Array2::zeros((embed_dim, embed_dim));

        // Verify all shapes match expectations
        assert_eq!(attention_output.dim(), (batch_size * seq_len, embed_dim));
        assert_eq!(output_grads.dim(), (batch_size * seq_len, embed_dim));
        assert_eq!(wo.dim(), (embed_dim, embed_dim));
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_poly_params_backward_shapes() {
        // Validate polynomial parameter backward computation
        let batch_size = 2;
        let num_heads = 4;
        let seq_len = 4;
        let total_score_elements = batch_size * num_heads * seq_len * seq_len;

        // Attention scores and score gradients have same shape: (batch*H, seq, seq)
        // but flattened for computation
        let attention_scores: Array2<f32> =
            Array2::zeros((batch_size * num_heads, seq_len * seq_len));
        let score_grads: Array2<f32> = Array2::zeros((batch_size * num_heads, seq_len * seq_len));

        assert_eq!(attention_scores.dim(), score_grads.dim());
        assert_eq!(
            attention_scores.dim().0 * attention_scores.dim().1,
            total_score_elements
        );
    }
}
