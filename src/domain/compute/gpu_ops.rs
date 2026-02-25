//! GPU Matrix Operations
//!
//! Provides abstraction for GPU-accelerated linear algebra operations.
//! Supports CUDA (cuBLAS), Metal (Metal Performance Shaders), and Vulkan compute.

use super::gpu_memory::{GpuBuffer, GpuMemoryPool};
use crate::common::errors::{ModelError, Result};

/// Parameters for Richards Curve calculation
///
/// Matches the GPU uniform buffer layout (64 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RichardsCurveParams {
    pub nu: f32,
    pub k: f32,
    pub m: f32,
    pub beta: f32,
    pub temp_reciprocal: f32,
    pub output_gain: f32,
    pub output_bias: f32,
    pub scale: f32,
    pub shift: f32,
    pub adaptive_scale: f32,
    pub adaptive_shift: f32,
    pub input_scale: f32,
    pub gate_scale: f32,
    pub gate_bias: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

/// BLAS operation scalar values
#[derive(Debug, Clone, Copy)]
pub struct Scalar(pub f32);

/// Trait for GPU matrix operations
///
/// Abstracts the differences between CUDA (cuBLAS), Metal (MPS), and Vulkan compute shaders.
/// Implementations must ensure numerical accuracy (target: ε ≤ 1e-4 vs CPU reference).
pub trait GpuMatrixOps: Send + Sync {
    // ─────────────────────────────────────────────────────────────────
    // Command Batching / Deferred Submission
    // ─────────────────────────────────────────────────────────────────

    /// Begin deferred recording mode.
    ///
    /// After calling this, dispatch calls should be recorded into an
    /// internal command buffer rather than submitted immediately.
    /// Default implementation is a no-op (immediate submission).
    fn begin_recording(&mut self) {}

    /// Flush all pending recorded commands to the GPU in one submission.
    ///
    /// This is the key performance primitive — call it once per training
    /// step to batch the entire forward + backward pass into a single
    /// GPU submission, eliminating per-kernel sync bubbles.
    /// Default implementation is a no-op (immediate submission).
    fn flush(&mut self) {}
    //
    // BLAS Level 3: Matrix-Matrix Operations
    //

    /// Fused Matrix-Matrix Multiply: `output = alpha * A @ B + beta * output`
    ///
    /// # Arguments
    /// * `pool` - Memory pool for buffer access
    /// * `alpha` - Scaling factor for product
    /// * `a` - Left matrix (m × k)
    /// * `b` - Right matrix (k × n)
    /// * `beta` - Scaling factor for output accumulation
    /// * `output` - Output matrix (m × n), accumulated with beta factor
    /// * `m`, `n`, `k` - Dimensions
    /// * `trans_a` - Whether to transpose A
    /// * `trans_b` - Whether to transpose B
    ///
    /// # Performance
    /// Expected: 50-100+ TFLOPS on modern GPUs (NVIDIA V100+, Apple M1+)
    fn gemm_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
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
    ) -> Result<()>;

    /// Batched Matrix-Matrix Multiply
    ///
    /// Performs `batch_count` GEMM operations.
    /// `output[b] = alpha * A[b] @ B[b] + beta * output[b]`
    ///
    /// All matrices are assumed to be stored contiguously with provided strides.
    fn gemm_batched_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        alpha: f32,
        a: &GpuBuffer,
        b: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize,
        n: usize,
        k: usize,
        batch_count: usize,
        strides: [usize; 3], // [stride_a, stride_b, stride_c] (in elements)
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()>;

    /// Matrix-Vector Multiply: `output = alpha * A @ x + beta * output`
    ///
    /// Optimized variant of GEMM for vector RHS (single column).
    fn gemv_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        alpha: f32,
        a: &GpuBuffer,
        x: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize,
        n: usize,
    ) -> Result<()>;

    //
    // Element-Wise Operations
    //

    /// Element-wise ReLU: `output = max(0, input)`
    fn relu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise GELU: `output = input * Φ(input)`
    fn gelu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise SiLU (Swish): `output = input * sigmoid(input)`
    fn silu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise Sigmoid: `output = 1 / (1 + exp(-input))`
    fn sigmoid(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise multiplication: `output = input1 * input2`
    fn mul(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input1: &GpuBuffer,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise addition: `output += scale * input`
    fn add_scaled(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scale: f32,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Element-wise multiplication: `output *= scale`
    fn scale(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scale: f32,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// In-place sign-preserving log scaling:
    /// `x <- sign(x) * log(1 + alpha * |x|) / alpha`
    fn signed_log1p_scale(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _buffer: &mut GpuBuffer,
        _alpha: f32,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "signed_log1p_scale kernel is not implemented for this backend".to_string(),
        })
    }

    /// PolyAttention scalar score transform (element-wise):
    /// `out = scale * (a * smooth_clip_tanh(x, clip_limit)^p + b)`
    fn poly_score_transform_scalar(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _a: f32,
        _b: f32,
        _scale: f32,
        _p: u32,
        _clip_limit: f32,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "poly_score_transform_scalar kernel is not implemented for this backend"
                .to_string(),
        })
    }

    /// PolyAttention scalar score-transform backward (element-wise + reduction contributions).
    ///
    /// Inputs:
    /// - `raw_scores`: pre-transform attention scores
    /// - `grad_transformed`: dL/d(transformed_scores)
    ///
    /// Outputs:
    /// - `grad_raw`: dL/d(raw_scores)
    /// - `grad_a_contrib`, `grad_b_contrib`, `grad_scale_contrib`: per-element contributions
    #[allow(clippy::too_many_arguments)]
    fn poly_score_transform_scalar_backward(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _raw_scores: &GpuBuffer,
        _grad_transformed: &GpuBuffer,
        _grad_raw: &mut GpuBuffer,
        _grad_a_contrib: &mut GpuBuffer,
        _grad_b_contrib: &mut GpuBuffer,
        _grad_scale_contrib: &mut GpuBuffer,
        _a: f32,
        _b: f32,
        _scale: f32,
        _p: u32,
        _clip_limit: f32,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message:
                "poly_score_transform_scalar_backward kernel is not implemented for this backend"
                    .to_string(),
        })
    }

    /// Element-wise multiply-add: `output = a * input1 + b * input2`
    fn axpy(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        a: f32,
        input1: &GpuBuffer,
        b: f32,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Richards Curve Element-wise function
    ///
    /// Computes the generalized sigmoid curve defined by the parameters.
    fn richards_curve(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
    ) -> Result<()>;

    /// Richards Curve input-gradient application:
    /// `output = upstream * d(richards_curve(input))/d(input)`
    ///
    /// Uses the same parameterization as `richards_curve`, including temperature and affine terms.
    fn richards_curve_backward_input(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _upstream: &GpuBuffer,
        _output: &mut GpuBuffer,
        _params: &RichardsCurveParams,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "richards_curve_backward_input kernel is not implemented for this backend"
                .to_string(),
        })
    }

    //
    // Normalization Operations
    //

    /// Layer Normalization: `output = gamma * (input - mean) / sqrt(var + eps) + beta`
    fn layer_norm(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
        feature_size: usize,
        eps: f32,
    ) -> Result<()>;

    /// Softmax normalization: `output = exp(input) / sum(exp(input))`
    fn softmax(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()>;

    /// Softmax backward (row-wise):
    /// `d_input[row, col] = softmax[row, col] * (d_output[row, col] - dot(d_output[row,:], softmax[row,:]))`
    fn softmax_backward(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _softmax_output: &GpuBuffer,
        _grad_output: &GpuBuffer,
        _grad_input: &mut GpuBuffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "softmax_backward kernel is not implemented for this backend".to_string(),
        })
    }

    /// Selective Scan forward kernel for SSM recurrence.
    ///
    /// Computes:
    /// - `h_t = A @ h_{t-1} + B @ x_t`
    /// - `y_t = C @ h_t + D @ x_t`
    ///
    /// Buffers are flat row-major:
    /// - `input`: [seq_len, embed_dim]
    /// - `a`: [state_dim, state_dim]
    /// - `b`: [state_dim, embed_dim]
    /// - `c`: [embed_dim, state_dim]
    /// - `d`: [embed_dim, embed_dim]
    /// - `h_init`: [state_dim]
    /// - `output`: [seq_len, embed_dim]
    /// - `h_final`: [state_dim]
    #[allow(clippy::too_many_arguments)]
    fn selective_scan_forward(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _a: &GpuBuffer,
        _b: &GpuBuffer,
        _c: &GpuBuffer,
        _d: &GpuBuffer,
        _h_init: &GpuBuffer,
        _output: &mut GpuBuffer,
        _h_final: &mut GpuBuffer,
        _seq_len: usize,
        _state_dim: usize,
        _embed_dim: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "selective_scan_forward kernel is not implemented for this backend"
                .to_string(),
        })
    }

    //
    // PolyAttention Operations
    //

    /// MoH Gate Activation
    ///
    /// Computes gate values: G = Richards(alpha * (Input @ W_g) + beta)
    /// Alpha and Beta are per-head vectors.
    #[allow(clippy::too_many_arguments)]
    fn moh_gate_activation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        logits: &GpuBuffer, // Input @ W_g
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        gate_params: &RichardsCurveParams,
        output: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
    ) -> Result<()>;

    /// MoH gate backward pointwise preparation (sigmoid-approx helper path).
    ///
    /// Computes, for each `[token, head]` element:
    /// - `z = alpha[head] * xw + beta[head]`
    /// - `g = sigmoid(clamp(z, -8, 8))`
    /// - `d_gate = eff_grads * g * (1 - g)`
    /// - `d_gate_scaled = d_gate * alpha[head]`
    ///
    /// This matches the existing simplified helper semantics used by `moh_gate_backward_gpu`.
    fn moh_gate_backward_prepare_sigmoid(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _xw: &GpuBuffer,
        _eff_grads: &GpuBuffer,
        _alpha: &GpuBuffer,
        _beta: &GpuBuffer,
        _d_gate: &mut GpuBuffer,
        _d_gate_scaled: &mut GpuBuffer,
        _num_tokens: usize,
        _num_heads: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message:
                "moh_gate_backward_prepare_sigmoid kernel is not implemented for this backend"
                    .to_string(),
        })
    }

    /// MoH gate backward per-head reductions for alpha/beta grads (sigmoid-approx helper path).
    ///
    /// Computes:
    /// - `grad_alpha[h] = sum_i d_gate[i,h] * xw[i,h]`
    /// - `grad_beta[h] = sum_i d_gate[i,h]`
    fn moh_gate_backward_reduce_alpha_beta(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _xw: &GpuBuffer,
        _d_gate: &GpuBuffer,
        _grad_alpha: &mut GpuBuffer,
        _grad_beta: &mut GpuBuffer,
        _num_tokens: usize,
        _num_heads: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message:
                "moh_gate_backward_reduce_alpha_beta kernel is not implemented for this backend"
                    .to_string(),
        })
    }

    /// Fused Polynomial Attention Kernel
    ///
    /// Computes:
    /// 1. S = S_content + S_pos + (Q_h @ K_comp^T)
    /// 2. S = (a * S + b)^p * scale
    /// 3. Output = S * Gate
    #[allow(clippy::too_many_arguments)]
    fn poly_attention_fused(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
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
    ) -> Result<()>;

    /// Compute `grad_transformed = grad_scores * gate_broadcast` for PolyAttention fused scores.
    ///
    /// Layouts:
    /// - `grad_scores`, `grad_transformed`: [B, H, S, S]
    /// - `gate`: [B*S, H] (token-major per-query token/head gate)
    #[allow(clippy::too_many_arguments)]
    fn poly_attention_gate_broadcast_mul(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _grad_scores: &GpuBuffer,
        _gate: &GpuBuffer,
        _grad_transformed: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "poly_attention_gate_broadcast_mul kernel is not implemented for this backend"
                .to_string(),
        })
    }

    /// Reduce fused-score gradients to per-query/head gate upstream gradients.
    ///
    /// Computes:
    /// `gate_upstream[b,s,h] = sum_j grad_scores[b,h,s,j] * transformed[b,h,s,j]`
    ///
    /// Layouts:
    /// - `grad_scores`, `transformed`: [B, H, S, S]
    /// - `gate_upstream`: [B*S, H] (token-major)
    #[allow(clippy::too_many_arguments)]
    fn poly_attention_gate_reduce_upstream(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _grad_scores: &GpuBuffer,
        _transformed: &GpuBuffer,
        _gate_upstream: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message:
                "poly_attention_gate_reduce_upstream kernel is not implemented for this backend"
                    .to_string(),
        })
    }

    /// BLR Projection Kernel
    ///
    /// Projects Q and K to low-rank components:
    /// Q_comp = MeanPool(Q)
    /// K_comp = MeanPool(K)
    /// Q_h = RichardsCurve(Q_comp)
    fn blr_projection(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
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
    ) -> Result<()>;

    /// Compute CoPE Scores
    ///
    /// Computes pos_scores[b, h, i, j] = Q[b, h, i] . P[i-j]
    #[allow(clippy::too_many_arguments)]
    fn compute_cope_scores(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        q: &GpuBuffer,
        pos_emb: &GpuBuffer,
        scores: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        max_pos: usize,
    ) -> Result<()>;

    /// Apply causal masking to attention score tensor laid out as [B, H, S, S].
    ///
    /// Sets entries with key index `j > i` to `mask_value`.
    #[allow(clippy::too_many_arguments)]
    fn causal_mask_attention_scores(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _scores: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _mask_value: f32,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "causal_mask_attention_scores kernel is not implemented for this backend"
                .to_string(),
        })
    }

    //
    // Reduction Operations
    //

    /// Compute sum of all elements
    fn sum(&mut self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32>;

    /// Compute mean of all elements
    fn mean(&mut self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32>;

    //
    // Data Transfer
    //

    /// Copy buffer contents from device to CPU
    fn download(
        &self,
        pool: &mut dyn GpuMemoryPool,
        gpu_buffer: &GpuBuffer,
        cpu_data: &mut [f32],
    ) -> Result<()>;

    /// Copy buffer contents from CPU to device
    fn upload(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        cpu_data: &[f32],
        gpu_buffer: &mut GpuBuffer,
    ) -> Result<()>;

    //
    // Sparse Routing Operations (MoE)
    //

    /// Find Top-K Experts per token
    #[allow(clippy::too_many_arguments)]
    fn compute_topk(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _routing_gates: &GpuBuffer,
        _topk_indices: &mut GpuBuffer,
        _topk_weights: &mut GpuBuffer,
        _num_tokens: usize,
        _num_experts: usize,
        _k: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "compute_topk kernel is not implemented for this backend".to_string(),
        })
    }

    /// Scatter tokens into contiguous expert buffers
    #[allow(clippy::too_many_arguments)]
    fn scatter_experts(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _hidden_states: &GpuBuffer,
        _topk_indices: &GpuBuffer,
        _global_expert_offsets: &GpuBuffer,
        _expert_counters: &mut GpuBuffer,
        _scattered_hidden: &mut GpuBuffer,
        _original_token_indices: &mut GpuBuffer,
        _num_tokens: usize,
        _hidden_dim: usize,
        _k: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "scatter_experts kernel is not implemented for this backend".to_string(),
        })
    }

    /// Gather expert outputs back to original token shape
    #[allow(clippy::too_many_arguments)]
    fn gather_experts(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _expert_outputs: &GpuBuffer,
        _topk_weights: &GpuBuffer,
        _topk_indices: &GpuBuffer,
        _global_expert_offsets: &GpuBuffer,
        _token_expert_slots: &GpuBuffer,
        _gathered_output: &mut GpuBuffer,
        _num_tokens: usize,
        _hidden_dim: usize,
        _k: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "gather_experts kernel is not implemented for this backend".to_string(),
        })
    }


    //
    // Titans Memory Kernels
    //

    /// Batched MLP forward for Titans neural memory.
    /// z = W1 @ keys + b1, h = ReLU(z), v_pred = W2 @ h + b2
    #[allow(clippy::too_many_arguments)]
    fn titans_mlp_forward(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _keys: &GpuBuffer,
        _w1: &GpuBuffer,
        _b1: &GpuBuffer,
        _w2: &GpuBuffer,
        _b2: &GpuBuffer,
        _z_out: &mut GpuBuffer,
        _h_out: &mut GpuBuffer,
        _v_pred: &mut GpuBuffer,
        _num_tokens: usize,
        _key_dim: usize,
        _hidden_dim: usize,
        _val_dim: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "titans_mlp_forward kernel is not implemented for this backend".to_string(),
        })
    }

    /// Accumulate W2/b2 gradients for Titans memory.
    #[allow(clippy::too_many_arguments)]
    fn titans_grad_w2(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _v_target: &GpuBuffer,
        _v_pred: &GpuBuffer,
        _h_act: &GpuBuffer,
        _grad_w2: &mut GpuBuffer,
        _grad_b2: &mut GpuBuffer,
        _num_tokens: usize,
        _hidden_dim: usize,
        _val_dim: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "titans_grad_w2 kernel is not implemented for this backend".to_string(),
        })
    }

    /// Accumulate W1/b1 gradients for Titans memory.
    #[allow(clippy::too_many_arguments)]
    fn titans_grad_w1(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _keys: &GpuBuffer,
        _v_target: &GpuBuffer,
        _v_pred: &GpuBuffer,
        _z: &GpuBuffer,
        _w2: &GpuBuffer,
        _grad_w1: &mut GpuBuffer,
        _grad_b1: &mut GpuBuffer,
        _num_tokens: usize,
        _key_dim: usize,
        _hidden_dim: usize,
        _val_dim: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "titans_grad_w1 kernel is not implemented for this backend".to_string(),
        })
    }

    /// Fused Titans per-element momentum + memory update.
    /// momentum = eta * momentum - theta * grad
    /// memory   = (1 - alpha) * memory + momentum
    #[allow(clippy::too_many_arguments)]
    fn titans_memory_update(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _grad: &GpuBuffer,
        _momentum: &mut GpuBuffer,
        _memory: &mut GpuBuffer,
        _num_elements: usize,
        _alpha: f32,
        _eta: f32,
        _theta: f32,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "titans_memory_update kernel is not implemented for this backend".to_string(),
        })
    }

    /// Copy within device (GPU-to-GPU)
    fn copy_within_device(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        src: &GpuBuffer,
        dst: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

    /// Copy a sub-range within device buffers (GPU-to-GPU).
    ///
    /// Offsets and `size` are in `f32` elements.
    fn copy_within_device_range(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _src: &GpuBuffer,
        _src_offset: usize,
        _dst: &mut GpuBuffer,
        _dst_offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "copy_within_device_range kernel is not implemented for this backend"
                .to_string(),
        })
    }

    /// Permute 4D tensor
    ///
    /// Generic permutation of a 4D tensor.
    /// Input: [d0, d1, d2, d3] with strides [s0, s1, s2, s3]
    /// Output: [od0, od1, od2, od3]
    ///
    /// This function computes `output[linear_idx] = input[permuted_idx]`.
    /// The caller must provide input strides and output strides/dims.
    ///
    /// Used to transpose (Batch, Seq, Heads, HeadDim) -> (Batch, Heads, Seq, HeadDim)
    ///
    /// # Arguments
    /// * `input_strides`: [s0, s1, s2, s3]
    /// * `output_dims`: [od0, od1, od2, od3]
    /// * `permuted_input_strides`: [pis0, pis1, pis2, pis3]
    ///   where `pis_k` is the stride of the input dimension that maps to output dimension k.
    fn permute_4d(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        output_dims: [usize; 4],
        permuted_input_strides: [usize; 4],
    ) -> Result<()>;

    /// Fill buffer with value
    fn fill_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        buffer: &mut GpuBuffer,
        value: f32,
    ) -> Result<()>;

    /// Row-wise broadcast addition: matrix[row, col] += bias[col]
    fn broadcast_add_rows(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _matrix: &mut GpuBuffer,
        _bias: &GpuBuffer,
        _batch_size: usize,
        _cols: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "broadcast_add_rows kernel is not implemented for this backend".to_string(),
        })
    }

    /// Compute Richards Curve Gating
    ///
    /// output = richards(alpha * input + beta)
    ///
    /// input: (N, H)
    /// alpha: (1, H)
    /// beta: (1, H)
    /// output: (N, H)
    fn richards_gate(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        batch_size: usize,
        num_heads: usize,
    ) -> Result<()>;

    /// Reduce scalar RichardsCurve parameter gradients over all elements.
    ///
    /// Writes the 9 canonical scalar gradients in this fixed order:
    /// `nu, k, m, beta, temperature, output_gain, output_bias, scale, shift`.
    /// Callers should filter/reorder by the curve's `*_learnable` flags.
    #[allow(clippy::too_many_arguments)]
    fn richards_scalar_param_grads_reduce(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _upstream: &GpuBuffer,
        _output_grads: &mut GpuBuffer,
        _params: &RichardsCurveParams,
        _size: usize,
        _variant_is_tanh: bool,
        _birch_exponential_tail: bool,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "richards_scalar_param_grads_reduce kernel is not implemented for this backend"
                .to_string(),
        })
    }

    //
    // Optimizer Operations
    //

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
    /// # Arguments
    /// * `params` - Parameters buffer (modified in-place)
    /// * `grads` - Gradients buffer
    /// * `m` - First moment estimate buffer
    /// * `v` - Second moment estimate buffer
    /// * `v_max` - Optional v_max buffer for AMSGrad
    /// * `lr` - Learning rate
    /// * `beta1` - First moment decay rate
    /// * `beta2` - Second moment decay rate
    /// * `epsilon` - Numerical stability constant
    /// * `inv_bias1` - Precomputed 1/(1-β₁^t)
    /// * `inv_bias2` - Precomputed 1/(1-β₂^t)
    /// * `weight_decay` - Weight decay coefficient
    /// * `use_decoupled_wd` - Use AdamW-style decoupled weight decay
    /// * `use_amsgrad` - Use AMSGrad variant
    /// * `size` - Number of elements
    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _params: &mut GpuBuffer,
        _grads: &GpuBuffer,
        _m: &mut GpuBuffer,
        _v: &mut GpuBuffer,
        _v_max: Option<&mut GpuBuffer>,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _inv_bias1: f32,
        _inv_bias2: f32,
        _weight_decay: f32,
        _use_decoupled_wd: bool,
        _use_amsgrad: bool,
        _size: usize,
    ) -> Result<()> {
        Err(ModelError::Backend {
            message: "adam_step kernel is not implemented for this backend".to_string(),
        })
    }
}

/// CPU-fallback matrix operations (for testing and non-GPU builds)
#[derive(Debug)]
pub struct CpuMatrixOps;

impl GpuMatrixOps for CpuMatrixOps {
    fn fill_f32(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _buffer: &mut GpuBuffer,
        _value: f32,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU fill_f32 not implemented".to_string(),
        })
    }

    fn richards_gate(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _alpha: &GpuBuffer,
        _beta: &GpuBuffer,
        _output: &mut GpuBuffer,
        _params: &RichardsCurveParams,
        _batch_size: usize,
        _num_heads: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU richards_gate not implemented".to_string(),
        })
    }

    fn gemm_f32(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _alpha: f32,
        _a: &GpuBuffer,
        _b: &GpuBuffer,
        _beta: f32,
        _output: &mut GpuBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _trans_a: bool,
        _trans_b: bool,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU gemm_f32 not implemented".to_string(),
        })
    }

    fn gemm_batched_f32(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _alpha: f32,
        _a: &GpuBuffer,
        _b: &GpuBuffer,
        _beta: f32,
        _output: &mut GpuBuffer,
        _m: usize,
        _n: usize,
        _k: usize,
        _batch_count: usize,
        _strides: [usize; 3],
        _trans_a: bool,
        _trans_b: bool,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU gemm_batched_f32 not implemented".to_string(),
        })
    }

    fn gemv_f32(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _alpha: f32,
        _a: &GpuBuffer,
        _x: &GpuBuffer,
        _beta: f32,
        _output: &mut GpuBuffer,
        _m: usize,
        _n: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU GEMV not implemented".to_string(),
        })
    }

    fn relu(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU ReLU not implemented".to_string(),
        })
    }

    fn gelu(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU GELU not implemented".to_string(),
        })
    }

    fn silu(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU SiLU not implemented".to_string(),
        })
    }

    fn sigmoid(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU Sigmoid not implemented".to_string(),
        })
    }

    fn mul(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input1: &GpuBuffer,
        _input2: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU Mul not implemented".to_string(),
        })
    }

    fn add_scaled(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _scale: f32,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU add_scaled not implemented".to_string(),
        })
    }

    fn scale(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _scale: f32,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU scale not implemented".to_string(),
        })
    }

    fn axpy(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _a: f32,
        _input1: &GpuBuffer,
        _b: f32,
        _input2: &GpuBuffer,
        _output: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU axpy not implemented".to_string(),
        })
    }

    fn richards_curve(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _params: &RichardsCurveParams,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU richards_curve not implemented".to_string(),
        })
    }

    fn layer_norm(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _gamma: &GpuBuffer,
        _beta: &GpuBuffer,
        _output: &mut GpuBuffer,
        _batch_size: usize,
        _feature_size: usize,
        _eps: f32,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU layer_norm not implemented".to_string(),
        })
    }

    fn softmax(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _rows: usize,
        _cols: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU softmax not implemented".to_string(),
        })
    }

    fn sum(&mut self, _pool: &mut dyn GpuMemoryPool, _buffer: &GpuBuffer, _size: usize) -> Result<f32> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU sum not implemented".to_string(),
        })
    }

    fn mean(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _buffer: &GpuBuffer,
        _size: usize,
    ) -> Result<f32> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU mean not implemented".to_string(),
        })
    }

    fn download(
        &self,
        _pool: &mut dyn GpuMemoryPool,
        _gpu_buffer: &GpuBuffer,
        _cpu_data: &mut [f32],
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU download not implemented".to_string(),
        })
    }

    fn upload(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _cpu_data: &[f32],
        _gpu_buffer: &mut GpuBuffer,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU upload not implemented".to_string(),
        })
    }

    fn copy_within_device(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _src: &GpuBuffer,
        _dst: &mut GpuBuffer,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU copy_within_device not implemented".to_string(),
        })
    }

    fn copy_within_device_range(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _src: &GpuBuffer,
        _src_offset: usize,
        _dst: &mut GpuBuffer,
        _dst_offset: usize,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU copy_within_device_range not implemented".to_string(),
        })
    }

    fn permute_4d(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _input: &GpuBuffer,
        _output: &mut GpuBuffer,
        _output_dims: [usize; 4],
        _permuted_input_strides: [usize; 4],
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU permute_4d not implemented".to_string(),
        })
    }

    fn poly_attention_fused(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _content_scores: &GpuBuffer,
        _pos_scores: &GpuBuffer,
        _q_h: &GpuBuffer,
        _k_comp: &GpuBuffer,
        _poly_a: &GpuBuffer,
        _poly_b: &GpuBuffer,
        _poly_scale: &GpuBuffer,
        _gate: &GpuBuffer,
        _output: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _max_pos: usize,
        _p: usize,
        _blr_rank: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU poly_attention_fused not implemented".to_string(),
        })
    }

    fn blr_projection(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _q: &GpuBuffer,
        _k: &GpuBuffer,
        _q_h: &mut GpuBuffer,
        _k_comp: &mut GpuBuffer,
        _richards_params: &RichardsCurveParams,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _head_dim: usize,
        _rank: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU blr_projection not implemented".to_string(),
        })
    }

    fn compute_cope_scores(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _q: &GpuBuffer,
        _pos_emb: &GpuBuffer,
        _scores: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
        _seq_len: usize,
        _head_dim: usize,
        _max_pos: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU compute_cope_scores not implemented".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn moh_gate_activation(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _logits: &GpuBuffer,
        _alpha: &GpuBuffer,
        _beta: &GpuBuffer,
        _gate_params: &RichardsCurveParams,
        _output: &mut GpuBuffer,
        _batch_size: usize,
        _num_heads: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU moh_gate_activation not implemented".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        &mut self,
        _pool: &mut dyn GpuMemoryPool,
        _params: &mut GpuBuffer,
        _grads: &GpuBuffer,
        _m: &mut GpuBuffer,
        _v: &mut GpuBuffer,
        _v_max: Option<&mut GpuBuffer>,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
        _inv_bias1: f32,
        _inv_bias2: f32,
        _weight_decay: f32,
        _use_decoupled_wd: bool,
        _use_amsgrad: bool,
        _size: usize,
    ) -> Result<()> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU adam_step not implemented".to_string(),
        })
    }
}

// CpuGpuMatrixOps has been removed. Use GpuDevice::auto_detect() to initialize
// backend-specific GPU matrix operations instead. This enables strict no-fallback
// semantics where GPU operations fail explicitly if no supported backend is available.
