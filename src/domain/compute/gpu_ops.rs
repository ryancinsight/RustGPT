//! GPU Matrix Operations
//!
//! Provides abstraction for GPU-accelerated linear algebra operations.
//! Supports CUDA (cuBLAS), Metal (Metal Performance Shaders), and Vulkan compute.

use super::gpu_memory::{GpuBuffer, GpuMemoryPool};
use crate::common::errors::Result;

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

    //
    // Reduction Operations
    //

    /// Compute sum of all elements
    fn sum(&self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32>;

    /// Compute mean of all elements
    fn mean(&self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32>;

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

    /// Copy within device (GPU-to-GPU)
    fn copy_within_device(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        src: &GpuBuffer,
        dst: &mut GpuBuffer,
        size: usize,
    ) -> Result<()>;

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

    fn sum(&self, _pool: &mut dyn GpuMemoryPool, _buffer: &GpuBuffer, _size: usize) -> Result<f32> {
        Err(crate::common::errors::ModelError::Backend {
            message: "CPU sum not implemented".to_string(),
        })
    }

    fn mean(
        &self,
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
}

// CpuGpuMatrixOps has been removed. Use GpuDevice::auto_detect() to initialize
// backend-specific GPU matrix operations instead. This enables strict no-fallback
// semantics where GPU operations fail explicitly if no supported backend is available.
