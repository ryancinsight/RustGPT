//! WGPU-based GPU Operations
//!
//! Cross-platform GPU implementation using wgpu (WebGPU API).
//! Supports Vulkan, Metal, DX12, and OpenGL backends automatically.
//!
//! ## Shader Implementation (Phase 5.3)
//!
//! This module provides working GPU kernels for:
//! - GEMM (tiled matrix multiplication)
//! - Softmax (numerically stable)
//! - Element-wise operations (ReLU, GELU, SiLU)
//! - Layer normalization
//! - Data transfer (upload/download)

use super::gpu_memory::{GpuBuffer, GpuMemoryPool, MemoryStats};
use super::gpu_ops::{GpuMatrixOps, RichardsCurveParams};
use crate::common::errors::{ModelError, Result};
use std::collections::HashMap;

#[cfg(feature = "wgpu")]
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, DeviceDescriptor,
    Features, Instance, InstanceDescriptor, Limits, MemoryHints, PowerPreference, Queue,
    RequestAdapterOptions, util::DeviceExt,
};

// ============================================================================
// WGSL Shader Source Code
// ============================================================================

/// Tiled GEMM shader for matrix multiplication
/// Computes C = alpha * A @ B^T + beta * C
/// Note: This implementation assumes B is transposed for attention-style operations
const SHADER_GEMM: &str = r#"
struct GemmParams {
    alpha: f32,
    beta: f32,
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: GemmParams;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= params.m || col >= params.n) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Compute dot product for this (row, col) of output
    for (var k_idx: u32 = 0u; k_idx < params.k; k_idx = k_idx + 1u) {
        var a_val: f32;
        if (params.trans_a != 0u) {
            // A is [K, M], read A[k, row]
            a_val = a[k_idx * params.m + row];
        } else {
            // A is [M, K], read A[row, k]
            a_val = a[row * params.k + k_idx];
        }
        
        var b_val: f32;
        if (params.trans_b != 0u) {
            // B is [N, K], read B[col, k] (row 'col' of B)
            b_val = b[col * params.k + k_idx];
        } else {
            // B is [K, N], read B[k, col]
            b_val = b[k_idx * params.n + col];
        }
        
        sum = sum + a_val * b_val;
    }
    
    let c_idx = row * params.n + col;
    
    // Apply alpha and beta
    if (params.beta == 0.0) {
        c[c_idx] = params.alpha * sum;
    } else {
        c[c_idx] = params.alpha * sum + params.beta * c[c_idx];
    }
}
"#;

/// Rust struct for GEMM parameters (matches WGSL GemmParams)
/// Note: WGSL requires 16-byte alignment for uniform buffers
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParamsRust {
    alpha: f32,
    beta: f32,
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    pad: u32,
}

/// Rust struct for batched GEMM parameters (matches WGSL GemmBatchedParams)
/// Note: WGSL requires 16-byte alignment for uniform buffers, so we pad to 64 bytes
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmBatchedParams {
    alpha: f32,
    beta: f32,
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    trans_a: u32,
    trans_b: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

/// Batched GEMM shader for matrix multiplication
/// Computes C[b] = alpha * A[b] @ B[b] + beta * C[b]
/// Supports transpose flags
const SHADER_GEMM_BATCHED: &str = r#"
struct GemmBatchedParams {
    alpha: f32,
    beta: f32,
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,
    trans_a: u32,
    trans_b: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: GemmBatchedParams;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;
    
    if (row >= params.m || col >= params.n || batch >= params.batch_count) {
        return;
    }
    
    let batch_offset_a = batch * params.stride_a;
    let batch_offset_b = batch * params.stride_b;
    let batch_offset_c = batch * params.stride_c;
    
    var sum: f32 = 0.0;
    
    // Compute dot product for this (row, col) of output
    for (var k_idx: u32 = 0u; k_idx < params.k; k_idx = k_idx + 1u) {
        let a_idx = batch_offset_a + row * params.k + k_idx;
        let b_idx = batch_offset_b + col * params.k + k_idx;  // B is transposed
        sum = sum + a[a_idx] * b[b_idx];
    }
    
    let c_idx = batch_offset_c + row * params.n + col;
    
    // Apply alpha and beta
    if (params.beta == 0.0) {
        c[c_idx] = params.alpha * sum;
    } else {
        c[c_idx] = params.alpha * sum + params.beta * c[c_idx];
    }
}
"#;

/// Numerically stable softmax shader
/// Computes softmax row-wise with log-sum-exp trick
#[allow(dead_code)]
const SHADER_SOFTMAX: &str = r#"
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SoftmaxParams;

// Workgroup shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.x / 256u;
    let tid = local_id.x;
    
    if (row >= params.rows) {
        return;
    }
    
    // 1. Find Max
    var max_val: f32 = -3.40282347e+38;
    for (var col = tid; col < params.cols; col += 256u) {
        let idx = row * params.cols + col;
        max_val = max(max_val, input[idx]);
    }
    shared_max[tid] = max_val;
    workgroupBarrier();
    
    // Reduce max
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
    }
    let row_max = shared_max[0];
    
    // 2. Compute Exp and Sum
    var sum_val: f32 = 0.0;
    for (var col = tid; col < params.cols; col += 256u) {
        let idx = row * params.cols + col;
        let val = exp(input[idx] - row_max);
        output[idx] = val; // Store temporarily
        sum_val += val;
    }
    shared_sum[tid] = sum_val;
    workgroupBarrier();
    
    // Reduce sum
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];
    
    // 3. Normalize
    let inv_sum = 1.0 / (row_sum + 1e-6);
    for (var col = tid; col < params.cols; col += 256u) {
        let idx = row * params.cols + col;
        output[idx] *= inv_sum;
    }
}
"#;

/// Permute 4D tensor shader
#[allow(dead_code)]
const SHADER_PERMUTE_4D: &str = r#"
struct PermuteParams {
    od0: u32,
    od1: u32,
    od2: u32,
    od3: u32,
    pis0: u32,
    pis1: u32,
    pis2: u32,
    pis3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PermuteParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.od0 * params.od1 * params.od2 * params.od3;
    
    if (idx >= total_size) {
        return;
    }
    
    let i3 = idx % params.od3;
    let tmp1 = idx / params.od3;
    let i2 = tmp1 % params.od2;
    let tmp2 = tmp1 / params.od2;
    let i1 = tmp2 % params.od1;
    let i0 = tmp2 / params.od1;
    
    let in_idx = i0 * params.pis0 + i1 * params.pis1 + i2 * params.pis2 + i3 * params.pis3;
    
    output[idx] = input[in_idx];
}
"#;

/// ReLU activation shader
#[allow(dead_code)]
const SHADER_RELU: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    output[idx] = max(0.0, input[idx]);
}
"#;

/// GELU activation shader (approximate)
#[allow(dead_code)]
const SHADER_GELU: &str = r#"
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
const SQRT_2_OVER_PI: f32 = 0.7978845608;

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    let x = input[idx];
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + 0.044715 * x3);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}
"#;

/// SiLU (Swish) activation shader
#[allow(dead_code)]
const SHADER_SILU: &str = r#"
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}
"#;

/// Sigmoid activation shader
#[allow(dead_code)]
const SHADER_SIGMOID: &str = r#"
// sigmoid(x) = 1 / (1 + exp(-x))

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    let x = input[idx];
    output[idx] = 1.0 / (1.0 + exp(-x));
}
"#;

/// MoH Gate Activation Shader
const SHADER_MOH_GATE_ACTIVATION: &str = r#"
struct RichardsParams {
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    input_scale: f32,
    gate_scale: f32,
    gate_bias: f32,
    pad1: u32,
    pad2: u32,
}

struct GateParams {
    total_tokens: u32,
    num_heads: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;      // [T, H]
@group(0) @binding(1) var<storage, read> alpha: array<f32>;      // [H]
@group(0) @binding(2) var<storage, read> beta: array<f32>;       // [H]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [T, H]
@group(0) @binding(4) var<uniform> gate_params: GateParams;
@group(0) @binding(5) var<uniform> richards: RichardsParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x; // Token index
    let h = global_id.y; // Head index
    
    if (t >= gate_params.total_tokens || h >= gate_params.num_heads) {
        return;
    }
    
    let idx = t * gate_params.num_heads + h;
    let x = input[idx];
    let a = alpha[h];
    let b = beta[h];
    
    // 1. Affine transform
    let z = a * x + b;
    
    // 2. Richards Curve
    let adaptive_normalized = richards.adaptive_scale * z + richards.adaptive_shift;
    let temp_scaled = adaptive_normalized * richards.temp_reciprocal;
    let inp = richards.input_scale * (richards.scale * temp_scaled + richards.shift);
    let exponent = -richards.k * (inp - richards.m);
    
    let ln_beta = log(richards.beta);
    let term = ln_beta + exponent;
    
    var sig: f32;
    if (term > 20.0) {
        sig = 0.0;
    } else if (term < -20.0) {
        sig = 1.0;
    } else {
        sig = 1.0 / (1.0 + exp(term));
    }
    
    let y = pow(sig, 1.0 / richards.nu);
    output[idx] = richards.output_gain * y + richards.output_bias;
}
"#;

/// Richards Curve shader
#[allow(dead_code)]
const SHADER_RICHARDS_CURVE: &str = r#"
struct RichardsCurveParams {
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    input_scale: f32,
    gate_scale: f32,
    gate_bias: f32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: RichardsCurveParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&output)) {
        return;
    }

    let xi = input[i];

    // 1. Adaptive Normalization
    let adaptive_normalized = params.adaptive_scale * xi + params.adaptive_shift;
    
    // 2. Temperature Scaling
    let temp_scaled = adaptive_normalized * params.temp_reciprocal;

    // 3. Input Transformation
    let inp = params.input_scale * (params.scale * temp_scaled + params.shift);

    // 4. Richards Exponent
    let exponent = -params.k * (inp - params.m);
    
    // 5. Compute Sigma (Stable log-space formulation)
    let ln_beta = log(params.beta);
    let t = ln_beta + exponent;
    var ln_base: f32;
    if (t > 20.0) {
        ln_base = t;
    } else {
        ln_base = log(1.0 + exp(t));
    }
    let sigma = exp((-1.0 / params.nu) * ln_base);

    // 6. Gate Transformation
    let gate = params.gate_scale * sigma + params.gate_bias;

    // 7. Output Transformation
    output[i] = params.output_gain * gate + params.output_bias;
}
"#;

/// MoH Gate Activation Shader
#[allow(dead_code)]
const SHADER_MOH_GATE: &str = r#"
struct MohParams {
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    input_scale: f32,
    gate_scale: f32,
    gate_bias: f32,
    num_heads: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;   // [B, S, H]
@group(0) @binding(1) var<storage, read> alpha: array<f32>;    // [H]
@group(0) @binding(2) var<storage, read> beta: array<f32>;     // [H]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [B, S, H]
@group(0) @binding(4) var<uniform> params: MohParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= arrayLength(&output)) {
        return;
    }

    let h = i % params.num_heads;
    
    // 1. Linear Mixing
    let val = logits[i] * alpha[h] + beta[h];

    // 2. Richards Curve
    let adaptive_normalized = params.adaptive_scale * val + params.adaptive_shift;
    let temp_scaled = adaptive_normalized * params.temp_reciprocal;
    let inp = params.input_scale * (params.scale * temp_scaled + params.shift);
    let exponent = -params.k * (inp - params.m);
    
    let ln_beta = log(params.beta);
    let t = ln_beta + exponent;
    var ln_base: f32;
    if (t > 20.0) {
        ln_base = t;
    } else {
        ln_base = log(1.0 + exp(t));
    }
    let sigma = exp((-1.0 / params.nu) * ln_base);

    let gate = params.gate_scale * sigma + params.gate_bias;
    output[i] = params.output_gain * gate + params.output_bias;
}
"#;

/// Element-wise multiplication shader
#[allow(dead_code)]
const SHADER_MUL: &str = r#"
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input1)) {
        return;
    }
    
    output[idx] = input1[idx] * input2[idx];
}
"#;

/// Add scaled shader: output += scale * input
#[allow(dead_code)]
const SHADER_ADD_SCALED: &str = r#"
struct AddParams {
    scale: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: AddParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    output[idx] = output[idx] + params.scale * input[idx];
}
"#;

/// Scale shader: output *= scale
#[allow(dead_code)]
const SHADER_SCALE: &str = r#"
struct ScaleParams {
    scale: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: ScaleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    output[idx] = output[idx] * params.scale;
}
"#;

/// AXPY shader: output = a * input1 + b * input2
#[allow(dead_code)]
const SHADER_AXPY: &str = r#"
struct AxpyParams {
    a: f32,
    b: f32,
    size: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: AxpyParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    output[idx] = params.a * input1[idx] + params.b * input2[idx];
}
"#;

/// Layer normalization shader
#[allow(dead_code)]
const SHADER_LAYER_NORM: &str = r#"
struct LayerNormParams {
    batch_size: u32,
    feature_size: u32,
    eps: f32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: LayerNormParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= params.batch_size) {
        return;
    }
    
    let row_start = batch_idx * params.feature_size;
    
    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.feature_size; i = i + 1u) {
        mean = mean + input[row_start + i];
    }
    mean = mean / f32(params.feature_size);
    
    // Compute variance
    var var_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.feature_size; i = i + 1u) {
        let diff = input[row_start + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f32(params.feature_size);
    
    // Normalize and apply affine transform
    let inv_std = 1.0 / sqrt(variance + params.eps);
    
    for (var i: u32 = 0u; i < params.feature_size; i = i + 1u) {
        let idx = row_start + i;
        let normalized = (input[idx] - mean) * inv_std;
        output[idx] = gamma[i] * normalized + beta[i];
    }
}
"#;

/// Fill shader: fill buffer with constant value
const SHADER_FILL: &str = r#"
struct FillParams {
    value: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: FillParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    output[idx] = params.value;
}
"#;

/// Richards Gate shader: output = richards(alpha * input + beta)
const SHADER_RICHARDS_GATE: &str = r#"
struct RichardsGateParams {
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    input_scale: f32,
    gate_scale: f32,
    gate_bias: f32,
    num_heads: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;      // [N, H]
@group(0) @binding(1) var<storage, read> alpha: array<f32>;     // [1, H]
@group(0) @binding(2) var<storage, read> beta: array<f32>;       // [1, H]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [N, H]
@group(0) @binding(4) var<uniform> params: RichardsGateParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = global_id.x; // batch index
    let h = global_id.y; // head index
    
    let total_tokens = params.num_heads; // N * H / H = N
    if (n >= total_tokens || h >= params.num_heads) {
        return;
    }
    
    let idx = n * params.num_heads + h;
    let x = input[idx];
    let a = alpha[h];
    let b = beta[h];
    
    // Affine transform
    let z = a * x + b;
    
    // Richards Curve
    let adaptive_normalized = params.adaptive_scale * z + params.adaptive_shift;
    let temp_scaled = adaptive_normalized * params.temp_reciprocal;
    let inp = params.input_scale * (params.scale * temp_scaled + params.shift);
    let exponent = -params.k * (inp - params.m);
    
    let ln_beta = log(params.beta);
    let t = ln_beta + exponent;
    var ln_base: f32;
    if (t > 20.0) {
        ln_base = t;
    } else {
        ln_base = log(1.0 + exp(t));
    }
    let sigma = exp((-1.0 / params.nu) * ln_base);
    
    let gate = params.gate_scale * sigma + params.gate_bias;
    output[idx] = params.output_gain * gate + params.output_bias;
}
"#;

// ============================================================================
// WGPU Memory Pool
// ============================================================================

/// WGPU-based memory pool for cross-platform GPU memory management
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub struct WgpuMemoryPool {
    device: Device,
    queue: Queue,
    buffers: HashMap<u64, Buffer>,
    next_id: u64,
    total_bytes: usize,
}

#[cfg(feature = "wgpu")]
impl WgpuMemoryPool {
    /// Create a new WGPU memory pool with automatic backend selection
    ///
    /// Uses strict GPU detection - will error if no GPU is available.
    pub async fn new() -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false, // Strict: no CPU fallback
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| ModelError::Backend {
                message: "No GPU adapter found. GPU is required but not available.".to_string(),
            })?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("RustGPT GPU Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::MemoryUsage,
                },
                None, // No trace path
            )
            .await
            .map_err(|e| ModelError::Backend {
                message: format!("Failed to create GPU device: {:?}", e),
            })?;

        Ok(Self {
            device,
            queue,
            buffers: HashMap::new(),
            next_id: 1,
            total_bytes: 0,
        })
    }

    /// Get the underlying wgpu device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the underlying wgpu queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get a buffer by ID
    pub fn get_buffer(&self, id: u64) -> Option<&Buffer> {
        self.buffers.get(&id)
    }

    /// Get a mutable buffer by ID
    pub fn get_buffer_mut(&mut self, id: u64) -> Option<&mut Buffer> {
        self.buffers.get_mut(&id)
    }

    /// Upload data to a buffer
    pub fn upload_to_buffer(&self, buffer_id: u64, data: &[f32]) -> Result<()> {
        let buffer = self
            .buffers
            .get(&buffer_id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found", buffer_id),
            })?;
        if data.len() * std::mem::size_of::<f32>() > buffer.size() as usize {
            return Err(ModelError::Backend {
                message: format!(
                    "WGPU upload exceeds buffer capacity: data={} f32, buffer={} bytes",
                    data.len(),
                    buffer.size()
                ),
            });
        }

        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.queue.write_buffer(buffer, 0, bytes);
        Ok(())
    }

    /// Download data from a buffer (async via staging buffer)
    pub async fn download_from_buffer_async(
        &self,
        buffer_id: u64,
        output: &mut [f32],
    ) -> Result<()> {
        let buffer = self
            .buffers
            .get(&buffer_id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found", buffer_id),
            })?;

        let size = output.len() * std::mem::size_of::<f32>();
        if size > buffer.size() as usize {
            return Err(ModelError::Backend {
                message: format!(
                    "WGPU download exceeds buffer capacity: requested={} bytes, buffer={} bytes",
                    size,
                    buffer.size()
                ),
            });
        }

        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size as u64);

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| ModelError::Backend {
                message: "Failed to receive buffer mapping result".to_string(),
            })?
            .map_err(|e| ModelError::Backend {
                message: format!("Buffer mapping failed: {:?}", e),
            })?;

        let data = buffer_slice.get_mapped_range();
        let mapped: &[f32] = bytemuck::cast_slice(&data);
        output.copy_from_slice(&mapped[..output.len()]);
        drop(data);
        staging_buffer.unmap();

        Ok(())
    }

    /// Download data from a buffer (blocking helper for sync callers).
    pub fn download_from_buffer(&self, buffer_id: u64, output: &mut [f32]) -> Result<()> {
        futures::executor::block_on(self.download_from_buffer_async(buffer_id, output))
    }

    /// Copy data between two device buffers.
    pub fn copy_between_buffers(&self, src_id: u64, dst_id: u64, size_f32: usize) -> Result<()> {
        if size_f32 == 0 || src_id == dst_id {
            return Ok(());
        }

        let src = self
            .buffers
            .get(&src_id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Source buffer {} not found", src_id),
            })?;
        let dst = self
            .buffers
            .get(&dst_id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Destination buffer {} not found", dst_id),
            })?;

        let size_bytes = size_f32 * std::mem::size_of::<f32>();
        if size_bytes > src.size() as usize || size_bytes > dst.size() as usize {
            return Err(ModelError::Backend {
                message: format!(
                    "WGPU copy exceeds buffer capacity: size={} bytes, src={} bytes, dst={} bytes",
                    size_bytes,
                    src.size(),
                    dst.size()
                ),
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Buffer Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size_bytes as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

#[cfg(feature = "wgpu")]
impl GpuMemoryPool for WgpuMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(&format!("RustGPT Buffer {}", self.next_id)),
            size: size_bytes as u64,
            usage: BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.buffers.insert(id, buffer);

        Ok(GpuBuffer { id, size_bytes })
    }

    fn upload(&mut self, data: &[f32]) -> Result<GpuBuffer> {
        let size_bytes = data.len() * std::mem::size_of::<f32>();
        let buffer = self.allocate(size_bytes)?;
        self.upload_to_buffer(buffer.id, data)?;
        Ok(buffer)
    }

    fn download(&mut self, buffer: &GpuBuffer, output: &mut [f32]) -> Result<()> {
        self.download_from_buffer(buffer.id, output)
    }

    fn deallocate(&mut self, buffer: GpuBuffer) {
        if self.buffers.remove(&buffer.id).is_some() {
            self.total_bytes -= buffer.size_bytes;
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.total_bytes = 0;
    }

    fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_bytes: self.total_bytes,
            used_bytes: self.total_bytes,
            free_bytes: 0,
            allocation_count: self.buffers.len() as u32,
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// ============================================================================
// WGPU Matrix Operations
// ============================================================================

/// WGPU-based matrix operations
#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub struct WgpuMatrixOps {
    device: Device,
    queue: Queue,
    /// Shader modules cached for reuse
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    /// Bind group layouts cached for reuse
    bind_group_layouts: HashMap<String, wgpu::BindGroupLayout>,
}

#[cfg(feature = "wgpu")]
impl WgpuMatrixOps {
    /// Create new WGPU matrix operations
    pub fn new(device: Device, queue: Queue) -> Self {
        Self {
            device,
            queue,
            pipelines: HashMap::new(),
            bind_group_layouts: HashMap::new(),
        }
    }

    /// Create or get cached pipeline
    fn get_or_create_pipeline(
        &mut self,
        name: &str,
        shader_code: &str,
        layout_entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Result<wgpu::ComputePipeline> {
        if let Some(pipeline) = self.pipelines.get(name) {
            return Ok(pipeline.clone());
        }

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} BindGroupLayout", name)),
                    entries: layout_entries,
                });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} PipelineLayout", name)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", name)),
                source: wgpu::ShaderSource::Wgsl(shader_code.into()),
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        self.bind_group_layouts
            .insert(name.to_string(), bind_group_layout);
        self.pipelines.insert(name.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    /// Resolve a GpuBuffer to a WGPU Buffer
    fn resolve_buffer<'a>(pool: &'a dyn GpuMemoryPool, id: u64) -> Result<&'a Buffer> {
        let wgpu_pool = pool
            .as_any()
            .downcast_ref::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;

        wgpu_pool
            .buffers
            .get(&id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found in WGPU pool", id),
            })
    }
}

// ============================================================================
// Parameter Structs
// ============================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmParams {
    alpha: f32,
    beta: f32,
    m: u32,
    n: u32,
    k: u32,
    trans_a: u32,
    trans_b: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams {
    rows: u32,
    cols: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AddParams {
    scale: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
    scale: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    a: f32,
    b: f32,
    size: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    batch_size: u32,
    feature_size: u32,
    eps: f32,
    pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FillParams {
    value: f32,
    size: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GateParams {
    total_tokens: u32,
    num_heads: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MohGateParams {
    num_heads: u32,
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    output_gain: f32,
    output_bias: f32,
    scale: f32,
    shift: f32,
    adaptive_scale: f32,
    adaptive_shift: f32,
    input_scale: f32,
    gate_scale: f32,
    gate_bias: f32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PermuteParams {
    od0: u32,
    od1: u32,
    od2: u32,
    od3: u32,
    pis0: u32,
    pis1: u32,
    pis2: u32,
    pis3: u32,
}

// ============================================================================
// GpuMatrixOps Implementation
// ============================================================================

#[cfg(feature = "wgpu")]
impl GpuMatrixOps for WgpuMatrixOps {
    fn fill_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        buffer: &mut GpuBuffer,
        value: f32,
    ) -> Result<()> {
        let buf = Self::resolve_buffer(pool, buffer.id)?;

        let params = FillParams {
            value,
            size: (buffer.size_bytes / 4) as u32,
            pad1: 0,
            pad2: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fill Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "fill",
            SHADER_FILL,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fill Bind Group"),
            layout: &self.bind_group_layouts["fill"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Fill Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fill Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (buffer.size_bytes / 4 + 255) / 256;
            cpass.dispatch_workgroups(workgroups as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

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
    ) -> Result<()> {
        let buf_input = Self::resolve_buffer(pool, input.id)?;
        let buf_alpha = Self::resolve_buffer(pool, alpha.id)?;
        let buf_beta = Self::resolve_buffer(pool, beta.id)?;
        let buf_output = Self::resolve_buffer(pool, output.id)?;

        let gp = GateParams {
            total_tokens: batch_size as u32,
            num_heads: num_heads as u32,
            pad1: 0,
            pad2: 0,
        };

        let gp_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Richards Gate Params"),
                contents: bytemuck::cast_slice(&[gp]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let richards_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Richards Params"),
                contents: bytemuck::cast_slice(&[*params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "richards_gate",
            SHADER_RICHARDS_GATE,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Richards Gate Bind Group"),
            layout: &self.bind_group_layouts["richards_gate"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_alpha.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_beta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gp_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Richards Gate Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Richards Gate Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let x_groups = (batch_size as u32 + 15) / 16;
            let y_groups = (num_heads as u32 + 15) / 16;
            cpass.dispatch_workgroups(x_groups, y_groups, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

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
    ) -> Result<()> {
        // Handle zero-dimension GEMM as no-op
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }

        let buf_a = Self::resolve_buffer(pool, a.id)?;
        let buf_b = Self::resolve_buffer(pool, b.id)?;
        let buf_c = Self::resolve_buffer(pool, output.id)?;

        let params = GemmParams {
            alpha,
            beta,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            trans_a: if trans_a { 1 } else { 0 },
            trans_b: if trans_b { 1 } else { 0 },
            pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gemm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "gemm",
            SHADER_GEMM,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gemm Bind Group"),
            layout: &self.bind_group_layouts["gemm"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Gemm Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gemm Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 16;
            let x_groups = (m as u32 + workgroup_size - 1) / workgroup_size;
            let y_groups = (n as u32 + workgroup_size - 1) / workgroup_size;
            cpass.dispatch_workgroups(x_groups, y_groups, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

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
        strides: [usize; 3],
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        let wgpu_pool = pool
            .as_any()
            .downcast_ref::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;

        let buf_a = wgpu_pool
            .get_buffer(a.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found", a.id),
            })?;
        let buf_b = wgpu_pool
            .get_buffer(b.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found", b.id),
            })?;
        let buf_c = wgpu_pool
            .get_buffer(output.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Buffer {} not found", output.id),
            })?;

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct GemmBatchedParams {
            alpha: f32,
            beta: f32,
            m: u32,
            n: u32,
            k: u32,
            batch_count: u32,
            stride_a: u32,
            stride_b: u32,
            stride_c: u32,
            trans_a: u32,
            trans_b: u32,
            pad: u32,
        }

        let params = GemmBatchedParams {
            alpha,
            beta,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            batch_count: batch_count as u32,
            stride_a: strides[0] as u32,
            stride_b: strides[1] as u32,
            stride_c: strides[2] as u32,
            trans_a: if trans_a { 1 } else { 0 },
            trans_b: if trans_b { 1 } else { 0 },
            pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gemm Batched Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "gemm_batched",
            SHADER_GEMM_BATCHED,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gemm Batched Bind Group"),
            layout: &self.bind_group_layouts["gemm_batched"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Gemm Batched Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gemm Batched Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let x_groups = (m as u32 + 15) / 16;
            let y_groups = (n as u32 + 15) / 16;
            let z_groups = batch_count as u32;
            cpass.dispatch_workgroups(x_groups, y_groups, z_groups);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

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
    ) -> Result<()> {
        // gemv is gemm with k=n, n=1
        // A is m x n, x is n x 1, output is m x 1
        self.gemm_f32(pool, alpha, a, x, beta, output, m, 1, n, false, false)
    }

    fn relu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "relu",
            SHADER_RELU,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReLU Bind Group"),
            layout: &self.bind_group_layouts["relu"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ReLU Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReLU Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn gelu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "gelu",
            SHADER_GELU,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GELU Bind Group"),
            layout: &self.bind_group_layouts["gelu"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("GELU Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GELU Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn silu(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "silu",
            SHADER_SILU,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SiLU Bind Group"),
            layout: &self.bind_group_layouts["silu"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("SiLU Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SiLU Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn sigmoid(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "sigmoid",
            SHADER_SIGMOID,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sigmoid Bind Group"),
            layout: &self.bind_group_layouts["sigmoid"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Sigmoid Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sigmoid Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn mul(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input1: &GpuBuffer,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in1 = Self::resolve_buffer(pool, input1.id)?;
        let buf_in2 = Self::resolve_buffer(pool, input2.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let pipeline = self.get_or_create_pipeline(
            "mul",
            SHADER_MUL,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mul Bind Group"),
            layout: &self.bind_group_layouts["mul"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_in2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Mul Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mul Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn add_scaled(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scale: f32,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params = AddParams {
            scale,
            size: size as u32,
            pad1: 0,
            pad2: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Add Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "add_scaled",
            SHADER_ADD_SCALED,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Scaled Bind Group"),
            layout: &self.bind_group_layouts["add_scaled"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Add Scaled Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Add Scaled Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn scale(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scale: f32,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params = ScaleParams {
            scale,
            size: size as u32,
            pad1: 0,
            pad2: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scale Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "scale",
            SHADER_SCALE,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scale Bind Group"),
            layout: &self.bind_group_layouts["scale"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Scale Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Scale Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn axpy(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        a: f32,
        input1: &GpuBuffer,
        b: f32,
        input2: &GpuBuffer,
        output: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let buf_in1 = Self::resolve_buffer(pool, input1.id)?;
        let buf_in2 = Self::resolve_buffer(pool, input2.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params = AxpyParams {
            a,
            b,
            size: size as u32,
            pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Axpy Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "axpy",
            SHADER_AXPY,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Axpy Bind Group"),
            layout: &self.bind_group_layouts["axpy"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_in2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Axpy Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Axpy Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn richards_curve(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsCurveParams,
        size: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Richards Params"),
                contents: bytemuck::cast_slice(&[*params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "richards_curve",
            SHADER_RICHARDS_CURVE,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Richards Bind Group"),
            layout: &self.bind_group_layouts["richards_curve"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Richards Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Richards Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

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
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_gamma = Self::resolve_buffer(pool, gamma.id)?;
        let buf_beta = Self::resolve_buffer(pool, beta.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params = LayerNormParams {
            batch_size: batch_size as u32,
            feature_size: feature_size as u32,
            eps,
            pad: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LayerNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "layer_norm",
            SHADER_LAYER_NORM,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LayerNorm Bind Group"),
            layout: &self.bind_group_layouts["layer_norm"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_gamma.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_beta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("LayerNorm Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LayerNorm Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (batch_size as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn softmax(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let buf_in = Self::resolve_buffer(pool, input.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let params = SoftmaxParams {
            rows: rows as u32,
            cols: cols as u32,
            pad1: 0,
            pad2: 0,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Softmax Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "softmax",
            SHADER_SOFTMAX,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Softmax Bind Group"),
            layout: &self.bind_group_layouts["softmax"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Softmax Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Softmax Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (rows as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    fn moh_gate_activation(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        logits: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        gate_params: &RichardsCurveParams,
        output: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
    ) -> Result<()> {
        let buf_logits = Self::resolve_buffer(pool, logits.id)?;
        let buf_alpha = Self::resolve_buffer(pool, alpha.id)?;
        let buf_beta = Self::resolve_buffer(pool, beta.id)?;
        let buf_out = Self::resolve_buffer(pool, output.id)?;

        let gp = GateParams {
            total_tokens: batch_size as u32,
            num_heads: num_heads as u32,
            pad1: 0,
            pad2: 0,
        };

        let gp_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MoH Gate Params"),
                contents: bytemuck::cast_slice(&[gp]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let richards_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MoH Richards Params"),
                contents: bytemuck::cast_slice(&[*gate_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "moh_gate_activation",
            SHADER_MOH_GATE_ACTIVATION,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MoH Gate Activation Bind Group"),
            layout: &self.bind_group_layouts["moh_gate_activation"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_logits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_alpha.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_beta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: gp_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: richards_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("MoH Gate Activation Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MoH Gate Activation Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let x_groups = (batch_size as u32 + 15) / 16;
            let y_groups = (num_heads as u32 + 15) / 16;
            cpass.dispatch_workgroups(x_groups, y_groups, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
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
        Err(ModelError::Backend {
            message: "WGPU poly_attention_fused not implemented".to_string(),
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
        Err(ModelError::Backend {
            message: "WGPU blr_projection not implemented".to_string(),
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
        Err(ModelError::Backend {
            message: "WGPU compute_cope_scores not implemented".to_string(),
        })
    }

    fn sum(&self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32> {
        // Fallback: download to CPU and sum
        let mut data = vec![0.0; size];
        self.download(pool, buffer, &mut data)?;
        Ok(data.iter().sum())
    }

    fn mean(&self, pool: &mut dyn GpuMemoryPool, buffer: &GpuBuffer, size: usize) -> Result<f32> {
        // Fallback: download to CPU and mean
        let mut data = vec![0.0; size];
        self.download(pool, buffer, &mut data)?;
        if size == 0 {
            return Ok(0.0);
        }
        Ok(data.iter().sum::<f32>() / size as f32)
    }

    fn download(
        &self,
        pool: &mut dyn GpuMemoryPool,
        gpu_buffer: &GpuBuffer,
        cpu_data: &mut [f32],
    ) -> Result<()> {
        let wgpu_pool = pool
            .as_any()
            .downcast_ref::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;
        wgpu_pool.download_from_buffer(gpu_buffer.id, cpu_data)
    }

    fn upload(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        cpu_data: &[f32],
        gpu_buffer: &mut GpuBuffer,
    ) -> Result<()> {
        let wgpu_pool = pool
            .as_any_mut()
            .downcast_mut::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;
        wgpu_pool.upload_to_buffer(gpu_buffer.id, cpu_data)
    }

    fn copy_within_device(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        src: &GpuBuffer,
        dst: &mut GpuBuffer,
        size: usize,
    ) -> Result<()> {
        let wgpu_pool = pool
            .as_any()
            .downcast_ref::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;
        wgpu_pool.copy_between_buffers(src.id, dst.id, size)
    }

    fn permute_4d(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        output_dims: [usize; 4],
        permuted_input_strides: [usize; 4],
    ) -> Result<()> {
        let wgpu_pool = pool
            .as_any()
            .downcast_ref::<WgpuMemoryPool>()
            .ok_or_else(|| ModelError::Backend {
                message: "Pool is not a WgpuMemoryPool".to_string(),
            })?;

        let input_buf = wgpu_pool
            .get_buffer(input.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Input buffer {} not found", input.id),
            })?;

        let output_buf = wgpu_pool
            .get_buffer(output.id)
            .ok_or_else(|| ModelError::Backend {
                message: format!("Output buffer {} not found", output.id),
            })?;

        let params = PermuteParams {
            od0: output_dims[0] as u32,
            od1: output_dims[1] as u32,
            od2: output_dims[2] as u32,
            od3: output_dims[3] as u32,
            pis0: permuted_input_strides[0] as u32,
            pis1: permuted_input_strides[1] as u32,
            pis2: permuted_input_strides[2] as u32,
            pis3: permuted_input_strides[3] as u32,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Permute Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let pipeline = self.get_or_create_pipeline(
            "permute_4d",
            SHADER_PERMUTE_4D,
            &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        )?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Permute Bind Group"),
            layout: &self.bind_group_layouts["permute_4d"],
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Permute Encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Permute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            let total_elements: usize = output_dims.iter().product();
            let workgroups = (total_elements as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

// ============================================================================
// Non-wgpu fallback (stub implementations)
// ============================================================================

#[cfg(not(feature = "wgpu"))]
pub struct WgpuMemoryPool;

#[cfg(not(feature = "wgpu"))]
impl WgpuMemoryPool {
    pub fn new() -> Result<Self> {
        Err(ModelError::Backend {
            message: "WGPU feature not enabled. Compile with --features gpu-wgpu".to_string(),
        })
    }
}

#[cfg(not(feature = "wgpu"))]
pub struct WgpuMatrixOps;

#[cfg(not(feature = "wgpu"))]
impl WgpuMatrixOps {
    pub fn new() -> Result<Self> {
        Err(ModelError::Backend {
            message: "WGPU feature not enabled. Compile with --features gpu-wgpu".to_string(),
        })
    }
}

// ============================================================================
// RichardsGluFusedParams (for external use)
// ============================================================================

/// Parameters for RichardsGLU Fused Kernel
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RichardsGluFusedParams {
    pub batch_size: u32,
    pub input_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
    pub nu: f32,
    pub k: f32,
    pub m: f32,
    pub beta: f32,
    pub temp_reciprocal: f32,
    pub gate_scale: f32,
    pub gate_bias: f32,
    pub gate_temp_reciprocal: f32,
    pub value_scale: f32,
    pub output_gain: f32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgpu_memory_pool_creation() {
        if let Ok(pool) = WgpuMemoryPool::new().await {
            let stats = pool.memory_stats();
            println!("GPU Memory Stats: {}", stats.format_human());
            assert_eq!(stats.allocation_count, 0);
        } else {
            println!("No GPU available, skipping test");
        }
    }

    #[tokio::test]
    async fn test_wgpu_buffer_allocation() {
        if let Ok(mut pool) = WgpuMemoryPool::new().await {
            let buffer = pool.allocate(1024 * std::mem::size_of::<f32>());
            assert!(buffer.is_ok());

            let stats = pool.memory_stats();
            assert_eq!(stats.allocation_count, 1);

            pool.deallocate(buffer.unwrap());
            let stats = pool.memory_stats();
            assert_eq!(stats.allocation_count, 0);
        } else {
            println!("No GPU available, skipping test");
        }
    }
}
