//! WGSL Kernel Implementations for Unified GPU Operations
//!
//! This module contains the WGSL source code for all GPU kernels.
//! Each kernel is designed for maximum portability across Vulkan, Metal, and DX12.
//!
//! ## Performance Tuning
//!
//! - **Workgroup size**: 256 for element-wise, 16×16 for matrix ops
//! - **Memory access**: Linear coalesced patterns for VRAM bandwidth
//! - **Precision**: f32 for numerical stability (ε ≤ 1e-4)

// ============================================================================
// Richards Curve Activation Kernel
// ============================================================================

/// Richards activation function: σ(x) = 1 / (1 + (k*m)^(1/m) * exp(-β*(x-ν)))
///
/// Computes element-wise Richards curve with stable exponential handling.
/// Prevents overflow/underflow through clamping and log-space formulation.
pub const SHADER_RICHARDS_CURVE: &str = r#"
struct RichardsParams {
    nu: f32,              // Center/inflection point
    k: f32,               // Growth rate steepness
    m: f32,               // Shape parameter (asymmetry)
    beta: f32,            // Scale/temperature
    temp_reciprocal: f32, // 1/temperature (for output scaling)
    output_gain: f32,     // Multiplicative output scaling
    output_bias: f32,     // Additive output scaling
    scale: f32,           // Scaling coefficient
    shift: f32,           // Shift coefficient
    adaptive_scale: f32,  // Adaptive scaling per-element
    adaptive_shift: f32,  // Adaptive shift per-element
    num_heads: u32,       // Number of attention heads (for multi-head activation)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: RichardsParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    let x = input[idx];
    
    // Stable log-space formulation:
    // Richards(x) = 1 / (1 + exp(-β * ln(1 + (k*m)^(1/m))))
    // Where the exponent is: -β * (x - ν)
    
    let center = x - params.nu;
    let exponent = -params.beta * center;
    
    // Numerically stable exponential
    let exp_val: f32;
    if (exponent > 20.0) {
        // exp(x) → ∞, so 1/(1+∞) → 0
        exp_val = 1e38; // Large value to compute 1/(1+large) safely
    } else if (exponent < -20.0) {
        // exp(x) → 0, so 1/(1+0) → 1
        exp_val = 1e-38; // Small value for numerical stability
    } else {
        exp_val = exp(exponent);
    }
    
    // Base: (k * m) ^ (1/m)
    let base_ln = log(params.k * params.m) / params.m;
    let base = exp(base_ln);
    
    // Richards: 1 / (1 + base * exp(exponent))
    let denominator = 1.0 + base * exp_val;
    let sigma = 1.0 / (denominator + 1e-8);
    
    // Apply output transformation
    let result = sigma * params.output_gain + params.output_bias;
    
    // Scale by temperature and adaptive parameters
    output[idx] = result * params.temp_reciprocal * params.adaptive_scale + params.adaptive_shift;
}
"#;

// ============================================================================
// Richards GLU Fused Kernel (Pass 1: Activation)
// ============================================================================

/// RichardsGLU Fused Pass 1: Compute activations
///
/// Computes:
/// - x1 = input @ w1
/// - x2 = input @ w2
/// - value = x1 * richards(x1)
/// - gate = richards(x2)
/// - gated = value * gate
///
/// All computation stays on GPU (zero-copy).
pub const SHADER_RICHARDS_GLU_PASS1: &str = r#"
struct RichardsGluParams {
    batch_size: u32,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    
    // Richards activation parameters
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    temp_reciprocal: f32,
    
    // Gate parameters
    gate_scale: f32,
    gate_bias: f32,
    gate_temp_reciprocal: f32,
    value_scale: f32,
    output_gain: f32,
    
    // Padding for 16-byte alignment (64 bytes total)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x1: array<f32>;      // input @ w1
@group(0) @binding(1) var<storage, read> x2: array<f32>;      // input @ w2
@group(0) @binding(2) var<storage, read_write> value: array<f32>;  // value buffer
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;   // gate buffer
@group(0) @binding(4) var<uniform> params: RichardsGluParams;

// Helper: Richards activation
fn richards_activation(x: f32) -> f32 {
    let center = x - params.nu;
    let exp_val = exp(-params.beta * center);
    let base = pow(params.k * params.m, 1.0 / params.m);
    let denom = 1.0 + base * exp_val;
    return 1.0 / (denom + 1e-8);
}

// Helper: Gate activation (simplified Richards)
fn gate_activation(x: f32) -> f32 {
    let scaled = x * params.gate_temp;
    let exp_val = exp(-params.gate_bias * scaled);
    return 1.0 / (1.0 + exp_val + 1e-8);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.hidden_dim;
    
    if (idx >= total) {
        return;
    }
    
    // Compute activations element-wise
    let x1_val = x1[idx];
    let x2_val = x2[idx];
    
    // value = x1 * richards(x1)
    let sigma1 = richards_activation(x1_val);
    value[idx] = x1_val * sigma1 * params.temp_reciprocal;
    
    // gate = richards(x2)
    gate[idx] = gate_activation(x2_val);
}
"#;

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise multiply: output = input1 * input2
pub const SHADER_MUL: &str = r#"
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

/// Element-wise add scaled: output += scale * input
pub const SHADER_ADD_SCALED: &str = r#"
struct AddScaledParams {
    scale: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: AddScaledParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input)) {
        return;
    }
    
    output[idx] += params.scale * input[idx];
}
"#;

/// Element-wise scale: output *= scale
pub const SHADER_SCALE: &str = r#"
struct ScaleParams {
    scale: f32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@group(0) @binding(1) var<uniform> params: ScaleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] *= params.scale;
}
"#;

/// Element-wise AXPY: output = a * input1 + b * input2
pub const SHADER_AXPY: &str = r#"
struct AxpyParams {
    a: f32,
    b: f32,
}

@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: AxpyParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&input1)) {
        return;
    }
    
    output[idx] = params.a * input1[idx] + params.b * input2[idx];
}
"#;

// ============================================================================
// Reduction Operations
// ============================================================================

/// Sum reduction with workgroup shared memory
pub const SHADER_SUM: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // Single element

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let idx = global_id.x;
    let total = arrayLength(&input);
    
    // Load data into shared memory
    var sum: f32 = 0.0;
    if (idx < total) {
        sum = input[idx];
    }
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    // Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (tid == 0u) {
        output[0] = shared_sum[0];
    }
}
"#;

/// Mean reduction: output = sum(input) / count
pub const SHADER_MEAN: &str = r#"
struct MeanParams {
    count: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // Single element
@group(0) @binding(2) var<uniform> params: MeanParams;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let idx = global_id.x;
    
    // Load data into shared memory
    var sum: f32 = 0.0;
    if (idx < params.count) {
        sum = input[idx];
    }
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    // Tree reduction
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (tid == 0u) {
        output[0] = shared_sum[0] / f32(params.count);
    }
}
"#;

// ============================================================================
// Gate Activation (for PolyAttention and other multi-head operations)
// ============================================================================

/// MoH Gate Activation: G = Richards(alpha * (Input @ W_g) + beta)
///
/// Applies Richards curve to gating logits with per-head scaling.
pub const SHADER_MOH_GATE: &str = r#"
struct MohGateParams {
    batch_size: u32,
    num_heads: u32,
    
    // Richards parameters
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;      // Input @ W_g
@group(0) @binding(1) var<storage, read> alpha: array<f32>;       // Per-head scaling
@group(0) @binding(2) var<storage, read> beta_vals: array<f32>;   // Per-head bias
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // Gate output
@group(0) @binding(4) var<uniform> params: MohGateParams;

fn richards_activation(x: f32) -> f32 {
    let center = x - params.nu;
    let exp_val = exp(-params.beta * center);
    let base = pow(params.k * params.m, 1.0 / params.m);
    let denom = 1.0 + base * exp_val;
    return 1.0 / (denom + 1e-8);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.num_heads;
    
    if (idx >= total) {
        return;
    }
    
    let head_idx = idx % params.num_heads;
    
    // Apply per-head scaling and bias
    let scaled_logit = logits[idx] * alpha[head_idx] + beta_vals[head_idx];
    
    // Apply Richards activation
    output[idx] = richards_activation(scaled_logit);
}
"#;
