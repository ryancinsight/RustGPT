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

// ============================================================================
// MoE Routing Kernels
// ============================================================================

/// Top-K Selection Shader
///
/// Finds the top-K experts for each token. 
/// Output format: topk_indices = [T, K] (u32), topk_weights = [T, K] (f32)
pub const SHADER_TOPK: &str = r#"
struct TopKParams {
    num_tokens: u32,
    num_experts: u32,
    k: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> routing_gates: array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_weights: array<f32>;
@group(0) @binding(3) var<uniform> params: TopKParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= params.num_tokens) {
        return;
    }

    // Very simple insertion sort/selection for small K (e.g. K=2)
    // K is usually <= 4 for MoE, so an O(N * K) loop is acceptable.
    
    // Initialize best K arrays
    var best_scores: array<f32, 8>;
    var best_indices: array<u32, 8>;
    for (var i = 0u; i < params.k; i++) {
        best_scores[i] = -1e38;
        best_indices[i] = 0u;
    }

    for (var e = 0u; e < params.num_experts; e++) {
        let score = routing_gates[token_idx * params.num_experts + e];
        
        // Insertion sort check
        if (score > best_scores[params.k - 1u]) {
            var insert_idx = params.k - 1u;
            while (insert_idx > 0u && score > best_scores[insert_idx - 1u]) {
                insert_idx--;
            }
            
            // Shift right
            for (var j = params.k - 1u; j > insert_idx; j--) {
                best_scores[j] = best_scores[j - 1u];
                best_indices[j] = best_indices[j - 1u];
            }
            
            best_scores[insert_idx] = score;
            best_indices[insert_idx] = e;
        }
    }

    // Normalize weights if using Softmax/TopK standard routing
    var sum = 0.0;
    for (var i = 0u; i < params.k; i++) {
        sum += best_scores[i];
    }
    let inv_sum = 1.0 / max(sum, 1e-8);

    for (var i = 0u; i < params.k; i++) {
        let out_idx = token_idx * params.k + i;
        topk_indices[out_idx] = best_indices[i];
        topk_weights[out_idx] = best_scores[i] * inv_sum;
    }
}
"#;

/// Scatter Tokens Shader
///
/// Creates contiguous buffers of tokens for each expert.
/// Note: We assume a CPU pre-pass or an atomic counter pass computes expert capacities/offsets.
pub const SHADER_SCATTER_EXPERTS: &str = r#"
struct ScatterParams {
    num_tokens: u32,
    hidden_dim: u32,
    k: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> hidden_states: array<f32>;
@group(0) @binding(1) var<storage, read> topk_indices: array<u32>;
@group(0) @binding(2) var<storage, read> global_expert_offsets: array<u32>; // Pre-computed offsets per expert
@group(0) @binding(3) var<storage, read_write> expert_counters: array<atomic<u32>>;\n@group(0) @binding(4) var<storage, read_write> scattered_hidden: array<f32>;
@group(0) @binding(5) var<storage, read_write> original_token_indices: array<u32>;
@group(0) @binding(6) var<uniform> params: ScatterParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= params.num_tokens) {
        return;
    }

    for (var i = 0u; i < params.k; i++) {
        let expert_idx = topk_indices[token_idx * params.k + i];
        
        // Get a target slot for this token in the expert's buffer
        let slot = atomicAdd(&expert_counters[expert_idx], 1u);
        let offset = global_expert_offsets[expert_idx];
        let write_idx = offset + slot;
        
        // Write reverse-mapping index
        original_token_indices[write_idx] = token_idx * params.k + i;
        
        // Copy hidden state
        for (var d = 0u; d < params.hidden_dim; d++) {
            scattered_hidden[write_idx * params.hidden_dim + d] = hidden_states[token_idx * params.hidden_dim + d];
        }
    }
}
"#;

/// Gather Tokens Shader (Token-parallel approach)
///
/// Since standard WGSL does not support `atomicAdd` for `f32`, we parallelize 
/// over the output tokens rather than the input packed tokens.
pub const SHADER_GATHER_EXPERTS: &str = r#"
struct GatherParams {
    num_tokens: u32,
    hidden_dim: u32,
    k: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> expert_outputs: array<f32>;
@group(0) @binding(1) var<storage, read> topk_weights: array<f32>;
@group(0) @binding(2) var<storage, read> topk_indices: array<u32>;
@group(0) @binding(3) var<storage, read> global_expert_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> token_expert_slots: array<u32>;
@group(0) @binding(5) var<storage, read_write> gathered_output: array<f32>;
@group(0) @binding(6) var<uniform> params: GatherParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= params.num_tokens) {
        return;
    }

    // For this target token, sum the results from its K chosen experts
    for (var d = 0u; d < params.hidden_dim; d++) {
        var sum_val: f32 = 0.0;
        
        for (var i = 0u; i < params.k; i++) {
            let expert_idx = topk_indices[token_idx * params.k + i];
            let weight = topk_weights[token_idx * params.k + i];
            
            // Find where this token's output is packed inside the expert's buffer
            let slot = token_expert_slots[token_idx * params.k + i];
            let offset = global_expert_offsets[expert_idx];
            let read_idx = offset + slot;
            
            sum_val += weight * expert_outputs[read_idx * params.hidden_dim + d];
        }
        
        gathered_output[token_idx * params.hidden_dim + d] = sum_val;
    }
}
"#;


// ============================================================================
// Titans Memory Kernels
// ============================================================================

/// Titans MLP Forward Kernel
///
/// Batched MLP forward for neural memory over num_tokens keys:
///   z = W1 @ k + b1, h = ReLU(z), v_pred = W2 @ h + b2
///
/// Dispatch: 1D over tokens.
pub const SHADER_TITANS_MLP_FORWARD: &str = r#"
struct TitansMlpParams {
    num_tokens: u32,
    key_dim: u32,
    hidden_dim: u32,
    val_dim: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> b1: array<f32>;
@group(0) @binding(3) var<storage, read> w2: array<f32>;
@group(0) @binding(4) var<storage, read> b2: array<f32>;
@group(0) @binding(5) var<storage, read_write> z_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> h_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_pred: array<f32>;
@group(0) @binding(8) var<uniform> params: TitansMlpParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token_idx = global_id.x;
    if (token_idx >= params.num_tokens) { return; }

    for (var h_i = 0u; h_i < params.hidden_dim; h_i++) {
        var acc: f32 = b1[h_i];
        for (var d = 0u; d < params.key_dim; d++) {
            acc += w1[h_i * params.key_dim + d] * keys[token_idx * params.key_dim + d];
        }
        z_out[token_idx * params.hidden_dim + h_i] = acc;
        h_out[token_idx * params.hidden_dim + h_i] = max(acc, 0.0);
    }

    for (var v_i = 0u; v_i < params.val_dim; v_i++) {
        var acc: f32 = b2[v_i];
        for (var h_i = 0u; h_i < params.hidden_dim; h_i++) {
            acc += w2[v_i * params.hidden_dim + h_i] * h_out[token_idx * params.hidden_dim + h_i];
        }
        v_pred[token_idx * params.val_dim + v_i] = acc;
    }
}
"#;

/// Titans W2/b2 Gradient Accumulation Kernel
///
/// Accumulates:
///   grad_w2[v,h] += (v_pred - v_target)[t,v] * h_act[t,h]
///   grad_b2[v]   += (v_pred - v_target)[t,v]
///
/// Dispatch: 2D workgroups of (val_dim, hidden_dim+1).
pub const SHADER_TITANS_GRAD_W2: &str = r#"
struct TitansGradW2Params {
    num_tokens: u32,
    hidden_dim: u32,
    val_dim: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> v_target: array<f32>;
@group(0) @binding(1) var<storage, read> v_pred: array<f32>;
@group(0) @binding(2) var<storage, read> h_act: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_w2: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_b2: array<f32>;
@group(0) @binding(5) var<uniform> params: TitansGradW2Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let v_idx = global_id.x;
    let col   = global_id.y;
    if (v_idx >= params.val_dim) { return; }

    var acc: f32 = 0.0;
    for (var t = 0u; t < params.num_tokens; t++) {
        let g = v_pred[t * params.val_dim + v_idx] - v_target[t * params.val_dim + v_idx];
        if (col < params.hidden_dim) {
            acc += g * h_act[t * params.hidden_dim + col];
        } else {
            acc += g;
        }
    }

    if (col < params.hidden_dim) {
        grad_w2[v_idx * params.hidden_dim + col] += acc;
    } else {
        grad_b2[v_idx] += acc;
    }
}
"#;

/// Titans W1/b1 Gradient Accumulation Kernel
///
/// Accumulates:
///   grad_h = W2^T @ grad_out
///   grad_z = grad_h * step(z)
///   grad_w1[h,k] += grad_z * k
///   grad_b1[h]   += grad_z
///
/// Dispatch: 2D workgroups of (hidden_dim, key_dim+1).
pub const SHADER_TITANS_GRAD_W1: &str = r#"
struct TitansGradW1Params {
    num_tokens: u32,
    key_dim: u32,
    hidden_dim: u32,
    val_dim: u32,
}

@group(0) @binding(0) var<storage, read> keys: array<f32>;
@group(0) @binding(1) var<storage, read> v_target: array<f32>;
@group(0) @binding(2) var<storage, read> v_pred: array<f32>;
@group(0) @binding(3) var<storage, read> z: array<f32>;
@group(0) @binding(4) var<storage, read> w2: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_w1: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_b1: array<f32>;
@group(0) @binding(7) var<uniform> params: TitansGradW1Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h_idx = global_id.x;
    let col   = global_id.y;
    if (h_idx >= params.hidden_dim) { return; }

    var acc: f32 = 0.0;
    for (var t = 0u; t < params.num_tokens; t++) {
        var grad_h: f32 = 0.0;
        for (var v = 0u; v < params.val_dim; v++) {
            let g = v_pred[t * params.val_dim + v] - v_target[t * params.val_dim + v];
            grad_h += w2[v * params.hidden_dim + h_idx] * g;
        }
        let grad_z = grad_h * select(0.0, 1.0, z[t * params.hidden_dim + h_idx] > 0.0);
        if (col < params.key_dim) {
            acc += grad_z * keys[t * params.key_dim + col];
        } else {
            acc += grad_z;
        }
    }

    if (col < params.key_dim) {
        grad_w1[h_idx * params.key_dim + col] += acc;
    } else {
        grad_b1[h_idx] += acc;
    }
}
"#;

/// Titans Memory Weight Update Kernel
///
/// Fused per-element momentum and memory update:
///   momentum = eta * momentum - theta * grad
///   memory   = (1 - alpha) * memory + momentum
///
/// Dispatch: 1D over num_elements.
pub const SHADER_TITANS_MEMORY_UPDATE: &str = r#"
struct TitansUpdateParams {
    num_elements: u32,
    alpha: f32,
    eta: f32,
    theta: f32,
}

@group(0) @binding(0) var<storage, read> grad: array<f32>;
@group(0) @binding(1) var<storage, read_write> momentum: array<f32>;
@group(0) @binding(2) var<storage, read_write> memory: array<f32>;
@group(0) @binding(3) var<uniform> params: TitansUpdateParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) { return; }
    let m = params.eta * momentum[idx] - params.theta * grad[idx];
    momentum[idx] = m;
    memory[idx]   = (1.0 - params.alpha) * memory[idx] + m;
}
"#;

// ============================================================================
// Selective State Space Model (SSM) Kernels
// ============================================================================

/// Selective Scan Forward Kernel
///
/// Implements the linear recurrence for State Space Models (Mamba-style):
///   h_t = A @ h_{t-1} + B @ x_t
///   y_t = C @ h_t + D @ x_t
///
/// This is the core O(N) SSM scan. Because each step depends on the previous
/// hidden state, the loop is sequential. We parallelise across the embed_dim
/// independently (one workgroup per element of the output vector).
///
/// Layouts (all row-major):
///   input:   [seq_len, embed_dim]
///   a:       [state_dim, state_dim]
///   b:       [state_dim, embed_dim]
///   c:       [embed_dim, state_dim]
///   d:       [embed_dim, embed_dim]
///   h_init:  [state_dim]
///   output:  [seq_len, embed_dim]
///   h_final: [state_dim]
///
/// Dispatch: (state_dim, 1, 1) for h update, then (embed_dim, 1, 1) for y.
pub const SHADER_SELECTIVE_SCAN_FORWARD: &str = r#"
struct ScanParams {
    seq_len:   u32,
    state_dim: u32,
    embed_dim: u32,
    pad:       u32,
}

// All matrices uploaded as flat row-major f32 arrays.
@group(0) @binding(0) var<storage, read>       input:   array<f32>; // [S, E]
@group(0) @binding(1) var<storage, read>       mat_a:   array<f32>; // [H, H]
@group(0) @binding(2) var<storage, read>       mat_b:   array<f32>; // [H, E]
@group(0) @binding(3) var<storage, read>       mat_c:   array<f32>; // [E, H]
@group(0) @binding(4) var<storage, read>       mat_d:   array<f32>; // [E, E]
@group(0) @binding(5) var<storage, read>       h_init:  array<f32>; // [H]
@group(0) @binding(6) var<storage, read_write> output:  array<f32>; // [S, E]
@group(0) @binding(7) var<storage, read_write> h_final: array<f32>; // [H]
@group(0) @binding(8) var<uniform>             params:  ScanParams;

// One workgroup runs the full sequence scan for all output elements.
// Workgroup size = 1 because the recurrence is strictly sequential.
// For parallel implementations (flash-linear-attn style), a chunked
// associative-scan approach should be used instead.
@compute @workgroup_size(1)
fn main() {
    let S = params.seq_len;
    let H = params.state_dim;
    let E = params.embed_dim;

    // h = h_init
    var h: array<f32, 256>; // H <= 256 assumed
    for (var i = 0u; i < H; i++) {
        h[i] = h_init[i];
    }

    for (var t = 0u; t < S; t++) {
        // new_h[i] = sum_j A[i,j]*h[j] + sum_k B[i,k]*x[t,k]
        var new_h: array<f32, 256>;
        for (var i = 0u; i < H; i++) {
            var acc: f32 = 0.0;
            for (var j = 0u; j < H; j++) {
                acc += mat_a[i * H + j] * h[j];
            }
            for (var k = 0u; k < E; k++) {
                acc += mat_b[i * E + k] * input[t * E + k];
            }
            new_h[i] = acc;
        }
        for (var i = 0u; i < H; i++) {
            h[i] = new_h[i];
        }

        // y[t,e] = sum_i C[e,i]*h[i] + sum_k D[e,k]*x[t,k]
        for (var e = 0u; e < E; e++) {
            var acc: f32 = 0.0;
            for (var i = 0u; i < H; i++) {
                acc += mat_c[e * H + i] * h[i];
            }
            for (var k = 0u; k < E; k++) {
                acc += mat_d[e * E + k] * input[t * E + k];
            }
            output[t * E + e] = acc;
        }
    }

    // Write final hidden state
    for (var i = 0u; i < H; i++) {
        h_final[i] = h[i];
    }
}
"#;

// ============================================================================
// Causal Masking Kernel
// ============================================================================

/// Causal Mask Attention Scores Kernel
///
/// Applies causal mask to a [B, H, S, S] attention score tensor.
/// Sets all entries where key_idx > query_idx to `mask_value` (typically -1e9).
///
/// Layout: scores[b, h, i, j] = scores[b*H*S*S + h*S*S + i*S + j]
pub const SHADER_CAUSAL_MASK: &str = r#"
struct CausalMaskParams {
    batch_size: u32,
    num_heads:  u32,
    seq_len:    u32,
    mask_value_bits: u32,  // bit-cast of f32 mask_value
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: CausalMaskParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let S   = params.seq_len;
    let H   = params.num_heads;
    let B   = params.batch_size;
    let idx = global_id.x;
    let total = B * H * S * S;
    if (idx >= total) { return; }

    let j   = idx % S;
    let i   = (idx / S) % S;
    // let h = (idx / (S * S)) % H;   // not needed for the mask test
    // let b = idx / (H * S * S);

    if (j > i) {
        // bitcast the stored u32 to f32
        scores[idx] = bitcast<f32>(params.mask_value_bits);
    }
}
"#;

// ============================================================================
// MoH Gate Backward Kernels
// ============================================================================

/// MoH Gate Backward Prepare Kernel
///
/// For each [token, head] element:
///   z = alpha[head] * xw[i,h] + beta[head]
///   g = sigmoid(clamp(z, -8, 8))
///   d_gate[i,h] = eff_grads[i,h] * g * (1 - g)
///   d_gate_scaled[i,h] = d_gate[i,h] * alpha[head]
///
/// Layouts:
///   xw, eff_grads, d_gate, d_gate_scaled: [num_tokens, num_heads]
///   alpha, beta: [num_heads]
pub const SHADER_MOH_GATE_BACKWARD_PREPARE: &str = r#"
struct MohGateBwdPrepParams {
    num_tokens: u32,
    num_heads:  u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> xw: array<f32>;
@group(0) @binding(1) var<storage, read> eff_grads: array<f32>;
@group(0) @binding(2) var<storage, read> alpha: array<f32>;
@group(0) @binding(3) var<storage, read> beta_vals: array<f32>;
@group(0) @binding(4) var<storage, read_write> d_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> d_gate_scaled: array<f32>;
@group(0) @binding(6) var<uniform> params: MohGateBwdPrepParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_tokens * params.num_heads) { return; }

    let h_idx = idx % params.num_heads;
    let z = clamp(alpha[h_idx] * xw[idx] + beta_vals[h_idx], -8.0, 8.0);
    let g = 1.0 / (1.0 + exp(-z));
    let dg = eff_grads[idx] * g * (1.0 - g);
    d_gate[idx] = dg;
    d_gate_scaled[idx] = dg * alpha[h_idx];
}
"#;

/// MoH Gate Backward Reduce Alpha/Beta Kernel
///
/// Computes per-head gradient reductions over all tokens:
///   grad_alpha[h] = sum_i d_gate[i,h] * xw[i,h]
///   grad_beta[h]  = sum_i d_gate[i,h]
///
/// Dispatch: (num_heads, 1, 1).
pub const SHADER_MOH_GATE_BACKWARD_REDUCE: &str = r#"
struct MohGateBwdReduceParams {
    num_tokens: u32,
    num_heads:  u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read> xw: array<f32>;
@group(0) @binding(1) var<storage, read> d_gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_alpha: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_beta: array<f32>;
@group(0) @binding(4) var<uniform> params: MohGateBwdReduceParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.x;
    if (h >= params.num_heads) { return; }

    var acc_alpha: f32 = 0.0;
    var acc_beta:  f32 = 0.0;
    for (var t = 0u; t < params.num_tokens; t++) {
        let idx = t * params.num_heads + h;
        acc_alpha += d_gate[idx] * xw[idx];
        acc_beta  += d_gate[idx];
    }
    grad_alpha[h] += acc_alpha;
    grad_beta[h]  += acc_beta;
}
"#;

// ============================================================================
// Broadcast & Element-wise Kernels
// ============================================================================

/// Broadcast Add Rows Kernel
///
/// Row-wise broadcast bias addition: matrix[row, col] += bias[col]
///
/// Layout: matrix [batch_size, cols], bias [cols]
pub const SHADER_BROADCAST_ADD_ROWS: &str = r#"
struct BroadcastAddParams {
    batch_size: u32,
    cols:       u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<uniform> params: BroadcastAddParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.batch_size * params.cols) { return; }
    let col = idx % params.cols;
    matrix[idx] += bias[col];
}
"#;

/// Signed Log1p Scale Kernel
///
/// In-place: x <- sign(x) * log(1 + alpha * |x|) / alpha
pub const SHADER_SIGNED_LOG1P_SCALE: &str = r#"
struct Log1pParams {
    num_elements: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> buffer: array<f32>;
@group(0) @binding(1) var<uniform> params: Log1pParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) { return; }
    let x = buffer[idx];
    let s = select(-1.0, 1.0, x >= 0.0);
    buffer[idx] = s * log(1.0 + params.alpha * abs(x)) / params.alpha;
}
"#;

// ============================================================================
// TitansMAG Gating Kernel
// ============================================================================

/// MAG Gate Forward Kernel
///
/// Vectorized over tokens for the TitansMAG gating combination:
///   concat = [y_swa, y_mem]               (2*dim)
///   z = concat @ gate_w + gate_b          (dim)
///   g = sigmoid(z)
///   output = g * y_swa + (1-g) * y_mem
///
/// Layouts:
///   y_swa, y_mem, output: [num_tokens, dim]
///   gate_w: [2*dim, dim]
///   gate_b: [dim]
pub const SHADER_MAG_GATE_FORWARD: &str = r#"
struct MagGateParams {
    num_tokens: u32,
    dim:        u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<storage, read>       y_swa:  array<f32>; // [T, D]
@group(0) @binding(1) var<storage, read>       y_mem:  array<f32>; // [T, D]
@group(0) @binding(2) var<storage, read>       gate_w: array<f32>; // [2D, D]
@group(0) @binding(3) var<storage, read>       gate_b: array<f32>; // [D]
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // [T, D]
@group(0) @binding(5) var<uniform>             params: MagGateParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x;
    if (t >= params.num_tokens) { return; }

    let D = params.dim;

    // Compute gate z = [y_swa[t], y_mem[t]] @ gate_w + gate_b  (dim)
    for (var d = 0u; d < D; d++) {
        var z: f32 = gate_b[d];
        for (var k = 0u; k < D; k++) {
            z += y_swa[t * D + k] * gate_w[k * D + d];        // first D rows
            z += y_mem[t * D + k] * gate_w[(D + k) * D + d];  // second D rows
        }
        let g = 1.0 / (1.0 + exp(-z));
        output[t * D + d] = g * y_swa[t * D + d] + (1.0 - g) * y_mem[t * D + d];
    }
}
"#;
