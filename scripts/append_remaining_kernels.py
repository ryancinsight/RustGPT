"""Append remaining GPU kernel WGSL shaders to wgsl_kernels.rs"""

# Check if these already exist
target = 'd:/RustGPT/src/domain/compute/wgsl_kernels.rs'
with open(target, 'r', encoding='utf-8') as f:
    existing = f.read()

shaders = []

# ─── Shader 1: Selective Scan Forward  ────────────────────────────────────────
if 'SHADER_SELECTIVE_SCAN_FORWARD' not in existing:
    shaders.append('''
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
''')

# ─── Shader 2: Causal Mask  ────────────────────────────────────────────────────
if 'SHADER_CAUSAL_MASK' not in existing:
    shaders.append('''
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
''')

# ─── Shader 3: MoH Gate Backward Prepare (sigmoid approx) ────────────────────
if 'SHADER_MOH_GATE_BACKWARD_PREPARE' not in existing:
    shaders.append('''
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
''')

# ─── Shader 4: Broadcast Add Rows  ────────────────────────────────────────────
if 'SHADER_BROADCAST_ADD_ROWS' not in existing:
    shaders.append('''
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
''')

# ─── Shader 5: MAG Gating Forward  ─────────────────────────────────────────────
if 'SHADER_MAG_GATE_FORWARD' not in existing:
    shaders.append('''
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
''')

# Write all new shaders
if shaders:
    with open(target, 'a', encoding='utf-8') as f:
        for shader in shaders:
            f.write(shader)
    print(f'Appended {len(shaders)} shader group(s) totalling {sum(len(s) for s in shaders)} bytes')
else:
    print('All shaders already present, nothing to append')
