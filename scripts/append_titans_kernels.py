"""Appends Titans WGSL shaders to wgsl_kernels.rs"""

titans_shaders = '''

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
'''

target = 'd:/RustGPT/src/domain/compute/wgsl_kernels.rs'
with open(target, 'a', encoding='utf-8') as f:
    f.write(titans_shaders)

print(f"Written {len(titans_shaders)} bytes of Titans WGSL shaders to {target}")
