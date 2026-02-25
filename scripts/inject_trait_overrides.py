"""Patch the canonical GpuMatrixOps impl block in wgpu_ops.rs to add
   trait-override methods that delegate to the new dispatch_ methods."""

target = 'd:/RustGPT/src/domain/compute/wgpu_ops.rs'
content = open(target, 'r', encoding='utf-8').read()

# The trait overrides to inject
OVERRIDES = '''
    fn selective_scan_forward(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        a: &GpuBuffer,
        b: &GpuBuffer,
        c: &GpuBuffer,
        d: &GpuBuffer,
        h_init: &GpuBuffer,
        output: &mut GpuBuffer,
        h_final: &mut GpuBuffer,
        seq_len: usize,
        state_dim: usize,
        embed_dim: usize,
    ) -> Result<()> {
        self.dispatch_selective_scan_forward(pool, input, a, b, c, d, h_init, output, h_final, seq_len, state_dim, embed_dim)
    }

    fn causal_mask_attention_scores(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        scores: &mut GpuBuffer,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        mask_value: f32,
    ) -> Result<()> {
        self.dispatch_causal_mask(pool, scores, batch_size, num_heads, seq_len, mask_value)
    }

    fn moh_gate_backward_prepare_sigmoid(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        xw: &GpuBuffer,
        eff_grads: &GpuBuffer,
        alpha: &GpuBuffer,
        beta: &GpuBuffer,
        d_gate: &mut GpuBuffer,
        d_gate_scaled: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> Result<()> {
        self.dispatch_moh_gate_backward_prepare(pool, xw, eff_grads, alpha, beta, d_gate, d_gate_scaled, num_tokens, num_heads)
    }

    fn moh_gate_backward_reduce_alpha_beta(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        xw: &GpuBuffer,
        d_gate: &GpuBuffer,
        grad_alpha: &mut GpuBuffer,
        grad_beta: &mut GpuBuffer,
        num_tokens: usize,
        num_heads: usize,
    ) -> Result<()> {
        self.dispatch_moh_gate_backward_reduce(pool, xw, d_gate, grad_alpha, grad_beta, num_tokens, num_heads)
    }

    fn broadcast_add_rows(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        matrix: &mut GpuBuffer,
        bias: &GpuBuffer,
        batch_size: usize,
        cols: usize,
    ) -> Result<()> {
        self.dispatch_broadcast_add_rows(pool, matrix, bias, batch_size, cols)
    }

    fn titans_mlp_forward(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        keys: &GpuBuffer,
        w1: &GpuBuffer,
        b1: &GpuBuffer,
        w2: &GpuBuffer,
        b2: &GpuBuffer,
        z_out: &mut GpuBuffer,
        h_out: &mut GpuBuffer,
        v_pred: &mut GpuBuffer,
        num_tokens: usize,
        key_dim: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> Result<()> {
        self.dispatch_titans_mlp_forward(pool, keys, w1, b1, w2, b2, z_out, h_out, v_pred, num_tokens, key_dim, hidden_dim, val_dim)
    }

    fn titans_grad_w2(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        v_target: &GpuBuffer,
        v_pred: &GpuBuffer,
        h_act: &GpuBuffer,
        grad_w2: &mut GpuBuffer,
        grad_b2: &mut GpuBuffer,
        num_tokens: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> Result<()> {
        self.dispatch_titans_grad_w2(pool, v_target, v_pred, h_act, grad_w2, grad_b2, num_tokens, hidden_dim, val_dim)
    }

    fn titans_grad_w1(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        keys: &GpuBuffer,
        v_target: &GpuBuffer,
        v_pred: &GpuBuffer,
        z: &GpuBuffer,
        w2: &GpuBuffer,
        grad_w1: &mut GpuBuffer,
        grad_b1: &mut GpuBuffer,
        num_tokens: usize,
        key_dim: usize,
        hidden_dim: usize,
        val_dim: usize,
    ) -> Result<()> {
        self.dispatch_titans_grad_w1(pool, keys, v_target, v_pred, z, w2, grad_w1, grad_b1, num_tokens, key_dim, hidden_dim, val_dim)
    }

    fn titans_memory_update(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        grad: &GpuBuffer,
        momentum: &mut GpuBuffer,
        memory: &mut GpuBuffer,
        num_elements: usize,
        alpha: f32,
        eta: f32,
        theta: f32,
    ) -> Result<()> {
        self.dispatch_titans_memory_update(pool, grad, momentum, memory, num_elements, alpha, eta, theta)
    }
'''

# Find the canonical "impl GpuMatrixOps for WgpuMatrixOps" block — it starts with
# "impl GpuMatrixOps for WgpuMatrixOps {" and contains "fn gemm_f32"
# We inject our overrides right before the closing brace of that block.
# Strategy: find the FIRST occurrence, find its end "}" following gemm_f32, insert before it.

import re

# Find the main impl block (the one that has 'fn gemm_f32')
# We look for "impl GpuMatrixOps for WgpuMatrixOps {" followed by "fn gemm_f32"
pattern = r'(impl GpuMatrixOps for WgpuMatrixOps \{[^}]*?)(fn gemm_f32)'
m = re.search(pattern, content, re.DOTALL)
if not m:
    print('Could not find canonical impl GpuMatrixOps for WgpuMatrixOps block')
    exit(1)

# Find position of "fn gemm_f32" in the main impl block
gemm_pos = content.find('fn gemm_f32')
# Find the closing brace of the impl block after gemm_f32
# We'll search for the right impl block — the one containing gemm_f32
# Walk backwards from gemm_pos to find the impl header
impl_start = content.rfind('impl GpuMatrixOps for WgpuMatrixOps {', 0, gemm_pos)
if impl_start == -1:
    print('Could not find canonical impl block header')
    exit(1)

print(f'Found canonical impl block at char {impl_start}')

# Walk forward from impl_start to find the matching closing brace
depth = 0
i = impl_start
while i < len(content):
    if content[i] == '{':
        depth += 1
    elif content[i] == '}':
        depth -= 1
        if depth == 0:
            impl_end = i  # position of the final '}'
            break
    i += 1
else:
    print('Could not find end of impl block')
    exit(1)

print(f'impl block closes at char {impl_end}')

# Check if our overrides are already inserted
if 'dispatch_selective_scan_forward' in content[impl_start:impl_end]:
    print('Overrides already present in the impl block, skipping')
    exit(0)

# Insert overrides before the closing brace
new_content = content[:impl_end] + OVERRIDES + '\n' + content[impl_end:]
open(target, 'w', encoding='utf-8').write(new_content)
print(f'Inserted {len(OVERRIDES)} bytes of trait overrides into the canonical impl block')
