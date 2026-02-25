"""Append Titans GPU trait methods and wgpu dispatch to gpu_ops.rs and wgpu_ops.rs"""
import re

# ─── 1. Add trait methods to gpu_ops.rs ─────────────────────────────────────

TRAIT_METHODS = '''
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
'''

ops_path = 'd:/RustGPT/src/domain/compute/gpu_ops.rs'
content = open(ops_path, 'r', encoding='utf-8').read()

# Insert before the first occurrence of "/// Copy within device (GPU-to-GPU)"
marker = '    /// Copy within device (GPU-to-GPU)'
if marker not in content:
    print('ERROR: marker not found in gpu_ops.rs')
    exit(1)

new_content = content.replace(marker, TRAIT_METHODS + '\n' + marker, 1)
open(ops_path, 'w', encoding='utf-8').write(new_content)
print(f'gpu_ops.rs: inserted Titans trait methods ({len(TRAIT_METHODS)} bytes)')
