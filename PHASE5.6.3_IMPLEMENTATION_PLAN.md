# Phase 5.6.3: GPU Backend Consolidation & Implementation Plan

**Date**: Feb 16, 2026  
**Priority**: Implement GPU kernels with automatic GPU detection, NO fallback to CPU  
**Status**: KICKOFF

## Objectives

1. **Consolidate** shared components (Diffusion, SSM, Transformer)
2. **Implement GPU kernels** with dual-pass approach for RichardsGLU
3. **Automatic GPU detection** with strict no-fallback semantics
4. **Optimize memory & performance** of shared temporal operations

## Architecture Overview

```
GPU Detection (auto_detect)
  ├─ CUDA (cudarc + cuBLAS)
  ├─ Metal (Metal Performance Shaders)
  └─ WGPU (Vulkan/Metal/DX12)

Unified Trait: GpuMatrixOps
  ├─ BLAS Operations (GEMM, GEMV)
  ├─ Element-wise (ReLU, GELU, SiLU, Richards Curve)
  ├─ Normalization (LayerNorm, Softmax)
  └─ PolyAttention Kernels (MOH Gate, COPE, BLR Projection)

Shared Components (Target: Diffusion, SSM, Transformer)
  ├─ RichardsGLU Fused Kernel (Two-Pass)
  ├─ TemporalMixing Variants
  └─ Unified Buffer Pool
```

## Phase 5.6.3 Implementation Tasks

### Priority 1: WGPU Backend Completion (Due: Feb 17)

**Status**: Shaders partially implemented, need verification

1. **Verify existing WGSL kernels**:
   - GEMM (tiled matrix multiplication)
   - Softmax (stable log-space)
   - Element-wise ops (ReLU, GELU, SiLU)
   - Layer norm

2. **Implement missing WGSL kernels**:
   - ✅ richards_curve (stable exponent computation)
   - ✅ moh_gate_activation (per-head gating)
   - ❌ richards_glu_fused (two-pass kernel)
   - ❌ poly_attention_fused (content + positional scoring)
   - ❌ blr_projection (low-rank projection)
   - ❌ cope_scores (COPE attention)

3. **Test & benchmark WGPU kernels**:
   ```bash
   cargo test --test gpu_shared_components_phase56 --features wgpu
   cargo bench --bench [bench_name] --features wgpu
   ```

### Priority 2: CUDA Backend Implementation (Due: Feb 18)

**Status**: Stubs only, ready for `.cu` kernels

1. **Create CUDA kernel implementations**:
   - `kernels/gemm.cu` - cuBLAS wrapper
   - `kernels/element_wise.cu` - ReLU, GELU, SiLU
   - `kernels/richards_curve.cu` - Stable Richards curve
   - `kernels/richards_glu_fused.cu` - Two-pass GLU kernel
   - `kernels/attention.cu` - Softmax, QKV projections

2. **Integrate cudarc bindings**:
   ```rust
   pub struct CudaMatrixOps {
       device: Arc<cudarc::driver::CudaDevice>,
       kernels: HashMap<String, cudarc::driver::CudaFunction>,
   }
   ```

### Priority 3: Metal Backend Implementation (Due: Feb 19)

**Status**: Stubs only, ready for `.metal` kernels

1. **Create Metal shader implementations** (`.metal` files)
2. **Integrate Metal Performance Shaders** (MPS)

## Two-Pass RichardsGLU Kernel Design

### Mathematical Formulation
```
Pass 1: Activation
  input:  [batch_size, input_dim]
  w_g1:   [input_dim, hidden_dim]
  w_g2:   [input_dim, hidden_dim]
  
  value = input @ w_g1          # [batch_size, hidden_dim]
  gate_logits = input @ w_g2    # [batch_size, hidden_dim]
  gated = gate_logits * value   # [batch_size, hidden_dim]
  
  output_p1: [batch_size, hidden_dim]

Pass 2: Projection
  gated:  [batch_size, hidden_dim]
  w_out:  [hidden_dim, output_dim]
  
  output = gated @ w_out        # [batch_size, output_dim]
  
  output_p2: [batch_size, output_dim]
```

### Zero-Copy Constraint
- Data remains on GPU throughout both passes
- No intermediate CPU transfers
- Buffers allocated once, reused across passes

### Dispatch Strategy
```
Pass 1: 256-thread workgroups for element-wise ops
  - Dispatch: (batch_size * hidden_dim + 255) / 256
  
Pass 2: GEMM kernel (existing tiled implementation)
  - Dispatch: (batch_size, output_dim) with TILE_SIZE=16
```

## GPU Detection & Fallback Semantics

### Priority Order (Strict)
```rust
GpuDevice::auto_detect() {
    if CUDA_available() && cudarc_linked() {
        return CUDA
    }
    if Metal_available() && macOS() {
        return Metal
    }
    if Vulkan_available() || DX12_available() {
        return WGPU(Vulkan/DX12)
    }
    return Err(ModelError::Backend { ... })  // NO CPU fallback
}
```

### Error Handling (Strict Mode)
- If a GPU operation is unimplemented for the detected backend → fail immediately
- No silent CPU fallback
- Developers must implement the kernel for the target backend

## Testing Strategy

### Unit Tests
```bash
cargo test --lib gpu_ops
cargo test --lib unified_gpu_kernels
cargo test --lib gpu_device
```

### Integration Tests (Phase 5.6)
```bash
cargo test --test gpu_shared_components_phase56 --features gpu-all
```

### Benchmarks
```bash
cargo bench --bench gpu_performance --features gpu-cuda
cargo bench --bench gpu_performance --features wgpu
```

## Success Criteria

- [ ] WGPU kernels: All core operations (GEMM, activation, normalization) pass tests
- [ ] CUDA stubs: Kernels compile with error messages pointing to `.cu` implementation
- [ ] Metal stubs: Kernels compile with error messages pointing to `.metal` implementation
- [ ] Automatic detection: Prioritizes CUDA > Metal > WGPU with NO CPU fallback
- [ ] RichardsGLU two-pass: Reduces kernel launches from 5+ to 2 (verified via profiling)
- [ ] Zero-copy: No intermediate GPU-CPU transfers in benchmark traces
- [ ] Memory efficiency: Power-of-2 aligned workspace buffers (verified via heap dump)

## Files to Modify/Create

### Existing (Verify/Update)
- `src/domain/compute/gpu_ops.rs` - Trait definitions (✅ done)
- `src/domain/compute/wgpu_ops.rs` - WGSL shaders (partial)
- `src/domain/compute/gpu_device.rs` - Auto-detection (✅ done)
- `src/domain/compute/unified_gpu_buffer_pool.rs` - Memory management (✅ done)

### New (Implement)
- `src/domain/compute/cuda/kernels/*.cu` - CUDA kernel implementations
- `src/domain/compute/metal/kernels/*.metal` - Metal shader implementations
- `src/domain/compute/wgpu/shaders/*.wgsl` - Additional WGSL kernels (two-pass GLU, etc.)

### Test Files
- `tests/gpu_shared_components_phase56.rs` - Comprehensive GPU integration tests

## Dependencies

- **cudarc**: CUDA runtime and kernel binding
- **wgpu**: Cross-platform GPU compute
- **metal**: macOS Metal API (optional)

## Next Session Handoff

1. Run comprehensive GPU tests with `--features gpu-all`
2. Profile kernel dispatch overhead (target: 2 launches for RichardsGLU)
3. Validate zero-copy memory transfers
4. Prepare CUDA `.cu` kernel templates

## References

- **Thread**: @T-019c64bf-ef38-742c-8f28-b9d3459e97d9 (Consolidation plan)
- **Two-Pass Kernel Strategy**: PHASE5.6_RICHARDSGL U_FUSED_KERNEL_GUIDE.md
- **GPU Detection**: src/domain/compute/gpu_device.rs#L50-L120
