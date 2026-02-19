# GPU Backend Implementation for Richards GLU (Phase 5.6.2)

## Status

**Phase**: 5.6.2 - GPU Backend Consolidation
**Date**: February 15, 2026
**Priority**: P1 (Critical path)

## Completed Actions

### 1. Backend Structure Setup ✅
- Unified trait `GpuMatrixOps` with consistent signatures across backends
- All backends now have explicit `pool: &mut dyn GpuMemoryPool` parameter
- CUDA, Metal, and WGPU all implement the trait

### 2. Method Signatures Aligned ✅
- Updated CUDA ops (`src/domain/compute/cuda/ops.rs`)
  - Added `richards_curve` method
  - Added `moh_gate_activation` method
  - Fixed parameter signatures
  
- Updated Metal ops (`src/domain/compute/metal/ops.rs`)
  - Added missing `pool` parameters to all methods
  - Added `richards_curve` method
  - Added `moh_gate_activation` method
  - Fixed `gemm_f32`, `gemv_f32` signatures

### 3. GPU Forward Pass Dispatch ✅
- File: `src/domain/compute/richards_glu_fused_kernel.rs`
- Implements two-pass fused kernel strategy:
  - **Pass 1**: GEMM operations (x1, x2) + Richards activation
  - **Pass 2**: Output projection (GEMM)
- Uses `device.richards_curve()` for GPU-accelerated activation

## Current Implementation Status

### WGPU (Primary Implementation) ✅
- **Status**: FULLY IMPLEMENTED
- **Location**: `src/domain/compute/wgpu_ops.rs`
- **Kernels Implemented**:
  - SHADER_GEMM (line 34-113): Tiled matrix multiplication
  - SHADER_RICHARDS_CURVE (line 451-513): Richards activation
  - SHADER_SOFTMAX: Numerically stable softmax
  - SHADER_RELU, SHADER_GELU, SHADER_SILU: Element-wise activations

### CUDA (Placeholder) ⚠️
- **Status**: NOT IMPLEMENTED (returns errors)
- **Why**: Requires native CUDA kernels (.cu files) and cuBLAS integration
- **Current Behavior**: Returns informative error message directing users to WGPU
- **Future**: Can be implemented with CUDA kernel files + cudarc bindings

### Metal (Placeholder) ⚠️
- **Status**: NOT IMPLEMENTED (returns errors)
- **Why**: Requires Metal Performance Shaders (MPS) kernels
- **Current Behavior**: Returns informative error message directing users to WGPU
- **Future**: Can be implemented with Metal compute kernels

## GPU Detection Strategy

### Automatic Backend Selection (Phase 5.6)
Priority order in `GpuDevice::auto_detect()`:
1. **CUDA** (if available on NVIDIA systems)
2. **Metal** (if on macOS)
3. **Vulkan/WGPU** (fallback, cross-platform)

### Strict No-Fallback Mode
- **Philosophy**: Fail fast with clear errors instead of silent CPU fallback
- **Behavior**: If GPU backend is selected but operation not implemented, error is returned
- **Rationale**: Ensures developers catch missing GPU implementations early

## Richards GLU Forward Pass Flow

```
Input (batch_size, input_dim) on GPU
         ↓
   [PASS 1: Activation]
   ├─ x1 = input @ w1 (GEMM)
   ├─ x2 = input @ w2 (GEMM)
   ├─ value = x1 * richards(x1) (Richards activation + multiply)
   ├─ gate = richards(x2) (Richards activation)
   └─ gated = value * gate (element-wise multiply)
         ↓
   [PASS 2: Projection]
   └─ output = gated @ w_out (GEMM)
         ↓
Output (batch_size, output_dim) on GPU
```

## Test Coverage

### Existing Tests ✅
- `test_richards_activation_bounds`: Validates activation curve
- `test_reference_forward_shapes`: CPU reference implementation
- `test_gpu_forward_dispatch`: GPU forward pass (skips if no GPU)

### Run Tests
```bash
# All Richards tests
cargo test --lib richards_glu 2>&1

# GPU-specific tests
cargo test --lib gpu 2>&1

# Full test suite
cargo test --lib 2>&1
```

## Implementation Roadmap

### Phase 5.6.2 (Current) - GPU Backend Consolidation
- [x] Align method signatures across all backends
- [x] Add missing GPU methods (richards_curve, moh_gate_activation)
- [x] Implement stub methods with informative errors
- [ ] Fix GPU detection automatic fallback logic (WGPU only on non-Linux)
- [ ] Document GPU feature flags

### Phase 5.6.3 - GPU Kernel Implementation (Future)
- [ ] Implement actual CUDA kernels (.cu files)
- [ ] Implement actual Metal kernels (.metal files)
- [ ] Extend cuBLAS integration for GEMM
- [ ] Add kernel performance benchmarks

### Phase 5.6.4 - Performance Optimization (Future)
- [ ] Fused kernel optimization (combine multiple ops)
- [ ] Shared memory optimizations (CUDA)
- [ ] Tensor core utilization (NVIDIA A100+)
- [ ] Profile vs CPU reference

## Compilation Flags

### Default (WGPU only)
```bash
cargo build --lib
```

### With CUDA Support
```bash
cargo build --lib --features gpu-cuda
```

### With Metal Support (macOS only)
```bash
cargo build --lib --features gpu-metal
```

### With All GPU Backends
```bash
cargo build --lib --features gpu-all
```

## Error Messages

When running on non-NVIDIA/non-macOS systems:
```
Error: GPU backend not available for this platform.
Use --features gpu-wgpu for cross-platform GPU support via Vulkan.
```

When GPU backend is selected but operation not implemented:
```
Error: CUDA richards_curve not yet implemented for size 24576.
Use WGPU backend or compile with native CUDA kernels.
```

## Key Files Modified

1. **src/domain/compute/cuda/ops.rs**
   - Added `richards_curve` method (lines 362-380)
   - Added `moh_gate_activation` method (lines 382-398)

2. **src/domain/compute/metal/ops.rs**
   - Fixed all method signatures to include `pool: &mut dyn GpuMemoryPool`
   - Added `richards_curve` method (lines 220-238)
   - Added `moh_gate_activation` method (lines 240-258)

3. **src/domain/compute/richards_glu_fused_kernel.rs**
   - Uses `device.richards_curve()` for GPU activation (line 254)
   - Two-pass fused kernel strategy (lines 268-354)

## Next Steps

1. **Immediate** (This session)
   - Verify all GPU tests pass ✅
   - Document GPU backend behavior
   - Clean up warnings

2. **Short-term** (Next session)
   - Implement GPU detection fallback (WGPU when CUDA/Metal unavailable)
   - Add GPU feature flag documentation
   - Create integration tests with actual GPU execution

3. **Medium-term** (Phase 5.6.3)
   - Implement native CUDA/Metal kernels
   - Optimize for mixed precision (float16)
   - Benchmark against CPU reference

## Known Limitations

1. **CUDA**: Requires .cu kernel files + cudarc bindings (not in current scope)
2. **Metal**: Requires .metal kernel files + metal-rs bindings (not in current scope)
3. **WGPU**: Cross-platform but may be slower than native implementations
4. **No Fallback**: Unlike PyTorch, this implementation fails fast instead of falling back to CPU

## References

- Implementation: `src/domain/compute/richards_glu_fused_kernel.rs`
- GPU Ops trait: `src/domain/compute/gpu_ops.rs`
- GPU Device: `src/domain/compute/gpu_device.rs`
- WGPU Shaders: `src/domain/compute/wgpu_ops.rs` (lines 451-513)
- Richards Curve: `src/domain/richards/richards_curve.rs`
