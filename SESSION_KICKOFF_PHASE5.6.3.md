# Phase 5.6.3 GPU Backend Consolidation - Session Kickoff

**Thread**: @T-019c64bf-ef38-742c-8f28-b9d3459e97d9  
**Date**: Feb 16, 2026  
**Objective**: Begin GPU backend consolidation with automatic detection and no CPU fallback  
**Status**: 🟢 READY TO BEGIN

## Executive Summary

We're proceeding with Phase 5.6.3 to consolidate and optimize shared GPU components across Diffusion, SSM, and Transformer models. The strategy is:

1. **Automatic GPU Detection**: CUDA > Metal > WGPU (strict, no CPU fallback)
2. **Two-Pass RichardsGLU Kernel**: Reduce kernel launches from 5+ to 2 (critical performance win)
3. **Backend Variants**: Implement WGPU (immediate), CUDA stubs (ready for `.cu`), Metal stubs (ready for `.metal`)
4. **Zero-Copy Memory**: Keep data on GPU throughout computation

## What's Already Done

✅ GPU device auto-detection with strict priority order  
✅ Unified `GpuMatrixOps` trait for all backends  
✅ WGPU WGSL shaders for ~20 core operations  
✅ CUDA/Metal stubs with informative error messages  
✅ Unified GPU kernels dispatcher (`UnifiedGpuKernels`)  
✅ GPU memory pool abstraction  

## What Needs to Be Done (Priority Order)

### Phase 5.6.3a: WGPU Kernel Completion (Feb 17)

**Critical Path**: RichardsGLU two-pass kernel

1. **Verify WGPU Implementation** (30 min)
   - Compile and run GPU kernel unit tests
   - Check shader compilation errors
   - Validate memory pool operations

2. **Implement RichardsGLU Two-Pass Kernel** (2 hours)
   - Location: `src/domain/compute/wgpu_ops.rs`
   - Design: Two-pass kernel structure
     - Pass 1: Input projection + gating (element-wise ops)
     - Pass 2: Output projection (GEMM)
   - Key constraint: Zero-copy (no GPU-CPU transfers)
   - Result: Reduce from 5+ kernel launches to 2

3. **Test & Verify** (1 hour)
   - Unit test for RichardsGLU correctness
   - Benchmark to confirm dispatch reduction
   - Memory profiling to verify zero-copy

4. **PolyAttention Kernels (If Needed)** (2 hours)
   - `poly_attention_fused` - Content + positional scoring
   - `blr_projection` - Low-rank projection
   - `compute_cope_scores` - COPE positional attention
   - Only if using PolyAttention layers

### Phase 5.6.3b: CUDA Kernel Templates (Feb 18)

Create skeleton `.cu` files ready for actual kernel implementation.

1. **Create CUDA Kernel Directory Structure**
   ```
   src/domain/compute/cuda/kernels/
     ├── element_wise.cu          # All element-wise ops
     ├── richards_curve.cu        # Stable Richards computation
     ├── richards_glu_fused.cu    # Two-pass GLU
     ├── activation.cu            # Softmax, LayerNorm
     └── Makefile.cu              # Build template
   ```

2. **Template Content** (each file ~50 lines)
   - Kernel function signatures
   - Block/thread dimension hints
   - TODO comments for implementation
   - Example element-wise op (e.g., ReLU)

3. **Rust Wrapper Updates**
   - Update `src/domain/compute/cuda/ops.rs` to reference kernel names
   - Keep error messages pointing to `.cu` files

### Phase 5.6.3c: Comprehensive GPU Tests (Feb 19)

Ensure all backends (WGPU, CUDA stubs, Metal stubs) work together.

1. **GPU Integration Test Suite**
   ```bash
   cargo test --test gpu_shared_components_phase56 \
     --features gpu-all -- --nocapture
   ```

2. **Test Coverage** (per kernel):
   - Correctness vs CPU reference
   - Numerical stability (especially Richards curve)
   - Memory bounds (no out-of-bounds access)
   - Zero-copy verification

## Session Execution Plan

### Hour 1: Verification & Planning (0:00 - 1:00)

```bash
# 1. Verify builds
cargo check --lib
cargo build --release --features wgpu

# 2. Run existing GPU tests (identify failures)
cargo test --lib gpu_device --features wgpu

# 3. Check which kernels are still stubbed
grep -n "Err.*Backend.*not implemented" \
  src/domain/compute/cuda/ops.rs | wc -l
```

**Deliverable**: List of missing kernels, build status report

### Hour 2-3: RichardsGLU Kernel Implementation (1:00 - 3:30)

```bash
# 1. Open unified_gpu_kernels.rs
# 2. Locate rg_lru_recurrent() function
# 3. Implement two-pass fused kernel:
#    - Pass 1: Input @ W_g1 + Input @ W_g2, element-wise gating
#    - Pass 2: Gated @ W_out (reuse existing GEMM)
# 4. Test with simple [32, 512] -> [32, 1024] -> [32, 512] shapes
```

**Deliverable**: Working RichardsGLU two-pass kernel in WGPU

### Hour 4: Testing & Benchmarking (3:30 - 4:30)

```bash
# 1. Create simple benchmark
cargo bench --bench gpu_performance \
  --features wgpu -- richards_glu

# 2. Verify dispatch reduction (2 launches vs 5+)
# 3. Check memory usage (power-of-2 aligned workspace)
# 4. Run correctness test vs CPU reference
```

**Deliverable**: Benchmark showing 2-pass dispatch, memory stats

### Hour 5: CUDA Template Creation (4:30 - 5:30)

```bash
# 1. Create cuda/kernels/ directory structure
# 2. Write kernel templates with TODO comments
# 3. Update cuda/ops.rs error messages to reference .cu files
# 4. Verify builds without CUDA toolkit (graceful errors)
```

**Deliverable**: CUDA kernel template files ready for implementation

## Key Decision Points

### 1. Auto-Detection Strictness
**Decision**: NO CPU fallback - GPU operations fail explicitly if backend is unavailable
- Rationale: Predictable performance, forces developers to handle missing implementations
- Implementation: `GpuDevice::auto_detect()` returns error if no GPU available

### 2. RichardsGLU Fusion Strategy
**Decision**: Two-pass kernel (element-wise ops + GEMM)
- Rationale: Simpler than single monolithic kernel, reuses existing GEMM
- Performance: Reduces launches from 5+ to 2 (target: 30% reduction in dispatch overhead)

### 3. Backend Priority Order
**Decision**: CUDA > Metal > WGPU (Vulkan/DX12)
- Rationale: CUDA for maximum performance on NVIDIA, Metal for macOS, WGPU for cross-platform
- Implementation: `GpuDevice::auto_detect()` checks in this order

## Testing Strategy

### Unit Tests (Immediate)
```bash
cargo test --lib gpu_ops --features wgpu
cargo test --lib gpu_device
```

### Integration Tests (After Implementation)
```bash
cargo test --test gpu_shared_components_phase56 \
  --features gpu-all

# Should test:
# - WGPU kernels (working)
# - CUDA stubs (error messages clear)
# - Metal stubs (error messages clear)
# - Auto-detection (correct backend selected)
```

### Benchmarks (Validation)
```bash
cargo bench --bench gpu_performance --features wgpu
# Target: RichardsGLU <1ms for [32, 512] -> [32, 1024]
```

## Known Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| WGSL shader compilation errors | Medium | Blocks kernel | Test early, reference WGSL docs |
| CUDA missing on dev machine | Medium | Can't test CUDA impl | Use stubs, defer `.cu` implementation |
| Zero-copy verification difficult | Low | Missed optimization | Add memory transfer profiling |
| PolyAttention kernels complex | Medium | Extends timeline | Implement only if needed for models |

## Success Criteria

- [x] GPU auto-detection working (no CPU fallback)
- [ ] WGPU builds without warnings
- [ ] RichardsGLU two-pass kernel implemented and tested
- [ ] Kernel dispatch reduced to 2 launches (verified via profiling)
- [ ] CUDA kernel templates created
- [ ] All GPU tests pass
- [ ] No GPU-CPU transfers in RichardsGLU path
- [ ] Memory pool tests pass

## Files Modified This Session

```
Core Implementation:
  src/domain/compute/wgpu_ops.rs      (add RichardsGLU kernel)
  src/domain/compute/gpu_ops.rs       (may add new trait methods)

New Files:
  src/domain/compute/cuda/kernels/element_wise.cu
  src/domain/compute/cuda/kernels/richards_curve.cu
  src/domain/compute/cuda/kernels/richards_glu_fused.cu

Documentation:
  PHASE5.6.3_IMPLEMENTATION_PLAN.md
  PHASE5.6.3_GPU_KERNEL_STATUS.md
  PHASE5.6.3_QUICKSTART.md
  SESSION_KICKOFF_PHASE5.6.3.md (this file)
```

## References

- **GPU Consolidation Plan**: @T-019c64bf-ef38-742c-8f28-b9d3459e97d9
- **Two-Pass Kernel Design**: PHASE5.6_RICHARDSGL U_FUSED_KERNEL_GUIDE.md
- **GPU Device**: src/domain/compute/gpu_device.rs
- **WGSL Docs**: https://gpuweb.github.io/gpuweb/wgsl/

---

## Next Steps

1. Verify current build status
2. Start RichardsGLU kernel implementation
3. Test with simple shapes first
4. Create CUDA templates
5. Run comprehensive tests

**Estimated completion**: ~5 hours for core Phase 5.6.3a (WGPU RichardsGLU)  
**Follow-up**: CUDA/Metal implementation in next session
