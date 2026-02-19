# Phase 5.6.3: GPU Backend Consolidation - Documentation Index

**Date**: Feb 16, 2026  
**Objective**: Consolidate GPU backends with automatic detection, implement fused kernels, optimize performance  
**Session Status**: 🟢 Ready to begin

## Quick Navigation

### For Quick Start
- **[PHASE5.6.3_QUICKSTART.md](PHASE5.6.3_QUICKSTART.md)** ← Start here
  - Build commands
  - Implementation checklist
  - Debugging tips

### For Session Execution
- **[SESSION_KICKOFF_PHASE5.6.3.md](SESSION_KICKOFF_PHASE5.6.3.md)** ← Read this for context
  - Executive summary
  - Hour-by-hour execution plan
  - Success criteria

### For Implementation Details
- **[PHASE5.6.3_RICHARDSON_GLU_IMPLEMENTATION_GUIDE.md](PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md)** ← Use during coding
  - Step-by-step implementation
  - Code snippets
  - Testing strategy
  - CUDA/Metal templates

### For Reference
- **[PHASE5.6.3_IMPLEMENTATION_PLAN.md](PHASE5.6.3_IMPLEMENTATION_PLAN.md)** ← Background & architecture
  - Three-phase plan (WGPU, CUDA, Metal)
  - Technical specifications
  - Success metrics

- **[PHASE5.6.3_GPU_KERNEL_STATUS.md](PHASE5.6.3_GPU_KERNEL_STATUS.md)** ← Kernel inventory
  - What's implemented
  - What's stubbed
  - Priority matrix

## Document Breakdown

### Setup & Planning

| Document | Purpose | Length | Priority |
|----------|---------|--------|----------|
| PHASE5.6.3_QUICKSTART.md | Build commands, quick reference | 150 lines | 🔴 HIGH |
| SESSION_KICKOFF_PHASE5.6.3.md | Executive summary, execution plan | 280 lines | 🔴 HIGH |
| PHASE5.6.3_IMPLEMENTATION_PLAN.md | Detailed roadmap, architecture | 200 lines | 🟡 MEDIUM |

### Implementation Guides

| Document | Purpose | Length | Priority |
|----------|---------|--------|----------|
| PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md | Step-by-step RichardsGLU kernel | 450 lines | 🔴 HIGH |
| PHASE5.6.3_GPU_KERNEL_STATUS.md | Kernel inventory & priorities | 250 lines | 🟡 MEDIUM |

### Architecture & References

| Document | Purpose | Lines |
|----------|---------|-------|
| PHASE5.6.3_DOCUMENTATION_INDEX.md | This file | Navigation |

## Key Files to Modify

### Priority 1: WGPU Backend (Feb 17)

```
src/domain/compute/wgpu_ops.rs
  └─ Add richards_glu_fused() kernel
     • Uses existing GEMM, element-wise ops
     • Two-pass structure: activation + projection
     • Target: 2 kernel launches

src/domain/compute/gpu_ops.rs
  └─ Update GpuMatrixOps trait
     • Add richards_glu_fused() method signature
     • Add moh_gate_activation() if missing
```

### Priority 2: CUDA Stubs (Feb 18)

```
src/domain/compute/cuda/ops.rs
  └─ Add error messages pointing to .cu files

New Files:
src/domain/compute/cuda/kernels/
  ├── element_wise.cu          # All element-wise ops
  ├── richards_curve.cu        # Stable Richards computation
  ├── richards_glu_fused.cu    # Two-pass GLU
  └── Makefile.cu              # Build template
```

### Priority 3: Metal Stubs (Feb 19)

```
src/domain/compute/metal/ops.rs
  └─ Add error messages pointing to .metal files

New Files:
src/domain/compute/metal/kernels/
  ├── element_wise.metal       # All element-wise ops
  ├── richards_curve.metal     # Richards computation
  └── attention.metal          # Normalization kernels
```

## Implementation Checklist

### Phase 5.6.3a: WGPU Completion (Feb 17)

- [ ] Verify WGPU builds (`cargo check --features wgpu`)
- [ ] Run existing GPU tests (`cargo test --lib gpu_ops --features wgpu`)
- [ ] Implement RichardsGLU two-pass kernel
  - [ ] Add method to `GpuMatrixOps` trait
  - [ ] Implement in `WgpuMatrixOps`
  - [ ] Add unit test with correctness check
  - [ ] Add benchmark with dispatch verification
- [ ] Verify kernel dispatch reduced from 5+ to 2 launches
- [ ] Zero-copy memory transfer verification
- [ ] Run integration tests (`cargo test --test gpu_shared_components_phase56 --features wgpu`)

### Phase 5.6.3b: CUDA Templates (Feb 18)

- [ ] Create `src/domain/compute/cuda/kernels/` directory
- [ ] Create kernel template files (`.cu` skeleton)
  - [ ] element_wise.cu
  - [ ] richards_curve.cu
  - [ ] richards_glu_fused.cu
  - [ ] activation.cu
- [ ] Update `cuda/ops.rs` error messages
- [ ] Verify builds with CUDA stubs
- [ ] Verify error messages are informative

### Phase 5.6.3c: Comprehensive Testing (Feb 19)

- [ ] Run all GPU tests with `--features gpu-all`
- [ ] Verify auto-detection priority (CUDA > Metal > WGPU)
- [ ] Verify no CPU fallback
- [ ] GPU memory pool tests pass
- [ ] Benchmark GPU vs CPU performance
- [ ] Document performance improvements

## Testing Strategy

### Unit Tests
```bash
cargo test --lib gpu_ops --features wgpu
cargo test --lib gpu_device
cargo test --lib unified_gpu_kernels --features wgpu
```

### Integration Tests
```bash
cargo test --test gpu_shared_components_phase56 --features gpu-all
cargo test --test gpu_backend_detection
```

### Benchmarks
```bash
cargo bench --bench gpu_performance --features wgpu
# Focus on: RichardsGLU, GEMM, softmax, element-wise
```

## GPU Architecture Diagram

```
Application Layer
    ↓
UnifiedGpuKernels (unified_gpu_kernels.rs)
    ↓
GpuMatrixOps Trait (gpu_ops.rs)
    ├─ WgpuMatrixOps (wgpu_ops.rs) [WGSL Shaders] ✅
    ├─ CudaMatrixOps (cuda/ops.rs) [cuBLAS + .cu] ❌ Stubs
    └─ MetalMatrixOps (metal/ops.rs) [MPS + .metal] ❌ Stubs
    ↓
GpuDevice::auto_detect()  (Strict priority: CUDA > Metal > WGPU)
    ↓
GpuMemoryPool (Memory management, zero-copy)
    ↓
Target Models: Diffusion, SSM, Transformer
```

## Key Concepts

### Auto-Detection (Strict, No Fallback)
```
GpuDevice::auto_detect() {
    if CUDA_available { return CUDA }
    if Metal_available { return Metal }
    if WGPU_available { return WGPU }
    return Err("No GPU backend available")  // NO CPU fallback
}
```

### Two-Pass RichardsGLU Kernel
```
Pass 1: Element-wise activation + gating
  [batch, input] @ W_g1 → [batch, hidden] (value)
  [batch, input] @ W_g2 → [batch, hidden] (gate_logits)
  Richards(gate_logits) * value → [batch, hidden] (gated)

Pass 2: Output projection (GEMM)
  [batch, hidden] @ W_out → [batch, output]

Result: 2 kernel launches instead of 5+
```

### Zero-Copy Memory
- Input uploaded at layer start
- All intermediates on GPU
- Output downloaded at layer end
- No GPU-CPU transfers during computation

## Performance Targets

| Metric | Target | Component |
|--------|--------|-----------|
| Kernel launch reduction | 30% (5+ → 2) | RichardsGLU |
| GEMM speedup | 10x+ vs CPU | WGPU/CUDA |
| Softmax speedup | 5x+ vs CPU | WGPU/CUDA |
| Memory bandwidth | >100 GB/s | GPU |
| Numerical error | <1e-4 | All kernels |

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| WGSL shader errors | Medium | Blocks kernel | Test early, WGSL docs |
| CUDA missing | Medium | Can't test CUDA | Use stubs, defer `.cu` |
| Complex fused kernel | Low | Schedule slip | Start with simple fusion |
| PolyAttention complexity | Medium | Extends timeline | Implement only if needed |

## Success Criteria

- ✅ Auto-detection working (strict, no CPU fallback)
- ✅ WGPU kernels verified (build + tests)
- [ ] RichardsGLU two-pass implemented and tested
- [ ] Kernel dispatch reduced to 2 (verified via profiling)
- [ ] CUDA templates created with clear TODOs
- [ ] All GPU tests pass
- [ ] Zero-copy verified (no GPU-CPU transfers in middle)
- [ ] Memory pool passes stress tests

## Session Timeline

```
Hour 1:   Verification & Planning
Hour 2-3: RichardsGLU Implementation (WGPU)
Hour 4:   Testing & Benchmarking
Hour 5:   CUDA Templates
Hour 6+:  PolyAttention Kernels (if needed)
```

## References

### Code Locations
- **GPU Device**: `src/domain/compute/gpu_device.rs`
- **GPU Ops Trait**: `src/domain/compute/gpu_ops.rs`
- **WGPU Implementation**: `src/domain/compute/wgpu_ops.rs`
- **CUDA Stubs**: `src/domain/compute/cuda/ops.rs`
- **Metal Stubs**: `src/domain/compute/metal/ops.rs`
- **Unified Dispatcher**: `src/domain/layers/components/unified_gpu_kernels.rs`
- **Memory Pool**: `src/domain/compute/gpu_memory.rs`

### External Documentation
- **WGSL Spec**: https://gpuweb.github.io/gpuweb/wgsl/
- **cudarc**: https://docs.rs/cudarc/latest/cudarc/
- **cuBLAS**: https://docs.nvidia.com/cuda/cublas/
- **Metal Shading Language**: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- **wgpu**: https://docs.rs/wgpu/latest/wgpu/

### Previous Sessions
- **Consolidation Plan**: @T-019c64bf-ef38-742c-8f28-b9d3459e97d9
- **Phase 5.6 GPU Kernels**: Previous session documentation

## Next Steps

1. **Read PHASE5.6.3_QUICKSTART.md** for immediate build commands
2. **Follow SESSION_KICKOFF_PHASE5.6.3.md** for execution plan
3. **Use PHASE5.6.3_RICHARDS_GLU_IMPLEMENTATION_GUIDE.md** during coding
4. **Reference PHASE5.6.3_GPU_KERNEL_STATUS.md** for kernel inventory

---

**Start with Quick Start guide → Execute session plan → Implement RichardsGLU → Create CUDA templates → Test**
