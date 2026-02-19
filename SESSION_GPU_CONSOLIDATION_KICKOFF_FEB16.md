# GPU Consolidation & Cleanup - Session Kickoff
**Date**: Feb 16, 2026  
**Thread**: @T-019c675f-91bb-7058-b594-cbc0e38d5091  
**Status**: Planning Complete → Ready for Implementation

---

## Executive Summary

**Objective**: Consolidate GPU implementations across Diffusion, SSM, and Transformer shared components. Implement strict no-fallback GPU semantics with unified auto-detection, fused kernel optimization, and zero-copy memory management.

**Key Targets**:
- ✓ Single unified `GpuDevice::auto_detect()` entry point (no duplicated logic)
- ✓ Strict no-fallback semantics (GPU operations error if backend unavailable)
- ✓ Memory pool reuse > 99% (power-of-2 sizing, zero-reallocation)
- ✓ Performance: 25x-30x speedup (RichardsGLU, Attention kernels)
- ✓ Numerical accuracy: < 1e-4 relative error (GPU vs CPU)

**Timeline**: 8 hours, 3 parallel workstreams

---

## What's Complete (Phase 5.5)

✓ GPU memory pool infrastructure  
✓ Unified kernel dispatcher (`UnifiedGpuKernels`)  
✓ Backend abstraction layer (`UnifiedGpuBackend`)  
✓ `GpuComponent` trait for component GPU management  
✓ CUDA/Metal/WGPU backend implementations (partial)  

---

## What Needs Implementation (Phase 5.6)

### Critical (Blocks all kernels)
- 🔴 **GpuDevice::auto_detect() consolidation** (2 hours)
  - Merge duplicated backend detection logic
  - Implement priority order: CUDA > Metal > Vulkan > WGPU
  - Strict error if no GPU (no CPU fallback)

### High Priority (Core functionality)
- 🟡 **SharedAttentionContext GPU kernel** (2 hours)
  - Similarity matrix computation
  - Softmax normalization
  - Residual combination
  - Target: 20x speedup

- 🟡 **SharedFeedforward RichardsGLU fused kernel** (2 hours)
  - Two-pass fused implementation
  - Bias and activation on GPU (not CPU)
  - Target: 25x speedup

- 🟡 **SharedTemporalProcessing kernels** (2 hours)
  - Attention: Q/K/V projection + scaled softmax + output
  - Mamba: Selective scan with state persistence
  - RG-LRU: Recurrent computation
  - Targets: 30x / 20x / 15x speedups

### Important (Validation)
- 🟢 **Numerical validation tests** (1 hour)
  - GPU vs CPU output comparison
  - Tolerance: < 1e-4 relative error
  - Test all batch sizes: 1, 32, 128, 512, 1024

- 🟢 **Performance benchmarks** (1 hour)
  - Latency measurement per kernel
  - Speedup verification
  - Memory usage profiling

---

## Architecture Decision: Strict No-Fallback

**Old Pattern** (still in some code):
```rust
match forward_gpu(input) {
    Ok(output) => output,
    Err(_) => forward_cpu(input),  // ❌ Silent fallback
}
```

**New Pattern** (Phase 5.6):
```rust
if gpu_device.is_some() {
    return forward_gpu(input)
        .expect("GPU forward failed - strict no-fallback semantics");
}
Ok(forward_cpu(input))  // Only if GPU not attached
```

**Rationale**:
- Predictable performance (no surprise CPU fallback)
- Easier debugging (errors surface immediately)
- Clearer intent (GPU-accelerated or CPU, not both)

---

## 3 Parallel Workstreams

### Workstream 1: GPU Backend Consolidation
**Owner**: TBD | **Duration**: 2 hours | **Blocker**: None

**Deliverables**:
- Consolidated `GpuDevice::auto_detect()` with priority order
- Standardized lock handling patterns across all components
- Merged memory pool implementations (CUDA/Metal/WGPU)

**Output Files**:
- `src/domain/compute/gpu_device.rs` (updated)
- `src/domain/compute/unified_gpu_buffer_pool.rs` (updated)
- Lock pattern refactor completed

**Success Metric**: `cargo test --lib --features gpu-all` passes

---

### Workstream 2: GPU Kernel Implementation
**Owner**: TBD | **Duration**: 4 hours | **Blocker**: Workstream 1

**Sub-tasks** (can be parallelized):

**2A: Attention Kernel** (2 hours)
- File: `src/domain/layers/components/attention_context_gpu.rs`
- Workspace structure with pre-allocated buffers
- GPU forward: similarity → softmax → residual add
- Numerical test: < 1e-4 vs CPU
- Target: 20x speedup (128 batch, 512 embed)

**2B: Feedforward RichardsGLU Fused** (2 hours)
- File: `src/domain/layers/components/feedforward_gpu.rs`
- True fused kernel: W1 → Richards → W2 (single GPU pass)
- Bias and activation on GPU (not post-download CPU)
- Memory pool integration
- Target: 25x speedup (1K batch)

**2C: Temporal Processing** (2 hours, concurrent with 2A/2B)
- File: `src/domain/layers/components/temporal_processing_gpu.rs`
- Attention kernel: QKV projection + scaled softmax + output projection
- Mamba selective scan: token-by-token with persistent state
- RG-LRU recurrent: state update + projection
- Targets: 30x / 20x / 15x speedups

**Output Files**:
- `*_gpu.rs` files with implementations
- Numerical test passing
- Benchmarks showing speedups

**Success Metrics**:
- All kernels pass numerical validation tests
- Speedup targets met (25x-30x for core ops)
- Memory pool reuse > 99%

---

### Workstream 3: Testing & Validation
**Owner**: TBD | **Duration**: 2 hours | **Blocker**: Workstream 2

**Deliverables**:

**3A: Numerical Validation Tests**
- File: `tests/gpu_shared_components_phase56.rs`
- Component-by-component GPU vs CPU comparison
- Multiple batch sizes (1, 32, 128, 512, 1024)
- Tolerance: < 1e-4 relative error
- Test all temporal types (Attention, Mamba, RG-LRU)

**3B: Performance Benchmarks**
- File: `benches/gpu_kernels_bench.rs`
- Latency measurement per kernel
- Speedup comparison (GPU vs CPU)
- Memory usage profiling
- Identify bottlenecks

**3C: Integration Tests**
- Full component pipeline: Attention → Feedforward → Temporal
- GPU device attach/detach scenarios
- Auto-detection on GPU/CPU-only systems
- Feature flag validation

**Output Files**:
- Test suite passing with `cargo test --lib --features gpu-all`
- Benchmark results document
- Performance analysis report

**Success Metrics**:
- All tests pass
- Benchmarks show target speedups (25x-30x)
- Numerical accuracy < 1e-4
- Coverage > 90% of GPU code paths

---

## File Organization

```
src/domain/
├── compute/
│   ├── gpu_device.rs                    ← CONSOLIDATE auto_detect()
│   ├── gpu_ops.rs                       ← Verify interface
│   ├── unified_gpu_buffer_pool.rs       ← MERGE implementations
│   ├── cuda/ops.rs
│   ├── metal/ops.rs
│   └── wgpu_ops.rs
│
└── layers/components/
    ├── attention_context.rs             ← CPU reference
    ├── attention_context_gpu.rs         ← IMPLEMENT kernel
    ├── feedforward.rs                   ← CPU reference
    ├── feedforward_gpu.rs               ← IMPLEMENT RichardsGLU
    ├── temporal_processing.rs           ← CPU reference
    ├── temporal_processing_gpu.rs       ← IMPLEMENT Attention/Mamba/RG-LRU
    ├── unified_gpu_kernels.rs           ← Dispatcher
    └── unified_gpu_backend.rs           ← Backend

tests/
└── gpu_shared_components_phase56.rs     ← NEW: Validation tests

benches/
└── gpu_kernels_bench.rs                 ← NEW: Performance benchmarks
```

---

## Key Concepts & Patterns

### Pattern 1: Workspace-Based GPU Operations
```rust
// Pre-allocate buffers once (power-of-2 sizing)
workspace.ensure_capacity(&mut device, batch_size, embed_dim, seq_len)?;

// Reuse buffers across multiple kernel calls
device.gemm_f32(&workspace.buf_input, &workspace.buf_weight, 
                 &mut workspace.buf_output)?;

// Reset workspace for next operation (no deallocation)
workspace.reset();  // Buffers reused!
```

**Benefit**: Allocation overhead < 1%, reuse rate > 99%

---

### Pattern 2: Fused Kernels
```
OLD (3 separate kernels):
  hidden = input @ W1           [GPU kernel 1]
  hidden = richardson(hidden)   [GPU kernel 2]  
  output = hidden @ W2          [GPU kernel 3]
  Total: 3 GPU launches, 2 intermediate downloads

NEW (1 fused kernel):
  output = fused_richards_glu(input, W1, W2)  [GPU kernel 1]
  Total: 1 GPU launch, 0 intermediate downloads
  Speedup: 3x+ from kernel fusion alone
```

---

### Pattern 3: Strict No-Fallback Error Handling
```rust
// GPU path is primary (no silent fallback)
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    if self.gpu_device.is_some() {
        return self.forward_gpu(input)
            .expect("GPU forward failed - strict no-fallback");
    }
    self.forward_cpu(input)
}

// GPU operations must error if backend unavailable
fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = self.device.lock()
        .map_err(|_| ModelError::Backend {
            message: "GPU lock failed in ComponentName::forward_gpu".to_string(),
        })?;
    
    device.gemm_f32(...)?;  // Error propagates (no fallback)
}
```

**Benefit**: Predictable behavior, early error detection, easier debugging

---

## Documentation Generated

Created during planning phase:

1. **CONSOLIDATION_GPU_PHASE5_6_ACTION_PLAN.md** (8 pages)
   - Detailed task breakdown
   - Implementation patterns
   - Success metrics

2. **GPU_CONSOLIDATION_DIAGNOSTIC_FEB16.md** (10 pages)
   - Current state assessment
   - Code quality issues found
   - Missing implementations
   - Immediate actions

3. **QUICK_REFERENCE_GPU_CONSOLIDATION_FEB16.md** (10 pages)
   - File locations
   - Implementation patterns
   - Testing patterns
   - Debugging checklist
   - Common errors & fixes

4. **GPU_CONSOLIDATION_IMPLEMENTATION_ROADMAP.md** (15 pages)
   - Step-by-step implementation guide
   - Code examples for each component
   - Build & test commands
   - Deployment checklist

5. **Architecture Diagram** (Mermaid)
   - GPU infrastructure stack
   - Component relationships
   - Data flow

---

## Build & Test Commands

### Quick Start
```bash
# Check everything compiles
cargo check --lib --features gpu-all

# Run unit tests
cargo test --lib gpu --features gpu-all

# Run integration tests
cargo test --test gpu_shared_components_phase56 --features gpu-all

# Benchmarks
cargo bench --bench gpu_kernels_bench --features gpu-all
```

### Debug Auto-Detection
```bash
cargo test test_auto_detect_no_fallback --lib --features gpu-all -- --nocapture
```

### Specific Component Tests
```bash
cargo test test_attention_context_gpu_vs_cpu --lib --features gpu-all
cargo test test_feedforward_gpu_vs_cpu --lib --features gpu-all
cargo test test_temporal_gpu_vs_cpu --lib --features gpu-all
```

---

## Success Criteria (Session End)

By end of this session:

✓ `GpuDevice::auto_detect()` consolidated (1 source of truth)  
✓ All lock handling patterns standardized (context-specific errors)  
✓ Memory pool implementations merged (zero duplication)  
✓ SharedAttentionContext GPU kernel fully implemented  
✓ SharedFeedforward RichardsGLU fused kernel fully implemented  
✓ SharedTemporalProcessing (Attention) GPU kernel implemented  
✓ Numerical validation tests passing (< 1e-4 error)  
✓ Performance targets verified (25x-30x speedups)  
✓ Integration tests passing (full pipeline)  
✓ Documentation complete (usage guides, best practices)  

---

## Risk Mitigation

### High Risk: GPU kernel correctness
- **Mitigation**: Numerical validation tests with < 1e-4 tolerance
- **Fallback**: CPU reference implementation always available

### High Risk: Performance targets
- **Mitigation**: Benchmark early and often
- **Fallback**: Document achieved speedups, adjust targets if needed

### Medium Risk: Backend-specific issues
- **Mitigation**: Test on actual CUDA/Metal/WGPU hardware
- **Fallback**: Feature flags allow disabling unsupported backends

### Low Risk: Code duplication
- **Mitigation**: Systematic consolidation with git tracking
- **Fallback**: Clear diff review before merging

---

## Next Steps

### Immediate (If continuing):
1. Start Workstream 1 (GPU Device consolidation) - 2 hours
2. Start Workstream 2 (kernel implementations) - 4 hours
3. Parallel with Workstream 3 (tests) - 2 hours

### If pausing for break:
1. All documentation is prepared
2. Implementation roadmap is detailed
3. Quick reference guide is available
4. Auto-detection consolidation is the natural starting point for next session

---

## References & Resources

**Documentation**:
- `AGENTS.md` - Build commands, project structure
- `CONSOLIDATION_GPU_PHASE5_6_ACTION_PLAN.md` - Detailed plan
- `GPU_CONSOLIDATION_DIAGNOSTIC_FEB16.md` - Current state & issues
- `QUICK_REFERENCE_GPU_CONSOLIDATION_FEB16.md` - Quick lookup
- `GPU_CONSOLIDATION_IMPLEMENTATION_ROADMAP.md` - Step-by-step guide

**Code Files**:
- Core: `src/domain/compute/gpu_*.rs`
- Components: `src/domain/layers/components/*_gpu.rs`
- Tests: `tests/gpu_shared_components_phase56.rs`
- Benchmarks: `benches/gpu_kernels_bench.rs`

**Thread**: https://ampcode.com/threads/T-019c675f-91bb-7058-b594-cbc0e38d5091

---

## Questions & Clarifications

### Q: Why strict no-fallback?
**A**: Predictable performance. If GPU is attached, all ops must use it. CPU fallback silently masks performance issues.

### Q: Why fused kernels?
**A**: Reduce kernel launches and intermediate downloads. Richardson GLU: 3 kernels → 1 kernel = 3x+ speedup from fusion alone.

### Q: What's the reallocation overhead?
**A**: Power-of-2 sizing means reallocations only when capacity exceeded. In typical usage: reallocation_count < total_allocations × 1%.

### Q: How to handle backward pass?
**A**: Forward + backward can be separate kernels or fused. Start with separate, fuse later if time permits.

### Q: What if target speedups not met?
**A**: Document actual speedups, investigate bottlenecks, optimize memory layout or kernel parameters.

---

## Session Status

**Status**: ✅ Planning Complete  
**Phase**: Phase 5.6 Ready for Implementation  
**Duration**: 8 hours planned  
**Workstreams**: 3 (can be parallelized)  
**Blockers**: None (ready to start)  

**Next Action**: Begin Workstream 1 (GPU Device auto-detection consolidation)

