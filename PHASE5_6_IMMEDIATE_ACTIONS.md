# Phase 5.6 Immediate Actions - GPU Backend Variants
**Date**: February 15, 2026  
**Session Focus**: Consolidation + Strict GPU Implementation  
**Duration**: This session

---

## Critical Path (Next 4 Hours)

### 1. Consolidate SharedFeedforward (HIGHEST PRIORITY)
**Target**: Merge `feedforward_gpu.rs` into `SharedFeedforward` + GpuComponent trait impl

```bash
Tasks:
├── [ ] Read feedforward_gpu.rs (understand current GPU impl)
├── [ ] Implement GpuComponent trait for SharedFeedforward
├── [ ] Add gpu_device field + enable_gpu_auto_detect()
├── [ ] Test: SharedFeedforward.enable_gpu_auto_detect() with strict no-fallback
└── [ ] Verify RichardsGLU GPU kernel integration
```

**Impact**: Unifies 2 files, enables strict GPU execution for primary feedforward

---

### 2. Implement GpuComponent for SharedTemporalProcessing
**Target**: Add GPU device management + auto-detection

```bash
Tasks:
├── [ ] Add gpu_device field to SharedTemporalProcessing
├── [ ] Implement GpuComponent trait methods
├── [ ] Update forward() to call forward_gpu() if device attached
├── [ ] Test auto-detection for temporal layers
└── [ ] Verify strict no-fallback behavior
```

**Impact**: Unified GPU management across temporal operators

---

### 3. Implement GpuComponent for SharedAttentionContext
**Target**: GPU-enable attention context with strict semantics

```bash
Tasks:
├── [ ] Add gpu_device field to SharedAttentionContext
├── [ ] Implement GpuComponent trait methods
├── [ ] Ensure capacity pre-allocation on GPU
└── [ ] Test GPU context modulation kernel
```

**Impact**: All 3 shared components now GPU-capable

---

### 4. Create GPU Backend Abstraction
**Target**: Trait-based GPU backend variant system

**File**: `src/domain/compute/gpu_backend.rs`

```rust
pub trait GpuBackendVariant: Send + Sync {
    fn name(&self) -> &'static str;
    fn is_available() -> bool;
    fn supports_feature(&self, feature: &str) -> bool;
}

pub struct WgpuBackend;
impl GpuBackendVariant for WgpuBackend {
    fn name(&self) -> &'static str { "WGPU" }
    fn is_available() -> bool { cfg!(feature = "wgpu") }
    fn supports_feature(&self, feature: &str) -> bool {
        matches!(feature, "gemm" | "softmax" | "layer_norm" | "element_wise")
    }
}
```

---

### 5. Diagnostic: GPU Detection & Error Handling
**Target**: Verify strict no-fallback enforcement

```bash
Tests:
├── [ ] GpuDevice::auto_detect() fails gracefully without GPU features
├── [ ] SharedFeedforward::enable_gpu_auto_detect() returns clear error
├── [ ] GPU ops return explicit errors (not silent fallback)
└── [ ] Error messages guide user to enable features
```

---

## Files to Modify This Session

### Create
```
src/domain/compute/gpu_backend.rs      (backend abstraction)
PHASE5_6_SESSION_PROGRESS.md           (progress tracking)
GPU_BACKEND_INTEGRATION_TESTS.md       (test plan)
```

### Modify
```
src/domain/layers/components/feedforward.rs              (GpuComponent impl)
src/domain/layers/components/temporal_processing.rs      (GpuComponent impl)
src/domain/layers/components/attention_context.rs        (GpuComponent impl)
src/domain/compute/mod.rs                                (export gpu_backend)
```

### Optional (Next Session)
```
src/domain/layers/components/feedforward_gpu.rs          (merge into feedforward.rs)
src/domain/layers/components/temporal_processing_gpu.rs  (deprecate after test)
```

---

## Acceptance Criteria

### ✅ Code Consolidation
- [ ] No duplicate GPU implementations across shared components
- [ ] All 3 shared components implement GpuComponent trait
- [ ] GPU device auto-detection works consistently

### ✅ Strict No-Fallback
- [ ] GPU operations error clearly when device not attached
- [ ] No silent CPU fallback behavior
- [ ] All error messages mention required feature flags

### ✅ Memory Efficiency
- [ ] Power-of-2 buffer allocation tracked
- [ ] AllocationStats verify reuse > 90% on repeated calls
- [ ] Zero-allocation reuse in forward passes

### ✅ Testing
- [ ] Unit tests: Each shared component with/without GPU
- [ ] Integration tests: Full forward pass GPU execution
- [ ] Error tests: Verify no-fallback semantics
- [ ] Benchmark: Measure GPU speedup (target: 20-30×)

---

## Session Rollback (if needed)

If consolidation reveals incompatibilities:
1. Revert all changes to shared components
2. Keep gpu_backend.rs abstraction (new, non-breaking)
3. Create issue for next session: "GPU implementation requires redesign"

---

## Next Session (After Consolidation)

Once shared components are consolidated:

1. **Phase 5.6.1a**: RichardsGLU WGPU Optimization
   - Kernel fusion: 2 GEMMs + Richards + gate + output proj
   - Target: 25× speedup (50ms → 2ms)
   
2. **Phase 5.6.1b**: MoE WGPU Kernels
   - Router GEMM + softmax, parallel experts
   - Target: 20× speedup (100ms → 5ms)
   
3. **Phase 5.6.2a**: PolyAttention WGPU Kernel
   - Polynomial basis + QKV + attention + gating
   - Target: 30× speedup (30ms → 1ms)

---

## Command Reference

```bash
# Build with WGPU backend
cargo build --release --features gpu-wgpu

# Build with all GPU backends
cargo build --release --features gpu-all

# Run tests (GPU optional)
cargo test --lib gpu_component -- --nocapture

# Benchmark GPU operations
cargo bench --bench gpu_shared_components

# Format code after changes
cargo fmt
```

---

## Notes

- **Strict No-Fallback**: If any GPU operation is requested but device unavailable, error immediately with helpful message
- **Power-of-2 Sizing**: All GPU buffers allocated with power-of-2 sizing to minimize reallocation frequency
- **Zero-Allocation Reuse**: Buffers are pre-allocated and reused; forward passes should not trigger reallocations
- **Feature Gating**: GPU code is feature-gated; compilation without GPU features succeeds but GPU ops return errors at runtime
