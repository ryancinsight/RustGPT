# GPU Backend Integration & Error Resolution Plan

## Current Status Summary
- **Compilation Errors**: 15 blocking errors
- **Warnings**: 13 deprecation/cleanup warnings  
- **Root Causes**: Missing type definitions, obsolete APIs, incomplete GPU implementations

---

## Critical Errors (Priority 1)

### 1. RichardsCurve Type Missing in poly_attention.rs (Line 407)
**Error**: `cannot find type RichardsCurve in this scope`
**Files**: `src/domain/attention/poly_attention.rs:407`
**Root Cause**: `RichardsCurve` is imported from `src/domain/richards/richards_curve.rs` but not included in poly_attention imports

**Resolution**:
```rust
// Add to poly_attention.rs imports
use crate::domain::richards::richards_curve::RichardsCurve;
```
**Scope**: 1 file, quick fix

---

### 2. CpuGpuMatrixOps References (3 files, 4 occurrences)
**Error**: `could not find CpuGpuMatrixOps in gpu_ops`
**Files**: 
- `src/domain/layers/components/attention_context_gpu.rs:158, 174`
- `src/domain/layers/components/feedforward_gpu.rs:123`
- `src/domain/layers/components/temporal_processing_gpu.rs:198, 214`

**Root Cause**: `CpuGpuMatrixOps` was removed from gpu_ops.rs (obsolete pattern)
  - It was a dual-implementation wrapper trying to support both CPU and GPU
  - Replaced by `UnifiedGpuExecutor` pattern which requires explicit GPU devices

**Resolution Strategy**:
- Replace `CpuGpuMatrixOps` with `GpuMatrixOps` trait object
- Add device requirement checks before calling GPU operations
- For stub implementations: return `ModelError::Backend` explicitly

**Updated Pattern**:
```rust
// OLD (REMOVED)
let mut ops = CpuGpuMatrixOps::new();
ops.gemm_f32(...)?;

// NEW
use crate::domain::compute::gpu_ops::GpuMatrixOps;
use crate::domain::compute::gpu_device::GpuDevice;

let device = GpuDevice::auto_detect()?;  // Explicit GPU requirement
let mut ops = device.create_ops()?;      // Get GPU-specific ops
ops.gemm_f32(...)?;
```

**Files to Fix**:
- `attention_context_gpu.rs` (2 errors, ~50 lines)
- `feedforward_gpu.rs` (1 error, ~50 lines)
- `temporal_processing_gpu.rs` (2 errors, ~50 lines)

---

### 3. SharedAttentionContext Not Found (2 occurrences)
**Error**: `could not find SharedAttentionContext in super`
**Files**: `src/domain/layers/components/attention_context_gpu.rs:154, 168`

**Root Cause**: Type is defined in `attention_context.rs` but not exported from module

**Resolution**:
```rust
// In src/domain/layers/components/mod.rs
pub use attention_context::SharedAttentionContext;
```

**Impact**: 2 lines in 1 file

---

### 4. RichardsGlu Variant Missing Fields (7 errors)
**Error**: `variant RichardsGlu has no field named w1, w2, w3, w_gate, gamma, beta, gate_scale, gate_bias, cached_input, input_norm_stats`
**File**: `src/domain/layers/components/feedforward_gpu.rs:106-115`

**Root Cause**: Feedforward component layout changed but GPU code path wasn't updated

**Investigation Needed**:
1. Check `src/domain/layers/components/common.rs` for actual RichardsGlu definition
2. Compare with CPU implementation in `feedforward.rs`
3. Update GPU path to match

**Resolution Approach**:
- Either restructure RichardsGlu to have those fields
- Or refactor feedforward_gpu.rs to match current RichardsGlu layout
- Consider using builder pattern for complex initialization

---

### 5. PolyAttention Missing Default Implementation (2 occurrences)
**Error**: `no function or associated item named default found for struct PolyAttention`
**Files**: `src/domain/layers/components/temporal_processing_gpu.rs:193, 209`

**Root Cause**: Code tries to call `PolyAttention::default()` but type doesn't derive/impl Default

**Resolution**:
```rust
// In poly_attention.rs
#[derive(Default)]  // Add this
pub struct PolyAttention { ... }

// OR implement manually if fields don't have sensible defaults
impl Default for PolyAttention {
    fn default() -> Self {
        // sensible defaults
    }
}
```

---

## Secondary Issues (Priority 2)

### Import Cleanup Warnings
**Files**:
- `poly_attention.rs:2-11` - Remove unused `Arc`, `Mutex`, `bytemuck`, GPU imports if not using them
- `unified_gpu_buffer_pool.rs:420` - Remove unused `super::*`
- `unified_gpu_executor.rs:386` - Remove unused `super::*`

**Fix**: `cargo clippy --all-targets --fix`

### Unused Variables
- `attention_context_gpu.rs:55, 64, 93, 95` - Define but not use `ops`, `batch_size`, `update_rate`
- `paged_attention_integration.rs:145` - `handle` variable
- `paged_attention.rs:690` - `retrieved_v` variable

**Fix**: Use prefixed underscore if intentional: `let _var = ...`

---

## Implementation Roadmap

### Phase 1: Unblock Compilation (2-3 hours)
1. ✅ Add missing `RichardsCurve` import to poly_attention.rs
2. ✅ Export `SharedAttentionContext` from components module
3. ✅ Update `CpuGpuMatrixOps` references (5 locations):
   - Replace with explicit `GpuMatrixOps` trait + device detection
   - Implement GPU-required error returns
4. ✅ Fix `RichardsGlu` field mismatch (investigate + refactor)
5. ✅ Add `Default` impl to `PolyAttention`
6. ✅ Clean up unused imports/variables

### Phase 2: GPU Backend Validation (3-4 hours)
1. Verify all GPU ops implementations match trait contracts
2. Test WGPU/CUDA device detection
3. Validate RichardsCurve GPU kernel accuracy
4. Test unified buffer pool allocation patterns

### Phase 3: Integration Testing (4-5 hours)
1. End-to-end DiffusionBlock GPU forward pass
2. SSM temporal processing GPU path
3. TransformerBlock + PolyAttention fusion
4. Cross-component workspace sharing validation

---

## Implementation Order

**Start with**:
```
1. Add RichardsCurve import (2 min)
2. Export SharedAttentionContext (2 min)
3. Replace CpuGpuMatrixOps in 5 files (1 hour)
4. Fix RichardsGlu variant access (30-45 min - needs investigation)
5. Add Default impl to PolyAttention (10 min)
6. Clean up imports/variables (15 min)
7. Test compilation (10 min)
```

**Then validate**:
- Full compilation: `cargo build --release`
- Check all GPU paths: `cargo test --lib --features gpu`
- Run integration tests: `cargo test --test '*'`

---

## Success Criteria
- [ ] Clean compilation with 0 errors, <5 warnings
- [ ] All GPU device detection working
- [ ] No silent CPU fallbacks (errors if GPU not available)
- [ ] Unified buffer pool managing all allocations
- [ ] All shared components (Diffusion, SSM, Transformer) using GPU paths
