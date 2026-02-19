# Phase 5.5 Implementation Checklist
**Duration**: February 14-20, 2026 (7 days)  
**Total Effort**: ~33 hours  
**Current Status**: Planning Complete → Ready to Start Coding

---

## Session 1: Infrastructure Foundation (Day 1-2)
**Target Hours**: 6.5  
**Goal**: Create UnifiedGpuBufferPool + GpuComponent trait with full tests

### Task 1.1: Create UnifiedGpuBufferPool Core
**File**: `src/domain/compute/unified_gpu_buffer_pool.rs`

- [ ] Create file with module documentation
- [ ] Define `BufferSpec` struct with power-of-2 padding
- [ ] Define `GpuCapacity` struct with defaults
- [ ] Define `MemoryStats` struct
- [ ] Implement `UnifiedGpuBufferPool` struct
  - [ ] `device: Arc<Mutex<GpuDevice>>` field
  - [ ] `memory_pools: HashMap` field
  - [ ] `buffer_cache: HashMap` field
  - [ ] `capacity: GpuCapacity` field
  - [ ] `stats: MemoryStats` field
- [ ] Implement `UnifiedGpuBufferPool::with_device(device)`
- [ ] Implement `UnifiedGpuBufferPool::auto_detect()` - STRICT NO FALLBACK
  - [ ] Returns error if no GPU
  - [ ] Error message indicates why GPU unavailable
  - [ ] No CPU fallback logic
- [ ] Implement `allocate(spec)` with buffer caching
- [ ] Implement `get_or_allocate(spec)`
- [ ] Implement `update_capacity(batch, seq, embed)`
- [ ] Implement `memory_stats()`
- [ ] Implement `device()` getter
- [ ] Add doc comments to all public methods
- [ ] Verify imports resolve (Arc, Mutex, HashMap, Result)

**Acceptance**: Compiles with no errors/warnings, BufferSpec tests pass

---

### Task 1.2: Create GpuComponent Trait
**File**: `src/domain/compute/gpu_component.rs`

- [ ] Create file with module documentation
- [ ] Define `GpuComponent` trait
  - [ ] `attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()>`
  - [ ] `detach_gpu_device(&mut self)`
  - [ ] `gpu_ready(&self) -> Result<()>`
  - [ ] `enable_gpu_auto_detect(&mut self) -> Result<()>`
- [ ] Document strict no-fallback design in trait docs
- [ ] Add example implementation in doc comments
- [ ] Verify all trait methods use `Result<T>`

**Acceptance**: Compiles, trait is properly documented

---

### Task 1.3: Update compute/mod.rs Exports
**File**: `src/domain/compute/mod.rs`

- [ ] Add `pub mod unified_gpu_buffer_pool;`
- [ ] Add `pub mod gpu_component;`
- [ ] Add `pub use unified_gpu_buffer_pool::{...};`
- [ ] Add `pub use gpu_component::{GpuComponent};`
- [ ] Verify no circular imports
- [ ] Check re-exports work: `use domain::compute::{UnifiedGpuBufferPool, GpuComponent};`

**Acceptance**: cargo check passes, exports accessible from domain

---

### Task 1.4: Write Infrastructure Tests
**File**: `tests/unified_gpu_buffer_pool_integration.rs`

- [ ] Test: `BufferSpec` power-of-2 padding
  ```rust
  assert!(spec.padded_size() >= spec.size);
  assert_eq!(spec.padded_size() % spec.alignment, 0);
  ```
- [ ] Test: `GpuCapacity` defaults are valid
  ```rust
  let cap = GpuCapacity::default();
  assert!(cap.max_batch_size > 0);
  ```
- [ ] Test: `auto_detect()` strict no-fallback
  ```rust
  match UnifiedGpuBufferPool::auto_detect() {
      Ok(_) => { /* GPU available */ }
      Err(e) => { /* Error message should indicate GPU unavailable */ }
  }
  ```
- [ ] Test: `with_device()` creates pool
- [ ] Test: `update_capacity()` updates state
- [ ] Test: `memory_stats()` returns valid stats
- [ ] Test: `device()` getter returns Arc
- [ ] Test: Multiple pools can exist independently
- [ ] Test: Buffer cache LRU behavior (if implemented)

**Expected**: All 8+ tests pass

**Acceptance**: `cargo test --lib unified_gpu_buffer_pool_integration`

---

## Session 2: Shared Component GPU Integration (Day 3-4)
**Target Hours**: 9.5  
**Goal**: All shared components implement GpuComponent trait with integration tests

### Task 2.1: SharedAttentionContext GPU Integration
**File**: `src/domain/layers/components/attention_context.rs`

- [ ] Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- [ ] Implement `GpuComponent` trait
  - [ ] `attach_gpu_device()`: Stores device in field
  - [ ] `detach_gpu_device()`: Sets field to None
  - [ ] `gpu_ready()`: Checks device is Some, returns error if None with clear message
  - [ ] `enable_gpu_auto_detect()`: Calls `GpuDevice::auto_detect()` and stores
- [ ] Update `apply_context_gpu_with_workspace()` to use device from field
- [ ] Add tests for trait implementation
- [ ] Verify `apply_context_gpu()` and `apply_context_gpu_with_workspace()` use GPU device

**Acceptance**: Compiles, `cargo test --lib attention_context` passes

---

### Task 2.2: SharedFeedforward GPU Integration
**File**: `src/domain/layers/components/feedforward.rs`

- [ ] Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- [ ] Implement `GpuComponent` trait (same pattern as AttentionContext)
- [ ] Update `forward_gpu()` to use device from field
- [ ] Update `forward_into()` to support GPU path
- [ ] Add tests for trait implementation

**Acceptance**: Compiles, `cargo test --lib feedforward_gpu` passes

---

### Task 2.3: TemporalMixingLayer GPU Integration
**File**: `src/domain/layers/components/temporal_processing.rs`

- [ ] Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- [ ] Implement `GpuComponent` trait
- [ ] Update `forward_gpu()` to use device from field
- [ ] Link to temporal_processing_gpu.rs implementation
- [ ] Mark SSM kernel implementations as TODO (for Phase 5.6)

**Acceptance**: Compiles, trait methods accessible

---

### Task 2.4: PolyAttention GPU Integration
**File**: `src/domain/attention/poly_attention.rs`

- [ ] Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- [ ] Implement `GpuComponent` trait
- [ ] Update `forward_gpu()` to use device from field
- [ ] Clean up orphaned kernel code
- [ ] Fix any syntax errors in parent file
- [ ] Document which kernel operations need implementation

**Acceptance**: Compiles with no errors, `cargo clippy` clean

---

### Task 2.5: Create Shared Component Integration Tests
**File**: `tests/shared_component_gpu_integration.rs`

- [ ] Test: AttentionContext GPU device attachment
- [ ] Test: Feedforward GPU device attachment
- [ ] Test: TemporalMixingLayer GPU device attachment
- [ ] Test: PolyAttention GPU device attachment
- [ ] Test: Auto-detect strict no-fallback on each component
- [ ] Test: Component chain (Attention → Temporal → Feedforward) with shared GPU device
- [ ] Test: GPU ready check works across component chain
- [ ] Test: Detach GPU reverts to CPU path (if applicable)

**Expected**: 8+ integration tests

**Acceptance**: `cargo test --lib shared_component_gpu_integration` passes

---

## Session 3: Block-Level GPU Implementations (Day 5)
**Target Hours**: 7.5  
**Goal**: DiffusionBlock + SSM stubs + TransformerBlock verification

### Task 3.1: DiffusionBlock GPU Forward
**File**: `src/domain/layers/diffusion/block.rs`

- [ ] Add GPU device field
- [ ] Create `forward_gpu()` method signature
  ```rust
  pub fn forward_gpu(
      &mut self,
      input: &Array2<f32>,
      time_step: f32,
      context: Option<&Array2<f32>>,
      pool: &mut UnifiedGpuBufferPool,
  ) -> Result<Array2<f32>>
  ```
- [ ] Implement GPU forward pipeline
  - [ ] Embed input on GPU
  - [ ] Process through denoising on GPU
  - [ ] Output projection on GPU
- [ ] Add implementation tests
- [ ] Document GPU memory requirements
- [ ] Verify numerical parity with CPU (ε ≤ 1e-4)

**Acceptance**: `cargo test --lib diffusion_block_gpu` passes, numerical parity verified

---

### Task 3.2: SSM GPU Kernel Stubs
**File**: `src/domain/layers/ssm/components/` (new GPU kernel files)

- [ ] Create `gpu_mamba_kernel.wgsl` with TODO placeholders
  - [ ] Mamba selective scan signature
  - [ ] Comments explaining operation
  - [ ] Marked for Phase 5.6 implementation
- [ ] Create `gpu_rg_lru_kernel.wgsl` with TODO placeholders
  - [ ] RG-LRU state update signature
  - [ ] Comments explaining operation
  - [ ] Marked for Phase 5.6 implementation
- [ ] Update `temporal_processing_gpu.rs` to reference new kernels
- [ ] Add TODO comments in Rust code
  ```rust
  // TODO Phase 5.6: Implement actual WGSL kernels
  // For now, returns error indicating not yet implemented
  ```

**Acceptance**: Files created, references updated, no build errors

---

### Task 3.3: TransformerBlock Full GPU Chain Verification
**File**: `src/domain/layers/transformer/block.rs`

- [ ] Verify GPU forward path exists:
  - [ ] Attention GPU forward
  - [ ] Temporal mixing GPU forward
  - [ ] Feedforward GPU forward
- [ ] Check that entire forward is GPU-to-GPU (no transfers mid-pipeline)
- [ ] Add test for full chain:
  ```rust
  #[test]
  fn test_transformer_block_full_gpu_chain() {
      // Input → Attention GPU → Temporal GPU → Feedforward GPU → Output
      // Verify no CPU fallback in middle
  }
  ```
- [ ] Verify skip connections work on GPU
- [ ] Verify post-norm on GPU
- [ ] Test numerical accuracy against CPU reference

**Acceptance**: Full GPU chain verified, no CPU fallback in middle, tests pass

---

### Task 3.4: Block-Level Integration Tests
**File**: `tests/block_gpu_forward_integration.rs`

- [ ] Test: DiffusionBlock GPU forward completes without error
- [ ] Test: DiffusionBlock GPU output shape correct
- [ ] Test: DiffusionBlock GPU vs CPU numerical parity
- [ ] Test: TransformerBlock full GPU chain (no CPU fallback)
- [ ] Test: TransformerBlock GPU memory usage reasonable
- [ ] Test: TemporalMixingLayer GPU forward (even with placeholder kernel)
- [ ] Test: GPU out-of-memory error handling graceful

**Expected**: 7+ integration tests

**Acceptance**: `cargo test --test block_gpu_forward_integration` passes

---

## Session 4: Deprecation & Cleanup (Day 6)
**Target Hours**: 4  
**Goal**: Mark legacy managers for removal, clean up orphaned code

### Task 4.1: Deprecate SharedComponentGpuManager
**File**: `src/domain/layers/components/shared_gpu_manager.rs`

- [ ] Add #[deprecated] attribute
  ```rust
  #[deprecated(
      since = "0.2.0",
      note = "use UnifiedGpuBufferPool from domain::compute instead"
  )]
  pub struct SharedComponentGpuManager { ... }
  ```
- [ ] Add deprecation docs at module level
- [ ] Create wrapper methods calling UnifiedGpuBufferPool
  ```rust
  pub fn as_unified_pool(&self) -> Result<UnifiedGpuBufferPool> {
      // Convert to unified pool
  }
  ```
- [ ] Document migration path in module docs
- [ ] Verify backward compatibility maintained (old code still works)

**Acceptance**: Deprecation warning shown, old API still functional

---

### Task 4.2: Deprecate GpuSharedOpsContext
**File**: `src/domain/layers/components/gpu_shared_ops.rs`

- [ ] Add #[deprecated] attribute
- [ ] Add deprecation docs
- [ ] Create wrapper methods calling UnifiedGpuBufferPool
- [ ] Update batch executor to use unified pool internally
- [ ] Document migration path

**Acceptance**: Deprecation warning shown, old API still functional

---

### Task 4.3: Fix ForwardContext Fields
**File**: `src/application/training/forward_context.rs`

- [ ] Find all ForwardContext initializations
- [ ] Add missing `low_rank_query_gate` field if absent
- [ ] Ensure GPU initialization sets device properly
- [ ] Update from_model_config() if exists
- [ ] Verify all initializers compile

**Acceptance**: No compilation errors in forward_context

---

### Task 4.4: PolyAttention Code Cleanup
**File**: `src/domain/attention/poly_attention.rs`

- [ ] Remove orphaned kernel code (no longer called)
- [ ] Fix syntax errors
- [ ] Ensure all public methods have clear signatures
- [ ] Update kernel invocation to use unified pool device
- [ ] Add TODO comments for unfinished work
- [ ] Run clippy to catch remaining issues

**Acceptance**: `cargo clippy` clean, no syntax errors

---

## Session 5: Validation & Benchmarking (Day 7)
**Target Hours**: 5.5  
**Goal**: Comprehensive testing, performance validation, documentation

### Task 5.1: Full Test Suite Execution
**Commands**:

- [ ] `cargo test --lib` (all 529+ tests)
  - [ ] Wait for completion
  - [ ] Record: `test result: ok. XXX passed; 0 failed`
  - [ ] Verify: 0 failures
- [ ] `cargo test --lib --features gpu-wgpu` (if GPU available)
  - [ ] Record: `test result: ok. XXX passed; 0 failed`
- [ ] Unit tests per module:
  - [ ] `cargo test --lib unified_gpu_buffer_pool`
  - [ ] `cargo test --lib shared_component_gpu`
  - [ ] `cargo test --lib diffusion_block_gpu`
  - [ ] `cargo test --lib transformer_block_gpu`

**Acceptance**: All tests pass (529+)

---

### Task 5.2: Build Quality Checks
**Commands**:

- [ ] `cargo fmt -- --check` (formatting)
  - [ ] No formatting changes needed
- [ ] `cargo clippy --all-targets` (linting)
  - [ ] No clippy warnings (except deprecated items)
- [ ] `cargo check --release` (final compilation)
  - [ ] No errors or warnings
  - [ ] Verify build time: <5 minutes

**Acceptance**: Clean build, no warnings

---

### Task 5.3: Benchmarking (optional, but recommended)
**File**: `benches/gpu_performance_baseline.rs` (if doesn't exist)

- [ ] Create GPU vs CPU benchmark
  - [ ] Single-batch latency (batch=1)
  - [ ] Multi-batch latency (batch=32)
  - [ ] Memory usage tracking
- [ ] Run benchmarks:
  ```bash
  cargo bench --bench gpu_performance_baseline 2>&1 | tee bench_results.txt
  ```
- [ ] Record baseline numbers:
  - [ ] CPU single-batch: __ ms
  - [ ] GPU single-batch: __ ms
  - [ ] GPU multi-batch: __ ms
  - [ ] GPU speedup: __ x
- [ ] Verify GPU speedup >= 1.5x on batch=32

**Expected Results**:
- Single-batch: GPU ≈ CPU (within 20%)
- Multi-batch: GPU 2-5x faster

**Acceptance**: Benchmarks run, results recorded

---

### Task 5.4: Documentation Update
**Files**:

- [ ] Update `GPU_BACKEND_IMPLEMENTATION_STATUS.md`
  - [ ] Mark Phase 5.5 complete
  - [ ] Update status table (SharedAttention ✓, Feedforward ✓, etc.)
  - [ ] List new files created
- [ ] Update `CONSOLIDATION_GPU_IMPLEMENTATION_PHASE5.5.md`
  - [ ] Add Session completion section
  - [ ] Record actual hours spent per session
  - [ ] Note any deviations from plan
- [ ] Create `PHASE5.5_COMPLETION_SUMMARY.md`
  - [ ] What was done
  - [ ] What works now (components, blocks)
  - [ ] What's next (Phase 5.6 priorities)

**Acceptance**: Documentation reflects current state

---

### Task 5.5: Commit & Session Complete
**Actions**:

- [ ] Review all changes:
  - [ ] New files created (7+)
  - [ ] Modified files updated (8+)
  - [ ] Tests added (30+)
  - [ ] Deprecations added (2+)
- [ ] Create git commit:
  ```bash
  git add --all
  git commit -m "Phase 5.5: GPU consolidation - UnifiedGpuBufferPool, GpuComponent, block implementations"
  ```
- [ ] (Optional) Create tag:
  ```bash
  git tag -a v0.2.0-phase5.5 -m "Phase 5.5 GPU consolidation complete"
  ```
- [ ] Write final summary in this file

**Acceptance**: All changes committed

---

## Overall Progress Tracking

### Files to Create
- [ ] `src/domain/compute/unified_gpu_buffer_pool.rs` (300+ lines)
- [ ] `src/domain/compute/gpu_component.rs` (100+ lines)
- [ ] `src/domain/layers/ssm/components/gpu_mamba_kernel.wgsl` (50+ lines)
- [ ] `src/domain/layers/ssm/components/gpu_rg_lru_kernel.wgsl` (50+ lines)
- [ ] `tests/unified_gpu_buffer_pool_integration.rs` (150+ lines)
- [ ] `tests/shared_component_gpu_integration.rs` (200+ lines)
- [ ] `tests/block_gpu_forward_integration.rs` (150+ lines)

**Total New Files**: 7  
**Total New Lines**: ~1000+

### Files to Modify
- [ ] `src/domain/compute/mod.rs` (3 lines add)
- [ ] `src/domain/layers/components/attention_context.rs` (20+ lines)
- [ ] `src/domain/layers/components/feedforward.rs` (20+ lines)
- [ ] `src/domain/layers/components/temporal_processing.rs` (20+ lines)
- [ ] `src/domain/attention/poly_attention.rs` (30+ lines)
- [ ] `src/domain/layers/diffusion/block.rs` (50+ lines)
- [ ] `src/domain/layers/components/shared_gpu_manager.rs` (50+ lines)
- [ ] `src/domain/layers/components/gpu_shared_ops.rs` (50+ lines)

**Total Modified Files**: 8  
**Total Modified Lines**: ~240+

### Test Coverage
- [ ] UnifiedGpuBufferPool: 8+ tests
- [ ] Shared components: 8+ tests
- [ ] Block-level: 7+ tests
- [ ] Deprecation warning: 2+ tests
- **Total New Tests**: 25+

---

## Session Completion Tracking

| Session | Status | Hours | Complete By | Notes |
|---------|--------|-------|-------------|-------|
| 1. Infrastructure | ⬜ TODO | 6.5h | Feb 14 | UnifiedPool + GpuComponent |
| 2. Components | ⬜ TODO | 9.5h | Feb 17 | 4 components implement trait |
| 3. Blocks | ⬜ TODO | 7.5h | Feb 18 | Diffusion + SSM + Verify Transform |
| 4. Cleanup | ⬜ TODO | 4h | Feb 19 | Deprecations + orphaned code |
| 5. Validation | ⬜ TODO | 5.5h | Feb 20 | Tests + benchmarks + docs |

**Total**: 33 hours over 7 days

---

## Failure Points & Mitigation

| Risk | When Detected | How to Fix |
|------|---------------|-----------|
| Build hangs | cargo check >30s | Kill with Ctrl+C, check for infinite loops |
| GPU auto_detect() returns Ok but GPU absent | During test | Implement strict error checking, not silent fallback |
| Arc<Mutex<>> deadlock | Component tests freeze | Ensure no locks held across operations |
| Buffer memory leak | Benchmark memory increasing | Audit allocate/deallocate pair for each component |
| Numerical mismatch GPU vs CPU | Validation tests fail | Check float precision, tolerance ε=1e-4 |
| Tests timeout | Full test suite >15min | Run individual tests, skip longest ones |

---

## Definition of Done: Phase 5.5

✅ **Phase 5.5 Complete When**:

1. **Code Quality**
   - [ ] 0 compilation errors
   - [ ] 0 clippy warnings (except deprecations)
   - [ ] Code formatted with rustfmt
   - [ ] All unsafe code documented

2. **Functionality**
   - [ ] UnifiedGpuBufferPool fully implemented
   - [ ] GpuComponent implemented on 4 components
   - [ ] DiffusionBlock has GPU forward
   - [ ] SSM kernels stubbed (to be implemented Phase 5.6)
   - [ ] TransformerBlock full GPU chain verified

3. **Testing**
   - [ ] 529+ unit tests pass
   - [ ] 25+ new tests added (GPU-specific)
   - [ ] GPU vs CPU numerical parity verified
   - [ ] No segfaults or panics
   - [ ] Memory leaks: 0 detected

4. **Deprecation**
   - [ ] SharedComponentGpuManager #[deprecated]
   - [ ] GpuSharedOpsContext #[deprecated]
   - [ ] Migration path documented
   - [ ] Old API still works (backward compatible)

5. **Performance**
   - [ ] Benchmarks established
   - [ ] GPU speedup measured (target: 1.5x+ on batch=32)
   - [ ] Memory usage tracked
   - [ ] No performance regressions vs Phase 5.4

6. **Documentation**
   - [ ] All public APIs documented
   - [ ] Strict no-fallback design documented
   - [ ] Migration guide for Phase 6
   - [ ] Benchmark results recorded

---

## Sign-Off Template (to fill in when complete)

```
Phase 5.5 GPU Consolidation - COMPLETION REPORT

Date Completed: [DATE]
Total Hours: [HOURS]
Deviations from Plan: [NOTES]

Test Results:
  Unit tests: [PASS/FAIL] XXX/XXX
  GPU tests: [PASS/FAIL] XXX/XXX
  Integration tests: [PASS/FAIL] XXX/XXX
  
Build Status: [PASS/FAIL]
  Warnings: [COUNT]
  Errors: [COUNT]

Performance Baseline:
  CPU single-batch: [MS]
  GPU single-batch: [MS]
  GPU multi-batch (32): [MS]
  GPU speedup: [X]x

Next Steps: Phase 5.6 - Implement actual SSM kernels

Signed: [NAME] on [DATE]
```

---

## Quick Status Check Commands

Use these throughout implementation to verify progress:

```bash
# Check current test status
cargo test --lib 2>&1 | tail -3

# Check compilation
cargo check --lib 2>&1 | grep -E "error|warning" || echo "✓ Clean"

# Count new tests
grep -r "^#\[test\]" tests/ | wc -l

# Check deprecated items
grep -r "#\[deprecated" src/domain/layers/components/

# Verify exports
cargo expand --lib domain::compute | head -20
```

---

## References

- **Full Specification**: CONSOLIDATION_GPU_IMPLEMENTATION_PHASE5.5.md
- **Execution Timeline**: PHASE5.5_EXECUTION_ROADMAP.md
- **Quick Start Guide**: PHASE5.5_QUICK_START.md
- **Session Summary**: SESSION_PHASE5.5_INITIALIZATION_SUMMARY.md
- **Previous Work**: CONSOLIDATION_GPU_BACKEND_SESSION_SUMMARY.md
- **Conventions**: AGENTS.md
- **Thread**: @T-019c5ef0-36a1-70be-a528-03e1253f1542

---

**Start Date**: February 14, 2026  
**Estimated Completion**: February 20, 2026  
**Status**: Ready to Begin ✓
