# Phase 5.5 GPU Consolidation - Week 1 Execution Roadmap

**Period**: Week of February 14-20, 2026  
**Scope**: UnifiedGpuBufferPool + Shared Component GPU Integration  
**Constraint**: Strict no-fallback, automatic GPU detection

---

## Session 1: Infrastructure Setup (Day 1-2)

### Goal
Establish unified GPU management foundation that all components depend on.

### Tasks

#### 1.1 Create `unified_gpu_buffer_pool.rs`
**File**: `src/domain/compute/unified_gpu_buffer_pool.rs`

**Core Types**:
```rust
/// Unified GPU buffer pool with automatic device detection
pub struct UnifiedGpuBufferPool {
    device: Arc<Mutex<GpuDevice>>,
    memory_pools: HashMap<PoolId, Arc<Mutex<dyn GpuMemoryPool>>>,
    buffer_cache: LruCache<BufferSpec, Arc<GpuBuffer>>,
    capacity: GpuCapacity,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BufferSpec {
    size: usize,
    alignment: BufferAlignment,
}

pub struct GpuCapacity {
    max_batch_size: usize,
    max_seq_length: usize,
    max_embedding_dim: usize,
}
```

**Key Methods**:
- `pub fn auto_detect() -> Result<Self>` - strict, no fallback
- `pub fn with_device(device: GpuDevice) -> Self`
- `pub fn allocate(&mut self, spec: BufferSpec) -> Result<Arc<GpuBuffer>>`
- `pub fn update_capacity(&mut self, batch: usize, seq: usize, embed: usize)`
- `pub fn memory_stats(&self) -> MemoryStats`

**Estimated Effort**: 2 hours

---

#### 1.2 Create `gpu_component.rs`
**File**: `src/domain/compute/gpu_component.rs`

**Core Trait**:
```rust
/// Standard interface for GPU-capable components
pub trait GpuComponent: Send + Sync {
    /// Attach an external GPU device
    fn attach_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()>;
    
    /// Detach GPU device (fallback to CPU)
    fn detach_gpu_device(&mut self);
    
    /// Check if GPU is ready
    fn gpu_ready(&self) -> Result<()>;
    
    /// Enable automatic GPU detection (strict: errors if no GPU)
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;
}
```

**Implementations (stubs)** for:
- SharedAttentionContext
- SharedFeedforward
- TemporalMixingLayer
- PolyAttention

**Estimated Effort**: 2 hours

---

#### 1.3 Update `compute/mod.rs`
**File**: `src/domain/compute/mod.rs`

**Changes**:
```rust
pub mod unified_gpu_buffer_pool;
pub mod gpu_component;

pub use unified_gpu_buffer_pool::UnifiedGpuBufferPool;
pub use gpu_component::GpuComponent;
```

**Estimated Effort**: 0.5 hours

---

#### 1.4 Write Infrastructure Tests
**File**: `tests/unified_gpu_buffer_pool_integration.rs`

**Tests**:
```rust
#[test]
fn test_unified_gpu_buffer_pool_creation() {}

#[test]
fn test_unified_gpu_auto_detect_strict_no_fallback() {}

#[test]
fn test_buffer_allocation_with_power_of_2_sizing() {}

#[test]
fn test_buffer_cache_lru_eviction() {}

#[test]
fn test_capacity_tracking_updates() {}

#[test]
fn test_memory_stats_reporting() {}
```

**Estimated Effort**: 2 hours

---

**Session 1 Subtotal**: ~6.5 hours
**Expected State**: Builds, tests pass, infrastructure ready for component migration

---

## Session 2: Shared Component GPU Migration (Day 3-4)

### Goal
Implement GpuComponent trait on all shared components.

### Tasks

#### 2.1 SharedAttentionContext GPU Integration
**File**: `src/domain/layers/components/attention_context_gpu.rs`

**Changes**:
- Implement `GpuComponent` trait
- Add device field: `gpu_device: Option<Arc<Mutex<GpuDevice>>>`
- Enhance `apply_context_gpu_with_workspace()` to use `UnifiedGpuBufferPool`

**Estimated Effort**: 1.5 hours

---

#### 2.2 SharedFeedforward GPU Integration
**File**: `src/domain/layers/components/feedforward_gpu.rs`

**Changes**:
- Implement `GpuComponent` trait
- Update `forward_gpu()` to allocate via unified pool
- Ensure GELU/SiLU operations use pool

**Estimated Effort**: 1.5 hours

---

#### 2.3 TemporalMixingLayer GPU Integration
**File**: `src/domain/layers/components/temporal_processing_gpu.rs`

**Changes**:
- Implement `GpuComponent` trait
- Update `forward_gpu()` to use unified pool
- Mark placeholder SSM kernels for replacement

**Estimated Effort**: 1.5 hours

---

#### 2.4 PolyAttention GPU Integration
**File**: `src/domain/attention/poly_attention_gpu.rs`

**Changes**:
- Implement `GpuComponent` trait
- Fix orphaned kernel code in parent `poly_attention.rs`
- Verify `forward_gpu()` path

**Estimated Effort**: 2 hours

---

#### 2.5 Integration Tests for Components
**File**: `tests/shared_component_gpu_integration.rs`

**Tests**:
```rust
#[test]
fn test_attention_context_gpu_component() {}

#[test]
fn test_feedforward_gpu_component() {}

#[test]
fn test_temporal_processing_gpu_component() {}

#[test]
fn test_poly_attention_gpu_component() {}

#[test]
fn test_component_chain_gpu_forward() {}
```

**Estimated Effort**: 2 hours

---

**Session 2 Subtotal**: ~9.5 hours
**Expected State**: All shared components implement GpuComponent, pass component integration tests

---

## Session 3: Block-Level GPU Implementations (Day 5)

### Goal
Implement GPU forward paths for Diffusion, SSM, and verify Transformer.

### Tasks

#### 3.1 DiffusionBlock GPU Forward
**File**: `src/domain/layers/diffusion/block.rs` (new GPU methods)

**Implementation**:
```rust
pub fn forward_gpu(
    &mut self,
    input: &Array2<f32>,
    time_step: f32,
    context: Option<&Array2<f32>>,
    pool: &mut UnifiedGpuBufferPool,
) -> Result<Array2<f32>> {
    self.gpu_ready()?;
    // GPU diffusion forward pipeline
}
```

**Estimated Effort**: 3 hours

---

#### 3.2 TemporalMixingLayer GPU Kernel Placeholder
**File**: `src/domain/layers/ssm/components/gpu_temporal_kernel.wgsl` (new)

**Content**:
- Placeholder for Mamba scan kernel
- Placeholder for RG-LRU kernel
- Clear TODOs for implementation

**Estimated Effort**: 1 hour

---

#### 3.3 TransformerBlock Full GPU Chain Verification
**File**: `src/domain/layers/transformer/block.rs`

**Verification**:
- Test full forward: Input → Attention GPU → Temporal GPU → Feedforward GPU → Output
- Ensure no CPU/GPU transfers mid-pipeline
- Validate numerical accuracy (ε ≤ 1e-4)

**Estimated Effort**: 1.5 hours

---

#### 3.4 Block-Level Tests
**File**: `tests/block_gpu_forward_integration.rs`

**Tests**:
```rust
#[test]
fn test_diffusion_block_gpu_forward() {}

#[test]
fn test_transformer_block_full_gpu_chain() {}

#[test]
fn test_temporal_mixing_gpu_forward() {}

#[test]
fn test_gpu_cpu_numerical_parity() {}
```

**Estimated Effort**: 2 hours

---

**Session 3 Subtotal**: ~7.5 hours
**Expected State**: Diffusion/Transformer have GPU paths, SSM kernels stubbed, tests pass

---

## Session 4: Deprecation & Cleanup (Day 6)

### Goal
Clean up legacy managers and prepare for Phase 6.

### Tasks

#### 4.1 Deprecate SharedComponentGpuManager
**File**: `src/domain/layers/components/shared_gpu_manager.rs`

**Changes**:
```rust
#[deprecated(
    since = "0.2.0",
    note = "use UnifiedGpuBufferPool from src::domain::compute instead"
)]
pub struct SharedComponentGpuManager { ... }
```

- Add wrapper methods calling `UnifiedGpuBufferPool`
- Document migration path in module docs

**Estimated Effort**: 1 hour

---

#### 4.2 Deprecate GpuSharedOpsContext
**File**: `src/domain/layers/components/gpu_shared_ops.rs`

**Changes**:
- Similar deprecation warnings
- Wrapper methods to `UnifiedGpuBufferPool`
- Update batch executor to use unified pool

**Estimated Effort**: 1 hour

---

#### 4.3 Fix ForwardContext Fields
**File**: `src/application/training/forward_context.rs`

**Changes**:
- Add missing `low_rank_query_gate` field
- Ensure GPU initialization sets device properly

**Estimated Effort**: 0.5 hours

---

#### 4.4 PolyAttention Cleanup
**File**: `src/domain/attention/poly_attention.rs`

**Changes**:
- Remove orphaned kernel code
- Fix syntax errors
- Update kernel invocation to use unified pool

**Estimated Effort**: 1.5 hours

---

**Session 4 Subtotal**: ~4 hours
**Expected State**: Legacy managers deprecated, no compilation warnings, orphaned code removed

---

## Session 5: Final Validation & Benchmarking (Day 7)

### Goal
Comprehensive testing and performance validation.

### Tasks

#### 5.1 Full Test Suite
```bash
# All 529 tests must pass
cargo test --lib
# GPU-specific tests
cargo test --lib --features gpu-wgpu
# Integration tests
cargo test --test unified_gpu_buffer_pool_integration
cargo test --test shared_component_gpu_integration
cargo test --test block_gpu_forward_integration
```

**Estimated Effort**: 2 hours (running tests)

---

#### 5.2 Benchmarking
**File**: `benches/gpu_performance_baseline.rs` (if doesn't exist)

**Benchmarks**:
```rust
// GPU vs CPU single batch
bench_diffusion_block_gpu_vs_cpu()
bench_transformer_block_gpu_vs_cpu()
bench_temporal_mixing_gpu_vs_cpu()

// Batch processing
bench_diffusion_block_gpu_batch_32()
bench_transformer_block_gpu_batch_32()
```

**Expected Results**:
- GPU should be faster on large batches (batch ≥ 16)
- Memory efficiency within 90% of peak

**Estimated Effort**: 1.5 hours

---

#### 5.3 Documentation Update
**Files**:
- `GPU_BACKEND_IMPLEMENTATION_STATUS.md` - Update phase completion
- `CONSOLIDATION_GPU_IMPLEMENTATION_PHASE5.5.md` - Final summary

**Estimated Effort**: 1 hour

---

#### 5.4 Commit & Summary
**Actions**:
- All changes committed with clear messages
- Phase 5.5 marked complete
- Next session priorities documented

**Estimated Effort**: 1 hour

---

**Session 5 Subtotal**: ~5.5 hours
**Expected State**: All tests pass, benchmarks established, documentation complete

---

## Daily Timeline

| Day | Session | Tasks | Hours | State |
|-----|---------|-------|-------|-------|
| Fri 2/14 | 1 | UnifiedPool + GpuComponent | 6.5 | Infrastructure ✓ |
| Mon 2/17 | 2 | Shared component migration | 9.5 | Components ✓ |
| Tue 2/18 | 3 | Block GPU implementations | 7.5 | Blocks ✓ |
| Wed 2/19 | 4 | Deprecation & cleanup | 4 | Cleanup ✓ |
| Thu 2/20 | 5 | Validation & benchmarks | 5.5 | Complete ✓ |

**Total Effort**: ~33 hours  
**Expected Completion**: February 20, 2026

---

## Critical Path & Dependencies

```
1. UnifiedGpuBufferPool (BLOCKER)
   ↓
2. GpuComponent trait
   ├→ 3. Shared component migration
   │  ├→ 4. DiffusionBlock GPU
   │  ├→ 5. SSM kernel stubs
   │  └→ 6. TransformerBlock verify
   └→ 7. Deprecation warnings
   ↓
8. Full test suite
   ↓
9. Benchmarking & docs
```

**Critical**: Tasks 1-2 must complete before 3-7 can proceed.

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Build timeout | HIGH | MEDIUM | Kill cargo, check for loops |
| GPU test failures | MEDIUM | HIGH | Run CPU-only first, then GPU |
| Numerical differences | LOW | HIGH | Establish ε=1e-4 tolerance upfront |
| Memory leaks | MEDIUM | HIGH | Add heap profiling to benchmarks |
| Component API conflicts | MEDIUM | MEDIUM | Wrapper methods + deprecation |

---

## Checklist for Phase Completion

### Infrastructure
- [ ] UnifiedGpuBufferPool created and tested
- [ ] GpuComponent trait defined and implemented
- [ ] compute/mod.rs exports new modules

### Shared Components
- [ ] SharedAttentionContext implements GpuComponent
- [ ] SharedFeedforward implements GpuComponent
- [ ] TemporalMixingLayer implements GpuComponent
- [ ] PolyAttention implements GpuComponent

### Block Implementations
- [ ] DiffusionBlock has forward_gpu()
- [ ] SSM kernels stubbed in .wgsl files
- [ ] TransformerBlock full GPU chain verified

### Cleanup
- [ ] SharedComponentGpuManager deprecated
- [ ] GpuSharedOpsContext deprecated
- [ ] ForwardContext fields fixed
- [ ] PolyAttention orphaned code removed

### Validation
- [ ] All 529 tests pass
- [ ] GPU-specific tests pass
- [ ] Benchmarks show expected speedup
- [ ] No deprecation warnings in build
- [ ] No memory leaks detected

---

## Next Steps After Phase 5.5

### Phase 5.6 (Immediate)
- Implement actual WGSL kernels for Mamba scan
- Implement actual WGSL kernels for RG-LRU
- Benchmark SSM on GPU

### Phase 6 (Remove Deprecations)
- Delete SharedComponentGpuManager entirely
- Delete GpuSharedOpsContext entirely
- Make UnifiedGpuBufferPool the only pool

### Performance Optimization Phase
- Kernel fusion (GEMM + Activation)
- Async GPU execution
- Multi-GPU support
- Mixed precision (FP16)

---

## References

- **Main Plan**: CONSOLIDATION_GPU_IMPLEMENTATION_PHASE5.5.md
- **Previous Session**: CONSOLIDATION_GPU_BACKEND_SESSION_SUMMARY.md
- **Build Guide**: AGENTS.md
- **Status Tracker**: GPU_BACKEND_IMPLEMENTATION_STATUS.md
