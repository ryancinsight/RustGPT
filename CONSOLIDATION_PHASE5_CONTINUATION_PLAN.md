# Phase 5 Continuation: Consolidation & Memory Optimization Plan
**Status**: ACTIVE  
**Date**: Feb 13, 2026  
**Focus**: RG-LRU Workspace Integration + In-Place Ops + Buffer Pooling

---

## Executive Summary

This document outlines the remaining high-impact consolidation tasks for Phase 5, continuing from previous work on `TransformerBlock` and `DiffusionBlock` workspace integration. The goal is to unify buffer management across all layer types (Transformer, Diffusion, SSM/RG-LRU/Mamba) while eliminating redundant allocations and improving performance.

**Expected Impact**:
- **Memory**: Reduce allocations per step by 40% (50-60 → 30-35)
- **Performance**: 10-15% speedup on inference
- **Code**: 300+ LOC reduction in `src/domain/layers/`

---

## Current State (As of Feb 13, 2026)

### ✅ Completed
- `TransformerBlock` ↔ `UnifiedLayerWorkspace` integration
- `DiffusionBlock` ↔ `UnifiedLayerWorkspace` integration
- `WorkspaceManaged` trait implementation in both blocks
- All 490 unit tests passing
- All 8 integration tests (transformer_block_verification) passing
- Compilation succeeds (cargo build --release)

### ⏳ In Progress / Remaining

#### Priority 1 (P0): Blocking Items
1. **RG-LRU Workspace Integration** (~3-4 hours)
   - Integrate `UnifiedLayerWorkspace` into `RgLru` struct
   - Implement `StreamingWorkspaceManaged` trait for RG-LRU
   - Replace manual streaming allocations with workspace buffers
   - Test both single-head and MoH variants
   - Expected impact: -80 LOC, unified SSM memory management

2. **Unified Streaming Workspace Consolidation** (~2-3 hours)
   - Consolidate 5+ streaming workspace types (RgLru, Mamba, PolyAttention, SlidingWindow, RingAttention)
   - Create `StreamingWorkspaceManaged` impl blocks for each
   - Define clear lifecycle (init → step → reset)
   - Expected impact: -120 LOC, clearer abstraction

#### Priority 2 (P1): Performance Optimization
3. **In-Place Operations (forward_into)** (~4-5 hours)
   - `SharedTemporalProcessing::forward_into()` for attention/mixing
   - `SharedFeedforward::forward_into()` for FFN layers
   - Eliminate intermediate allocations in hot paths
   - Profile before/after using criterion benchmarks
   - Expected speedup: 10-15% on inference

4. **Global Buffer Pooling** (~3-4 hours)
   - Consolidate `IntermediateBufferPool` + workspace pools into `GlobalBufferPool`
   - Implement power-of-2 sizing hierarchy
   - Add TLS-backed pooling for streaming ops
   - Measure heap fragmentation reduction
   - Expected impact: 20% reduction in allocation overhead headers

#### Priority 3 (P2): Advanced Features
5. **Selective Gradient Computation** (~2-3 hours)
   - Create `GradientComputeMask` for frozen/pruned layers
   - Skip unnecessary backward passes in non-trainable sections
   - Measure memory savings in training mode

6. **Batch Norm / Residual Fusion** (~3-4 hours)
   - Fuse normalization + residual + mixing into single ops
   - Reduce memory bandwidth and intermediate buffers
   - Benchmark on transformer training

7. **Mixed Precision Support** (~2-3 hours)
   - Keep f32 for activations, f16 for historical context matrices
   - Implement automatic cast/uncast in attention contexts
   - Measure memory reduction (~50% for context buffers)

---

## Task Breakdown

### Task P0-1: RG-LRU Workspace Integration

**Files to Modify**:
- `src/domain/layers/ssm/rg_lru.rs` - Main implementation
- `src/domain/layers/ssm/mod.rs` - Re-exports if needed
- `src/domain/layers/components/workspace_managed.rs` - Already has trait

**Implementation Steps**:

1. **Add `unified_workspace` field to `RgLru`**
   ```rust
   pub struct RgLru {
       // ... existing fields ...
       pub unified_workspace: Option<UnifiedLayerWorkspace>,
   }
   ```

2. **Implement `WorkspaceManaged` for `RgLru`**
   - Delegate `ensure_capacity()` to `unified_workspace`
   - Implement buffer access methods for streaming state
   - Clear cache fields alongside workspace

3. **Implement `StreamingWorkspaceManaged` for `RgLru`**
   - Manage `streaming_state` and `context_buffer` lifecycle
   - Initialize at first token, preserve across steps
   - Reset on new sequence

4. **Test Coverage**:
   - Single-step forward/backward
   - Multi-step streaming consistency
   - Memory reuse across sequences
   - MoHRgLru multi-head coordination

**Success Criteria**:
- ✅ All existing RG-LRU tests pass
- ✅ Streaming workspace properly reused
- ✅ No additional allocations per step after first

---

### Task P0-2: Unified Streaming Workspace Consolidation

**Affected Streaming Workspaces**:
1. `RgLruStreamingWorkspace` (rg_lru.rs:18)
2. `MoHRgLruStreamingWorkspace` (rg_lru.rs:28)
3. `MambaStreamingState` (mamba.rs)
4. `PolyAttentionStreamingWorkspace` (poly_attention.rs:78)
5. `SlidingWindowStreamingWorkspace` (sliding_window_attention.rs:92)
6. `RingAttentionStreamingWorkspace` (ring_attention.rs:400)

**Implementation Steps**:

1. **Audit Current Patterns**
   - Document buffer shapes, allocation sizes, lifecycle
   - Identify commonalities (all are Option<Array1/2>)
   - Note special cases (PolyAttention's `with_exact_capacity`, RingAttention's blocking)

2. **Create Consolidation Layer**
   - Extend `StreamingWorkspaceManaged` trait
   - Add methods: `init_streaming()`, `step_forward()`, `reset_streaming()`
   - Define standard error handling

3. **Implement for Each Type**
   - Minimize changes to core computation logic
   - Wrap allocation logic in trait methods
   - Add metrics collection (bytes allocated, step count)

4. **Test Coverage**:
   - Before/after equivalence tests
   - Memory profile comparisons
   - Benchmark regression detection

**Success Criteria**:
- ✅ All 8 transformer_block_verification tests pass
- ✅ All diffusion and SSM tests pass
- ✅ No functional changes to core algorithms
- ✅ 120+ LOC reduction

---

### Task P1-1: In-Place Operations (forward_into)

**Files to Modify**:
- `src/domain/layers/components/shared_temporal_processing.rs`
- `src/domain/layers/components/shared_feedforward.rs`
- Attention implementations (poly, sliding, ring)

**Implementation Steps**:

1. **Add `forward_into()` Methods**
   - `SharedTemporalProcessing::forward_into(input, output_buf) -> Result`
   - `SharedFeedforward::forward_into(input, output_buf) -> Result`
   - Accepts pre-allocated output buffer to avoid new allocation

2. **Profile Hot Paths**
   - Use `cargo bench` to identify top allocators
   - Measure allocation count before/after
   - Compare peak memory usage

3. **Update Call Sites**
   - TransformerBlock's forward pass
   - DiffusionBlock's forward pass
   - SSM forward paths

4. **Benchmark Coverage**:
   - `cargo bench transformer_throughput`
   - `cargo bench diffusion_inference_speed`
   - Memory profiler (cargo-valgrind or custom)

**Success Criteria**:
- ✅ 10-15% speedup on forward pass
- ✅ No allocation count growth with sequence length variance
- ✅ Backward pass unchanged in correctness

---

### Task P1-2: Global Buffer Pooling

**Files to Create/Modify**:
- `src/domain/layers/components/global_buffer_pool.rs` (NEW)
- `src/domain/layers/components/mod.rs` - Re-export
- `src/domain/layers/components/workspace_managed.rs` - Trait extensions

**Implementation Steps**:

1. **Design Power-of-2 Hierarchy**
   - Size buckets: 2^8, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20
   - Round up allocations to nearest bucket
   - Maintain per-bucket pool

2. **Implement GlobalBufferPool**
   - TLS-backed pools for thread safety
   - Lazy initialization
   - Metrics: pool hit rate, fragmentation, total allocated

3. **Integrate with UnifiedLayerWorkspace**
   - Use pool for large allocations (> 1KB)
   - Bypass pool for tiny/temporary allocations
   - Measure allocation overhead reduction

4. **Benchmark**:
   - Allocation count in typical inference
   - Heap fragmentation before/after
   - Cache locality improvement

**Success Criteria**:
- ✅ Allocation count reduced by 20-30%
- ✅ Pool hit rate > 80% on repeated forward passes
- ✅ No memory leaks in pool management

---

## Performance Targets

| Metric | Baseline | Target | Improvement |
|:-------|:---------|:-------|:------------|
| Allocations per step | 50-60 | 30-35 | -40% |
| Peak memory (1 batch) | 2.0 GB | 1.6 GB | -20% |
| Forward+Backward time | 450ms | 380ms | -15% |
| Code lines (layers/) | 8500 | 8200 | -300 LOC |

---

## Testing Strategy

### Unit Tests
- Verify each workspace integration
- Check buffer reuse (same pointer across steps)
- Validate capacity calculations

### Integration Tests
- Streaming consistency (transformer_block_verification)
- Diffusion forward/backward correctness
- SSM BPTT correctness (new)

### Performance Tests
```bash
cargo bench --bench transformer_throughput
cargo bench --bench diffusion_speed
cargo test --lib  # All unit tests
cargo test --test transformer_block_verification
cargo test --test diffusion_block_verification  # if exists
```

### Memory Profiling
- Before: `cargo build --release && valgrind ./target/release/...`
- After: Same command, compare output
- Check for allocation count growth

---

## Risk Assessment

### Low Risk ✅
- RG-LRU workspace integration (isolated component)
- Streaming consolidation (wraps existing logic)

### Medium Risk ⚠️
- In-place ops (requires careful buffer lifecycle management)
- Global buffer pooling (TLS state, synchronization)

### Mitigation
- Feature flags to enable/disable new pooling
- Gradual rollout: enable for inference first, then training
- Extensive regression testing

---

## Timeline Estimate

| Task | Effort | Days | Priority |
|:-----|:-------|:-----|:---------|
| P0-1: RG-LRU Integration | 3-4h | 1 | High |
| P0-2: Streaming Consolidation | 2-3h | 1 | High |
| P1-1: In-Place Ops | 4-5h | 2 | Medium |
| P1-2: Global Buffer Pooling | 3-4h | 1-2 | Medium |
| P2 Tasks | 7-10h | 2-3 | Low |
| **Total** | **19-26h** | **5-7 days** | |

**Recommended Daily Allocation**: 4-5 hours of focused development

---

## Success Metrics (Phase 5 Complete)

- ✅ All layer types (Transformer, Diffusion, SSM) use unified workspace
- ✅ 40% reduction in allocations per step
- ✅ 15% speedup on inference
- ✅ 300+ LOC reduction in layers/
- ✅ All tests passing (490 unit + 8+ integration)
- ✅ No functional regressions
- ✅ Memory profiling shows improvement

---

## References

### Trait Definitions
- `WorkspaceManaged` - [src/domain/layers/components/workspace_managed.rs#L40-L56](file:///d:/RustGPT/src/domain/layers/components/workspace_managed.rs#L40-L56)
- `StreamingWorkspaceManaged` - [src/domain/layers/components/workspace_managed.rs#L92-L116](file:///d:/RustGPT/src/domain/layers/components/workspace_managed.rs#L92-L116)

### Existing Integrations
- `TransformerBlock` - [src/domain/layers/transformer/block.rs](file:///d:/RustGPT/src/domain/layers/transformer/block.rs)
- `DiffusionBlock` - [src/domain/layers/diffusion/diffusion_block.rs](file:///d:/RustGPT/src/domain/layers/diffusion/diffusion_block.rs)

### Streaming Workspaces
- RG-LRU - [src/domain/layers/ssm/rg_lru.rs#L18-L32](file:///d:/RustGPT/src/domain/layers/ssm/rg_lru.rs#L18-L32)
- PolyAttention - [src/domain/attention/poly_attention.rs#L78-L116](file:///d:/RustGPT/src/domain/attention/poly_attention.rs#L78-L116)
- SlidingWindow - [src/domain/attention/sliding_window_attention.rs#L92-L110](file:///d:/RustGPT/src/domain/attention/sliding_window_attention.rs#L92-L110)

---

**Next Action**: Start with Task P0-1 (RG-LRU integration). Estimated completion: 1 day.
