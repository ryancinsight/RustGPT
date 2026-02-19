# Phase 5: Streaming Workspace Unification & Performance Consolidation

**Status**: In Progress  
**Date**: Feb 13, 2026  
**Focus**: Consolidate remaining 5+ streaming workspace implementations into `StreamingWorkspaceManaged` trait  

## Executive Summary

This phase unifies all streaming workspace implementations (PolyAttention, SlidingWindow, RingAttention, Mamba, Mamba2) into the `StreamingWorkspaceManaged` trait established in Phase 5.1. This eliminates redundant code, improves maintainability, and enables consistent performance optimizations across all stateful layers.

---

## Tasks (P0 - Critical Path)

### Task 1: PolyAttention Consolidation
**File**: `src/domain/attention/poly_attention.rs`  
**Current State**: Has custom `PolyAttentionStreamingWorkspace` struct, no `StreamingWorkspaceManaged` impl  
**Action**:
- [ ] Add `StreamingWorkspaceManaged` trait implementation to `PolyAttention`
- [ ] Consolidate `PolyAttentionStreamingWorkspace` buffers into `UnifiedLayerWorkspace` (optional buffers)
- [ ] Update `forward_streaming()` to use workspace trait methods
- [ ] Update initialization path to call `init_streaming()`
- [ ] Validate: All PolyAttention tests pass

**Expected Impact**: -40 LOC, +1 trait impl (10 lines)

---

### Task 2: SlidingWindowAttention Consolidation
**File**: `src/domain/attention/sliding_window_attention.rs`  
**Current State**: Has `SlidingWindowStreamingWorkspace`, no trait impl  
**Action**:
- [ ] Add `StreamingWorkspaceManaged` trait implementation
- [ ] Consolidate buffers into workspace pattern
- [ ] Validate: All tests pass

**Expected Impact**: -30 LOC, +1 trait impl

---

### Task 3: RingAttention Consolidation
**File**: `src/domain/attention/ring_attention.rs`  
**Current State**: Has `RingAttentionStreamingWorkspace`, no trait impl  
**Action**:
- [ ] Add `StreamingWorkspaceManaged` trait implementation
- [ ] Consolidate block-sized buffers
- [ ] Validate: All tests pass

**Expected Impact**: -35 LOC, +1 trait impl

---

### Task 4: Mamba Consolidation
**File**: `src/domain/layers/ssm/mamba.rs`  
**Current State**: Disparate optional fields (`streaming_workspace`, `streaming_ssm_state`, `streaming_conv_queue`)  
**Action**:
- [ ] Create unified `MambaStreamingWorkspace` struct consolidating all three fields
- [ ] Add `StreamingWorkspaceManaged` trait implementation
- [ ] Update `forward_streaming()` and inference paths
- [ ] Validate: All tests pass

**Expected Impact**: -50 LOC, +1 unified struct, +1 trait impl

---

### Task 5: Mamba2 Consolidation
**File**: `src/domain/layers/ssm/mamba2.rs`  
**Current State**: Has `MoHMamba2StreamingWorkspace`, no trait impl  
**Action**:
- [ ] Add `StreamingWorkspaceManaged` trait implementation
- [ ] Consolidate workspace pattern
- [ ] Validate: All tests pass

**Expected Impact**: -25 LOC, +1 trait impl

---

## Tasks (P1 - Performance Optimizations)

### Task 6: In-Place Operations (Forward-into variants)
**Priority**: P1  
**Files**: 
- `src/domain/attention/poly_attention.rs`
- `src/domain/layers/components/shared_feedforward.rs`

**Action**:
- [ ] Implement `forward_into()` variants for PolyAttention scoring
- [ ] Implement `forward_into()` for SharedFeedforward
- [ ] Add benchmarks comparing allocating vs in-place variants
- [ ] Target: 10-15% speedup on inference

**Expected Impact**: +3-5% overall inference speedup

---

### Task 7: Global Buffer Pooling Integration
**Priority**: P1  
**Files**:
- `src/domain/layers/components/mod.rs`
- Create: `src/domain/layers/components/global_buffer_pool.rs`

**Action**:
- [ ] Design `GlobalBufferPool` trait with power-of-2 sizing
- [ ] Implement pooling for workspace allocations
- [ ] Integrate with `UnifiedLayerWorkspace`
- [ ] Benchmark heap fragmentation before/after
- [ ] Target: -20% memory usage, -5% allocation time

**Expected Impact**: -20% peak memory, improved cache locality

---

## Tasks (P2 - Advanced)

### Task 8: Selective Gradient Computation
**Priority**: P2  
**Action**:
- [ ] Design `GradientComputeMask` for frozen layer skipping
- [ ] Implement in attention and feedforward components
- [ ] Add tests validating zero-gradient propagation

**Expected Impact**: -10% training time for large models with frozen layers

---

### Task 9: Batch Norm Fusion
**Priority**: P2  
**Action**:
- [ ] Fuse norm + residual operations into single pass
- [ ] Implement for TransformerBlock and DiffusionBlock
- [ ] Benchmark memory bandwidth improvements

**Expected Impact**: -5-10% forward/backward time

---

## Memory & Performance Targets

| Metric | Baseline | Target | Improvement |
|--------|----------|--------|-------------|
| Allocations per step | 50-60 | 30-35 | -40% |
| Peak Memory (1B) | 2.0 GB | 1.6 GB | -20% |
| Forward + Backward | 450ms | 380ms | -15% |
| Code Lines (layers/) | 8500 | 8200 | -300 LOC |

---

## Completion Checklist

### Phase 5.1 Workspace Unification (Completed)
- [x] TransformerBlock integration
- [x] DiffusionBlock integration  
- [x] RgLru integration with `StreamingWorkspaceManaged`
- [x] `UnifiedLayerWorkspace` implementation

### Phase 5.2 Streaming Consolidation (Current)
- [ ] PolyAttention `StreamingWorkspaceManaged` impl
- [ ] SlidingWindowAttention consolidation
- [ ] RingAttention consolidation
- [ ] Mamba unified streaming workspace
- [ ] Mamba2 consolidation
- [ ] Integration test: All streaming layers work end-to-end

### Phase 5.3 In-Place Operations (Planned)
- [ ] Attention forward_into variants
- [ ] Feedforward forward_into variants
- [ ] Gradient computation in-place
- [ ] Benchmarks showing 10-15% speedup

### Phase 5.4 Global Buffer Pooling (Planned)
- [ ] GlobalBufferPool trait design
- [ ] Integration with workspace management
- [ ] Benchmarks showing -20% memory usage

### Phase 5.5 Advanced Optimizations (Future)
- [ ] Selective gradient computation
- [ ] Batch norm fusion
- [ ] Mixed precision for historical matrices

---

## Key Design Principles

1. **Zero-Cost Abstraction**: Trait overhead should be eliminated at compile time
2. **Lazy Allocation**: Buffers allocated only when needed
3. **Reuse-First**: Reuse allocated buffers when shape is unchanged
4. **Power-of-2 Sizing**: Minimize reallocations and heap fragmentation
5. **Contiguous Memory**: Prefer single large allocation over scattered buffers

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking streaming layer behavior | Add comprehensive integration tests for each layer |
| Memory leaks in workspace reuse | Add property tests verifying buffer clearing |
| Compilation time regression | Monitor build times, profile clippy warnings |
| Performance regression | Benchmark before/after each optimization |

---

## Next Steps

1. Implement PolyAttention `StreamingWorkspaceManaged` (Task 1)
2. Validate with integration tests
3. Consolidate remaining attention variants (Tasks 2-3)
4. Consolidate SSM layers (Tasks 4-5)
5. Implement in-place operations (Task 6)
6. Integrate global buffer pooling (Task 7)
