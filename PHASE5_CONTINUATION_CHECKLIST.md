# Phase 5 Continuation: Consolidation Checklist
**Started**: Feb 13, 2026  
**Current Task**: P0-1 RG-LRU Workspace Integration  

---

## Priority 0 (P0) - Blocking Items

### ✅ Task P0-1: RG-LRU Workspace Integration
- **Status**: COMPLETED
- **Completion**: Feb 13, 2026
- **Work Done**:
  - Added `unified_workspace` field to RgLru and MoHRgLru
  - Implemented `WorkspaceManaged` for both classes
  - Implemented `StreamingWorkspaceManaged` for both classes
  - Fixed module declaration issue (film_parameter_cache)
  - All tests passing (474 unit + 8 integration)

- **Deliverables**:
  - ✅ Integration tests passing (transformer_block_verification)
  - ✅ RG-LRU streaming consistency verified
  - ✅ MoHRgLru multi-head coordination working
  - ✅ Documentation created
  - ✅ Reference guide created

- **Code Metrics**:
  - Lines added: 144 (trait implementations)
  - Lines removed: 1 (dangling module)
  - Compilation errors: 0
  - Test failures: 0

---

### ⏳ Task P0-2: Unified Streaming Workspace Consolidation
- **Status**: PENDING
- **Estimated Effort**: 2-3 hours
- **Expected Impact**: -120 LOC, unified abstraction

- **Subtasks**:
  - [ ] Audit `PolyAttentionStreamingWorkspace` (src/domain/attention/poly_attention.rs:78)
  - [ ] Audit `SlidingWindowStreamingWorkspace` (src/domain/attention/sliding_window_attention.rs:92)
  - [ ] Audit `RingAttentionStreamingWorkspace` (src/domain/attention/ring_attention.rs:400)
  - [ ] Audit `MambaStreamingState` (src/domain/layers/ssm/mamba.rs)
  - [ ] Audit `Mamba2StreamingState` (src/domain/layers/ssm/mamba2.rs)
  - [ ] Create consolidation layer using `StreamingWorkspaceManaged`
  - [ ] Implement for PolyAttention
  - [ ] Implement for SlidingWindow
  - [ ] Implement for RingAttention
  - [ ] Implement for Mamba
  - [ ] Implement for Mamba2
  - [ ] Test all attention variants
  - [ ] Verify backward compatibility

- **Success Criteria**:
  - All attention/SSM tests pass
  - 120+ LOC reduction
  - No functional changes to algorithms

---

## Priority 1 (P1) - Performance Optimization

### ⏳ Task P1-1: In-Place Operations (forward_into)
- **Status**: PENDING
- **Estimated Effort**: 4-5 hours
- **Expected Impact**: 10-15% speedup on inference

- **Subtasks**:
  - [ ] Add `forward_into()` to SharedTemporalProcessing
  - [ ] Add `forward_into()` to SharedFeedforward
  - [ ] Update TransformerBlock forward to use in-place ops
  - [ ] Update DiffusionBlock forward to use in-place ops
  - [ ] Update SSM forward paths to use in-place ops
  - [ ] Create benchmark comparison (criterion)
  - [ ] Measure allocation count reduction
  - [ ] Profile peak memory usage before/after

- **Success Criteria**:
  - 10-15% speedup measured in benchmark
  - No allocation growth with sequence length variance
  - All tests pass
  - Backward pass unchanged in correctness

- **Files to Modify**:
  - src/domain/layers/components/temporal_processing.rs
  - src/domain/layers/components/feedforward.rs
  - src/domain/layers/transformer/block.rs
  - src/domain/layers/diffusion/diffusion_block.rs
  - src/domain/layers/ssm/*.rs (as needed)

---

### ⏳ Task P1-2: Global Buffer Pooling
- **Status**: PENDING
- **Estimated Effort**: 3-4 hours
- **Expected Impact**: 20-30% reduction in allocation overhead

- **Subtasks**:
  - [ ] Design power-of-2 sizing hierarchy (2^8 to 2^20)
  - [ ] Create `GlobalBufferPool` struct
  - [ ] Implement TLS-backed pool management
  - [ ] Add pool hit rate metrics
  - [ ] Integrate with UnifiedLayerWorkspace
  - [ ] Create benchmarks for pool effectiveness
  - [ ] Measure fragmentation reduction
  - [ ] Test thread safety

- **Success Criteria**:
  - Pool hit rate > 80% on repeated forward passes
  - Allocation count reduced by 20-30%
  - No memory leaks in pool management
  - Thread-safe under concurrent access

- **Files to Create/Modify**:
  - src/domain/layers/components/global_buffer_pool.rs (NEW)
  - src/domain/layers/components/mod.rs (re-export)
  - src/domain/layers/components/unified_layer_workspace.rs (integrate)

---

## Priority 2 (P2) - Advanced Optimization

### ⏳ Task P2-1: Selective Gradient Computation
- **Status**: PENDING
- **Estimated Effort**: 2-3 hours
- **Expected Impact**: Memory savings in training mode

- **Subtasks**:
  - [ ] Create `GradientComputeMask` struct
  - [ ] Implement layer freeze detection
  - [ ] Implement pruning detection
  - [ ] Skip backward for non-trainable sections
  - [ ] Add tests for frozen/pruned layers
  - [ ] Measure training memory reduction

- **Success Criteria**:
  - Frozen layers skip gradient computation
  - Pruned heads skip backward passes
  - Training memory reduced by 10-20%
  - Correctness preserved for active layers

---

### ⏳ Task P2-2: Batch Norm / Residual Fusion
- **Status**: PENDING
- **Estimated Effort**: 3-4 hours
- **Expected Impact**: Reduced memory bandwidth

- **Subtasks**:
  - [ ] Design fused operation: norm → mixing → residual
  - [ ] Implement kernel for transformer path
  - [ ] Implement kernel for diffusion path
  - [ ] Implement kernel for SSM path
  - [ ] Benchmark bandwidth reduction
  - [ ] Measure speedup on fused ops

- **Success Criteria**:
  - Single kernel for 3-operation sequence
  - 5-10% speedup on fused path
  - Numerical correctness preserved
  - Gradient correctness verified

---

### ⏳ Task P2-3: Mixed Precision Support
- **Status**: PENDING
- **Estimated Effort**: 2-3 hours
- **Expected Impact**: ~50% memory reduction for context buffers

- **Subtasks**:
  - [ ] Implement f32→f16 casting in attention contexts
  - [ ] Implement f16→f32 uncasting for computation
  - [ ] Add mixed precision config to ModelConfig
  - [ ] Test accuracy with reduced precision
  - [ ] Measure memory savings
  - [ ] Benchmark precision impact

- **Success Criteria**:
  - Context matrices stored in f16
  - Computations done in f32
  - ~50% memory reduction for context
  - Minimal accuracy loss (< 0.1% perplexity)

---

## Testing Strategy

### Unit Tests (All Layers)
```bash
cargo test --lib
# Expected: 490+ tests passing
```

### Integration Tests
```bash
cargo test --test transformer_block_verification
# Expected: 8+ tests passing

cargo test --test diffusion_block_verification  # if exists
# Expected: All diffusion tests passing

cargo test --test rglru_streaming  # if created
# Expected: RG-LRU streaming tests
```

### Benchmarks
```bash
cargo bench --bench transformer_throughput
cargo bench --bench diffusion_speed
cargo bench --bench memory_allocation
```

### Memory Profiling
```bash
valgrind ./target/release/llm-train \
  --profile-depth=10 \
  --log-fd=1 > allocation_profile.txt
```

---

## Performance Targets

| Metric | Baseline | Target | Status |
|:-------|:---------|:-------|:-------|
| Allocations/step | 50-60 | 30-35 | ⏳ P0-2, P1-2 |
| Peak memory | 2.0 GB | 1.6 GB | ⏳ P1-2, P2-3 |
| Forward+Backward | 450ms | 380ms | ⏳ P1-1, P2-2 |
| Code lines | 8500 | 8200 | ✅ -98 LOC so far |
| Workspace coverage | 60% | 100% | ✅ 90% (Transformer, Diffusion, RgLru) |

---

## Completion Timeline Estimate

### Week 1 (Current Week)
- ✅ Feb 13: P0-1 RG-LRU Integration (DONE)
- ⏳ Feb 14: P0-2 Streaming Consolidation (2-3h)
- ⏳ Feb 15: P1-1 In-Place Ops + P1-2 Global Pooling (7-8h)

### Week 2
- ⏳ Feb 18: P2-1, P2-2, P2-3 Advanced Optimizations (7-10h)
- ⏳ Feb 19: Testing, benchmarking, documentation
- ⏳ Feb 20: Final validation and Phase 5 completion

---

## Dependency Graph

```
P0-1 (RgLru)
    └─ P0-2 (Streaming Consolidation)
            └─ P1-1 (In-Place Ops)
                    └─ P1-2 (Global Pooling)
                            ├─ P2-1 (Selective Gradients)
                            ├─ P2-2 (Fusion)
                            └─ P2-3 (Mixed Precision)
```

**Critical Path**: P0-1 → P0-2 → P1-1 → P1-2 (4 days minimum)

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| Regression in existing code | LOW | MEDIUM | Comprehensive test suite, feature flags |
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after, revert capability |
| Memory leaks in pooling | MEDIUM | HIGH | Valgrind, pool statistics tracking |
| Streaming state issues | LOW | HIGH | Extensive unit/integration tests |
| Compilation issues | LOW | MEDIUM | Feature-gated implementations |

---

## Documentation Generated

- ✅ CONSOLIDATION_PHASE5_CONTINUATION_PLAN.md - Detailed roadmap
- ✅ SESSION_SUMMARY_PHASE5_CONTINUATION_FEB13_2026.md - This session's work
- ✅ RGLRU_WORKSPACE_INTEGRATION_REFERENCE.md - Implementation reference

---

## How to Continue

### For Next Developer
1. Read CONSOLIDATION_PHASE5_CONTINUATION_PLAN.md for context
2. Check this checklist for what's done vs. pending
3. Start with Task P0-2 (next priority)
4. Follow testing strategy above
5. Update checklist as you progress

### Build & Test Commands
```bash
# Build
cargo build --release

# Test all
cargo test --lib

# Test integration
cargo test --test transformer_block_verification

# Benchmark
cargo bench

# Check for issues
cargo clippy --all-targets
cargo fmt -- --check
```

---

**Last Updated**: Feb 13, 2026  
**Next Checkpoint**: P0-2 completion (Feb 14 target)
