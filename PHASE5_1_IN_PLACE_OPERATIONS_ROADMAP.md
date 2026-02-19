# Phase 5.1: In-Place Operations Implementation Roadmap

**Date**: February 13, 2026  
**Status**: Ready for Implementation  
**Target**: 10-15% per-layer speedup, 40 KB/step memory reduction

---

## Overview

Phase 5.1 focuses on implementing `forward_into()` variants across shared components to eliminate intermediate allocations. This is the highest-impact optimization with measurable performance gains.

### Baseline Metrics
- Current memory per step: 129 KB
- Target memory per step: 89 KB (40 KB reduction)
- Expected latency improvement: 10-15% per layer, 25-30% total model speedup

---

## Implementation Strategy

### Core Principle
Replace allocating forward passes with in-place variants that write directly to pre-allocated output buffers. This eliminates:
- Intermediate Array2 allocations
- Copy operations between layers
- Temporary buffer memory pressure

### Code Pattern (Reference)
```rust
// Current (allocating):
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ... computation ...
    output  // Returns newly allocated buffer
}

// New (in-place):
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    // ... computation written directly to output ...
    Ok(())
}
```

---

## Phase 5.1 Task Breakdown

### Task 1: SharedTemporalProcessing - In-Place Forward
**Files**: `src/domain/layers/components/temporal_processing.rs`  
**Scope**: Implement `forward_into()` wrapper and delegate to underlying layer

**Subtasks**:
1. Add `forward_into()` method to SharedTemporalProcessing
2. Delegate to `temporal_mixing.forward_into()` (through Layer trait)
3. Add test: `test_temporal_forward_into_equivalence()`
4. Verify all temporal mixing variants support forward_into

**Expected Impact**: ~8-10% latency reduction per layer

---

### Task 2: TemporalMixingLayer Variants - In-Place Support
**Files**: 
- `src/domain/attention/poly_attention.rs` (PolyAttention)
- `src/domain/ssm/rg_lru.rs` (RgLru)
- `src/domain/ssm/mamba.rs` (Mamba)
- Other temporal mixing implementations

**Scope**: Implement `forward_into()` for each variant

**Subtasks** (per variant):
1. Add `forward_into()` method signature
2. Eliminate intermediate allocations:
   - Attention: Output written directly (avoid query/key/value temp matrices)
   - RG-LRU: Gate and value computations into output
   - Mamba: SSM output written directly to buffer
3. Reuse workspace buffers where possible
4. Add test: `test_{variant}_forward_into_matches_forward()`

**Expected Impact**: ~5-8% latency reduction per variant

---

### Task 3: SharedFeedforward - In-Place Forward
**Files**: `src/domain/layers/components/feedforward.rs`  
**Scope**: Implement `forward_into()` wrapper

**Subtasks**:
1. Add `forward_into()` method to SharedFeedforward
2. Delegate to `feedforward.forward_into()` (through FeedForwardVariant)
3. Add test: `test_feedforward_forward_into_equivalence()`

**Expected Impact**: ~5-7% latency reduction per layer

---

### Task 4: FeedForwardVariant Implementations
**Files**:
- `src/domain/layers/richards_glu.rs` (RichardsGLU)
- `src/domain/mixtures/moe.rs` (MoE)

**Scope**: Implement in-place forward for each variant

**Subtasks** (per variant):
1. Add `forward_into()` method
2. Eliminate intermediate buffers:
   - RichardsGLU: Hidden layer output written directly
   - MoE: Router output and expert outputs written directly
3. Maximize buffer reuse from workspace
4. Add test: `test_{variant}_forward_into_matches_forward()`

**Expected Impact**: ~4-6% latency reduction per variant

---

### Task 5: Block-Level Integration
**Files**:
- `src/domain/layers/transformer_block.rs`
- `src/domain/layers/diffusion_block.rs`
- `src/domain/layers/ssm_block.rs` (if present)

**Scope**: Integrate in-place operations into block forward passes

**Subtasks**:
1. Modify TransformerBlock::forward to use in-place operations:
   - Pre-attention norm → temporal mixing → residual
   - Pre-FFN norm → feedforward → residual
2. Integrate with UnifiedLayerWorkspace buffer reuse
3. Add test: `test_block_forward_into_matches_forward()`
4. Measure memory and latency improvements

**Expected Impact**: ~10-15% latency reduction per block

---

### Task 6: Validation & Performance Analysis
**Files**: `tests/` directory

**Subtasks**:
1. Run comprehensive equivalence tests (all 476+ library tests must pass)
2. Profile latency improvements:
   - Per-component benchmarks (PHASES timing)
   - Per-layer benchmarks (using `criterion`)
   - End-to-end model benchmarks
3. Verify memory savings:
   - Peak memory usage measurement
   - Allocation count reduction
   - Cache locality improvement analysis
4. Stress test: 1000-step training run without memory leaks

**Success Criteria**:
- ✅ All tests pass
- ✅ 25-30% total model speedup confirmed
- ✅ Memory reduced from 129 KB to ~89 KB/step
- ✅ No regressions in output quality

---

## Implementation Timeline

### Phase 5.1a: Foundation (Feb 14-15)
- [ ] Task 1: SharedTemporalProcessing forward_into
- [ ] Task 3: SharedFeedforward forward_into

### Phase 5.1b: Temporal Mixing Variants (Feb 16-17)
- [ ] Task 2: All temporal mixing layer variants

### Phase 5.1c: Feedforward Variants (Feb 18)
- [ ] Task 4: RichardsGLU and MoE forward_into

### Phase 5.1d: Integration & Testing (Feb 19-20)
- [ ] Task 5: Block-level integration
- [ ] Task 6: Comprehensive validation

---

## Layer Trait Integration

The Layer trait already supports `forward_into()` through this pattern:
```rust
pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    
    // New method added in this phase
    fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        // Default: allocate then copy (backward compatible)
        let result = self.forward(input);
        output.assign(&result);
        Ok(())
    }
}
```

All temporal mixing variants inherit from Layer, so overriding this method provides optimized in-place behavior.

---

## Memory Reuse Pattern

### Pre-Allocated Buffers (from UnifiedLayerWorkspace)
```rust
// In block forward pass:
workspace.ensure_capacity(seq_len, batch_size, embed_dim)?;

// Reuse these buffers:
let norm1_out = workspace.get_norm1_out_mut();
let temporal_out = workspace.get_temporal_out_mut();
let residual1 = workspace.get_residual1_mut();
let norm2_out = workspace.get_norm2_out_mut();
let ffn_out = workspace.get_ffn_out_mut();

// Forward pass reuses buffers:
temporal_layer.forward_into(norm1_out, temporal_out)?;
add_residual_into(temporal_out, residual1, input)?;
// ... continue with FFN ...
```

---

## Risk Mitigation

### Backward Compatibility
- All existing `forward()` methods remain unchanged
- New `forward_into()` is optional (default implementation delegates to forward)
- Existing code continues to work without modification

### Testing Strategy
1. Equivalence tests: Ensure `forward()` ≈ `forward_into()` output
2. Numerical stability: Check for precision loss from increased in-place operations
3. Integration tests: Verify entire block forward pass produces same results
4. Stress tests: 1000+ step training runs to catch memory leaks

### Validation Gates
- ✅ All 476+ library tests must pass before proceeding to next phase
- ✅ 99.9% numerical equivalence (< 1e-5 max element difference)
- ✅ Zero new clippy warnings
- ✅ Benchmark shows ≥ 25% speedup (or investigate bottleneck)

---

## Success Metrics

### Performance Targets
| Metric | Baseline | Target | Achieved |
|--------|----------|--------|----------|
| Inference latency/layer | 100ms | 85-90ms | — |
| Training memory/step | 129 KB | 89 KB | — |
| Allocations/step | 24 | 12 | — |
| Total model speedup | 1x | 1.25-1.3x | — |

### Code Quality Targets
| Metric | Target |
|--------|--------|
| Test coverage | ≥ 99% |
| Clippy warnings | 0 |
| Compilation time | ≤ 180s |
| Rustfmt compliance | 100% |

---

## Next Steps After Phase 5.1

### Phase 5.2: Global Buffer Pooling
- Consolidate IntermediateBufferPool and workspace pools
- Implement GlobalBufferPool with power-of-2 sizing
- Expected additional 10-15% speedup

### Phase 5.3: Advanced Optimizations
- Selective gradient computation (skip frozen layers)
- Batch norm fusion
- Mixed precision (f32 activations, f16 historical context)

---

## References

- **CONSOLIDATION_COMPONENTS_MANIFEST.md**: Component inventory and integration map
- **OPTIMIZATION_PATTERNS_GUIDE.md**: Reusable optimization patterns
- **PHASE5_IMPLEMENTATION_ROADMAP.md**: Full Phase 5 strategy

