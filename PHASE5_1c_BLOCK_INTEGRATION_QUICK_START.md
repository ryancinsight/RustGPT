# Phase 5.1c Quick Start: Block Integration

**Session**: 5.1c - Transform/Diffusion Block Integration  
**Estimated Duration**: 2-3 hours  
**Prerequisite**: Phase 5.1b complete (all SSM forward_into() methods implemented) ✅

---

## Overview

Integrate the completed `forward_into()` methods into `TransformerBlock` and `DiffusionBlock` inference paths to eliminate redundant allocations and achieve 5-8% per-layer speedup.

---

## What's Already Done ✅

- ✅ All temporal mixing variants have `forward_into()`
- ✅ Feedforward layers have `forward_into()`
- ✅ TemporalMixingLayer dispatch updated
- ✅ 504 tests passing
- ✅ Zero regressions

---

## Task 1: TransformerBlock Integration (30-45 min)

**File**: `src/domain/layers/transformer/block.rs`

### Current Pattern (Allocates Output)
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ... normalization
    let temporal_out = self.temporal_mixing.forward(norm1_out);  // ❌ Allocation
    // ... residuals, ffn, etc.
}
```

### New Pattern (Reuses Buffer)
```rust
fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let (seq_len, embed_dim) = input.dim();
    
    // Ensure workspace capacity
    self.unified_workspace.ensure_capacity(seq_len, 1, embed_dim);
    
    // Normalization
    let mut norm1_out = self.unified_workspace.norm1_out.take().unwrap();
    norm1_out = self.pre_attention_norm.forward(&input); // ✅ Reused
    
    // Temporal mixing (ZERO ALLOCATION)
    let mut temporal_out = self.unified_workspace.temporal_out.take().unwrap();
    self.temporal_mixing.forward_into(&norm1_out, &mut temporal_out)?;
    
    // Continue with residuals and FFN using pre-allocated buffers...
    
    Ok(())
}
```

### Implementation Checklist
- [ ] Add `forward_into()` method signature
- [ ] Validate input/output dimensions
- [ ] Extract pre-allocated buffers from `unified_workspace`
- [ ] Call `temporal_mixing.forward_into()` instead of `forward()`
- [ ] Call `feedforward.forward_into()` instead of `forward()`
- [ ] Write final result to `output` buffer
- [ ] Add 3-4 tests (basic, dimension validation, equivalence)

### Expected Memory Savings
- Per-call: 1-2 allocations eliminated (temporal + ffn outputs)
- Per-step: 15-20 KB for typical 256-token sequence

---

## Task 2: DiffusionBlock Integration (30-45 min)

**File**: `src/domain/layers/diffusion/block.rs`

### Same Pattern as TransformerBlock
- Add `forward_into()` with pre-allocated buffers
- Use temporal mixing and feedforward in-place methods
- Leverage `unified_workspace` for intermediate tensors

### Key Differences
- Include FiLM modulation parameters in forward_into
- Handle time embedding computation
- Optional: Cache time-dependent computations

### Expected Memory Savings
- Per-call: 2-3 allocations (temporal + ffn + outputs)
- Per-step: 20-25 KB

---

## Task 3: Integration Tests (20-30 min)

### TransformerBlock Tests to Add
```rust
#[test]
fn test_transformer_block_forward_into_equivalence() {
    // Compare forward() vs forward_into() outputs
    // Check numerical equivalence (< 1e-4)
}

#[test]
fn test_transformer_block_forward_into_dimension_validation() {
    // Wrong output dimensions should fail
    // Correct dimensions should succeed
}

#[test]
fn test_transformer_block_forward_into_memory_efficiency() {
    // Verify no unexpected allocations
    // Check workspace reuse patterns
}
```

### DiffusionBlock Tests
- Similar structure to TransformerBlock tests
- Include time embedding validation
- Verify FiLM modulation correctness

---

## Build & Verify Commands

### Quick Check
```bash
cargo check
```

### Run All Tests
```bash
cargo test --lib 2>&1 | tail -20
```

### Build Release
```bash
cargo build --release
```

### Target: All tests pass, no warnings, no regressions
- Expected test count: 510-520 (from current 504 + new tests)
- Build time: ~3-4 min (release)

---

## Performance Validation (Optional)

### Before/After Comparison
```rust
// Benchmark TransformerBlock inference throughput
// Expected improvement: 5-8% per layer
// With 12 layers: 40-60% total speedup (if bottleneck is memory allocation)
```

### Memory Profiling
```bash
# Optional: Profile heap allocations
# Verify forward_into() path uses pre-allocated buffers
```

---

## Files to Modify

1. **src/domain/layers/transformer/block.rs**
   - Add TransformerBlock::forward_into()
   - Add 3-4 integration tests

2. **src/domain/layers/diffusion/block.rs**
   - Add DiffusionBlock::forward_into()
   - Add 3-4 integration tests

3. **src/domain/layers/components/common.rs** (if needed)
   - Update any dispatch methods

---

## Fallback Strategy

If block-level integration proves complex:
1. Keep forward_into() methods in isolation
2. Mark as "private" internal pattern
3. Use in later optimization phase
4. No regressions - tests still pass

---

## Success Criteria

✅ **Phase 5.1c Complete When**:
1. TransformerBlock::forward_into() implemented and tested
2. DiffusionBlock::forward_into() implemented and tested
3. All 510+ tests passing
4. No new clippy warnings
5. No regressions from Phase 5.1b
6. Release build succeeds in < 5 minutes

---

## Expected Time Breakdown

| Task | Est. Time | Status |
|------|-----------|--------|
| TransformerBlock impl | 30-45 min | Not started |
| DiffusionBlock impl | 30-45 min | Not started |
| Integration tests | 20-30 min | Not started |
| Debugging/Verification | 10-20 min | Not started |
| **Total** | **90-140 min** | **Ready to start** |

---

## Tips for Success

1. **Copy the Pattern**: Use RgLru::forward_into() as a template
2. **Test First**: Write tests before implementation
3. **Memory Safety**: Validate dimensions at entry point
4. **Reuse Buffers**: Always use unified_workspace pre-allocated buffers
5. **Error Handling**: Return Result<()>, not panics
6. **No Allocations**: forward_into() should NOT allocate new tensors
7. **Assignment**: Use .assign() for zero-copy when dimensions match

---

## References

- **Previous Session**: SESSION_SUMMARY_PHASE5_1b_SSM_VARIANTS_FEB13_2026.md
- **Design Doc**: PHASE5_1_EXECUTION_PLAN.md
- **Component Registry**: CONSOLIDATION_COMPONENTS_MANIFEST.md
- **RgLru Example**: src/domain/layers/ssm/rg_lru.rs (lines 736-806)

---

## Notes

- Phase 5.1c is the final piece of the per-layer optimization
- Phase 5.2 will focus on global buffer pooling
- Block integration enables cascade benefits throughout inference pipeline
- Expected 40-60% total speedup for full 12-layer model

**Good luck! Start with TransformerBlock—it's the simpler of the two.**
