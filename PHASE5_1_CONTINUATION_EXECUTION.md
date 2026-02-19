# Phase 5.1 Continuation: SSM Variants & Feedforward In-Place Operations
**Session Start**: February 13, 2026 (Evening)  
**Focus**: Complete SSM variant forward_into() implementations and benchmark  
**Target Progress**: Tasks 3-6 (40% overall completion)

---

## Executive Summary

- **Current Status**: 25% complete (RgLru + SharedFeedforward foundation done)
- **All 490 tests passing** ✅
- **Next 4 hours**: Implement Mamba and Mamba2 forward_into() with in-place support
- **Memory Savings Target**: 15-20 KB/step additional
- **Expected Speedup**: 5-8% per-layer (non-allocation paths only)

---

## Task Breakdown (Immediate Priorities)

### Task 3 & 4: MoH Variants (30 min each)
**Files**: `src/domain/layers/ssm/mamba.rs`, `src/domain/mixtures/moh_gating.rs`  
**Pattern**: Copy RgLru's approach, adapt for Mamba structure

```rust
impl MoHRgLru {
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        // 1. Reuse self.workspace (MoHStreamingWorkspace) for intermediate buffers
        // 2. Call self.forward() to get results
        // 3. Write directly to output buffer
        // 4. Return Ok(())
    }
}
```

**Rationale**: 
- Both MoH variants use streaming workspace pools
- Reusing workspace prevents heap fragmentation
- Tests already validate equivalence (copy from RgLru patterns)

---

### Task 5 & 6: Mamba & Mamba2 (45 min each)
**Files**: `src/domain/layers/ssm/mamba.rs`  
**Challenge**: No streaming workspace yet; need to leverage forward_cached pattern

**Implementation Strategy**:
```rust
impl Mamba {
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        // Reuse forward_cached (which already does in-place style)
        let result = self.forward_cached(input);
        // Direct assignment (no deep clone)
        if output.raw_dim() == result.raw_dim() {
            output.assign(&result);
        } else {
            *output = result;
        }
        Ok(())
    }
}
```

**Key Insight**: Mamba's forward_cached already minimizes allocations; forward_into just provides the interface.

---

## Expected Memory Impact

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| MoHRgLru::forward_into | 2 allocations | 0 | 8-12 KB |
| Mamba::forward_into | 3 allocations | 1 | 10-15 KB |
| Mamba2::forward_into | 3 allocations | 1 | 10-15 KB |
| **Subtotal (12 layers)** | **~120 KB** | **~24 KB** | **~96 KB/step** |

---

## Implementation Checklist

- [ ] **MoHRgLru::forward_into()** (src/domain/layers/ssm/rg_lru.rs lines ~800)
  - [ ] Method signature and docstring
  - [ ] Implementation (reuse workspace pattern)
  - [ ] Test: forward_into_equivalence
  - [ ] Verify 490+ tests pass

- [ ] **Mamba::forward_into()** (src/domain/layers/ssm/mamba.rs lines ~1870)
  - [ ] Method signature and docstring
  - [ ] Implementation (leverage forward_cached)
  - [ ] Test: mamba_forward_into_equivalence
  - [ ] Verify 490+ tests pass

- [ ] **Mamba2::forward_into()** (src/domain/layers/ssm/mamba.rs lines ~2485)
  - [ ] Method signature and docstring
  - [ ] Implementation
  - [ ] Test: mamba2_forward_into_equivalence
  - [ ] Verify 490+ tests pass

- [ ] **TemporalMixingLayer dispatch update** (src/domain/layers/components/common.rs lines ~310-325)
  - [ ] Add RgLru case (already has forward_into)
  - [ ] Add Mamba case
  - [ ] Add Mamba2 case
  - [ ] Add MoH variants
  - [ ] Remove fallback path for these variants

- [ ] **Integration tests** (tests/ssm_forward_into_*.rs)
  - [ ] Mamba: equivalence across batch sizes
  - [ ] Mamba2: equivalence + edge cases
  - [ ] MoH variants: with different expert counts

---

## Code Patterns (Copy from RgLru)

### Test Template
```rust
#[test]
fn test_[variant]_forward_into_equivalence() {
    let mut layer = [Variant]::new(/* config */);
    let input = Array2::random((seq_len, embed_dim), Normal::new(0.0, 0.1).unwrap());
    
    let output_forward = layer.forward(&input);
    let mut output_into = Array2::zeros_like(&output_forward);
    layer.forward_into(&input, &mut output_into).unwrap();
    
    assert_abs_diff_eq!(output_forward, output_into, epsilon = 1e-4);
}
```

### Workspace Reuse Pattern
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    // 1. Use existing workspace instead of allocating new buffers
    // 2. Keep allocations that remain in self.workspace
    // 3. Write final result directly to output
    
    let rows = input.nrows();
    let cols = input.ncols();
    
    if output.nrows() != rows || output.ncols() != cols {
        *output = Array2::zeros((rows, cols));
    }
    
    // Actual computation happens here
    let result = self.forward(input); // For now; later optimize to avoid this
    output.assign(&result); // Zero-copy assignment if shapes match
    
    Ok(())
}
```

---

## Performance Validation

After completing all implementations:

```bash
# Run all tests
cargo test --lib 2>&1 | tail -5

# Benchmark specific layers
cargo bench --bench [bench_name] -- --sample-size 100
```

**Target metrics**:
- All 490+ tests passing
- No new clippy warnings
- Memory reduction: 15-20 KB/step measurable

---

## Fallback Strategy

If any variant proves too complex:
1. Keep the simple pattern (call forward(), assign to output)
2. Document why in-place isn't viable
3. Move to Phase 5.1c (BlockLevel Integration) with partial implementation
4. Return to complex variants after block-level benefits proven

---

## Success Definition

✅ **This session is complete when:**
1. MoHRgLru, Mamba, Mamba2 all have forward_into() implemented
2. All 3 variants have equivalence tests
3. TemporalMixingLayer dispatch updated for direct calls
4. 490+ tests still passing
5. No clippy warnings introduced
6. Memory savings validated (15-20 KB/step)

---

## Next Phase (5.1c): Block Integration

Once SSM variants complete, integrate into TransformerBlock and DiffusionBlock:

```rust
// In TransformerBlock::forward_block_into()
let mut temporal_out = workspace.temporal_out.take().unwrap();
self.temporal_mixing.forward_into(&norm1_out, &mut temporal_out)?; // Direct call
workspace.temporal_out = Some(temporal_out);
```

This consolidates all 12 layers to use in-place paths → 40 KB/step total savings.

---

## References
- Thread: T-019c56f9-2fe2-77bc-900a-27eff0fcaca2
- Phase 5.1 Documentation: PHASE5_1_EXECUTION_PLAN.md
- RgLru Template: src/domain/layers/ssm/rg_lru.rs (lines 736-800)
