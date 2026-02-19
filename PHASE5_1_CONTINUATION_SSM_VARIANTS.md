# Phase 5.1 Continuation: SSM Variants In-Place Operations
**Date**: February 13-15, 2026  
**Status**: Ready for Implementation  
**Previous Progress**: Foundation Layer Complete (20%)

---

## Overview

Continuing Phase 5.1 (In-Place Operations) from foundation layer to SSM variants. This session focuses on implementing `forward_into()` for state-space models (RG-LRU, Mamba, Mamba2) and their Mixture-of-Heads variants.

### Key Metrics
- **Current Memory/Step**: 129 KB (Phase 4.1)
- **Phase 5.1 Target**: 89 KB (40 KB reduction)
- **SSM Variants Target**: 8-10 fewer allocations/step

---

## Implementation Tasks

### Task 1: RgLru::forward_into() ⏳ (Next)

**File**: `src/domain/layers/ssm/rg_lru.rs`

**Core Pattern**:
The RgLru forward pass:
1. Compute gates (r, i, a) from input using weights
2. Compute state (h) from gates and input
3. Cache intermediate values for backward

**Implementation Strategy**:
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let (t, d) = input.dim();
    
    // Validate output dimensions
    if output.dim() != (t, d) {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Output dimension mismatch"
        )));
    }
    
    // Reuse or allocate intermediate buffers for gates
    // Gates: r, i, a (3 × (t, d))
    if self.cached_r.as_ref().is_none_or(|x| x.dim() != (t, d)) {
        self.cached_r = Some(Array2::<f32>::zeros((t, d)));
    }
    if self.cached_i.as_ref().is_none_or(|x| x.dim() != (t, d)) {
        self.cached_i = Some(Array2::<f32>::zeros((t, d)));
    }
    if self.cached_a.as_ref().is_none_or(|x| x.dim() != (t, d)) {
        self.cached_a = Some(Array2::<f32>::zeros((t, d)));
    }
    if self.cached_hprev.as_ref().is_none_or(|x| x.dim() != (t, d)) {
        self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
    }
    
    let r = self.cached_r.as_mut().unwrap();
    let i = self.cached_i.as_mut().unwrap();
    let a = self.cached_a.as_mut().unwrap();
    let hprev = self.cached_hprev.as_mut().unwrap();
    
    // Compute gates in-place into cached buffers
    Self::compute_gates_into_parts(
        input,
        GatesParams {
            w_a: &self.w_a,
            b_a: &self.b_a,
            w_x: &self.w_x,
            b_x: &self.b_x,
            lambda: &self.lambda,
        },
        r,
        i,
        a,
    );
    
    // Compute state directly into output buffer
    Self::compute_state_into(input, i, a, hprev, output);
    
    // Cache input for backward
    self.cached_input = Some(input.clone());
    
    Ok(())
}
```

**Equivalence Test Pattern**:
```rust
#[test]
fn test_rg_lru_forward_into_equivalence() {
    let mut rg = RgLru::new(/* ... */);
    let input = /* ... */;
    
    // Compute using standard forward
    let output_standard = rg.forward(&input);
    
    // Compute using forward_into
    let mut output_into = Array2::zeros(input.dim());
    rg.forward_into(&input, &mut output_into).unwrap();
    
    // Assert equivalence (< 1e-5 relative error)
    assert_abs_diff_eq!(output_standard, output_into, epsilon = 1e-5);
}
```

**Memory Savings**:
- Eliminates: 1 × (t, d) allocation for `h` output
- Reduces allocations per step: ~1-2
- Total: ~4-8 KB/step savings

---

### Task 2: MoHRgLru::forward_into()

**File**: `src/domain/layers/ssm/rg_lru.rs`

**Complexity**: Higher due to head routing

**Pattern**:
1. Compute per-head routing (via MoH gating)
2. Apply RgLru computation per routed head subset
3. Aggregate outputs

**Key Consideration**: MoH adds `moh_outputs` allocation that should be reused from workspace

---

### Task 3: Mamba::forward_into()

**File**: `src/domain/layers/ssm/mamba.rs`

**SSM-specific Challenge**: Mamba has state transitions and scan operations

**Likely Intermediates**:
- Expanded state (`expand_dim` × embed_dim)
- Delta computation
- Scan output

**Implementation Approach**: Reuse unified_workspace buffers for temporal outputs, scan results

---

### Task 4: Mamba2::forward_into()

**File**: `src/domain/layers/ssm/mamba2.rs`

**Similar to Mamba** but with additional normalization and gating

---

### Task 5: MoH Variants (MoHMamba, MoHMamba2)

**Implementation**: Apply same head routing pattern as MoHRgLru

---

## Code Quality Checklist

- [ ] Dimension validation on output buffer
- [ ] Clear error messages (InvalidInput, etc.)
- [ ] Caching for backward compatibility
- [ ] Equivalence tests (< 1e-5 error)
- [ ] No clippy warnings
- [ ] Documentation with examples
- [ ] Test coverage for edge cases (t=0, d=1, large tensors)

---

## Build & Test Plan

### Build Verification
```bash
cargo build --release
```

### Test Suite
```bash
# All library tests (should pass 476+)
cargo test --lib

# SSM-specific forward_into tests
cargo test --lib forward_into -- --nocapture

# Specific test (after implementation)
cargo test --lib test_rg_lru_forward_into_equivalence
```

### Benchmarking (Post-Implementation)
```bash
cargo bench --bench [benchmark_name]
```

---

## Success Criteria

- [x] Plan documented
- [ ] RgLru::forward_into() implemented and tested
- [ ] All 476+ tests pass
- [ ] < 1e-5 equivalence error across all SSM variants
- [ ] Memory usage reduction measured (target: 40 KB/step total)
- [ ] No new clippy warnings

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| State management in SSM | Keep cached intermediates for backward compatibility |
| MoH routing complexity | Start with base RgLru, then layer MoH pattern |
| Backward compatibility | Equivalence tests validate correctness |
| Performance regression | Benchmark before/after latency |

---

## Session Checkpoints

**Phase 5.1a (SSM Variants)**:
- Checkpoint 1: RgLru::forward_into() complete + tests passing ✓
- Checkpoint 2: Mamba variants started
- Checkpoint 3: All SSM variants implemented
- Gate: All 476+ library tests still pass

**Phase 5.1b (FFN Variants)**: Next session
**Phase 5.1c (Block Integration)**: Week of Feb 19
**Phase 5.1d (Validation)**: Week of Feb 20

---

## References

- **Foundation Layer Implementations**: `src/domain/layers/components/temporal_processing.rs`
- **Reference Implementation**: `src/domain/attention/poly_attention.rs#L1548-L1641`
- **SSM Core**: `src/domain/layers/ssm/`
- **Phase 5.1 Roadmap**: PHASE5_1_IN_PLACE_OPERATIONS_ROADMAP.md
- **Execution Plan**: PHASE5_1_EXECUTION_PLAN.md
