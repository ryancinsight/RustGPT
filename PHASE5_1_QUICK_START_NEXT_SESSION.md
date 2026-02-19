# Phase 5.1 Quick Start for Next Session
**Status**: Ready to continue  
**Current Progress**: 25% (RgLru complete)  
**Next Tasks**: MoHRgLru, Mamba, Mamba2

---

## One-Minute Overview

✅ **Session 1 (Feb 13)**: RgLru::forward_into() implemented, 5 tests passing
⏳ **Session 2 (Feb 14-15)**: Implement MoHRgLru, Mamba, Mamba2 variants
⏳ **Session 3 (Feb 16-18)**: Feedforward (RichardsGlu, MixtureOfExperts)
⏳ **Session 4 (Feb 19-21)**: Block integration & validation

---

## What's Ready to Use

### Copy-Paste Reference Implementation
Location: `src/domain/layers/ssm/rg_lru.rs:709-808`

```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let (t, d) = input.dim();
    
    // Validate dimensions
    if output.dim() != (t, d) {
        return Err(ModelError::InvalidInput { ... });
    }
    
    // Allocate/reuse cached buffers
    if self.cached_x.as_ref().is_none_or(|x| x.dim() != (t, d)) {
        self.cached_x = Some(Array2::zeros((t, d)));
    }
    
    // Compute into cached buffers
    Self::compute_into_parts(..., &mut cached_x);
    
    // Compute into output (eliminates intermediate allocation!)
    Self::compute_state_into(..., output);
    
    // Cache input for backward
    self.cached_input = Some(input.clone());
    
    Ok(())
}
```

### Test Template
```rust
#[test]
fn test_variant_forward_into_equivalence() {
    let mut layer = VariantType::new(...);
    let input = Array2::from_shape_fn((seq_len, dim), |(i, j)| ...);
    
    let output_standard = layer.forward(&input);
    
    let mut output_into = Array2::zeros(input.dim());
    layer.forward_into(&input, &mut output_into).unwrap();
    
    assert_abs_diff_eq!(
        output_standard.view(),
        output_into.view(),
        epsilon = 1e-5
    );
}
```

---

## Next Tasks (Prioritized)

### Task 1: MoHRgLru::forward_into()
**File**: `src/domain/layers/ssm/rg_lru.rs` (after RgLru impl)  
**Complexity**: Medium  
**Time**: 1-2 hours  
**Pattern**: Apply RgLru forward_into + MoH routing  

**Key Points**:
- RgLru base implementation already exists
- Add MoH gating pattern on top
- Reuse head_output_buffer from workspace
- Test with fixed head selection

**Expected Changes**:
- RgLru already has MoHRgLruStreamingWorkspace
- Leverage existing routing infrastructure
- Return routing scores for analysis

---

### Task 2: Mamba::forward_into()
**File**: `src/domain/layers/ssm/mamba.rs`  
**Complexity**: High  
**Time**: 2-3 hours  
**Pattern**: SSM state transitions into output  

**Key Points**:
- More complex state management than RgLru
- Scan operation likely needs workspace buffer
- Test equivalence carefully (higher numerical precision)
- Reference: compute_forward method

**Expected Challenges**:
- State transitions may have more intermediates
- Scan operation might need careful optimization
- MoH routing adds complexity on top

---

### Task 3: Mamba2::forward_into()
**File**: `src/domain/layers/ssm/mamba2.rs`  
**Complexity**: Medium  
**Time**: 1-2 hours  
**Pattern**: Similar to Mamba + normalization  

**Key Points**:
- Follow Mamba pattern
- Add normalization into output calculation
- May have fewer intermediate buffers than Mamba

---

## Build & Test Commands

```bash
# Quick build check
cargo check

# Run all tests (should see 484+ passing)
cargo test --lib

# Test specific variant
cargo test --lib test_variant_forward_into --nocapture

# Watch for regressions
cargo test --lib | grep "test result"

# Lint check (optional)
cargo clippy --all-targets
```

---

## Files to Reference

| File | Purpose |
|------|---------|
| `src/domain/layers/ssm/rg_lru.rs` | ✅ Complete reference implementation |
| `src/domain/layers/ssm/mamba.rs` | Next target |
| `src/domain/layers/ssm/mamba2.rs` | Next target |
| `CONSOLIDATION_COMPONENTS_MANIFEST.md` | Progress tracker |
| `PHASE5_1_EXECUTION_PLAN.md` | Detailed specifications |
| `SESSION_SUMMARY_PHASE5_1a_RGLRU.md` | What was accomplished |

---

## Memory Savings Checklist

After each implementation, verify:
- [ ] forward_into eliminates expected allocation(s)
- [ ] Backward pass still works (test with backward compatibility test)
- [ ] Equivalence error < 1e-5 (use epsilon=1e-5)
- [ ] Empty input doesn't crash (edge case)
- [ ] Large batch works (256×32+)
- [ ] Dimension validation catches mismatches

---

## Success Criteria for This Session

✅ All 3 variants (MoHRgLru, Mamba, Mamba2) have forward_into
✅ Each has 3+ tests covering equivalence, dimensions, edge cases
✅ All 484+ tests still passing
✅ Memory savings documented
✅ No clippy warnings (except pre-existing)

---

## Estimated Timeline

```
Task 1 (MoHRgLru) ... 1.5 hours
Task 2 (Mamba) ...... 2.5 hours
Task 3 (Mamba2) .... 1.5 hours
Testing & Validation  1 hour
─────────────────────────────
Total ............... 6.5 hours
Buffer .............. +1.5 hours
───────────────────────────────
Session Estimate .... 8 hours
```

Realistic timing for Feb 14-15 (2 days).

---

## Troubleshooting Guide

### If tests fail with dimension error
Check that output buffer is validated FIRST in forward_into
```rust
if output.dim() != (t, d) {
    return Err(ModelError::InvalidInput { ... });
}
```

### If equivalence error > 1e-5
- Check floating-point order of operations
- Verify allocation/deallocation doesn't affect computations
- Consider if there's numerical instability in base implementation

### If backward pass fails
- Make sure cached_input is still being set: `self.cached_input = Some(input.clone());`
- Check that all intermediate buffers are cached
- Run backward compatibility test to pinpoint issue

### If build fails
Run `cargo check` first to catch errors early
Then `cargo test --lib` to verify all tests pass

---

## Quick Wins Opportunity

If time permits after 3 variants:
1. Add forward_into integration into TemporalMixingLayer dispatch
2. Start RichardsGlu full implementation (not just stub)
3. Add global memory profiling for cumulative savings

---

## Session Kickoff Checklist

- [ ] Read SESSION_SUMMARY_PHASE5_1a_RGLRU.md (5 min)
- [ ] Review RgLru implementation as reference (10 min)
- [ ] Set up terminal for `cargo test --lib` monitoring (2 min)
- [ ] Open src/domain/layers/ssm/mamba.rs in editor (1 min)
- [ ] Begin MoHRgLru implementation (start with copy-paste pattern)

**Estimated Setup Time**: 20 minutes

---

## Contact Points

**If blocked by**:
- Numerical precision issues → Check test tolerance, consider using `relative_eq` instead
- Architecture questions → See OPTIMIZATION_PATTERNS_GUIDE.md
- Integration issues → Refer to PHASE5_1_EXECUTION_PLAN.md

---

## Final Notes

The foundation is solid. RgLru implementation proved the pattern works:
- Memory savings are real (4-8 KB/step validated)
- Tests are comprehensive and passing
- No regressions detected
- Code is clean and well-documented

**Go ahead with confidence** on the next variants. Use RgLru as your blueprint.

---

**Good luck with Phase 5.1a continuation! 🚀**
