# Phase 5.1c Quick Reference Card

**Status**: Ready for implementation (70% complete, 2 tasks remaining)  
**Est. Time**: 2-2.5 hours  
**Target**: 5-8% per-layer speedup via batch-path optimization

---

## The 3-Step Plan

### ✅ Step 1: Workspace Buffer Methods (30 min)

**File**: `src/domain/layers/components/unified_layer_workspace.rs`

**Add to `impl UnifiedLayerWorkspace`**:
```rust
// Pattern: take/return for each buffer
pub fn take_norm1_out(&mut self) -> Array2<f32> {
    self.norm1_out.take().unwrap_or_else(|| Array2::zeros((1, 1)))
}
pub fn return_norm1_out(&mut self, buf: Array2<f32>) {
    self.norm1_out = Some(buf);
}

// Repeat for: temporal_out, residual1, norm2_out, ffn_out

// Batch operations
pub fn take_all_buffers(&mut self) -> (Array2, Array2, Array2, Array2, Array2) { ... }
pub fn return_all_buffers(&mut self, n1: Array2, t: Array2, r1: Array2, n2: Array2, f: Array2) { ... }
```

---

### ✅ Step 2: RichardsNorm::forward_into() (20 min)

**File**: Find with `grep -r "impl RichardsNorm" src/`

**Add method**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
    let result = self.forward(input);
    if output.dim() != result.dim() {
        *output = Array2::zeros(result.dim());
    }
    output.assign(&result);
}
```

---

### ✅ Step 3: TransformerBlock::forward() Refactor (45 min)

**File**: `src/domain/layers/transformer/block.rs` (lines 763-917)

**Key changes** (6 replacements):

| Line | Current | New |
|------|---------|-----|
| ~774 | N/A | Add: `let (mut n1, mut mix, mut r1, mut n2, mut f) = workspace.take_all_buffers();` |
| ~797 | `let norm1_out = pre_norm.forward(...)` | `pre_norm.forward_into(&input_used, &mut n1);` |
| ~841 | `let mix_out = temporal.forward_with_titan(...)` | `temporal.forward_into(&n1, &mut mix);` |
| ~870 | `let residual1 = mix_out.clone(); residual1 += input;` | `r1.assign(&mix); r1 += &input_used;` |
| ~876 | `let norm2_out = pre_ffn_norm.forward(&r1)` | `pre_ffn_norm.forward_into(&r1, &mut n2);` |
| ~879 | `let ffn_out = feedforward.forward_with_...` | `feedforward.forward_into(&n2, &mut f);` |
| ~916 | Return `ffn_out` | Add: `workspace.return_all_buffers(n1, mix, r1, n2, f.clone()); ffn_out` |

**Pattern**:
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ... existing setup ...
    self.unified_workspace.ensure_capacity(...);
    
    // NEW: Take workspace buffers
    let (mut n1, mut mix, mut r1, mut n2, mut f) = self.unified_workspace.take_all_buffers();
    
    // ... forward pass using forward_into() ...
    self.pre_attention_norm.forward_into(&input_used, &mut n1);
    self.temporal_mixing.forward_into(&n1, &mut mix);
    r1.assign(&mix); r1 += &input_used;
    self.pre_ffn_norm.forward_into(&r1, &mut n2);
    self.feedforward.forward_into(&n2, &mut f);
    f += &r1;
    
    // ... cache intermediates ...
    
    // NEW: Return workspace buffers
    self.unified_workspace.return_all_buffers(n1, mix, r1, n2, f.clone());
    
    f  // Return output
}
```

---

## Testing Checklist

```bash
# Full test suite (should show 504 passing)
cargo test --lib 2>&1 | tail -5

# Check for warnings
cargo clippy --all-targets 2>&1 | grep -i warning

# Optional: benchmark
cargo bench --bench transformer_block 2>&1 | head -20
```

**Success**: 504/504 tests pass, no new warnings

---

## Common Pitfalls & Fixes

| Problem | Fix |
|---------|-----|
| "Cannot move out of self" | Use `take()`/`return()` pattern, not direct assignment |
| Type mismatch Array2 vs Option | Ensure `take_*` returns `Array2`, not `Option` |
| Workspace dimensions wrong | Check `ensure_capacity()` is called before `take_all_buffers()` |
| Tests fail after refactor | Verify cached intermediates still use Arc clones for backward pass |
| No performance improvement | Check workspace buffers are actually being reused (not reassigned) |

---

## Memory Impact (Per Layer)

**Before**: ~12 KB allocations per forward  
**After**: ~1-2 KB (context only)  
**Savings**: 10-11 KB per layer

**Scaling** (12-layer model):
- 100-step batch: 1.2-1.3 MB → 0.12-0.15 MB
- Reduction: **90%** for workspace allocations

---

## File Checklist

- [ ] `src/domain/layers/components/unified_layer_workspace.rs` - Add take/return methods
- [ ] Find RichardsNorm location - Add forward_into
- [ ] `src/domain/layers/transformer/block.rs` - Refactor forward()
- [ ] Run tests - Verify 504/504 pass
- [ ] Update manifest - Mark Phase 5.1c complete
- [ ] Benchmark - Document speedup

---

## Success Criteria

✅ When all of these are true:

1. Code compiles: `cargo build --release 2>&1 | grep -i error` → no output
2. Tests pass: `cargo test --lib 2>&1 | tail -3` → `504 passed`
3. No warnings: `cargo clippy --all-targets 2>&1` → clean
4. Output correct: Manual test shows reasonable numerical output
5. Memory reduced: Workspace buffers taken/returned consistently
6. Documentation: Manifest updated with completion status

---

## Time Breakdown

| Task | Time | Status |
|------|------|--------|
| Read & understand workspace structure | 10 min | ⏳ |
| Implement take/return methods | 15 min | ⏳ |
| Add RichardsNorm::forward_into() | 10 min | ⏳ |
| Refactor TransformerBlock::forward() | 35 min | ⏳ |
| Fix compilation errors | 15 min | ⏳ |
| Run tests & validate | 10 min | ⏳ |
| Benchmark (optional) | 10 min | ⏳ |
| Document results | 5 min | ⏳ |
| **Total** | **110 min** | ⏳ |

---

## Key Implementation Files

```
src/domain/layers/components/
├── unified_layer_workspace.rs     ← Add take/return methods
├── normalization.rs               ← Add RichardsNorm::forward_into()
└── temporal_processing.rs         ← ✅ Already has forward_into()

src/domain/layers/transformer/
└── block.rs                       ← Refactor forward() to use in-place ops

src/domain/richards/
└── richards_glu.rs               ← ✅ Already has forward_into()

src/domain/mixtures/
└── moe.rs                         ← ✅ Already has forward_into()
```

---

## Next: Phase 5.2 (Preview)

Once 5.1c complete:
- Global buffer pooling (consolidate layer pools)
- Selective gradient computation (frozen layers)
- Mixed precision (f16 for historical context)

**Timeline**: Next session (est. 2-3 hours)  
**Expected additional savings**: 30-40 KB/step

---

## Quick Command Reference

```bash
# Find RichardsNorm
grep -r "impl RichardsNorm" src/ | head -1

# Find UnifiedLayerWorkspace  
ls -la src/domain/layers/components/unified_layer_workspace.rs

# Build & test
cargo build --release 2>&1 | tail -20
cargo test --lib 2>&1 | tail -5

# Format & lint
cargo fmt
cargo clippy --all-targets
```

---

**Ready to start? Begin with Step 1!** ✨
