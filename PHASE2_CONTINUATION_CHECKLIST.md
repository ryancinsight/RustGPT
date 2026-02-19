# Phase 2 Continuation Checklist

**Next Session Target:** Activate Transformer Workspace + Integrate AdaptiveResiduals Workspace  
**Estimated Time:** 4-6 hours  
**Expected Gains:** 30-40% total memory, 25-30% latency

---

## Pre-Session Setup (5 minutes)

- [ ] Pull latest changes
- [ ] Verify 456 tests still passing
- [ ] Check git status (should be clean)
- [ ] Open these files in editor:
  - `src/domain/layers/transformer/block.rs` (main file)
  - `src/domain/layers/components/adaptive_residuals.rs` (integration target)
  - `src/domain/layers/components/adaptive_residuals_workspace.rs` (new workspace)

---

## Task 1: Activate Transformer Workspace (1-2 hours)

### Step 1.1: Review Existing Code (15 min)
- [ ] Read `TransformerWorkspace` struct (line 383-443 in block.rs)
- [ ] Understand `ensure_capacity()` method
- [ ] Identify where `batch_workspace` field is initialized

### Step 1.2: Locate Forward Pass Allocations (15 min)
- [ ] Find `fn forward(&mut self, input: &Array2<f32>)` (line 865)
- [ ] Identify these allocations:
  ```rust
  let norm1_out = self.pre_attention_norm.forward(input_used_arc.as_ref());
  let mix_out = self.temporal_mixing.forward_with_titan_fusion_default(...);
  // ... more allocations ...
  ```
- [ ] Note: Already caches with Arc, but allocates each time

### Step 1.3: Activate Workspace (30 min)
```rust
// At start of forward():
if self.batch_workspace.is_none() {
    self.batch_workspace = Some(TransformerWorkspace::new(
        input.nrows(),
        self.config.embed_dim,
    ));
}

let ws = self.batch_workspace.as_mut().unwrap();
ws.ensure_capacity(input.nrows(), self.config.embed_dim);

// Then in forward loop:
// Instead of: let norm1_out = self.pre_attention_norm.forward(...);
// Option 1: Use workspace buffers directly if pre_attention_norm supports in-place
// Option 2: Still allocate but capture for reuse in backward
```

### Step 1.4: Update CachedIntermediates (30 min)
- [ ] Decide: Cache Arc references or use workspace buffers?
- [ ] If using workspace: Update cached_intermediates to reference workspace
- [ ] If caching Arc: Ensure no duplication with workspace

### Step 1.5: Test & Validate (15 min)
```bash
cargo build --lib
cargo test --lib transformer 2>&1 | grep "test result"
```
- [ ] All tests pass
- [ ] No new warnings
- [ ] Latency measurements show improvement

### Step 1.6: Benchmark (15 min)
```bash
# Informal: run forward 1000 times, measure time
# Formal: Create benches/transformer_workspace.rs
```
- [ ] Measure latency before
- [ ] Measure latency after
- [ ] Calculate improvement %

**Expected Result:** 15-20% latency improvement, ~20% memory reduction

---

## Task 2: Integrate AdaptiveResiduals Workspace (2-3 hours)

### Step 2.1: Understand Current Structure (15 min)
- [ ] Open `adaptive_residuals.rs`
- [ ] Review current `scratch_*` fields (lines 80-97 in original)
- [ ] Count how many methods use these fields

### Step 2.2: Add Workspace Field (10 min)
```rust
// In AdaptiveResiduals struct, replace scratch_* with:
pub workspace: Option<Arc<Mutex<AdaptiveResidualsWorkspace>>>,

// Or simpler approach (if not multi-threaded):
pub internal_workspace: AdaptiveResidualsWorkspace,
```

### Step 2.3: Add Workspace Methods (10 min)
```rust
pub fn set_workspace(&mut self, ws: Arc<AdaptiveResidualsWorkspace>) {
    self.workspace = Some(ws);
}

fn get_workspace_mut(&mut self) -> &mut AdaptiveResidualsWorkspace {
    if self.workspace.is_some() {
        // Can't mutate Arc directly without Mutex
        &mut self.internal_workspace
    } else {
        &mut self.internal_workspace
    }
}
```

### Step 2.4: Update Constructor (10 min)
- [ ] Update `new()` to initialize workspace
- [ ] Update `new_minimal()` similarly
- [ ] Ensure Clone trait still works

### Step 2.5: Migrate Methods (1-2 hours)
Replace all `self.scratch_*` references with workspace:

**Pattern:**
```rust
// Before:
self.scratch_nx.resize(embed_dim, 0.0f64);
self.scratch_nx.fill(0.0f64);
// ... use self.scratch_nx[i] ...

// After:
let ws = self.get_workspace_mut();
ws.resize_for_dim(embed_dim);
// ... use ws.nx[i] ...
```

**Methods to update:**
1. `apply_attention_residual_step_into()` - uses `channel_scales`
2. `apply_attention_residual_with_moh()` - uses `channel_scales`
3. `apply_ffn_residual_step_into()` - uses `channel_scales`
4. `similarity_stats_from_weights()` - uses `nx`, `ny`, `dot`
5. `compute_gradients()` - doesn't use workspace
6. `apply_gradients()` - doesn't use workspace

**Effort breakdown:**
- apply_attention_residual methods: 45 min (3 methods × 15 min each)
- apply_ffn_residual methods: 45 min (3 methods × 15 min each)
- similarity_stats method: 15 min (complex but just array indexing changes)
- Testing: 15 min

### Step 2.6: Test Thoroughly (15 min)
```bash
cargo build --lib
cargo test --lib adaptive_residuals 2>&1 | grep "test result"
cargo test --lib 2>&1 | tail -3
```
- [ ] All 456 tests pass
- [ ] No regressions
- [ ] New workspace is correctly sized

### Step 2.7: Memory Profile (15 min)
- [ ] Before integration: Note memory usage
- [ ] After integration: Compare
- [ ] Expected: 25-30% reduction for workspace buffers

**Expected Result:** 25-30% memory reduction, workspace pooling ready

---

## Task 3: Benchmarking & Validation (1 hour)

### Step 3.1: Create Benchmark File (15 min)
```rust
// benches/phase2_consolidation.rs
#[bench]
fn bench_transformer_forward_with_workspace(b: &mut Bencher) {
    let mut block = TransformerBlock::new(...);
    let input = Array2::zeros((256, 768));
    b.iter(|| block.forward(&input));
}

#[bench]
fn bench_adaptive_residuals_with_workspace(b: &mut Bencher) {
    // Similar benchmark for adaptive residuals
}
```

### Step 3.2: Run Benchmarks (15 min)
```bash
cargo bench --bench phase2_consolidation 2>&1 | grep "time"
```
- [ ] Document baseline numbers
- [ ] Run again after implementation
- [ ] Calculate improvement percentage

### Step 3.3: Memory Profiling (15 min)
- [ ] Use `valgrind` or similar if available
- [ ] Or use Rust's memory safety checks
- [ ] Track allocation counts
- [ ] Compare before/after

### Step 3.4: Documentation Update (15 min)
- [ ] Update PHASE2_INPROGRESS.md with completion status
- [ ] Add benchmark results
- [ ] Document any gotchas encountered
- [ ] Update performance metrics tables

**Expected Result:** Document 30-40% total memory savings, 25%+ latency improvement

---

## Testing Checklist

### Before Starting
- [ ] `cargo build --lib` passes
- [ ] `cargo test --lib` passes (456 tests)
- [ ] Git status is clean

### After Each Task
- [ ] Compilation: `cargo build --lib`
- [ ] Unit tests: `cargo test --lib`
- [ ] Specific tests: `cargo test --lib transformer` (or adaptive_residuals)
- [ ] No new warnings added

### Before Finishing Session
- [ ] All 456 tests pass
- [ ] No compiler warnings related to changes
- [ ] Benchmarks run successfully
- [ ] Git diff is clean and logical
- [ ] Documentation is updated

---

## Troubleshooting Guide

### Issue: "Cannot borrow as mutable multiple times"
**Cause:** Trying to get multiple mutable references to workspace  
**Solution:** Use single `ws` variable, index into it
```rust
// ❌ Don't do this:
let ws1 = self.get_workspace_mut();
let ws2 = self.get_workspace_mut();  // ERROR

// ✅ Do this:
let ws = self.get_workspace_mut();
use ws.channel_scales and ws.nx in same scope
```

### Issue: Tests fail after workspace integration
**Cause:** Workspace not properly initialized  
**Solution:** Ensure `resize_for_dim()` called before using workspace
```rust
// Always do this before accessing workspace:
ws.resize_for_dim(embed_dim);
// Then access ws.nx, ws.channel_scales, etc.
```

### Issue: Memory usage hasn't improved
**Cause:** Workspace allocations still happening  
**Solution:** Check `ensure_capacity()` logic
```rust
// Make sure you're calling:
ws.resize_for_dim(embed_dim);
// And dimensions match expectation
```

### Issue: Latency is slower after changes
**Cause:** Added unnecessary clones or copies  
**Solution:** Review changes for unintended allocations
```rust
// Look for:
// - .clone() calls (should use references)
// - Array2::zeros() (should reuse workspace)
// - .to_vec() (should use views)
```

---

## Time Allocation

```
Phase 2 Continuation (4-6 hours total):

Task 1: Activate Transformer Workspace   1.5-2.0 hours
  ├─ Review existing code              0.25h
  ├─ Locate allocations               0.25h
  ├─ Activate workspace               0.5h
  ├─ Update caching                   0.5h
  ├─ Test & validate                  0.25h
  └─ Benchmark                        0.25h

Task 2: Integrate AdaptiveResiduals WS  2.0-2.5 hours
  ├─ Review structure                 0.25h
  ├─ Add workspace field              0.25h
  ├─ Update all methods               1.5-2.0h
  └─ Test thoroughly                  0.25h

Task 3: Benchmarking & Docs            0.5-1.0 hours
  ├─ Create benchmarks                0.25h
  ├─ Run & profile                    0.25h
  ├─ Update documentation             0.25h

Buffer for issues/refinement:          0.5 hours
```

---

## Success Criteria

### Transformer Workspace Activation
- [ ] `batch_workspace` is used in `forward()`
- [ ] `ensure_capacity()` called at start of forward
- [ ] Latency improved by 15-20%
- [ ] All tests still pass
- [ ] No memory increase

### AdaptiveResiduals Workspace Integration
- [ ] Workspace field added to struct
- [ ] All scratch_* references replaced
- [ ] Methods accept and use workspace
- [ ] Memory reduced by 25-30%
- [ ] All tests still pass

### Overall Phase 2
- [ ] Both optimizations complete
- [ ] 456 tests passing
- [ ] Combined memory savings: 30-40% (Phase 1 + 2)
- [ ] Combined latency improvement: 25-30%
- [ ] Documentation updated
- [ ] Code ready for Phase 3

---

## Quick Command Reference

```bash
# Build
cargo build --lib

# Test everything
cargo test --lib

# Test specific module
cargo test --lib transformer
cargo test --lib adaptive_residuals

# Benchmark
cargo bench --bench phase2_consolidation

# Check for issues
cargo clippy --lib

# Format code
cargo fmt

# Git workflow
git status
git diff src/domain/layers/transformer/block.rs
git add -A
git commit -m "Phase 2: Activate Transformer workspace"
```

---

## Session Goals Summary

**Start:** Phase 2 at 50% completion (in-place ops done, infrastructure created)  
**Target:** Phase 2 at 100% completion (both major optimizations active)  
**Expected Outcome:** 30-40% total memory savings, 25%+ latency improvement  
**Quality Gate:** All 456 tests passing, zero breaking changes

---

## Notes for Next Session Lead

- Transformer workspace is fully implemented, just unused
- AdaptiveResiduals workspace is created and tested, ready to integrate
- In-place context mixing is working and tested
- Estimated 4-6 hours of focused work
- Lower risk than it appears (infrastructure already exists)
- Major challenge will be updating all adaptive_residuals methods (repetitive but straightforward)

**Recommended approach:** Start with Transformer workspace (quick win), then tackle AdaptiveResiduals (longer but systematic).

Good luck! 🚀
