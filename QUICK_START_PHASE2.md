# Phase 2 Quick Start Guide

## Overview
Continue consolidation with workspace pooling and caching optimizations.

---

## Phase 2 Tasks (in order)

### Task 1: AdaptiveResiduals Workspace Pooling (2-3 hours)

**File:** `src/domain/layers/components/adaptive_residuals.rs`

**Create new file:** `src/domain/layers/components/adaptive_residuals_workspace.rs`
```rust
pub struct AdaptiveResidualsWorkspace {
    pub nx: Vec<f64>,
    pub ny: Vec<f64>,
    pub mean_x: Vec<f64>,
    pub mean_y: Vec<f64>,
    pub mean_z: Vec<f64>,
    pub perf_values: Vec<f64>,
    pub channel_scales: Vec<f32>,
    pub dot: Vec<f64>,
    pub z: Vec<f64>,
}

impl AdaptiveResidualsWorkspace {
    pub fn resize_for_dim(&mut self, embed_dim: usize) {
        let capacity = embed_dim.next_power_of_two();
        // Resize all buffers to capacity
    }
}
```

**Modifications to AdaptiveResiduals:**
1. Remove individual `scratch_*` Vec fields (use workspace instead)
2. Add `workspace: Option<Arc<Mutex<AdaptiveResidualsWorkspace>>>`
3. Update all methods to use shared workspace
4. Add `set_workspace()` method

**Expected gain:** 25-30% memory reduction

---

### Task 2: Transformer Workspace Generational Buffers (1-2 hours)

**File:** `src/domain/layers/transformer/block.rs`

**Create workspace struct:**
```rust
pub struct TransformerWorkspace {
    last_dims: Option<(usize, usize, usize)>,  // (batch, seq, embed_dim)
    norm_out: Array2<f32>,
    temporal_out: Array2<f32>,
    ffn_out: Array2<f32>,
}

impl TransformerWorkspace {
    pub fn ensure_capacity(&mut self, batch: usize, seq: usize, embed_dim: usize) {
        if self.last_dims != Some((batch, seq, embed_dim)) {
            // Reallocate
            self.last_dims = Some((batch, seq, embed_dim));
        } else {
            // Clear only
            self.norm_out.fill(0.0);
            // ... clear others
        }
    }
}
```

**Modifications to TransformerBlock:**
1. Add workspace field
2. Call `ensure_capacity()` before forward
3. Use workspace buffers in forward pass

**Expected gain:** 15-20% latency improvement, ~20% memory reduction

---

### Task 3: In-place Context Application (1 hour)

**File:** `src/domain/layers/components/attention_context.rs`

**Add new method:**
```rust
pub fn apply_context_into(
    &self, 
    input: &Array2<f32>, 
    output: &mut Array2<f32>
) {
    if output.dim() != input.dim() {
        *output = input.clone();
        return;
    }
    
    if let Some(context) = &self.incoming_context {
        let scale = self.get_strength() / (input.ncols() as f32).max(1.0);
        
        // Use linalg for in-place mixing
        ndarray::linalg::general_mat_mul(scale, input, context, 1.0, output);
    } else {
        output.assign(input);
    }
}
```

**Usage in hot paths:**
- Replace `let out = input.apply_context()` with:
  ```rust
  let mut out = /* pre-allocated buffer */;
  ctx.apply_context_into(input, &mut out);
  ```

**Expected gain:** 20-30% faster mixing, no allocations

---

## Testing Checklist

For each task:
- [ ] Compilation succeeds
- [ ] All existing tests pass (606 total)
- [ ] New unit tests added for new functionality
- [ ] No performance regressions (use benchmarks)
- [ ] Memory profiling shows expected savings
- [ ] Backward compatibility maintained

---

## Benchmarking

### Before Phase 2
```bash
cargo bench --bench consolidation_bench
# (Currently non-existent, will create in Phase 3)
```

### After each task
```bash
cargo test --lib 2>&1 | tail -5
```

---

## Expected Outcomes

| Task | Memory Saved | Latency Gain | Priority |
|------|--------------|------------|----------|
| Workspace pooling | 25-30% | 5-10% | HIGH |
| Generational buffers | 15-20% | 15-20% | HIGH |
| In-place mixing | 5-10% | 20-30% | MEDIUM |
| **Total Phase 2** | **45-60%** | **40-60%** | - |

---

## Files to Modify/Create

### New Files
- `src/domain/layers/components/adaptive_residuals_workspace.rs`

### Modified Files
- `src/domain/layers/components/adaptive_residuals.rs`
- `src/domain/layers/components/mod.rs` (add new module)
- `src/domain/layers/transformer/block.rs`
- `src/domain/layers/components/attention_context.rs`

---

## Rollback Plan

If any task causes issues:
```bash
git diff src/domain/layers/components/adaptive_residuals.rs  # Review changes
git checkout src/domain/layers/components/adaptive_residuals.rs  # Rollback
```

All changes are atomic per task, easy to revert.

---

## Notes for Next Session

1. **Start with workspace pooling** - Highest impact, lowest risk
2. **Test after each task** - Don't batch changes
3. **Profile memory** - Use `memory_usage_bytes()` helpers added in Phase 1
4. **Document as you go** - Add inline comments for cache logic
5. **Watch for allocation patterns** - Check for unexpected new allocations

---

## Success Criteria for Phase 2

- [ ] All 3 tasks completed
- [ ] 606+ tests passing
- [ ] Memory usage reduced by 40-50% overall (combined with Phase 1)
- [ ] Latency improved by 30%+
- [ ] Zero breaking API changes
- [ ] Checkpoint compatibility maintained

---

## Time Estimate
- Task 1: 2-3 hours
- Task 2: 1-2 hours  
- Task 3: 1 hour
- Testing: 1 hour
- **Total:** 5-7 hours (doable in one focused session)

Good luck! 🚀
