# Consolidation & Optimization Action Plan
**Session**: Phase 2 Continuation | **Date**: 2026-02-12

## Overview
Complete the consolidation and memory efficiency optimizations for shared components between Diffusion, SSM, and Transformer architectures. Focus on activating unused infrastructure and reducing hot-path allocations.

---

## Phase 1: Workspace Activation (HIGH PRIORITY)

### 1.1 Activate TransformerBlock Batch Workspace
**File**: `src/domain/layers/transformer/block.rs`
**Current State**: `batch_workspace` field exists but is never used in `forward()`
**Impact**: Eliminates 4 major allocations per forward pass (norm, mix, residual, FFN)

**Tasks**:
- [x] Modify `TransformerBlock::forward()` to call `batch_workspace.ensure_capacity(seq_len, embed_dim)`
- [ ] Replace inline `Arc::new(norm1_out)` calls with pre-allocated buffers
- [ ] Route intermediate computations through workspace buffers:
  - `norm_scratch` ← norm1 output
  - `mix_scratch` ← temporal mixing output  
  - `residual_scratch` ← residual1 (input + mix_out)
  - `ffn_scratch` ← FFN output
- [ ] Wrap final results in Arc for cached_intermediates
- [ ] Measure allocation reduction via criterion benchmark

**Status**: Workspace capacity initialization added. Next: integrate buffer reuse for Arc allocation pooling.

**Pseudo-code**:
```rust
// In forward():
let seq_len = input.nrows();
let embed_dim = self.config.embed_dim;

// Ensure workspace has capacity
let workspace = self.batch_workspace.get_or_insert_with(|| {
    TransformerWorkspace::new(seq_len, embed_dim)
});
workspace.ensure_capacity(seq_len, embed_dim);

// Use buffers:
let norm1_out = {
    let buf = workspace.norm_buffer();
    self.pre_attention_norm.forward_into(&input_used, buf);
    buf.clone() // Or keep as view
};
```

---

### 1.2 Integrate AdaptiveResiduals Workspace
**File**: `src/domain/layers/components/adaptive_residuals.rs`
**Current State**: Creates 9 scratch buffers internally per instance
**Issue**: Each layer allocates these independently; no sharing across layers
**Target**: Reduce allocations by ~75% per training step

**Tasks**:
- [x] Update `AdaptiveResiduals` to accept optional `AdaptiveResidualsWorkspace`
- [x] Add workspace field and accessor methods to AdaptiveResiduals struct
- [x] Refactor `apply_attention_residual_step_into()` to use workspace.channel_scales
- [ ] Refactor remaining forward methods to use workspace buffers:
  - `update_similarity_sketch()` → reuse workspace.nx, ny, mean_x, mean_y
  - All corr/covar computations → reuse workspace.dot matrix
- [x] Add `new_with_workspace()` method to `AdaptiveResiduals::new()`
- [x] Implement fallback logic for backward compatibility (uses internal scratch if no workspace)
- [ ] Modify compute_gradients to use workspace instead of scratch_*
- [ ] Create model-level workspace pool in `LLMModel` or training context
- [ ] Update TransformerBlock to pass workspace to AdaptiveResiduals

**Status**: Foundation laid (struct field, constructors, buffer getters). Next: integrate remaining forward paths.

**Workspace Allocation Savings**:
- Before: 9 vecs × 12 layers × 8-24 bytes = ~10-30 KB per model
- After: 1 shared workspace = ~1-2 KB (reused across 12+ layers)

---

## Phase 2: Hot-Path Optimization (MEDIUM PRIORITY)

### 2.1 Eliminate Intermediate Allocations in Context Mixing
**File**: `src/domain/layers/components/attention_context.rs`
**Current State**: `apply_context()` uses `.dot()` which allocates intermediate array
**Code Location**: Lines 115, 362, 380
**Issue**: `.dot()` creates temporary array for result
**Solution**: Use `general_mat_mul()` with beta=0.0 to reuse output buffer

**Tasks**:
- [ ] Profile current allocations with `valgrind --tool=massif`
- [ ] Replace line 115 in `update_outgoing_context()`:
  ```rust
  // Before: let cov = sub_x.t().dot(&sub_y);
  // After:
  let mut cov = Array2::zeros((embed_dim, embed_dim));
  ndarray::linalg::general_mat_mul(1.0, &sub_x.t(), &sub_y, 0.0, &mut cov);
  ```
- [ ] Same for lines 362, 380 in gradient computation
- [ ] Benchmark: expect 20-30% faster mixing at scale

---

### 2.2 Cache Weight Norms with Dirty Flags
**Files**: 
- `src/domain/layers/components/adaptive_residuals.rs`
- `src/domain/layers/components/attention_context.rs`
**Current State**: `weight_norm()` recomputed every call (iterates all params)
**Cost**: ~10μs per layer per call in models with 12+ layers = measurable overhead
**Solution**: Cache with dirty flag cleared on `apply_gradients()`

**Tasks**:
- [ ] Add to `SharedAttentionContext`:
  ```rust
  cached_weight_norm: Option<f32>,
  weight_norm_dirty: bool,
  ```
- [ ] Modify `weight_norm()` to check dirty flag
- [ ] Set dirty=true in `set_strength()` and setters
- [ ] Set dirty=false after `apply_gradients()` completes
- [ ] Same pattern for `AdaptiveResiduals`

**Expected Savings**: ~0.5-1% wall-clock reduction in training (for norm-heavy models)

---

## Phase 3: Memory Efficiency (MEDIUM PRIORITY)

### 3.1 Optimize Scratch Buffer Sizing in AdaptiveResidualsWorkspace
**File**: `src/domain/layers/components/adaptive_residuals_workspace.rs`
**Current State**: Power-of-2 rounding (line 68) may over-allocate
**Issue**: embed_dim=768 → rounds to 1024 (25% waste)
**Solution**: Hybrid approach with small fixed overhead

**Tasks**:
- [ ] Modify `resize_for_dim()`:
  ```rust
  // For large dims, round to next 64 (instead of power-of-2)
  let new_capacity = if embed_dim > 256 {
      ((embed_dim + 63) / 64) * 64  // Round to nearest 64
  } else {
      embed_dim.next_power_of_two().max(32)  // Power-of-2 for small
  };
  ```
- [ ] Reduces 768→768 (0% waste), 768→1024→820 for matrix (overhead only)
- [ ] Benchmark memory reduction for typical models

### 3.2 Lazy Allocation for Diffusion Intermediates
**File**: `src/domain/layers/diffusion.rs` (if exists)
**Target**: Defer allocation of ODE solver reverse-pass scratch until needed
**Pattern**: Match `SharedAttentionContext.outgoing_context` (Option<Array2>)

**Tasks**:
- [ ] Identify largest allocations in diffusion forward/backward
- [ ] Wrap in `Option<>` with lazy init
- [ ] Estimate memory saved (likely 2-5 MB for larger diffusion models)

---

## Phase 4: Code Cleanup (LOW PRIORITY)

### 4.1 Remove Unused scratch_* from AdaptiveResiduals
**File**: `src/domain/layers/components/adaptive_residuals.rs` lines 80-98
**Status**: After workspace integration, these can be removed
**Tasks**:
- [ ] Remove fields: scratch_nx, scratch_ny, scratch_mean_x, ..., scratch_z
- [ ] Verify all compute paths use workspace
- [ ] Update serde skip directives

### 4.2 Consolidate Context Setters
**File**: `src/domain/layers/components/attention_context.rs` lines 154-196
**Observation**: Three similar methods for setting incoming context
**Refactor**:
- [ ] Merge `set_incoming_context()` and `set_incoming_context_reuse()` 
- [ ] Single method with optional reuse logic
- [ ] Reduces API surface, clarifies intent

---

## Validation & Testing

### 4.1 Unit Tests
- [ ] `TransformerWorkspace::ensure_capacity()` reuses buffers correctly
- [ ] `AdaptiveResidualsWorkspace` allocates only once per dimension
- [ ] No buffer overflow when seq_len or embed_dim increases

### 4.2 Integration Tests
- [ ] Model trains with workspace pooling (loss trajectory unchanged)
- [ ] Memory profiling shows expected reduction (use criterion + memtrack)
- [ ] No performance regression in latency

### 4.3 Benchmarks to Update
**Files**: `benches/`
- [ ] `transformer_block_forward.rs` → expect 5-15% faster
- [ ] `adaptive_residuals_forward.rs` → expect 10-20% less memory
- [ ] Add new benchmark: workspace reuse over 10 forward passes

---

## Success Criteria

| Optimization | Target Reduction | Measurement |
|---|---|---|
| Workspace Activation | 4 allocations/fwd | `criterion` batch benchmark |
| Scratch Integration | 75% of AdaptiveRes alloc | `valgrind --tool=massif` |
| Context Hot-Path | 20-30% mixing time | `criterion` attention sub-bench |
| Weight Norm Caching | 0.5-1% training wall-clock | `criterion` full epoch |
| Memory Overall | 10-20 MB @ 768D × 12L | Peak RSS in training logs |

---

## Implementation Order

1. **Workspace Activation** (1-2 hours): Highest ROI, tests clear
2. **Context Hot-Path** (30-45 min): Profile-driven, low risk
3. **Scratch Integration** (1.5-2 hours): Moderate complexity, measurable gain
4. **Weight Norm Caching** (30-45 min): Cleanup pattern reuse
5. **Memory Optimization** (1 hour): Fine-tuning, lower priority

**Estimated Total**: ~5-6 hours of engineering + testing

---

## Blockers / Risks

- [ ] `batch_workspace` initialization in cloned blocks (line 138) - may need Arc<Mutex>
- [ ] `AdaptiveResidualsWorkspace` lifetime management if model held in parallel contexts
- [ ] Benchmark noise: allocator variance may mask small gains; use `--sample-size 100`

---

## References

- Previous consolidation report: `CONSOLIDATION_DIFFUSION_TRANSFORMER_SSM.md`
- Workspace pattern: `src/domain/layers/transformer/block.rs` lines 379-470
- Shared components: `src/domain/layers/components/`
