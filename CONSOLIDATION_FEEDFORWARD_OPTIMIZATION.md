# Feedforward Component Consolidation & Optimization

## Status: Phase 5 Consolidation - Feedforward Optimization

Date: Feb 13, 2026  
Thread: @T-019c56e1-5d51-725b-a8f7-608ea73bdb2e

---

## Overview

This phase optimizes the shared feedforward components across Transformer, Diffusion, and SSM blocks. The focus is on eliminating redundant allocations, improving memory efficiency, and unifying workspace patterns.

### Current Architecture

```
SharedFeedforward
├── FeedForwardVariant::RichardsGlu
│   ├── w1, w2, w_out matrices
│   ├── Streaming workspace (token-by-token)
│   └── Batch workspace (unused/stub)
└── FeedForwardVariant::MixtureOfExperts
    ├── Expert layers
    ├── Router networks
    ├── Streaming workspace
    └── Batch workspace (unused/stub)
```

---

## Memory Efficiency Issues

### Issue 1: RichardsGlu `forward_into()` is Not Truly In-Place
**File**: `src/domain/richards/richards_glu.rs:547-572`

```rust
// Current: Creates intermediate allocation, then copies
pub(crate) fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    let result = self.forward(input);  // <-- Allocates full output
    output.assign(&result);             // <-- Copies (wasteful)
    Ok(())
}
```

**Impact**: Defeats purpose of `forward_into()` by allocating intermediate buffer.

**Solution**: Implement true in-place computation by reusing batch_workspace buffers.

### Issue 2: MixtureOfExperts `forward_into()` is Not Truly In-Place
**File**: `src/domain/mixtures/moe.rs:1794-1819`

Same issue as RichardsGlu.

### Issue 3: Unused Batch Workspace Buffers
**RichardsGlu**: Has `batch_workspace: Option<RichardsGluBatchWorkspace>` (never initialized)  
**MixtureOfExperts**: Has workspace buffers (never pre-allocated for batch ops)

### Issue 4: SharedFeedforward Doesn't Manage Workspace Lifecycle
**File**: `src/domain/layers/components/feedforward.rs`

```rust
pub struct SharedFeedforward {
    pub feedforward: FeedForwardVariant,
    // Missing: workspace pools, reusable buffers
}
```

No capacity management, no workspace clearing, no memory pooling.

### Issue 5: FiLM Modulation Clones During Forward Pass
**File**: `src/domain/layers/components/feedforward.rs:125-142`

```rust
pub fn forward_with_film(...) -> Array2<f32> {
    let conditioned = apply_optional_delta_film(...);  // <-- Allocates
    self.forward_with_token_head_activity(
        conditioned.as_ref(),  // <-- Uses allocation
        ...
    )
}
```

Creates intermediate Cow<> that may allocate.

---

## Optimization Plan

### Phase 5.1.1: Implement True In-Place RichardsGlu Batch Forward

**File**: `src/domain/richards/richards_glu.rs`

1. **Define RichardsGluBatchWorkspace**:
   ```rust
   pub struct RichardsGluBatchWorkspace {
       x1: Array2<f32>,           // [batch, hidden]
       x2: Array2<f32>,           // [batch, hidden]
       value: Array2<f32>,        // [batch, hidden]
       gate_sigma: Array2<f32>,   // [batch, hidden]
       gated: Array2<f32>,        // [batch, hidden]
   }
   ```

2. **Implement in-place forward using workspace**:
   ```rust
   pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
       let (batch_size, embed_dim) = input.dim();
       let hidden_dim = self.w1.ncols();
       
       // Lazy initialize workspace with power-of-2 capacity
       self.ensure_batch_workspace(batch_size, hidden_dim);
       let ws = self.batch_workspace.as_mut().unwrap();
       
       // All computations happen in-place using workspace buffers
       // x1 = input @ W1
       general_mat_mul(1.0, input, &self.w1, 0.0, &mut ws.x1);
       // ... rest of computation ...
       // result stored in output buffer directly
       
       Ok(())
   }
   ```

3. **Benefits**:
   - Eliminates intermediate allocation
   - Reuses buffers across batch calls
   - ~50% memory reduction for forward pass

### Phase 5.1.2: Implement True In-Place MixtureOfExperts Batch Forward

**File**: `src/domain/mixtures/moe.rs`

Similar approach: pre-allocate expert computation buffers, reuse across batches.

### Phase 5.1.3: Add Workspace Management to SharedFeedforward

**File**: `src/domain/layers/components/feedforward.rs`

```rust
pub struct SharedFeedforward {
    pub feedforward: FeedForwardVariant,
    
    // NEW: Workspace management
    workspace_cache: Option<FeedforwardWorkspace>,
    last_batch_size: usize,
    last_embed_dim: usize,
}

impl SharedFeedforward {
    pub fn ensure_workspace(&mut self, batch_size: usize, embed_dim: usize) -> Result<()> {
        // Initialize/resize workspace if needed (power-of-2 sizing)
        if self.workspace_cache.is_none() || 
           self.last_batch_size != batch_size ||
           self.last_embed_dim != embed_dim {
            // Trigger underlying variant's workspace allocation
            match &mut self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => {
                    layer.ensure_batch_workspace(batch_size, ...);
                }
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.ensure_batch_workspace(batch_size, ...);
                }
            }
        }
        Ok(())
    }
    
    pub fn clear_workspace(&mut self) {
        // Clear internal caches while keeping allocations
        self.workspace_cache = None;
        // delegate to variant
    }
}
```

### Phase 5.1.4: Eliminate FiLM Intermediate Allocations

**File**: `src/domain/layers/components/feedforward.rs:125-142`

Add in-place variant:
```rust
pub fn forward_with_film_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
    gamma: Option<&Array1<f32>>,
    beta: Option<&Array1<f32>>,
    head_activity_ratio: Option<f32>,
    head_activity_vec: Option<&[f32]>,
    token_head_activity_vec: Option<&[f32]>,
) -> Result<()> {
    // Apply FiLM directly to input in-place or using workspace buffer
    // Then forward_into() to output
    Ok(())
}
```

### Phase 5.1.5: Profile & Benchmark

- Measure allocation count before/after
- Track memory peak usage
- Benchmark latency improvement

---

## Expected Memory Savings

| Component | Before (12 layers) | After | Savings |
|-----------|-------------------|-------|---------|
| RichardsGlu intermediate buffers | 96 KB | 0 KB (reused) | 100% |
| MoE expert temp buffers | 48 KB | 0 KB (reused) | 100% |
| FiLM intermediate Cow | 24 KB | 0 KB (in-place) | 100% |
| **Total per forward pass** | **168 KB** | **0 KB** | **100%** |

**Cumulative over 1000 inference steps**: ~165 MB freed

---

## Implementation Roadmap

```
Week 1: 
  - [ ] RichardsGlu batch workspace + forward_into
  - [ ] MoE batch workspace + forward_into
  - [ ] Tests + benchmarks

Week 2:
  - [ ] SharedFeedforward workspace management
  - [ ] FiLM in-place optimization
  - [ ] Integration with UnifiedLayerWorkspace
  - [ ] E2E testing (TransformerBlock, DiffusionBlock)

Week 3:
  - [ ] Profile & optimize hot paths
  - [ ] Update documentation
  - [ ] Final integration tests
```

---

## Success Criteria

1. ✅ All 484 unit tests pass
2. ✅ No intermediate allocations in forward_into() paths
3. ✅ Memory reduction >= 100 KB per forward pass
4. ✅ Latency improvement >= 5% in batch inference
5. ✅ Workspace reuse ratio >= 95% (>= 95% of calls reuse pre-allocated buffers)

---

## Files to Modify

1. `src/domain/richards/richards_glu.rs` - Batch workspace + in-place forward
2. `src/domain/mixtures/moe.rs` - Batch workspace + in-place forward
3. `src/domain/layers/components/feedforward.rs` - Workspace management
4. `src/domain/layers/components/unified_layer_workspace.rs` - Integration hook
5. Tests: Add benchmarks for memory efficiency

---

## Related Documentation

- Thread: @T-019c56e1-5d51-725b-a8f7-608ea73bdb2e (Phase 5 consolidation)
- Phase 5 summary: CONSOLIDATION_PHASE5_COMPLETION_REPORT_FEB13_2026.md
- Pattern reference: RgLru streaming workspace in `src/domain/layers/ssm/rg_lru.rs`
