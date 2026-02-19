# Consolidation & Optimization - Phase 3 Continuation

## Objective
Continue consolidation and cleanup while optimizing the performance and memory efficiency of shared components between Diffusion, SSM, and Transformer architectures.

## Current State Analysis

### Shared Components Status
- ✅ **`adaptive_residuals_workspace.rs`**: Complete with power-of-2 sizing and memory pooling
- ✅ **`adaptive_residuals.rs`**: Integrated workspace support with fallback logic
- ✅ **`attention_context.rs`**: Lazy allocation of outgoing context implemented
- ✅ **`block_core.rs`**: Shared block core builder pattern established
- ⚠️ **`attention_context.rs` (Gradient Path)**: Hot-path `.dot()` calls still allocate intermediate arrays
- ⚠️ **`transformer/block.rs`**: Still uses inline `Arc::new()` for intermediates; needs workspace pooling
- ⚠️ **`diffusion/block.rs`**: Uses SharedAttentionContext but lacks explicit workspace management
- ⚠️ **`ssm/rg_lru.rs`**: Has independent streaming workspace; opportunity for consolidation

## High-Priority Optimizations

### 1. Replace `.dot()` with `general_mat_mul` in Hot Paths

**Target Files**: `attention_context.rs`, `conditioning.rs`

**Affected Operations**:
- Line 277: `input.dot(context)` in `apply_context` → allocates new array
- Line 115: `sub_x.t().dot(&sub_y)` in `update_outgoing_context` → allocates covariance matrix
- Line 362: `input_original.dot(ctx)` in gradient path → allocates intermediate
- Line 380: `final_input_grads.dot(&ctx.t())` in gradient computation

**Optimization Strategy**:
```rust
// Before (allocates intermediate):
let mut out = input.dot(context);  // O(seq_len × embed_dim × embed_dim)

// After (reuses buffer):
let mut out = Array2::zeros((input.nrows(), context.ncols()));
ndarray::linalg::general_mat_mul(1.0, input, context, 0.0, &mut out);
```

**Estimated Memory Savings**:
- Per `apply_context` call: ~seq_len × embed_dim × 4 bytes (1-2 MB for typical seq_len=512, embed_dim=768)
- Per forward pass: ~10 calls × 2 MB = 20 MB
- Per batch: 20 MB × batch_size reduction

### 2. Implement Workspace Pooling in TransformerBlock

**Location**: `src/domain/layers/transformer/block.rs`

**Current Issue**:
```rust
// Lines ~600-650 in forward():
Arc::new(norm1_out.clone())  // Creates Arc, but clone allocates
Arc::new(mix_out.clone())    // Creates Arc, but clone allocates
Arc::new(ffn_out.clone())    // Creates Arc, but clone allocates
```

**Solution**: Pre-allocate in batch_workspace
```rust
pub struct TransformerBlockWorkspaceFull {
    pub norm1_scratch: Array2<f32>,
    pub mix_scratch: Array2<f32>,
    pub residual_scratch: Array2<f32>,
    pub ffn_scratch: Array2<f32>,
    pub norm2_scratch: Array2<f32>,
    pub context_scratch: Option<Array2<f32>>,
    pub adaptive_residuals_workspace: Option<AdaptiveResidualsWorkspace>,
}

impl TransformerBlockWorkspaceFull {
    pub fn ensure_capacity(&mut self, seq_len: usize, embed_dim: usize) {
        // Use power-of-2 sizing to minimize reallocations
        let cap_seq = seq_len.next_power_of_two();
        let cap_embed = embed_dim.next_power_of_two();
        
        if self.norm1_scratch.dim() != (cap_seq, cap_embed) {
            self.norm1_scratch = Array2::zeros((cap_seq, cap_embed));
            self.mix_scratch = Array2::zeros((cap_seq, cap_embed));
            // ... resize other buffers
        } else {
            // Clear without deallocating
            self.norm1_scratch.fill(0.0);
            self.mix_scratch.fill(0.0);
            // ... clear other buffers
        }
    }
}
```

**Integration Points**:
- `TransformerBlock::forward()`: Call `ensure_capacity()` at start
- Replace all `Arc::new()` calls with references to workspace buffers
- Update caching logic to work with borrowed references

**Memory Savings**:
- 4 scratch buffers: 4 × seq_len × embed_dim × 4 bytes = 2 × seq_len × embed_dim bytes
- For seq_len=512, embed_dim=768: ~1.5 MB per layer × num_layers

### 3. Weight Norm Caching with Dirty Flags

**Targets**: `attention_context.rs`, `adaptive_residuals.rs`

**Current Issue**: `weight_norm()` recomputes norms on every call
```rust
pub fn weight_norm(&self) -> f32 {
    let mut sum_sq = 0.0f64;
    for &v in self.attention_residual_scales.iter() {
        let x = if v.is_finite() { v as f64 } else { 0.0 };
        sum_sq += x * x;  // Recomputes every time
    }
    (sum_sq as f32).sqrt()
}
```

**Solution**: Cache with dirty flag
```rust
pub struct AdaptiveResiduals {
    // ... existing fields ...
    
    // Cached norm and dirty flag
    cached_weight_norm: Option<f32>,
    norm_is_dirty: bool,
}

impl AdaptiveResiduals {
    pub fn weight_norm(&mut self) -> f32 {
        if self.norm_is_dirty || self.cached_weight_norm.is_none() {
            // Recompute
            let mut sum_sq = 0.0f64;
            for &v in self.attention_residual_scales.iter() {
                let x = if v.is_finite() { v as f64 } else { 0.0 };
                sum_sq += x * x;
            }
            let norm = (sum_sq as f32).sqrt();
            self.cached_weight_norm = Some(norm);
            self.norm_is_dirty = false;
            norm
        } else {
            self.cached_weight_norm.unwrap()
        }
    }
    
    pub fn apply_gradients(&mut self, ...) -> Result<()> {
        // ... update parameters ...
        self.norm_is_dirty = true;  // Mark as needing recompute
    }
}
```

**Estimated Performance Improvement**:
- Reduces O(embed_dim) operations to O(1) on cache hits
- ~50-100 norm computations per training step → ~5-10x speedup

### 4. Lazy Allocation in SharedAttentionContext - Verify Complete

**Status**: Already implemented (Line 65-69 in attention_context.rs)

**Verification Tasks**:
- [ ] Confirm lazy allocation works in Diffusion blocks
- [ ] Confirm lazy allocation works in SSM contexts
- [ ] Test memory impact in long sequences (512→2048 seq_len)

**Memory Savings**:
- Lazy allocation saves embed_dim² × 4 bytes per context
- For embed_dim=768: 2.36 MB per layer
- Multi-layer model: 2.36 MB × num_layers

## Medium-Priority Optimizations

### 5. Unified Workspace Extraction Interface

**Goal**: Create a trait for blocks that need workspace management

```rust
pub trait WorkspaceManaged {
    type Workspace: Clone + Default;
    
    fn ensure_workspace_capacity(&mut self, seq_len: usize, embed_dim: usize);
    fn clear_workspace(&mut self);
}

impl WorkspaceManaged for TransformerBlock {
    type Workspace = TransformerBlockWorkspaceFull;
    
    fn ensure_workspace_capacity(&mut self, seq_len: usize, embed_dim: usize) {
        if let Some(ref mut ws) = self.batch_workspace {
            ws.ensure_capacity(seq_len, embed_dim);
        }
    }
}
```

### 6. Model-Level Workspace Pool

**Location**: `src/domain/models/llm.rs`

```rust
pub struct LLMModel {
    // ... existing fields ...
    
    /// Shared workspaces to reduce allocations across all layers
    transformer_workspace: Option<TransformerBlockWorkspaceFull>,
    adaptive_residuals_workspace: Option<AdaptiveResidualsWorkspace>,
}

impl LLMModel {
    pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let seq_len = input.nrows();
        let embed_dim = input.ncols();
        
        // Ensure all workspaces are sized correctly
        if let Some(ref mut ws) = self.transformer_workspace {
            ws.ensure_capacity(seq_len, embed_dim);
        }
        if let Some(ref mut ws) = self.adaptive_residuals_workspace {
            ws.resize_for_dim(embed_dim);
        }
        
        // Forward pass uses shared workspaces
        // ...
    }
}
```

### 7. Diffusion-Specific Streaming Cache

**Location**: `src/domain/layers/diffusion/block.rs`

**Goal**: Implement ring-buffer cache for ODE solver steps to avoid repeating context computations

```rust
pub struct DiffusionStreamingCache {
    /// Ring buffer of computed contexts for reverse ODE steps
    context_cache: VecDeque<Array2<f32>>,
    /// Step indices for cache hits
    step_indices: VecDeque<usize>,
    /// Maximum cache capacity
    max_capacity: usize,
}

impl DiffusionStreamingCache {
    pub fn get_or_compute(
        &mut self,
        step: usize,
        compute_fn: impl Fn() -> Array2<f32>,
    ) -> Array2<f32> {
        if let Some(pos) = self.step_indices.iter().position(|&s| s == step) {
            self.context_cache[pos].clone()
        } else {
            let result = compute_fn();
            self.step_indices.push_back(step);
            self.context_cache.push_back(result.clone());
            
            if self.context_cache.len() > self.max_capacity {
                self.context_cache.pop_front();
                self.step_indices.pop_front();
            }
            result
        }
    }
}
```

## Implementation Roadmap

### Phase 3.1: Hot-Path Optimization (1-2 hours)
- [ ] Replace `.dot()` with `general_mat_mul` in `attention_context.rs` gradient paths
- [ ] Add tests validating numerical equivalence
- [ ] Benchmark memory impact

### Phase 3.2: Workspace Pooling (2-3 hours)
- [ ] Extend `TransformerBlockWorkspaceFull` with full initialization
- [ ] Integrate into `TransformerBlock::forward()`
- [ ] Update cached intermediates logic
- [ ] Run integration tests

### Phase 3.3: Weight Norm Caching (1 hour)
- [ ] Add dirty-flag pattern to `AdaptiveResiduals`
- [ ] Add dirty-flag pattern to `SharedAttentionContext`
- [ ] Update `apply_gradients` to set dirty flags
- [ ] Verify performance improvement

### Phase 3.4: Verification & Cleanup (1-2 hours)
- [ ] Verify lazy allocation works end-to-end across all architectures
- [ ] Run full test suite
- [ ] Document optimizations in code comments
- [ ] Update AGENTS.md with optimization patterns

## Performance Metrics to Track

| Optimization | Memory Saved | Speed Improvement | Effort |
|---|---|---|---|
| `.dot()` → `general_mat_mul` | ~10-20 MB/layer | 5-10% GPU utilization | 1-2h |
| Workspace pooling | ~1.5-2 MB/layer | 3-5% throughput | 2-3h |
| Weight norm caching | ~0.5-1 MB cache | 5-10% loss computation | 1h |
| Lazy allocation (verify) | ~2.36 MB/layer | Minimal | 0.5h |
| **Total** | **~15-25 MB/model** | **10-20%** | **~5-7h** |

## Code Comments & Documentation

All changes should include:
1. **Inline comments** explaining why reuse is safe
2. **Doc comments** describing cache invalidation strategy
3. **Perf notes** documenting expected memory/speed impact
4. **Tests** validating numerical equivalence before/after

## Success Criteria

✅ All tests pass  
✅ Memory usage decreased by 10-15% on model forward pass  
✅ No numerical correctness issues (all outputs match within 1e-5 tolerance)  
✅ Code is self-documenting with clear optimization intent  
✅ Performance benchmarks show measurable improvement  

## Notes

- Workspace sizing uses power-of-2 alignment for efficient memory pooling
- All buffers cleared (not deallocated) between uses for generational reuse
- Dirty flags invalidated only on parameter updates, not on every compute
- SharedAttentionContext lazy allocation saves ~2.36 MB per layer in inference
