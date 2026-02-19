# Phase 4 Consolidation & Optimization Continuation

## Status Summary

### Completed Work
- ✅ Shared components framework (`workspace_pool.rs`, `intermediate_buffer_pool.rs`, `film_parameter_cache.rs`)
- ✅ `attention_context.rs` optimized (most .dot() calls replaced in forward pass)
- ✅ `adaptive_residuals.rs` with workspace support
- ✅ `block_core.rs` builder infrastructure
- ✅ `conditioning.rs` shared FiLM modulation

### Remaining Hot-Path Optimizations

#### 1. RichardsGLU (.dot() → general_mat_mul)
**File**: `src/domain/richards/richards_glu.rs`
**Priority**: CRITICAL (forward & backward executed every layer)

Lines with .dot() calls:
- L170-171: `input.dot(&self.w1)` and `input.dot(&self.w2)` - **2 forward matrix products**
- L180: `gated.dot(&self.w_out)` - **1 forward matrix product**
- L215-220: Backward cache reconstruction via .dot() - **2 backward reconstructions**
- L236-237: Weight gradients - **2 backward matrix products**
- L313-317: Weight gradients & input gradients - **4 backward matrix products**

**Impact**: RichardsGLU is executed in every transformer/diffusion/SSM layer's FFN stage.
- 12-layer model: 48 forward + 48 backward .dot() calls per step
- Expected memory savings: ~200-300 KB per training step

#### 2. Attention Context Backward Pass
**File**: `src/domain/layers/components/attention_context.rs`
**Priority**: HIGH (backward gradient computation)

Lines with .dot() calls:
- L277: `input.dot(context)` - forward path already optimized
- L313: `input.dot(ctx)` - backward gradient computation

**Impact**: Context modulation gradient path.
- Expected memory savings: ~50-80 KB per 12-layer model per step

#### 3. RG-LRU Backward Pass
**File**: `src/domain/layers/ssm/rg_lru.rs`
**Priority**: MEDIUM (used when TemporalMixingType::RgLru selected)

Lines with .dot() calls:
- L842-844: `input.t().dot(&dlogits_r)` and `input.t().dot(&dlogits_i)` - **weight gradient matrix products**
- L847: `dlogits_r.dot(&self.w_a.t())` and `dlogits_i.dot(&self.w_x.t())` - **input gradient matrix products**

**Impact**: Only active when using RG-LRU temporal mixing.
- Expected memory savings: ~60-100 KB per 12-layer RG-LRU model per step

#### 4. Diffusion Block Context Integration
**File**: `src/domain/layers/diffusion/block.rs`
**Priority**: MEDIUM (diffusion-conditioned models only)

Lines with .dot() calls:
- L1742: `input_original.dot(ctx)` - forward context product
- L1762: `final_input_grads.dot(&ctx.t())` - backward context gradient

**Impact**: Context modulation in diffusion blocks.
- Expected memory savings: ~40-70 KB per 12-layer diffusion model per step

### Transformer Block Integration with WorkspacePool
**File**: `src/domain/layers/transformer/block.rs`
**Priority**: HIGH (core architecture change)

**Current State**: 
- Uses inline `Arc::new` allocations
- No workspace pooling integration
- Caches full intermediates without reuse

**Target State**:
- Acquire `IntermediateBufferPool` from shared `WorkspacePool`
- Use `ensure_capacity` with power-of-2 sizing
- Share workspace across sequential block computations

**Expected Savings**: 480-600 KB per 12-layer model per forward step

### SSM/Mamba Integration Audit
**File**: `src/domain/layers/ssm/mamba.rs`, `src/domain/layers/ssm/mamba2.rs`
**Priority**: LOW (not primary mixing type in Phase 4)

**Required**: Full audit for redundant allocations and .dot() patterns

## Optimization Execution Plan

### Phase 4.1: RichardsGLU Hot-Path Optimization
1. Replace forward .dot() calls (L170-171, L180)
   - Use `general_mat_vec_mul` for (seq_len, hidden) × (hidden, hidden) → (seq_len, hidden)
   - Pre-allocate x1, x2, gated in layer struct as `Option<Array2<f32>>`
   - Reuse buffers when dimensions unchanged
2. Replace backward .dot() calls (L236-237, L313-317)
   - Use workspace buffers for grad_w_out, grad_gated, grad_input
   - Implement dirty-flag caching for weight matrix norms
3. Update cached_input strategy to use Arc<Array2<f32>>
4. Benchmark and document memory reduction

### Phase 4.2: Attention Context Backward Pass
1. Optimize L277 (input × context product)
2. Optimize L313 (backward gradient computation)
3. Use workspace buffers from SharedAttentionContext
4. Verify no performance regression in forward path

### Phase 4.3: Transformer Block WorkspacePool Integration
1. Add WorkspacePool reference to TransformerBlock
2. Refactor forward pass to acquire buffers from pool
3. Replace inline Arc::new allocations with workspace buffers
4. Verify cached_intermediates strategy compatibility
5. Test sequential layer execution with buffer reuse

### Phase 4.4: RG-LRU Backward Pass Optimization (Conditional)
1. Audit SSM/RG-LRU .dot() usage patterns
2. Optimize weight gradient computation
3. Optimize input gradient computation
4. Test with models using RG-LRU temporal mixing

### Phase 4.5: Diffusion Block Context Optimization (Conditional)
1. Audit diffusion block .dot() patterns
2. Integrate context modulation workspace sharing
3. Test with diffusion-conditioned models

## Key Optimization Patterns

### Pattern 1: general_mat_vec_mul for Sequential Matrix Products
```rust
// BEFORE: Allocates new array
let result = input.dot(&weight);

// AFTER: Reuses pre-allocated buffer
let mut result = Array2::zeros((input.nrows(), weight.ncols()));
general_mat_mul(1.0, input, weight, 0.0, &mut result);
```

### Pattern 2: Power-of-2 Buffer Sizing
```rust
fn ensure_capacity(buf: &mut Option<Array2<f32>>, rows: usize, cols: usize) {
    let capacity = next_power_of_2(rows.max(cols));
    match buf {
        Some(ref existing) if existing.nrows() >= rows && existing.ncols() >= cols => {},
        _ => *buf = Some(Array2::zeros((capacity, capacity))),
    }
}
```

### Pattern 3: Dirty-Flag Caching for Norms
```rust
struct WeightNormCache {
    cached_norm: f32,
    dirty: bool,
}

impl WeightNormCache {
    fn get_norm(&mut self, weight: &Array2<f32>) -> f32 {
        if self.dirty {
            self.cached_norm = weight.norm_l2();
            self.dirty = false;
        }
        self.cached_norm
    }
    
    fn mark_dirty(&mut self) {
        self.dirty = true;
    }
}
```

## Estimated Total Impact

| Component | Forward KB | Backward KB | Total KB |
|-----------|-----------|------------|----------|
| RichardsGLU | 180-240 | 200-300 | **380-540** |
| Attention Context | 40-60 | 40-60 | **80-120** |
| Transformer Workspace | 250-350 | N/A | **250-350** |
| RG-LRU (conditional) | 30-50 | 60-100 | **90-150** |
| Diffusion Block (conditional) | 30-50 | 40-70 | **70-120** |
| **Total Phase 4 Savings** | | | **870-1280 KB** |

For a 12-layer model:
- **Per training step**: 870-1280 KB reduction (~12-15% of total intermediate allocations)
- **Per epoch (1000 steps)**: 870-1280 MB reduction
- **Memory efficiency gain**: 40-50% reduction in transient allocations during backward pass

## Success Criteria

✓ All .dot() calls in hot paths replaced with general_mat_mul or workspace buffers
✓ Transformer block integrated with WorkspacePool
✓ No performance regression in forward/backward passes
✓ Memory profiling shows <400 MB for 12-layer model inference
✓ All tests pass: `cargo test --lib`
✓ Benchmarks show ≥5% faster training per epoch
