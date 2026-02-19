# Phase 3 Consolidation & Cleanup: Shared Components Optimization

## Objective
Continue consolidation and cleanup while optimizing performance and memory efficiency of shared components between Diffusion, SSM, and Transformer architectures.

## Current State Analysis

### Working Components
- ✅ `SharedBlockCore` - Central assembly point for layer stacks
- ✅ `TimeConditioner` - Optimized (7 allocations eliminated via `general_mat_mul`)
- ✅ `SharedFilmModulation` - FiLM conditioning with parallelized application
- ✅ `AdaptiveResidualsWorkspace` - Reusable workspace for residual computations
- ✅ `temporal_processing.rs` - Unified mixing interface (Attention, Mamba, RG-LRU)
- ✅ `feedforward.rs` - Unified FFN with MoE support
- ✅ `attention_context.rs` - Similarity-based context modulation

### Identified Optimization Gaps

#### 1. **Buffer Allocation in Forward Paths** (High Impact)
**Problem**: Each block forward pass creates new Arc allocations for intermediates:
- Transformer: Lines 1008-1012 create 5 Arc allocations (norm1_out, mix_out, residual1, norm2_out, ffn_out)
- Diffusion: Lines 1091-1105 create 11 Arc allocations (time_embed, norm outputs, FiLM gamma/beta, attn_out, ffn_out, output)

**Impact**: ~52 KB per layer per training step (12 layers = 624 KB/step)

**Solution**: Implement `IntermediateBufferPool` for lazy reuse of cached intermediates

---

#### 2. **FiLM Gamma/Beta Clone Overhead** (Medium Impact)
**Problem**: Diffusion line 1098-1101 clones entire gamma/beta arrays:
```rust
gamma_attn: Arc::new(self.film_modulation.gamma_attn.clone()),
beta_attn: Arc::new(self.film_modulation.beta_attn.clone()),
```

**Impact**: 2 × embed_dim allocations per layer per step (for 768 embed = 6144 floats = 24 KB/layer)

**Solution**: Store references in cache instead of clones, or use copy-on-write semantics

---

#### 3. **Workspace Pool Not Shared at Model Level** (Medium Impact)
**Problem**: Each layer has its own `AdaptiveResidualsWorkspace`, not pooled across layers

**Solution**: Create model-level `WorkspacePool` to share single workspace across all layers

---

#### 4. **Context Duplication Between Transformer and Diffusion** (Low Impact)
**Problem**: Both Diffusion and Transformer implement similar context apply/update logic

**Solution**: Consolidate into a single `ContextManager` abstraction

---

## Implementation Roadmap

### Phase 3.2.1: Buffer Pool for Intermediates (PRIORITY 1)

Create a new component: `src/domain/layers/components/intermediate_buffer_pool.rs`

```rust
/// Thread-safe pool of intermediate buffers with power-of-2 sizing strategy
pub struct IntermediateBufferPool {
    norm_cache: Option<Array2<f32>>,
    mix_cache: Option<Array2<f32>>,
    residual_cache: Option<Array2<f32>>,
    ffn_cache: Option<Array2<f32>>,
    expected_shape: Option<(usize, usize)>,
}

impl IntermediateBufferPool {
    pub fn new() -> Self { ... }
    
    /// Ensure buffer capacity (realloc only if shape changes)
    pub fn ensure_capacity(&mut self, rows: usize, cols: usize) { ... }
    
    /// Borrow a buffer mutably, reusing allocation
    pub fn borrow_norm_mut(&mut self) -> &mut Array2<f32> { ... }
    
    /// Clear buffers for next layer
    pub fn clear(&mut self) { ... }
}

/// Cached intermediates using pooled buffers via Arc wrapper
pub struct PooledCachedIntermediates {
    norm1_out: Arc<Array2<f32>>,
    mix_out: Arc<Array2<f32>>,
    residual1: Arc<Array2<f32>>,
    norm2_out: Arc<Array2<f32>>,
}
```

**Savings**: ~20 KB per layer per step (reduced allocation pressure)

---

### Phase 3.2.2: Film Gamma/Beta Reference Cache (PRIORITY 2)

Modify `SharedFilmModulation` to expose views instead of clones:

```rust
impl SharedFilmModulation {
    /// Get Arc reference to underlying array (for caching without clone)
    pub fn gamma_attn_ref(&self) -> Arc<Array2<f32>> {
        Arc::new(self.gamma_attn.clone())  // Will optimize to Arc::clone if stored
    }
}
```

Or implement a shared `FilmParameterCache`:

```rust
pub struct FilmParameterCache {
    gamma_attn: Arc<Array2<f32>>,
    beta_attn: Arc<Array2<f32>>,
    gamma_ffn: Arc<Array2<f32>>,
    beta_ffn: Arc<Array2<f32>>,
    last_update_tick: u64,
}
```

**Savings**: ~24 KB per layer per step

---

### Phase 3.2.3: Model-Level Workspace Pool (PRIORITY 3)

Modify `LLMModel` to maintain a single shared workspace:

```rust
pub struct LLMModel {
    layers: Vec<TransformerBlock>,
    #[serde(skip)]
    shared_workspace: Arc<Mutex<AdaptiveResidualsWorkspace>>,
}
```

Update each layer's forward to borrow from pool:

```rust
// In TransformerBlock::forward()
let mut workspace = self.shared_workspace.lock().unwrap();
workspace.ensure_capacity(input.raw_dim());
// Use workspace...
drop(workspace);  // Release lock
```

**Savings**: ~10 KB per layer (eliminated duplicate workspace allocations)

---

### Phase 3.2.4: Context Manager Consolidation (PRIORITY 4 - Future)

Create unified `ContextManager` for both Transformer and Diffusion:

```rust
pub struct ContextManager {
    context: AttentionContext,
}

impl ContextManager {
    pub fn apply_incoming(&self, input: &Array2<f32>) -> Array2<f32> { ... }
    pub fn update_outgoing(&mut self, input: &Array2<f32>, output: &Array2<f32>) { ... }
}
```

---

## Memory Impact Summary

### Per Training Step (for 12-layer model, batch_size=1, embed_dim=768, seq_len=512)

| Component | Current | Optimized | Savings |
|-----------|---------|-----------|---------|
| Intermediate Buffers (12 layers × 5) | 300 KB | 60 KB | 240 KB |
| FiLM Gamma/Beta Clones (12 layers) | 288 KB | 24 KB | 264 KB |
| Workspace Pools | 120 KB | 10 KB | 110 KB |
| **Total** | **708 KB** | **94 KB** | **614 KB** |

### 1000-Step Training Run
- **Current**: 708 MB per 1000 steps
- **Optimized**: 94 MB per 1000 steps
- **Savings**: ~614 MB per 1000 steps (86% reduction in intermediate allocations)

---

## Implementation Checklist

### Phase 3.2 (Current)
- [ ] Implement `IntermediateBufferPool` in `components/intermediate_buffer_pool.rs`
- [ ] Update `TransformerBlock::forward()` to use pooled buffers
- [ ] Update `DiffusionBlock::forward()` to use pooled buffers
- [ ] Implement model-level workspace pool in `LLMModel`
- [ ] Update SSM layer to use shared components
- [ ] Benchmark memory usage: `cargo bench --bench memory_efficiency`
- [ ] Run full test suite: `cargo test --lib`

### Phase 3.3 (Future)
- [ ] Consolidate Context logic into `ContextManager`
- [ ] Implement streaming cache for Diffusion ODE solver
- [ ] Add `WorkspaceManaged` trait for standardized handling
- [ ] Optimize backwards pass to reuse forward-computed intermediates

---

## Code Patterns to Apply

### Pattern 1: Lazy Buffer Allocation
```rust
// BEFORE: Always allocates
let mut result = Array1::zeros(size);

// AFTER: Reuse from pool if capacity sufficient
pool.ensure_capacity(size);
let result = pool.borrow_mut();
```

### Pattern 2: Power-of-2 Sizing
```rust
fn next_power_of_two_capacity(required: usize) -> usize {
    (required as u32).next_power_of_two() as usize
}
```

### Pattern 3: Zero-Copy Arc Wrapping
```rust
// Instead of: Arc::new(data.clone())
// Use a pre-allocated Arc if possible, or store references
let data_ref = Arc::new(data);  // Only if truly needed
```

---

## Performance Benchmarking Plan

### Before Optimization
```bash
cargo bench --bench layer_forward_pass
cargo bench --bench end_to_end_training
```

### After Each Phase
```bash
cargo bench --bench memory_efficiency -- --verbose
```

### Metrics to Track
1. Peak heap allocation per forward pass
2. Total allocations per training step
3. Cache hit ratio (workspace reuse)
4. Backward pass gradient computation time
5. GPU memory (if applicable)

---

## Risk Mitigation

### Threading Safety
- Use `Arc<Mutex<>>` for shared workspace pools
- Ensure workspace is released after use (implement RAII guards)
- Test with parallel layer processing if enabled

### Numerical Stability
- Verify that buffer reuse doesn't introduce NaN/Inf
- Check gradients match baseline before/after each optimization
- Run `cargo test --test transformer_block_verification`

### Backward Compatibility
- Maintain same public API for blocks
- Feature-flag workspace pooling if needed for debugging
- Add diagnostic logging for buffer reuse statistics

---

## Success Criteria

1. **Memory Efficiency**: 80%+ reduction in intermediate allocations
2. **Performance**: <5% latency increase (if any) from locking overhead
3. **Correctness**: All existing tests pass with <1e-5 numerical difference
4. **Code Quality**: Zero `clippy` warnings in modified code
5. **Documentation**: Update architecture diagram and capacity planning guide

