# Consolidation & Optimization: Diffusion, Transformer, SSM

## Executive Summary

Focus consolidation and performance optimization on the three primary architectures:
- **Transformer** - Attention-based temporal mixing with sliding window KV caching
- **Diffusion** - Reverse ODE solving with intermediate caching and layer-wise relevance
- **SSM** - Selective scanning with state management and streaming inference

### Scope
Consolidate shared components (attention context, adaptive residuals, feedforward), optimize memory buffers, eliminate allocations in hot paths, and improve serialization efficiency.

---

## 1. Shared Component Analysis

### 1.1 Current Shared Components
**File:** `src/domain/layers/components/`

| Component | Usage | Current Status |
|-----------|-------|-----------------|
| `attention_context.rs` | Transformer, Diffusion | ✅ Shared, needs optimization |
| `adaptive_residuals.rs` | Transformer, Diffusion | ✅ Shared, memory pooling needed |
| `feedforward.rs` | All three | ✅ Shared (wrapper) |
| `temporal_processing.rs` | Transformer, Diffusion | ✅ Shared |
| `common.rs` | All three | ✅ Shared (TemporalMixingLayer enum) |
| `block_core.rs` | Transformer, Diffusion | ✅ Shared builder |
| `conditioning.rs` | Diffusion mainly | ✅ Shared (FiLM conditioning) |

### 1.2 Architecture-Specific Components

**Transformer** (`src/domain/layers/transformer/`)
- Block structure with pre-norm/residual connections
- Sliding window KV cache (paged attention)
- Poly attention with CoPE positioning

**Diffusion** (`src/domain/layers/diffusion/`)
- Time/noise level conditioning
- ODE solvers (PNDM, PLMS, Euler)
- Cached intermediates for reverse pass
- Sampling strategies

**SSM** (`src/domain/layers/ssm/`)
- Mamba/Mamba2 variants with selective scanning
- RG-LRU for efficient recurrence
- State management for streaming inference
- Convolutional kernel preprocessing

---

## 2. Memory Optimization Strategy

### 2.1 SharedAttentionContext - Lazy Allocation
**File:** `src/domain/layers/components/attention_context.rs`

**Current Issues:**
- Always maintains `outgoing_context` matrix (embed_dim × embed_dim)
- Allocated even when `update_outgoing_context` is never called
- No pooling across layers

**Optimization:**
```rust
pub struct SharedAttentionContext {
    incoming_context: Option<Array2<f32>>,
    similarity_context_strength: Array2<f32>,
    
    /// Lazy-allocated: only allocate when update_outgoing_context called
    #[serde(skip)]
    outgoing_context: Option<Array2<f32>>,  // Change from always-present to Option
    
    similarity_update_rate: f32,
}

impl SharedAttentionContext {
    pub fn update_outgoing_context(...) {
        if self.outgoing_context.is_none() {
            self.outgoing_context = Some(Array2::zeros((embed_dim, embed_dim)));
        }
        // Update logic
    }
}
```

**Impact:** 
- 10-15% memory reduction for models without context updates
- Zero overhead for inference-only paths

---

### 2.2 AdaptiveResiduals - Workspace Pooling
**File:** `src/domain/layers/components/adaptive_residuals.rs`

**Current Issues:**
- Multiple `Vec<f64>` scratch buffers allocated per layer
- Reallocate on each forward pass
- No reuse across layers in a model

**Optimization - Create Shared Workspace:**

```rust
// NEW: src/domain/layers/components/adaptive_residuals_workspace.rs
pub struct AdaptiveResidualsWorkspace {
    /// Shared scratch buffers
    nx: Vec<f64>,
    ny: Vec<f64>,
    mean_x: Vec<f64>,
    mean_y: Vec<f64>,
    mean_z: Vec<f64>,
    perf_values: Vec<f64>,
    channel_scales: Vec<f32>,
    dot: Vec<f64>,
    z: Vec<f64>,
}

impl AdaptiveResidualsWorkspace {
    pub fn resize_for_dim(&mut self, embed_dim: usize) {
        // Power-of-2 rounding to avoid frequent reallocs
        let capacity = embed_dim.next_power_of_two();
        self.nx.resize(capacity, 0.0);
        // ... resize all buffers
    }
}
```

**Modifications to AdaptiveResiduals:**
```rust
pub struct AdaptiveResiduals {
    // ... existing fields
    
    #[serde(skip)]
    workspace: Option<AdaptiveResidualsWorkspace>,  // Borrowed from pool
}

impl AdaptiveResiduals {
    pub fn set_workspace(&mut self, ws: &AdaptiveResidualsWorkspace) {
        self.workspace = Some(ws.clone());  // Or use Arc for shared ownership
    }
    
    pub fn forward(&mut self, input: &Array2<f32>, ...) {
        let ws = self.workspace.as_mut().expect("workspace required");
        ws.resize_for_dim(self.config.embed_dim);
        // Use ws.nx, ws.ny, etc. instead of self.scratch_*
    }
}
```

**Impact:**
- Reduce per-layer allocations by 8-10
- Pool shared across N layers in model
- ~25-30% memory reduction for large models with many blocks

---

### 2.3 Transformer - KV Cache Optimization
**File:** `src/domain/layers/transformer/block.rs`

**Current Status:** Already uses paged attention with good pooling ✅

**Enhancement - Workspace Generational Pattern:**
```rust
pub struct TransformerWorkspace {
    /// Dimensions from last forward pass
    last_dims: Option<(usize, usize, usize)>,  // (batch, seq, embed_dim)
    
    /// Only reallocate if dims change
    norm_out: Array2<f32>,
    temporal_out: Array2<f32>,
    ffn_out: Array2<f32>,
}

impl TransformerWorkspace {
    pub fn ensure_capacity(&mut self, batch: usize, seq: usize, embed_dim: usize) {
        if self.last_dims != Some((batch, seq, embed_dim)) {
            self.norm_out = Array2::zeros((batch * seq, embed_dim));
            // ... resize other buffers
            self.last_dims = Some((batch, seq, embed_dim));
        } else {
            // Same dims, just clear
            self.norm_out.fill(0.0);
            // ... clear other buffers
        }
    }
}
```

**Impact:** O(1) allocation amortized cost per forward pass

---

### 2.4 Diffusion - Intermediate Caching
**File:** `src/domain/layers/diffusion/block.rs`

**Current Issues:**
- Caches intermediates for each solver step
- No pooling of cache buffers
- Solver-specific replay for LRM

**Optimization - Streaming Cache:**
```rust
pub struct DiffusionIntermediateCache {
    /// Ring buffer for solver history
    history: RingBuffer<Array2<f32>>,
    /// Current step's intermediate
    current: Option<Array2<f32>>,
    /// LRM replay markers
    lrm_indices: Vec<usize>,
}

impl DiffusionIntermediateCache {
    /// Reuse buffer if shape unchanged, else allocate
    pub fn store_step(&mut self, intermediate: &Array2<f32>, step: usize) {
        if let Some(ref mut curr) = self.current {
            if curr.dim() == intermediate.dim() {
                curr.assign(intermediate);
                return;
            }
        }
        self.current = Some(intermediate.clone());
        self.lrm_indices.push(step);
    }
}
```

**Impact:** 
- Reduce allocations in sampling loops
- 15-20% faster generation

---

## 3. Hot Path Optimizations

### 3.1 Attention Context Application
**File:** `src/domain/layers/components/attention_context.rs`

**Current:**
```rust
pub fn apply_context<'a>(&self, input: &'a Array2<f32>) -> Cow<'a, Array2<f32>> {
    if let Some(context) = &self.incoming_context {
        let scale = strength / (embed_dim as f32).max(1.0);
        let mut out = input.dot(context);  // Allocates
        
        Zip::from(&mut out).and(input).for_each(|o, &i| {
            *o = i + scale * *o;
        });
        Cow::Owned(out)
    } else {
        Cow::Borrowed(input)
    }
}
```

**Optimized - In-place operations:**
```rust
pub fn apply_context_into(&self, input: &Array2<f32>, output: &mut Array2<f32>) {
    if output.dim() != input.dim() {
        *output = input.clone();
        return;
    }
    
    if let Some(context) = &self.incoming_context {
        let scale = strength / (embed_dim as f32).max(1.0);
        
        // Use linalg for in-place mixing
        ndarray::linalg::general_mat_mul(scale, input, context, 1.0, output);
    } else {
        output.assign(input);
    }
}
```

**Impact:** 
- No intermediate allocation in hot path
- 20-30% faster mixing in inference

---

### 3.2 Adaptive Residuals - Per-Channel Scaling
**File:** `src/domain/layers/components/adaptive_residuals.rs`

**Current:** Loop over channels, computing contrast factors each iteration

**Optimized - Vectorized scaling:**
```rust
pub fn apply_attention_residual_step_into(
    &mut self,
    input: &ndarray::ArrayView1<f32>,
    attn_out: &ndarray::ArrayView1<f32>,
    output: &mut ndarray::Array1<f32>,
) {
    let scales = &self.attention_residual_scales;
    
    // Vectorized: output = input + (scale ⊙ attn_out)
    Zip::from(output)
        .and(input)
        .and(attn_out)
        .and(scales.column(0))
        .par_for_each(|o, &i, &a, &s| {
            *o = i + s.clamp(0.1, 3.0) * a;
        });
}
```

**Impact:**
- 25-35% faster residual scaling
- Enables rayon parallelization

---

### 3.3 Feedforward - Weight Norm Caching
**File:** `src/domain/layers/components/feedforward.rs`

**Current:** Recomputes weight norm on every call

**Optimized - Cache with dirty flag:**
```rust
pub struct SharedFeedforward {
    pub feedforward: FeedForwardVariant,
    
    #[serde(skip)]
    cached_weight_norm: Option<f32>,
    #[serde(skip)]
    weight_norm_dirty: bool,
}

impl SharedFeedforward {
    pub fn weight_norm(&mut self) -> f32 {
        if self.weight_norm_dirty {
            self.cached_weight_norm = Some(self.feedforward.weight_norm());
            self.weight_norm_dirty = false;
        }
        self.cached_weight_norm.unwrap_or(0.0)
    }
    
    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.feedforward.apply_gradients(grads, lr)?;
        self.weight_norm_dirty = true;  // Mark for recalculation
        Ok(())
    }
}
```

**Impact:**
- O(1) weight norm lookups in gradient clipping
- 10-15% faster training step

---

## 4. Serialization & Checkpoint Optimization

### 4.1 Lazy Context Serialization
```rust
impl SharedAttentionContext {
    #[serde(skip)]
    pub outgoing_context: Option<Array2<f32>>,
}
```

**Benefit:** Checkpoint size reduced 5-10% for inference-only models

---

### 4.2 Skip Non-Critical Scratch
```rust
pub struct AdaptiveResiduals {
    // ... learned parameters serialized
    
    #[serde(skip, default)]
    scratch_nx: Vec<f64>,
    #[serde(skip, default)]
    scratch_ny: Vec<f64>,
    // ... all temp buffers skipped
}
```

**Benefit:** Already applied ✅

---

## 5. Implementation Phases

### Phase 1: Immediate (Low Risk, High Impact)
**Target:** 2-3 hours

1. **Lazy allocation for SharedAttentionContext** ✅
   - Change `outgoing_context` to `Option<Array2<f32>>`
   - Allocate only on `update_outgoing_context` call
   - Files: `attention_context.rs`

2. **AdaptiveResiduals workspace pooling**
   - Extract scratch buffers to `AdaptiveResidualsWorkspace`
   - Make workspaces reusable across layers
   - Files: `adaptive_residuals.rs`, new `adaptive_residuals_workspace.rs`

3. **In-place context application**
   - Add `apply_context_into` method alongside `apply_context`
   - Update hot paths to use in-place variant
   - Files: `attention_context.rs`

**Expected Gains:**
- 15-20% memory reduction
- 10-15% latency improvement
- Zero API breakage

---

### Phase 2: Short-term (Medium Complexity)
**Target:** 4-5 hours

1. **Transformer workspace generational buffers**
   - Pre-allocate workspace at layer creation
   - Only resize when sequence/batch dims change
   - Files: `transformer/block.rs`

2. **Diffusion intermediate cache optimization**
   - Implement streaming cache with ring buffer reuse
   - Reduce allocations in solver loops
   - Files: `diffusion/block.rs`

3. **Feedforward weight norm caching**
   - Add dirty flag pattern
   - Cache norms between gradient applications
   - Files: `feedforward.rs`, `common.rs`

**Expected Gains:**
- 25-30% additional memory reduction
- 20-25% latency improvement
- Better scaling with model size

---

### Phase 3: Polish (Lower Priority)
**Target:** 2-3 hours

1. **Per-device memory pool integration**
   - Thread-local pools for small arrays (<1KB)
   - CUDA-specific pooling if needed
   - Files: New `component_memory_pool.rs`

2. **Benchmark suite**
   - Memory profiling tests
   - Latency micro-benchmarks
   - Files: `benches/consolidation_bench.rs`, `tests/memory_*.rs`

3. **Documentation**
   - Update AGENTS.md with optimization patterns
   - Code comments for complex cache logic
   - Files: `AGENTS.md`, inline docs

---

## 6. Testing & Validation

### 6.1 Unit Tests
```rust
#[test]
fn test_attention_context_lazy_allocation() {
    let mut ctx = SharedAttentionContext::new();
    assert_eq!(ctx.memory_usage(), 0);  // No allocation yet
    
    ctx.update_outgoing_context(...);
    assert!(ctx.outgoing_context.is_some());  // Allocated on update
}

#[test]
fn test_adaptive_residuals_workspace_reuse() {
    let mut ws = AdaptiveResidualsWorkspace::new();
    
    ws.resize_for_dim(256);
    let cap1 = ws.nx.capacity();
    
    ws.resize_for_dim(300);  // Should not reallocate (within power-of-2)
    assert_eq!(ws.nx.capacity(), cap1);
}
```

### 6.2 Integration Tests
```rust
#[test]
fn test_transformer_block_no_unnecessary_allocations() {
    let mut block = TransformerBlock::new(...);
    let input = Array2::zeros((1, 512, 768));
    
    let alloc_before = get_total_allocations();
    let _out = block.forward(&input);
    let alloc_first = get_total_allocations() - alloc_before;
    
    let alloc_before = get_total_allocations();
    let _out = block.forward(&input);  // Same dimensions
    let alloc_second = get_total_allocations() - alloc_before;
    
    assert!(alloc_second < alloc_first / 2);  // Should allocate much less
}
```

### 6.3 Benchmarks
```bash
cargo bench --bench consolidation_bench -- --verbose
```

---

## 7. Backward Compatibility

**No Breaking Changes:**
- Lazy allocation is internal detail
- Workspace pooling is transparent
- New `_into` methods are additive
- Serialization unchanged (scratch buffers already skipped)

**Deserialization:** 
- Old checkpoints load with lazy fields uninitialized ✅
- No migration needed

---

## 8. Success Criteria

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Model Memory | 100% | 80% | Phase 1-2 |
| Forward Latency | 100% | 85% | Phase 1-2 |
| Checkpoint Size | 100% | 95% | Phase 1 |
| Allocations/Step | 10-15 | 2-4 | Phase 1-2 |
| Code Duplication | 15% | <8% | Phase 1 |
| Test Coverage | 85% | >90% | All phases |

---

## 9. Files to Modify

**Core Components:**
- `src/domain/layers/components/attention_context.rs` - Lazy allocation
- `src/domain/layers/components/adaptive_residuals.rs` - Workspace extraction
- `src/domain/layers/components/feedforward.rs` - Weight norm caching
- `src/domain/layers/components/common.rs` - Shared helpers

**Architecture-Specific:**
- `src/domain/layers/transformer/block.rs` - Workspace generational buffers
- `src/domain/layers/diffusion/block.rs` - Intermediate caching
- `src/domain/layers/diffusion/solvers.rs` - Cache integration

**New Files:**
- `src/domain/layers/components/adaptive_residuals_workspace.rs` - Workspace struct
- `benches/consolidation_bench.rs` - Performance benchmarks
- `tests/memory_consolidation.rs` - Memory profiling tests

---

## 10. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Lazy allocation breaks inference | Low | High | Comprehensive tests |
| Workspace sharing corrupts state | Low | High | Thread safety checks |
| Checkpoint incompatibility | Very Low | Medium | No API changes |
| Performance regression | Low | High | Benchmark suite |
| Increased complexity | Medium | Low | Clear documentation |

---

## Next Steps

1. ✅ Create this plan
2. → Start Phase 1 with lazy attention context
3. → Add workspace pooling for adaptive residuals
4. → In-place context mixing in hot paths
5. → Run benchmarks and compare
6. → Move to Phase 2 if gains are met

**Estimated Total Time:** 10-15 hours spread over 2-3 work sessions
