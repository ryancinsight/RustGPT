# Phase 5 Implementation Roadmap - Detailed Technical Plan

**Timeline**: Week of Feb 13-20, 2026  
**Current Status**: 476 tests passing, all components operational  
**Target**: 25-30% performance improvement + 50% memory reduction

---

## Priority 1: In-Place Forward Operations (70% of optimization gains)

### 1.1 Temporal Mixing In-Place Operations

**Files to Modify**:
- `src/domain/layers/components/temporal_processing.rs` - Main interface
- `src/domain/attention/poly_attention.rs` - PolyAttention layer
- `src/domain/mamba/` - Mamba and Mamba2 layers
- `src/domain/ssm/rg_lru.rs` - RG-LRU layer
- `src/domain/attention/sliding_window.rs` - SlidingWindowAttention
- `src/domain/attention/ring_attention.rs` - RingAttention

**Step 1.1.1: SharedTemporalProcessing Interface**

Add to `src/domain/layers/components/temporal_processing.rs`:

```rust
impl SharedTemporalProcessing {
    /// Forward pass with in-place output
    /// 
    /// Reuses provided output buffer, eliminating intermediate allocations.
    /// Expected to save 40 KB/step per layer.
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        self.prepare_forward();
        self.temporal_mixing.forward_into(input, output)
    }

    /// Forward with causal masking and in-place output
    pub fn forward_with_causal_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        causal: bool,
    ) -> Result<()> {
        self.prepare_forward();
        self.temporal_mixing.forward_with_causal_into(input, output, causal)
    }

    /// Forward with FiLM and in-place output
    pub fn forward_with_film_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        gamma: Option<&Array1<f32>>,
        beta: Option<&Array1<f32>>,
        causal: bool,
    ) -> Result<()> {
        let conditioned = apply_optional_delta_film(
            input,
            gamma.map(|g| g.view()),
            beta.map(|b| b.view()),
        );
        self.forward_with_causal_into(conditioned.as_ref(), output, causal)
    }
}
```

**Step 1.1.2: TemporalMixingLayer Trait**

Modify `src/domain/layers/components/common.rs` to add to the enum:

```rust
impl TemporalMixingLayer {
    /// Dispatch forward_into to the appropriate implementation
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        match self {
            Self::Attention(layer) => layer.forward_into(input, output),
            Self::Titans(layer) => layer.forward_into(input, output),
            Self::Mamba(layer) => layer.forward_into(input, output),
            Self::Mamba2(layer) => layer.forward_into(input, output),
            Self::RgLru(layer) => layer.forward_into(input, output),
            Self::PolyAttention(layer) => layer.forward_into(input, output),
            Self::SlidingWindow(layer) => layer.forward_into(input, output),
            Self::RingAttention(layer) => layer.forward_into(input, output),
        }
    }

    /// With causal variant
    pub fn forward_with_causal_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        causal: bool,
    ) -> Result<()> {
        match self {
            Self::Attention(layer) => layer.forward_with_causal_into(input, output, causal),
            // ... etc for other variants
        }
    }
}
```

**Step 1.1.3: PolyAttention Implementation**

In `src/domain/attention/poly_attention.rs`, add:

```rust
impl PolyAttention {
    /// Forward pass with in-place output
    /// 
    /// Processes input and writes results directly to output buffer.
    /// No intermediate allocations beyond workspace buffers.
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let (batch_size, seq_len, embed_dim) = (input.nrows(), input.ncols(), input.ncols());
        
        // Ensure output has correct shape
        if output.dim() != (batch_size, embed_dim) {
            *output = Array2::zeros((batch_size, embed_dim));
        }

        // Use workspace from streaming_workspace or allocate locally
        let workspace = self.ensure_workspace(batch_size, seq_len, embed_dim);
        
        // 1. Project inputs (Q, K, V) into workspace buffers
        // No allocation: use workspace pools
        
        // 2. Compute polynomial attention scores
        // Reuse workspace for intermediate scores
        
        // 3. Apply gating and output projection
        // Write directly to output buffer
        
        Ok(())
    }

    /// With causal masking
    pub fn forward_with_causal_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        causal: bool,
    ) -> Result<()> {
        self.set_causal_masking(causal);
        self.forward_into(input, output)
    }
}
```

**Verification**:
- Add test: `test_temporal_processing_forward_into_matches_forward()`
- Profile memory allocation count
- Benchmark: expect 8-10% speedup per layer

---

### 1.2 Feedforward In-Place Operations

**Files to Modify**:
- `src/domain/layers/components/feedforward.rs` - Main wrapper
- `src/domain/richards/glu.rs` - RichardsGLU
- `src/domain/mixtures/moe.rs` - MixtureOfExperts

**Implementation**:

```rust
impl SharedFeedforward {
    /// Forward pass with in-place output
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        self.feedforward.forward_into(input, output)
    }

    /// With FiLM conditioning
    pub fn forward_with_film_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
        gamma: Option<&Array1<f32>>,
        beta: Option<&Array1<f32>>,
    ) -> Result<()> {
        let conditioned = apply_optional_delta_film(
            input,
            gamma.map(|g| g.view()),
            beta.map(|b| b.view()),
        );
        self.feedforward.forward_into(conditioned.as_ref(), output)
    }
}

// In FeedForwardVariant enum
impl FeedForwardVariant {
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        match self {
            Self::RichardsGlu(layer) => layer.forward_into(input, output),
            Self::MixtureOfExperts(layer) => layer.forward_into(input, output),
        }
    }
}
```

**Verification**:
- Add test: `test_feedforward_forward_into_matches_forward()`
- Benchmark FFN only: expect 3-5% speedup

---

### 1.3 Block-Level Integration

**Files to Modify**:
- `src/domain/layers/transformer/block.rs` - TransformerBlock
- `src/domain/layers/diffusion/block.rs` - DiffusionBlock
- `src/domain/layers/ssm/block.rs` - SSMBlock (if exists)

**Pattern**:

```rust
impl TransformerBlock {
    /// Forward with unified workspace (in-place operations)
    pub fn forward_with_unified_workspace(
        &mut self,
        input: &Array2<f32>,
        workspace: &mut UnifiedLayerWorkspace,
    ) -> Result<Array2<f32>> {
        let batch_size = input.nrows();
        let seq_len = input.ncols();
        let embed_dim = self.embed_dim;

        // Ensure workspace capacity
        workspace.ensure_capacity(batch_size, seq_len, embed_dim);

        // Pre-attention norm (in-place)
        self.pre_attention_norm.forward_into(
            input,
            workspace.norm1_out_mut().unwrap(),
        );

        // Temporal mixing (in-place)
        self.temporal_mixing.forward_into(
            workspace.norm1_out().unwrap(),
            workspace.temporal_out_mut().unwrap(),
        )?;

        // Residual addition (in-place)
        add_residual_inplace(
            workspace.norm1_out().unwrap(),
            workspace.temporal_out_mut().unwrap(),
        );

        // Pre-FFN norm (in-place)
        self.pre_ffn_norm.forward_into(
            workspace.temporal_out().unwrap(),
            workspace.norm2_out_mut().unwrap(),
        );

        // FFN (in-place)
        self.feedforward.forward_into(
            workspace.norm2_out().unwrap(),
            workspace.ffn_out_mut().unwrap(),
        )?;

        // Final residual (in-place)
        add_residual_inplace(
            workspace.temporal_out().unwrap(),
            workspace.ffn_out_mut().unwrap(),
        );

        // Return output
        Ok(workspace.ffn_out().unwrap().clone())
    }
}

// Helper for residual addition
#[inline]
fn add_residual_inplace(
    residual: &Array2<f32>,
    output: &mut Array2<f32>,
) {
    use ndarray::Zip;
    Zip::from(output)
        .and(residual)
        .for_each(|o, &r| *o += r);
}
```

**Verification**:
- Integration test: `test_transformer_block_forward_with_workspace()`
- Validate output matches old forward()
- Memory profiling: measure allocation reduction

---

## Priority 2: GlobalBufferPool Implementation (20% of optimization gains)

**Files to Create/Modify**:
- `src/domain/layers/components/global_buffer_pool.rs` - New file
- `src/domain/models/llm.rs` - LLMModel integration
- `src/common/config.rs` - Configuration

**Step 2.1: GlobalBufferPool Component**

```rust
// File: src/domain/layers/components/global_buffer_pool.rs

use ndarray::Array2;
use std::collections::HashMap;

/// Statistics for buffer pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_acquired: u64,
    pub total_released: u64,
    pub current_buffers: usize,
    pub size_class_distribution: HashMap<usize, (u32, usize)>,
}

/// Global buffer pool for reducing fragmentation
pub struct GlobalBufferPool {
    /// Size class (power-of-2) -> Vec<Buffer>
    pools: HashMap<usize, Vec<Array2<f32>>>,
    
    /// Statistics
    stats: PoolStats,
    
    /// Maximum buffers per size class
    max_per_class: usize,
}

impl GlobalBufferPool {
    pub fn new(max_per_class: usize) -> Self {
        Self {
            pools: HashMap::new(),
            stats: PoolStats {
                total_acquired: 0,
                total_released: 0,
                current_buffers: 0,
                size_class_distribution: HashMap::new(),
            },
            max_per_class,
        }
    }

    /// Acquire buffer of at least `elements` f32 values
    pub fn acquire(&mut self, elements: usize) -> Array2<f32> {
        let size_class = Self::size_class(elements);
        let pool = self.pools.entry(size_class).or_insert_with(Vec::new);

        let buffer = if let Some(buf) = pool.pop() {
            self.stats.current_buffers -= 1;
            buf
        } else {
            // Allocate new buffer (size_class can hold this many elements)
            Array2::zeros((1, size_class))
        };

        self.stats.total_acquired += 1;
        buffer
    }

    /// Release buffer back to pool
    pub fn release(&mut self, buffer: Array2<f32>) {
        let elements = buffer.len();
        let size_class = Self::size_class(elements);
        
        let pool = self.pools.entry(size_class).or_insert_with(Vec::new);
        
        if pool.len() < self.max_per_class {
            pool.push(buffer);
            self.stats.current_buffers += 1;
        }
        
        self.stats.total_released += 1;
    }

    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }

    /// Clear all pooled buffers
    pub fn clear(&mut self) {
        self.pools.clear();
        self.stats.current_buffers = 0;
    }

    /// Compute size class (power-of-2 bucket)
    fn size_class(elements: usize) -> usize {
        if elements <= 1024 {
            1024
        } else {
            elements.next_power_of_two()
        }
    }
}
```

**Step 2.2: LLMModel Integration**

In `src/domain/models/llm.rs`:

```rust
pub struct LLMModel {
    pub layers: Vec<Layer>,
    pub buffer_pool: Option<Arc<Mutex<GlobalBufferPool>>>,
}

impl LLMModel {
    pub fn with_buffer_pool(mut self, max_buffers_per_class: usize) -> Self {
        self.buffer_pool = Some(Arc::new(Mutex::new(
            GlobalBufferPool::new(max_buffers_per_class)
        )));
        self
    }

    pub fn forward_with_pool(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let mut output = input.clone();
        let mut workspace = UnifiedLayerWorkspace::new();

        for layer in &mut self.layers {
            output = layer.forward_with_unified_workspace(&output, &mut workspace)?;
        }

        Ok(output)
    }
}
```

**Verification**:
- Test: `test_global_buffer_pool_acquire_release()`
- Profile fragmentation: compare with/without pool
- Benchmark: expect 5-10% speedup from better cache locality

---

## Priority 3: Gradient Computation Optimization (5% of gains)

**Files to Create/Modify**:
- `src/domain/layers/components/gradient_router.rs` - Add masking
- `src/domain/layers/transformer/block.rs` - Use mask

**Step 3.1: GradientComputeMask**

```rust
// In src/domain/layers/components/gradient_router.rs

#[derive(Debug, Clone)]
pub struct GradientComputeMask {
    pub layer_masks: Vec<bool>, // false = skip gradients
}

impl GradientComputeMask {
    pub fn new(num_layers: usize, default: bool) -> Self {
        Self {
            layer_masks: vec![default; num_layers],
        }
    }

    pub fn set_layer(&mut self, layer_idx: usize, compute: bool) {
        if layer_idx < self.layer_masks.len() {
            self.layer_masks[layer_idx] = compute;
        }
    }

    pub fn should_compute(&self, layer_idx: usize) -> bool {
        self.layer_masks.get(layer_idx).copied().unwrap_or(true)
    }
}
```

---

## Testing & Validation Strategy

### Phase 1: Unit Tests (Per Component)
```rust
#[test]
fn test_temporal_processing_forward_into_matches_forward() {
    let mut tp = create_test_temporal_processing();
    let input = Array2::from_elem((2, 128), 0.5f32);
    
    // Old path
    let result_old = tp.forward(&input);
    
    // New path
    let mut result_new = Array2::zeros(result_old.dim());
    tp.forward_into(&input, &mut result_new).unwrap();
    
    // Compare (within numerical tolerance)
    assert_close(&result_old, &result_new, 1e-5);
}

#[test]
fn test_feedforward_forward_into_matches_forward() {
    let mut ff = create_test_feedforward();
    let input = Array2::from_elem((2, 256), 0.5f32);
    
    let result_old = ff.forward(&input);
    
    let mut result_new = Array2::zeros(result_old.dim());
    ff.forward_into(&input, &mut result_new).unwrap();
    
    assert_close(&result_old, &result_new, 1e-5);
}
```

### Phase 2: Integration Tests (Block-Level)
```rust
#[test]
fn test_transformer_block_with_unified_workspace() {
    let mut block = create_test_transformer_block();
    let mut workspace = UnifiedLayerWorkspace::new();
    let input = Array2::from_elem((2, 64, 128), 0.5f32);
    
    let result = block.forward_with_unified_workspace(&input, &mut workspace).unwrap();
    
    // Validate shape and values
    assert_eq!(result.shape(), [2, 64, 128]);
    assert!(result.iter().all(|x| x.is_finite()));
}
```

### Phase 3: Performance Benchmarks
```bash
cargo bench --bench layer_forward_comparison
# Expected: 10-15% speedup for in-place ops
```

### Phase 4: Memory Profiling
```bash
valgrind --tool=massif --massif-out-file=massif.out cargo test --lib
ms_print massif.out
```

---

## Rollout Plan

### Week 1 (Feb 13-20)
- [ ] Day 1-2: Implement SharedTemporalProcessing forward_into()
- [ ] Day 2-3: Implement PolyAttention forward_into()
- [ ] Day 3: Implement SharedFeedforward forward_into()
- [ ] Day 4: Block-level integration
- [ ] Day 5: Full test suite validation

### Week 2 (Feb 20-27)
- [ ] GlobalBufferPool implementation
- [ ] LLMModel integration
- [ ] Comprehensive benchmarking
- [ ] Performance profiling

### Week 3 (Feb 27-Mar 5)
- [ ] Gradient optimization (optional)
- [ ] Documentation updates
- [ ] Final validation
- [ ] Merge to main

---

## Success Metrics

### Performance
- [ ] 10-15% inference speedup (per-layer)
- [ ] 25-30% total model speedup
- [ ] 40 KB/step memory reduction per layer

### Quality
- [ ] All 476 tests still passing
- [ ] No performance regressions
- [ ] < 3 compiler warnings

### Coverage
- [ ] 500+ integration tests
- [ ] All temporal mixing variants tested
- [ ] All FFN variants tested

---

## Risk Mitigation

**Risk**: Numerical instability in in-place operations
**Mitigation**: Rigorous tolerance testing, keep old path available

**Risk**: Memory leaks in GlobalBufferPool
**Mitigation**: Rust's ownership system, comprehensive tests

**Risk**: Performance regression in certain scenarios
**Mitigation**: Benchmark suite with regression detection

---

## References

- Base implementation: CONSOLIDATION_COMPONENTS_MANIFEST.md
- Previous work: T-019c5596-85e3-710a-a671-a28d6bf3db20
- Architecture guide: OPTIMIZATION_PATTERNS_GUIDE.md
- Build commands: AGENTS.md
