# Phase 5.1c: Block-Level In-Place Integration

**Status**: Immediate implementation phase  
**Timeline**: 2-3 hours (development + testing)  
**Target**: 5-8% per-layer speedup via batch path optimization

---

## Overview

Phase 5.1a-b completed component-level `forward_into()` methods for all temporal and feedforward layers. Phase 5.1c integrates these into the block-level batch forward paths, eliminating redundant allocations at the orchestration level.

**Expected Impact**:
- Memory: 20-30 KB/step savings (block-level allocation elimination)
- Speed: 5-8% per-layer improvement via reduced GC pressure
- No API changes, backward compatible

---

## 1. TransformerBlock::forward() Integration

### Current State (Batch Path - lines 763-917)

**Key allocations to eliminate**:
1. `input.clone()` (line 788) - Input duplication for context application
2. `norm1_out` temporary allocation - Could reuse workspace buffer
3. `mix_out` temporary allocation - Could reuse workspace buffer  
4. `residual1` clone + addition (lines 870-872) - Can use pre-allocated buffer
5. `norm2_out` temporary allocation - Could reuse workspace buffer
6. `ffn_out` allocation + clone (line 879 + line 895) - Can chain forward_into
7. Multiple Arc wrapping/cloning for caching (lines 887-899)

**Current flow**:
```
input (Arc clone)
  → context.apply_context() → Cow (allocated if modified)
  → norm1 → norm1_out (new allocation)
  → temporal mixing → mix_out (new allocation) 
  → residual + normalization → norm2_out (new allocation)
  → feedforward → ffn_out (new allocation)
  → final residual → output (reuse ffn_out)
```

### Optimized Flow (In-Place)

```rust
// Pre-allocate workspace buffers (already allocated once in ensure_capacity)
let mut norm1_out = workspace.take_norm1_out();      // Pre-allocated
let mut mix_out = workspace.take_temporal_out();     // Pre-allocated
let mut residual1 = workspace.take_residual1();      // Pre-allocated
let mut norm2_out = workspace.take_norm2_out();      // Pre-allocated
let mut ffn_out = workspace.take_ffn_out();          // Pre-allocated

// Forward pass using in-place operations
// 1. Context application (avoid clone when possible)
let input_effective = if let Cow::Owned(owned) = context.apply_context(...) {
    // Use owned; reuse in-place
    owned
} else {
    // Borrowed; reference only
    input.clone() // Only fallback clone
};

// 2. Normalization (into workspace)
self.pre_attention_norm.forward_into(&input_effective, &mut norm1_out);

// 3. Temporal mixing (into workspace)
self.temporal_mixing.forward_into(&norm1_out, &mut mix_out);

// 4. Residual connection (in-place)
residual1.assign(&mix_out);  // Copy mix_out
residual1 += &input_effective;  // Add residual

// 5. Normalization (into workspace)
self.pre_ffn_norm.forward_into(&residual1, &mut norm2_out);

// 6. Feedforward (into workspace)
self.feedforward.forward_into(&norm2_out, &mut ffn_out);

// 7. Final residual (reuse ffn_out)
ffn_out += &residual1;

// Return workspace buffers
workspace.return_buffers(norm1_out, mix_out, residual1, norm2_out, ffn_out);
output = ffn_out;
```

### Implementation Steps

#### Step 1: Add workspace buffer access methods to UnifiedLayerWorkspace

**File**: `src/domain/layers/components/unified_layer_workspace.rs`

```rust
impl UnifiedLayerWorkspace {
    /// Take mutable ownership of norm1_out for forward_into operations
    pub fn take_norm1_out(&mut self) -> Array2<f32> {
        self.norm1_out.take().unwrap_or_else(|| Array2::zeros((1, 1)))
    }
    
    /// Return norm1_out to workspace
    pub fn return_norm1_out(&mut self, buf: Array2<f32>) {
        self.norm1_out = Some(buf);
    }
    
    // Similar for: temporal_out, residual1, norm2_out, ffn_out
    // ... (5 more pairs of take/return methods)
    
    /// Batch take for all required buffers
    pub fn take_all_buffers(&mut self) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        (
            self.take_norm1_out(),
            self.take_temporal_out(),
            self.take_residual1(),
            self.take_norm2_out(),
            self.take_ffn_out(),
        )
    }
    
    /// Batch return for all buffers
    pub fn return_all_buffers(
        &mut self,
        norm1_out: Array2<f32>,
        temporal_out: Array2<f32>,
        residual1: Array2<f32>,
        norm2_out: Array2<f32>,
        ffn_out: Array2<f32>,
    ) {
        self.return_norm1_out(norm1_out);
        self.return_temporal_out(temporal_out);
        self.return_residual1(residual1);
        self.return_norm2_out(norm2_out);
        self.return_ffn_out(ffn_out);
    }
}
```

#### Step 2: Add forward_into helper methods to component layers

**File**: `src/domain/layers/components/normalization.rs` or add to RichardsNorm

```rust
impl RichardsNorm {
    /// Forward pass writing directly into output buffer
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
        let result = self.forward(input);
        *output = result;  // Assign result into pre-allocated buffer
    }
}
```

**File**: `src/domain/layers/transformer/block.rs` - Verify these exist:
- `SharedTemporalProcessing::forward_into()` ✅ Already implemented
- `SharedFeedforward::forward_into()` ✅ Already implemented

#### Step 3: Refactor TransformerBlock::forward() to use in-place ops

**File**: `src/domain/layers/transformer/block.rs` (lines 763-917)

**Key changes**:
1. Take workspace buffers after ensure_capacity
2. Replace temporary allocations with workspace buffers
3. Use forward_into where available
4. Minimize Arc cloning
5. Return buffers before exit

**Pattern** (see example below):

```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // ... existing setup code (titan_memory reset, ensure_capacity, etc.) ...
    
    // NEW: Take workspace buffers
    let (mut norm1_out, mut mix_out, mut residual1, mut norm2_out, mut ffn_out) = 
        self.unified_workspace.take_all_buffers();
    
    // Context application (same logic, but track input reference)
    let input_original_arc = Arc::new(input.clone());
    let input_used_arc = match self.context.apply_context(input_original_arc.as_ref()) {
        Cow::Borrowed(_) => input_original_arc.clone(),
        Cow::Owned(owned) => Arc::new(owned),
    };
    
    // Normalization INTO workspace (rather than allocating new)
    self.pre_attention_norm.forward_into(&input_used_arc, &mut norm1_out);
    
    // Temporal mixing INTO workspace
    self.temporal_mixing.forward_into(&norm1_out, &mut mix_out);
    
    // Update context
    self.context.update_outgoing_context(...);
    
    // Residual connection IN-PLACE
    residual1.assign(&mix_out);
    residual1 += &input_used_arc;
    
    // Normalization INTO workspace
    self.pre_ffn_norm.forward_into(&residual1, &mut norm2_out);
    
    // Feedforward INTO workspace
    self.feedforward.forward_into(&norm2_out, &mut ffn_out);
    
    // Final residual IN-PLACE
    ffn_out += &residual1;
    
    // Cache outputs
    *self.cached_intermediates.write().unwrap() = Some(CachedIntermediates {
        input_original: input_original_arc,
        input_used: input_used_arc,
        norm1_out: Arc::new(norm1_out.clone()),  // For backward pass
        mix_out: Arc::new(mix_out.clone()),
        residual1: Arc::new(residual1.clone()),
        norm2_out: Arc::new(norm2_out.clone()),
        ffn_out: Arc::new(ffn_out.clone()),
    });
    
    // NEW: Return workspace buffers
    self.unified_workspace.return_all_buffers(
        norm1_out,
        mix_out,
        residual1,
        norm2_out,
        ffn_out.clone(),  // Clone for return, but reuse copy
    );
    
    ffn_out  // Return output (moved from workspace)
}
```

---

## 2. DiffusionBlock::forward_with_timestep() Integration

### Current State

**File**: `src/domain/layers/diffusion/block.rs` (lines ~800-1100)

**Key allocations**:
1. `x_t.clone()` - Input duplication
2. `time_embed` allocation - Can reuse workspace
3. `film_params` from TimeConditioner
4. Multiple intermediate allocations in feedforward
5. `Arc::new(...)` wrappers for caching

### Optimization Strategy

Similar to TransformerBlock but with diffusion-specific optimizations:

1. Reuse `UnifiedLayerWorkspace` diffusion-specific buffers
2. Use `SharedFeedforward::forward_into()` for feedforward path
3. Apply FiLM modulation in-place where possible
4. Minimize Arc allocations

### Implementation Steps

#### Step 1: Extend UnifiedLayerWorkspace for diffusion buffers

**File**: `src/domain/layers/components/unified_layer_workspace.rs`

Diffusion-specific buffers already defined (lines ~160):
```rust
pub input_buffer: Option<Array2<f32>>,
pub time_embed: Option<Array1<f32>>,
pub film_modulation_scale: Option<Vec<f32>>,
pub film_modulation_shift: Option<Vec<f32>>,
pub output_buffer: Option<Array2<f32>>,
```

Add getter/setter methods:
```rust
pub fn take_time_embed(&mut self) -> Array1<f32> { ... }
pub fn take_film_scale(&mut self) -> Vec<f32> { ... }
pub fn take_film_shift(&mut self) -> Vec<f32> { ... }
```

#### Step 2: Convert DiffusionBlock forward to use in-place operations

**File**: `src/domain/layers/diffusion/block.rs`

**Key refactors**:
1. Take workspace buffers at start
2. Use pre-allocated buffers for:
   - Time embeddings
   - FiLM parameters
   - Intermediate computation buffers
3. Chain `forward_into()` calls for:
   - Temporal mixing
   - Feedforward with FiLM
4. Return buffers at end

---

## 3. Testing Strategy

### Unit Tests (new in `tests/block_integration_tests.rs`)

```rust
#[test]
fn test_transformer_block_forward_into_zero_allocation() {
    // Verify workspace buffer reuse
    // Check memory usage before/after
    // Validate numerical equivalence with original forward
}

#[test]
fn test_diffusion_block_forward_into_preserves_semantics() {
    // Verify FiLM modulation is applied correctly in-place
    // Check output equivalence
}
```

### Regression Tests

Run full test suite:
```bash
cargo test --lib 2>&1 | tail -20
```

**Expected**: All 504 tests pass, no new warnings

### Benchmark Validation

Create micro-benchmarks:
```bash
cargo bench --bench [new_bench_name]
```

Measure:
- Time per forward pass
- Memory allocations per step
- GC pressure reduction

---

## 4. Implementation Checklist

### Phase 5.1c: TransformerBlock

- [ ] Add workspace buffer take/return methods to UnifiedLayerWorkspace
- [ ] Add RichardsNorm::forward_into() method
- [ ] Refactor TransformerBlock::forward() to use in-place ops
- [ ] Update cached intermediates handling
- [ ] Test equivalence with original forward
- [ ] Validate no regressions (504 tests pass)
- [ ] Benchmark memory/speed improvements

### Phase 5.1c: DiffusionBlock (if time permits)

- [ ] Extend UnifiedLayerWorkspace diffusion buffer access
- [ ] Refactor DiffusionBlock::forward_with_timestep() to use in-place
- [ ] Apply FiLM modulation in-place
- [ ] Test diffusion generation quality
- [ ] Benchmark improvements

### Phase 5.1d: Comprehensive Validation

- [ ] Profile full training loop
- [ ] Measure cumulative memory reduction (target: 20-30 KB/step per layer)
- [ ] Validate loss curves unchanged
- [ ] Check attention patterns preserved
- [ ] Generate benchmark report

---

## 5. Expected Performance Impact

### Memory Savings (Per Layer)
| Component | Current | With forward_into | Savings |
|-----------|---------|-------------------|---------|
| Norm outputs | 3 allocations | 0 (workspace) | 3×(seq×embed) |
| Temporal output | 1 allocation | 0 (workspace) | seq×embed |
| Residual buffers | 2 allocations | 0 (workspace) | 2×(seq×embed) |
| FFN intermediate | 1 allocation | 0 (workspace) | seq×2048+ |
| **Per-layer total** | **~7 allocations** | **~1 context** | **~12 KB** |

### Scaling (12-layer model)
- **Single forward**: 12 KB × 12 layers = 144 KB
- **Training batch (8)**: 1.15 MB per step
- **100 steps**: 115 MB

### Speed Impact
- Reduced memory fragmentation
- Better L1/L2 cache locality
- Less GC pressure
- **Estimated**: 5-8% per-layer improvement

---

## 6. Code Pattern Reference

### General forward_into pattern

```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
    // Ensure output has correct shape
    if output.dim() != (input.nrows(), desired_output_cols) {
        *output = Array2::zeros((input.nrows(), desired_output_cols));
    }
    
    // Compute in-place (either directly or via intermediate)
    let result = self.forward(input);
    *output = result;  // Move into pre-allocated
}
```

### Workspace management pattern

```rust
// Take at start of batch operation
let (mut buf1, mut buf2) = workspace.take_buffers();

// Use throughout
buf1.assign(&data);
buf2.fill(0.0);

// Return at end
workspace.return_buffers(buf1, buf2);
```

### Cached intermediates pattern

```rust
// Keep original clones for backward pass (necessary for backprop)
let cached = CachedIntermediates {
    norm1_out: Arc::new(norm1_out.clone()),  // Clone for caching
    // ...
};

// But reuse workspace for forward computation
```

---

## 7. Next Steps

1. **Immediate** (1 hour):
   - Add workspace buffer take/return methods
   - Add RichardsNorm::forward_into()
   
2. **Short-term** (1 hour):
   - Refactor TransformerBlock::forward()
   - Test and validate
   
3. **Follow-up** (0.5 hour):
   - DiffusionBlock optimization
   - Final benchmarking

4. **Phase 5.2** (Next session):
   - Global buffer pooling
   - Selective gradient computation
   - Mixed precision optimization
