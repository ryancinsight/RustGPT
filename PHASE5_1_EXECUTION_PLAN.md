# Phase 5.1 Execution Plan: In-Place Operations Consolidation

**Date**: February 13, 2026  
**Status**: Ready for Implementation  
**Target**: 10-15% per-layer speedup, 40 KB/step memory reduction

---

## Current State Analysis

### What Already Exists ✅
1. **PolyAttention**: Has `forward_into()` and `forward_into_with_causal()` implementations
   - Location: `src/domain/attention/poly_attention.rs#L1548-L1641`
   - Pattern: Writes output to pre-allocated buffer
   - Features: Supports Titan memory fusion, metrics caching
   
2. **SharedTemporalProcessing**: Has forward delegation infrastructure
   - Delegates to TemporalMixingLayer trait
   - Supports `forward_with_causal()` and `forward_with_film()`
   
3. **SharedAttentionContext**: In-place operations partially implemented
   - `apply_context_into()`: In-place context application
   - `apply_step_into()`: Step-mode in-place application
   
4. **UnifiedLayerWorkspace**: Pre-allocated buffer pool ready
   - Lazy allocation with power-of-2 sizing
   - Memory tracking and stats

### What Needs Implementation 🚧
1. **SharedTemporalProcessing**:
   - Add `forward_into()` wrapper method
   - Add `forward_with_causal_into()` method
   
2. **Temporal Mixing Variants**:
   - RgLru: Add `forward_into()` method
   - Mamba: Add `forward_into()` method
   - Mamba2: Add `forward_into()` method (if exists)
   
3. **SharedFeedforward**:
   - Add `forward_into()` wrapper method
   
4. **FeedForwardVariant Implementations**:
   - RichardsGlu: Add `forward_into()` method
   - MixtureOfExperts: Add `forward_into()` method
   
5. **Block Integration**:
   - TransformerBlock: Use in-place forward pass with workspace
   - DiffusionBlock: Use in-place forward pass with workspace

### Test Coverage Status
- Total tests: 476+ library tests
- New tests needed: 8-12 equivalence tests for forward_into variants
- Regression tests: Ensure all existing tests still pass

---

## Detailed Implementation Tasks

### Task 1: SharedTemporalProcessing Extensions
**File**: `src/domain/layers/components/temporal_processing.rs`

**Methods to Add**:
```rust
/// Forward pass with in-place output buffer
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    self.prepare_forward();
    self.temporal_mixing.forward_into(input, output)
}

/// Forward pass with causal control and in-place output
pub fn forward_with_causal_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
    causal: bool,
) -> Result<()> {
    self.prepare_forward();
    self.temporal_mixing.forward_with_causal_into(input, output, causal)
}
```

**Tests to Add**:
- `test_temporal_forward_into_basic()`: Verify basic functionality
- `test_temporal_forward_into_equivalence()`: Compare with `forward()`
- `test_temporal_forward_causal_into_equivalence()`: Compare causal variant

**Estimated Impact**: Reduces 1 allocation per forward pass

---

### Task 2: TemporalMixingLayer Trait Extension
**File**: `src/domain/layers/components/common.rs`

**Pattern**:
```rust
impl TemporalMixingLayer {
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        match self {
            Self::Attention(attn) => attn.forward_into(input, output),
            Self::RgLru(rg_lru) => rg_lru.forward_into(input, output),
            Self::Mamba(mamba) => mamba.forward_into(input, output),
            // ... other variants
        }
    }
}
```

**Tests to Add**:
- `test_temporal_mixing_layer_dispatch()`: Verify dispatch mechanism

---

### Task 3: RgLru In-Place Forward
**File**: `src/domain/layers/ssm/rg_lru.rs`

**Implementation Pattern**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    let (seq_len, embed_dim) = input.dim();
    
    if output.dim() != (seq_len, embed_dim) {
        return Err(Box::new(DimensionError::new("Output buffer dimension mismatch")));
    }
    
    // 1. Compute gate without allocating
    // 2. Compute value path directly into output
    // 3. Apply residuals in-place
    
    Ok(())
}
```

**Tests to Add**:
- `test_rg_lru_forward_into_basic()`: Basic functionality
- `test_rg_lru_forward_into_matches_forward()`: Numerical equivalence
- `test_rg_lru_forward_into_state_consistency()`: Verify state updates

**Estimated Impact**: ~5-8% latency reduction

---

### Task 4: Mamba In-Place Forward
**File**: `src/domain/layers/ssm/mamba.rs`

**Implementation Pattern**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    let (seq_len, _) = input.dim();
    
    // 1. Compute projections without allocating intermediates
    // 2. SSM computations written directly to output buffer
    // 3. Projection to output dimension
    
    Ok(())
}
```

**Tests to Add**:
- `test_mamba_forward_into_basic()`: Basic functionality
- `test_mamba_forward_into_matches_forward()`: Numerical equivalence

**Estimated Impact**: ~5-8% latency reduction

---

### Task 5: SharedFeedforward Extension
**File**: `src/domain/layers/components/feedforward.rs`

**Method to Add**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    self.feedforward.forward_into(input, output)
}
```

**Tests to Add**:
- `test_feedforward_forward_into_basic()`: Basic functionality
- `test_feedforward_forward_into_equivalence()`: Compare with `forward()`

**Estimated Impact**: Reduces 1 allocation per forward pass

---

### Task 6: RichardsGlu In-Place Forward
**File**: `src/domain/richards.rs` (or `src/domain/layers/richards_glu.rs`)

**Implementation Pattern**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    let (seq_len, embed_dim) = input.dim();
    
    if output.dim() != (seq_len, embed_dim) {
        return Err(Box::new(DimensionError::new("Output dimension mismatch")));
    }
    
    // 1. Compute projection to hidden dim without allocating
    // 2. Apply Richards activation in-place
    // 3. Gate computation and application
    // 4. Output projection directly to output buffer
    
    Ok(())
}
```

**Tests to Add**:
- `test_richards_glu_forward_into_basic()`: Basic functionality
- `test_richards_glu_forward_into_matches_forward()`: Numerical equivalence
- `test_richards_glu_forward_into_gradient_consistency()`: Verify backward pass

**Estimated Impact**: ~4-6% latency reduction

---

### Task 7: MixtureOfExperts In-Place Forward
**File**: `src/domain/mixtures/moe.rs`

**Implementation Pattern**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    let (seq_len, embed_dim) = input.dim();
    
    if output.dim() != (seq_len, embed_dim) {
        return Err(Box::new(DimensionError::new("Output dimension mismatch")));
    }
    
    // 1. Router computation without allocating
    // 2. Expert selection without allocating
    // 3. Expert outputs written directly to output buffer
    // 4. Weighted combination in-place
    
    Ok(())
}
```

**Tests to Add**:
- `test_moe_forward_into_basic()`: Basic functionality
- `test_moe_forward_into_matches_forward()`: Numerical equivalence

**Estimated Impact**: ~4-6% latency reduction

---

### Task 8: TransformerBlock Integration
**File**: `src/domain/layers/transformer/block.rs`

**Pattern**:
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    let seq_len = input.nrows();
    let batch_size = 1; // or from input context
    let embed_dim = input.ncols();
    
    // Ensure workspace capacity
    self.unified_workspace.ensure_capacity(seq_len, batch_size, embed_dim)?;
    
    // 1. First norm
    let norm1_out = self.unified_workspace.get_norm1_out_mut();
    self.pre_attention_norm.forward_into(input, norm1_out)?;
    
    // 2. Temporal mixing in-place
    let temporal_out = self.unified_workspace.get_temporal_out_mut();
    self.temporal_mixing.forward_into(norm1_out, temporal_out)?;
    
    // 3. Residual
    let residual1 = self.unified_workspace.get_residual1_mut();
    add_residual_into(temporal_out, residual1, input)?;
    
    // 4. Second norm
    let norm2_out = self.unified_workspace.get_norm2_out_mut();
    self.pre_ffn_norm.forward_into(residual1, norm2_out)?;
    
    // 5. FFN in-place
    let ffn_out = self.unified_workspace.get_ffn_out_mut();
    self.feedforward.forward_into(norm2_out, ffn_out)?;
    
    // 6. Final residual
    add_residual_into(ffn_out, output, residual1)?;
    
    Ok(())
}
```

**Tests to Add**:
- `test_transformer_block_forward_into_basic()`: Basic functionality
- `test_transformer_block_forward_into_matches_forward()`: Numerical equivalence
- `test_transformer_block_forward_into_memory_efficient()`: Verify allocation count

**Estimated Impact**: ~10-15% latency reduction per block

---

### Task 9: DiffusionBlock Integration
**File**: `src/domain/layers/diffusion/block.rs`

**Pattern**: Similar to TransformerBlock but includes:
- Time embedding computation
- FiLM modulation application
- Extra diffusion-specific buffers in workspace

**Tests to Add**:
- `test_diffusion_block_forward_into_basic()`: Basic functionality
- `test_diffusion_block_forward_into_with_conditioning()`: With time conditioning

**Estimated Impact**: ~10-15% latency reduction per block

---

### Task 10: Comprehensive Validation
**Files**: `tests/` directory

**Validation Steps**:
1. Run all 476+ existing library tests
   ```
   cargo test --lib
   ```

2. Run new in-place equivalence tests
   ```
   cargo test --lib forward_into
   ```

3. Profile latency improvements
   - Create benchmark comparing `forward()` vs `forward_into()`
   - Measure memory usage with peak allocation tracking
   - Run 100-step inference to confirm improvements

4. Stress test
   - 1000-step training run without memory leaks
   - Verify gradient computation consistency
   - Check numerical stability over long sequences

**Success Criteria**:
- ✅ All 476+ tests pass
- ✅ All new forward_into tests pass with < 1e-5 error
- ✅ Zero clippy warnings
- ✅ Benchmark shows ≥ 10% speedup per layer
- ✅ Memory reduced from 129 KB to ≤ 89 KB/step
- ✅ No allocations in hot path (verified with profiler)

---

## Implementation Sequence & Checkpoints

### Week of Feb 14-15 (Foundation)
1. Implement SharedTemporalProcessing::forward_into
2. Implement TemporalMixingLayer dispatch
3. Create test suite foundation
4. Checkpoint: 476+ tests still pass

### Week of Feb 16-17 (SSM Variants)
1. Implement RgLru::forward_into
2. Implement Mamba::forward_into
3. Add equivalence tests
4. Checkpoint: 476+ tests + SSM tests pass

### Week of Feb 18 (Feedforward)
1. Implement SharedFeedforward::forward_into
2. Implement RichardsGlu::forward_into
3. Implement MixtureOfExperts::forward_into
4. Add equivalence tests
5. Checkpoint: 476+ tests + FFN tests pass

### Week of Feb 19-20 (Integration & Validation)
1. Integrate into TransformerBlock
2. Integrate into DiffusionBlock
3. Run comprehensive benchmarks
4. Final validation and stress tests
5. Checkpoint: Performance targets met

---

## Memory Optimization Checklist

- [ ] No intermediate allocations in forward_into path
- [ ] Workspace buffers reused when possible
- [ ] Power-of-2 sizing applied to scratch buffers
- [ ] Lazy allocation for optional buffers
- [ ] Allocation tracking in workspace stats
- [ ] Zero-copy operations via general_mat_mul
- [ ] Peak memory tracked per step

---

## Build & Test Commands

```bash
# Build optimized
cargo build --release

# Run all tests
cargo test --lib

# Run in-place tests only
cargo test --lib forward_into

# Run specific component test
cargo test --lib test_transformer_block_forward_into

# Benchmark
cargo bench --bench transformer_block

# Check for regressions
cargo clippy --all-targets
cargo fmt -- --check
```

---

## Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Regression in output quality | Strict numerical equivalence tests (< 1e-5 error) |
| Memory leaks in streaming | Long-running stress tests (1000 steps) |
| Cache locality issues | Profile memory access patterns |
| Backward compatibility | Keep `forward()` methods unchanged |
| Complexity explosion | Clear pattern replication, comprehensive tests |

---

## Next Phase (5.2)

After successful completion of Phase 5.1:
- **Global Buffer Pooling**: Consolidate all workspace pools
- **Batch Norm Fusion**: Combine norm, mixing, residual into single kernel
- **Mixed Precision**: f32 activations, f16 context matrices
- **Target Impact**: Additional 15-25% speedup

