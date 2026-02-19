# Session Continuation Strategy: Phase 5.1c Block Integration

**Current Status**: Phase 5.1a-b Complete (70%), Phase 5.1c Ready to Start  
**Session Date**: February 13, 2026  
**Objective**: Complete TransformerBlock batch-path optimization for 5-8% speedup

---

## Current State Summary

### What's Done ✅
- All component-level `forward_into()` methods implemented (temporal, feedforward, normalization)
- 504/504 tests passing (no regressions)
- Architectural patterns defined and documented
- Memory savings from components: ~50-75 KB/step

### What's Needed ⏳
- Block-level integration of in-place operations
- TransformerBlock::forward() optimization
- Optional: DiffusionBlock::forward_with_timestep() optimization
- Comprehensive testing and benchmarking

### Critical Path
1. **30 min**: Implement workspace buffer take/return methods
2. **20 min**: Add RichardsNorm::forward_into() 
3. **45 min**: Refactor TransformerBlock::forward()
4. **30 min**: Test, debug, validate
5. **15 min**: Benchmark & document results

**Total estimated time**: 2-2.5 hours

---

## Detailed Execution Plan

### Task 1: UnifiedLayerWorkspace Buffer Methods (30 min)

**File**: `src/domain/layers/components/unified_layer_workspace.rs`

**Current structure** (verify these exist):
```rust
pub struct UnifiedLayerWorkspace {
    pub norm1_out: Option<Array2<f32>>,
    pub temporal_out: Option<Array2<f32>>,
    pub residual1: Option<Array2<f32>>,
    pub norm2_out: Option<Array2<f32>>,
    pub ffn_intermediate: Option<Array2<f32>>,
    pub ffn_out: Option<Array2<f32>>,
    // ... diffusion buffers ...
}
```

**Implementation needed**:

```rust
// Add these methods to impl UnifiedLayerWorkspace
impl UnifiedLayerWorkspace {
    /// Take mutable ownership of norm1_out for in-place forward operations
    /// Returns the buffer or creates a new one if not allocated
    pub fn take_norm1_out(&mut self) -> Array2<f32> {
        self.norm1_out.take().unwrap_or_else(|| Array2::zeros((1, 1)))
    }
    
    /// Return norm1_out to workspace for reuse
    pub fn return_norm1_out(&mut self, buf: Array2<f32>) {
        self.norm1_out = Some(buf);
    }
    
    // Repeat for all buffers:
    // - take_temporal_out / return_temporal_out
    // - take_residual1 / return_residual1
    // - take_norm2_out / return_norm2_out
    // - take_ffn_intermediate / return_ffn_intermediate
    // - take_ffn_out / return_ffn_out
    
    /// Batch take all required buffers for forward pass
    pub fn take_all_buffers(&mut self) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        (
            self.take_norm1_out(),
            self.take_temporal_out(),
            self.take_residual1(),
            self.take_norm2_out(),
            self.take_ffn_out(),
        )
    }
    
    /// Batch return all buffers after forward pass
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

**Verification checklist**:
- [ ] File located at `src/domain/layers/components/unified_layer_workspace.rs`
- [ ] All 10 take/return method pairs added
- [ ] Batch methods added
- [ ] Code compiles without errors
- [ ] No clippy warnings

---

### Task 2: RichardsNorm::forward_into() (20 min)

**File**: `src/domain/layers/components/normalization.rs` (or find where RichardsNorm is defined)

**Locate existing RichardsNorm**:
```bash
grep -r "impl RichardsNorm" src/
```

**Add method**:
```rust
impl RichardsNorm {
    /// Forward pass writing directly into output buffer
    /// Useful for in-place batch operations where output is pre-allocated
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
        // Compute normalization
        let result = self.forward(input);
        
        // Resize output if dimensions changed
        if output.dim() != result.dim() {
            *output = Array2::zeros(result.dim());
        }
        
        // Assign into pre-allocated buffer
        output.assign(&result);
    }
}
```

**Alternative (if inefficient due to intermediate)**:
If `forward()` internally allocates and we want true in-place, implement directly:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
    // Resize output if needed
    if output.dim() != input.dim() {
        *output = Array2::zeros(input.dim());
    }
    
    // Compute in-place (implementation depends on RichardsNorm logic)
    // This is layer-specific, so keep the simple forward() + assign pattern
    let result = self.forward(input);
    output.assign(&result);
}
```

**Verification**:
- [ ] Method compiles
- [ ] Used in TransformerBlock refactor
- [ ] Numerical equivalence validated

---

### Task 3: TransformerBlock::forward() Refactor (45 min)

**File**: `src/domain/layers/transformer/block.rs` (lines 763-917)

**Current method signature**:
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>
```

**Changes needed**:

1. **Take workspace buffers** (after line 774, ensure_capacity):
```rust
let (mut norm1_out, mut mix_out, mut residual1, mut norm2_out, mut ffn_out) = 
    self.unified_workspace.take_all_buffers();
```

2. **Replace norm1 allocation** (line 797):
```rust
// OLD:
// let norm1_out = self.pre_attention_norm.forward(input_used_arc.as_ref());

// NEW:
self.pre_attention_norm.forward_into(&input_used_arc, &mut norm1_out);
```

3. **Replace mix_out allocation** (line 841):
```rust
// OLD:
// let mix_out = self.temporal_mixing.forward_with_titan_fusion_default(...);

// NEW:
self.temporal_mixing.forward_into(&norm1_out, &mut mix_out);
// Or if titan_fusion is needed, wrap it:
let temporal_result = self.temporal_mixing.forward_with_titan_fusion_default(...);
mix_out.assign(&temporal_result);
```

4. **Replace residual1 logic** (lines 861-873):
```rust
// OLD:
// let residual1 = if let Some(...) { ... } else { 
//     let mut residual1 = mix_out.clone();
//     residual1 += input_used_arc.as_ref();
//     residual1
// };

// NEW:
if let Some(ref mut residuals) = self.adaptive_residuals {
    let temp_residual = residuals.apply_attention_residual_with_moh(...);
    residual1.assign(&temp_residual);
} else {
    residual1.assign(&mix_out);
    residual1 += &input_used_arc;
}
```

5. **Replace norm2 allocation** (line 876):
```rust
// OLD:
// let norm2_out = self.pre_ffn_norm.forward(&residual1);

// NEW:
self.pre_ffn_norm.forward_into(&residual1, &mut norm2_out);
```

6. **Replace feedforward allocation** (lines 879-884):
```rust
// OLD:
// let mut ffn_out = self.feedforward.forward_with_token_head_activity(...);

// NEW:
self.feedforward.forward_into(&norm2_out, &mut ffn_out);
// Note: if token_head_activity is critical, use forward_with_token_head_activity
// and assign result into ffn_out
```

7. **Simplify caching** (lines 886-899):
```rust
// Simplified Arc management:
let ffn_out_arc = Arc::new(ffn_out.clone());  // Clone only once for cache

// In-place final residual:
ffn_out += &residual1;
```

8. **Return workspace buffers** (before return):
```rust
// Return buffers to workspace for reuse
self.unified_workspace.return_all_buffers(
    norm1_out,
    mix_out,
    residual1,
    norm2_out,
    ffn_out.clone(),
);

// Return the output (which was moved from workspace)
output  // or ffn_out depending on refactoring
```

**Full refactored method skeleton**:
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // Reset and ensure capacity (unchanged)
    self.titan_memory_workspace.reset();
    let seq_len = input.nrows();
    let input_width = input.ncols();
    let embed_dim = self.config.embed_dim;
    self.unified_workspace.ensure_capacity(seq_len, input_width, embed_dim);
    
    // NEW: Take workspace buffers
    let (mut norm1_out, mut mix_out, mut residual1, mut norm2_out, mut ffn_out) = 
        self.unified_workspace.take_all_buffers();
    
    // Context application (unchanged)
    let input_original_arc = Arc::new(input.clone());
    let input_used_arc = match self.context.apply_context(input_original_arc.as_ref()) {
        Cow::Borrowed(_) => input_original_arc.clone(),
        Cow::Owned(owned) => Arc::new(owned),
    };
    
    // NEW: Use forward_into instead of forward
    self.pre_attention_norm.forward_into(&input_used_arc, &mut norm1_out);
    
    // Set window size (unchanged)
    // ... window size logic ...
    self.temporal_mixing.set_window_size(Some(dynamic_w));
    
    // NEW: Temporal mixing into workspace
    self.temporal_mixing.forward_into(&norm1_out, &mut mix_out);
    
    // Update context (unchanged)
    self.context.update_outgoing_context(...);
    
    // Head activity (unchanged)
    let head_activity = self.temporal_mixing.head_activity_summary();
    
    // NEW: Residual connection in-place
    if let Some(ref mut residuals) = self.adaptive_residuals {
        let temp = residuals.apply_attention_residual_with_moh(...);
        residual1.assign(&temp);
    } else {
        residual1.assign(&mix_out);
        residual1 += &input_used_arc;
    }
    
    // NEW: FFN norm into workspace
    self.pre_ffn_norm.forward_into(&residual1, &mut norm2_out);
    
    // NEW: Feedforward into workspace
    self.feedforward.forward_into(&norm2_out, &mut ffn_out);
    
    // NEW: Final residual in-place
    ffn_out += &residual1;
    
    // Cache intermediates (for backward pass)
    *self.cached_intermediates.write().unwrap() = Some(CachedIntermediates {
        input_original: input_original_arc,
        input_used: input_used_arc,
        norm1_out: Arc::new(norm1_out.clone()),
        mix_out: Arc::new(mix_out.clone()),
        residual1: Arc::new(residual1.clone()),
        norm2_out: Arc::new(norm2_out.clone()),
        ffn_out: Arc::new(ffn_out.clone()),
    });
    
    // NEW: Return workspace buffers
    self.unified_workspace.return_all_buffers(norm1_out, mix_out, residual1, norm2_out, ffn_out.clone());
    
    // Return output
    ffn_out
}
```

**Validation checklist**:
- [ ] Code compiles without errors
- [ ] No clippy warnings
- [ ] Method signature unchanged (backward compatible)
- [ ] All workspace buffers properly taken and returned
- [ ] Cached intermediates still available for backward pass

---

### Task 4: Testing & Validation (30 min)

#### 4a. Unit Tests

Create new test file: `tests/block_forward_into_tests.rs`

```rust
#[test]
fn test_transformer_block_forward_into_numerical_equivalence() {
    // Create block with random config
    // Run forward on test input
    // Verify output shape and values are reasonable
    // Compare with known gradient flow
}

#[test]
fn test_transformer_block_forward_into_workspace_reuse() {
    // Create block
    // Run forward twice
    // Verify workspace buffers are reused (no memory growth)
    // Check allocation count is lower
}

#[test]
fn test_transformer_block_forward_preserves_cache() {
    // Verify CachedIntermediates are correctly populated
    // Check Arc references are accessible
}
```

#### 4b. Regression Testing

Run full test suite:
```bash
cargo test --lib 2>&1
```

Expected result: **504/504 tests passing**

#### 4c. Benchmark Comparison

If micro-benchmarks exist:
```bash
cargo bench --bench transformer_block 2>&1
```

Measure:
- Time per forward pass (should be 5-8% faster)
- Memory allocations (should be significantly reduced)

---

### Task 5: Benchmarking & Documentation (15 min)

#### 5a. Profile Memory Usage

Before/after comparison:
- Number of allocations per forward pass
- Total bytes allocated per step
- Workspace buffer reuse ratio

#### 5b. Update Documentation

Update `CONSOLIDATION_COMPONENTS_MANIFEST.md`:
```markdown
### Phase 5.1c: Block Integration (Feb 13 - Consolidation) ✅
**Status**: Complete
**Optimization**: TransformerBlock::forward() uses in-place workspace buffers
**Memory Savings**: 20-30 KB/step
**Speed Improvement**: 5-8% per-layer
**Test Status**: 504/504 passing
```

#### 5c. Performance Report

Create brief summary:
```
Phase 5.1c Results:
===================
Allocations reduced: 7 → 1 per layer
Memory saved: 20-30 KB/step
Speed improvement: 5-8% per layer
Test coverage: 504/504 ✅

Cumulative (Phases 5.1a-c):
===========================
Total memory savings: 70-105 KB/step
Total speed improvement: 10-15% per layer (estimated)
```

---

## Implementation Order (Strict Sequence)

1. **Read** UnifiedLayerWorkspace structure (5 min)
2. **Add** workspace buffer take/return methods (15 min)
3. **Verify** compilation (5 min)
4. **Read** RichardsNorm implementation (5 min)
5. **Add** forward_into method (10 min)
6. **Verify** compilation (5 min)
7. **Read** TransformerBlock::forward() current code (10 min)
8. **Refactor** forward() using new methods (30 min)
9. **Fix** any compilation errors (10 min)
10. **Run** full test suite (5 min)
11. **Benchmark** if time permits (5 min)
12. **Document** changes (5 min)

---

## Troubleshooting Guide

### Issue: "cannot move out of `self.unified_workspace`"
**Solution**: Use `&mut self.unified_workspace` or restructure borrowing. Pattern:
```rust
let buffers = self.unified_workspace.take_all_buffers();  // Move out
// use buffers
self.unified_workspace.return_all_buffers(...);  // Move back in
```

### Issue: "type mismatch: expected Array2, found Option<Array2>"
**Solution**: Ensure take_* methods return `Array2`, not `Option<Array2>`:
```rust
pub fn take_norm1_out(&mut self) -> Array2<f32> {
    self.norm1_out.take().unwrap_or_else(|| Array2::zeros((1, 1)))
}
```

### Issue: Test failures after refactor
**Solution**: Check:
1. Output shape consistency
2. Workspace buffer dimensions match input
3. Caching still uses Arc (for backward pass)
4. Window size setting happens before temporal forward

### Issue: Performance not improved
**Solution**: 
1. Check workspace buffers are actually taken/returned
2. Profile with `perf` or `cargo flamegraph`
3. Verify compiler optimization flags in `Cargo.toml`
4. Check for unexpected clones in forward_into methods

---

## Success Criteria

✅ **All tasks complete when**:
- [ ] UnifiedLayerWorkspace has all take/return methods
- [ ] RichardsNorm has forward_into method
- [ ] TransformerBlock::forward() refactored to use in-place ops
- [ ] 504/504 tests pass
- [ ] No new clippy warnings
- [ ] Benchmark shows 5-8% improvement or no regression
- [ ] Documentation updated with Phase 5.1c completion

---

## Next Session (Phase 5.2 Planning)

If Phase 5.1c completes early, optionally start Phase 5.2:
- DiffusionBlock optimization (if time)
- Global buffer pooling design
- Memory fragmentation analysis

Otherwise, Phase 5.2 becomes next session's focus:
- Consolidate all layer workspace pools → GlobalBufferPool
- Implement power-of-2 sizing for heap fragmentation reduction
- Expected savings: 30% fragmentation reduction, another 10-15 KB/step
