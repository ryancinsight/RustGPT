# Phase 5.2b: Quick Integration Guide for DiffusionBlock

## TL;DR - One Page Reference

### Current State
- ✅ `UnifiedLayerWorkspace` extended with 5 Diffusion fields
- ✅ 487/487 tests passing
- ⏳ Ready for `DiffusionBlock` refactoring

### What Needs to Change in DiffusionBlock

**Remove:**
```rust
pub cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
```

**Add:**
```rust
pub cached_gamma_beta: Option<Array1<f32>>,  // From time_conditioner
pub cached_h_vec: Option<Array1<f32>>,        // From time_conditioner
```

### Buffer Mapping

| Current (RwLock) | New (UnifiedLayerWorkspace) | Field Path |
|---|---|---|
| `input_original` | `unified_workspace.input_buffer()` | Direct mapping |
| `input_used` | `unified_workspace.input_buffer()` | Reuse same buffer |
| `time_embed` | `unified_workspace.time_embed()` | Direct mapping |
| `norm1_out` | `unified_workspace.norm1_out()` | Already mapped |
| `attn_out` | `unified_workspace.temporal_out()` | Already mapped |
| `residual1` | `unified_workspace.residual1()` | Already mapped |
| `norm2_out` | `unified_workspace.norm2_out()` | Already mapped |
| `ffn_out` | `unified_workspace.ffn_out()` | Already mapped |
| `output` | `unified_workspace.output_buffer()` | NEW buffer |
| `gamma_attn` | `unified_workspace.film_modulation_scale().slice(s![.., 0..embed])` | Slice access |
| `beta_attn` | `unified_workspace.film_modulation_scale().slice(s![.., embed..2*embed])` | Slice access |
| `gamma_ffn` | `unified_workspace.film_modulation_scale().slice(s![.., 2*embed..3*embed])` | Slice access |
| `beta_ffn` | `unified_workspace.film_modulation_scale().slice(s![.., 3*embed..])` | Slice access |

### Integration Steps (Order Matters)

#### Step 1: Initialize Workspace in DiffusionBlock::new()
```rust
// In DiffusionBlock::new()
let mut unified_workspace = UnifiedLayerWorkspace::new();
// Initialize Diffusion-specific buffers
unified_workspace.input_buffer = Some(Array2::zeros((1, 1)));
unified_workspace.time_embed = Some(Array1::zeros(embed_dim));
unified_workspace.film_modulation_scale = Some(Array2::zeros((1, 4 * embed_dim)));
unified_workspace.film_modulation_shift = Some(Array2::zeros((1, 4 * embed_dim)));
unified_workspace.output_buffer = Some(Array2::zeros((1, 1)));

Self {
    // ...
    unified_workspace,
    // Remove: cached_intermediates: RwLock::new(None),
    // Add: cached_gamma_beta: None,
    // Add: cached_h_vec: None,
}
```

#### Step 2: First Call in forward_with_timestep()
```rust
pub fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) -> Array2<f32> {
    // Ensure capacity for Diffusion buffers
    self.unified_workspace.ensure_capacity(x_t.nrows(), x_t.ncols(), self.config.embed_dim);
    
    // ... rest of forward pass ...
}
```

#### Step 3: Store Inputs
```rust
// Instead of Arc::new(x_model_in)
*self.unified_workspace.input_buffer_mut().unwrap() = x_model_in;

// Track both input_original and input_used in same buffer
// (Use flags or compute on-demand if distinction needed)
```

#### Step 4: Store Intermediate Results
```rust
// Instead of: norm1_out: Arc::new(norm1_out),
*self.unified_workspace.norm1_out_mut().unwrap() = norm1_out;

// Repeat for all other intermediates:
// - temporal_out ← attn_out
// - residual1 ← residual1
// - norm2_out ← norm2_out
// - ffn_out ← ffn_out
// - output_buffer ← output (before EDM scaling)
```

#### Step 5: Store Conditioning State
```rust
self.cached_gamma_beta = Some(gamma_beta.clone());
self.cached_h_vec = Some(h.row(0).to_owned());
```

#### Step 6: Update Backward Pass
```rust
pub fn backward(&mut self, d_output: &Array2<f32>, learning_rate: f32) -> Array2<f32> {
    // Access buffers from unified_workspace instead of RwLock
    let residual1 = self.unified_workspace.residual1().unwrap();
    let ffn_out = self.unified_workspace.ffn_out().unwrap();
    let norm2_out = self.unified_workspace.norm2_out().unwrap();
    
    // ... rest of backward pass (unchanged) ...
}
```

### Testing Strategy

```bash
# 1. Ensure all 487 tests still pass
cargo test --lib

# 2. Run DiffusionBlock-specific tests
cargo test --lib diffusion_block

# 3. Benchmark allocation count (should be -40%)
cargo bench --bench diffusion_memory

# 4. Verify gradients match old implementation
# (Run numerical gradient test)
```

### Common Pitfalls to Avoid

❌ **Don't** forget to call `ensure_capacity()` first  
→ ✅ Call at start of `forward_with_timestep()`

❌ **Don't** mix Arc access patterns  
→ ✅ Use `unwrap()` consistently; buffers guaranteed to exist after ensure_capacity()

❌ **Don't** assume buffer reuse across calls  
→ ✅ Each forward pass may reallocate (power-of-2 sizing); old pointers may invalidate

❌ **Don't** store direct references to workspace buffers  
→ ✅ Store clones if needed; workspace ownership stays in DiffusionBlock

❌ **Don't** forget FiLM parameter slicing  
→ ✅ Use ndarray slicing: `slice![.., start..end]` to extract gamma/beta from consolidated buffer

### Key Accessors Needed

```rust
// Read accessors
self.unified_workspace.input_buffer().unwrap()
self.unified_workspace.time_embed().unwrap()
self.unified_workspace.norm1_out().unwrap()
self.unified_workspace.temporal_out().unwrap()
self.unified_workspace.residual1().unwrap()
self.unified_workspace.norm2_out().unwrap()
self.unified_workspace.ffn_out().unwrap()
self.unified_workspace.output_buffer().unwrap()
self.unified_workspace.film_modulation_scale().unwrap()

// Write accessors (mutable)
self.unified_workspace.input_buffer_mut().unwrap()
self.unified_workspace.time_embed_mut().unwrap()
// ... etc ...
```

### Performance Expectations

**After Integration Complete:**
| Metric | Target | How to Validate |
|---|---|---|
| Allocations/forward | -40% | Count Arc allocations before/after |
| Peak memory | -20% | Measure heap usage with profiler |
| Latency | <-10% | `cargo bench --bench diffusion_forward_pass` |
| Test pass rate | 487+/487+ | `cargo test --lib` |

### Rollback Strategy

If something breaks:
```bash
# Quick revert
git diff HEAD  # See changes
git checkout -- src/domain/layers/diffusion/block.rs  # Revert single file

# Or revert entire commit
git revert <commit-hash>
```

### Files to Modify

**Only modify:**
1. `src/domain/layers/diffusion/block.rs` - DiffusionBlock struct + methods
2. Possibly `src/domain/layers/diffusion/mod.rs` - If imports change

**Do NOT modify:**
- ✅ `src/domain/layers/components/unified_layer_workspace.rs` (already done)
- ✅ Any block type except DiffusionBlock
- ✅ Test infrastructure

### Validation Checklist

- [ ] Code compiles without errors
- [ ] All 487+ tests pass
- [ ] `cached_intermediates` field removed
- [ ] `cached_gamma_beta` and `cached_h_vec` added
- [ ] `ensure_capacity()` called before buffer access
- [ ] FiLM parameters properly sliced from consolidated buffer
- [ ] Backward pass uses unified_workspace pointers
- [ ] No RwLock usage in DiffusionBlock (except if explicitly needed)
- [ ] Allocation count reduced by ~40% (verified via benchmark)
- [ ] Memory usage reduced by ~20% (verified via profiler)

### Time Estimate
- **Code changes**: 2-3 hours
- **Testing & validation**: 1-2 hours
- **Benchmarking**: 30 min - 1 hour
- **Total**: 3.5 - 6 hours

### Questions?
Refer to:
- `PHASE5_2b_DIFFUSION_INTEGRATION_PLAN.md` - Detailed integration plan
- `SESSION_SUMMARY_PHASE5_WORKSPACE_EXTENSION.md` - Extension details
- `src/domain/layers/components/unified_layer_workspace.rs` - Implementation reference

---

**Status**: Ready to implement  
**Confidence**: High (all prerequisites complete, architecture validated)  
**Next Step**: Execute DiffusionBlock refactoring
