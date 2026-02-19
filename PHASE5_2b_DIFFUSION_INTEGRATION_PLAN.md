# Phase 5.2b: DiffusionBlock Integration Plan

## Objective
Refactor `DiffusionBlock` to eliminate `DiffusionCachedIntermediates` RwLock and consolidate all buffer management into `UnifiedLayerWorkspace`, reducing Arc allocation overhead by 75-80%.

## Current State Analysis

### DiffusionBlock Cache Structure
```rust
pub struct DiffusionBlock {
    // ... other fields ...
    pub cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
    pub unified_workspace: UnifiedLayerWorkspace,  // ← Already present but underutilized
}
```

### DiffusionCachedIntermediates (16 Arc fields)
```rust
pub struct DiffusionCachedIntermediates {
    pub input_original: Arc<Array2<f32>>,           // ← input_buffer
    pub input_used: Arc<Array2<f32>>,               // ← input_buffer (alias)
    pub time_embed: Arc<Array1<f32>>,               // ← time_embed
    pub gamma_beta: Arc<Array1<f32>>,               // ✓ Keep (3-param vector)
    pub norm1_out: Arc<Array2<f32>>,                // ✓ Already in unified_workspace
    pub norm1_mod: Arc<Array2<f32>>,                // ✗ Temp (modulation only)
    pub attn_out: Arc<Array2<f32>>,                 // ✓ temporal_out
    pub residual1: Arc<Array2<f32>>,                // ✓ residual1
    pub norm2_out: Arc<Array2<f32>>,                // ✓ norm2_out
    pub norm2_mod: Arc<Array2<f32>>,                // ✗ Temp (modulation only)
    pub ffn_out: Arc<Array2<f32>>,                  // ✓ ffn_out
    pub output: Arc<Array2<f32>>,                   // ✓ final output (new field needed)
    pub h_vec: Arc<Array1<f32>>,                    // ✗ Temp (conditioning state)
    pub gamma_attn: Arc<Array2<f32>>,               // ✓ film_modulation_scale[batch, :embed]
    pub beta_attn: Arc<Array2<f32>>,                // ✓ film_modulation_scale[batch, embed:2*embed]
    pub gamma_ffn: Arc<Array2<f32>>,                // ✓ film_modulation_scale[batch, 2*embed:3*embed]
    pub beta_ffn: Arc<Array2<f32>>,                 // ✓ film_modulation_scale[batch, 3*embed:]
    pub timestep: usize,                            // ✓ Keep metadata
}
```

### Consolidation Mapping

| Field | Target | Rationale | Action |
|---|---|---|---|
| `input_original` | `unified_workspace.input_buffer` | Direct mapping | Move |
| `input_used` | `unified_workspace.input_buffer` | Reuse same buffer | Move |
| `time_embed` | `unified_workspace.time_embed` | Direct mapping | Move |
| `gamma_beta` | **Keep in DiffusionBlock** | 3-param vector, used in conditioning | Keep separate |
| `norm1_out` | `unified_workspace.norm1_out` | Already mapped | Reuse |
| `norm1_mod` | **Eliminate** | Temporary modulation (computed on-the-fly) | Remove |
| `attn_out` | `unified_workspace.temporal_out` | Already mapped | Reuse |
| `residual1` | `unified_workspace.residual1` | Already mapped | Reuse |
| `norm2_out` | `unified_workspace.norm2_out` | Already mapped | Reuse |
| `norm2_mod` | **Eliminate** | Temporary modulation (computed on-the-fly) | Remove |
| `ffn_out` | `unified_workspace.ffn_out` | Already mapped | Reuse |
| `output` | **Add new field** | Final output buffer | Add to unified_workspace |
| `h_vec` | **Keep in DiffusionBlock** | Temporary state from time_conditioner | Keep separate |
| `gamma_attn` | `unified_workspace.film_modulation_scale[batch, :embed]` | FiLM scale consolidation | Use view/slice |
| `beta_attn` | `unified_workspace.film_modulation_scale[batch, embed:2*embed]` | FiLM scale consolidation | Use view/slice |
| `gamma_ffn` | `unified_workspace.film_modulation_scale[batch, 2*embed:3*embed]` | FiLM scale consolidation | Use view/slice |
| `beta_ffn` | `unified_workspace.film_modulation_scale[batch, 3*embed:]` | FiLM scale consolidation | Use view/slice |
| `timestep` | **Metadata only** | Track current timestep | Keep separate |

## Refactoring Steps

### Step 1: Add Output Buffer to UnifiedLayerWorkspace (DONE in extension)
Already added during Phase 5.2b extension:
- `film_modulation_scale: Option<Array2<f32>>` (shape: `[batch, 4*embed_dim]`)
- `film_modulation_shift: Option<Array2<f32>>` (shape: `[batch, 4*embed_dim]`)

**Note**: Review FiLM parameter layout—currently using `film_modulation_scale` and `film_modulation_shift`, but the original code uses `gamma_{attn,ffn}` and `beta_{attn,ffn}`. Need to clarify if these are aliases or separate.

### Step 2: DiffusionBlock Initialization
Replace:
```rust
pub cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
```

With:
```rust
// unified_workspace already exists; ensure Diffusion fields are initialized:
pub cached_gamma_beta: Option<Array1<f32>>,  // From time_conditioner
pub cached_h_vec: Option<Array1<f32>>,        // From time_conditioner
pub current_timestep: usize,                   // Already exists
```

### Step 3: Forward Pass Integration

**Current flow:**
```rust
fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) {
    // 1. Time embedding
    let time_embed = self.time_embedding.forward(t, ...);
    
    // 2. FiLM modulation
    let (gamma_beta, h) = self.time_conditioner.forward(&time_embed, ...);
    self.film_modulation.update(gamma_beta.as_slice().unwrap(), ...);
    
    // 3. Norm + modulation
    let norm1_out = self.pre_attention_norm.forward(&input_used);
    let norm1_mod = self.film_modulation.apply_attn_conditioning(&norm1_out);
    
    // 4. Temporal mixing
    let attn_out = self.temporal_mixing.forward_with_titan_fusion(&norm1_mod, ...);
    
    // 5. Residual
    let residual1 = adaptive_residuals.apply_attention_residual_with_moh(...);
    
    // 6. Second norm + modulation
    let norm2_out = self.pre_ffn_norm.forward(&residual1);
    let norm2_mod = self.film_modulation.apply_ffn_conditioning(&norm2_out);
    
    // 7. FFN
    let ffn_out = self.feedforward.forward_with_token_head_activity(&norm2_mod, ...);
    
    // 8. Final residual
    let output = adaptive_residuals.apply_ffn_residual(&residual1, &ffn_out);
    
    // 9. Store in RwLock
    *self.cached_intermediates.write().unwrap() = Some(DiffusionCachedIntermediates { ... });
}
```

**Refactored flow:**
```rust
fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) {
    // Ensure workspace capacity
    self.unified_workspace.ensure_capacity(x_t.nrows(), x_t.ncols(), self.config.embed_dim);
    
    // 1. Time embedding → unified_workspace.time_embed
    let time_embed = self.time_embedding.forward(t, ...);
    *self.unified_workspace.time_embed_mut().unwrap() = time_embed.clone();
    
    // 2. FiLM modulation
    let (gamma_beta, h) = self.time_conditioner.forward(&self.unified_workspace.time_embed().unwrap(), ...);
    self.cached_gamma_beta = Some(gamma_beta);
    self.cached_h_vec = Some(h);
    self.film_modulation.update(...);
    
    // 3. Input buffer
    *self.unified_workspace.input_buffer_mut().unwrap() = x_t.clone();
    
    // 4. Norm1 → unified_workspace.norm1_out
    let input_used = self.context.apply_context(&self.unified_workspace.input_buffer().unwrap());
    let norm1_out = self.pre_attention_norm.forward(&input_used);
    *self.unified_workspace.norm1_out_mut().unwrap() = norm1_out;
    
    // 5. Apply modulation in-place
    self.film_modulation.apply_attn_conditioning(
        self.unified_workspace.norm1_out_mut().unwrap()
    );
    
    // 6. Temporal mixing → unified_workspace.temporal_out
    let attn_out = self.temporal_mixing.forward_with_titan_fusion(
        self.unified_workspace.norm1_out().unwrap(),
        ...
    );
    *self.unified_workspace.temporal_out_mut().unwrap() = attn_out;
    
    // 7. Residual → unified_workspace.residual1
    let residual1 = adaptive_residuals.apply_attention_residual_with_moh(
        &self.unified_workspace.input_buffer().unwrap(),
        self.unified_workspace.temporal_out().unwrap(),
        ...
    );
    *self.unified_workspace.residual1_mut().unwrap() = residual1;
    
    // 8. Norm2 → unified_workspace.norm2_out
    let norm2_out = self.pre_ffn_norm.forward(self.unified_workspace.residual1().unwrap());
    *self.unified_workspace.norm2_out_mut().unwrap() = norm2_out;
    
    // 9. Apply FFN modulation in-place
    self.film_modulation.apply_ffn_conditioning(
        self.unified_workspace.norm2_out_mut().unwrap()
    );
    
    // 10. FFN → unified_workspace.ffn_out
    let ffn_out = self.feedforward.forward_with_token_head_activity(
        self.unified_workspace.norm2_out().unwrap(),
        ...
    );
    *self.unified_workspace.ffn_out_mut().unwrap() = ffn_out;
    
    // 11. Final output (ADD NEW FIELD: output_buffer)
    let output = adaptive_residuals.apply_ffn_residual(
        self.unified_workspace.residual1().unwrap(),
        self.unified_workspace.ffn_out().unwrap(),
    );
    // Store output for backward pass
    self.unified_workspace.output_buffer = Some(output.clone());
    
    // 12. EDM scaling
    let prediction = if edm_on {
        (x_t * c_skip) + (&output * c_out)
    } else {
        output
    };
    
    // 13. Validation
    assert!(prediction.iter().all(|v| v.is_finite()), "Non-finite values in forward pass");
    
    // 14. Store timestep metadata
    self.current_timestep = t;
    
    // 15. NO RwLock needed anymore!
    prediction
}
```

### Step 4: Backward Pass Integration

**Current backward() method reads from cached_intermediates**:
- Extract gradients from cached Arc pointers
- Compute parameter gradients
- Back-propagate through layers

**Refactored approach**:
- Use unified_workspace pointers directly
- No RwLock overhead
- Gradient computation unchanged (same underlying arrays)

```rust
pub fn backward(
    &mut self,
    d_output: &Array2<f32>,
    learning_rate: f32,
) -> Array2<f32> {
    // Access buffers directly from unified_workspace
    let d_residual1 = &self.unified_workspace.residual1().unwrap() * d_output;
    let d_ffn_out = self.feedforward.backward(&d_output, learning_rate);
    
    // ... rest of backward pass using unified_workspace pointers ...
}
```

### Step 5: Remove DiffusionCachedIntermediates

Once all usages are migrated:
```bash
git rm src/domain/layers/diffusion/cached_intermediates.rs  # if separate file
# Or delete the struct definition from block.rs
```

### Step 6: Add New UnifiedLayerWorkspace Fields

**Extension needed**:
```rust
pub struct UnifiedLayerWorkspace {
    // ... existing fields ...
    
    /// Final output buffer (new for Diffusion)
    #[serde(skip)]
    pub output_buffer: Option<Array2<f32>>,
}
```

Update:
- `ensure_capacity()` to allocate `output_buffer`
- `estimate_memory_usage()` to include `output_buffer`
- Tests for `output_buffer` allocation and clearing

## Validation Strategy

### Unit Tests
1. **TestDiffusionBlockForwardBackward**: Ensure forward/backward produces same gradients as cached version
2. **TestAllocationCount**: Verify -75% reduction in Arc allocations
3. **TestMemoryUsage**: Verify peak memory reduction
4. **TestWorkspaceReuse**: Ensure buffers are reused across forward calls

### Integration Tests
- Run existing DiffusionBlock forward/backward tests
- Verify loss convergence on toy dataset
- Compare numerical gradients with autodiff

### Performance Benchmarks
```bash
cargo bench --bench diffusion_forward_pass  # Measure latency
cargo bench --bench diffusion_memory        # Measure allocations
```

## Risk Analysis

### Risk 1: Gradient Computation Breaks
**Likelihood**: Medium  
**Mitigation**: Comprehensive gradient tests; numerical gradient validation; side-by-side comparison with old code

### Risk 2: Memory Tracking Becomes Inaccurate
**Likelihood**: Low  
**Mitigation**: Use `estimate_memory_usage()` consistently; validate against actual allocations

### Risk 3: Buffer Aliasing Issues (input_original vs input_used)
**Likelihood**: Medium  
**Mitigation**: Store separate flags or use slices; document buffer reuse patterns; test cases for both input variants

### Risk 4: Backward Pass Compatibility
**Likelihood**: Medium  
**Mitigation**: Ensure Arc pointers from unified_workspace work identically to cached Arc fields; use shared_ptr semantics

## Implementation Order

1. **First**: Add `output_buffer` to `UnifiedLayerWorkspace` (extend Phase 5.2b)
2. **Second**: Update DiffusionBlock `__new__()` to initialize workspace Diffusion fields
3. **Third**: Refactor `forward_with_timestep()` to use unified_workspace
4. **Fourth**: Refactor `backward()` to use unified_workspace pointers
5. **Fifth**: Remove `cached_intermediates` field and struct
6. **Sixth**: Benchmark allocation reduction and memory usage
7. **Seventh**: Update integration tests
8. **Eighth**: Commit with detailed message

## Estimated Effort

- **Code changes**: 3-4 hours (refactor forward/backward, update tests)
- **Testing & validation**: 2-3 hours (gradient tests, benchmarks)
- **Documentation**: 1 hour (update architecture docs)
- **Total**: 6-8 hours

## Success Criteria

✅ All 486+ tests pass  
✅ Backward compatibility maintained (same gradients)  
✅ -75% Arc allocation reduction verified  
✅ -20% peak memory reduction measured  
✅ Zero code duplication with other blocks  
✅ Performance within -10% latency (acceptable tradeoff for memory)  
✅ Clean git history

## Open Questions

1. **FiLM parameter layout**: Are `film_modulation_scale` and `film_modulation_shift` sufficient, or do we need separate tracking of gamma vs beta?
2. **Input buffer reuse**: How to handle `input_original` vs `input_used` distinction with a single buffer?
3. **gradient_gamma_beta storage**: Where to store gradients for `gamma_beta` if not in RwLock?

---

## Appendix: Buffer Lifecycle During Forward Pass

### Current (RwLock-based)
```
1. allocate DiffusionCachedIntermediates (16 Arc)
2. compute forward pass
3. store results in RwLock
4. backward pass reads from RwLock
5. drop RwLock when DiffusionBlock is dropped
```

### Refactored (Unified workspace)
```
1. ensure_capacity() allocates unified_workspace buffers once
2. forward pass writes to workspace
3. backward pass reads from same workspace
4. workspace reused across forward calls
5. drop when DiffusionBlock is dropped
```

**Net effect**: Single allocation point + reuse pattern → -40% allocations per step.

