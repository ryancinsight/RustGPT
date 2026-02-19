# Phase 5 Continuation: Consolidation & Memory Optimization

## Objective
Complete consolidation of shared components between Diffusion, SSM/RG-LRU, and Transformer blocks while achieving:
- **-40% allocation churn** (from 50-60 to 30-35 allocations per forward pass)
- **-20% peak memory** (from 2.0 GB to 1.6 GB)
- **-100% code duplication** in shared component interfaces
- **484/484 tests passing** (maintain 100% test coverage)

## Status: 82% → Target 95%+

### Current Architecture
| Component | Status | Integration |
|---|---|---|
| `UnifiedLayerWorkspace` | ✅ Deployed | All blocks use this |
| `SharedAttentionContext` | ✅ Deployed | Attention & Diffusion |
| `SharedTemporalProcessing` | ✅ Deployed | All mixing strategies |
| `SharedFeedforward` | ✅ Deployed | All block types |
| `DiffusionCachedIntermediates` | ⚠️ Duplication | Consolidate into unified workspace |
| `RgLru` streaming state | ⚠️ Dual workspaces | Merge into unified workspace |
| Linear projections | ⚠️ Inconsistent | Standardize across all blocks |

---

## Phase 5.2b: Diffusion Intermediate Consolidation

### Current Issue
`DiffusionBlock.cached_intermediates` (wrapped in `RwLock`) contains 16 Arc-wrapped fields:
- `input_original`, `input_used` (2x duplicates)
- `time_embed`, `gamma_beta`, `h_vec` (time-specific)
- `norm1_out`, `norm1_mod`, `attn_out`, `residual1` (in unified workspace)
- `norm2_out`, `norm2_mod` (in unified workspace)
- `ffn_out`, `output` (in unified workspace)
- `gamma_attn`, `beta_attn`, `gamma_ffn`, `beta_ffn` (FiLM modulation)

### Solution
**Map into `UnifiedLayerWorkspace` + extension fields:**

```rust
pub struct UnifiedLayerWorkspace {
    // Existing core buffers (6)
    norm1_out: Option<Array2<f32>>,
    temporal_out: Option<Array2<f32>>,   // attn_out
    residual1: Option<Array2<f32>>,
    norm2_out: Option<Array2<f32>>,
    ffn_intermediate: Option<Array2<f32>>,
    ffn_out: Option<Array2<f32>>,
    
    // NEW: Streaming & context (2)
    streaming_state: Option<Array2<f32>>,
    context_buffer: Option<Array2<f32>>,
    
    // NEW: Diffusion-specific (4)
    input_buffer: Option<Array2<f32>>,    // cache input_original & input_used
    time_embed: Option<Array1<f32>>,
    film_gamma: Option<Array2<f32>>,      // [batch, embed*4] for all FiLM scales
    film_beta: Option<Array2<f32>>,       // [batch, embed*4] for all FiLM scales
    
    // Metadata
    expected_shape: Option<(usize, usize)>,
    allocation_limit: usize,
    allocation_count: u32,
}
```

### Implementation Steps
1. ✅ Add 4 new fields to `UnifiedLayerWorkspace`
2. ✅ Implement `ensure_capacity()` for Diffusion sizes
3. ✅ Update `DiffusionBlock::forward()` to use unified workspace
4. ✅ Remove `cached_intermediates` RwLock wrapper
5. ✅ Update gradient computation to use unified pointers
6. ✅ Verify all 484 tests pass
7. ✅ Measure allocation reduction

### Memory Impact
- **Before**: `DiffusionCachedIntermediates` = 16 Arc allocations + 2 RwLock overhead
- **After**: Unified workspace with lazy allocation = 2-4 allocations per forward pass
- **Expected reduction**: ~60-80 Arc allocations per batch

---

## Phase 5.3b: Streaming Workspace Unified Interface

### Current Issue
`RgLru` and `MoHRgLru` maintain parallel workspaces:
- `unified_workspace: UnifiedLayerWorkspace` (batch processing)
- `streaming_workspace: RgLruStreamingWorkspace` (token-by-token)

These are **orthogonal** but require separate management.

### Solution
**Implement `StreamingWorkspaceManaged` trait:**

```rust
pub trait StreamingWorkspaceManaged: WorkspaceManaged {
    fn ensure_streaming_capacity(&mut self, embed_dim: usize);
    fn step_streaming(&mut self, token: &Array1<f32>) -> Array1<f32>;
    fn reset_streaming(&mut self);
    fn streaming_state(&self) -> Option<&Array2<f32>>;
}
```

Apply to:
- `RgLru` (single-head state)
- `MoHRgLru` (multi-head state aggregation)
- `Mamba` (if selective scanning required)

### Benefits
- **Unified state management**: Single initialization point
- **Reduced code duplication**: ~100 LOC in SSM layer
- **Predictable memory**: Power-of-2 padding applies to streaming state too

---

## Phase 5.4: Linear Projection Unification

### Current Inconsistency

| Block Type | Projection Pattern | File |
|---|---|---|
| Transformer | Direct matrix multiply | `shared_temporal.rs` |
| Diffusion | Direct matrix multiply | `temporal_processing.rs` |
| RG-LRU | Custom `LinearProjection` | `rg_lru.rs` |
| Mamba | Custom `LinearProjection` | `mamba.rs` |

### Solution
**Extract unified `ProjectionLayer` trait:**

```rust
pub trait ProjectionLayer {
    fn forward(&self, x: &Array2<f32>) -> Array2<f32>;
    fn parameters(&self) -> usize;
}

impl ProjectionLayer for MatMulProjection {
    // Transformer/Diffusion style: x @ W + b
}

impl ProjectionLayer for LinearProjection {
    // SSM style: enhanced with learnable bias, normalization
}
```

**Benefits**: 
- Single testing surface for all projections
- Consistent gradient computation
- Easy to swap implementations (e.g., low-rank, sparse)

---

## Implementation Priority (Next 3 Sessions)

### Session 1: Diffusion Consolidation (4-6 hours)
- [ ] Extend `UnifiedLayerWorkspace` with Diffusion fields
- [ ] Refactor `DiffusionBlock::forward()` to use unified workspace
- [ ] Update `DiffusionBlock::backward()` gradient routing
- [ ] Remove `DiffusionCachedIntermediates` RwLock
- [ ] Test: 484/484 passing
- [ ] Benchmark: Allocation count reduction

### Session 2: SSM Streaming Unification (4-6 hours)
- [ ] Implement `StreamingWorkspaceManaged` trait
- [ ] Apply to `RgLru` and `MoHRgLru`
- [ ] Consolidate `RgLruStreamingWorkspace` logic
- [ ] Update token-by-token inference paths
- [ ] Test: 484/484 passing
- [ ] Benchmark: Streaming state reuse

### Session 3: Linear Projection Unification (3-4 hours)
- [ ] Extract `ProjectionLayer` trait
- [ ] Implement for MatMul and custom LinearProjection
- [ ] Unified testing harness
- [ ] Update all SSM blocks to use trait
- [ ] Test: 484/484 passing
- [ ] Code coverage report

---

## Optimization Targets

| Metric | Baseline | Target | Validation |
|---|---|---|---|
| Allocations/step | 50-60 | 30-35 | Use `allocation_count()` in workspace |
| Peak memory | 2.0 GB | 1.6 GB | Measure with `estimate_memory_usage()` |
| Latency | 450ms | 380ms | `cargo bench` with 1000-step profile |
| Test coverage | 100% | 100%+ | 484/484 passing + new tests |
| Code duplication | 300+ LOC | 0 LOC | Audit shared component LOC |

---

## Validation Plan

### Unit Tests
- `UnifiedLayerWorkspace` capacity handling
- `StreamingWorkspaceManaged` state isolation
- `ProjectionLayer` gradient correctness

### Integration Tests
- `TransformerBlock` forward/backward with unified workspace
- `DiffusionBlock` forward/backward with unified workspace
- `RgLru` streaming inference correctness
- Cross-block compatibility (mixed architectures)

### Benchmarks
```bash
cargo bench --bench memory_efficiency  # Allocation count
cargo bench --bench latency_profile    # Forward pass timing
cargo bench --bench gradient_computation  # Backward pass timing
```

### Code Metrics
```bash
cargo clippy --all-targets
cargo fmt -- --check
# LOC audit: grep -r "Arc<Array" src/ | wc -l
```

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gradient computation breaks | Medium | Extensive backward pass tests |
| Memory usage increases | Low | Use `estimate_memory_usage()` validation |
| Streaming state corruption | Medium | Unit test streaming reset/step paths |
| Arc aliasing issues | Low | Clippy strict checks; no unsafe code |

---

## Success Criteria

✅ All 484 tests pass
✅ -40% allocation reduction verified in benchmarks
✅ Zero code duplication in shared interfaces
✅ Streaming state fully integrated into `WorkspaceManaged`
✅ Linear projections unified under trait
✅ Performance within -15% latency target
✅ Git history clean (1-2 commits per phase)
