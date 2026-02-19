# Phase 5.3: RgLru (SSM) Integration Plan

## Status: PLANNING (Ready to Start)

### Objective
Consolidate RgLru and MoHRgLru workspace management by implementing `StreamingWorkspaceManaged` trait, unifying state management with `UnifiedLayerWorkspace`.

### Current RgLru Structure
```rust
RgLru {
  streaming_workspace: Option<RgLruStreamingWorkspace>,  // 1D state buffers
  cached_input: Option<Array2<f32>>,                      // For backward pass
  cached_r: Option<Array2<f32>>,                          // For backward pass
  cached_i: Option<Array2<f32>>,                          // For backward pass
  cached_a: Option<Array2<f32>>,                          // For backward pass
  cached_hprev: Option<Array2<f32>>,                      // For backward pass
}

RgLruStreamingWorkspace {
  h_prev: Array1<f32>,    // Recurrent state
  r_pre: Array1<f32>,     // Reset gate pre-activation
  i_pre: Array1<f32>,     // Input gate pre-activation
  r: Array1<f32>,         // Reset gate
  i: Array1<f32>,         // Input gate
  a: Array1<f32>,         // Recurrence coefficient
}

MoHRgLru {
  streaming_workspace: Option<MoHRgLruStreamingWorkspace>,  // MoH-specific state
  heads: Vec<RgLru>,
  cached_input: Option<Array2<f32>>,
  cached_eff: Option<Array2<f32>>,
  cached_head_out: Option<Vec<Array2<f32>>>,
  moh: MoHGating,
}
```

### Integration Strategy

#### Phase 5.3a: Add UnifiedLayerWorkspace (optional)
- Note: RgLru uses 1D stepping, so unified workspace (2D) may be less useful
- Keep streaming_workspace as-is for token-by-token processing
- Could add unified workspace for batch processing if needed in future

#### Phase 5.3b: Implement StreamingWorkspaceManaged
```rust
pub trait StreamingWorkspaceManaged: WorkspaceManaged {
    fn init_streaming(&mut self, batch_size: usize, embed_dim: usize) -> Result<()>;
    fn step_streaming(&mut self, token_idx: usize) -> Result<()>;
    fn reset_streaming(&mut self) -> Result<()>;
}
```

For RgLru:
- `init_streaming`: Initialize h_prev, r_pre, i_pre, r, i, a buffers
- `step_streaming`: Validate token_idx is in bounds
- `reset_streaming`: Zero h_prev and all activations

#### Phase 5.3c: Batch Processing Path
- RgLru currently supports both:
  - Streaming forward (token-by-token with state)
  - Batch forward (full sequence at once)
- Batch path uses cached buffers for gradient computation
- Keep both paths independent

### Complexity Assessment
- **Difficulty**: LOW-MEDIUM
  - Streaming workspace already decoupled
  - Trait implementation is straightforward
  - Limited integration points with unified workspace

- **Lines to change**: ~100-150 LOC
  - StreamingWorkspaceManaged impl
  - Maybe small refactors in forward/backward paths

### Testing Requirements
- All 484 existing tests must pass
- RgLru-specific tests: test_rg_lru_forward_shape, test_rg_lru_grad_shapes, etc.
- MoHRgLru tests: test_moh_rg_lru_forward_shape, test_moh_rg_lru_grad_shapes
- Gradient correctness tests

### Timeline Estimate
- Implementation: 1-2 hours
- Testing & validation: 1 hour
- **Total**: 2-3 hours

### Dependencies
- ✓ Phase 5.1 completed (UnifiedLayerWorkspace working)
- ✓ Phase 5.2 completed (DiffusionBlock updated)
- ✓ StreamingWorkspaceManaged trait exists (in workspace_managed.rs)

### Expected Outcomes
After Phase 5.3:
- ✓ All three block types (Transformer, Diffusion, SSM/RgLru) use unified pattern
- ✓ StreamingWorkspaceManaged trait implemented for RgLru
- ✓ Workspace management consistent across architectures
- ✓ 484 tests passing

### Next: Phase 5.4 - Performance Validation
- Benchmark allocation count: 30-35/step (vs current 50-60)
- Benchmark latency: -15% improvement (vs baseline)
- Memory peak: -20% reduction
- Lock in improvements with performance tests
