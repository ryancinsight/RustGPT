# Phase 5.2: DiffusionBlock Integration Plan

## Status: PLANNING (Ready to Start)

### Objective
Consolidate DiffusionBlock's separate workspace fields and cached intermediate structures with UnifiedLayerWorkspace to reduce duplication and maintain unified memory management.

### Current DiffusionBlock Structure
```rust
DiffusionBlock {
  cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
  titan_memory_workspace: TitanMemoryWorkspace,
  // No explicit workspace buffers - uses cached_intermediates for all
}

DiffusionCachedIntermediates {
  input_original: Arc<Array2<f32>>,      // Can reuse via unified_workspace
  input_used: Arc<Array2<f32>>,           // Can reuse via unified_workspace  
  time_embed: Arc<Array1<f32>>,           // Separate - time conditioning (keep)
  gamma_beta: Arc<Array1<f32>>,           // Separate - time conditioning (keep)
  norm1_out: Arc<Array2<f32>>,            // → unified_workspace.norm1_out
  norm1_mod: Arc<Array2<f32>>,            // Separate - FILM modulation (keep)
  attn_out: Arc<Array2<f32>>,             // → unified_workspace.temporal_out
  residual1: Arc<Array2<f32>>,            // → unified_workspace.residual1
  norm2_out: Arc<Array2<f32>>,            // → unified_workspace.norm2_out
  norm2_mod: Arc<Array2<f32>>,            // Separate - FILM modulation (keep)
  ffn_out: Arc<Array2<f32>>,              // → unified_workspace.ffn_out
  output: Arc<Array2<f32>>,               // Final output (keep separately)
  h_vec: Arc<Array1<f32>>,                // Separate - for backward (keep)
  gamma_attn: Arc<Array2<f32>>,           // Separate - FILM (keep)
  beta_attn: Arc<Array2<f32>>,            // Separate - FILM (keep)
  gamma_ffn: Arc<Array2<f32>>,            // Separate - FILM (keep)
  beta_ffn: Arc<Array2<f32>>,             // Separate - FILM (keep)
  timestep: usize,                         // Metadata (keep)
}
```

### Integration Strategy

#### Phase 5.2a: Add UnifiedLayerWorkspace field
- Add `unified_workspace: UnifiedLayerWorkspace` to DiffusionBlock
- Keep cached_intermediates struct but reduce what's stored in it

#### Phase 5.2b: Reorganize cached intermediates  
- Keep only non-workspace buffers in DiffusionCachedIntermediates:
  - Time conditioning: `time_embed`, `gamma_beta`, `h_vec`
  - FILM modulation: `norm1_mod`, `norm2_mod`, `gamma_attn`, `beta_attn`, `gamma_ffn`, `beta_ffn`
  - Final outputs: `input_original`, `input_used`, `output`
  - Metadata: `timestep`
  
- Move to unified_workspace:
  - `norm1_out` → unified_workspace.norm1_out()
  - `attn_out` → unified_workspace.temporal_out()
  - `residual1` → unified_workspace.residual1()
  - `norm2_out` → unified_workspace.norm2_out()
  - `ffn_out` → unified_workspace.ffn_out()

### Complexity Assessment
- **Difficulty**: MEDIUM
  - Diffusion uses Arc for gradients (deep integration)
  - FILM modulation requires orthogonal storage (not in unified workspace)
  - Multiple backward compatibility concerns (cached intermediates API)

- **Lines to change**: ~200-300 LOC
  - forward_with_timestep method (major change)
  - Gradient computation (references to cached buffers)
  - Intermediate caching logic

### Testing Requirements
- All 484 existing tests must pass
- DiffusionBlock-specific tests must verify gradient computation
- Backward compatibility for cache retrieval API

### Timeline Estimate
- Implementation: 2-3 hours
- Testing & validation: 1 hour
- **Total**: 3-4 hours

### Dependencies
- ✓ Phase 5.1 completed (UnifiedLayerWorkspace working)
- ✓ WorkspaceManaged trait tested
- ✓ All infrastructure in place

### Next Phases After 5.2
- **Phase 5.3**: RgLru integration (StreamingWorkspaceManaged trait)
- **Phase 5.4**: Performance benchmarking and validation
- **Phase 5.5**: Documentation and knowledge transfer
