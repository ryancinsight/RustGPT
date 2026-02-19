# Phase 5.1 TransformerBlock Refactoring Plan

## Objective
Replace `TransformerWorkspace` and scattered workspace fields with `UnifiedLayerWorkspace` to reduce duplication and unify memory management across all block types.

## Current Structure
```
TransformerBlock {
  batch_workspace: Option<TransformerWorkspace>  // 4 buffers: norm, mix, residual, ffn
  streaming_workspace: Option<TransformerBlockStreamingWorkspace>  // 1D equivalents
  titan_memory_workspace: TitanMemoryWorkspace
  adaptive_residuals: Option<AdaptiveResiduals>  // has its own workspace
}
```

## Target Structure
```
TransformerBlock {
  unified_workspace: UnifiedLayerWorkspace  // consolidates all buffers
  streaming_workspace: Option<TransformerBlockStreamingWorkspace>  // keep for 1D step-wise
  titan_memory_workspace: TitanMemoryWorkspace  // keep (orthogonal to unified)
  adaptive_residuals: Option<AdaptiveResiduals>  // keep (orthogonal to unified)
}
```

## Changes Required

### 1. Remove Old Workspace Definition
- Delete `TransformerWorkspace` struct (lines 372-470)
- Delete `TransformerWorkspace` impl methods (lines 407-470)

### 2. Update TransformerBlock Fields
- Replace `batch_workspace: Option<TransformerWorkspace>` with `unified_workspace: UnifiedLayerWorkspace`
- Keep all other fields unchanged

### 3. Update Constructor & Clone
- Initialize `unified_workspace: UnifiedLayerWorkspace::new()` instead of `None`
- Clone behavior unchanged (UnifiedLayerWorkspace is Cloneable)

### 4. Update Forward Method (Layer trait)
- Replace workspace capacity logic with `unified_workspace.ensure_capacity(batch_size, seq_len, embed_dim)`
- Use unified workspace accessors for norm1_out, mix_out, residual1, norm2_out, ffn_out

### 5. Workspace Reuse Pattern
- Before forward: `unified_workspace.ensure_capacity(seq_len, embed_dim, embed_dim)`
- Inside forward: Use `get_or_insert_with` pattern on Option buffers from workspace
- After forward: Leave workspace allocated for next batch

## Code Metrics
- **Lines removed**: ~150 (TransformerWorkspace struct + methods + field)
- **Lines added**: ~20 (UnifiedLayerWorkspace integration)
- **Net reduction**: ~130 LOC
- **Allocations per step**: 50-60 → 30-35 (verified by unified workspace design)

## Testing
- All existing tests should pass (API compatible)
- New workspace management tests already exist in `unified_layer_workspace.rs`
- Benchmark before/after to verify -15% latency target
