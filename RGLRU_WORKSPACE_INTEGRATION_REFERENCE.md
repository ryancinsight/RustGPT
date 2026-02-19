# RG-LRU Workspace Integration Reference
**Completed**: Feb 13, 2026  
**Task**: P0-1 - RG-LRU Workspace Integration  
**Status**: ✅ COMPLETE

---

## Quick Summary

RgLru and MoHRgLru now implement unified workspace management via two traits:
1. **WorkspaceManaged** - For batch processing buffer allocation/lifecycle
2. **StreamingWorkspaceManaged** - For token-by-token inference state management

---

## Key Implementation Details

### RgLru Workspace Field
```rust
pub struct RgLru {
    // ... existing fields ...
    pub unified_workspace: UnifiedLayerWorkspace,
}
```

### WorkspaceManaged Implementation
Delegates buffer management to `unified_workspace`:
```rust
impl WorkspaceManaged for RgLru {
    fn ensure_capacity(&mut self, batch_size, seq_len, embed_dim) {
        self.unified_workspace.ensure_capacity(batch_size, seq_len, embed_dim);
    }
    
    fn clear_workspace(&mut self) {
        self.unified_workspace.clear_workspace();
        // Also clear caches
        self.cached_input = None;
        // ... etc
    }
}
```

### StreamingWorkspaceManaged Implementation
Manages streaming state lifecycle:
```rust
impl StreamingWorkspaceManaged for RgLru {
    fn init_streaming(&mut self, batch_size, _embed_dim) -> Result<()> {
        // 1. Ensure unified workspace has capacity
        self.unified_workspace.ensure_capacity(batch_size, 1, self.embed_dim);
        
        // 2. Enable streaming state buffer
        self.unified_workspace.set_streaming_state_enabled(true);
        
        // 3. Initialize streaming workspace with zeroed buffers
        let h_prev = Array1::zeros(self.embed_dim);
        let r_pre = Array1::zeros(self.embed_dim);
        let i_pre = Array1::zeros(self.embed_dim);
        let r = Array1::zeros(self.embed_dim);
        let i = Array1::zeros(self.embed_dim);
        let a = Array1::zeros(self.embed_dim);
        
        self.streaming_workspace = Some(RgLruStreamingWorkspace {
            h_prev, r_pre, i_pre, r, i, a,
        });
        
        Ok(())
    }
    
    fn reset_streaming_state(&mut self) {
        // Called between sequences - zero all state buffers
        if let Some(ref mut ws) = self.streaming_workspace {
            ws.h_prev.fill(0.0);
            ws.r_pre.fill(0.0);
            ws.i_pre.fill(0.0);
            ws.r.fill(0.0);
            ws.i.fill(0.0);
            ws.a.fill(0.0);
        }
    }
    
    fn is_streaming(&self) -> bool {
        self.streaming_workspace.is_some()
    }
}
```

### MoHRgLru Coordination
Multi-head variant delegates to individual heads:
```rust
impl WorkspaceManaged for MoHRgLru {
    fn ensure_capacity(&mut self, batch_size, seq_len, _embed_dim) {
        for head in &mut self.heads {
            head.ensure_capacity(batch_size, seq_len, self.head_dim);
        }
    }
}

impl StreamingWorkspaceManaged for MoHRgLru {
    fn init_streaming(&mut self, batch_size, _embed_dim) -> Result<()> {
        // Initialize each head
        for head in &mut self.heads {
            head.init_streaming(batch_size, self.head_dim)?;
        }
        
        // Create MoH-specific streaming workspace
        let moh_ws = MoHRgLruStreamingWorkspace {
            moh_workspace: MoHStreamingWorkspace {
                xw: Array1::zeros(self.moh.w_g.nrows()),
                g: Array1::zeros(self.num_heads),
                m: Array1::zeros(self.num_heads),
            },
            output_buffer: Array1::zeros(self.embed_dim),
            head_output_buffer: Array1::zeros(self.head_dim),
        };
        
        self.streaming_workspace = Some(moh_ws);
        Ok(())
    }
}
```

---

## Usage Patterns

### Batch Processing
```rust
let mut rglru = RgLru::new(embed_dim);

// Before batch forward pass
rglru.ensure_capacity(batch_size, seq_len, embed_dim);

// Forward/backward passes
let output = rglru.forward(&input);
let grad_input = rglru.backward(&grad_output, learning_rate);

// After batch - clear buffers
rglru.clear_workspace();
```

### Streaming Inference (Token-by-Token)
```rust
let mut rglru = RgLru::new(embed_dim);

// Initialize streaming before first token
rglru.init_streaming(batch_size, embed_dim)?;

// Process tokens one at a time
for token_input in token_stream {
    let output = rglru.forward(&token_input);
    // h_prev, r_pre, i_pre retained for next token
}

// Reset between sequences
rglru.reset_streaming_state();

// Process next sequence
for token_input in next_sequence {
    // ...
}
```

---

## Memory Management

### Workspace Buffers
The `UnifiedLayerWorkspace` provides:
- `streaming_state: Option<Array2<f32>>` - RNN state
- `context_buffer: Option<Array2<f32>>` - Attention context

### RgLru-Specific State
- `h_prev: Array1<f32>` - Hidden state from previous timestep
- `r_pre: Array1<f32>` - Reset gate pre-activation
- `i_pre: Array1<f32>` - Input gate pre-activation
- `r, i, a: Array1<f32>` - Gate and alpha values

### Allocation Strategy
- Power-of-2 sizing in `UnifiedLayerWorkspace`
- Lazy allocation (first call to `ensure_capacity`)
- Reuse across steps in streaming mode
- Explicit clearing to free memory

---

## Integration Points

### With TransformerBlock
RgLru can be used as temporal mixing layer in TransformerBlock:
```rust
pub struct TransformerBlock {
    temporal_mixing: TemporalMixingType,  // Can be RgLru
    unified_workspace: UnifiedLayerWorkspace,
}

// Both delegate to same unified workspace
```

### With Forward Pass
RgLru's `forward()` method is called via `Layer` trait:
```rust
impl Layer for RgLru {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.compute_forward_cached(input)
    }
}
```

Workspace is allocated separately via `WorkspaceManaged`:
```rust
block.ensure_capacity(batch_size, seq_len, embed_dim);
output = rglru.forward(&input);  // Uses allocated buffers
```

---

## Testing

### Unit Tests (All Pass)
- `test_rg_lru_forward_shape` - Verifies output shape
- `test_rg_lru_grad_shapes` - Checks gradient dimensions
- `test_rg_lru_gate_ranges` - Validates gate activation ranges
- `test_rg_lru_recurrence_matches_state_computation` - Streaming correctness

### Integration Tests (All Pass)
- `test_transformer_block_streaming_consistency_rglru` - Batch vs streaming equivalence
- `test_transformer_block_streaming_consistency_rglru_moh` - Multi-head coordination

---

## Next Tasks (Dependency Chain)

### P0-2: Streaming Workspace Consolidation (NEXT)
- Apply same pattern to: PolyAttention, SlidingWindow, RingAttention, Mamba, Mamba2
- Share streaming workspace implementation
- Expected: 120+ LOC reduction

### P1-1: In-Place Operations
- Add `forward_into()` methods
- Eliminate intermediate allocations
- Expected: 10-15% speedup

### P1-2: Global Buffer Pooling
- Create power-of-2 sizing pool
- Share across all layers
- Expected: 20% reduction in allocation overhead

---

## Potential Issues & Solutions

### Issue: Unbounded Streaming State
**Problem**: If streaming state is not reset, memory grows indefinitely
**Solution**: Call `reset_streaming_state()` between sequences or `clear_workspace()` explicitly

### Issue: Mismatched Dimensions
**Problem**: `init_streaming()` called with wrong embed_dim
**Solution**: Use `self.embed_dim` instead of parameter (now correctly implemented)

### Issue: Workspace Not Allocated
**Problem**: `forward()` called without prior `ensure_capacity()`
**Solution**: Always call `ensure_capacity()` before streaming or use `Layer::forward()` which assumes allocation

---

## File Locations

- Implementation: `src/domain/layers/ssm/rg_lru.rs` (lines 964-1036 for RgLru traits)
- Trait definitions: `src/domain/layers/components/workspace_managed.rs`
- Unified workspace: `src/domain/layers/components/unified_layer_workspace.rs`
- Tests: `tests/transformer_block_verification.rs` (integration tests)

---

## Migration Checklist (for other SSM layers)

- [ ] Add `unified_workspace: UnifiedLayerWorkspace` field
- [ ] Implement `WorkspaceManaged` trait
- [ ] Implement `StreamingWorkspaceManaged` trait
- [ ] Update all struct constructors/initialization
- [ ] Add tests for workspace allocation/reuse
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Benchmark before/after memory usage

---

**Status**: Complete and ready for production use.
