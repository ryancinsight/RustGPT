# Session Execution Plan - February 14, 2026

**Time Available**: Varies (planning for flexible engagement)  
**Primary Objectives**:
1. Fix build issue & verify tests pass
2. Implement streaming workspace consolidation (P0)
3. Begin in-place operations framework (P1)

---

## Phase 0: Build Verification (15-30 min)

### Task 0.1: Verify Compilation Fix ✅ DONE
- **Change**: Removed duplicate `moh_gate_activation` from gpu_ops.rs
- **Status**: Applied (lines 276-290 removed)

### Task 0.2: Trigger Clean Build
```bash
cargo clean  # Optional, if cache seems stale
cargo check  # Quick check
cargo build --release  # Full build
```

**Expected Time**: 3-5 minutes (slow due to WGPU shader compilation)

### Task 0.3: Run Tests
```bash
cargo test --lib --lib 2>&1 | tail -20  # Check final result
cargo test --test transformer_block_verification  # Integration test
```

**Expected Result**: 529 tests passing

---

## Phase 1: Streaming Workspace Consolidation (2-3 hours)

**Objective**: Implement `StreamingWorkspaceManaged` trait for remaining 4 components

### Overview: Current State
- ✅ RgLru - `impl StreamingWorkspaceManaged` (lines 1316-1362)
- ✅ MoHRgLru - `impl StreamingWorkspaceManaged` (lines 1706+)
- ⏳ Mamba - Manual `MambaStreamingState`, needs trait impl
- ⏳ PolyAttention - Manual `PolyAttentionStreamingWorkspace`, needs trait impl
- ⏳ SlidingWindow - Manual workspace, needs trait impl
- ⏳ RingAttention - Manual workspace, needs trait impl

### Task 1.1: Audit Mamba Streaming (30 min)

**File**: `src/domain/layers/ssm/mamba.rs`

1. Locate `MambaStreamingWorkspace` struct definition
2. Identify initialization points and lifecycle
3. Check for buffer allocation patterns
4. Document required traits to implement

**Success Criteria**:
- Understand current streaming state management
- Identify `init_streaming()`, `reset_streaming()`, `is_streaming()` patterns

### Task 1.2: Implement StreamingWorkspaceManaged for Mamba (45 min)

**Implementation Pattern** (based on RgLru):

```rust
impl StreamingWorkspaceManaged for Mamba {
    fn init_streaming(&mut self, batch_size: usize, _embed_dim: usize) -> Result<()> {
        // 1. Ensure unified_workspace has capacity
        self.unified_workspace.ensure_capacity(batch_size, 1, self.embed_dim);
        
        // 2. Enable streaming state
        self.unified_workspace.set_streaming_state_enabled(true);
        
        // 3. Initialize streaming workspace (h, c, x, etc.)
        self.streaming_workspace = Some(MambaStreamingWorkspace {
            // Fields from current MambaStreamingState
        });
        
        Ok(())
    }
    
    fn reset_streaming_state(&mut self) {
        if let Some(ref mut ws) = self.streaming_workspace {
            ws.h.fill(0.0);
            ws.c.fill(0.0);
            // ... reset all buffers
        }
    }
    
    fn is_streaming(&self) -> bool {
        self.streaming_workspace.is_some()
    }
}
```

**Steps**:
1. Add `unified_workspace: UnifiedLayerWorkspace` field (if not present)
2. Implement `init_streaming()` - initialize workspace + create streaming state
3. Implement `reset_streaming_state()` - zero out buffers between sequences
4. Implement `is_streaming()` - check if streaming is active
5. Update constructor: initialize `unified_workspace` to default

**Files to Modify**:
- `src/domain/layers/ssm/mamba.rs` - Add impl + field
- `src/domain/layers/ssm/mod.rs` - Ensure exports if needed

### Task 1.3: Implement StreamingWorkspaceManaged for PolyAttention (45 min)

**File**: `src/domain/attention/poly_attention.rs`

Follow same pattern as Mamba:
1. Add `unified_workspace` field (if not present)
2. Implement `init_streaming()` 
3. Implement `reset_streaming_state()`
4. Implement `is_streaming()`

**Specific Attention**:
- PolyAttention has head_dim varying with `with_exact_capacity` mode
- Must handle both fixed and variable-capacity streaming

### Task 1.4: Quick Pass on SlidingWindow & RingAttention (Optional, 30 min)

If time permits:
- Repeat pattern for `SlidingWindowStreamingWorkspace`
- Repeat pattern for `RingAttentionStreamingWorkspace`

---

## Phase 2: In-Place Operations Framework (1-2 hours, if time)

**Objective**: Add `forward_into()` methods to reduce allocations

### Task 2.1: Implement forward_into for SharedFeedforward (45 min)

**File**: `src/domain/layers/components/shared_feedforward.rs`

**Pattern**:
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    // 1. Validate dimensions
    assert_eq!(output.nrows(), input.nrows());
    assert_eq!(output.ncols(), self.embed_dim);
    
    // 2. Reuse intermediate buffers from workspace
    let mut intermediate = /* get buffer from workspace */;
    
    // 3. Compute forward without creating new arrays
    // output = W_out @ (activation(input @ W_in + b_in)) + b_out
    self.gemm_or_workspace_mul(input, &self.w_in, &mut intermediate)?;
    self.activation_in_place(&mut intermediate)?;
    self.gemm_or_workspace_mul(&intermediate, &self.w_out, output)?;
    
    Ok(())
}
```

**Success Criteria**:
- No new allocations during forward_into
- Same numerical output as forward()
- Tests pass

### Task 2.2: Implement forward_into for SharedTemporalProcessing (45 min)

**File**: `src/domain/layers/components/shared_temporal_processing.rs`

Follow same pattern - dispatch to variant's forward_into

---

## Testing & Validation

### Unit Tests for Streaming Consolidation
```bash
# After implementing Mamba streaming
cargo test --lib ssm::mamba 

# After PolyAttention
cargo test --lib attention::poly_attention

# Full test suite
cargo test --lib
```

### Integration Tests
```bash
cargo test --test transformer_block_verification
```

### Benchmarking (Optional)
```bash
# Before/after streaming consolidation
cargo bench --bench transformer_throughput

# Before/after in-place ops
cargo bench --bench diffusion_speed
```

---

## Session Checklist

- [ ] **Phase 0**: Build verification complete
  - [ ] Duplicate definition removed
  - [ ] `cargo check` passes
  - [ ] `cargo test --lib` shows 529 passing
  
- [ ] **Phase 1**: Streaming consolidation (Priority P0)
  - [ ] Mamba: `impl StreamingWorkspaceManaged` ✅
  - [ ] PolyAttention: `impl StreamingWorkspaceManaged` ✅
  - [ ] SlidingWindow: `impl StreamingWorkspaceManaged` (Optional)
  - [ ] RingAttention: `impl StreamingWorkspaceManaged` (Optional)
  - [ ] All tests passing
  
- [ ] **Phase 2**: In-place operations (Priority P1, if time)
  - [ ] SharedFeedforward: `forward_into()` (Optional)
  - [ ] SharedTemporalProcessing: `forward_into()` (Optional)
  - [ ] Benchmark & validate
  
- [ ] **Documentation**: Update status files
  - [ ] Session completion summary
  - [ ] Next session action items

---

## Expected Outcomes

### At Minimum (Phase 0-1, 1-2 hours)
- ✅ Clean build
- ✅ 529 tests passing
- ✅ Mamba streaming workspace integrated
- ✅ PolyAttention streaming workspace integrated
- **Impact**: -80 LOC, unified streaming API, ready for Phase 5.4

### If Extended (Phase 0-2, 3-4 hours)
- ✅ Everything above
- ✅ SharedFeedforward `forward_into()` framework
- ✅ SharedTemporalProcessing `forward_into()` framework
- **Impact**: Additional -60 LOC, 5-8% inference speedup potential

---

## Priority Principles

1. **Build First**: No forward progress without clean build
2. **Tests Always**: Every change must pass tests before moving on
3. **Consolidation Before Features**: Finish P0 streaming before P1 in-place ops
4. **Documentation**: Always end session with updated status files

---

## Key File Locations

**Trait Definition**:
- `src/domain/layers/components/workspace_managed.rs` - `StreamingWorkspaceManaged` trait

**Reference Implementations**:
- `src/domain/layers/ssm/rg_lru.rs:1316-1362` - RgLru impl (reference)
- `src/domain/layers/ssm/rg_lru.rs:1706+` - MoHRgLru impl (reference)

**Components to Modify**:
- `src/domain/layers/ssm/mamba.rs` - Mamba struct + impl
- `src/domain/attention/poly_attention.rs` - PolyAttention struct + impl
- `src/domain/attention/sliding_window_attention.rs` - SlidingWindow struct + impl
- `src/domain/attention/ring_attention.rs` - RingAttention struct + impl

**Shared Components**:
- `src/domain/layers/components/shared_feedforward.rs` - FFN
- `src/domain/layers/components/shared_temporal_processing.rs` - Temporal

---

**Next Action**: Start Phase 0 by triggering `cargo check` and monitoring build output.

