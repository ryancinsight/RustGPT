# Session Execution Plan - Consolidation & GPU Optimization (Feb 14, 2026)

## Objective
Continue consolidation and cleanup while optimizing performance and memory efficiency of shared components between diffusion, SSM, and transformer architectures. Implement GPU backend variants with automatic GPU detection (no fallback for troubleshooting).

---

## Current Status (Before Session)
- **Build**: ✅ 529 tests passing
- **Phase 5.3**: ✅ 95% complete (GPU infrastructure)
- **Consolidation**: 44% complete (focus on streaming workspace unification)

---

## Priority Queue (Ordered by Impact + Effort)

### P0: Streaming Workspace Consolidation (2-3 hours)
**Objective**: Unify streaming state management across Mamba, PolyAttention, SlidingWindow, RingAttention

**Current State**:
- ✅ RgLru: Already implements `StreamingWorkspaceManaged`
- ⏳ Mamba: Has `MambaStreamingWorkspace` struct, needs trait impl
- ⏳ PolyAttention: Has `PolyAttentionStreamingWorkspace`, needs trait impl
- ⏳ SlidingWindow: Has `SlidingWindowStreamingWorkspace`, needs trait impl
- ⏳ RingAttention: Has `RingAttentionStreamingWorkspace`, needs trait impl

**Impact**:
- Unified API across all streaming components (-120 LOC)
- Enable batch streaming inference mode
- Foundation for async execution

**Tasks**:
1. **Mamba** (45 min)
   - Implement `StreamingWorkspaceManaged` for `Mamba`
   - Add `init_streaming()` using existing `MambaStreamingWorkspace`
   - Update lifecycle methods (reset, finalize)
   
2. **PolyAttention** (30 min)
   - Implement trait for PolyAttention
   - Consolidate manual workspace patterns

3. **SlidingWindow & RingAttention** (30 min)
   - Quick implementations leveraging existing patterns

---

### P1: In-Place Operations Framework (4-5 hours)
**Objective**: Eliminate intermediate allocations in hot paths (10-15% speedup expected)

**Current State**:
- Trait patterns defined in `WorkspaceManaged`
- SharedFeedforward & SharedTemporalProcessing have GPU methods but no CPU `forward_into()`

**Impact**:
- 10-15% inference speedup
- Reduced allocator pressure

**Tasks**:
1. Implement `forward_into()` for SharedFeedforward
2. Implement `forward_into()` for SharedTemporalProcessing
3. Update TransformerBlock/DiffusionBlock call sites
4. Benchmark before/after

---

### P1: Global Buffer Pooling (3-4 hours)
**Objective**: Implement power-of-2 buffer sizing hierarchy with global reuse

**Current State**:
- `UnifiedLayerWorkspace` exists but no global pooling
- `IntermediateBufferPool` is separate
- No power-of-2 sizing hierarchy

**Impact**:
- 20% reduction in allocation overhead
- Consistent memory footprint

**Tasks**:
1. Design `GlobalBufferPool` with buckets: 2^8 → 2^20
2. Integrate with `UnifiedLayerWorkspace`
3. Add TLS-backed pooling for streaming ops
4. Add metrics (pool hit rate, fragmentation)

---

### P2: Mixed Precision Support (2-3 hours)
**Objective**: FP16/BF16 context buffers (~50% memory reduction)

---

## Implementation Sequence

### Phase 1: Streaming Consolidation (First)
```rust
// Goal: This pattern repeated for 4 components
impl WorkspaceManaged for Mamba {
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        self.unified_workspace.ensure_capacity(batch_size, seq_len, embed_dim);
    }
    fn clear_workspace(&mut self) {
        self.unified_workspace.clear_workspace();
        self.streaming_workspace = None;
        self.streaming_ssm_state = None;
        self.streaming_conv_queue = None;
    }
    fn workspace_stats(&self) -> WorkspaceStats {
        self.unified_workspace.workspace_stats()
    }
}

impl StreamingWorkspaceManaged for Mamba {
    fn init_streaming(&mut self, batch_size: usize, embed_dim: usize) -> Result<()> {
        self.unified_workspace.ensure_capacity(batch_size, 1, embed_dim);
        self.streaming_ssm_state = Some(Array2::zeros((batch_size, self.state_size)));
        self.streaming_conv_queue = Some(VecDeque::new());
        self.streaming_workspace = Some(Box::new(MambaStreamingWorkspace::new(
            self.embed_dim, self.state_size, self.conv_kernel
        )));
        self.is_streaming_mode = true;
        Ok(())
    }
    
    fn reset_streaming_state(&mut self) {
        if let Some(state) = &mut self.streaming_ssm_state {
            state.fill(0.0);
        }
        if let Some(queue) = &mut self.streaming_conv_queue {
            queue.clear();
        }
    }
    
    fn is_streaming(&self) -> bool {
        self.is_streaming_mode
    }
}
```

### Phase 2: Verify Consolidation
- Run full test suite (529+ tests)
- Verify streaming inference still works
- Check memory stats

### Phase 3: In-Place Operations (If Time)
- Implement CPU `forward_into()` methods
- Update hot paths in TransformerBlock

---

## Build & Test Strategy

### Quick Check (After Each Task)
```bash
cargo check                    # Syntax only
cargo clippy --all-targets     # Linting
```

### Full Verification (After Phase Complete)
```bash
cargo test --lib              # All 529+ tests
cargo test --lib --features gpu-wgpu  # GPU tests
```

### Benchmarking (For Performance Claims)
```bash
cargo bench --bench unified_workspace_bench
```

---

## Success Criteria

### P0 (Streaming): ✅ MUST DO
- [ ] Mamba implements `StreamingWorkspaceManaged`
- [ ] PolyAttention implements `StreamingWorkspaceManaged`
- [ ] All 529 tests pass
- [ ] Streaming inference mode works correctly

### P1 (In-Place): SHOULD DO
- [ ] SharedFeedforward has `forward_into()` implementation
- [ ] At least one `forward_into()` in call path
- [ ] Benchmark shows measurable speedup

### P1 (Global Pooling): NICE TO HAVE
- [ ] GlobalBufferPool design document
- [ ] Basic integration with UnifiedLayerWorkspace
- [ ] Metrics collection

---

## Files to Modify/Create

### Consolidation (P0)
- `src/domain/layers/ssm/mamba.rs` - Add trait impls
- `src/domain/attention/poly_attention.rs` - Add trait impls
- `src/domain/attention/sliding_window_attention.rs` - Add trait impls
- `src/domain/attention/ring_attention.rs` - Add trait impls

### Optimization (P1)
- `src/domain/layers/components/feedforward.rs` - Add `forward_into()`
- `src/domain/layers/components/temporal_processing.rs` - Add `forward_into()`
- `src/domain/blocks/transformer_block.rs` - Update call sites

### New (P1)
- `src/domain/compute/global_buffer_pool.rs` - NEW file

---

## GPU Detection Strategy (No Fallback)

All GPU operations follow this pattern:
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Strict requirement: GPU must be available
    self.require_gpu_ready()?;  // Errors if GPU not attached
    
    // Use GPU
    self.gpu_device
        .as_ref()
        .ok_or("GPU device not available")?
        .forward(&input)
}

// Helper
fn require_gpu_ready(&self) -> Result<()> {
    self.gpu_device
        .as_ref()
        .ok_or("Automatic GPU detection failed: no supported GPU backend was detected")?;
    Ok(())
}
```

---

## Metrics & Logging

### Memory Tracking
- Enable `RUST_LOG=debug` for workspace allocation logs
- Track allocation/deallocation events in consolidation phase

### Performance Profiling
- Use `cargo flamegraph` for hot path analysis
- Compare shared_feedforward CPU vs forward_into

---

## Time Estimate

| Task | Effort | Priority | Complexity |
|------|--------|----------|------------|
| Mamba trait impl | 45 min | P0 | Medium |
| PolyAttention trait | 30 min | P0 | Low |
| SlidingWindow/RingAttention | 30 min | P0 | Low |
| **Subtotal (P0)** | **105 min** | | |
| In-place ops framework | 4-5h | P1 | High |
| Global pooling | 3-4h | P1 | High |
| **Total** | **8-10h** | | |

---

## Next Session Handoff

If consolidation completes early, focus should shift to:
1. **Async GPU execution** - Enable overlapped compute/transfer
2. **Mixed precision** - FP16 context buffers
3. **Kernel fusion** - Combine GEMM + Activation
4. **Batch streaming** - Multi-token inference mode

---

## References

- **Status**: `CONSOLIDATION_PRIORITY_MATRIX_FEB14.md`
- **GPU Backend**: `CONSOLIDATION_GPU_BACKEND_SESSION_SUMMARY.md`
- **Traits**: `src/domain/layers/components/workspace_managed.rs`
- **Examples**: `src/domain/layers/ssm/rg_lru.rs` (reference implementation)
