# GPU Consolidation - Phase 5.6 (February 15, 2026)

## Objective
Consolidate and optimize shared components between diffusion, SSM, and transformer while implementing GPU backend variants with automatic GPU detection and strict no-fallback semantics.

## Current Status

### ✅ Completed
1. **GPU Infrastructure**: GpuDevice, GpuMatrixOps trait, GpuMemoryPool trait
2. **Unified GPU Executor**: UnifiedGpuExecutor, UnifiedGpuBufferPool
3. **WGPU Kernels**: GEMM, Softmax, Element-wise ops, Layer norm (4711 lines)
4. **Test Coverage**: 539 passing tests (no failures)

### 🔄 In Progress
1. **gpu_ops.rs Deprecation**: CpuGpuMatrixOps still deprecated (3 warnings)
2. **Placeholder GPU Implementations**: feedforward_gpu.rs, temporal_processing_gpu.rs have stubs
3. **Shared Component Integration**: Need GpuComponent trait implementations

## Work Items

### Phase 1: Remove Deprecated CpuGpuMatrixOps (Priority: HIGH)

**File**: `src/domain/compute/gpu_ops.rs`

**Change**: Remove the deprecated `CpuGpuMatrixOps` struct and its implementation.

**Rationale**:
- It's marked as deprecated
- Users should use GpuDevice::auto_detect() + backend-specific GpuMatrixOps
- Enables strict no-fallback semantics
- Reduces technical debt

**Impact**: 
- Removes 3 compiler warnings
- Forces explicit GPU backend selection
- Maintains API stability (GpuMatrixOps trait remains)

---

### Phase 2: Memory Optimization - Power-of-2 Sizing Verification

**Files**:
- `src/domain/compute/unified_gpu_buffer_pool.rs`
- `src/domain/compute/unified_gpu_executor.rs`

**Objectives**:
1. Verify all buffer allocations follow power-of-2 sizing
2. Add metrics for allocation efficiency
3. Benchmark reuse rates across forward passes

**Implementation**:
```rust
// Add allocation stats tracking
pub struct AllocationStats {
    total_allocated: usize,
    total_wasted_padding: usize,  // Power-of-2 waste
    reuse_count: usize,
    resize_count: usize,
}

impl AllocationStats {
    pub fn efficiency(&self) -> f32 {
        (self.total_allocated - self.total_wasted_padding) as f32 / self.total_allocated as f32
    }
}
```

---

### Phase 3: Shared Component GPU Integration

**Components Needing GpuComponent Implementation**:

#### A. SharedFeedforward (`src/domain/layers/components/feedforward.rs`)
- Status: CPU only
- GPU Path Needed: RichardsGLU kernel fusion
- Kernel Design:
  ```
  // Fused kernel
  x1, x2 = linear_split(input)  // 2 GEMMs
  x2 = richards(x2)              // Activation 
  output = x1 * x2               // Element-wise multiply
  ```

#### B. SharedTemporalProcessing (`src/domain/layers/components/temporal_processing.rs`)
- Status: Stubs in temporal_processing_gpu.rs
- GPU Paths Needed:
  1. **PolyAttention**: Polynomial basis + gating kernel
  2. **Mamba/RG-LRU**: Recurrent scan kernel
  3. **TransformerAttention**: Scaled dot-product attention

#### C. SharedAttentionContext (`src/domain/layers/components/attention_context.rs`)
- Status: Baseline implementations exist
- GPU Optimization: Context modulation kernel fusion

---

### Phase 4: Performance Profiling & Kernel Optimization

**Target Metrics**:
- **GEMM**: 50-100+ TFLOPS
- **Numerical Accuracy**: ε ≤ 1e-4 vs CPU reference
- **Memory Transfer**: <1% of compute time
- **Reallocation Frequency**: ≤ 2 per full training epoch

**Benchmarks**:
- GEMM sizes: 256x256x512, 1024x1024x1024 (common transformer sizes)
- Attention sizes: 16x64 (heads x head_dim), variable seq_len
- Feedforward: embedding_dim x 4*embedding_dim splits

---

### Phase 5: Feature-Gated Backend Support

**Current**: WGPU (primary focus)
**Roadmap**:
- Phase 5.7: CUDA (via cuBLAS + custom kernels)
- Phase 5.8: Metal (Apple Silicon optimization)

**Feature Flags**:
```toml
[features]
gpu-wgpu = ["wgpu", "pollster"]
gpu-cuda = ["cuda-sys", "cublas-sys"]
gpu-metal = ["metal", "metal-performance-shaders"]
```

---

## Implementation Order

1. **Immediate** (Session Priority):
   - [ ] Remove CpuGpuMatrixOps deprecation
   - [ ] Verify wgpu_ops kernel correctness
   - [ ] Add allocation stats to UnifiedGpuBufferPool

2. **Near-term**:
   - [ ] Implement GpuComponent trait for SharedFeedforward
   - [ ] Implement GpuComponent trait for SharedTemporalProcessing
   - [ ] Add benchmarks for memory efficiency

3. **Medium-term**:
   - [ ] Replace placeholder GPU kernels with actual WGPU code
   - [ ] Performance profiling and optimization
   - [ ] Integration tests for end-to-end GPU paths

4. **Future**:
   - [ ] CUDA backend implementation
   - [ ] Metal backend implementation
   - [ ] Distributed GPU support

---

## Testing Strategy

### Unit Tests
- Kernel correctness (GEMM, softmax, activations)
- Memory allocation and reuse
- Auto-detection with no-fallback

### Integration Tests
- End-to-end forward pass (GPU only)
- Multi-layer sequences
- Batch processing

### Benchmarks
- GEMM throughput (TFLOPS)
- Attention memory bandwidth
- Full model forward pass latency

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| GPU OOM on large batches | Power-of-2 sizing + pool stats |
| Numerical drift | Compare vs CPU reference (ε ≤ 1e-4) |
| Device incompatibility | Explicit error on missing GPU backend |
| Kernel correctness | Comprehensive test coverage + benchmarks |

---

## Success Criteria

- ✅ All tests pass (539 passing)
- ✅ Zero compiler warnings (remove CpuGpuMatrixOps)
- ✅ Memory efficiency ≥ 85% (power-of-2 waste < 15%)
- ✅ Numerical accuracy within tolerance
- ✅ No CPU fallbacks in strict no-fallback mode
