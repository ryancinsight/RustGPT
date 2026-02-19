# Phase 5.6.3: Consolidation & GPU Optimization Action Plan

**Date**: February 15, 2026  
**Focus**: Continue consolidation and cleanup while optimizing shared components for GPU with strict no-fallback auto-detection  
**Status**: In Progress

---

## 1. Consolidation Goals

### A. Shared Component Optimization
- [ ] **SharedAttentionContext** (`src/domain/layers/components/attention_context.rs`)
  - Optimize GPU context modulation: Reduce GEMM overhead with fused kernels
  - Verify GpuComponent trait implementation and auto-detection path
  - Benchmark similarity computation on GPU vs CPU

- [ ] **SharedFeedforward** (`src/domain/layers/components/feedforward.rs`)
  - Complete RichardsGLU fused kernel integration
  - Implement MoE variant for GPU (if applicable)
  - Verify zero-allocation forward paths on GPU

- [ ] **SharedTemporalProcessing** (`src/domain/layers/components/temporal_processing.rs`)
  - Optimize temporal mixing dispatch for all variants (Attention, Mamba, RG-LRU)
  - Ensure all Layer trait implementations work with GPU
  - Benchmark temporal operations on GPU

- [ ] **PolyAttention** (`src/domain/attention/poly_attention.rs`)
  - Verify polynomial attention on GPU
  - Implement fused Q/K/V projection → scoring → softmax pipeline
  - Test streaming workspace compatibility with GPU buffers

---

## 2. GPU Backend Consolidation

### A. Auto-Detection Strategy (No-Fallback)
- [ ] Verify `GpuDevice::auto_detect()` returns error if no GPU (never falls back to CPU)
- [ ] Test detection priority: CUDA > Metal > Vulkan > WGPU
- [ ] Confirm conditional feature compilation (`#[cfg(...)]` guards)

### B. GPU Component Trait Implementation
- [ ] Audit all shared components for complete `GpuComponent` implementation
- [ ] Verify `enable_gpu_auto_detect()` for strict no-fallback
- [ ] Ensure `is_gpu_ready()` correctly reflects GPU availability
- [ ] Test `ensure_capacity()` for buffer pre-allocation

### C. Memory Pool Consolidation
- [ ] Verify `UnifiedGpuBufferPool` power-of-2 sizing
- [ ] Confirm buffer reuse across components
- [ ] Benchmark allocation fragmentation

---

## 3. Performance Targets

| Component | Operation | CPU Ref | GPU Target | Speedup |
|-----------|-----------|---------|------------|---------|
| RichardsGLU | 1K batch | 50ms | 2ms | 25x |
| PolyAttention | 512 batch | 30ms | 1ms | 30x |
| Mamba Scan | 512 batch | 40ms | 2ms | 20x |
| AttentionContext | 1K batch | 15ms | 0.5ms | 30x |

---

## 4. Implementation Checklist

### Phase 4.1: RichardsGLU Fused Kernel Finalization
- [ ] Complete two-pass strategy in `richards_glu_fused_kernel.rs`
  - Pass 1: W1 @ input → x1, W2 @ input → x2, Richards activation → value, gate
  - Pass 2: (value * gate) @ W_out → output
- [ ] Implement GPU kernel dispatch (CUDA, Metal, WGPU)
- [ ] Add benchmark: Compare 5-launch naive vs 2-pass fused
- [ ] Integration test: Verify gradients flow correctly

### Phase 4.2: Shared Component GPU Wiring
- [ ] **AttentionContext GPU Path**
  - Wire `apply_incoming_context_gpu()` for forward
  - Ensure gradients work with GPU buffers
  - Test with attention similarity computation

- [ ] **Feedforward GPU Path**
  - Wire RichardsGLU GPU forward from `ComputeBackend`
  - Verify batch processing without allocations
  - Test gradient computation

- [ ] **TemporalProcessing GPU Path**
  - Dispatch to appropriate GPU kernel based on `TemporalMixingType`
  - Ensure zero-copy data flow through Attention/Mamba/RG-LRU variants
  - Benchmark temporal mixing on GPU

### Phase 4.3: PolyAttention GPU Optimization
- [ ] Implement fused attention kernel
  - Combine Q/K/V projection, scoring, softmax into single pass
  - Target 1-pass GPU implementation vs 3+ on CPU
- [ ] Optimize streaming workspace for GPU
- [ ] Benchmark poly attention (512 batch, 768 dims)

### Phase 4.4: Consolidation & Cleanup
- [ ] Remove dead code paths (CPU-only duplicate implementations)
- [ ] Consolidate error handling (strict no-fallback semantics)
- [ ] Update all tests to use GPU auto-detection
- [ ] Audit conditional compilation flags
- [ ] Verify zero test failures on all platforms (GPU and CPU-only builds)

---

## 5. Strict No-Fallback Verification

### Test Scenarios
- [ ] **GPU Available**: All GPU components initialized, GPU operations execute
- [ ] **GPU Not Available (No Fallback)**:
  - `GpuDevice::auto_detect()` returns `ModelError::Backend`
  - `component.enable_gpu_auto_detect()` returns error (NOT silent CPU fallback)
  - `component.forward_gpu()` returns error if GPU not attached
- [ ] **Partial GPU (Feature Flags)**:
  - CUDA runtime detected but binary compiled without `--features gpu-cuda`
  - Clear error message guiding user to recompile
- [ ] **Zero-Copy Forward**:
  - Input → Attention → Temporal → Feedforward → Output (all on GPU)
  - No downloads until final output retrieval

---

## 6. Code Review Checkpoints

1. **GPU Device & Component Trait** (`gpu_device.rs`, `gpu_component.rs`)
   - Verify auto-detect logic returns errors (never falls back)
   - Check feature guard compilation

2. **Fused Kernels** (`richards_glu_fused_kernel.rs`)
   - Two-pass strategy correct
   - GPU kernel dispatch proper error handling
   - Intermediate buffer pool management

3. **Shared Component Wiring** (`attention_context.rs`, `feedforward.rs`, `temporal_processing.rs`)
   - GPU paths called before CPU fallback (if any)
   - GpuComponent trait fully implemented
   - Buffer lifecycle management

4. **PolyAttention** (`poly_attention.rs`)
   - Fused GPU pipeline implemented
   - Streaming workspace compatible with GPU

---

## 7. Build & Test Commands

```bash
# Build with all GPU backends
cargo build --release --features gpu-all

# Test GPU components
cargo test --lib gpu_device
cargo test --lib gpu_component
cargo test --test gpu_shared_components_phase56

# Benchmark GPU vs CPU
cargo bench --bench richards_glu_fused

# Verify no-fallback mode
cargo test --lib --doc -- --nocapture | grep -i "no gpu"

# CPU-only build verification
cargo build --release
cargo test --lib
```

---

## 8. Known Issues & Resolutions

### Critical Fixes Applied (Phase 5.6)
- ✅ **AttentionContext GPU import**: Fixed `ndarray::linalg::general_mat_mul` missing import
- ✅ **Conditional GPU imports**: Applied `#[cfg(...)]` guards to prevent CPU-only build failures
- ✅ **Pre-existing kernels**: Discovered UnifiedGpuBackend already has functional implementations

### Current Focus Areas
- RichardsGLU fused kernel completion and integration
- Verification of strict no-fallback semantics across all components
- Performance benchmarking and optimization

---

## 9. Next Steps (Immediate)

1. **Run current GPU test suite**
   ```bash
   cargo test --test gpu_shared_components_phase56
   ```

2. **Audit RichardsGLU fused kernel**
   - Verify two-pass strategy in `richards_glu_fused_kernel.rs`
   - Check GPU kernel dispatch

3. **Verify SharedComponent implementations**
   - Confirm all `GpuComponent` trait methods present
   - Test `enable_gpu_auto_detect()` errors on no GPU

4. **Performance baseline**
   - Run benchmarks to establish CPU reference times
   - Profile GPU operations for bottleneck identification

---

## Document Index
- [Phase 5.6 GPU Consolidation](./PHASE5.6_GPU_CONSOLIDATION_FINAL_PLAN.md)
- [Richards GLU Fused Kernel Guide](./PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md)
- [GPU Component Trait](./src/domain/compute/gpu_component.rs)
- [Unified GPU Executor](./src/domain/compute/unified_gpu_executor.rs)
- [Shared Components Module](./src/domain/layers/components/mod.rs)
