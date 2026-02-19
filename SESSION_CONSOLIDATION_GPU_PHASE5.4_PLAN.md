# GPU Backend Consolidation & Optimization - Phase 5.4
**Date**: February 14, 2026  
**Status**: Planning & Execution  
**Focus**: GPU Backend Variants, Automatic Detection with Strict No-Fallback

---

## Executive Summary

Building on Phase 5.3's unified GPU architecture (95% complete), Phase 5.4 focuses on:
1. **Consolidating duplicate GPU managers** (`SharedComponentGpuManager` + `GpuSharedOpsContext` → `UnifiedGpuBufferPool`)
2. **Implementing GPU variants** for Diffusion, SSM, and Transformer blocks
3. **Memory efficiency optimizations** in kernel pipeline integration
4. **Strict no-fallback verification** with automatic GPU detection

---

## Consolidation Opportunities

### Priority 1: Merge Duplicate GPU Managers (P1)

**Current State**:
- `SharedComponentGpuManager` (shared_gpu_manager.rs) - Basic capacity tracking
- `GpuSharedOpsContext` (gpu_shared_ops.rs) - Buffer pooling with pre-allocation
- **Problem**: Duplicate interfaces, inconsistent API, unnecessary abstraction layers

**Target**:
- Merge into single `UnifiedGpuBufferPool` in `unified_gpu_buffer_pool.rs`
- Implement `WorkspaceManaged` trait for all components
- Keep single entry point: `UnifiedGpuBufferPool`

**Impact**: 
- 2-3 hours consolidation
- Eliminates 200+ lines of duplicate code
- Simplifies shared component GPU integration

### Priority 2: GPU Forward Implementations (P1)

**Missing Implementations**:
1. **DiffusionBlock GPU variant** - Currently CPU-only
2. **Mamba/RG-LRU placeholder kernels** - in `temporal_processing_gpu.rs` (lines 47-59)
3. **Complete GPU path** for TransformerBlock end-to-end

**Target**:
- Implement `forward_gpu()` for DiffusionBlock
- Replace Mamba/RG-LRU placeholders with actual WGSL recurrent scan
- Test full GPU pipeline with actual forward passes

**Impact**:
- 3-4 hours per block type
- ~15% GPU throughput improvement when fully utilized
- Enables streaming inference on GPU

### Priority 3: Memory Efficiency (P2)

**Opportunities**:
1. **In-place attention scoring** - Reuse input buffer for attention scores
2. **Kernel fusion** - Combine GEMM + Softmax for attention
3. **Buffer reuse patterns** - Implement free-list tracking in memory pool

**Target**:
- Implement in-place variants where applicable
- Add kernel fusion for common patterns
- Benchmark vs. current approach

**Impact**:
- 20-30% memory reduction for large batches
- 10-15% latency improvement from fewer memory transfers

### Priority 4: GPU Variant Trait (P2)

**Design**:
```rust
pub trait GpuVariant: Sized {
    fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>;
    fn is_gpu_ready(&self) -> bool;
    fn gpu_backend(&self) -> Option<&str>;
}
```

**Implementation**:
- Add to all shared components
- Blanket impl for components without GPU support (returns error)
- Auto-detection with fallback to CPU (explicit)

---

## Files to Modify

### 1. Consolidation Layer (New)
- **Merge**: `unified_gpu_buffer_pool.rs` (finalize implementation)
- **Remove**: `shared_gpu_manager.rs`, `gpu_shared_ops.rs` (deprecate)
- **Update**: All component imports

### 2. GPU Forward Implementations
- `src/domain/blocks/diffusion_block_gpu.rs` (new)
- `src/domain/temporal/temporal_processing_gpu.rs` (complete Mamba/RG-LRU)
- `src/domain/blocks/transformer_block_gpu.rs` (verify full path)

### 3. Kernel Pipeline Integration
- `src/domain/compute/wgpu_ops.rs` - Verify shader binding
- `src/domain/compute/unified_gpu_executor.rs` - Add profiling hooks

### 4. Testing & Benchmarks
- `tests/gpu_consolidation_verification.rs` (new)
- `benches/gpu_vs_cpu_throughput.rs` (new)

---

## Implementation Roadmap

### Phase 5.4.1: Consolidation (2-3 hours)
- [ ] Merge GPU managers into `UnifiedGpuBufferPool`
- [ ] Update all component imports
- [ ] Verify tests still pass (529 tests)

### Phase 5.4.2: GPU Implementations (4-5 hours)
- [ ] Implement DiffusionBlock GPU variant
- [ ] Complete Mamba recurrent scan kernel
- [ ] Complete RG-LRU recurrent kernel
- [ ] Full TransformerBlock GPU pipeline test

### Phase 5.4.3: Memory Optimization (2-3 hours)
- [ ] In-place attention variants
- [ ] Kernel fusion candidates
- [ ] Buffer pooling improvements

### Phase 5.4.4: Verification (1-2 hours)
- [ ] Strict GPU detection tests
- [ ] End-to-end GPU pipeline tests
- [ ] Performance benchmarks

---

## Success Criteria

✅ All GPU managers consolidated into single unified interface  
✅ DiffusionBlock has GPU variant with forward_gpu()  
✅ Mamba/RG-LRU placeholders replaced with actual recurrent kernels  
✅ 529+ tests passing with GPU detection strict mode  
✅ Memory usage reduced by 20-30% for batch operations  
✅ GPU throughput improvement measurable in benchmarks  

---

## Strict No-Fallback Design (Enforcement)

**Principle**: GPU operations **error explicitly** if GPU unavailable, never silently fall back.

**Enforcement Points**:
1. `GpuDevice::auto_detect()` - Returns error if no GPU (no silent CPU fallback)
2. `forward_gpu()` methods - Return `Result`, caller decides fallback
3. `UnifiedGpuBufferPool::enable_gpu()` - Must succeed or return clear error

**Testing**:
- `test_gpu_strict_no_fallback_mode` - Verify error paths are explicit
- All GPU paths should be Optional in interfaces

---

## References

- Previous work: @T-019c5d26-5186-7533-bc45-5e5cca1b0cbc
- GPU implementation status: GPU_BACKEND_IMPLEMENTATION_STATUS.md
- Shared components: src/domain/layers/components/
- Compute backend: src/domain/compute/
