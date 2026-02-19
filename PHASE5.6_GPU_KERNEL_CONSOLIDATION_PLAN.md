# Phase 5.6: GPU Kernel Consolidation and Optimization Plan

## Objective
Consolidate RichardGLU and PolyAttention GPU backends with automatic GPU detection (no fallback) and optimize shared components for memory efficiency.

## Phase Breakdown

### Phase 5.6.1: RichardsGLU GPU Kernel Consolidation (CURRENT)

**Goal**: Implement fused RichardsGLU kernel with strict GPU-only execution

**Tasks**:
1. ✓ Verify GPU infrastructure in place (GpuDevice, GpuComponent trait)
2. ✓ RichardsGlu has GPU device field and ensure_gpu_cache
3. **NEXT**: Implement GpuComponent trait for RichardsGlu
   - set_gpu_device()
   - enable_gpu_auto_detect()
   - is_gpu_ready()
   - gpu_backend_name()
   - ensure_capacity()
4. **NEXT**: Create fused RichardsGLU kernel
   - Single GPU kernel combining: GEMM(w1) + GEMM(w2) + Richards(x1) + Gate(x2) + Mul + GEMM(w_out)
   - Reduces global memory roundtrips from 5 to 1
   - Target: 50ms CPU → 2ms GPU (25x speedup)
5. **NEXT**: Implement GPU backward pass for RichardsGLU
   - Gradient computation on GPU
   - Weight update operations

**Deliverables**:
- [ ] RichardsGlu implements GpuComponent
- [ ] Fused kernel for RichardsGLU forward pass
- [ ] GPU backward pass (gradient computation)
- [ ] Performance benchmarks
- [ ] Numerical accuracy validation (ε ≤ 1e-4)

---

### Phase 5.6.2: Shared Components GPU Integration (DEPENDENT)

**Goal**: Implement GpuComponent for all shared layer components

**Tasks**:
1. SharedFeedforward: Implement GpuComponent
   - Delegate to underlying FeedForwardVariant (RichardsGlu, MoE)
   - Capacity management for batch/embed_dim
2. SharedTemporalProcessing: Implement GpuComponent
   - Support PolyAttention, Mamba, RG-LRU GPU paths
   - Auto-detect and attach GPU device
3. SharedAttentionContext: Implement GpuComponent
   - GPU-accelerated similarity computation
   - Context matrix operations on GPU

**Deliverables**:
- [ ] GpuComponent impl for SharedFeedforward
- [ ] GpuComponent impl for SharedTemporalProcessing
- [ ] GpuComponent impl for SharedAttentionContext
- [ ] Integration tests for each component

---

### Phase 5.6.3: PolyAttention GPU Kernel (FOLLOW-UP)

**Goal**: Implement PolyAttention GPU kernels with polynomial basis computation

**Tasks**:
1. Implement polynomial basis computation kernel
2. Implement attention scoring with polynomial gates
3. Implement learnable degree adaptation
4. GPU backward pass for gradient computation

**Target**: 30ms CPU → 1ms GPU (30x speedup)

**Deliverables**:
- [ ] PolyAttention GPU kernel
- [ ] GPU backward pass
- [ ] Degree adaptation on GPU
- [ ] Performance benchmarks

---

### Phase 5.6.4: Shared Component Memory Optimization

**Goal**: Optimize shared buffer pool and memory management

**Tasks**:
1. Unify CPU/GPU buffer pool management in UnifiedGpuBufferPool
2. Implement zero-copy forward pass patterns
3. Optimize power-of-2 sizing and reuse logic
4. Profile memory usage patterns

**Deliverables**:
- [ ] Unified buffer pool for CPU/GPU
- [ ] Zero-copy forward pipeline
- [ ] Memory efficiency metrics
- [ ] Profiling results

---

## Code Patterns

### GpuComponent Implementation Template

```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for MyComponent {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
        // Propagate to sub-components if needed
    }

    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        Ok(())
    }

    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }

    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device.as_ref().and_then(|device| {
            let dev = device.lock().ok()?;
            Some(dev.backend().as_str())
        })
    }

    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()> {
        if let Some(device_arc) = &self.gpu_device {
            let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device".to_string(),
            })?;
            // Pre-allocate buffers
            let _ = device.allocate_f32(batch_size * embed_dim)?;
            // ... more allocations
        }
        Ok(())
    }
}
```

---

## Performance Targets

| Component | Operation | CPU Time | GPU Target | Gain |
|-----------|-----------|----------|-----------|------|
| RichardsGLU | 1K batch forward | ~50ms | ~2ms | 25x |
| PolyAttention | Forward pass | ~30ms | ~1ms | 30x |
| Mamba scan | SSM forward | ~40ms | ~2ms | 20x |
| AttentionContext | Similarity + softmax | ~15ms | ~0.5ms | 30x |

---

## Strict No-Fallback Design

- `enable_gpu_auto_detect()` returns `Err` if no GPU detected
- GPU operations never silently fall back to CPU
- Clear error messages indicating exact issue and solution
- Auto-detection priority: CUDA > Metal > Vulkan
- Feature-flag validation ensures binary matches hardware

---

## Testing Strategy

1. **Compilation**: `cargo check --lib`
2. **Unit tests**: Each component GpuComponent impl
3. **Integration**: Shared component GPU forward pass
4. **Numerical accuracy**: Compare GPU vs CPU output (ε ≤ 1e-4)
5. **Performance**: Benchmark GPU vs CPU execution time
6. **Memory**: Track allocation/reuse efficiency

---

## Status

- [x] GPU infrastructure established (GpuDevice, GpuComponent trait)
- [x] RichardsGlu has GPU fields and basic methods
- [ ] **CURRENT**: Implement RichardsGlu GpuComponent
- [ ] Implement fused RichardsGLU kernel
- [ ] Shared component GpuComponent impls
- [ ] PolyAttention GPU kernels
- [ ] Full integration testing
