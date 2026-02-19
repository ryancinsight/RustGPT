# Phase 5.2: GPU Backend Consolidation & Optimization

**Status**: Active  
**Date Started**: Feb 13, 2026  
**Focus**: GPU backend implementation with automatic detection, no-fallback policy, and shared component consolidation

---

## Objectives

1. **GPU-First Architecture**: Implement CUDA, Metal, and wgpu backends with automatic detection
2. **No Fallback Policy**: `require_cpu_implemented()` panics on missing GPU kernels
3. **Shared Component Optimization**: Consolidate diffusion/ssm/transformer for GPU efficiency
4. **Memory Efficiency**: Power-of-2 buffer sizing, workspace pooling, minimal allocations

---

## Phase Breakdown

### Phase 5.2.1: CUDA Backend Implementation
**Target**: GPU-accelerated kernels for core operations

**Components to GPU**:
1. `SharedAttentionContext` - GEMM-based attention
2. `SharedFeedforward` - RichardsGLU / MoE operations
3. `RichardsNorm` - Layer normalization
4. `Mamba/RgLru` - SSM state transitions

**CUDA Requirements**:
- `cuBLAS` for GEMM (0-copy via output buffer reuse)
- Element-wise kernels (ReLU, GELU, SiLU, scaled add)
- Normalization kernels (layer_norm, softmax)
- Fused operations where applicable

**Milestones**:
- [ ] Create `CudaMemoryPool` + test allocate/deallocate
- [ ] Create `CudaMatrixOps` GEMM implementation
- [ ] Implement ReLU, GELU, SiLU, add_scaled
- [ ] Implement layer_norm, softmax
- [ ] Integrate into shared components

---

### Phase 5.2.2: Metal Backend Implementation (macOS)
**Target**: Apple GPU acceleration via Metal Performance Shaders

**Metal Requirements**:
- Metal compute shaders for matrix ops
- Metal kernels for element-wise and normalization ops
- Texture-based memory management

**Milestones**:
- [ ] Create `MetalMemoryPool` with MTLBuffer backing
- [ ] Create `MetalMatrixOps` with MPS wrappers
- [ ] Implement compute kernels (GEMM via MPS, custom kernels)
- [ ] Test on macOS systems

---

### Phase 5.2.3: WebGPU Backend (Cross-Platform)
**Target**: Cross-platform GPU via wgpu (fallback for Linux/Windows without CUDA)

**wgpu Requirements**:
- `wgpu` compute shaders for all operations
- Portable shader compilation (WGSL)

**Milestones**:
- [ ] Create `WgpuMemoryPool` with wgpu buffers
- [ ] Create `WgpuMatrixOps` with compute shader dispatches
- [ ] Implement WGSL shaders for core ops
- [ ] Test on Linux/Windows/macOS

---

### Phase 5.2.4: Shared Component GPU Integration
**Target**: Consolidate diffusion/ssm/transformer with GPU variants

**Components**:
- `UnifiedLayerWorkspace` - GPU buffer layout optimization
- `SharedAttentionContext` - GPU-accelerated attention (no intermediate allocations)
- `SharedFeedforward` - GPU FFN with fused activations
- `SharedTemporalProcessing` - GPU SSM/Mamba state management
- `SharedFilmModulation` - GPU time conditioning

**Integration Pattern**:
1. Check backend at layer creation time
2. Call backend-specific kernel dispatch
3. Use `require_cpu_implemented()` if GPU backend path missing
4. Track workspace stats for profiling

**Milestones**:
- [ ] Update `TransformerBlock` with GPU variants
- [ ] Update `DiffusionBlock` with GPU variants
- [ ] Update SSM layers (Mamba, RG-LRU) with GPU variants
- [ ] Profile memory usage and allocation counts

---

### Phase 5.2.5: Performance Profiling & Optimization
**Target**: Identify and eliminate bottlenecks

**Profiling Tools**:
- CUDA: `nsys`, `ncu` (NVIDIA tools)
- Metal: Xcode Instruments
- wgpu: GPU debug markers

**Optimization Targets**:
- Reduce PCIe transfers (keep data on GPU)
- Fuse operations (layer_norm + residual + activation)
- Reduce kernel launch overhead via batch operations
- Mixed precision (f32 activations, f16 history)

**Milestones**:
- [ ] Profile baseline performance
- [ ] Identify top 3 bottlenecks
- [ ] Implement fusion optimizations
- [ ] Target 50+ GB/s memory bandwidth utilization

---

## Implementation Status

### Completed
- [x] GPU backend infrastructure (`GpuDevice`, `GpuMemoryPool`, `GpuMatrixOps` traits)
- [x] Automatic GPU detection with priority: CUDA > Metal > wgpu
- [x] No-fallback mode via `require_cpu_implemented()`
- [x] CPU memory pool for testing/non-GPU builds
- [x] Feature flags: `gpu-cuda`, `gpu-metal`, `gpu-wgpu`, `gpu-all`
- [x] ComputeBackend preference resolution

### In Progress
- [ ] CUDA backend implementation
- [ ] Shared component GPU integration

### Pending
- [ ] Metal backend implementation
- [ ] wgpu backend implementation
- [ ] Comprehensive profiling suite
- [ ] Optimization passes

---

## Testing Strategy

### Unit Tests
- Memory allocation/deallocation tracking
- Operation correctness vs CPU reference
- Numerical stability (target: ε ≤ 1e-4)

### Integration Tests
- Full forward pass with GPU backend
- Gradient computation with mixed precision
- Multi-layer transformers / diffusion blocks

### Benchmarks
- GEMM throughput (TFLOPS)
- Memory bandwidth utilization
- End-to-end inference latency

---

## Key Files

- `src/domain/compute_backend.rs` - Backend detection & resolution
- `src/domain/compute/gpu_device.rs` - GPU device abstraction
- `src/domain/compute/gpu_memory.rs` - Memory management traits
- `src/domain/compute/gpu_ops.rs` - Operation traits
- `src/domain/compute/cuda/` - CUDA backend (to be created)
- `src/domain/compute/metal/` - Metal backend (to be created)
- `src/domain/compute/wgpu/` - wgpu backend (to be created)

---

## Next Immediate Steps

1. Build with `--features gpu-cuda` to verify feature gate
2. Create `src/domain/compute/cuda/mod.rs` with `CudaMemoryPool`
3. Implement basic GEMM via cuBLAS
4. Test on system with CUDA installed
5. Integrate `SharedAttentionContext` with GPU GEMM
6. Profile allocation patterns
