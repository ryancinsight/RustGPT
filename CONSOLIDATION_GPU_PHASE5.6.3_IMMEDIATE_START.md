# GPU Backend Consolidation & Optimization (Phase 5.6.3)
## Immediate Implementation Start
**Status**: Ready for Implementation  
**Date**: 2026-02-16  
**Duration**: This Session

---

## 1. Executive Summary

This consolidation session focuses on:
1. **Unify GPU kernel implementations** across diffusion, SSM, and transformer shared components
2. **Implement fused kernels** for Richards GLU and attention operations
3. **Optimize memory efficiency** through workspace management and power-of-2 sizing
4. **Enable automatic GPU detection** with strict no-fallback semantics for troubleshooting

### Performance Targets
- **RichardsGLU Fused**: 25x speedup (50ms → 2ms on 1K batch)
- **PolyAttention Fused**: 30x speedup (30ms → 1ms on 512 batch)  
- **Mamba Selective Scan**: 20x speedup (40ms → 2ms on 512 batch)

---

## 2. Current State Assessment

### ✅ Already Implemented
- **GpuDevice abstraction** (gpu_device.rs): Backend detection, memory management
- **GpuMatrixOps trait**: Unified interface across CUDA/Metal/Vulkan
- **WGPU implementation**: GEMM, softmax, element-wise ops
- **UnifiedGpuBackend**: High-level entry point
- **UnifiedGpuKernels**: Parameter structures and dispatcher skeleton

### ⚠️ Needs Completion (Priority Order)

#### Priority 1: Fused Kernel Implementations
- [ ] **RichardsGLU Fused Kernel** (WGPU + CUDA + Metal)
  - File: `src/domain/compute/richards_glu_fused_kernel.rs` (exists, needs finalization)
  - Combine: W1 projection → Richards activation → W2 projection → Gate → W_out
  - Two-pass kernel to minimize global memory traffic
  
- [ ] **PolyAttention Unified Kernel** (WGPU)
  - Combine: Q @ K → softmax → gate multiplication → V projection
  - Reduce kernel launches from 3+ to 1

#### Priority 2: GPU Memory Optimization
- [ ] **Workspace-based buffer allocation** in UnifiedGpuKernels
  - Pre-allocate power-of-2 sized buffers
  - Reuse across kernel calls
  - Track capacity and resize when needed

- [ ] **Temporal operation optimization**
  - Mamba selective scan kernel
  - RG-LRU recurrent kernel  
  - Streaming data on GPU (no CPU round-trips)

#### Priority 3: Automatic GPU Detection & Testing
- [ ] **Strict no-fallback mode** enforcement
  - Return `ModelError::Backend` instead of silent CPU fallback
  - Add comprehensive error messages for missing implementations

- [ ] **GPU detection integration tests**
  - Test each backend (CUDA, Metal, WGPU) independently
  - Verify no CPU fallback occurs

#### Priority 4: Performance Profiling
- [ ] **Kernel dispatch timing**
- [ ] **Memory bandwidth utilization**
- [ ] **Comparison: CPU vs GPU performance**

---

## 3. Implementation Roadmap

### Phase 3A: RichardsGLU Fused Kernel Finalization (2 hours)

**Current Status**: Skeleton exists in `src/domain/compute/richards_glu_fused_kernel.rs`

**Tasks**:
1. **Review & complete WGPU implementation**
   - Implement Pass 1: W1 @ input → Richards activation
   - Implement Pass 2: RG(output) @ W2 → gate logic → W_out
   - Add stable log-space Richards computation

2. **Add CUDA stub with `.cu` kernel template**
   - Create `src/domain/compute/kernels/richards_glu.cu`
   - Implement fused kernel using CUDA intrinsics
   - Link via cudarc

3. **Add Metal stub with `.metal` kernel template**
   - Create `src/domain/compute/kernels/richards_glu.metal`
   - Use Metal Performance Shaders primitives
   - Link via metal-rs

**Code Pattern** (WGSL):
```wgsl
// Pass 1: Activation
let x1 = input @ w1;
let x2 = input @ w2_init;
let g = richards_curve(x2, params);
let value = x1;
let gated = value * g;

// Pass 2: Projection
let output = gated @ w_out;
```

---

### Phase 3B: Workspace Memory Optimization (1.5 hours)

**Files to Update**:
- `src/domain/layers/components/unified_gpu_kernels.rs` (GpuKernelWorkspace)
- Implement proper buffer lifecycle management

**Tasks**:
1. **Implement GpuKernelWorkspace properly**
   - Track allocated buffers
   - Implement power-of-2 sizing strategy
   - Reuse buffers across calls without reallocation

2. **Add workspace finalization methods**
   - `cleanup()` - deallocate all buffers
   - `reset()` - mark all buffers as reusable
   - `memory_stats()` - track usage

3. **Integrate with kernel operations**
   - Update `attention_forward()` to use workspace
   - Update SSM operations to use workspace
   - Update normalization to use workspace

---

### Phase 3C: Temporal Operations GPU Optimization (2 hours)

**Files to Create/Update**:
- `src/domain/compute/selective_scan_kernel.rs` (new)
- `src/domain/compute/rg_lru_kernel.rs` (new)

**Tasks**:
1. **Selective Scan Kernel** (Mamba)
   - Input: u[t], dt[t], A, B[t], C[t], D
   - Output: y[t]
   - State update: h[t] = A @ h[t-1] + B[t] * u[t]
   - Target: 20x speedup on 512 batch

2. **RG-LRU Recurrent Kernel**
   - State update with Richards gate
   - Input: x[t], state (batch, embed_dim)
   - Output: y[t]
   - Target: 15x speedup on 512 batch

---

### Phase 3D: GPU Detection & Error Handling (1 hour)

**Files to Update**:
- `src/domain/compute/gpu_device.rs`
- `src/domain/layers/components/unified_gpu_backend.rs`

**Tasks**:
1. **Enforce strict no-fallback**
   - Add `ModelError::Backend` for all missing GPU ops
   - Remove any silent CPU computation paths
   - Add informative error messages

2. **Add GPU feature detection**
   - Check for CUDA compute capability
   - Check for Metal shader version
   - Check for WGPU limits
   - Report diagnostics to user

3. **Testing**
   - Unit tests for each backend
   - Integration tests with actual tensors
   - Error case handling

---

## 4. Code Organization & Structure

### Module Layout
```
src/domain/
├── compute/
│   ├── gpu_device.rs (★ Entry point for all GPU ops)
│   ├── gpu_ops.rs (★ GpuMatrixOps trait)
│   ├── gpu_memory.rs (★ GpuMemoryPool, GpuBuffer)
│   ├── wgpu_ops.rs (WGPU backend - mostly done)
│   ├── cuda/ (CUDA backend - stubs ready)
│   ├── metal/ (Metal backend - stubs ready)
│   ├── richards_glu_fused_kernel.rs (★ Priority 1)
│   ├── selective_scan_kernel.rs (Priority 3)
│   └── rg_lru_kernel.rs (Priority 3)
│
└── layers/components/
    ├── unified_gpu_kernels.rs (★ Dispatcher + workspace)
    ├── unified_gpu_backend.rs (High-level API)
    └── ...
```

### Trait Implementation Pattern

For each GPU backend, implement:
```rust
impl GpuMatrixOps for WgpuMatrixOps {
    fn richards_curve(...) -> Result<()> { /* WGSL kernel */ }
    fn moh_gate_activation(...) -> Result<()> { /* WGSL kernel */ }
    // ...
}

#[cfg(feature = "gpu-cuda")]
impl GpuMatrixOps for CudaMatrixOps {
    fn richards_curve(...) -> Result<()> { /* CUDA kernel dispatch */ }
    fn moh_gate_activation(...) -> Result<()> { /* CUDA kernel dispatch */ }
    // ...
}
```

---

## 5. Performance Optimization Strategies

### Kernel Dispatch Optimization
- **Workgroup sizes**: 256 threads for element-wise, 16×16 for matrix ops
- **Shared memory**: < 48KB for temporary data
- **Coalesced access**: Linear buffer reads/writes

### Memory Access Patterns
- **Zero-copy**: Keep data on GPU between operations
- **Power-of-2 sizing**: Align buffers to 256-byte boundaries
- **Reusable workspaces**: Pre-allocate, never deallocate

### Numerical Stability
- **Richards curve**: Use log-space formulation for exponents
- **Softmax**: Apply log-sum-exp trick
- **Normalization**: Use Welford algorithm for stability

### Kernel Fusion Examples

**RichardsGLU Before (5 kernels)**:
1. W1 projection
2. W2 projection  
3. Richards activation
4. Element-wise multiply (gate)
5. W_out projection

**RichardsGLU After (2 kernels)**:
1. **Pass 1**: W1 → Richards activation
2. **Pass 2**: Gated @ W_out

---

## 6. Testing Strategy

### Unit Tests
- Parameter initialization
- Dimension validation
- Workspace capacity management

### Integration Tests  
- End-to-end forward pass on GPU
- Compare GPU output vs CPU reference
- Numerical accuracy (ε ≤ 1e-4)

### Performance Tests
- Benchmark kernel execution time
- Track memory allocation/deallocation
- Profile bandwidth utilization

### Backend-Specific Tests
- WGPU: All platforms (Windows/Mac/Linux)
- CUDA: NVIDIA devices only
- Metal: macOS only

---

## 7. Troubleshooting with No-Fallback Mode

**Benefits of Strict No-Fallback**:
1. **Predictable performance**: Know exactly when code runs on GPU
2. **Early error detection**: GPU compilation/execution failures visible immediately
3. **Easier debugging**: No silent performance degradation to CPU
4. **Clear error messages**: Developers know exactly what's not implemented

**Typical Error Scenarios**:
```
Error: GPU operation 'richards_curve' not implemented for CUDA
→ Implement in `src/domain/compute/kernels/richards_curve.cu`

Error: No GPU detected; cannot create GpuDevice
→ Run on system with GPU or use CPU-only build

Error: WGPU backend requires Vulkan/Metal/DX12
→ Install graphics drivers or use different platform
```

---

## 8. Immediate Action Items (This Session)

### Phase 3A: RichardsGLU Fused Kernel
- [ ] Finalize WGPU implementation
- [ ] Test on sample tensors
- [ ] Create CUDA `.cu` template
- [ ] Create Metal `.metal` template

### Phase 3B: Workspace Management
- [ ] Complete GpuKernelWorkspace
- [ ] Integrate with all kernel calls
- [ ] Add capacity tracking

### Phase 3C: GPU Detection
- [ ] Ensure strict no-fallback mode
- [ ] Add comprehensive error messages
- [ ] Write detection tests

### Phase 3D: Documentation
- [ ] Update kernel dispatch guide
- [ ] Document GPU backend priority order
- [ ] Add performance tuning tips

---

## 9. Success Criteria

✅ **Session Complete When**:
1. RichardsGLU fused kernel works end-to-end on WGPU
2. Workspace properly manages buffers with power-of-2 sizing
3. All GPU operations return errors (not silent CPU fallback)
4. Unit tests pass for all implemented operations
5. Documentation updated with implementation patterns
6. CUDA/Metal kernels have stubs ready for implementation

---

## 10. Related Files & References

### Core GPU Infrastructure
- `src/domain/compute/gpu_device.rs` - Device management
- `src/domain/compute/gpu_ops.rs` - Operation trait definitions
- `src/domain/compute/gpu_memory.rs` - Buffer allocation
- `src/domain/compute/wgpu_ops.rs` - WGPU backend (mostly complete)

### Shared Components
- `src/domain/layers/components/unified_gpu_kernels.rs` - Kernel dispatcher
- `src/domain/layers/components/unified_gpu_backend.rs` - High-level API
- `src/domain/layers/components/attention_context_gpu.rs` - Attention GPU ops
- `src/domain/layers/components/feedforward_gpu.rs` - Feedforward GPU ops

### Performance Targets (from Phase 5.6.3)
- Multi-head attention: 30x speedup (30ms → 1ms on 512 batch)
- Mamba selective scan: 20x speedup (40ms → 2ms on 512 batch)
- RG-LRU recurrent: 15x speedup (30ms → 2ms on 512 batch)

---

## Next Session Handoff

When consolidation is complete, document:
1. Fused kernel dispatch implementation
2. Workspace memory lifecycle
3. GPU detection priority and error handling
4. CUDA/Metal kernel stubs ready for future implementation
5. Performance benchmarks achieved

