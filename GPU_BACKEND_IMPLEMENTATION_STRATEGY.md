# GPU Backend Implementation Strategy - Phase 5.2a Complete

**Status**: ✅ Foundation Complete → Ready for Backend-Specific Implementations  
**Date**: Feb 13, 2026  
**Progress**: GPU compute abstractions in place, unit tests passing

---

## What Was Built (Phase 5.2a)

### 1. GPU Memory Abstraction (`src/domain/compute/gpu_memory.rs`)
- **GpuBuffer**: Opaque handle to device-resident memory (id, size_bytes)
- **GpuMemoryPool trait**: Unified allocation/deallocation interface
- **MemoryStats**: Tracks allocation count, usage, utilization percentage
- **CpuMemoryPool**: Reference implementation for testing
- **Tests**: ✅ 3 tests passing (allocate, deallocate, capacity suggestions)

```rust
pub trait GpuMemoryPool: Send + Sync {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer>;
    fn deallocate(&mut self, buffer: GpuBuffer);
    fn memory_stats(&self) -> MemoryStats;
}
```

### 2. GPU Matrix Operations (`src/domain/compute/gpu_ops.rs`)
- **GpuMatrixOps trait**: 20+ GPU-accelerated operations
- **Core BLAS**: GEMM, GEMV (matrix multiply)
- **Element-Wise**: ReLU, GELU, SiLU, scaled add, multiply
- **Normalization**: LayerNorm, Softmax
- **Reductions**: Sum, Mean
- **Data Transfer**: Upload, Download, Copy within device
- **CpuMatrixOps**: Stub implementation that fails with clear error messages

```rust
pub trait GpuMatrixOps: Send + Sync {
    fn gemm_f32(&mut self, alpha, a, b, beta, output, m, n, k) -> Result<()>;
    fn relu(&mut self, input, output, size) -> Result<()>;
    fn softmax(&mut self, input, output, rows, cols) -> Result<()>;
    // ... 17 more methods
}
```

### 3. GPU Device Context (`src/domain/compute/gpu_device.rs`)
- **GpuDevice**: Unified device abstraction
- **Backend selection**: Dispatches to CUDA/Metal/Vulkan
- **Memory management**: Allocate, deallocate, clear, stats
- **Operation dispatch**: Router for all matrix ops
- **Device info**: Name, backend type, format_info()

```rust
pub struct GpuDevice {
    backend: ComputeBackend,
    memory: Box<dyn GpuMemoryPool>,
    ops: Box<dyn GpuMatrixOps>,
    name: String,
}
```

### 4. Module Integration
- Added `pub mod compute` to `src/domain/mod.rs`
- Created `src/domain/compute/mod.rs` with re-exports
- All code follows AGENTS.md conventions (error handling, documentation)

---

## Test Results

```
running 5 tests (GPU-specific)
test domain::compute::gpu_memory::tests::memory_stats_utilization ... ok
test domain::compute::gpu_memory::tests::suggest_capacity_power_of_two ... ok
test domain::compute::gpu_memory::tests::cpu_pool_allocate_deallocate ... ok
test domain::compute::gpu_device::tests::gpu_device_format_info ... ok
test domain::compute::gpu_device::tests::gpu_device_memory_tracking ... ok

Total: 511 tests passing ✅
```

---

## Current Limitations & Next Steps

### What's Implemented
- ✅ Abstract interfaces (trait definitions)
- ✅ CPU reference implementations (for testing)
- ✅ Memory tracking and capacity management
- ✅ Operation routing infrastructure

### What's Missing (For Each Backend)
The current code will **panic with clear messages** when GPU backend is selected:

```rust
// Example error when CUDA backend used without CUDA impl
Err(ModelError::Backend {
    message: "Backend 'cuda' selected for 'GEMM', but this path does not have GPU kernels yet. \
              No CPU fallback is allowed in strict backend mode."
})
```

---

## Implementation Roadmap: Backends

### Phase 5.2b-1: CUDA Backend (Highest Priority)

**Dependencies to add**:
```toml
[dependencies]
cudarc = "0.12"      # CUDA runtime bindings
# Optional: cublas = "0.2" for cuBLAS integration
```

**Implementation scope**:
```rust
// src/domain/compute/cuda/mod.rs
pub mod cuda_memory;    // CudaMemoryPool: cudaMalloc, cudaFree
pub mod cuda_ops;       // CudaMatrixOps: cuBLAS kernels

impl GpuMemoryPool for CudaMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        // Call cudarc to allocate on default CUDA device
        // Return CudaDeviceBuffer wrapper
    }
}

impl GpuMatrixOps for CudaMatrixOps {
    fn gemm_f32(&mut self, alpha, a, b, beta, output, m, n, k) {
        // Use cuBLAS cublasGemmEx (or Sgemm for f32)
        // Handle async execution with synchronization
    }
}
```

**Key considerations**:
- Device selection (multi-GPU support)
- Stream management (async operations)
- Error handling (CUDA error codes → Result)
- Synchronization points (ensure operations complete before CPU access)

### Phase 5.2b-2: Metal Backend (Apple-Specific)

**Dependencies**:
```toml
[dependencies]
metal = "0.28"  # Metal API bindings
metal-rs = "0.3" # Higher-level wrapper
```

**Implementation**:
```rust
// src/domain/compute/metal/mod.rs
pub mod metal_memory;   // MetalMemoryPool: MTLDevice alloc
pub mod metal_ops;      // MetalMatrixOps: MetalPerformanceShaders

impl GpuMemoryPool for MetalMemoryPool {
    fn allocate(&mut self, size_bytes: usize) -> Result<GpuBuffer> {
        // Call MTLDevice::newBuffer()
        // Return MetalBuffer wrapper
    }
}
```

**Key considerations**:
- Shared memory pool (Metal has unified memory)
- Command buffer management
- Shader compilation (or link to MPS)
- Device selection (for multi-GPU Macs)

### Phase 5.2b-3: Vulkan Backend (Linux/Cross-Platform)

**Dependencies**:
```toml
[dependencies]
vulkano = "0.35" # High-level Vulkan wrapper (recommended)
# OR ash = "0.39" for low-level Vulkan
```

**Implementation**:
```rust
// src/domain/compute/vulkan/mod.rs
pub mod vulkan_memory;  // VulkanMemoryPool: GPU memory management
pub mod vulkan_ops;     // VulkanMatrixOps: compute shaders
```

**Key considerations**:
- Descriptor set management
- Compute shader compilation from GLSL/SPIR-V
- Command buffer recording and submission
- Synchronization (semaphores, fences)

---

## Priority Implementation Order

### 1. **CUDA** (Days 1-3 of Phase 5.2b)
   - Highest impact: Most users have NVIDIA GPUs
   - cuBLAS is mature and fast
   - Smallest implementation effort

### 2. **FALLBACK PATH** (Days 2-3, parallel)
   - For components without GPU variants yet
   - Allow CPU path when GPU backend not selected
   - Modify `require_cpu_implemented()` to allow graceful degradation during development

### 3. **Metal** (Days 4-5)
   - Apple ecosystem coverage
   - Simpler than Vulkan (unified memory)

### 4. **Vulkan** (Post-Phase)
   - Linux/Windows cross-platform
   - Lower priority initially

---

## Integration with Shared Components

Once a backend is implemented, integrate like this:

### SharedTemporalProcessing (Example)

```rust
impl SharedTemporalProcessing {
    pub fn forward_gpu(
        &mut self, 
        device: &mut GpuDevice, 
        x: &GpuBuffer
    ) -> Result<GpuBuffer> {
        match self.temporal_mixing {
            TemporalMixingLayer::Attention(ref mut attn) => {
                // GPU-accelerated attention
                attn.forward_gpu(device, x)
            }
            TemporalMixingLayer::RgLru(ref mut rg) => {
                // GPU-accelerated RG-LRU with parallel scan
                rg.forward_gpu(device, x)
            }
            // ... other variants
        }
    }
}
```

### Block Integration

```rust
impl TransformerBlock {
    pub fn forward_gpu(
        &mut self, 
        device: &mut GpuDevice, 
        x: &GpuBuffer
    ) -> Result<GpuBuffer> {
        // Pre-norm
        let norm1 = self.pre_attn_norm.forward_gpu(device, x)?;
        
        // Temporal mix (Attention/SSM)
        let mix = self.temporal_mixing.forward_gpu(device, &norm1)?;
        
        // Add with residual
        let residual1 = device.add_scaled(1.0, x, &mut mix.clone(), ...)?;
        
        // Continue with FFN, etc.
        // ...
    }
}
```

---

## Testing Strategy

### Unit Tests (Per Backend)
```rust
#[cfg(feature = "gpu-cuda")]
mod cuda_tests {
    #[test]
    fn cuda_gemm_matches_ndarray() {
        // Compare CUDA GEMM vs ndarray reference
        // Tolerance: ε ≤ 1e-4 for f32
    }
    
    #[test]
    fn cuda_memory_pool_lifecycle() {
        // Test alloc → use → dealloc
    }
}
```

### Integration Tests
```rust
#[test]
fn transformer_block_gpu_vs_cpu() {
    let x_cpu = /* ... */;
    let x_gpu = device.upload_to_gpu(&x_cpu)?;
    
    let cpu_out = block.forward(&x_cpu);
    let gpu_out = block.forward_gpu(&mut device, &x_gpu)?;
    let gpu_out_cpu = device.download_from_gpu(&gpu_out)?;
    
    assert_abs_diff_eq!(cpu_out, gpu_out_cpu, epsilon = 1e-4);
}
```

### Benchmark Suite
```bash
cargo bench --bench transformer_block_gpu -- cuda metal cpu
# Expected: CUDA 3-10× faster depending on sequence length
```

---

## Feature Flags (Recommended Additions to Cargo.toml)

```toml
[features]
default = ["cpu"]
cpu = []
gpu-cuda = ["cudarc", "cublas"]
gpu-metal = ["metal"]
gpu-vulkan = ["vulkano"]
gpu-all = ["gpu-cuda", "gpu-metal", "gpu-vulkan"]
```

**Build variants**:
```bash
cargo build --release --features gpu-cuda        # CUDA only
cargo build --release --features gpu-all          # All backends
cargo build --release                             # CPU only (default)
```

---

## Success Metrics for Phase 5.2b

- [ ] CUDA backend implemented (GEMM, softmax, activations)
- [ ] Metal backend implemented (or documented for future)
- [ ] `TransformerBlock::forward_gpu()` working with GPU tensors
- [ ] `DiffusionBlock::forward_with_timestep_gpu()` working
- [ ] GPU output matches CPU reference (ε ≤ 1e-4)
- [ ] Benchmark shows 3-10× speedup vs CPU
- [ ] All tests passing (511 + new GPU tests)
- [ ] Zero clippy warnings

---

## Known Challenges & Mitigation

| Challenge | Mitigation |
|-----------|-----------|
| **Async execution complexity** | Use synchronous cuBLAS for phase 1, stream APIs in phase 2 |
| **Memory fragmentation** | Implement buddy allocator or memory pool consolidation |
| **Numerical differences** | Accept ε ≤ 1e-4 tolerance (inherent in GPU float ops) |
| **Build system complexity** | Use feature flags, conditional compilation |
| **Multi-GPU support** | Start single-GPU, add multi-GPU context in phase 2 |
| **Debugging GPU code** | Use `compute-sanitizer` (CUDA) or `MetalCPUValidation` (Metal) |

---

## Next Session Checklist

- [ ] Choose backend to implement first (CUDA recommended)
- [ ] Add dependencies to Cargo.toml
- [ ] Create backend-specific module structure
- [ ] Implement GEMM operation (highest ROI)
- [ ] Write unit tests comparing to ndarray reference
- [ ] Create simple benchmark (e.g., matrix multiply)
- [ ] Document any build system changes

---

## References

- **Thread**: T-019c571e-9d07-753d-ac9a-1b5c34fc1949 (Original consolidation context)
- **Abstract interfaces**: `src/domain/compute/{gpu_memory,gpu_ops,gpu_device}.rs`
- **Strict mode**: `src/domain/compute_backend.rs` (AutoGpu + no fallback)
- **Component manifests**: `CONSOLIDATION_COMPONENTS_MANIFEST.md`
- **Build guide**: `AGENTS.md`

---

## Summary

**Completed**: Foundation is solid. Abstract interfaces (traits) are defined, tested, and ready for backend-specific implementations. Code follows Rust best practices and error handling conventions.

**Ready for**: CUDA/Metal/Vulkan kernel implementations in Phase 5.2b.

**Quick Start for Next Session**:
1. Add cudarc to Cargo.toml (or your chosen backend)
2. Implement `CudaMemoryPool` struct
3. Implement `CudaMatrixOps` struct with GEMM kernel
4. Add tests comparing to ndarray reference
5. Integrate into `SharedTemporalProcessing::forward_gpu()`
