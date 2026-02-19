# GPU Backend Consolidation & Implementation - Phase 5.2

**Status**: Planning → Implementation  
**Start Date**: Feb 13, 2026  
**Target Completion**: Feb 20, 2026  
**Focus**: Automatic GPU detection with strict no-fallback, shared component GPU variants  

---

## Executive Summary

This phase implements GPU backend variants for shared components between Diffusion, SSM, and Transformer architectures. Current state:

- ✅ Backend detection framework in place (`ComputeBackend`, `ComputeBackendPreference`)
- ✅ Strict `auto-gpu` mode that panics when no GPU available (no CPU fallback)
- ⏳ **GPU kernel implementations missing** - components currently error on GPU backend selection
- ⏳ **Device memory management** - `UnifiedLayerWorkspace` tracks backend but doesn't use it
- ⏳ **GPU-accelerated operations** - GEMM, attention, FFN need GPU variants

---

## Current State Analysis

### Backend Detection (Working ✅)
```
ComputeBackend Detection Priority: CUDA > Metal > Vulkan
├── CUDA: nvidia-smi, nvcc probing
├── Metal: system_profiler on macOS
└── Vulkan: vulkaninfo probing
```

### Components with GPU Panics (Need Implementation 🔴)
| Component | File | Issue |
|-----------|------|-------|
| SharedFeedforward::forward_into() | feedforward.rs:85-92 | Panics on GPU backend |
| UnifiedLayerWorkspace | unified_layer_workspace.rs | Tracks backend but no GPU memory management |
| SharedAttentionContext | attention_context.rs | CPU-only buffer operations |
| SharedTemporalProcessing | temporal_processing.rs | CPU-only forward paths |

### Environment Detection (Working ✅)
```rust
RUSTGPT_GPU_BACKEND env var override:
- "auto-gpu" / "auto" → AutoGpu (strict)
- "cpu" → Cpu
- "cuda" → Cuda (fail if unavailable)
- "metal" → Metal (fail if unavailable)
- "vulkan" → Vulkan (fail if unavailable)
```

---

## Implementation Strategy

### Phase 5.2a: GPU-Accelerated Primitives (Days 1-2)

**Objective**: Implement core GPU operations that will be reused across components.

#### 1. GPU Memory Pool (`src/domain/compute/gpu_memory.rs`)
```rust
pub trait GpuMemoryPool: Send + Sync {
    fn allocate(&mut self, size_bytes: usize) -> GpuBuffer;
    fn deallocate(&mut self, buffer: GpuBuffer);
    fn clear(&mut self);
    fn memory_usage(&self) -> MemoryStats;
}

// Backend-specific implementations
pub struct CudaMemoryPool { /* ... */ }
pub struct MetalMemoryPool { /* ... */ }
pub struct VulkanMemoryPool { /* ... */ }
```

**Why**: Unified interface for device-specific memory allocation across all components.

#### 2. GPU Matrix Operations (`src/domain/compute/gpu_ops.rs`)
```rust
pub trait GpuMatrixOps: Send + Sync {
    // GEMM: output = alpha*A@B + beta*output
    fn gemm_f32(&mut self, alpha: f32, a: &GpuBuffer, b: &GpuBuffer, 
                beta: f32, output: &mut GpuBuffer, m: usize, n: usize, k: usize);
    
    // Element-wise ops
    fn relu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize);
    fn gelu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize);
    
    // Normalization
    fn layer_norm(&mut self, input: &GpuBuffer, gamma: &GpuBuffer, beta: &GpuBuffer,
                  output: &mut GpuBuffer, size: usize);
}

// Backend-specific implementations
pub struct CudaMatrixOps { /* cuBLAS, cuDNN kernels */ }
pub struct MetalMatrixOps { /* Metal Performance Shaders */ }
pub struct VulkanMatrixOps { /* Vulkan compute shaders */ }
```

**Why**: Abstracts CUDA/Metal/Vulkan kernel differences, enables feature-parity across backends.

#### 3. Device-Host Transfer (`src/domain/compute/gpu_transfer.rs`)
```rust
pub trait GpuTransfer: Send + Sync {
    fn upload(&mut self, cpu_data: &[f32], gpu_buffer: &mut GpuBuffer) -> Result<()>;
    fn download(&self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()>;
    fn copy_within_device(&mut self, src: &GpuBuffer, dst: &mut GpuBuffer, size: usize);
}
```

**Why**: Minimizes CPU↔GPU transfers, enables in-place GPU operations.

---

### Phase 5.2b: GPU Variants for Shared Components (Days 2-4)

#### Component Update Pattern

For each component, implement GPU variant alongside CPU:

```rust
impl SharedTemporalProcessing {
    // Existing CPU path
    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>> { /* ... */ }
    
    // NEW: GPU path (auto-selected by compute_backend)
    pub fn forward_gpu(&self, device: &mut GpuDevice, x: &GpuBuffer) -> Result<GpuBuffer> {
        match self.compute_backend {
            ComputeBackend::Cuda => self.forward_cuda(device, x),
            ComputeBackend::Metal => self.forward_metal(device, x),
            ComputeBackend::Vulkan => self.forward_vulkan(device, x),
            ComputeBackend::Cpu => panic!("CPU path should use forward()"),
        }
    }
    
    // Router method (called by blocks)
    pub fn forward_any(&self, x: &ComputeTensor) -> Result<ComputeTensor> {
        match x {
            ComputeTensor::Cpu(arr) => self.forward(arr).map(ComputeTensor::Cpu),
            ComputeTensor::Gpu(buf) => {
                let device = self.get_gpu_device();
                self.forward_gpu(&device, buf).map(ComputeTensor::Gpu)
            }
        }
    }
}
```

#### Shared Computational Patterns for GPU Implementation

1. **Temporal Mixing (Attention/SSM)**
   - Attention: Use cuDNN for QK^T scaling, softmax, weighted sum
   - Mamba/RG-LRU: Parallel scan (BlellochScan, TitanScan) for recurrence
   - Advantage: 5-10× speedup vs CPU for large sequence lengths

2. **Feedforward with RichardsGLU**
   - GEMM for gate/value projections
   - Fused element-wise operations (RichardsGLU activation)
   - FiLM modulation (element-wise multiply/add)
   - Advantage: 3-5× speedup via kernel fusion

3. **LayerNorm with Residuals**
   - Fused normalization + residual add in single kernel
   - Optional affine parameters
   - Advantage: Reduced memory bandwidth, 2-3× speedup

4. **Attention Context Updates**
   - Sparse matrix operations for context pooling
   - Efficient similarity matrix computation via GEMM
   - Advantage: Near-peak GPU memory bandwidth utilization

---

### Phase 5.2c: Integration with Blocks (Days 4-5)

**Goal**: Use GPU variants in `TransformerBlock` and `DiffusionBlock` forward passes.

```rust
// TransformerBlock GPU forward
impl TransformerBlock {
    pub fn forward_gpu(&self, device: &mut GpuDevice, 
                       x: &GpuBuffer, context: Option<&GpuBuffer>) -> Result<GpuBuffer> {
        // Pre-attention normalization
        let norm1_out = self.pre_attention_norm.forward_gpu(device, x)?;
        
        // Temporal mixing (attention/SSM)
        let mix_out = self.temporal_mixing.forward_gpu(device, &norm1_out)?;
        
        // First residual
        let residual1 = device.add(&x, &mix_out)?;
        
        // Pre-FFN normalization
        let norm2_out = self.pre_ffn_norm.forward_gpu(device, &residual1)?;
        
        // Feedforward
        let ffn_out = self.feedforward.forward_gpu(device, &norm2_out)?;
        
        // Second residual
        device.add(&residual1, &ffn_out)
    }
}
```

---

## Recommended Dependencies

### For CUDA Support
```toml
[dependencies]
cudarc = "0.12"  # CUDA runtime bindings
cublas = "0.2"   # cuBLAS bindings
cudnn = "0.12"   # cuDNN bindings (optional, for advanced ops)
```

### For Metal Support
```toml
[dependencies]
metal = "0.28"      # Metal API bindings
metal-rs = "0.3"    # Higher-level Metal wrapper
```

### For Vulkan Support
```toml
[dependencies]
vulkan = "0.6"      # Vulkan API bindings
# Consider: gpu-alloc, gpu-descriptor for memory management
```

### For Cross-Platform Matrix Operations
```toml
[dependencies]
polars = "0.x"      # Has GPU acceleration via GPU plugin (future)
# OR use direct backend bindings above
```

---

## Workload Prioritization

### High Priority (Large Impact)
1. **GEMM Operations** - Used in feedforward, attention all-gather
2. **Softmax** - Critical for attention numerics and performance
3. **Element-wise Operations** - ReLU, GELU, add, multiply (used everywhere)

### Medium Priority (Moderate Impact)
1. **LayerNorm** - Normalization bottleneck, benefits from kernel fusion
2. **Attention Context** - Similarity pooling with sparse operations
3. **Parallel Scan** - Essential for Mamba/RG-LRU efficiency

### Lower Priority (Smaller Impact, Can Start CPU)
1. **Conditioning** - Time embeddings, FiLM parameters (small tensors)
2. **Adaptive Residuals** - Head activity tracking (post-hoc)

---

## Strict No-Fallback Enforcement

Current implementation in `ComputeBackend::require_cpu_implemented()` already enforces this:

```rust
pub fn require_cpu_implemented(self, op_name: &str) {
    if self.is_gpu() {
        panic!(
            "Backend '{}' selected for '{}', but this path does not have GPU kernels yet. \
             No CPU fallback is allowed in strict backend mode.",
            self.as_str(),
            op_name
        );
    }
}
```

**Usage in components**:
```rust
pub fn forward_gpu(&self, device: &GpuDevice, x: &GpuBuffer) -> Result<GpuBuffer> {
    self.compute_backend.require_cpu_implemented("SharedTemporalProcessing::forward_gpu");
    // Kernel implementation follows
}
```

This ensures:
1. ✅ GPU backend selected → GPU kernels must exist or panic
2. ✅ No silent CPU fallback (no silent performance cliffs)
3. ✅ Clear error messages for incomplete implementations

---

## Testing & Validation

### Unit Tests Per Component
```rust
#[test]
fn gpu_temporal_mixing_cuda_output_matches_cpu() {
    let x_cpu = /* ... */;
    let x_gpu = upload_to_gpu(&x_cpu);
    
    let cpu_out = temporal_mixing.forward(&x_cpu).unwrap();
    let gpu_out = temporal_mixing.forward_gpu(&device, &x_gpu).unwrap();
    let gpu_out_cpu = download_from_gpu(&gpu_out);
    
    assert_abs_diff_eq!(cpu_out, gpu_out_cpu, epsilon = 1e-4);
}
```

### Benchmark Comparisons
```bash
cargo bench --bench temporal_mixing_gpu -- cuda metal cpu
# Expected: GPU 3-10× faster depending on sequence length
```

### Memory Profiling
```rust
// Track allocations before/after GPU transfer
let mem_before = device.memory_usage();
device.allocate(tensor_bytes);
let mem_after = device.memory_usage();
println!("GPU alloc: {} MB", (mem_after.used - mem_before.used) / 1024 / 1024);
```

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|-----------|
| CUDA/Metal APIs complex | High | Start with GEMM (most mature), use high-level bindings |
| Device-specific bugs | High | Extensive unit tests with CPU fallback in tests |
| Memory fragmentation | Medium | Use memory pools with buddy allocator |
| Numerical precision | Medium | Validate GPU output vs CPU reference (ε = 1e-4 for f32) |
| Build system complexity | Medium | Feature flags: `cuda`, `metal`, `vulkan` for optional compilation |

---

## Feature Flags (Recommended)

```toml
[features]
default = ["cpu"]
cpu = []
gpu-cuda = ["cudarc", "cublas"]
gpu-metal = ["metal", "metal-rs"]
gpu-vulkan = ["vulkan"]
gpu-all = ["gpu-cuda", "gpu-metal", "gpu-vulkan"]
```

**Build variants**:
```bash
cargo build --release                    # CPU only
cargo build --release --features gpu-all # All GPU backends
cargo build --release --features gpu-cuda # CUDA only (smaller binary)
```

---

## Implementation Roadmap

### Week 1 (Feb 13-19)
- [ ] **Day 1-2**: GPU primitives (memory pools, matrix ops)
- [ ] **Day 3**: Temporal mixing GPU variants
- [ ] **Day 4**: Feedforward GPU variants
- [ ] **Day 5**: Attention context GPU variant

### Week 2 (Feb 20)
- [ ] **Day 1-2**: Block integration (Transformer + Diffusion)
- [ ] **Day 3**: End-to-end testing (CPU ↔ GPU)
- [ ] **Day 4-5**: Performance profiling and documentation

---

## Success Criteria

- ✅ GPU detection works with `auto-gpu` strict mode
- ✅ No CPU fallback when GPU backend selected
- ✅ GPU variants implemented for ≥5 core components
- ✅ Unit tests: CPU vs GPU output matches (ε ≤ 1e-4)
- ✅ Benchmark: GPU 3-10× faster than CPU for typical workloads
- ✅ Memory tracking: GPU memory usage < 2× model parameters
- ✅ Zero clippy warnings, all tests passing

---

## Next Steps

1. **Validate dependencies available**: Check if CUDA/Metal libraries available on target platform
2. **Create GPU memory pool abstraction**: `src/domain/compute/gpu_memory.rs`
3. **Implement CUDA GEMM wrapper**: Use `cudarc` for proof-of-concept
4. **Add GPU path to SharedTemporalProcessing**: Attention first (highest ROI)
5. **Create end-to-end test**: Verify forward pass works GPU→CPU

---

## References

- **Backend Detection**: `src/domain/compute_backend.rs`
- **Component Manifest**: `CONSOLIDATION_COMPONENTS_MANIFEST.md`
- **Previous Thread**: T-019c571e-9d07-753d-ac9a-1b5c34fc1949
- **GPU Acceleration Concepts**: WebGPU spec, CUDA Best Practices, Metal Performance Shaders docs
