# Phase 4: GPU Kernel Implementation Roadmap
**Date**: February 15, 2026  
**Status**: Ready for execution  
**Focus**: Implement fused GPU kernels for shared components

---

## Overview

GPU dispatch is now active across all 3 shared components:
- **SharedAttentionContext** ✅ Dispatch active → needs kernel
- **SharedFeedforward** ✅ Dispatch active → uses variant kernels
- **SharedTemporalProcessing** ✅ Dispatch active → uses variant kernels

Phase 4 focuses on implementing the actual GPU kernels that the dispatch code calls.

---

## Kernel Implementation Priority

### Priority 1: Attention Context Kernel (Lowest Hanging Fruit)

**Kernel**: `UnifiedGpuBackend::forward_attention_context()`  
**File**: `src/domain/layers/components/unified_gpu_kernels.rs`

**Operation**:
```
output = input @ context_strength * strength + input
```

**Current Status**: Exists but incomplete (needs backend implementation)

**Complexity**: ⭐ (Matrix multiplication + element-wise addition)

**Implementation Steps**:

1. **Matrix Multiplication Pass**
   ```rust
   pub fn forward_attention_context(
       &mut self,
       input: &Array2<f32>,         // (batch, embed_dim)
       context: &Array2<f32>,       // (embed_dim, embed_dim)
       strength: f32,               // scalar modifier
   ) -> Result<Array2<f32>> {
       // 1. Upload input to GPU
       let batch_size = input.nrows();
       let embed_dim = input.ncols();
       
       // 2. Allocate output buffer on GPU
       let output_size = batch_size * embed_dim;
       
       // 3. Dispatch kernel based on backend
       match &self.backend_type {
           BackendType::Cuda => self.cuda_matrix_mul(...),
           BackendType::Wgpu => self.wgpu_matrix_mul(...),
           BackendType::Metal => self.metal_matrix_mul(...),
           BackendType::Vulkan => self.vulkan_matrix_mul(...),
       }
   }
   ```

2. **CUDA Implementation** (if CUDA enabled)
   - Use cuBLAS for matrix multiplication
   - Use custom kernel for element-wise operations
   - Memory: input + context + output + intermediate

3. **WGPU Implementation** (cross-platform)
   - Use GPU compute shader for matrix multiply
   - Tile-based optimization (64x64 tile size)
   - Shared memory for efficiency

4. **Metal Implementation** (macOS)
   - Use Metal Performance Shaders (MPS)
   - MetalAI Matrix library for acceleration

**Expected Performance**:
- CPU: 15ms (1K batch, 64D)
- GPU: 0.5ms (CUDA/Metal) - **30x speedup**

---

### Priority 2: RichardsGLU Fused Kernel

**Kernel**: `UnifiedGpuKernels::richards_glu_fused()`  
**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Operation** (5 launches → 2 passes):
```
Pass 1 (Fused):
  x1 = input @ W1                          [launch 1 → 1]
  x1 = richards_curve.forward(x1)          [merged into launch 1]
  x2 = input @ W2                          [launch 1 → 2]
  gated = x1 * sigmoid(x2)                 [merged into launch 2]

Pass 2:
  output = gated @ W_out + input           [launch 3]
```

**Current Status**: Variant calls own GPU forward (not fused yet)

**Complexity**: ⭐⭐ (Multiple operations fused)

**Implementation Steps**:

1. **Kernel Stub Location**
   ```rust
   #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
   pub fn richards_glu_fused(
       input: &Array2<f32>,
       w1: &Array2<f32>,
       w2: &Array2<f32>,
       w_out: &Array2<f32>,
       backend: &mut UnifiedGpuBackend,
   ) -> Result<Array2<f32>>
   ```

2. **CUDA Kernel**
   ```cuda
   __global__ void richards_glu_kernel(
       float* input,           // [batch, embed_dim]
       float* w1, float* w2,   // [embed_dim, hidden_dim]
       float* x1_out,          // [batch, hidden_dim]
       int batch, int embed_dim, int hidden_dim
   ) {
       // Tile: [block_size, block_size]
       // 1. Compute x1 = input @ W1 (with Richards activation)
       // 2. Compute x2 = input @ W2
       // 3. Compute gated = x1 * sigmoid(x2)
       // Write to shared memory then global
   }
   ```

3. **Buffer Management**
   - Pre-allocate x1, x2, gated buffers in pool
   - Reuse across multiple forward passes
   - No CPU transfers until final output

**Expected Performance**:
- CPU: 50ms (1K batch, 4K→16K)
- GPU: 2ms (fused) - **25x speedup**
- Non-fused GPU: 8ms - **6x speedup**

---

### Priority 3: PolyAttention Fused Kernel

**Kernel**: `UnifiedGpuKernels::poly_attention_fused()`  
**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Operation** (3+ launches → 1 pass):
```
Single Fused Kernel:
  Q = input @ W_Q
  K = input @ W_K
  V = input @ W_V
  scores = (Q @ K.T) / sqrt(embed_dim)
  attn_weights = softmax(scores)
  output = attn_weights @ V
  output = output @ W_O
```

**Current Status**: Uses standard attention kernels

**Complexity**: ⭐⭐⭐ (Multi-head, batching)

**Implementation Steps**:

1. **Kernel Signature**
   ```rust
   pub fn poly_attention_fused(
       input: &Array2<f32>,         // [batch*seq, embed_dim]
       w_q: &Array2<f32>,           // [embed_dim, num_heads*head_dim]
       w_k: &Array2<f32>,
       w_v: &Array2<f32>,
       w_o: &Array2<f32>,           // [num_heads*head_dim, embed_dim]
       num_heads: usize,
       backend: &mut UnifiedGpuBackend,
   ) -> Result<Array2<f32>>
   ```

2. **Tiled Computation**
   - Tile Q/K/V projections: [128, 128] tile
   - Block-level softmax: max/exp/sum/norm per head
   - Output projection: [64, 128] tile

3. **Shared Memory Optimization**
   - Load Q/K into shared memory (96KB per block)
   - Compute scores in registers
   - Reduce softmax computation

**Expected Performance**:
- CPU: 30ms (512 batch, 64D, 8 heads)
- GPU: 1ms (fused) - **30x speedup**

---

### Priority 4: Mamba Scan Kernel

**Kernel**: `UnifiedGpuKernels::mamba_scan_kernel()`  
**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Operation**: Vectorized selective state space scan
```
State: h[t] = A[t] * h[t-1] + B[t] * x[t]
Output: y[t] = C[t] * h[t]
```

**Current Status**: Uses sequential variant kernels

**Complexity**: ⭐⭐⭐⭐ (Recurrent, selective masking)

**Implementation Steps**:

1. **Scan Decomposition**
   ```rust
   pub fn mamba_scan_kernel(
       input: &Array2<f32>,         // [seq_len, state_dim]
       a: &Array2<f32>,             // A matrices
       b: &Array2<f32>,             // B matrices
       c: &Array2<f32>,             // C matrices
       backend: &mut UnifiedGpuBackend,
   ) -> Result<Array2<f32>>
   ```

2. **Parallel Scan Algorithm**
   - Split sequence into chunks: [512 elements]
   - Compute prefix products locally
   - Reduce across chunks (parallel prefix)
   - Propagate state forward

3. **Selective Masking**
   - Apply delta modulation per step
   - Zero out inactive paths (sparse computation)
   - Memory-efficient state tracking

**Expected Performance**:
- CPU: 40ms (512 seq, 64D)
- GPU: 2ms (vectorized) - **20x speedup**

---

## Implementation Architecture

### Backend Abstraction Layer

```rust
// In unified_gpu_kernels.rs

pub enum BackendType {
    #[cfg(feature = "gpu-cuda")]
    Cuda(CudaContext),
    #[cfg(feature = "gpu-wgpu")]
    Wgpu(WgpuDevice),
    #[cfg(feature = "gpu-metal")]
    Metal(MetalDevice),
    #[cfg(feature = "gpu-vulkan")]
    Vulkan(VulkanDevice),
}

impl UnifiedGpuKernels {
    pub fn matrix_multiply(
        &mut self,
        a: &GpuBuffer,           // (m, k)
        b: &GpuBuffer,           // (k, n)
        c: &mut GpuBuffer,       // (m, n)
        alpha: f32,
    ) -> Result<()> {
        match &self.backend {
            BackendType::Cuda(ctx) => ctx.cublas_matmul(a, b, c, alpha),
            BackendType::Wgpu(dev) => dev.compute_shader_matmul(a, b, c, alpha),
            BackendType::Metal(dev) => dev.mps_matmul(a, b, c, alpha),
            BackendType::Vulkan(dev) => dev.compute_matmul(a, b, c, alpha),
        }
    }
}
```

### Buffer Lifecycle Management

```rust
// In unified_buffer_pool.rs

pub struct UnifiedBufferPool {
    buffers: HashMap<usize, Vec<GpuBuffer>>,  // size -> [buffers]
    max_lifetime: Duration,
}

impl UnifiedBufferPool {
    pub fn allocate(&mut self, size: usize) -> Result<GpuBuffer> {
        let power_of_2 = size.next_power_of_two();
        // Reuse from pool if available
        if let Some(mut vec) = self.buffers.get_mut(&power_of_2) {
            if let Some(buf) = vec.pop() {
                return Ok(buf);
            }
        }
        // Allocate new buffer
        let buf = self.backend.allocate(power_of_2)?;
        Ok(buf)
    }

    pub fn recycle(&mut self, buffer: GpuBuffer) {
        let size = buffer.capacity();
        self.buffers.entry(size).or_insert_with(Vec::new).push(buffer);
    }
}
```

---

## Integration with Dispatch Layer

### Attention Context Flow

```
apply_context()
├── Check GPU ready
├── Auto-detect backend
├── Call apply_incoming_context_gpu()
│   └── UnifiedGpuBackend::forward_attention_context()
│       ├── Buffer pool: allocate(batch * embed_dim)
│       ├── Upload input, context, strength
│       ├── Kernel dispatch:
│       │   ├── CUDA: matrix_multiply kernel
│       │   ├── WGPU: compute shader
│       │   └── Metal: Metal Performance Shaders
│       ├── Download result
│       └── Buffer pool: recycle()
└── Return result
```

### Feedforward Flow

```
forward()
├── Check compute_backend.is_gpu()
├── Call forward_gpu()
│   └── FeedForwardVariant::forward_gpu()
│       ├── RichardsGLU:
│       │   └── GPU kernels (via trait method)
│       └── MoE:
│           └── GPU expert routing + compute
└── Return result
```

---

## Validation & Testing Strategy

### Correctness Tests

**AttentionContext Kernel**:
```rust
#[test]
fn test_attention_context_gpu_vs_cpu() {
    let input = Array2::random((128, 64));
    let context = Array2::random((64, 64));
    
    // CPU path
    let cpu_result = context_cpu(&input, &context);
    
    // GPU path
    let mut backend = UnifiedGpuBackend::auto_detect()?;
    let gpu_result = backend.forward_attention_context(&input, &context, 1.0)?;
    
    // Compare (tolerance: 1e-4)
    assert_close(&cpu_result, &gpu_result, 1e-4);
}
```

**Performance Benchmarks**:
```bash
cargo bench --bench gpu_kernels_phase56 -- --features gpu-wgpu
cargo bench --bench gpu_kernels_phase56 -- --features gpu-cuda
```

### Coverage Targets
- ✅ All kernel entry points
- ✅ Edge cases (empty input, size-1, very large)
- ✅ Numerical stability (accumulation errors)
- ✅ Memory alignment (buffer pooling)

---

## Development Schedule

### Week 1: AttentionContext Kernel
- Implement WGPU compute shader (cross-platform)
- Add CUDA cuBLAS wrapper
- Add Metal MPS wrapper
- Tests + benchmarks

### Week 2: RichardsGLU Fused Kernel
- Implement fused computation
- Benchmark launch reduction (5→2)
- Optimize register usage

### Week 3: PolyAttention Fused Kernel
- Implement multi-head attention fusion
- Shared memory optimization
- Head dimension tiling

### Week 4: Mamba Scan Kernel
- Implement parallel prefix scan
- Selective masking
- State propagation optimization

---

## Success Criteria

| Kernel | CPU Time | GPU Target | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- |
| AttentionContext | 15ms | 0.5ms | 30x | ⏳ Ready |
| RichardsGLU | 50ms | 2ms | 25x | ⏳ Ready |
| PolyAttention | 30ms | 1ms | 30x | ⏳ Ready |
| Mamba Scan | 40ms | 2ms | 20x | ⏳ Ready |

---

## Files to Create/Modify

### New Files
1. `src/domain/layers/components/gpu_kernels_attention_context.rs`
2. `src/domain/layers/components/gpu_kernels_richardsglu.rs`
3. `src/domain/layers/components/gpu_kernels_poly_attention.rs`
4. `src/domain/layers/components/gpu_kernels_mamba.rs`
5. `tests/gpu_kernels_phase56_verification.rs`
6. `benches/gpu_kernels_phase56.rs`

### Modified Files
1. `src/domain/layers/components/unified_gpu_kernels.rs` (add dispatchers)
2. `src/domain/layers/components/fused_kernels_module.rs` (remove stubs, add implementations)
3. `src/domain/layers/components/mod.rs` (export new modules)

---

## Risk Mitigation

| Risk | Mitigation |
| :--- | :--- |
| Kernel doesn't match CPU numerics | Tolerance tests (1e-4), error bounds |
| GPU memory exhaustion | Buffer pool with size limits, early allocation checks |
| Backend differences (CUDA vs WGPU) | Implement all backends, test with all |
| Performance doesn't meet target | Profile with NSight (CUDA) / Xcode (Metal) / GPU validation layers |

---

**Status**: Phase 4 ready to begin. All dispatch infrastructure in place. Next: Implement AttentionContext kernel.
