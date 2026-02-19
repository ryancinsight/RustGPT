# CUDA Kernel Implementation Guide

**Quick Reference for Filling GPU Stub Implementations**

---

## Prerequisites

- cudarc 0.12 (already in Cargo.toml)
- CUDA 12.0+ with cuBLAS, CUDA Runtime
- nvcc compiler (for custom kernels if not using cuBLAS)

---

## 1. CudaMatrixOps::gemm_f32 - Matrix Multiply

**Current Stub**: `src/domain/compute/cuda/ops.rs:31`

```rust
fn gemm_f32(
    &mut self,
    alpha: f32,
    a: &GpuBuffer,      // m × k matrix
    b: &GpuBuffer,      // k × n matrix
    beta: f32,
    output: &mut GpuBuffer, // m × n matrix
    m: usize, n: usize, k: usize,
) -> Result<()> {
```

### Implementation Pattern (cuBLAS)

```rust
fn gemm_f32(
    &mut self,
    alpha: f32,
    a: &GpuBuffer,
    b: &GpuBuffer,
    beta: f32,
    output: &mut GpuBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    // Get device pointers from GpuBuffer IDs
    // NOTE: GpuBuffer only stores (id, size_bytes) - need to map to actual pointers
    //       This requires modifying CudaMemoryPool to expose a method like:
    //       pub fn get_slice(&self, id: u64) -> Option<&CudaSlice<f32>>
    
    // Option A: Use CudaDevice directly if cuBLAS is available
    // Option B: Use cudarc's built-in operations (if available)
    // Option C: Implement custom CUDA kernel
    
    // Example (requires CudaMemoryPool modification):
    /*
    let a_slice = self.memory.get_slice(a.id())
        .ok_or_else(|| ModelError::Backend {
            message: "Buffer A not found".to_string(),
        })?;
    let b_slice = self.memory.get_slice(b.id())?;
    let out_slice = self.memory.get_slice_mut(output.id())?;
    
    // cuBLAS call (if available via cudarc)
    cublas_handle.sgemm(..., alpha, a_slice, b_slice, beta, out_slice)
        .map_err(|e| ModelError::Backend {
            message: format!("GEMM failed: {}", e),
        })
    */
    
    Ok(())
}
```

### Key Points

1. **Buffer Mapping Issue**: GpuBuffer doesn't store actual device pointers, only IDs
   - **Solution**: Add method to CudaMemoryPool to retrieve CudaSlice by ID
   - Store device pointers in HashMap<u64, CudaSlice<f32>>

2. **cuBLAS Integration**: Check if cudarc provides cuBLAS bindings
   - If yes: Use directly
   - If no: Implement custom kernel or use alternative (e.g., `nalgebra-gpu`)

3. **Error Handling**: Map cudarc errors to `ModelError::Backend`

---

## 2. Element-Wise Operations (ReLU, GELU, SiLU)

**Stubs**: `ops.rs:111-165`

### Custom CUDA Kernel Pattern

```cuda
// kernel.cu
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

__global__ void gelu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximation: gelu(x) ≈ 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
        float cdf = 0.5f * (1.0f + tanhf(sqrtf(2.0f / 3.14159265f) * 
                    (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

__global__ void silu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * (1.0f / (1.0f + expf(-x)));  // x * sigmoid(x)
    }
}
```

### Rust Integration

```rust
fn relu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, size: usize) -> Result<()> {
    // 1. Get device pointers from memory pool
    let in_ptr = self.get_device_ptr(input.id())?;
    let out_ptr = self.get_device_ptr_mut(output.id())?;
    
    // 2. Calculate grid/block dimensions
    let threads_per_block = 256;
    let blocks = (size + threads_per_block - 1) / threads_per_block;
    
    // 3. Launch kernel
    unsafe {
        relu_kernel<<<blocks, threads_per_block>>>(
            in_ptr as *const f32,
            out_ptr as *mut f32,
            size as u32
        );
    }
    
    // 4. Synchronize and check for errors
    self.device.synchronize()?;
    Ok(())
}
```

---

## 3. Normalization Operations (Layer Norm, Softmax)

### Layer Norm Pattern

```rust
fn layer_norm(
    &mut self,
    input: &GpuBuffer,
    gamma: &GpuBuffer,
    beta: &GpuBuffer,
    output: &mut GpuBuffer,
    batch_size: usize,
    feature_size: usize,
    eps: f32,
) -> Result<()> {
    // Algorithm:
    // For each batch element i:
    //   1. Compute mean: mean = sum(input[i, :]) / feature_size
    //   2. Compute variance: var = sum((input[i, :] - mean)^2) / feature_size
    //   3. Normalize: norm[i, :] = (input[i, :] - mean) / sqrt(var + eps)
    //   4. Affine: output[i, :] = gamma[j] * norm[i, j] + beta[j]
    
    // This requires parallel reduction for efficiency
    // Consider:
    // - One kernel per batch element with block-level reduction
    // - Or use cuDNN if available (cudnnLayerNorm)
    
    Ok(())
}
```

### Softmax Pattern (Log-Sum-Exp Trick)

```rust
fn softmax(
    &mut self,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    // Algorithm (numerically stable):
    // For each row i:
    //   1. Find max: max_i = max(input[i, :])
    //   2. Compute: exp_vals = exp(input[i, :] - max_i)
    //   3. Sum: sum_exp = sum(exp_vals)
    //   4. Output: output[i, :] = exp_vals / sum_exp
    
    // Requires two kernels:
    // 1. Reduction kernel to find max and sum per row
    // 2. Division kernel to normalize
    
    Ok(())
}
```

---

## 4. Data Transfer Operations

### Upload (CPU → GPU)

```rust
fn upload(&mut self, cpu_data: &[f32], gpu_buffer: &mut GpuBuffer) -> Result<()> {
    let in_ptr = self.get_device_ptr_mut(gpu_buffer.id())?;
    
    // Use cudarc's memcpy_htod
    self.device.htod_sync_copy(cpu_data, in_ptr)
        .map_err(|e| ModelError::Backend {
            message: format!("CUDA upload failed: {}", e),
        })
}
```

### Download (GPU → CPU)

```rust
fn download(&self, gpu_buffer: &GpuBuffer, cpu_data: &mut [f32]) -> Result<()> {
    let out_ptr = self.get_device_ptr(gpu_buffer.id())?;
    
    // Use cudarc's memcpy_dtoh
    self.device.dtoh_sync_copy(out_ptr, cpu_data)
        .map_err(|e| ModelError::Backend {
            message: format!("CUDA download failed: {}", e),
        })
}
```

---

## 5. Reduction Operations (Sum, Mean)

```rust
fn sum(&self, buffer: &GpuBuffer, size: usize) -> Result<f32> {
    // Algorithm:
    // 1. Launch reduction kernel to compute partial sums in parallel
    // 2. Copy result back to CPU
    // 3. Return sum
    
    // Example with thrust (if available):
    // let result = thrust::reduce(ptr, ptr + size, 0.0f, thrust::plus<float>());
    
    Ok(0.0) // Placeholder
}
```

---

## 6. Code Organization

### Step-by-Step Implementation Plan

1. **Phase 1**: Fix GpuBuffer ↔ Device Pointer Mapping
   - Modify CudaMemoryPool to expose `get_slice(id)` method
   - Update CudaMatrixOps to take &mut CudaMemoryPool reference

2. **Phase 2**: Implement GEMM
   - Try cuBLAS via cudarc first
   - Fall back to custom kernel if needed
   - Test against ndarray GEMM reference

3. **Phase 3**: Implement Element-Wise Ops
   - Start with ReLU (simplest)
   - Add GELU, SiLU
   - Use same kernel pattern for all

4. **Phase 4**: Implement Normalization
   - Layer norm (use cuDNN if available)
   - Softmax (numerically stable)

5. **Phase 5**: Implement Data Transfer & Reductions
   - Upload/download (straightforward)
   - Sum/mean (reduction patterns)

6. **Phase 6**: Integration & Testing
   - Plug CudaMatrixOps into SharedAttentionContext
   - Test forward pass end-to-end
   - Benchmark vs CPU

---

## Testing Template

```rust
#[test]
#[cfg(feature = "gpu-cuda")]
fn test_cuda_gemm_correctness() {
    let mut pool = CudaMemoryPool::new(0).unwrap();
    let mut ops = CudaMatrixOps::new(&mut pool);
    
    // Create CPU reference data
    let a_cpu = Array2::<f32>::random((64, 128));
    let b_cpu = Array2::<f32>::random((128, 64));
    
    // Compute on GPU
    // ...copy to GPU...
    // ...call gemm...
    // ...copy back from GPU...
    
    // Compute on CPU
    let expected = a_cpu.dot(&b_cpu);
    
    // Compare (allow ε ≤ 1e-4)
    assert_abs_diff_eq!(gpu_result, expected, epsilon = 1e-4);
}
```

---

## Performance Profiling

Use NVIDIA tools:

```bash
# Profile GEMM throughput (TFLOPS)
nsys profile -o profile.nsys ./target/release/main

# Detailed metrics
ncu --set=full --export=report.ncu-sqlite ./target/release/main

# Memory bandwidth analysis
nvprof --metrics all ./target/release/main
```

### Target Metrics
- **GEMM**: 50-100+ TFLOPS
- **Memory Bandwidth**: 300+ GB/s
- **Kernel Launch Overhead**: <1μs
- **Data Transfer**: 200+ GB/s (PCIe 4.0)

---

## Common Pitfalls

1. **Forgetting device synchronization** - Always sync before reading results
2. **Memory leaks** - Ensure all GPU allocations are freed in CudaMemoryPool
3. **Dimension mismatches** - Validate m, n, k before launching kernels
4. **Numeric instability** - Use log-sum-exp for softmax, subtract mean for norm
5. **Insufficient shared memory** - Consider shared memory limits for reduction ops

---

## Next Priority

Start with **CUDA GEMM (cuBLAS)** - it's the highest-impact kernel for transformer models.
