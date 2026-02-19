# Quick Start: Phase 5.6.4c - CUDA & Metal GEMM Implementations

**Status**: WGPU complete, ready for CUDA/Metal  
**Remaining Priority 1**: 2 backends × 4-6 hours = 8-12 hours work  

## CUDA Implementation Checklist

### 1. CudaGemmKernel Structure (src/domain/layers/components/gpu_gemm_kernels.rs:135-190)

**Current placeholder**:
```rust
#[cfg(feature = "gpu-cuda")]
mod cuda_gemm {
    pub struct CudaGemmKernel {
        _context: (),  // TODO: Add actual CUDA context
    }
}
```

**Implementation steps**:

1. **Add CUDA types**:
   ```rust
   use cudarc::driver::{CudaDevice, CudaStream};
   
   pub struct CudaGemmKernel {
       device: CudaDevice,
       stream: CudaStream,
   }
   ```

2. **Implement gemm() method**:
   - Use `cublas` for matrix multiplication
   - Call `cublasSgemm()` for f32
   - Handle transposition with `CUBLAS_OP_T`
   - Synchronize with `cudaStreamSynchronize()`

3. **Implement gemm_t() method**:
   - Set operation flag to `CUBLAS_OP_T` for A
   - Otherwise identical to gemm()

### 2. Key CUDA Integration Points

**cuBLAS call signature**:
```c
cublasStatus_t cublasSgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda,
    const float *B, int ldb,
    const float *beta,
    float *C, int ldc
);
```

**Rust wrapper**:
```rust
fn gemm(
    &mut self,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a_ptr: *const f32,
    b_ptr: *const f32,
    beta: f32,
    c_ptr: *mut f32,
) -> Result<()> {
    // 1. Create CUDA device pointers from raw pointers
    // 2. Call cublasSgemm with CUBLAS_OP_N
    // 3. Synchronize stream
    // 4. Validate error codes
}
```

### 3. Error Handling

**CUDA errors to handle**:
- `CUDA_ERROR_INVALID_DEVICE`
- `CUDA_ERROR_OUT_OF_MEMORY`
- `CUDA_ERROR_INVALID_ARGUMENT`
- `CUBLAS_STATUS_EXECUTION_FAILED`

**Map to ModelError**:
```rust
match cuda_error {
    CudaError::InvalidDevice => ModelError::Backend { ... },
    CudaError::OutOfMemory => ModelError::Backend { ... },
    _ => ModelError::Computation { ... },
}
```

### 4. Testing

Create `tests/gpu_gemm_cuda_phase56.rs`:
```rust
#[test]
#[cfg(feature = "gpu-cuda")]
fn test_cuda_gemm_basic() {
    let device = CudaDevice::new(0).unwrap();
    let mut kernel = CudaGemmKernel::new(device);
    
    // Allocate GPU memory
    // Copy data
    // Call gemm()
    // Verify results
}
```

---

## Metal Implementation Checklist

### 1. MetalGemmKernel Structure (src/domain/layers/components/gpu_gemm_kernels.rs:192-247)

**Current placeholder**:
```rust
#[cfg(feature = "gpu-metal")]
mod metal_gemm {
    pub struct MetalGemmKernel {
        _device: (),  // TODO: Add Metal device
    }
}
```

**Implementation steps**:

1. **Add Metal types**:
   ```rust
   use metal::*;
   
   pub struct MetalGemmKernel {
       device: Device,
       command_queue: CommandQueue,
   }
   ```

2. **Use Metal Performance Shaders (MPS)**:
   - `MTLMatrix` for matrix operations
   - `MPSMatrixMultiplication` for GEMM
   - Handle transposition with `MPSMatrix` layout flags

3. **Implement gemm() method**:
   ```rust
   // Create MTLBuffer objects from raw pointers
   // Create MTLMatrix descriptors
   // Create MPSMatrixMultiplication kernel
   // Encode into command buffer
   // Commit and wait
   ```

### 2. Metal GEMM Flow

```rust
fn gemm(
    &mut self,
    m: usize, n: usize, k: usize,
    alpha: f32,
    a_ptr: *const f32,
    b_ptr: *const f32,
    beta: f32,
    c_ptr: *mut f32,
) -> Result<()> {
    // 1. Create command buffer
    // 2. Create MTLBuffer from data
    // 3. Create MTLMatrix descriptors with proper row/column strides
    // 4. Create MPSMatrixMultiplication kernel
    // 5. Set kernel parameters (alpha, beta)
    // 6. Encode into command buffer
    // 7. Commit and wait for completion
    // 8. Synchronize results
}
```

### 3. Transposition in Metal

Metal uses matrix descriptors:
```rust
// For A^T multiplication, set rows/columns in descriptor
let matrix_a_descriptor = MTLMatrixDescriptor::new(
    rows: k as u64,      // Transposed
    columns: m as u64,   // Transposed
    rowBytes: stride,    // Must match layout
    dataType: .float32
);
```

### 4. Error Handling

**Metal errors**:
- `NSError` from encode operations
- Device not available
- Buffer allocation failures
- Command submission failures

**Map to ModelError**:
```rust
match metal_error {
    NSError if code == ... => ModelError::Backend { ... },
    _ => ModelError::Computation { ... },
}
```

### 5. Testing

Create `tests/gpu_gemm_metal_phase56.rs`:
```rust
#[test]
#[cfg(feature = "gpu-metal")]
fn test_metal_gemm_basic() {
    let device = Device::system_default().unwrap();
    let mut kernel = MetalGemmKernel::new(device);
    // Similar to CUDA test
}
```

---

## Implementation Order

### Session 1: CUDA (4-6 hours)
1. Add CUDA types and constructors
2. Implement `gemm()` with cuBLAS
3. Implement `gemm_t()` with transposition
4. Add error handling
5. Write and pass tests
6. Performance validation

### Session 2: Metal (4-6 hours)
1. Add Metal types and constructors
2. Implement `gemm()` with MPS
3. Implement `gemm_t()` with matrix descriptors
4. Add error handling
5. Write and pass tests
6. Performance validation

### Session 3: Benchmarking & Optimization
1. Compare WGPU vs CUDA vs Metal
2. Identify bottlenecks
3. Optimize transfer overhead
4. Batch operations
5. Update performance targets

---

## Integration Testing

**Cross-backend tests**:
```rust
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[test]
fn test_all_backends_equivalent() {
    let input = Array2::<f32>::random((32, 64));
    
    // Test on each available backend
    // Verify numerical equivalence (tolerance: 1e-4)
    // Check performance differences
}
```

---

## Performance Benchmarks to Establish

**For each backend**:
```
gemm(256, 256, 256):
  WGPU:   0.05-0.1ms  (expected)
  CUDA:   0.05-0.1ms  (target)
  Metal:  0.05-0.1ms  (target)
  CPU:    0.5-1.0ms   (baseline)

Speedup: 5-20x expected
```

---

## Key Files to Modify

1. `src/domain/layers/components/gpu_gemm_kernels.rs`
   - CUDA implementation (lines 135-190)
   - Metal implementation (lines 192-247)

2. `tests/gpu_gemm_cuda_phase56.rs` (create new)
3. `tests/gpu_gemm_metal_phase56.rs` (create new)

4. `Cargo.toml` (dependencies already present)
   - `cudarc` for CUDA
   - `metal` for Metal (likely need to add)

---

## Success Criteria

- [x] WGPU GEMM working (Phase 5.6.4c ✅)
- [ ] CUDA GEMM working & tested
- [ ] Metal GEMM working & tested
- [ ] All 3 backends produce equivalent results (tolerance: 1e-4)
- [ ] All 3 backends pass 552+ tests
- [ ] Performance targets met (15-30x speedup)
- [ ] No GPU memory leaks
- [ ] Error handling complete

---

**Ready to start?** Just follow the checklist above and you'll have all 3 backends complete in the next 2-3 sessions!
