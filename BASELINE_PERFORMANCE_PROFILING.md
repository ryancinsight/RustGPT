# GPU Performance Baseline Profiling - Phase A

## Objective
Establish baseline metrics to guide optimization decisions.

## Approach: Hot Path Analysis

Instead of long training runs, we'll analyze where time is spent:

### Hot Path Candidates (Based on Code Analysis)

1. **RichardsGlu Forward Pass** (Line 151-189)
   ```rust
   - Input upload: pool.upload(input_slice)
   - Output allocation: pool.allocate(output_size)
   - Kernel execution: forward_gpu_kernel()
   - Output download: pool.download()
   ```

2. **PolyAttention Forward Pass** (Line 1615-1695)
   ```rust
   - Multiple weight uploads (Q, K, V, O weights)
   - Large matrix multiplications (attention kernel)
   - Output download
   ```

3. **GEMM Kernels** (GPU compute intensive)
   ```rust
   - Matrix multiply operations
   - Largest numerical workload
   - Potential for kernel fusion
   ```

## Metrics to Collect

### Memory Transfers (Potential Bottleneck 1)
```
For each forward pass:
- Input upload size (MB)
- Weight upload size (MB)  
- Output download size (MB)
- Total bandwidth (MB/s)
```

### Kernel Execution (Potential Bottleneck 2)
```
Per kernel:
- Launch time
- Execution time
- Data dependencies
- Occupancy
```

### Memory Allocation (Potential Bottleneck 3)
```
- Allocation count per iteration
- Pool fragmentation
- Reuse efficiency
```

## Profiling Implementation Strategy

### Step 1: Add Timing Instrumentation

File: `src/domain/richardson/richards_glu.rs` (Line 151+)

```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let start = std::time::Instant::now();
    
    // ... existing code ...
    
    // Cache input for backward pass
    self.cached_input = Some(input.clone());
    
    let device_arc = self.gpu_device.as_ref()...?;
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();

    // PROFILING: Upload time
    let upload_start = std::time::Instant::now();
    let input_slice = input.as_slice().ok_or_else(|| ...)?;
    let input_buf = pool.upload(input_slice)?;
    let upload_ms = upload_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  GPU Upload: {:.3}ms", upload_ms);

    // PROFILING: Kernel execution time
    let kernel_start = std::time::Instant::now();
    let batch_size = input.nrows();
    let embedding_dim = self.w_out.ncols();
    let output_size = batch_size * embedding_dim * 4;
    let mut output_buf = pool.allocate(output_size)?;
    self.forward_gpu_kernel(pool, ops, &input_buf, &mut output_buf, batch_size)?;
    let kernel_ms = kernel_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  GPU Kernel: {:.3}ms", kernel_ms);

    // PROFILING: Download time
    let download_start = std::time::Instant::now();
    let mut output_array = Array2::zeros((batch_size, embedding_dim));
    let output_slice = output_array.as_slice_mut().unwrap();
    pool.download(&output_buf, output_slice)?;
    let download_ms = download_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  GPU Download: {:.3}ms", download_ms);

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("RichardsGlu::forward_gpu Total: {:.3}ms", total_ms);

    Ok(output_array)
}
```

### Step 2: Add Memory Tracking

File: `src/domain/compute/unified_gpu_buffer_pool.rs`

```rust
pub fn upload(&mut self, data: &[f32]) -> Result<GpuBuffer> {
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
    eprintln!("  → Uploading {:.2}MB", size_mb);
    
    // existing upload logic
    self.allocate(size_bytes)
}

pub fn download(&mut self, buffer: &GpuBuffer, data: &mut [f32]) -> Result<()> {
    let size_bytes = data.len() * std::mem::size_of::<f32>();
    let size_mb = size_bytes as f64 / (1024.0 * 1024.0);
    eprintln!("  ← Downloading {:.2}MB", size_mb);
    
    // existing download logic
}
```

### Step 3: Profile Kernel Count

Measure how many kernels are launched per iteration:
- QKV projection kernels
- Attention kernels
- Output projection kernels
- GLU kernels
- Element-wise kernels

## Quick Win: Remove Unused ModelError Imports

Let me clean those warnings first (should be 0 impact but reduces noise):
