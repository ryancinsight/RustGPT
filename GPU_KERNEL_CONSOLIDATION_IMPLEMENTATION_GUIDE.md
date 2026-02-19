# GPU Kernel Consolidation & Implementation Guide - Phase 5.6.3

**Date**: Feb 16, 2026  
**Status**: IN PROGRESS  
**Focus**: Consolidate shared components (Diffusion, SSM, Transformer) with optimized GPU kernels

## Quick Start

### 1. GPU Auto-Detection (Already Implemented)
```rust
// Automatic detection with priority: CUDA > Metal > Vulkan
// NO CPU fallback - will error if no GPU available
let kernels = UnifiedGpuKernels::auto_detect()?;

// Or with specific backend
let kernels = UnifiedGpuKernels::new(ComputeBackend::Cuda)?;
```

### 2. Memory Management Pattern
```rust
// Workspace is pre-allocated with power-of-2 sizing
kernels.ensure_capacity(batch_size, embed_dim, seq_len)?;

// Use workspace buffers for all operations
let output = kernels.activation_forward(&input, activation)?;

// Reset for reuse (keeps buffers allocated)
kernels.reset_workspace();

// Cleanup when done
kernels.cleanup_workspace()?;
```

### 3. Building New GPU Kernels

#### Step 1: Define Parameters
```rust
#[derive(Debug, Clone)]
pub struct MyKernelParams {
    pub batch_size: usize,
    pub embed_dim: usize,
    pub seq_len: usize,
    pub temperature: f32,
    // ... other parameters
}
```

#### Step 2: Implement CPU Reference (for validation)
```rust
pub fn forward_reference_cpu(
    input: &Array2<f32>,
    params: &MyKernelParams,
) -> Result<Array2<f32>> {
    // CPU implementation for testing/debugging
    // This serves as the ground truth for GPU validation
    
    let mut output = input.clone();
    for i in 0..params.batch_size {
        for j in 0..params.embed_dim {
            // Apply operation
            output[[i, j]] = /* computation */;
        }
    }
    Ok(output)
}
```

#### Step 3: Implement GPU Dispatch
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    params: &MyKernelParams,
) -> Result<GpuBuffer> {
    let (batch_size, embed_dim) = (params.batch_size, params.embed_dim);
    let total_size = batch_size * embed_dim;
    
    // Allocate output buffer
    let output_size = total_size * std::mem::size_of::<f32>();
    let mut output = device.allocate(output_size)?;
    
    // Call backend-specific kernel
    // Backend will automatically dispatch to CUDA/Metal/WGPU
    device.my_kernel_op(input, &mut output, params)?;
    
    Ok(output)
}
```

#### Step 4: Integrate into UnifiedGpuKernels
```rust
impl UnifiedGpuKernels {
    pub fn my_kernel_forward(
        &mut self,
        input: &Array2<f32>,
        params: &MyKernelParams,
    ) -> Result<Array2<f32>> {
        let mut device = self.device.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire GPU device lock".to_string(),
        })?;
        
        self.ensure_capacity(params.batch_size, params.embed_dim, params.seq_len)?;
        
        // Upload input
        let input_size = params.batch_size * params.embed_dim * std::mem::size_of::<f32>();
        let mut input_buf = device.allocate(input_size)?;
        device.upload(input.as_slice().unwrap(), &mut input_buf)?;
        
        // Execute GPU kernel
        let output_buf = forward_gpu(&mut device, &input_buf, params)?;
        
        // Download result
        let (batch_size, embed_dim) = input.dim();
        let total_size = batch_size * embed_dim;
        let mut output = vec![0.0f32; total_size];
        device.download(&output_buf, &mut output)?;
        
        // Cleanup
        device.deallocate(input_buf);
        device.deallocate(output_buf);
        
        Ok(Array2::from_shape_vec((batch_size, embed_dim), output)?)
    }
}
```

## Implementation Priority

### Priority 1: Core Kernels (Critical Path)
These kernels have the highest impact on performance:

1. **RichardsGLU Fused Kernel** ✓ (Already implemented)
   - Location: `src/domain/compute/richards_glu_fused_kernel.rs`
   - Two-pass strategy: W1→Richards→W2 (minimal global memory traffic)
   - Target: 25x speedup (50ms → 2ms on 1K batch)
   - Status: GPU dispatch implemented, needs backend-specific optimization

2. **Attention Operations** (HIGH PRIORITY)
   - File to create: `src/domain/layers/components/attention_gpu_kernel.rs`
   - Operations:
     - QKV projection: Q, K, V = input @ W_q, W_k, W_v
     - Softmax attention: softmax(Q @ K^T / √d) @ V
     - Output projection: attention @ W_o
   - Target: 30x speedup (30ms → 1ms on 512 batch)
   - Kernel size: 256-thread workgroups, thread-block reduction for softmax

3. **Selective Scan (Mamba)** (HIGH PRIORITY)
   - File to create: `src/domain/layers/components/mamba_selective_scan_gpu.rs`
   - Operation: Parallel prefix sum + recurrent state update
   - Target: 20x speedup (40ms → 2ms on 512 batch)
   - Challenge: Sequential nature requires careful GPU mapping

### Priority 2: Secondary Kernels (Good Speedup)
4. **RG-LRU Recurrent** (MEDIUM PRIORITY)
   - File to create: `src/domain/layers/components/rg_lru_gpu_kernel.rs`
   - Operation: State update with recurrent connectivity
   - Target: 15x speedup (30ms → 2ms on 512 batch)

5. **Normalization Kernels** (MEDIUM PRIORITY)
   - File to create: `src/domain/layers/components/norm_gpu_kernel.rs`
   - Operations: Layer norm, RMS norm
   - Target: 5-10x speedup

### Priority 3: Utility Kernels (Foundation)
6. **Elementwise Operations** (LOW PRIORITY - foundation)
   - Already implemented in `GpuDevice`
   - Operations: Add, multiply, activation functions
   - Essential for building fused kernels

## Current Status

### ✅ Completed
- [x] Auto GPU detection (CUDA > Metal > Vulkan) with strict no-fallback
- [x] `UnifiedGpuKernels` dispatcher with workspace management
- [x] Power-of-2 buffer sizing
- [x] Buffer naming for tracking
- [x] Memory estimation in workspace stats
- [x] RichardsGLU reference implementation and GPU dispatch skeleton
- [x] Compilation passing with warnings cleaned up

### 🔄 In Progress
- [ ] Attention kernel implementation (QKV projection, softmax, output)
- [ ] Selective scan kernel (Mamba)
- [ ] RG-LRU kernel
- [ ] Normalization kernels

### 📋 TODO
- [ ] Fused kernel combinations
- [ ] Performance profiling and optimization
- [ ] Numerical stability improvements
- [ ] Error handling and troubleshooting guides
- [ ] Comprehensive testing

## Memory Management Details

### Buffer Pool Strategy
Each operation allocates from a pre-sized pool:

```
Total Memory per Workspace (power-of-2):
├── Activation Buffers (2): 2 * batch * embed * 4 bytes
├── QKV Buffers (3): 3 * batch * embed * 4 bytes
├── Attention Scores: batch * seq * seq * 4 bytes
├── Output Buffer: batch * embed * 4 bytes
└── Weight Buffer: embed * embed * 4 bytes

Example (batch=512, embed=768, seq=512):
├── Activation: 2 * 512 * 768 * 4 = 3.1 MB
├── QKV: 3 * 512 * 768 * 4 = 4.7 MB
├── Scores: 512 * 512 * 512 * 4 = 512 MB  ← Largest!
├── Output: 512 * 768 * 4 = 1.5 MB
└── Weight: 768 * 768 * 4 = 2.3 MB
Total: ~523 MB
```

### Power-of-2 Sizing
- `batch_size = 512` → kept as 512 (already power-of-2)
- `batch_size = 500` → rounds to 512
- `embed_dim = 768` → rounds to 1024
- Benefits: GPU memory coalescing, cache-friendly alignment

### Reuse Strategy
1. **Allocate**: Pre-allocate all buffers in `ensure_capacity()`
2. **Reuse**: Call `reset_workspace()` to reuse without reallocation
3. **Track**: Monitor allocation/reallocation counts
4. **Deallocate**: Final cleanup with `cleanup_workspace()`

```
First call (allocate):
┌─────────────────────────────┐
│ ensure_capacity()           │
├─────────────────────────────┤
│ allocation_count = 1        │
│ reallocation_count = 0      │
└─────────────────────────────┘

Subsequent calls (reuse):
┌─────────────────────────────┐
│ reset_workspace() [no-op]   │
│ [forward pass uses buffers] │
├─────────────────────────────┤
│ allocation_count = 1        │ ← No change
│ reallocation_count = 0      │ ← No change
└─────────────────────────────┘

Resize if needed:
┌──────────────────────────────┐
│ ensure_capacity() resize     │
├──────────────────────────────┤
│ allocation_count = 2         │
│ reallocation_count = 1       │
└──────────────────────────────┘
```

## Error Handling Pattern

### Strict No-Fallback Design
```rust
// ❌ DO NOT implement CPU fallback
pub fn gpu_operation(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = self.device.lock().map_err(|_| ModelError::Backend {
        message: "Failed to acquire GPU device lock".to_string(),
    })?;
    
    // Execute GPU operation
    device.my_gpu_op(&input_buf, &mut output_buf)?;
    // If this fails, return error - no CPU fallback
    
    Ok(output)
}

// ✅ DO provide detailed error context
device.my_gpu_op(&input_buf, &mut output_buf)
    .map_err(|e| ModelError::Backend {
        message: format!(
            "GPU operation failed for {} backend: {}. \
             This operation requires GPU. \
             Ensure your GPU is properly configured.",
            device.backend().as_str(),
            e
        ),
    })?;
```

## Performance Profiling Guide

### Measuring Speedup
```rust
use std::time::Instant;

// CPU version
let start = Instant::now();
let cpu_output = cpu_forward(&input, &params)?;
let cpu_time = start.elapsed();

// GPU version
let start = Instant::now();
let gpu_output = kernels.gpu_forward(&input, &params)?;
let gpu_time = start.elapsed();

let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
println!("Speedup: {:.1}x ({:.1}ms CPU → {:.1}ms GPU)",
    speedup,
    cpu_time.as_secs_f64() * 1000.0,
    gpu_time.as_secs_f64() * 1000.0
);
```

### Monitoring Memory Usage
```rust
let stats = kernels.workspace_stats();
println!("Workspace Statistics:");
println!("  Capacity: batch={}, embed={}, seq={}",
    stats.capacity.0, stats.capacity.1, stats.capacity.2);
println!("  Buffers: {} allocated", stats.buffer_count);
println!("  Allocations: {} total, {} reallocations", 
    stats.allocation_count, stats.reallocation_count);
println!("  Memory: {:.1} MB", 
    stats.estimated_memory_bytes as f64 / (1024.0 * 1024.0));
```

## Testing Template

```rust
#[test]
fn test_my_kernel_gpu() {
    // Skip if no GPU available
    if !crate::domain::compute::GpuDevice::auto_detect().is_ok() {
        println!("No GPU available, skipping test");
        return;
    }
    
    let mut kernels = UnifiedGpuKernels::auto_detect().expect("GPU should be available");
    
    // Create test data
    let batch_size = 8;
    let embed_dim = 64;
    let input = Array2::<f32>::ones((batch_size, embed_dim));
    let params = MyKernelParams {
        batch_size,
        embed_dim,
        seq_len: 32,
        temperature: 1.0,
    };
    
    // Ensure workspace capacity
    kernels.ensure_capacity(batch_size, embed_dim, 32).expect("Capacity");
    
    // Test GPU forward
    let gpu_output = kernels.my_kernel_forward(&input, &params)
        .expect("GPU forward should succeed");
    
    // Test against CPU reference
    let cpu_output = forward_reference_cpu(&input, &params)
        .expect("CPU forward should succeed");
    
    // Verify outputs match (with tolerance for numerical errors)
    let max_diff = gpu_output
        .iter()
        .zip(cpu_output.iter())
        .map(|(g, c)| (g - c).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    
    assert!(max_diff < 1e-4, "GPU and CPU outputs differ by {}", max_diff);
    
    // Cleanup
    kernels.cleanup_workspace().expect("Cleanup");
}
```

## Next Steps

1. **Implement Attention Kernel**
   - Create `src/domain/layers/components/attention_gpu_kernel.rs`
   - Implement QKV projection, softmax, and output operations
   - Add tests with attention context examples

2. **Implement Selective Scan**
   - Create `src/domain/layers/components/mamba_selective_scan_gpu.rs`
   - Handle parallel prefix sum efficiently on GPU
   - Test with Mamba layer examples

3. **Performance Profiling**
   - Run microbenchmarks on actual GPU hardware
   - Profile memory usage patterns
   - Identify optimization opportunities

4. **Documentation**
   - Create architecture diagrams
   - Document performance characteristics
   - Build troubleshooting guides

