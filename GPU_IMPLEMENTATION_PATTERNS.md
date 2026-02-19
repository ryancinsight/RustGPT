# GPU Implementation Patterns (Phase 5.6)
## Quick Reference for Shared Component GPU Kernels

---

## Pattern 1: Standard GPU Forward Pass

### Template for Component::forward_gpu()

```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // 1. Get GPU device
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU device not set".to_string(),
        })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // 2. Upload input
    let input_slice = input.as_slice()
        .ok_or_else(|| ModelError::InvalidInput {
            message: "Input must be contiguous".to_string(),
        })?;
    let input_buf = pool.upload(input_slice)?;
    
    // 3. Allocate output
    let (batch_size, embed_dim) = input.dim();
    let output_size = batch_size * embed_dim * std::mem::size_of::<f32>();
    let mut output_buf = pool.allocate(output_size)?;
    
    // 4. Execute kernel
    self.forward_gpu_kernel(pool, ops, &input_buf, &mut output_buf, batch_size)?;
    
    // 5. Download result
    let mut output_array = Array2::zeros((batch_size, embed_dim));
    let output_slice = output_array.as_slice_mut().unwrap();
    pool.download(&output_buf, output_slice)?;
    
    Ok(output_array)
}
```

### Kernel Pattern (Low-Level)

```rust
pub fn forward_gpu_kernel(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    batch_size: usize,
) -> Result<()> {
    // Allocate intermediate buffers (power-of-2 sizing for efficiency)
    let mut buf1 = pool.allocate(batch_size * hidden_dim * 4)?; // f32 = 4 bytes
    let mut buf2 = pool.allocate(batch_size * hidden_dim * 4)?;
    
    // Execute operations (kept on GPU)
    ops.gemm_f32(pool, 1.0, input, &weights1, 0.0, &mut buf1, ...)?;
    ops.some_activation(pool, &buf1, ...)?;
    ops.gemm_f32(pool, 1.0, &buf1, &weights2, 0.0, output, ...)?;
    
    Ok(())
}
```

---

## Pattern 2: GEMM Operation (Standard)

### For A @ B = C computation

```rust
ops.gemm_f32(
    pool,
    1.0,                    // alpha
    input_A,                // A: (M x K)
    weights_B,              // B: (K x N)
    0.0,                    // beta
    output_C,               // C: (M x N)
    m,                      // rows of A
    n,                      // cols of B
    k,                      // cols of A / rows of B
    false,                  // transpose_a
    false,                  // transpose_b
)?;
```

### For A @ B^T = C computation (e.g., attention scores)

```rust
ops.gemm_f32(
    pool,
    1.0,
    query,                  // (batch, seq_len, d_k)
    key,                    // (batch, seq_len, d_k)
    0.0,
    scores,                 // (batch, seq_len, seq_len)
    batch * seq_len,        // M
    seq_len,                // N
    d_k,                    // K
    false,
    true,                   // transpose_b = true for key
)?;
```

---

## Pattern 3: Fused Operations (Optimization)

### Example: GEMM + Activation Fusion

**CPU Implementation** (3 kernels):
```rust
// 1. GEMM: z = input @ weights
gemm(input, weights, &mut z);
// 2. Activation: a = activation(z)
activation(&z, &mut a);
// 3. Multiply: output = input * a
mul(&input, &a, &mut output);
```

**GPU Implementation** (1 fused kernel):
```rust
ops.gemm_f32(pool, 1.0, input, weights, 0.0, &mut z, ...)?;
ops.fused_activation_mul(pool, input, &z, &mut output, ...)?;
// OR: Manually fuse into single WGPU shader
```

**Benefit**: Reduces GPU memory roundtrips by 2× (z and a don't leave GPU)

---

## Pattern 4: Numerical Accuracy Validation

### CPU Reference vs GPU Comparison

```rust
#[test]
fn gpu_vs_cpu_numerical_accuracy() {
    // Create test component with GPU device
    let mut component = Component::new(...);
    
    // Set GPU device
    let device = GpuDevice::auto_detect().expect("GPU required");
    component.set_gpu_device(Arc::new(Mutex::new(device)));
    
    // Create test input
    let input = Array2::from_shape_fn((batch_size, embed_dim), |(i, j)| {
        ((i * embed_dim + j) as f32 * 0.1).sin()
    });
    
    // CPU forward
    let cpu_result = component.forward(&input);
    
    // GPU forward
    let gpu_result = component.forward_gpu(&input).expect("GPU forward failed");
    
    // Validate: max element-wise difference ≤ 1e-4
    let max_diff = gpu_result.iter()
        .zip(cpu_result.iter())
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, |a, b| a.max(b));
    
    assert!(max_diff <= 1e-4, "Max diff: {}", max_diff);
}
```

---

## Pattern 5: Power-of-2 Buffer Sizing

### Alignment for GPU Efficiency

```rust
// Never allocate exact sizes; round up to power of 2
fn allocate_gpu_buffer(pool: &mut dyn GpuMemoryPool, requested_size: usize) -> Result<GpuBuffer> {
    // For 1001 elements: allocate 1024 (next power of 2)
    let allocated_size = 1 << (requested_size.next_power_of_two().trailing_zeros());
    pool.allocate(allocated_size * std::mem::size_of::<f32>())
}

// Track waste for monitoring
let wasted_bytes = (allocated_size - requested_size) * std::mem::size_of::<f32>();
stats.total_wasted_padding += wasted_bytes;
```

---

## Pattern 6: Batch Size Variation Tests

### Testing with Different Sizes

```rust
#[test]
fn gpu_kernel_batch_size_robustness() {
    let embed_dim = 768;
    
    for batch_size in &[1, 16, 32, 64, 128, 256] {
        let input = create_test_input(*batch_size, embed_dim);
        
        // CPU forward
        let cpu_result = component.forward(&input);
        
        // GPU forward (should work with any batch size)
        let gpu_result = component.forward_gpu(&input)
            .expect(&format!("GPU forward failed for batch_size={}", batch_size));
        
        // Validate accuracy
        let max_diff = max_element_diff(&gpu_result, &cpu_result);
        assert!(max_diff <= 1e-4, "batch_size={}: max_diff={:.2e}", batch_size, max_diff);
    }
}
```

---

## Pattern 7: GPU Device Attachment

### Automatic Initialization

```rust
impl Component {
    pub fn set_gpu_device(
        &mut self,
        device: Arc<Mutex<GpuDevice>>,
    ) {
        self.gpu_device = Some(device);
    }
    
    pub fn ensure_gpu_device_auto_detect(&mut self) -> Result<()> {
        if self.gpu_device.is_none() {
            let device = GpuDevice::auto_detect()?;
            self.set_gpu_device(Arc::new(Mutex::new(device)));
        }
        Ok(())
    }
}
```

### Usage in Forward Pass

```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Ensure GPU is available (strict: errors if not)
    self.ensure_gpu_device_auto_detect()?;
    
    // Proceed with GPU computation
    let device_arc = self.gpu_device.as_ref().unwrap().clone();
    // ...
}
```

---

## Pattern 8: AllocationStats Monitoring

### Tracking Memory Efficiency

```rust
pub struct AllocationStats {
    pub total_allocated: usize,        // Total bytes allocated
    pub total_wasted_padding: usize,   // Power-of-2 padding overhead
    pub reuse_count: usize,            // Times buffers were reused
    pub resize_count: usize,           // Reallocation events
}

// During training loop, monitor:
let stats = pool.allocation_stats();
println!("Reuse: {}, Resizes: {}, Waste: {} bytes",
    stats.reuse_count,
    stats.resize_count,
    stats.total_wasted_padding
);

// Target:
// - reuse_count >> resize_count (mostly reusing, not reallocating)
// - total_wasted_padding < 5% of total_allocated
```

---

## Pattern 9: Error Handling (Strict No-Fallback)

### Never Silently Fall Back to CPU

```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // 1. Check GPU device is attached
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU device not attached for component".to_string(),
        })?;
    
    // 2. Ensure GPU is available at runtime
    let mut device = device_arc.lock()
        .map_err(|e| ModelError::Backend {
            message: format!("GPU device lock failed: {}", e),
        })?;
    
    // 3. Execute kernel (may fail)
    let result = self.forward_gpu_kernel(...)?;
    
    // 4. Never do: if result.is_err() { return self.forward_cpu(...) }
    // Instead, propagate error
    Ok(result)
}
```

---

## Pattern 10: Kernel Fusion Example (RichardsGlu)

### Multi-Step Computation → Single Fused Kernel

**Operations**:
1. x1 = input @ w1
2. x2 = input @ w2
3. activated_x1 = x1 * richards(x1)
4. gate = richards(x2)
5. gated = activated_x1 * gate
6. output = input + gated @ w_out

**GPU Implementation** (in one kernel function):
```rust
pub fn forward_gpu_kernel(...) -> Result<()> {
    // Step 1-2: Two GEMMs (x1, x2)
    ops.gemm_f32(...)?; // x1 = input @ w1
    ops.gemm_f32(...)?; // x2 = input @ w2
    
    // Step 3: Activation (Richards curve kernel)
    ops.richards_curve(...)?; // Fused: x1 * sigma(x1)
    
    // Step 4-5: Gating and multiplication
    ops.richards_curve(...)?; // gate = sigma(x2)
    ops.mul(...)?;             // gated = activated_x1 * gate
    
    // Step 6: Final projection
    ops.gemm_f32(...)?; // output = gated @ w_out
    
    Ok(())
}
```

**Key Insight**: All 6 steps execute on GPU without downloading intermediates to CPU.

---

## Verification Checklist

For each GPU kernel implementation:

- [ ] GPU device auto-detection works (no silent fallback)
- [ ] Power-of-2 buffer sizing applied
- [ ] CPU vs GPU numerical accuracy validated (ε ≤ 1e-4)
- [ ] Batch size variation tested (1, 16, 32, 64, 128)
- [ ] Memory tracking enabled (AllocationStats)
- [ ] Error handling is strict (no fallback)
- [ ] Documentation explains GPU path
- [ ] Performance benchmarked vs CPU
- [ ] Integration with other components tested
- [ ] Edge cases tested (very small/large batches)

---

## References

- Implementation plan: `/d:/RustGPT/PHASE5.6_GPU_CONSOLIDATION_IMPLEMENTATION.md`
- RichardsGlu reference: `src/domain/richards/richards_glu.rs::forward_gpu_kernel`
- GpuDevice API: `src/domain/compute/gpu_device.rs`
- GpuMemoryPool: `src/domain/compute/gpu_memory.rs`
- GpuMatrixOps: `src/domain/compute/gpu_ops.rs`
- Tests: `tests/gpu_shared_components_phase56.rs`
