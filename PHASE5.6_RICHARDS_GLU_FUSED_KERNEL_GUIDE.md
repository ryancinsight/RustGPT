# RichardsGLU Fused Kernel Implementation Guide

**Priority**: P0 (Blocking)  
**Target**: 25x speedup (50ms → 2ms for 1K batch)  
**Backend Priority**: CUDA > Metal > Vulkan  

---

## Overview

RichardsGLU uses two GPU passes to minimize global memory roundtrips:

**Pass 1** (Fused Kernel):
- W1 projection + Richards activation + W2 projection + Sigmoid gating
- Keeps intermediate values in GPU cache

**Pass 2** (Standard GEMM):
- Output projection + Residual addition
- Downloads result or chains to next layer

---

## Implementation Structure

### File: `src/domain/layers/components/fused_kernels_module.rs`

Current status: Skeleton with TODO comments (lines 108-123)

```rust
pub fn execute(
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &Array2<f32>,
    w1: &Array2<f32>,
    w2: &Array2<f32>,
    w_out: &Array2<f32>,
    params: &RichardsGluFusedKernelParams,
) -> Result<Array2<f32>> {
    // TODO: Implement two-pass fused kernel execution
    // Phase 5.6.3 implementation
    
    // For now, return placeholder
    Ok(input.clone())
}
```

### Implementation Plan

#### Step 1: Lock GPU Device

```rust
pub fn execute(
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &Array2<f32>,
    w1: &Array2<f32>,
    w2: &Array2<f32>,
    w_out: &Array2<f32>,
    params: &RichardsGluFusedKernelParams,
) -> Result<Array2<f32>> {
    let mut device = device.lock().map_err(|_| ModelError::Backend {
        message: "Failed to acquire GPU device lock for RichardsGLU kernel".to_string(),
    })?;

    let batch_size = params.batch_size as usize;
    let input_dim = params.input_dim as usize;
    let hidden_dim = params.hidden_dim as usize;
    let output_dim = params.output_dim as usize;
```

#### Step 2: Validate Input Dimensions

```rust
    // Validate input shapes
    if input.dim().0 != batch_size || input.dim().1 != input_dim {
        return Err(ModelError::InvalidShape {
            expected: vec![batch_size, input_dim],
            actual: vec![input.dim().0, input.dim().1],
        });
    }

    if w1.dim() != (input_dim, hidden_dim) {
        return Err(ModelError::InvalidShape {
            expected: vec![input_dim, hidden_dim],
            actual: vec![w1.dim().0, w1.dim().1],
        });
    }

    if w2.dim() != (input_dim, hidden_dim) {
        return Err(ModelError::InvalidShape {
            expected: vec![input_dim, hidden_dim],
            actual: vec![w2.dim().0, w2.dim().1],
        });
    }

    if w_out.dim() != (hidden_dim, output_dim) {
        return Err(ModelError::InvalidShape {
            expected: vec![hidden_dim, output_dim],
            actual: vec![w_out.dim().0, w_out.dim().1],
        });
    }
```

#### Step 3: Allocate GPU Buffers

```rust
    let (_, ops) = device.execution_context();

    // Buffer sizes in bytes
    let input_size = batch_size * input_dim * std::mem::size_of::<f32>();
    let hidden_size = batch_size * hidden_dim * std::mem::size_of::<f32>();
    let output_size = batch_size * output_dim * std::mem::size_of::<f32>();
    let w1_size = input_dim * hidden_dim * std::mem::size_of::<f32>();
    let w2_size = input_dim * hidden_dim * std::mem::size_of::<f32>();
    let w_out_size = hidden_dim * output_dim * std::mem::size_of::<f32>();

    // Upload input
    let mut input_buf = ops.allocate(input_size)?;
    ops.upload(input.as_slice().unwrap(), &mut input_buf)?;

    // Upload weights
    let mut w1_buf = ops.allocate(w1_size)?;
    ops.upload(w1.as_slice().unwrap(), &mut w1_buf)?;

    let mut w2_buf = ops.allocate(w2_size)?;
    ops.upload(w2.as_slice().unwrap(), &mut w2_buf)?;

    let mut w_out_buf = ops.allocate(w_out_size)?;
    ops.upload(w_out.as_slice().unwrap(), &mut w_out_buf)?;

    // Allocate intermediate buffers (Pass 1)
    let mut x1_buf = ops.allocate(hidden_size)?;  // x @ W1
    let mut x2_buf = ops.allocate(hidden_size)?;  // x @ W2
    let mut value_buf = ops.allocate(hidden_size)?;  // x1 * richards(x1)
    let mut gate_buf = ops.allocate(hidden_size)?;  // sigmoid(x2)
    let mut gated_buf = ops.allocate(hidden_size)?;  // value * gate

    // Allocate output buffer (Pass 2)
    let mut output_buf = ops.allocate(output_size)?;
```

#### Step 4: Pass 1 - Projection + Activation + Gating

```rust
    // ============================================================
    // PASS 1: Fused kernel operations
    // ============================================================

    // Step 1: First projection - x @ W1 → (batch_size, hidden_dim)
    ops.gemm(
        1.0,
        &input_buf,
        &w1_buf,
        0.0,
        &mut x1_buf,
        batch_size,
        hidden_dim,
        input_dim,
        false,  // transpose_a
        false,  // transpose_b
    )?;

    // Step 2: Second projection - x @ W2 → (batch_size, hidden_dim)
    ops.gemm(
        1.0,
        &input_buf,
        &w2_buf,
        0.0,
        &mut x2_buf,
        batch_size,
        hidden_dim,
        input_dim,
        false,
        false,
    )?;

    // Step 3: Apply Richards activation to x1
    // Download x1 for CPU-side Richards curve computation
    // (TODO: Implement GPU Richards kernel for future optimization)
    let mut x1_data = vec![0.0f32; batch_size * hidden_dim];
    ops.download(&x1_buf, &mut x1_data)?;

    // Apply Richards curve: value = x1 * richards(x1; nu, k, m, beta)
    let nu = params.richards_nu;
    let k = params.richards_k;
    let m = params.richards_m;
    let beta = params.richards_beta;
    let temp_inv = params.activation_temp_inv;

    for i in 0..batch_size * hidden_dim {
        let x1_val = x1_data[i];
        // Richards curve: x / ((1 + |x - m|^nu * exp(-k * (x - m)))^(1/nu))
        // Simplified form for efficiency
        let diff = x1_val - m;
        let abs_diff = diff.abs();
        let curve_val = if abs_diff > 1e-6 {
            let factor = 1.0 + abs_diff.powf(nu) * (-k * diff).exp();
            x1_val / factor.powf(1.0 / nu)
        } else {
            x1_val
        };
        x1_data[i] = curve_val * temp_inv;
    }

    // Upload activated x1 back (now as "value")
    ops.upload(&x1_data, &mut value_buf)?;

    // Step 4: Apply sigmoid to x2 for gating
    // Download x2 for CPU-side sigmoid computation
    // (TODO: Implement GPU sigmoid kernel)
    let mut x2_data = vec![0.0f32; batch_size * hidden_dim];
    ops.download(&x2_buf, &mut x2_data)?;

    // Apply sigmoid: gate = 1 / (1 + exp(-x2 * temp_inv))
    let gate_temp_inv = params.gate_temp_inv;
    for i in 0..batch_size * hidden_dim {
        let x2_val = x2_data[i] * gate_temp_inv;
        x2_data[i] = 1.0 / (1.0 + (-x2_val).exp());
    }

    // Upload sigmoid output (gate)
    ops.upload(&x2_data, &mut gate_buf)?;

    // Step 5: Element-wise multiply to get gated activation
    // gated = value * gate (element-wise)
    // For now, download and do on CPU (TODO: GPU element-wise kernel)
    let mut value_data = vec![0.0f32; batch_size * hidden_dim];
    ops.download(&value_buf, &mut value_data)?;

    let mut gated_data = vec![0.0f32; batch_size * hidden_dim];
    for i in 0..batch_size * hidden_dim {
        gated_data[i] = value_data[i] * x2_data[i];
    }

    // Upload gated output
    ops.upload(&gated_data, &mut gated_buf)?;
```

#### Step 5: Pass 2 - Output Projection + Residual

```rust
    // ============================================================
    // PASS 2: Output projection + residual connection
    // ============================================================

    // Compute output = gated @ W_out → (batch_size, output_dim)
    ops.gemm(
        1.0,
        &gated_buf,
        &w_out_buf,
        0.0,
        &mut output_buf,
        batch_size,
        output_dim,
        hidden_dim,
        false,
        false,
    )?;

    // Add residual: output += input
    // If output_dim == input_dim, add element-wise
    if output_dim == input_dim {
        ops.add_scaled(
            1.0,
            &input_buf,
            &mut output_buf,
            batch_size * input_dim,
        )?;
    } else {
        eprintln!("Warning: output_dim ({}) != input_dim ({}), residual not added",
            output_dim, input_dim);
    }

    // Download result
    let mut output = Array2::zeros((batch_size, output_dim));
    ops.download(&output_buf, output.as_slice_mut().unwrap())?;

    // Cleanup GPU buffers
    device.deallocate(input_buf);
    device.deallocate(w1_buf);
    device.deallocate(w2_buf);
    device.deallocate(w_out_buf);
    device.deallocate(x1_buf);
    device.deallocate(x2_buf);
    device.deallocate(value_buf);
    device.deallocate(gate_buf);
    device.deallocate(gated_buf);
    device.deallocate(output_buf);

    Ok(output)
}
```

---

## Optimization Opportunities

### 1. GPU-Side Richards Curve Kernel

Currently computing Richards curve on CPU. Create a CUDA/Metal/Vulkan kernel:

```rust
// In unified_gpu_kernels.rs (Phase 5.6.3)
pub fn apply_richards_curve_kernel(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    params: &RichardsCurveParams,
    size: usize,
) -> Result<()> {
    // Launch Richards curve kernel
    // Kernel pseudocode:
    // for i in parallel:
    //   x = input[i]
    //   diff = x - m
    //   abs_diff = abs(diff)
    //   factor = 1.0 + pow(abs_diff, nu) * exp(-k * diff)
    //   output[i] = x / pow(factor, 1.0 / nu)
}
```

### 2. GPU-Side Sigmoid Kernel

Similar optimization for sigmoid activation:

```rust
pub fn apply_sigmoid_kernel(
    device: &mut GpuDevice,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    temp_inv: f32,
    size: usize,
) -> Result<()> {
    // Launch sigmoid kernel
    // Kernel pseudocode:
    // for i in parallel:
    //   x = input[i] * temp_inv
    //   output[i] = 1.0 / (1.0 + exp(-x))
}
```

### 3. GPU-Side Element-Wise Multiply

Fuse gating operation into GPU:

```rust
pub fn element_wise_multiply_kernel(
    device: &mut GpuDevice,
    a: &GpuBuffer,
    b: &GpuBuffer,
    output: &mut GpuBuffer,
    size: usize,
) -> Result<()> {
    // Launch element-wise multiply kernel
    // Kernel pseudocode:
    // for i in parallel:
    //   output[i] = a[i] * b[i]
}
```

---

## Integration with SharedFeedforward

### File: `src/domain/layers/components/feedforward.rs`

Add `forward_gpu()` method:

```rust
impl SharedFeedforward {
    /// GPU-accelerated forward pass for RichardsGlu
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        use crate::domain::compute::GpuComponent;
        use crate::domain::layers::components::fused_kernels_module::richards_glu_fused;

        let gpu_device = require_gpu_or_error(&self.gpu_device, "forward_gpu")?;

        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(richards) => {
                let (batch_size, embed_dim) = input.dim();
                let hidden_dim = richards.hidden_dim;

                // Get weights and biases
                let w1 = &richards.w1;
                let w2 = &richards.w2;
                let w_out = &richards.w_out;

                // Build kernel parameters from Richards curve config
                let params = richards_glu_fused::RichardsGluFusedKernelParams {
                    batch_size: batch_size as u32,
                    input_dim: embed_dim as u32,
                    hidden_dim: hidden_dim as u32,
                    output_dim: embed_dim as u32,
                    richards_nu: richards.nu,
                    richards_k: richards.k,
                    richards_m: richards.m,
                    richards_beta: richards.beta,
                    activation_temp_inv: 1.0 / richards.activation_temp,
                    gate_nu: richards.gate_nu,
                    gate_k: richards.gate_k,
                    gate_temp_inv: 1.0 / richards.gate_temp,
                };

                // Execute fused kernel
                let mut device = gpu_device.lock().map_err(|_| ModelError::Backend {
                    message: "Failed to acquire GPU device lock".to_string(),
                })?;

                let (pool, ops) = device.execution_context();

                richards_glu_fused::execute(
                    &gpu_device,
                    pool,
                    ops,
                    input,
                    w1,
                    w2,
                    w_out,
                    &params,
                )
            }
            FeedForwardVariant::MixtureOfExperts(_) => {
                // TODO: Implement MoE GPU variant
                Err(ModelError::NotImplemented {
                    message: "GPU MoE not yet implemented".to_string(),
                })
            }
        }
    }

    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_gpu(&mut self, _input: &Array2<f32>) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message: "GPU features not enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal".to_string(),
        })
    }
}
```

---

## Testing Strategy

### Unit Test: RichardsGLU Correctness

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_richards_glu_fused_correctness() {
        if GpuDevice::auto_detect().is_err() {
            println!("No GPU, skipping GPU test");
            return;
        }

        let batch_size = 32;
        let input_dim = 512;
        let hidden_dim = 2048;
        let output_dim = 512;

        // Create input and weights
        let input = Array2::<f32>::random((batch_size, input_dim), Normal::new(0.0, 0.1).unwrap());
        let w1 = Array2::<f32>::random((input_dim, hidden_dim), Normal::new(0.0, 0.01).unwrap());
        let w2 = Array2::<f32>::random((input_dim, hidden_dim), Normal::new(0.0, 0.01).unwrap());
        let w_out = Array2::<f32>::random((hidden_dim, output_dim), Normal::new(0.0, 0.01).unwrap());

        // Parameters
        let params = RichardsGluFusedKernelParams::new(batch_size, input_dim, hidden_dim, output_dim);

        // Get GPU device
        let device = Arc::new(Mutex::new(GpuDevice::auto_detect().unwrap()));
        let mut d = device.lock().unwrap();
        let (pool, ops) = d.execution_context();

        // Execute GPU kernel
        let gpu_result = richards_glu_fused::execute(
            &device,
            pool,
            ops,
            &input,
            &w1,
            &w2,
            &w_out,
            &params,
        ).unwrap();

        // Verify output shape
        assert_eq!(gpu_result.dim(), (batch_size, output_dim));
    }
}
```

### Benchmark: Performance Validation

```rust
#[cfg(test)]
mod benches {
    use super::*;

    #[test]
    fn bench_richards_glu_1k_batch() {
        if GpuDevice::auto_detect().is_err() {
            println!("No GPU, skipping GPU benchmark");
            return;
        }

        let batch_size = 1024;
        let input_dim = 768;
        let hidden_dim = 3072;
        let output_dim = 768;

        // Warm up
        for _ in 0..5 {
            // Run forward pass
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..100 {
            // Run forward pass
        }
        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f32 / 100.0;

        println!("RichardsGLU 1K batch: {:.2}ms (target: 2ms, speedup: {:.1}x)",
            avg_ms, 50.0 / avg_ms);

        assert!(avg_ms < 5.0, "Performance degradation: {:.2}ms > 5ms", avg_ms);
    }
}
```

---

## Debugging Checklist

- [ ] GPU device auto-detects correctly (CUDA > Metal > Vulkan)
- [ ] Input buffer uploaded successfully
- [ ] Weights uploaded successfully
- [ ] GEMM operations produce correct matrix multiply results
- [ ] Richards curve values are reasonable (not NaN/Inf)
- [ ] Sigmoid values are in [0, 1] range
- [ ] Gating produces correct output dimensions
- [ ] Output buffer downloaded correctly
- [ ] Residual addition preserves output shape
- [ ] All GPU buffers deallocated (no memory leaks)

---

## Next Steps

1. Implement basic `execute()` following the structure above
2. Test with CPU reference implementation for correctness
3. Benchmark against target (2ms for 1K batch)
4. If needed, optimize with GPU-side kernels (Phase 5.6.3)
5. Integrate with SharedFeedforward
6. Move to PolyAttention fused kernel
