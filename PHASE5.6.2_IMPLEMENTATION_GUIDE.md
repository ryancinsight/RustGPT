# Phase 5.6.2 Implementation Guide: Fused GPU Kernels

**Status**: Ready for Implementation
**Focus**: RichardsGLU Fused Kernel (WGPU)
**Estimated Time**: 4-6 hours

---

## Overview

This guide covers the implementation of fused GPU kernels for Phase 5.6.2, starting with RichardsGLU on WGPU.

### What is a Fused Kernel?

Instead of:
```
1. x1 = input @ W1        (GEMM, global memory write)
2. x2 = input @ W2        (GEMM, global memory write)
3. value = richards(x1)   (activation, global memory write)
4. gated = value * x2     (multiply, global memory write)
5. output = gated @ W_out (GEMM, global memory write)
```

A fused kernel does all 5 operations in a single GPU kernel with minimal global memory traffic:
```
// Single kernel
for each (batch, hidden_dim):
    x1 = input @ W1
    x2 = input @ W2
    value = richards(x1)
    gated = value * x2
    output = gated @ W_out
    write to global memory once
```

**Benefits**:
- 5x fewer global memory writes (bottleneck on GPUs)
- Intermediate results stay in registers/shared memory
- ~25x speedup on large batches

---

## Step 1: WGSL Shader for RichardsGLU Fused Kernel

### Shader Structure

Add to `src/domain/compute/wgpu_ops.rs` after the existing shaders (around line 463):

```wgsl
// ============================================================================
// WGSL: RichardsGLU Fused Kernel (Phase 5.6.2)
// ============================================================================
// 
// Single-pass kernel combining:
// 1. x1 = input @ W1
// 2. x2 = input @ W2
// 3. value = x1 * richards_curve(x1)
// 4. gate = richards_curve(x2)
// 5. gated = value * gate
// 6. output = gated @ W_out
//
// Reduces global memory roundtrips from 5 to 1
```

### Key Algorithm

```
INPUT: input[batch_size, input_dim], W1, W2, W_out[hidden_dim, output_dim]
       Richards parameters (nu, k, m, beta, temperature)

FOR EACH (batch, hidden_dim) in parallel:
    // Step 1-2: Compute projections (matrix-vector product)
    x1[hidden_dim] = 0
    x2[hidden_dim] = 0
    for k in 0..input_dim:
        x1[hidden_dim] += input[batch, k] * W1[k, hidden_dim]
        x2[hidden_dim] += input[batch, k] * W2[k, hidden_dim]
    
    // Step 3: Apply Richards activation to value
    // y = x * richards_curve(x)
    richards_x1 = richards_curve(x1)
    value[hidden_dim] = x1[hidden_dim] * richards_x1
    
    // Step 4: Apply Richards to gate
    gate[hidden_dim] = richards_curve(x2)
    
    // Step 5: Gate the value
    gated[hidden_dim] = value[hidden_dim] * gate[hidden_dim]
    
    // Step 6: Final projection
    output[output_dim] = 0
    for hidden in 0..hidden_dim:
        output[output_dim] += gated[hidden] * W_out[hidden, output_dim]
    
OUTPUT: output[batch_size, output_dim]
```

---

## Step 2: Parameters Structure

Add to `src/domain/compute/wgpu_ops.rs`:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RichardsGluFusedParams {
    // Dimensions
    batch_size: u32,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    
    // Richards curve parameters (shared for value and gate)
    nu: f32,
    k: f32,
    m: f32,
    beta: f32,
    
    // Temperature (for softening/hardening)
    temp_reciprocal: f32,
    
    // Gate-specific parameters
    gate_scale: f32,
    gate_bias: f32,
    gate_temp_reciprocal: f32,
    
    // Value-specific parameters
    value_scale: f32,
    output_gain: f32,
    _pad1: u32,
    _pad2: u32,
}
```

---

## Step 3: WGSL Implementation

Key sections:

### Richards Curve Helper
```wgsl
fn richards_curve(x: f32, p: RichardsGluFusedParams) -> f32 {
    // Richards curve: 1 / (1 + exp(-(k*x + m*x^2 - nu)))
    // Simplified: return 1.0 / (1.0 + exp(-x * k))
    // Full: let sig = 1.0 / (1.0 + exp(-k * x - m * x * x + nu));
    let scaled = x * p.k + x * x * p.m - p.nu;
    let sig = 1.0 / (1.0 + exp(-clamp(scaled, -20.0, 20.0)));
    return p.beta + (1.0 - p.beta) * sig;
}
```

### Main Compute Loop
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let hidden_idx = global_id.y;
    
    if (batch_idx >= params.batch_size || hidden_idx >= params.hidden_dim) {
        return;
    }
    
    // Compute x1 = input @ W1 (this row, column)
    var x1: f32 = 0.0;
    var x2: f32 = 0.0;
    for (var k = 0u; k < params.input_dim; k++) {
        let input_idx = batch_idx * params.input_dim + k;
        let w1_idx = k * params.hidden_dim + hidden_idx;
        let w2_idx = k * params.hidden_dim + hidden_idx;
        
        x1 += input[input_idx] * w1[w1_idx];
        x2 += input[input_idx] * w2[w2_idx];
    }
    
    // Apply Richards activation to value
    let richards_x1 = richards_curve(x1, params);
    let value = x1 * richards_x1;
    
    // Apply Richards to gate
    let gate = richards_curve(x2 * params.gate_temp_reciprocal, params);
    
    // Combine
    let gated = value * gate;
    
    // Final projection: gated @ W_out
    var output_val: f32 = 0.0;
    for (var out = 0u; out < params.output_dim; out++) {
        let w_out_idx = hidden_idx * params.output_dim + out;
        output_val = gated * w_out[w_out_idx];
        
        // Accumulate to output atomically
        let output_idx = batch_idx * params.output_dim + out;
        atomicAdd(&output[output_idx], output_val);
    }
}
```

---

## Step 4: Rust Implementation

### Add method to `WgpuMatrixOps`

```rust
impl WgpuMatrixOps {
    /// Fused RichardsGLU kernel: x1 = input @ W1; x2 = input @ W2
    /// value = x1 * richards(x1); gate = richards(x2); 
    /// output = (value * gate) @ W_out
    pub fn richards_glu_fused(
        &mut self,
        input: &GpuBuffer,
        w1: &GpuBuffer,
        w2: &GpuBuffer,
        w_out: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        params: RichardsGluFusedParams,
    ) -> Result<()> {
        // 1. Create pipeline
        let pipeline = self.get_pipeline(SHADER_RICHARDS_GLU_FUSED);
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        
        // 2. Create params buffer
        let params_buffer = self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("RichardsGLU Fused Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        
        // 3. Create bind group (input, w1, w2, w_out, output, params)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RichardsGLU Fused Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: w_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // 4. Dispatch with appropriate workgroup counts
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("RichardsGLU Fused Encoder"),
            },
        );
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RichardsGLU Fused Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            
            // Dispatch: (batch_size / 16) x (hidden_dim / 16) x 1
            let batch_groups = (batch_size as u32 + 15) / 16;
            let hidden_groups = (hidden_dim as u32 + 15) / 16;
            cpass.dispatch_workgroups(batch_groups, hidden_groups, 1);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}
```

---

## Step 5: Integration with RichardsGlu

Modify `src/domain/richards/richards_glu.rs`:

```rust
pub fn forward_gpu_fused(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU device not set for RichardsGlu".to_string(),
        })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // Upload input
    let input_slice = input.as_slice().ok_or_else(|| 
        ModelError::InvalidInput {
            message: "Input array must be contiguous".to_string(),
        })?;
    let input_buf = pool.upload(input_slice)?;
    
    // Allocate output
    let batch_size = input.nrows();
    let embedding_dim = self.w_out.ncols();
    let output_size = batch_size * embedding_dim * 4;
    let mut output_buf = pool.allocate(output_size)?;
    
    // Run fused kernel
    self.forward_gpu_fused_kernel(pool, ops, &input_buf, &mut output_buf, batch_size)?;
    
    // Download result
    let mut output_array = Array2::zeros((batch_size, embedding_dim));
    let output_slice = output_array.as_slice_mut().unwrap();
    pool.download(&output_buf, output_slice)?;
    
    Ok(output_array)
}

fn forward_gpu_fused_kernel(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &GpuBuffer,
    output: &mut GpuBuffer,
    batch_size: usize,
) -> Result<()> {
    self.ensure_gpu_cache(pool, ops)?;
    
    let cache = self.gpu_cache.as_ref().unwrap();
    let embedding_dim = self.w1.nrows();
    let hidden_dim = self.w1.ncols();
    
    let params = RichardsGluFusedParams {
        batch_size: batch_size as u32,
        input_dim: embedding_dim as u32,
        hidden_dim: hidden_dim as u32,
        output_dim: embedding_dim as u32,
        nu: self.richards_activation.richards_curve.nu,
        k: self.richards_activation.richards_curve.k,
        m: self.richards_activation.richards_curve.m,
        beta: self.richards_activation.richards_curve.beta,
        temp_reciprocal: 1.0 / self.richards_activation.richards_curve.temperature,
        gate_scale: self.gate.curve.k,
        gate_bias: self.gate.curve.beta,
        gate_temp_reciprocal: 1.0 / self.gate.curve.temperature,
        value_scale: 1.0,
        output_gain: 1.0,
        _pad1: 0,
        _pad2: 0,
    };
    
    ops.richards_glu_fused(
        pool,
        input,
        &cache.w1,
        &cache.w2,
        &cache.w_out,
        output,
        batch_size,
        embedding_dim,
        hidden_dim,
        embedding_dim,
        params,
    )
}
```

---

## Step 6: Testing

Create `tests/gpu_richards_glu_fused.rs`:

```rust
#[test]
#[cfg(feature = "wgpu")]
fn test_richards_glu_fused_numerical_accuracy() {
    // 1. Create RichardsGlu with GPU device
    let mut glu = RichardsGlu::new(64, 128);
    
    // 2. Enable GPU (strict, no fallback)
    assert!(glu.set_gpu_device_auto_detect().is_ok());
    assert!(glu.is_gpu_ready());
    
    // 3. Create input
    let input = Array2::from_shape_fn((32, 64), |(i, j)| {
        ((i * 64 + j) as f32).sin()
    });
    
    // 4. Forward pass (CPU reference)
    let cpu_output = glu.forward(&input).unwrap();
    
    // 5. Forward pass (GPU fused)
    let gpu_output = glu.forward_gpu_fused(&input).unwrap();
    
    // 6. Validate accuracy
    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    
    println!("Max difference GPU vs CPU: {}", max_diff);
    assert!(max_diff <= 1e-4, "GPU accuracy not within tolerance: {}", max_diff);
}

#[test]
#[cfg(feature = "wgpu")]
fn test_richards_glu_fused_performance() {
    let mut glu = RichardsGlu::new(512, 2048);
    glu.set_gpu_device_auto_detect().unwrap();
    
    let input = Array2::zeros((1024, 512));
    
    // Warmup
    let _ = glu.forward_gpu_fused(&input);
    
    // Benchmark
    use std::time::Instant;
    let start = Instant::now();
    for _ in 0..100 {
        let _ = glu.forward_gpu_fused(&input);
    }
    let elapsed = start.elapsed();
    
    let time_per_pass = elapsed.as_micros() as f32 / 100.0;
    println!("RichardsGLU fused kernel: {:.2} µs/pass (1K batch)", time_per_pass);
    
    // Target: 2ms = 2000 µs for 1K batch
    assert!(time_per_pass < 2000.0, "Performance below target");
}
```

---

## Step 7: Backward Pass (Optional for Phase 5.6.2)

Once forward pass is working, add:

```rust
pub fn backward_gpu_fused(
    &mut self,
    grad_output: &Array2<f32>,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    // Gradient computation:
    // ∂L/∂input = grad_output @ W_out^T @ (gate-aware jacobian)
    // ∂L/∂W1 = input^T @ (grad_output @ W_out^T * ∂gated/∂x1)
    // ∂L/∂W2 = input^T @ (grad_output @ W_out^T * ∂gated/∂x2)
    // ∂L/∂W_out = gated^T @ grad_output
    
    // Full implementation requires storing intermediate activations
    // For now: placeholder
    todo!()
}
```

---

## Validation Checklist

- [ ] Shader compiles without errors
- [ ] Parameters structure matches WGSL
- [ ] Buffer binding order is correct
- [ ] Workgroup size is valid (16x16 or smaller)
- [ ] Numerical accuracy within ε ≤ 1e-4
- [ ] Performance: > 20x speedup on 1K batch
- [ ] Memory: No memory leaks (checked with profiler)
- [ ] Handles edge cases (batch_size=1, small hidden_dim)

---

## Success Criteria

1. **Correctness**: GPU output matches CPU reference (tolerance ε ≤ 1e-4)
2. **Performance**: 25x speedup on 1K batch = ~2ms vs 50ms
3. **Integration**: Works with GpuComponent trait (automatic detection)
4. **Memory**: Zero allocation overhead after warmup
5. **Robustness**: Handles all input sizes without errors

---

## Debugging Tips

If shader fails to compile:
```
cargo build --features gpu-wgpu 2>&1 | grep -A5 "wgpu"
```

If numerical accuracy fails:
```
// Reduce tolerance initially to debug
assert!(max_diff <= 1e-2); // Debug threshold
```

If performance is slow:
- Check workgroup size (16x16 is good balance)
- Verify no unnecessary synchronization
- Profile with GPU profiler

---

## Next Steps After Forward Pass

1. Implement backward pass (gradients)
2. Implement PolyAttention GPU kernels
3. Integrate Mamba/RG-LRU GPU kernels
4. Full zero-copy pipeline
5. Multi-GPU support
