# Phase 5.6.3: Fused Kernels Implementation Guide

## Overview
This document provides detailed implementation patterns for fused GPU kernels in Phase 5.6.

**Module Location**: `src/domain/layers/components/fused_kernels_module.rs`

**Status**: Foundation laid (parameter structures defined, placeholder implementations)

**Next Steps**: Implement kernel execution logic for each component

---

## 1. RichardsGLU Fused Kernel (PRIORITY 1)

### Current State
- **File**: `src/domain/richards/richards_glu.rs`
- **Current Approach**: 5 separate GPU launches
  1. W1 projection: `input @ W1` (GEMM)
  2. W2 projection: `input @ W2` (GEMM)
  3. Richards activation: `richards(x1)` (custom kernel)
  4. Gating: `sigmoid(x2)` (element-wise)
  5. Output projection: `gated @ W_out` (GEMM)

### Target
- **Two GPU launches**: Combined Pass 1 + standard Pass 2
- **Memory traffic reduction**: ~60% fewer global memory writes
- **Speedup**: 25x (50ms → 2ms on 1K batch)

### Two-Pass Strategy

#### Pass 1: Combined Projection + Activation + Gating
```
Kernel: richardson_glu_fused_pass1

Input:
- x: (batch_size, input_dim) on GPU
- W1: (input_dim, hidden_dim) transposed on GPU
- W2: (input_dim, hidden_dim) transposed on GPU
- params: RichardsGluFusedKernelParams

Computation (per thread block):
1. Load input row into shared memory
2. Compute x1 = dot(x_row, W1_col) for all hidden_dim in parallel
3. Apply Richards activation: value = x1 * richards(x1)
4. Compute x2 = dot(x_row, W2_col) for all hidden_dim in parallel
5. Apply gate: gate = sigmoid(x2)
6. Combine: gated = value * gate
7. Write to global memory: output[row, :] = gated

Output:
- gated: (batch_size, hidden_dim) on GPU
```

#### Pass 2: Standard Output Projection
```
Kernel: Standard GEMM (existing GPU ops)

Input:
- gated: (batch_size, hidden_dim) from Pass 1
- x: (batch_size, input_dim) residual
- W_out: (hidden_dim, output_dim) transposed

Computation:
- output = x + (gated @ W_out)

Output:
- output: (batch_size, output_dim) on GPU
```

### Implementation Steps

#### Step 1: Define Kernel Structure (DONE)
```rust
// File: src/domain/layers/components/fused_kernels_module.rs
pub mod richards_glu_fused {
    pub struct RichardsGluFusedKernelParams { ... }
    pub fn execute(...) -> Result<Array2<f32>> { ... }
}
```

#### Step 2: Implement WGSL Kernel
**File**: Create `src/domain/compute/gpu_kernels/richards_glu_fused.wgsl`

```wgsl
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w1: array<f32>;
@group(0) @binding(2) var<storage, read> w2: array<f32>;
@group(0) @binding(3) var<storage, read_write> gated: array<f32>;

struct Params {
    batch_size: u32,
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    richards_nu: f32,
    richards_k: f32,
    richards_m: f32,
    richards_beta: f32,
    activation_temp_inv: f32,
}

@group(0) @binding(4) var<uniform> params: Params;

// Richards curve: sigma(x) = 1 / (1 + (m / x)^k)
fn richards_curve(x: f32, params: Params) -> f32 {
    let shifted = x - params.richards_m;
    if abs(shifted) < 1e-6 { return 0.5; }
    let ratio = params.richards_m / shifted;
    let powered = pow(ratio, params.richards_k);
    return 1.0 / (1.0 + powered);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let hidden_idx = global_id.y;
    
    if (batch_idx >= params.batch_size || hidden_idx >= params.hidden_dim) {
        return;
    }
    
    // Step 1: Compute x1 = x @ W1
    var x1 = 0.0f32;
    for (var i = 0u; i < params.input_dim; i = i + 1u) {
        let x_val = x[batch_idx * params.input_dim + i];
        let w1_val = w1[i * params.hidden_dim + hidden_idx];
        x1 += x_val * w1_val;
    }
    
    // Step 2: Apply Richards activation
    let richards_val = richards_curve(x1, params);
    let value = x1 * richards_val;  // RichardsActivation: x * richards(x)
    
    // Step 3: Compute x2 = x @ W2
    var x2 = 0.0f32;
    for (var i = 0u; i < params.input_dim; i = i + 1u) {
        let x_val = x[batch_idx * params.input_dim + i];
        let w2_val = w2[i * params.hidden_dim + hidden_idx];
        x2 += x_val * w2_val;
    }
    
    // Step 4: Apply sigmoid gating
    let gate = 1.0 / (1.0 + exp(-x2 * params.activation_temp_inv));
    
    // Step 5: Combine value and gate
    let gated_val = value * gate;
    gated[batch_idx * params.hidden_dim + hidden_idx] = gated_val;
}
```

#### Step 3: Implement Rust Execution Wrapper
**File**: Modify `src/domain/layers/components/fused_kernels_module.rs`

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
    let mut device_guard = device.lock().map_err(|_| {
        ModelError::Backend { 
            message: "Failed to acquire GPU device lock".to_string() 
        }
    })?;
    
    let (pool, ops) = device_guard.execution_context();
    
    let batch_size = params.batch_size as usize;
    let input_dim = params.input_dim as usize;
    let hidden_dim = params.hidden_dim as usize;
    let output_dim = params.output_dim as usize;
    
    // 1. Upload inputs
    let input_buf = pool.upload(input.as_slice().unwrap())?;
    let w1_buf = pool.upload(w1.as_slice().unwrap())?;
    let w2_buf = pool.upload(w2.as_slice().unwrap())?;
    let w_out_buf = pool.upload(w_out.as_slice().unwrap())?;
    
    // 2. Allocate intermediate buffer (gated output from Pass 1)
    let mut gated_buf = pool.allocate(batch_size * hidden_dim * 4)?;
    
    // 3. Execute Pass 1 fused kernel
    ops.richard_glu_fused_pass1(
        pool,
        &input_buf,
        &w1_buf,
        &w2_buf,
        &mut gated_buf,
        params,
    )?;
    
    // 4. Allocate output buffer
    let mut output_buf = pool.allocate(batch_size * output_dim * 4)?;
    
    // 5. Execute Pass 2 (standard GEMM with residual)
    // output = input + (gated @ W_out)
    ops.copy_within_device(pool, &input_buf, &mut output_buf, batch_size * output_dim)?;
    ops.gemm_f32(
        pool,
        1.0,
        &gated_buf,
        &w_out_buf,
        1.0,  // Add to existing output (residual)
        &mut output_buf,
        batch_size,
        output_dim,
        hidden_dim,
        false,
        false,
    )?;
    
    // 6. Download result
    let mut output = Array2::zeros((batch_size, output_dim));
    pool.download(&output_buf, output.as_slice_mut().unwrap())?;
    
    Ok(output)
}
```

#### Step 4: Integrate into RichardsGlu
**File**: `src/domain/richards/richards_glu.rs`

```rust
pub fn forward_gpu_fused(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU device not set".to_string()
        })?
        .clone();
    
    let mut dev = device.lock().unwrap();
    let (pool, _ops) = dev.execution_context();
    
    self.ensure_gpu_cache(pool, _ops)?;
    
    let params = RichardsGluFusedKernelParams::new(
        input.nrows(),
        input.ncols(),
        self.w1.ncols(),
        self.w_out.ncols(),
    );
    
    // Call fused kernel executor
    crate::domain::layers::components::fused_kernels_module::richards_glu_fused::execute(
        &device,
        pool,
        _ops,
        input,
        &self.w1,
        &self.w2,
        &self.w_out,
        &params,
    )
}
```

#### Step 5: Testing
**File**: Create/modify `tests/gpu_richardson_glu_fused.rs`

```rust
#[test]
fn test_richardson_glu_fused_execution() {
    if let Ok(mut device) = GpuDevice::auto_detect() {
        let batch_size = 32;
        let input_dim = 768;
        let hidden_dim = 3072;
        let output_dim = 768;
        
        let input = Array2::random((batch_size, input_dim), Normal::new(0.0, 0.1).unwrap());
        let w1 = Array2::random((input_dim, hidden_dim), Normal::new(0.0, 0.1).unwrap());
        let w2 = Array2::random((input_dim, hidden_dim), Normal::new(0.0, 0.1).unwrap());
        let w_out = Array2::random((hidden_dim, output_dim), Normal::new(0.0, 0.1).unwrap());
        
        let params = RichardsGluFusedKernelParams::new(batch_size, input_dim, hidden_dim, output_dim);
        
        let (pool, ops) = device.execution_context();
        
        let result = richards_glu_fused::execute(
            &Arc::new(Mutex::new(device)),
            pool,
            ops,
            &input,
            &w1,
            &w2,
            &w_out,
            &params,
        ).expect("fused kernel execution");
        
        assert_eq!(result.dim(), (batch_size, output_dim));
    }
}

#[test]
fn test_richardson_glu_fused_vs_unfused() {
    if let Ok(device) = GpuDevice::auto_detect() {
        // Create RichardsGlu layer
        let mut layer = RichardsGlu::new(768, 3072);
        layer.set_gpu_device(Arc::new(Mutex::new(device.clone())));
        
        let input = Array2::random((32, 768), Normal::new(0.0, 0.1).unwrap());
        
        // Compare fused vs unfused
        let unfused = layer.forward_gpu(&input).expect("unfused forward");
        let fused = layer.forward_gpu_fused(&input).expect("fused forward");
        
        // Numerical accuracy: ≤ 1e-4
        let max_diff = unfused.iter()
            .zip(fused.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        println!("Max difference: {:.2e}", max_diff);
        assert!(max_diff <= 1e-4, "Fused and unfused differ by {}", max_diff);
    }
}

#[test]
fn test_richardson_glu_fused_kernel_launches() {
    if let Ok(device) = GpuDevice::auto_detect() {
        let mut layer = RichardsGlu::new(768, 3072);
        layer.set_gpu_device(Arc::new(Mutex::new(device.clone())));
        
        // Reset stats
        let initial_stats = device.memory_stats();
        
        let input = Array2::random((32, 768), Normal::new(0.0, 0.1).unwrap());
        let _ = layer.forward_gpu_fused(&input);
        
        let final_stats = device.memory_stats();
        
        // Should have 2 kernel launches (Pass 1 + GEMM)
        println!("Kernel launches: {}", final_stats.kernel_launches);
        assert_eq!(final_stats.kernel_launches, 2, "Expected 2 GPU launches");
    }
}
```

### Code Checklist
- [ ] WGSL kernel implemented (`richards_glu_fused.wgsl`)
- [ ] CUDA kernel implemented (`.cu` file)
- [ ] Metal kernel implemented (`.metal` file)
- [ ] Rust wrapper updated in `richards_glu.rs`
- [ ] Tests pass for all GPU backends
- [ ] Numerical accuracy verified (ε ≤ 1e-4)
- [ ] Performance benchmarked (target: 25x speedup)
- [ ] Kernel launch count verified (exactly 2)

---

## 2. PolyAttention Fused Kernel (PRIORITY 2)

### Current State
- **File**: `src/domain/layers/components/temporal_processing_gpu.rs`
- **Current Approach**: Placeholder (currently returns CPU computation)
- **Operations**: Q/K/V projections → polynomial scoring → softmax → output projection

### Target
- **Single GPU launch**: Combine all operations
- **Speedup**: 30x (30ms → 1ms on 512 batch)

### Implementation Pattern
Similar to RichardsGLU, but with polynomial basis computation instead of Richards curves.

---

## 3. Mamba Selective Scan (PRIORITY 3)

### Current State
- **File**: `src/domain/layers/components/temporal_processing_gpu.rs`
- **Challenge**: Inherently recurrent (can't parallelize scan operations)
- **Strategy**: GPU implementation of sequential scan with optimized state updates

### Target
- **Optimized recurrent kernel**: 20x speedup via efficient state transitions
- **Speedup**: 20x (40ms → 2ms on 512 batch)

---

## 4. AttentionContext GPU Ops (PRIORITY 4)

### Current State
- **File**: `src/domain/layers/components/attention_context_gpu.rs`
- **Operations**: Matrix multiplications (input @ context_strength)
- **Strategy**: Use existing GPU GEMM operations

### Target
- **Direct GEMM usage**: 30x speedup
- **No custom kernel needed**: Use existing high-performance ops

---

## GPU Device API Integration

### Required GpuMatrixOps Methods

Your GPU backend must implement:

```rust
pub trait GpuMatrixOps {
    // Existing methods
    fn gemm_f32(...) -> Result<()>;
    fn copy_within_device(...) -> Result<()>;
    
    // NEW for RichardsGLU fused
    fn richard_glu_fused_pass1(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        w1: &GpuBuffer,
        w2: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &RichardsGluFusedKernelParams,
    ) -> Result<()>;
    
    // NEW for PolyAttention
    fn poly_attention_fused(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,
        wq: &GpuBuffer,
        wk: &GpuBuffer,
        wv: &GpuBuffer,
        wo: &GpuBuffer,
        output: &mut GpuBuffer,
        params: &PolyAttentionFusedParams,
    ) -> Result<()>;
}
```

---

## Testing Strategy

1. **Unit Tests**: Test individual kernels in isolation
2. **Integration Tests**: Test within layer (RichardsGlu, PolyAttention, etc.)
3. **Numerical Validation**: Compare GPU vs CPU outputs
4. **Performance Benchmarks**: Measure actual speedup
5. **Regression Tests**: Ensure no accuracy degradation

---

## Performance Monitoring

Use `GpuBackendStats` to track:
- `kernel_launches`: Should be 2 (RichardsGLU fused) or 1 (PolyAttention)
- `bytes_uploaded`: Track data transfer overhead
- `bytes_downloaded`: Verify output size

---

## Common Pitfalls

1. **Tensor Layout**: Ensure W matrices are transposed correctly for GEMM
2. **Shared Memory**: Watch for bank conflicts in WGSL workgroups
3. **Warp Divergence**: Minimize conditional branches in kernels
4. **Register Pressure**: Balance occupancy vs computation
5. **Numerical Stability**: Use appropriate epsilon values for sigmoid/softmax

---

## Success Criteria (Phase 5.6.3)

- ✅ All 4 kernels have working GPU implementations
- ✅ Fused kernels reduce launches by 50%+
- ✅ Numerical accuracy ≤ 1e-3 (≤ 1e-4 for attention)
- ✅ All tests pass on auto-detected GPU
- ✅ Performance targets met (25x, 30x, 20x, 30x speedups)
- ✅ Code compiles cleanly (no warnings)
