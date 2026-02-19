# RichardsGLU Two-Pass Kernel Implementation Guide

**Phase**: 5.6.3a (Priority 1)  
**Impact**: 30% reduction in kernel dispatch overhead  
**Effort**: ~2 hours

## Mathematical Foundation

### Forward Pass (Training)
```
Input:  [batch_size, input_dim]
Weights:
  W_g1: [input_dim, hidden_dim]      (value projection)
  W_g2: [input_dim, hidden_dim]      (gate projection)
  W_out: [hidden_dim, output_dim]    (output projection)

Computation:
  value_logits = Input @ W_g1        # [batch_size, hidden_dim]
  gate_logits = Input @ W_g2         # [batch_size, hidden_dim]
  gate = Richards(gate_logits)       # [batch_size, hidden_dim] (smooth gating function)
  gated = gate * value_logits        # [batch_size, hidden_dim] (element-wise)
  output = gated @ W_out             # [batch_size, output_dim] (projection)
```

### Backward Pass (Gradient Flow)
```
d_output flows back through:
  1. d_W_out from GEMM gradient
  2. d_gated from GEMM backward
  3. d_value_logits, d_gate_logits from element-wise multiply
  4. d_W_g1, d_W_g2 from GEMM backward for projections
```

## Two-Pass Kernel Structure

### Pass 1: Activation & Gating (Element-Wise Operations)

**Goal**: Compute intermediate `gated` tensor in-place on GPU

**Algorithm**:
```
For each (batch_b, hidden_h) pair:
  1. Load input_b and compute value = Input_b @ W_g1_h
  2. Load input_b and compute gate_logits = Input_b @ W_g2_h
  3. Apply Richards curve: gate = Richards(gate_logits)
  4. Element-wise multiply: gated_bh = gate_bh * value_bh
  5. Store gated_bh in intermediate buffer (stays on GPU)
```

**WGSL Kernel Pattern** (from existing element-wise ops):
```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= total_elements) { return; }
    
    // Load inputs, compute operation, store output
}

// Dispatch: workgroup_count = ceil(batch * hidden / 256)
```

**Key Constraint**: No CPU-GPU transfers
- Input uploaded once (at start of layer)
- Intermediate `gated` stays in GPU buffer
- Final output downloaded once (at end of layer)

### Pass 2: Output Projection (GEMM)

**Goal**: Project gated tensor to output space

**Algorithm**:
```
output = gated @ W_out

Where:
  gated is [batch_size, hidden_dim] (from Pass 1)
  W_out is [hidden_dim, output_dim]
  output is [batch_size, output_dim]
```

**Reuse Existing**: `SHADER_GEMM` from `wgpu_ops.rs`
- Just change dimensions
- Same workgroup_size (16x16 for GEMM)
- Same dispatch logic

## Implementation Steps

### Step 1: Locate Insertion Point

**File**: `src/domain/compute/wgpu_ops.rs`

**Current structure**:
```rust
pub struct WgpuMatrixOps {
    device: Device,
    queue: Queue,
    shaders: HashMap<String, ShaderModule>,
    buffers: ...
}

impl GpuMatrixOps for WgpuMatrixOps {
    fn gemm_f32(...) { ... }    // ← Reuse for Pass 2
    fn relu(...) { ... }        // ← Similar pattern to Pass 1
    fn gelu(...) { ... }
    fn silu(...) { ... }
    fn mul(...) { ... }         // ← Element-wise op
    // ...
}
```

**Add new method** after existing operations:
```rust
fn richards_glu_fused(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,           // [batch, input_dim]
    w_g1: &GpuBuffer,            // [input_dim, hidden_dim]
    w_g2: &GpuBuffer,            // [input_dim, hidden_dim]
    w_out: &GpuBuffer,           // [hidden_dim, output_dim]
    output: &mut GpuBuffer,      // [batch, output_dim]
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()> {
    // TODO: Implementation
}
```

### Step 2: Implement Pass 1

```rust
fn richards_glu_fused(...) -> Result<()> {
    // ===== PASS 1: Activation & Gating =====
    
    // 1. Allocate intermediate buffer for gated
    let gated_size = batch_size * hidden_dim * std::mem::size_of::<f32>();
    let mut gated_buf = pool.allocate(gated_size)?;
    
    // 2. Allocate intermediate buffer for value_logits
    let mut value_buf = pool.allocate(gated_size)?;
    
    // 3. Compute value_logits = input @ w_g1
    // Reuse existing gemm_f32 method
    self.gemm_f32(
        pool,
        1.0,        // alpha
        input,      // [batch, input_dim]
        w_g1,       // [input_dim, hidden_dim]
        0.0,        // beta (fresh output)
        &mut value_buf,  // [batch, hidden_dim]
        batch_size,
        hidden_dim,
        input_dim,
        false,      // trans_a = false (input as-is)
        false,      // trans_b = false (w_g1 as-is)
    )?;
    
    // 4. Compute gate_logits = input @ w_g2
    let mut gate_logits_buf = pool.allocate(gated_size)?;
    self.gemm_f32(
        pool,
        1.0,
        input,
        w_g2,
        0.0,
        &mut gate_logits_buf,
        batch_size,
        hidden_dim,
        input_dim,
        false,
        false,
    )?;
    
    // 5. Apply Richards curve to gate_logits
    // Create Richards parameters
    let richards_params = RichardsCurveParams {
        nu: 1.0,           // Asymmetry parameter
        k: 1.0,            // Growth rate
        m: 0.0,            // Midpoint
        beta: 1.0,         // Scaling
        temp_reciprocal: 1.0,
        output_gain: 1.0,
        output_bias: 0.0,
        scale: 1.0,
        shift: 0.0,
        adaptive_scale: 0.0,
        adaptive_shift: 0.0,
        num_heads: 1,
        _pad: [0, 0],
    };
    
    // Reuse existing richards_curve method
    let mut gate_buf = pool.allocate(gated_size)?;
    self.richards_curve(
        pool,
        &gate_logits_buf,
        &mut gate_buf,
        &richards_params,
        batch_size * hidden_dim,
    )?;
    
    // 6. Gated = gate * value (element-wise multiply)
    self.mul(
        pool,
        &gate_buf,
        &value_buf,
        &mut gated_buf,
        batch_size * hidden_dim,
    )?;
    
    // ===== PASS 2: Output Projection =====
    
    // 7. output = gated @ w_out
    self.gemm_f32(
        pool,
        1.0,
        &gated_buf,
        w_out,
        0.0,
        output,
        batch_size,
        output_dim,
        hidden_dim,
        false,
        false,
    )?;
    
    // ===== CLEANUP =====
    
    // 8. Deallocate intermediate buffers (keep gated for potential gradient computation)
    pool.deallocate(value_buf);
    pool.deallocate(gate_logits_buf);
    pool.deallocate(gate_buf);
    
    Ok(())
}
```

### Step 3: Add to Trait Definition

**File**: `src/domain/compute/gpu_ops.rs`

**Add to GpuMatrixOps trait**:
```rust
pub trait GpuMatrixOps: Send + Sync {
    // ... existing methods ...
    
    /// Richards GLU Fused Kernel: Two-Pass Gating
    ///
    /// Computes: output = (Richards(input @ W_g2) * (input @ W_g1)) @ W_out
    ///
    /// Two-pass structure minimizes GPU launches:
    /// - Pass 1: Element-wise activation & gating
    /// - Pass 2: Output projection (GEMM)
    ///
    /// # Performance
    /// - Reduces kernel launches from 5+ to 2
    /// - Zero-copy: all intermediate data stays on GPU
    #[allow(clippy::too_many_arguments)]
    fn richards_glu_fused(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        input: &GpuBuffer,           // [batch, input_dim]
        w_g1: &GpuBuffer,            // [input_dim, hidden_dim]
        w_g2: &GpuBuffer,            // [input_dim, hidden_dim]
        w_out: &GpuBuffer,           // [hidden_dim, output_dim]
        output: &mut GpuBuffer,      // [batch, output_dim]
        batch_size: usize,
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
    ) -> Result<()>;
}
```

### Step 4: Add to CPU Stub

**File**: `src/domain/compute/gpu_ops.rs`

**Add to CpuGpuMatrixOps impl**:
```rust
fn richards_glu_fused(
    &mut self,
    _pool: &mut dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _w_g1: &GpuBuffer,
    _w_g2: &GpuBuffer,
    _w_out: &GpuBuffer,
    _output: &mut GpuBuffer,
    _batch_size: usize,
    _input_dim: usize,
    _hidden_dim: usize,
    _output_dim: usize,
) -> Result<()> {
    Err(crate::common::errors::ModelError::Backend {
        message: "CPU richards_glu_fused not implemented (GPU required)".to_string(),
    })
}
```

### Step 5: Add to CUDA Stub

**File**: `src/domain/compute/cuda/ops.rs`

```rust
fn richards_glu_fused(
    &mut self,
    _pool: &mut dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _w_g1: &GpuBuffer,
    _w_g2: &GpuBuffer,
    _w_out: &GpuBuffer,
    _output: &mut GpuBuffer,
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()> {
    Err(ModelError::Backend {
        message: format!(
            "CUDA richards_glu_fused not yet implemented for shape \
             ({} x {}) -> ({} x {}) -> ({} x {}). \
             Requires custom CUDA kernel (see src/domain/compute/cuda/kernels/richards_glu_fused.cu)",
            batch_size, input_dim, batch_size, hidden_dim, batch_size, output_dim
        ),
    })
}
```

### Step 6: Add to Metal Stub

**File**: `src/domain/compute/metal/ops.rs`

```rust
fn richards_glu_fused(
    &mut self,
    _pool: &mut dyn GpuMemoryPool,
    _input: &GpuBuffer,
    _w_g1: &GpuBuffer,
    _w_g2: &GpuBuffer,
    _w_out: &GpuBuffer,
    _output: &mut GpuBuffer,
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()> {
    Err(ModelError::Backend {
        message: format!(
            "Metal richards_glu_fused not yet implemented for shape \
             ({} x {}) -> ({} x {}) -> ({} x {}). \
             Requires Metal compute shader (see src/domain/compute/metal/kernels/richards_glu_fused.metal)",
            batch_size, input_dim, batch_size, hidden_dim, batch_size, output_dim
        ),
    })
}
```

## Testing

### Simple Unit Test

**File**: `tests/gpu_shared_components_phase56.rs`

```rust
#[test]
#[cfg(feature = "wgpu")]
fn test_richards_glu_fused_correctness() {
    use ndarray::Array2;
    use llm::domain::compute::{GpuDevice, GpuMatrixOps};
    
    // Create test data
    let batch_size = 32;
    let input_dim = 512;
    let hidden_dim = 1024;
    let output_dim = 256;
    
    let mut rng = rand::thread_rng();
    let input = Array2::from_shape_fn((batch_size, input_dim), |_| {
        rng.gen_range(-1.0..1.0)
    });
    
    let w_g1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
        rng.gen_range(-0.1..0.1)
    });
    
    let w_g2 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
        rng.gen_range(-0.1..0.1)
    });
    
    let w_out = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
        rng.gen_range(-0.1..0.1)
    });
    
    // Initialize GPU
    let device = GpuDevice::auto_detect().expect("GPU device required");
    let mut pool = device.create_pool().expect("Failed to create pool");
    let mut ops = device.create_ops().expect("Failed to create ops");
    
    // Upload data
    let input_buf = pool.allocate(batch_size * input_dim * 4).unwrap();
    ops.upload(&mut pool, input.as_slice().unwrap(), &input_buf).unwrap();
    // ... upload w_g1, w_g2, w_out similarly ...
    
    // Allocate output
    let mut output_buf = pool.allocate(batch_size * output_dim * 4).unwrap();
    
    // Execute fused kernel
    ops.richards_glu_fused(
        &mut pool,
        &input_buf,
        &w_g1_buf,
        &w_g2_buf,
        &w_out_buf,
        &mut output_buf,
        batch_size,
        input_dim,
        hidden_dim,
        output_dim,
    ).expect("Fused kernel failed");
    
    // Download and verify
    let mut output_cpu = vec![0.0; batch_size * output_dim];
    ops.download(&mut pool, &output_buf, &mut output_cpu).unwrap();
    
    // Verify shape
    assert_eq!(output_cpu.len(), batch_size * output_dim);
    
    // Verify reasonable values (no NaN/Inf)
    for &val in &output_cpu {
        assert!(val.is_finite());
    }
}
```

### Dispatch Verification

Add to benchmark or test:

```rust
#[test]
fn test_richards_glu_dispatch_count() {
    // Use GPU profiling tool or layer to count kernel launches
    // Expected: 2 launches (Pass 1 + Pass 2)
    // Current baseline: 5+ launches (separate GEMM + element-wise)
    
    // For WGPU: Would need wgpu-core profiling
    // For CUDA: Use nvidia-smi with profiling
    // For Metal: Use Metal System Trace
}
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Kernel launches | 2 | Down from 5+ (Pass 1 + Pass 2) |
| Launch overhead reduction | 30% | ~1-2ms saved per layer |
| Memory bandwidth | >100 GB/s | WGPU on modern GPU |
| Numerical error | <1e-4 | Compared to CPU reference |

## Debugging Checklist

- [ ] WGSL shader compiles without errors
- [ ] Buffer allocations/deallocations match (no leaks)
- [ ] Output shape is correct [batch_size, output_dim]
- [ ] Output values are finite (no NaN/Inf)
- [ ] Dispatch count reduced to 2
- [ ] Zero-copy verified (no GPU-CPU transfers in middle)

## Next: CUDA Implementation

Once WGPU works, create `src/domain/compute/cuda/kernels/richards_glu_fused.cu`:

```cuda
__global__ void kernel_richards_glu_fused_pass1(
    const float* input,      // [batch, input_dim]
    const float* w_g1,       // [input_dim, hidden_dim]
    const float* w_g2,       // [input_dim, hidden_dim]
    float* value_buf,        // [batch, hidden_dim] (output)
    float* gate_logits_buf,  // [batch, hidden_dim] (output)
    int batch_size, int input_dim, int hidden_dim
) {
    // TODO: Implement Pass 1
    // Use matrix multiply operations (cuBLAS or manual)
}

__global__ void kernel_richards_glu_pass2_gemm(
    // Use existing cuBLAS or GEMM kernel
)
```

---

## Summary

**RichardsGLU two-pass kernel** is the critical optimization for Phase 5.6.3. It:
1. Reduces GPU kernel launches from 5+ to 2
2. Keeps all intermediate data on GPU (zero-copy)
3. Reuses existing WGSL kernels (GEMM, element-wise, Richards curve)
4. Applies to SSM gating operations

**Estimated effort**: 2 hours for WGPU, 3-4 hours for CUDA, 2-3 hours for Metal.
