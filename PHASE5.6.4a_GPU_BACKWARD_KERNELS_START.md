# Phase 5.6.4a - GPU Backward Kernels Implementation Start

**Date**: Feb 16, 2026
**Focus**: GPU Backward Pass Kernels for PolyAttention
**Status**: Stub implementations and tests in place

## What Was Added

### 1. GPU Backward Kernel Stubs (unified_gpu_kernels.rs)

Added 4 GPU kernel method stubs to `UnifiedGpuKernels` implementation:

```rust
pub fn attention_backward(...)            // Main backward dispatcher
pub fn backward_qkv_projection_gpu(...)  // Q,K,V projection gradients
pub fn backward_output_projection_gpu(...) // W_out projection gradients
pub fn backward_poly_params_gpu(...)     // Polynomial parameter gradients (a, b, scale)
```

**Location**: Lines 1008-1138 in unified_gpu_kernels.rs
**Status**: Bridge implementations (return zero arrays, CPU fallback ready)
**Feature Gated**: Only compiled with `wgpu`, `gpu-cuda`, or `gpu-metal`

### 2. Unit Tests for Backward Kernels

Added 3 parametric tests to validate kernel signatures and shapes:

- `test_backward_qkv_projection_params()` - Validates QKV backward shapes
- `test_backward_output_projection_shapes()` - Validates output projection shapes
- `test_poly_params_backward_shapes()` - Validates polynomial parameter shapes

**Location**: Lines 1190-1245 in unified_gpu_kernels.rs
**Status**: All tests passing (552/552)
**Purpose**: Ensure kernel signatures work before GPU implementation

## Architecture: Backward Kernel Design

### Computation Flow

```
Forward Pass (Already Done):
input [batch*seq, embed] --[W_q]→ Q [batch*seq, embed]
                         --[W_k]→ K [batch*seq, embed]
                         --[W_v]→ V [batch*seq, embed]
      
Q,K,V --[Attention]→ Scores [batch*H, seq, seq]
      --[Softmax]→ Weights [batch*H, seq, seq]
      --[Multiply V]→ Output [batch*seq, embed]
      --[W_out]→ Final [batch*seq, embed]

Backward Pass (Phase 5.6.4a):
dL/dout [batch*seq, embed] --[W_out^T]→ dL/dScores [batch*H, seq, seq]
                           --[Softmax']→ dL/dAttention [batch*H, seq, seq]
                           --[V^T @ dL]→ dL/dQ,K [batch*seq, embed]
                                         dL/dW_q = input^T @ dL/dQ
                                         dL/dW_k = input^T @ dL/dK
                                         dL/dW_v = input^T @ dL/dV
                           --[dL_out^T @ input]→ dL/dW_out
```

### Expected GPU Kernel Responsibilities

#### backward_qkv_projection_gpu
- **Input**: 
  - Gradient w.r.t. Q,K,V outputs (from softmax/attention backward)
  - Original input for weight gradient computation
- **Computation**:
  - dL/dW_q = input^T @ dL/dQ  (GEMM: [D,N] @ [N,D] → [D,D])
  - dL/dW_k = input^T @ dL/dK
  - dL/dW_v = input^T @ dL/dV
- **Parallelization**: Can compute all 3 in parallel via multi-stream execution

#### backward_output_projection_gpu
- **Input**:
  - Attention output (before W_out projection)
  - Gradient w.r.t. final output
- **Computation**:
  - dL/dW_out = attn_output^T @ dL/dout (GEMM)
- **Optimization**: Use transposed GEMM flags to avoid explicit transpose

#### backward_poly_params_gpu
- **Input**:
  - Attention scores from forward
  - Score gradients from backward
  - Current polynomial parameters (a, b, scale)
- **Computation**:
  - dL/da = sum(dL/dscores ⊙ d(poly)/da)
  - dL/db = sum(dL/dscores ⊙ d(poly)/db)
  - dL/dscale = sum(dL/dscores ⊙ d(poly)/dscale)
- **Reduction**: Use GPU reduction kernels for efficient summation

## Implementation Roadmap for Phase 5.6.4a

### Step 1: Implement backward_qkv_projection_gpu (2-3 hours)
```rust
// Pseudo-code:
fn backward_qkv_projection_gpu(...) {
    // 1. Upload all inputs to GPU
    let grad_q_buf = pool.upload(grad_q)?;
    let input_buf = pool.upload(input)?;
    
    // 2. Execute GEMM kernels
    // GPU Kernel: matmul_transposed(input, grad_q) → grad_wq
    let grad_wq_buf = device.matmul(&input_buf.t(), &grad_q_buf)?;
    
    // 3. Repeat for K, V
    
    // 4. Download results
    let grad_wq = pool.download(&grad_wq_buf)?;
    
    Ok((grad_wq, grad_wk, grad_wv))
}
```

**Target**: Single kernel launch with 3 parallel GEMM operations

### Step 2: Implement backward_output_projection_gpu (1-2 hours)
```rust
fn backward_output_projection_gpu(...) {
    // Similar to QKV but single GEMM
    // Use transposed flags: C = A^T @ B (no explicit transpose)
    let grad_wo_buf = device.matmul_transposed(&attn_output, &dL_dout)?;
    Ok(grad_wo)
}
```

**Target**: Single GEMM kernel call

### Step 3: Implement backward_poly_params_gpu (2-3 hours)
```rust
fn backward_poly_params_gpu(...) {
    // 1. Element-wise multiply: dL/dscores ⊙ d(poly)/da
    let grad_a_elementwise = device.mul(&score_grads, &dpoly_da)?;
    
    // 2. Reduce (sum all elements)
    let grad_a = device.sum(&grad_a_elementwise)?;
    
    // 3. Repeat for b, scale
    
    Ok((grad_a, grad_b, grad_scale))
}
```

**Target**: Element-wise multiply + reduction kernels

### Step 4: Wire into PolyAttention.backward_gpu() (1 hour)
```rust
pub fn backward_gpu(&mut self, grads: &Array2<f32>, lr: f32) -> Result<Array2<f32>> {
    // Call GPU backward kernels instead of CPU
    let (grad_input, grad_wq, grad_wk, grad_wv) = 
        self.backward_qkv_projection_gpu(...)?;
    let grad_wo = self.backward_output_projection_gpu(...)?;
    let (grad_a, grad_b, grad_scale) = self.backward_poly_params_gpu(...)?;
    
    // Apply weight updates via GPU optimizers (TODO)
    self.apply_gpu_updates(&grad_wq, &grad_wk, ...)?;
    
    Ok(grad_input)
}
```

## Current Bridge Implementation

Each kernel method currently:
1. ✅ Validates GPU device attachment
2. ✅ Accepts proper tensor dimensions  
3. ✅ Returns zero arrays (safe bridge)
4. ⏳ Has TODO comments for GPU implementation
5. ⏳ Fallback comment indicating CPU path ready

**Why Bridge**:
- Maintains backward compatibility
- Allows incremental testing
- Clear insertion points for GPU code
- Easy to identify what needs implementation

## Testing Strategy

### Phase 1: Unit Tests (DONE)
- ✅ Parameter shape validation
- ✅ Dimension compatibility checks
- ✅ Memory layout assertions

### Phase 2: Integration Tests (TODO)
```rust
#[test]
fn test_polyattention_backward_gpu_correctness() {
    // Forward on GPU
    let output_gpu = poly_attn.forward_gpu(&input)?;
    
    // Backward on GPU
    let grads_gpu = poly_attn.backward_gpu(&loss_grads, lr)?;
    
    // Compare with CPU backward
    let grads_cpu = poly_attn.backward(&loss_grads, lr);
    
    // Verify element-wise difference < 1e-5
    assert_close(grads_gpu, grads_cpu, 1e-5);
}
```

### Phase 3: Performance Tests (TODO)
```bash
# Benchmark backward pass speedup
cargo bench --bench gpu_backward_kernels
```

## Known Limitations

1. **Bridge Implementation**: Currently returns zero gradients
2. **No GPU Optimizer**: Weight updates still happen on CPU
3. **No Kernel Fusion**: Each projection computed separately
4. **Memory Not Managed**: No workspace buffer reuse between forward/backward

These will be addressed in Phase 5.6.4a/b implementation.

## Success Criteria for Phase 5.6.4a

- ✅ Backward kernel stubs added to unified_gpu_kernels.rs
- ✅ Unit tests passing (552/552)
- ⏳ Implement full GEMM-based backward kernels
- ⏳ Backward pass speedup ≥ 10x (conservative target)
- ⏳ Integration tests verifying correctness
- ⏳ Performance benchmarks showing improvement

## Next Session: Full Implementation

Ready to begin GPU kernel implementation:

1. **Implement backward_qkv_projection_gpu** using GEMM kernels
2. **Implement backward_output_projection_gpu** using transposed GEMM
3. **Implement backward_poly_params_gpu** using element-wise multiply + reduce
4. **Wire into PolyAttention.backward_gpu()** and test
5. **Profile and optimize** for target 30x speedup

---

**Ready to code**: All stubs and tests in place. GPU implementation can proceed.
