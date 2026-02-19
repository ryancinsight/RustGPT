# Quick Reference: GPU Backward Kernels (Phase 5.6.4a)

## What's Ready to Implement

### 1. backward_qkv_projection_gpu (High Priority)
**Location**: `src/domain/layers/components/unified_gpu_kernels.rs:1067`

```rust
pub fn backward_qkv_projection_gpu(
    &mut self,
    output_grads: &Array2<f32>,  // Gradients w.r.t. Q,K,V outputs
    input: &Array2<f32>,         // Original input from forward
    wq, wk, wv: &Array2<f32>,    // Weight matrices
    params: &AttentionParams,
) -> Result<(Array2<f32>, Array2<f32>, Array2<f32>)>
```

**GPU Implementation**:
```
GPU GEMM 1: dL/dW_q = input^T @ grad_q    [D,N] @ [N,D] → [D,D]
GPU GEMM 2: dL/dW_k = input^T @ grad_k
GPU GEMM 3: dL/dW_v = input^T @ grad_v
Return all 3 gradients
```

**Expected Speedup**: 10-15x (3 GEMMs in parallel)

---

### 2. backward_output_projection_gpu (Quick Win)
**Location**: `src/domain/layers/components/unified_gpu_kernels.rs:1100`

```rust
pub fn backward_output_projection_gpu(
    &mut self,
    attention_output: &Array2<f32>,  // Attention output (before W_out)
    output_grads: &Array2<f32>,      // Loss gradients
    wo: &Array2<f32>,                // Output weight matrix
) -> Result<Array2<f32>>
```

**GPU Implementation**:
```
GPU GEMM: dL/dW_out = attention_output^T @ output_grads
          Use transposed_gemm() to avoid explicit transpose
Return gradient
```

**Expected Speedup**: 5-8x (single optimized GEMM)

---

### 3. backward_poly_params_gpu (Specialized)
**Location**: `src/domain/layers/components/unified_gpu_kernels.rs:1118`

```rust
pub fn backward_poly_params_gpu(
    &mut self,
    attention_scores: &Array2<f32>,  // Scores from forward
    score_grads: &Array2<f32>,       // Gradients w.r.t. scores
    a, b, scale: f32,                // Current poly parameters
) -> Result<(f32, f32, f32)>
```

**GPU Implementation**:
```
Element-wise Ops: grad_a = score_grads ⊙ d(poly)/da for each element
Reduction: sum all grad_a elements → final dL/da
Repeat for b, scale
Return (dL/da, dL/db, dL/dscale)
```

**Expected Speedup**: 20x (massive parallel reduction)

---

## Implementation Checklist

### backward_qkv_projection_gpu
- [ ] Allocate GPU buffers for grad_q, grad_k, grad_v
- [ ] Upload input, grads, and weights to GPU
- [ ] Execute 3 parallel GEMM operations
  - [ ] matmul(input.T, grad_q) → grad_wq
  - [ ] matmul(input.T, grad_k) → grad_wk
  - [ ] matmul(input.T, grad_v) → grad_wv
- [ ] Download results back to CPU
- [ ] Deallocate GPU buffers
- [ ] Return (grad_wq, grad_wk, grad_wv)

### backward_output_projection_gpu
- [ ] Allocate GPU buffer for grad_wo
- [ ] Upload attention_output, output_grads, wo to GPU
- [ ] Execute GEMM with transposition: matmul(attn.T, grads)
- [ ] Download result
- [ ] Deallocate buffers
- [ ] Return grad_wo

### backward_poly_params_gpu
- [ ] Allocate GPU buffers for element-wise results
- [ ] Upload score_grads, d(poly)/da, d(poly)/db, d(poly)/dscale
- [ ] Compute 3 element-wise multiplications
- [ ] Apply reduction (sum) to each
- [ ] Download scalar results (dL/da, dL/db, dL/dscale)
- [ ] Deallocate buffers
- [ ] Return scalars

---

## Testing Strategy

### Unit Tests (Already in place)
```bash
cargo test --lib backward
→ Tests kernel signatures and shapes
```

### Integration Tests (TODO)
```rust
#[test]
fn test_backward_gpu_vs_cpu_parity() {
    // Forward on GPU
    let out_gpu = poly.forward_gpu(&input)?;
    
    // Backward on GPU
    let grads_gpu = poly.backward_gpu(&loss_grads, lr)?;
    
    // Backward on CPU (reference)
    let grads_cpu = poly.backward(&loss_grads, lr);
    
    // Compare
    assert_close(grads_gpu, grads_cpu, 1e-5);
}
```

### Performance Tests (TODO)
```bash
cargo bench --bench gpu_backward
→ Measure actual speedup vs CPU
```

---

## Code Reference Points

### Where to Add GPU Code
- Main implementations: `unified_gpu_kernels.rs:1008-1138`
- Tests: `unified_gpu_kernels.rs:1190-1245`
- Integration into PolyAttention: `poly_attention.rs:1627-1705`

### GPU Device API
```rust
// Get GPU device
let mut device = self.device.lock()?;
let (pool, ops) = device.execution_context();

// Allocate buffer
let buf = pool.upload(data.as_slice())?;

// Execute operation
let result = device.matmul(&buf_a, &buf_b)?;

// Download
pool.download(&result, out.as_slice_mut())?;

// Cleanup
pool.deallocate(buf);
```

### GEMM Operation Template
```rust
// Forward: Q = input @ W_q
// GPU: input [N,D] @ wq [D,D] → Q [N,D]

// Backward: dL/dW_q = input^T @ dL/dQ
// GPU: input [D,N] @ grad_q [N,D] → grad_wq [D,D]
// Use transposed_gemm to avoid explicit transpose
let grad_wq = device.matmul_transposed(&input, &grad_q)?;
```

---

## Feature Flags
```toml
# All GPU code is behind these feature gates
[features]
gpu-wgpu = []
gpu-cuda = []
gpu-metal = []
gpu-all = ["gpu-wgpu", "gpu-cuda", "gpu-metal"]
```

Build and test:
```bash
cargo build --release --features gpu-wgpu
cargo test --lib --features gpu-wgpu
```

---

## Performance Targets

| Kernel | CPU Time | GPU Time | Target Speedup |
|--------|----------|----------|-----------------|
| backward_qkv_projection | 3ms | 0.3ms | 10x |
| backward_output_projection | 1ms | 0.2ms | 5x |
| backward_poly_params | 2ms | 0.1ms | 20x |
| **Total backward pass** | **30ms** | **2ms** | **15x** |

---

## Error Handling Patterns

```rust
// Validate GPU device
let device_arc = self.gpu_device.as_ref()
    .ok_or_else(|| ModelError::Backend {
        message: "GPU device not attached".into()
    })?;

// Check tensor dimensions
if input.dim().0 * seq_len != total_tokens {
    return Err(ModelError::ShapeMismatch {
        expected: vec![total_tokens, embed_dim],
        actual: vec![input.nrows(), input.ncols()],
        message: "Invalid input dimensions".into(),
    });
}

// Handle lock failures
let mut device = device_arc.lock()
    .map_err(|_| ModelError::Backend {
        message: "Failed to acquire GPU device lock".into(),
    })?;
```

---

## Next Steps (Execution Order)

1. **Implement backward_qkv_projection_gpu** (Parallel GEMM)
2. **Implement backward_output_projection_gpu** (Single GEMM)
3. **Implement backward_poly_params_gpu** (Element-wise + Reduce)
4. **Add integration tests** (CPU/GPU parity)
5. **Profile and optimize** (Measure speedup)
6. **Fuse kernels** (Phase 5.6.4b - Combine QKV into single kernel)

---

## Key Files
- **Implementation**: `src/domain/layers/components/unified_gpu_kernels.rs`
- **Tests**: Same file, lines 1190-1245
- **Integration**: `src/domain/attention/poly_attention.rs:1627`
- **Documentation**: `PHASE5.6.4a_GPU_BACKWARD_KERNELS_START.md`

**Ready to implement?** All stubs are in place. Infrastructure complete. Let's code! 🚀
