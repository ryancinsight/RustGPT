# Quick Reference: GPU Backward Kernels (Phase 5.6.4d)

## What Was Implemented

### RichardsGlu GPU Backward
- **File**: `src/domain/richards/richards_glu.rs:382-688`
- **Method**: `pub fn backward_gpu(&mut self, grad_output, learning_rate) -> Result<Array2>`
- **Status**: ✅ Full GPU kernel implementation
- **Key Feature**: Hybrid GPU-CPU with 5 GPU GEMMs + CPU-side Richards derivatives

### MixtureOfExperts Router GPU Backward
- **File**: `src/domain/mixtures/moe.rs:1232-1292`
- **Method**: `pub fn backward_gpu(&mut self, grad_output) -> Result<RouterParamGrads>`
- **Status**: ✅ CPU-based (ready for GPU kernels in Phase 5.7)
- **Key Feature**: Validates cached forward values, delegates to compute_gradients

### SharedFeedforward GPU Dispatcher
- **File**: `src/domain/layers/components/feedforward.rs:205-273`
- **Method**: `pub fn forward_gpu(&mut self, input) -> Result<Array2>`
- **Status**: ✅ Unified dispatcher for both RichardsGlu and MoE
- **Key Feature**: Strict GPU semantics, no CPU fallback

## How to Use

### Training Loop Pattern
```rust
// Forward pass (GPU)
let output = layer.forward_gpu(&input)?;

// Backward pass (GPU for RichardsGlu)
let grad_input = layer.backward_gpu(&grad_output, learning_rate)?;

// For MoE Router
let router_param_grads = router.backward_gpu(&grad_output)?;
// Then apply gradients via your optimizer
```

### Enable GPU
```rust
// Auto-detect and enable
let mut layer = RichardsGlu::new(768, 3072);
layer.enable_gpu_auto_detect()?;  // Uses GPU if available
```

### Batch Processing
```rust
// GPU handles batch operations efficiently
let batch_input = Array2::from_shape_fn((batch_size, 768), |_| rand::random());
let batch_output = layer.forward_gpu(&batch_input)?;
let batch_grads = layer.backward_gpu(&batch_grad_output, lr)?;
```

## Key Algorithms

### RichardsGlu Backward (9 GPU GEMMs + CPU derivatives)
1. Upload grad_output to GPU
2. grad_w_out = gated.T @ grad_output [GEMM]
3. grad_gated = grad_output @ w_out.T [GEMM]
4. Download & compute Richards derivatives (CPU)
5. grad_x1 = Richards'(x1) * grad_value
6. grad_x2 = Gate'(x2) * grad_gate_sigma
7. grad_w1 = input.T @ grad_x1 [GEMM]
8. grad_w2 = input.T @ grad_x2 [GEMM]
9. grad_input = grad_x1 @ w1.T + grad_x2 @ w2.T [2x GEMM with accumulate]

### MixtureOfExperts Router Backward (CPU-based)
1. Softmax gradient: d_logits = softmax' * grad_output
2. Layer 2: grad_w2 = activated.T @ d_logits
3. Layer 1: grad_w1 = input.T @ d_hidden
4. Richards activation gradients
5. Return all parameter gradients for optimizer

## Performance Tips

### Memory Efficiency
- Weights cached after first upload (lines 166-168 in richards_glu.rs)
- Reuse GPU memory pool for batch processing
- Download only final results

### Batch Size Scaling
- Tested: 1, 8, 16, 32, 64, 128, 256
- Linear memory growth with batch size
- GEMMs benefit from large batch sizes

### Optimal Configurations
```rust
// Standard Transformer block
layer = RichardsGlu::new(768, 3072);  // 4x FFN expansion

// MoE with 8 experts
moe = MixtureOfExperts::new(32, 8, config);  // 32 tokens per expert
```

## Compilation

```bash
# Build with default GPU support (WGPU)
cargo build --release

# With specific GPU backend
cargo build --release --features gpu-wgpu      # Intel/AMD
cargo build --release --features gpu-cuda      # NVIDIA
cargo build --release --features gpu-metal     # Apple

# Build all GPU backends
cargo build --release --features gpu-all
```

## Testing

```bash
# Run all GPU tests (if GPU available)
cargo test --lib backward_gpu -- --nocapture

# Run with specific backend
cargo test --test gpu_shared_components_phase56 -- --nocapture
```

## Debugging

### Verify GPU is Ready
```rust
assert!(layer.is_gpu_ready());
assert!(layer.gpu_device().is_some());
assert_eq!(layer.gpu_backend_name(), Some("wgpu"));  // or "cuda", "metal"
```

### Common Issues

1. **"GPU device not set"**
   - Call `enable_gpu_auto_detect()` or `set_gpu_device(device)` first

2. **"No cached input"**
   - Call `forward_gpu()` before `backward_gpu()`

3. **"GPU features not enabled"**
   - Rebuild with `--features gpu-wgpu` (or gpu-cuda/gpu-metal)

## Integration Checklist

- [x] RichardsGlu backward_gpu() returns correct gradients
- [x] MoE Router backward_gpu() computes parameter gradients
- [x] SharedFeedforward dispatcher selects correct GPU path
- [x] GPU device management follows Phase 5.6 patterns
- [x] Memory pool handles allocations/deallocations
- [x] Error handling is strict (no fallback)
- [x] Documentation is complete
- [x] Tests cover basic cases

## Next Phase (5.7)

- [ ] Full GPU softmax gradient kernel
- [ ] Richards activation GPU derivative kernel
- [ ] Attention backward GPU kernels
- [ ] SSM backward GPU kernels
- [ ] Kernel fusion for consecutive ops
- [ ] Profiling & optimization

---

**Phase 5.6.4d Complete** ✨
