# Phase 5.8: Attention GPU Backward Pass - KICKOFF

**Date**: Feb 18, 2026  
**Status**: Planning & Kickoff  
**Target**: Implement GPU backward pass for PolyAttention (Days 6-8 of Phase 5.7 roadmap)

## Problem Statement

PolyAttention `forward_gpu()` exists but doesn't cache intermediate values needed for `backward_gpu()`:
- Q, K, V projections (batch_size × num_heads × head_dim)
- Attention weights/scores before softmax
- Head outputs before concatenation
- Currently backward_gpu() falls back to CPU (line 1765)

## Architecture & Intermediates Needed

### Forward Pass (CPU reference at line 2181-2183)
```rust
let q_all = input.dot(&self.w_q);  // (n, embed_dim)
let k_all = input.dot(&self.w_k);
let v_all = input.dot(&self.w_v);

// Then for each head (0..num_heads):
//   q = q_all[.., h_idx*head_dim..(h_idx+1)*head_dim]
//   k = k_all[.., h_idx*head_dim..(h_idx+1)*head_dim]
//   v = v_all[.., h_idx*head_dim..(h_idx+1)*head_dim]
//   scores = q @ k.T / sqrt(head_dim)
//   attn_weights = softmax(scores)  // CRITICAL FOR BACKWARD
//   head_output = attn_weights @ v
```

### Intermediates to Cache
1. **Q, K, V Projections** (n × embed_dim each)
   - Used in all gradient computations
   - Required for: grad_w_q, grad_w_k, grad_w_v, grad_input

2. **Attention Weights** (n × n) per head
   - Used to compute grad_q, grad_k, grad_v
   - Required for backward through softmax (scaled)

3. **Head Outputs** (n × head_dim) per head
   - Used to compute grad_w_out, grad_v
   - Required for weight gradient accumulation

### Backward Path (from CPU backward lines 2122-2400)
```
grad_output (n × embed_dim)
    ↓
[per head] grad_y_gated_all = grad_output @ w_out_block.T
    ↓
[softmax backward] grad_attn_weights = grad_y_gated_all @ v.T, then scale by softmax
    ↓
[linear backward]
    grad_q = grad_attn_weights @ k
    grad_k = grad_attn_weights.T @ q
    grad_v = grad_attn_weights.T @ grad_y_gated_all
    ↓
[projection backward]
    grad_input += grad_q @ w_q.T + grad_k @ w_k.T + grad_v @ w_v.T
    grad_w_q += input.T @ grad_q
    grad_w_k += input.T @ grad_k
    grad_w_v += input.T @ grad_v
```

## Implementation Plan

### Phase 5.8.1: Forward Caching (Days 1-2)

**Goal**: Modify PolyAttention::forward_gpu() to download & cache intermediates

**Steps**:
1. Modify attention_gpu_kernel::forward_gpu() to return intermediate handles
   - Q, K, V GPU buffers (from projection step)
   - Attention weights GPU buffers (from softmax step)
   - Head outputs GPU buffers

2. Update PolyAttention::forward_gpu() to:
   - Receive intermediate buffer handles from kernel
   - Download to CPU after kernel execution
   - Cache in struct fields (similar to RichardsGlu)

3. Add cache fields to PolyAttention struct:
   - `cached_q: Option<Array2<f32>>`  // (n, embed_dim)
   - `cached_k: Option<Array2<f32>>`  // (n, embed_dim)
   - `cached_v: Option<Array2<f32>>`  // (n, embed_dim)
   - `cached_attn_weights: Option<Vec<Array2<f32>>>` // Per-head weights
   - `cached_head_outputs: Option<Vec<Array2<f32>>>` // Per-head outputs

### Phase 5.8.2: Softmax Gradient Kernel (Days 1-2)

**Goal**: Create GPU softmax backward kernel (similar to existing softmax kernel)

**Implementation**:
- Use existing GpuSoftmaxGradientKernel (from Phase 5.7)
- Applies to attention weight gradients
- Kernel: `d(softmax) = softmax * (grad_out - (softmax * grad_out).sum())`

### Phase 5.8.3: Attention Backward Kernel (Days 3-4)

**Goal**: Implement GPU attention backward computation

**Steps**:
1. Create AttentionBackwardKernel in src/domain/compute/
   - GEMM for grad_q = grad_attn @ k
   - GEMM for grad_k = grad_attn.T @ q
   - GEMM for grad_v = grad_attn.T @ grad_y
   - Element-wise for grad accumulation

2. Key operations:
   ```rust
   // Softmax backward on attention weights
   grad_attn_weights = softmax_grad(attn_weights, grad_y_gated @ v.T)
   
   // Project gradients back through attention
   grad_q = grad_attn_weights @ k           (n × head_dim)
   grad_k = grad_attn_weights.T @ q         (n × head_dim)
   grad_v = grad_attn_weights.T @ grad_y    (n × head_dim)
   ```

3. Tests: 8-10 comprehensive tests covering:
   - Single head attention
   - Multi-head attention
   - Causal masking effects
   - Numerical validation vs CPU

### Phase 5.8.4: Weight Gradient Computation (Days 5-6)

**Goal**: Compute gradients for w_q, w_k, w_v, w_out via GPU

**Steps**:
1. Use existing GEMM kernels for:
   - grad_w_q = input.T @ grad_q_accumulated
   - grad_w_k = input.T @ grad_k_accumulated
   - grad_w_v = input.T @ grad_v_accumulated
   - grad_w_out = head_outputs.T @ grad_output

2. Implementation:
   ```rust
   // Per-head accumulation via reduction kernels
   grad_q_total = reduce_cat(grad_q_per_head)  // Combine head gradients
   grad_w_q = input.T @ grad_q_total
   
   // Repeat for k, v, out
   ```

3. Tests: Validate weight gradient shapes and values

### Phase 5.8.5: Integration & Testing (Days 7-8)

**Goal**: Full backward_gpu() implementation with validation

**Steps**:
1. Update backward_gpu() to use GPU kernels:
   - Replace CPU fallback with GPU operations
   - Download final gradients to CPU
   - Apply with optimizers (Adam, etc.)

2. Comprehensive tests:
   - test_attention_backward_gpu_basic
   - test_attention_backward_numerical_validation
   - test_attention_backward_gradient_shapes
   - test_attention_backward_with_heads (multi-head)
   - test_attention_backward_with_causal (causal masking)

3. Performance profiling:
   - Compare GPU vs CPU backward time
   - Memory usage analysis
   - Target: 2-3x speedup over CPU

## Architecture Pattern (from Phase 5.7)

Same pattern as RichardsGlu:

```
Forward GPU:
  1. Compute intermediates on GPU
  2. Download to CPU  ← KEY ADDITION
  3. Cache in struct fields
  4. Return final output

Backward GPU:
  1. Access cached CPU intermediates
  2. Compute gradients (GPU/CPU hybrid)
  3. Download results to CPU
  4. Update parameters via optimizers
```

## Files to Modify

### Primary
- `src/domain/attention/poly_attention.rs`
  - Add cache fields (lines 400-450)
  - Update forward_gpu() (lines 1620-1691)
  - Implement backward_gpu() (lines 1702-1767)

- `src/domain/compute/mod.rs`
  - Export AttentionBackwardKernel (if created as new module)

### Secondary
- `src/domain/layers/components/attention_gpu_kernel.rs`
  - Modify forward_gpu() to return intermediate handles
  - Or create new attention_backward_kernel module

## Testing Strategy

### Unit Tests (8-10 tests)
- Located in `domain::attention::poly_attention::tests`
- Test both GPU implementations and CPU fallback
- Numerical validation via finite differences

### Integration Tests
- Backward through full PolyAttention layer
- Multi-head attention scenarios
- With causal masking enabled

### Performance Benchmarks
- Compare backward GPU vs CPU time
- Memory overhead measurement
- Scalability across batch sizes (1-256)

## Success Criteria

✅ All backward GPU tests pass  
✅ GPU backward numerical match to CPU (rel error < 1e-4)  
✅ GPU backward 2-3x faster than CPU  
✅ No regression in forward GPU path  
✅ Proper gradient flow to weight optimizers  

## Dependencies

- Phase 5.7: GPU forward caching pattern ✅ (RichardsGlu done)
- Existing GPU kernels: GEMM, Softmax (already available)
- GpuSoftmaxGradientKernel ✅ (from Phase 5.7)

## Next Phase (5.9)

Apply same pattern to SSM (Mamba) backward pass:
- Selective scan backward kernels
- State gradient computation
- Similar 2-3x speedup target

---

**Status**: Ready to start Phase 5.8.1 implementation  
**Estimated Duration**: 3-4 days (Days 6-9 of overall Phase 5.7)
