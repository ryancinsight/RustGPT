# Phase 5.6.4 - Immediate Next Steps for Full GPU Implementation

**Current Status**: Bridge implementations complete. All 552 tests passing. Ready for GPU kernel implementation.

## Summary of Bridge Implementations (Completed)

All temporal mixing layer types now have `forward_gpu()` methods:

| Layer | File | Status | Bridge Impl | Target Speedup |
|-------|------|--------|-------------|-----------------|
| PolyAttention | poly_attention.rs | ✅ | Uses CPU forward (GPU weights cached) | 30x |
| Mamba | mamba.rs | ✅ | Delegates to forward_mamba2 | 20x |
| Mamba2 | mamba2.rs | ✅ | Delegates to inner Mamba | 20x |
| RgLru | rg_lru.rs | ✅ | Uses CPU forward_into | 15x |
| MoHMamba2 | mamba2.rs | ✅ | Uses CPU forward_into | 20x |
| Backward (PolyAttention) | poly_attention.rs | ✅ | Falls back to CPU backward | 30x |

## Next Task: Full GPU Backward Kernels for PolyAttention

**Priority**: Phase 5.6.4a (blocking for training optimization)

### Step 1: Implement Gradient Projection Kernels

**File**: `src/domain/layers/components/attention_gpu_kernel.rs`

Add these GPU kernel signatures:

```rust
pub fn backward_qkv_projection_gpu(
    device: &mut GpuDevice,
    output_grads: &GpuBuffer,        // [batch, seq, embed]
    w_q: &GpuBuffer,                 // [embed, embed] (transposed)
    w_k: &GpuBuffer,                 // [embed, embed] (transposed)
    w_v: &GpuBuffer,                 // [embed, embed] (transposed)
    input: &GpuBuffer,               // [batch, seq, embed] (cached from forward)
    attention_weights: &GpuBuffer,   // [batch, heads, seq, seq] (cached from forward)
    params: &AttentionParams,
) -> Result<(GpuBuffer, GpuBuffer)>; // (grad_input, grad_weights)

pub fn backward_output_projection_gpu(
    device: &mut GpuDevice,
    attention_output: &GpuBuffer,    // [batch, heads, seq, head_dim]
    output_grads: &GpuBuffer,        // [batch, seq, embed]
    w_out: &GpuBuffer,               // [embed, embed] (transposed)
    params: &AttentionParams,
) -> Result<GpuBuffer>; // grad_w_out
```

### Step 2: Implement SSM Selective Scan on GPU

**File**: `src/domain/layers/components/ssm_gpu_kernel.rs` (NEW)

```rust
pub fn selective_scan_forward_gpu(
    device: &mut GpuDevice,
    input: &GpuBuffer,      // [batch, seq, dim]
    dt: &GpuBuffer,         // [batch, seq, dim]
    B: &GpuBuffer,          // [batch, seq, state_dim]
    C: &GpuBuffer,          // [batch, seq, state_dim]
    A: &GpuBuffer,          // [1, dim, state_dim]
    params: &SsmParams,
) -> Result<(GpuBuffer, GpuBuffer)>; // (output, state)

pub fn selective_scan_backward_gpu(
    device: &mut GpuDevice,
    // ... (gradient computation)
) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer)>; // (grad_input, grad_dt, grad_state)
```

### Step 3: Wire Backward Kernels into PolyAttention.backward_gpu()

**File**: `src/domain/attention/poly_attention.rs` (Line 1627)

Replace CPU fallback with GPU kernel calls:

```rust
pub fn backward_gpu(&mut self, grads: &Array2<f32>, lr: f32) -> Result<Array2<f32>> {
    // 1. Upload gradients
    let grad_buf = pool.upload(grads.as_slice())?;
    
    // 2. Call backward_qkv_projection_gpu
    let (grad_input, grad_weights) = attention_gpu_kernel::backward_qkv_projection_gpu(
        &mut device,
        &grad_buf,
        &cached_wq_buf,
        &cached_wk_buf,
        &cached_wv_buf,
        &cached_input_buf,
        &cached_attention_weights_buf,
        &params,
    )?;
    
    // 3. Download grad_input
    let mut input_grads = Array2::zeros(input.dim());
    pool.download(&grad_input, input_grads.as_slice_mut().unwrap())?;
    
    // 4. Apply gradient updates via GPU optimizers (TODO)
    
    Ok(input_grads)
}
```

## Optimization Opportunities (Phase 5.6.4b)

### Fused Kernels
- Fuse Q,K,V projections → single kernel with 3 output branches
- Fuse softmax + output projection → reduce memory loads
- Expected improvement: 2-3x additional speedup

### Memory Optimization
- Use workspace-managed pools for all intermediate buffers
- Implement buffer reuse across forward/backward
- Reduce peak memory usage by 40-50%

### Kernel Launch Optimization
- Reduce number of kernel launches (currently ~10 per forward pass)
- Target: 4-5 launches for full forward+backward cycle

## Files to Modify

**Primary**:
- `src/domain/layers/components/attention_gpu_kernel.rs` - Add backward kernels
- `src/domain/attention/poly_attention.rs` - Wire backward kernels into backward_gpu

**Secondary**:
- `src/domain/layers/components/ssm_gpu_kernel.rs` - NEW file for selective scan
- `src/domain/layers/ssm/mamba.rs` - Wire forward_gpu to GPU kernels
- `src/domain/layers/ssm/rg_lru.rs` - Wire forward_gpu to GPU kernels

## Testing Strategy

1. **Unit Tests** (SSM GPU kernels):
   ```rust
   #[test]
   fn test_backward_qkv_projection_gpu() { ... }
   
   #[test]
   fn test_selective_scan_forward_gpu() { ... }
   ```

2. **Integration Tests**:
   ```rust
   #[test]
   fn test_polyattention_backward_gpu_correctness() {
       // Verify backward_gpu produces same gradients as CPU backward
       // within numerical tolerance (1e-5 for f32)
   }
   ```

3. **Performance Tests**:
   ```bash
   cargo bench --bench attention_gpu_kernels -- --verbose
   ```

## Success Criteria

✅ All GPU backward kernels implemented and tested  
✅ 552 existing tests still passing  
✅ New GPU tests added (>30 tests for backward paths)  
✅ Backward pass speedup ≥15x vs CPU (conservative estimate)  
✅ No GPU memory leaks in extended training runs  

## Timeline Estimate

- Phase 5.6.4a (GPU Backward Kernels): 4-6 hours
- Phase 5.6.4b (Fused Kernels + Optimization): 3-4 hours
- Phase 5.6.5 (SSM GPU Implementation): 6-8 hours
