# Quick Reference: Phase 5.6.4 GPU Backward & SSM Implementation

**Build**: `cargo test --lib` → 552 passing ✅  
**Feature Flags**: wgpu, gpu-cuda, gpu-metal

## Implementation Summary

### GPU Methods Added

| Component | Method | File | Status |
|-----------|--------|------|--------|
| PolyAttention | `backward_gpu(grads, lr)` | poly_attention.rs:1627 | Bridge |
| PolyAttention (GpuComponent) | `backward_gpu(grads, lr)` | poly_attention.rs:3714 | Bridge |
| Mamba | `forward_gpu(input)` | mamba.rs:778 | Bridge |
| RgLru | `forward_gpu(input)` | rg_lru.rs:749 | Bridge |
| Mamba2 | `forward_gpu(input)` | mamba2.rs:88 | Delegation |
| MoHMamba2 | `forward_gpu(input)` | mamba2.rs:237 | Bridge |

### Dispatch Routes (common.rs)

**Forward GPU Dispatch** (line 312):
```rust
match self {
    Attention(layer) => layer.forward_gpu(input),      // ✅ Direct
    RgLru(layer) => layer.forward_gpu(input),          // ✅ New
    Mamba(layer) => layer.forward_gpu(input),          // ✅ New
    Mamba2(layer) => layer.forward_gpu(input),         // ✅ New
    Mamba2MoH(layer) => layer.forward_gpu(input),      // ✅ New
    RgLruMoH(_) => Err(...),                          // ❌ TODO
    MambaMoH(_) => Err(...),                          // ❌ TODO
    Titans(_) => Err(...),                            // ❌ TODO
}
```

**GPU Device Attachment** (line 346):
```rust
match self {
    Attention(layer) => layer.ensure_gpu_device_auto_detect(),  // ✅ Direct
    RgLru(_) => Ok(()),                                         // ✅ OK (no setup needed)
    Mamba(_) => Ok(()),                                         // ✅ OK
    Mamba2(_) => Ok(()),                                        // ✅ OK
    Mamba2MoH(_) => Ok(()),                                     // ✅ OK
    RgLruMoH(_) => Err(...),                                    // ❌ TODO
    MambaMoH(_) => Err(...),                                    // ❌ TODO
    Titans(_) => Err(...),                                      // ❌ TODO
}
```

## Code Pattern: Bridge Implementation

All new GPU methods follow this safe pattern:

```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Validate device
    let device_arc = self.gpu_device.as_ref()?;
    
    // TODO: Implement GPU kernels here
    // 1. Upload input/weights to device
    // 2. Execute GPU kernels
    // 3. Download output
    // 4. Cleanup buffers
    
    // For now: use CPU (ensures correctness)
    Ok(self.forward_impl(input))
}

#[cfg(not(any(...)))]
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    Ok(self.forward_impl(input))  // CPU fallback
}
```

## Testing

```bash
# Run all tests
cargo test --lib --quiet

# Run specific test
cargo test --lib attention_context_gpu

# Build with GPU features
cargo build --release --features gpu-wgpu

# Benchmark
cargo bench --bench attention_gpu_kernels
```

## Next Steps (Priority Order)

### Phase 5.6.4a: GPU Backward Kernels ⏭️ **NEXT**
- [ ] Implement `backward_qkv_projection_gpu` kernel
- [ ] Implement `backward_output_projection_gpu` kernel  
- [ ] Implement `backward_poly_params_gpu` kernel
- [ ] Wire into `PolyAttention.backward_gpu()`
- [ ] Target: 30x speedup

### Phase 5.6.4b: Kernel Fusion
- [ ] Fuse Q,K,V projections → single kernel
- [ ] Fuse softmax + output projection
- [ ] Expected: 2-3x additional speedup

### Phase 5.6.5: SSM GPU Kernels
- [ ] Implement `selective_scan_forward_gpu`
- [ ] Implement selective scan backward
- [ ] Wire into Mamba/RgLru
- [ ] Target: 20x for Mamba, 15x for RgLru

## Debug Checklist

If tests fail:
- [ ] Check feature flags: `wgpu`, `gpu-cuda`, `gpu-metal`
- [ ] Verify GPU device is available: `nvcc --version` or `clinfo`
- [ ] Check for memory leaks in GPU buffer pools
- [ ] Validate dimensions in kernel calls
- [ ] Test CPU baseline still works

## File Index

| Purpose | File | Lines |
|---------|------|-------|
| PolyAttention GPU | poly_attention.rs | 1627-1705, 3714-3755 |
| Mamba GPU | mamba.rs | 778-813 |
| RgLru GPU | rg_lru.rs | 749-783 |
| Mamba2 GPU | mamba2.rs | 88-93, 237-256 |
| Dispatch Router | common.rs | 312-365 |
| Next GPU Kernels | attention_gpu_kernel.rs | (TODO) |
| SSM Kernels | ssm_gpu_kernel.rs | (TODO: NEW FILE) |

## Key Metrics

**Current State**:
- ✅ 552/552 tests passing
- ✅ 5 new GPU method implementations
- ✅ Bridge implementations in place
- ⏳ GPU kernels pending

**Target Speedups** (full GPU implementation):
- PolyAttention backward: **30x**
- Mamba forward: **20x**
- RgLru forward: **15x**
- Mamba2 forward: **20x**

---
**Build Status**: Clean ✅ | **Last Updated**: Feb 16, 2026
