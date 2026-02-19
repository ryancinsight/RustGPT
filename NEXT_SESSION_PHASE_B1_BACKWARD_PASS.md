# Next Session: Phase B.1 - Backward Pass Weight Caching

## Quick Summary
Apply the same weight caching optimization to backward passes. Expected +5-10% speedup on training.

## Status
- ✅ Forward pass weight caching completed
- ⏳ Backward pass weight caching ready to implement
- ⏳ Kernel fusion (Phase C) waiting

## What to Do

### 1. PolyAttention Backward Pass (src/domain/attention/poly_attention.rs:1700+)

**Current location**: `backward_gpu()` method around line 1700

**Current issue**: Likely also re-uploads weights on every backward pass

**Changes needed**:
```rust
pub fn backward_gpu(&mut self, output_grads: &Array2<f32>) -> Result<...> {
    // ... existing code ...
    
    // ADD THIS:
    self.ensure_gpu_weights(pool, ops)?;
    
    // Replace weight uploads with cached versions:
    // OLD: let wq_buf = pool.upload(...)?;
    // NEW: let gpu_weights = self.gpu_weights.as_ref()?;
    //      use gpu_weights.w_q
}
```

**Expected time**: 10-15 minutes
**Expected speedup**: +5-7%

### 2. RichardsGlu Backward Pass (src/domain/richards/richards_glu.rs:387+)

**Current location**: `backward_gpu()` method around line 387

**Current issue**: Also re-uploads weights

**Changes needed**:
```rust
pub fn backward_gpu(...) -> Result<...> {
    // ... existing code ...
    
    // ADD THIS:
    self.ensure_gpu_cache(pool, ops)?;
    
    // Use cached weights from self.gpu_cache
}
```

**Expected time**: 10-15 minutes
**Expected speedup**: +3-5%

## Verification Checklist

- [ ] Find the backward_gpu() methods in both files
- [ ] Add ensure_gpu_weights/ensure_gpu_cache calls
- [ ] Replace weight uploads with cached references
- [ ] Compile: `cargo check --lib --features gpu-wgpu`
- [ ] Test: `cargo test --lib gpu --features gpu-wgpu`
- [ ] Verify 81/81 GPU tests still pass
- [ ] Build release: `cargo build --release --features gpu-wgpu`

## Code Pattern to Follow

**From forward pass** (already done):
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // 1. Ensure weights cached
    self.ensure_gpu_weights(pool, ops)?;
    
    // 2. Upload input only
    let input_buf = pool.upload(input_slice)?;
    
    // 3. Get cached weights
    let gpu_weights = self.gpu_weights.as_ref()?;
    
    // 4. Use cached weights
    kernel::forward_gpu(
        &mut device,
        &input_buf,
        &gpu_weights.w_q,   // <- cached
        &gpu_weights.w_k,   // <- cached
        // ...
    )?;
}
```

**Apply same pattern to backward passes**

## Files to Examine First

```bash
# View backward pass location
grep -n "pub fn backward_gpu" d:/RustGPT/src/domain/attention/poly_attention.rs
grep -n "pub fn backward_gpu" d:/RustGPT/src/domain/richards/richards_glu.rs

# View what weights they're currently uploading
grep -n "pool.upload" d:/RustGPT/src/domain/attention/poly_attention.rs | head -20
```

## Expected Completion Time

- Reading/understanding: 10 min
- Implementation: 20 min
- Testing: 10 min
- **Total: 40 minutes**

## Testing Commands

```bash
# Quick check
cargo check --lib --features gpu-wgpu

# Full GPU test suite
cargo test --lib gpu --features gpu-wgpu

# Specific backward tests
cargo test --lib backward_gpu --features gpu-wgpu

# Build for deployment
cargo build --release --features gpu-wgpu
```

## Expected Result

After completing B.1:
- ✅ Both forward AND backward passes use cached weights
- ✅ Total GPU memory transfer reduced by 50-60%
- ✅ Training speedup: +5-10% overall
- ✅ All 81 tests still passing

## Then Continue to Phase C

After B.1 is complete, Phase C (kernel fusion) becomes next priority:

**Phase C options**:
1. **RichardsGlu fusion** (easier): Fuse Linear + GLU + Activation
2. **QKV projection fusion** (harder): Combine 3 GEMMs → 1

## Documentation References

- **Phase B completed**: GPU_OPTIMIZATION_PHASE_B_MEMORY_CACHING.md
- **Session summary**: SESSION_GPU_PERFORMANCE_OPTIMIZATION_SUMMARY.md
- **Build guide**: AGENTS.md
- **Optimization roadmap**: PHASE5.6_GPU_OPTIMIZATION_ROADMAP.md

## Success Criteria

✅ Compilation passes
✅ 81/81 tests pass
✅ No GPU memory leaks
✅ Backward pass runs without errors

---

**This is a straightforward optimization. You've got this! Follow the forward pass pattern and apply to backward passes.**
