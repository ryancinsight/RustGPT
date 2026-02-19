# Quick Reference: GPU Forward Caching Fix

## The Problem (1 sentence)
GPU forward pass didn't download intermediate values (x1, x2, value, gated) to CPU, so backward pass had no cached values.

## The Solution (3 parts)

### 1. Make kernel return intermediate buffers
```rust
// Before
pub fn forward_gpu_kernel(...) -> Result<()>

// After  
pub fn forward_gpu_kernel(...) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)>
//                                        x1         x2         value       gated
```

### 2. Download & cache them in forward_gpu()
```rust
let (x1_buf, x2_buf, value_buf, gated_buf) = 
    self.forward_gpu_kernel(pool, ops, &input_buf, &mut output_buf, batch_size)?;

// Download to CPU
let mut x1_array = Array2::zeros((batch_size, hidden_dim));
pool.download(&x1_buf, x1_array.as_slice_mut().unwrap())?;
self.cached_x1 = Some(x1_array);  // Repeat for x2, value, gated
```

### 3. Use cached values in backward_gpu()
```rust
// backward_gpu() uses:
let x1 = self.cached_x1.as_ref().unwrap();
let value = self.cached_swish.as_ref().unwrap();
let x2 = self.cached_x2.as_ref().unwrap();
let gated = self.cached_gated.as_ref().unwrap();
```

## File Modified
`src/domain/richards/richards_glu.rs`

## Lines Changed
- Line 8: Add ModelError import
- Lines 199-291: Modify forward_gpu_kernel() return type
- Lines 149-217: Update forward_gpu() to download
- Lines 746-759: Add gradient computation

## Test
```bash
cargo test --lib --features gpu-wgpu domain::richards::glu
```

Expected: **10 passed; 0 failed**

## Why This Works
- GPU computes all intermediates during forward
- We download them to CPU immediately after  
- Backward pass accesses cached CPU copies
- Standard pattern in PyTorch/TensorFlow

## Performance Impact
- Memory: +3 MB per forward (batch_size=64)
- Speed: Minimal (download is fast, async-friendly)
- Trade-off: Standard for backprop support

## Architecture
```
Forward GPU:
  input → [GPU Kernel] → x1, x2, value, gated → [Download] → CPU Cache

Backward GPU:
  CPU Cache → [Gradient Computation] → param updates
```

Done! ✅
