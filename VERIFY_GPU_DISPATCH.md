# GPU Dispatch Verification Guide

## What Was Fixed
The training pipeline now dispatches to GPU kernels for compute-heavy layers:
- **RichardsGlu** → `forward_gpu()` 
- **PolyAttention** → `forward_gpu()`
- **TransformerBlock** → delegates to internal components
- **DiffusionBlock** → delegates to internal components

## Building with GPU Support

### WGPU (Vulkan/Metal)
```bash
cargo build --release --features gpu-wgpu
```

### CUDA (NVIDIA GPUs)
```bash
cargo build --release --features gpu-cuda
```

### All GPU Backends
```bash
cargo build --release --features gpu-all
```

## Running Training

### Start Training with GPU
```bash
cargo run --bin main --features gpu-wgpu -- \
  --pretrain-epochs 2 \
  --instruction-epochs 1 \
  --pretrain-batch-size 16
```

### Key Parameters for GPU Utilization
```bash
# Increase batch size for better GPU utilization
--pretrain-batch-size 32      # Default: 4 (too small for GPU)
--instruction-batch-size 32   # Default: 4

# Enable gradient accumulation for larger effective batch
--pretrain-gradient-accumulation-steps 4
--instruction-gradient-accumulation-steps 4
```

**Example: Good GPU utilization**
```bash
cargo run --bin main --features gpu-wgpu -- \
  --pretrain-epochs 2 \
  --instruction-epochs 1 \
  --pretrain-batch-size 32 \
  --pretrain-gradient-accumulation-steps 4 \
  --instruction-batch-size 32 \
  --instruction-gradient-accumulation-steps 4
```

## Monitoring GPU Usage

### Windows (Task Manager)
1. Open **Task Manager**
2. Click **Performance** tab
3. Select **GPU** in left sidebar
4. Watch for:
   - **GPU** line: Should show >50% during training
   - **Memory**: Should show active usage
   - **Engine**: Should show "3D" activity

### Linux (nvidia-smi)
```bash
# Real-time monitoring
nvidia-smi -l 1

# Detailed output
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader -l 1
```

### macOS (Activity Monitor)
1. Open **Activity Monitor**
2. Click **Window** → **GPU History**
3. Look for GPU utilization during training

## Logging GPU Initialization

The training logs will show GPU initialization:

```
INFO  GPU initialization for training complete
```

If you see this message, GPU was found and initialized. If you don't see it:
- GPU features may not be compiled in
- No GPU detected on system
- GPU initialization failed (check stderr)

## Expected Behavior

### With GPU Enabled ✓
```
cargo run --bin main --features gpu-wgpu
```
- Training logs show "GPU initialization for training complete"
- Task Manager GPU % jumps to 50-90% during training
- Training speed: ~2-5x faster than CPU-only

### Without GPU Flags
```
cargo run --bin main
```
- No GPU initialization message
- Task Manager GPU % stays ~0%
- Training uses CPU only (slow)

## Troubleshooting Low GPU Utilization

### 1. Verify GPU Detected
```bash
# WGPU backends
RUST_LOG=debug cargo run --bin main --features gpu-wgpu 2>&1 | grep -i gpu

# CUDA
nvidia-smi
```

### 2. Check Batch Size
GPU kernels have high per-call overhead. Small batches won't saturate GPU:
```bash
# Too small (won't use GPU fully)
--pretrain-batch-size 4

# Recommended (good GPU utilization)
--pretrain-batch-size 32
```

### 3. Verify GPU Code Path
Add temporary debug logging in `src/domain/models/llm.rs`:
```rust
fn try_forward_gpu_richards(...) {
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
    {
        if let Ok(output) = layer.forward_gpu(input) {
            eprintln!("✓ RichardsGlu GPU forward");  // Add this
            return output;
        }
    }
    layer.forward(input)
}
```

### 4. Check GPU Memory
If GPU memory is full, kernels will fail and fallback to CPU:
```bash
nvidia-smi  # Check Memory-Usage
# Or Windows Task Manager → Performance → GPU Memory
```

## Files Modified
1. `src/domain/models/llm.rs`
   - Lines 32-70: GPU dispatch helpers
   - Lines 1520-1541: GPU initialization in `train_with_warmup_with_accumulation()`
   - Lines 2875-2897: GPU initialization in `train_diffusion_ce_with_accumulation()`
   - Lines 1104-1145: Layer dispatch to GPU forward

## Performance Expectations

### RichardsGlu (Fused GLU Kernel)
- **CPU-only**: ~500-1000 ms per forward
- **GPU**: ~50-200 ms per forward
- **Expected**: 2-5x speedup

### PolyAttention 
- **CPU-only**: ~1000-3000 ms per forward
- **GPU**: ~100-500 ms per forward
- **Expected**: 5-10x speedup

### Full Training
- **CPU-only**: ~24 hours for 2 epochs (small model)
- **GPU**: ~4-6 hours for 2 epochs
- **Expected**: 4-6x overall speedup

## Known Limitations

### Phase B.1 (Not Yet Implemented)
**Backward pass** still re-uploads weights every iteration:
- Estimated 10-15% performance improvement available
- Backward pass weight caching is planned

### Phase C (Not Yet Implemented)
**Kernel Fusion** can further optimize:
- Fuse RichardsGlu: Linear + GLU + Activation → single kernel
- Fuse QKV Projections: 3 GEMMs → 1 fused GEMM
- Estimated 15-25% additional improvement

## Next Steps (If Needed)

If GPU utilization is still low:
1. Increase batch size significantly (32-64)
2. Check `nvidia-smi` for kernel launches
3. Enable profiling with `RUST_LOG=debug`
4. File issue with GPU diagnostics

## Quick Test
```bash
# Build with GPU
cargo build --release --features gpu-wgpu

# Run with good GPU settings
./target/release/main \
  --pretrain-epochs 1 \
  --instruction-epochs 0 \
  --pretrain-batch-size 32 \
  --pretrain-gradient-accumulation-steps 2

# Monitor in parallel terminal
# Windows: Task Manager → Performance → GPU
# Linux: watch nvidia-smi
# macOS: Activity Monitor → Window → GPU History
```

Expected result: GPU utilization jumps to 50-90% during training.
