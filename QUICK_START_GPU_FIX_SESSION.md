# Quick Start: GPU Utilization Fix Session

## Status
**Build**: ✅ Complete (`cargo build --release --features gpu-wgpu` succeeds)
**GPU Mode**: Strict no-fallback (panics on GPU errors)
**Next Action**: Test GPU is actually being used during training

## What Changed
1. **Strict GPU dispatch** in `src/domain/models/llm.rs`: No more silent CPU fallback when GPU fails
2. **Fixed imports**: Added missing `ModelError` imports across GPU modules
3. **Removed invalid GPU calls**: Fixed non-existent method in `temporal_processing.rs`
4. **GPU auto-init**: Training loop automatically enables GPU for RichardsGlu and PolyAttention layers

## How to Test GPU Usage

### Method 1: Simple Panic Test
```bash
# Build with GPU
cargo build --release --features gpu-wgpu

# Run main binary
cargo run --release --features gpu-wgpu --bin main

# If GPU works: No panic, training proceeds normally
# If GPU fails: Panics immediately with detailed error message showing the issue
```

### Method 2: Monitor GPU During Training
On Windows (Task Manager):
1. Run: `cargo run --release --features gpu-wgpu --bin main`
2. Open Task Manager > Performance > GPU
3. Watch utilization percentage during training

On Linux (nvidia-smi):
```bash
watch -n 1 nvidia-smi
```

### Method 3: Enable Debug Logging
```bash
RUST_LOG=debug cargo run --release --features gpu-wgpu --bin main 2>&1 | grep -i gpu
```

## If GPU Fails
The panic message will tell you exactly which layer failed and why. Example:
```
panicked at 'RichardsGlu GPU forward failed (GPU enabled, no fallback): Backend { 
    message: "WGPU device not initialized" 
}'
```

Then check:
1. Is GPU driver installed and working?
2. Does wgpu support this GPU? (Run a WGPU diagnostic if needed)
3. Check `GpuDevice::auto_detect()` in `src/domain/layers/components/unified_gpu_backend.rs`

## Expected GPU Behavior
- **With batch_size >= 32**: Good GPU utilization (30-50%+)
- **With batch_size = 4** (default): Low GPU utilization (5-10%) due to kernel launch overhead
- **Solution**: Increase batch size or enable gradient accumulation for larger effective batches

## Compilation Check
```bash
# Verify build without running
cargo check --lib --features gpu-wgpu
```

## Files Modified
- `src/domain/models/llm.rs` - Strict GPU dispatch
- `src/domain/compute/unified_gpu_*.rs` - Import fixes
- `src/domain/richards/richards_glu.rs` - Import fix
- `src/domain/layers/components/*.rs` - Multiple compilation fixes

## Next Session Tasks
1. [ ] Run training and verify GPU utilization in Task Manager
2. [ ] If GPU underutilized, increase batch size to 32+
3. [ ] Implement gradient accumulation on GPU for larger effective batches
4. [ ] Profile memory transfers vs kernel execution time
5. [ ] Consider kernel fusion optimizations if still needed

## Reference Docs
- GPU Implementation Status: Previous thread context
- Phase B Optimization: Weight caching, kernel fusion, on-device intermediates
