# Quick GPU Test - Next Session

## TL;DR
1. Build: `cargo build --release --features gpu-wgpu`
2. Test: `RUST_LOG=debug cargo run --release --features gpu-wgpu --bin main 2>&1 | grep gpu`
3. Monitor: Task Manager > Performance > GPU (watch utilization %)

## What Was Fixed
- GPU detection now works (was failing silently on Windows due to missing `vulkaninfo` tool)
- Strict no-fallback logging added (will panic with details if GPU fails)
- GPU will be initialized during training start

## Expected Behavior

### If GPU Works ✅
- Logs show: `"RichardsGlu GPU initialization successful"`
- Task Manager GPU % goes up during training
- All logs mention "GPU forward succeeded"

### If GPU Fails ❌
- **Training panics** with error message showing exactly what failed
- Error message tells you what to fix (missing drivers, incompatible GPU, etc.)
- No silent fallback to CPU anymore!

## Performance Expectations

### GPU Utilization by Batch Size
| Batch Size | GPU Util | Status |
|------------|----------|--------|
| 2-4 | 5-15% | Low (kernel overhead dominates) |
| 8-16 | 20-40% | Medium (improving) |
| 32+ | 50%+ | Good (saturated) |

**Recommendation**: Use batch size 32+ for good GPU utilization.

## One-Minute Diagnostic

```bash
# Build
cargo build --release --features gpu-wgpu 2>&1 | tail -3

# Run with debug logs, capture GPU lines
RUST_LOG=debug cargo run --release --features gpu-wgpu --bin main 2>&1 | grep -i "gpu\|backend"

# Expected lines:
# - "GPU features not compiled? NO"  (if WGPU enabled)
# - "Attempting GPU auto-detection for X layers"
# - "RichardsGlu GPU initialization successful" (if GPU available)
# - "GPU forward succeeded" (if GPU working in forward pass)
```

## If GPU Still ~0% Utilization

Check in this order:

1. **Did GPU init succeed?**
   - Look for: `"RichardsGlu GPU initialization successful"`
   - If NOT there, GPU available but initialization failing
   - Check error message from initialization attempt

2. **Is batch size too small?**
   - Default: 4 (too small for GPU)
   - Try: 32+ batch size
   - Code: `llm.train_with_warmup_with_accumulation(..., batch_size=32, ...)`

3. **Check driver/hardware compatibility**
   - Run: `RUST_LOG=trace cargo run ... 2>&1 | grep -i "wgpu\|vulkan\|backend"`
   - Look for any initialization errors

## Files to Know
- GPU detection: `src/domain/compute_backend.rs:253-277`
- GPU dispatch: `src/domain/models/llm.rs:37-108`
- GPU init logging: `src/domain/models/llm.rs:1591-1645`

## Previous Work Reference
See `GPU_DIAGNOSTIC_SESSION.md` for root cause analysis of why GPU wasn't being used before.
