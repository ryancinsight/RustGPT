# Next Session: GPU Kernel Verification & Optimization

## Current Status (Feb 17, 2026)

✅ **COMPLETED**:
- All compilation errors fixed
- Feature flag consolidation (wgpu → gpu-wgpu)
- GPU auto-detection verified (WGPU+Vulkan+NVIDIA initialized successfully)
- All 7 compiler warnings cleaned up
- CPU and GPU builds both passing

## Architecture Overview

### GPU Component Hierarchy
```
TransformerBlock / DiffusionBlock
  ├── attention: PolyAttention
  │   └── forward_gpu() implemented
  ├── temporal_mixing: TemporalMixing
  │   └── SharedTemporalProcessing
  │       └── forward_gpu() implemented
  └── feedforward: SharedFeedforward
      └── RichardsGlu
          └── forward_gpu() implemented
```

### GPU Initialization Flow
1. Components have `set_gpu_device(device)` methods
2. Blocks have `enable_gpu_auto_detect()` method
3. `GpuDevice::auto_detect()` handles backend selection
4. Strict no-fallback semantics (errors if GPU not available when enabled)

### Key GPU Methods Found
- ✅ RichardsGlu::forward_gpu() - IMPLEMENTED
- ✅ PolyAttention::forward_gpu() - IMPLEMENTED
- ✅ SharedTemporalProcessing::forward_gpu() - IMPLEMENTED
- ✅ MixtureOfExperts::forward_gpu() - IMPLEMENTED (x2 variants)
- ✅ Mamba::forward_gpu() - IMPLEMENTED (x2 variants)
- ✅ RgLru::forward_gpu() - IMPLEMENTED (x2 variants)

## Testing Plan for Next Session

### Phase 1: Baseline Verification
```bash
# 1. Build release binary with GPU
cargo build --release --features gpu-wgpu

# 2. Run 10 training iterations
./target/release/main.exe

# 3. Verify NO errors/warnings related to:
#    - Missing GPU kernels
#    - Fallback to CPU
#    - Memory allocation failures
#    - GPU device not found

# Expected output:
# - WGPU initialization messages
# - GPU device detected
# - Training iterations complete
# - No panics or errors
```

### Phase 2: Performance Profiling
```bash
# Measure GPU vs CPU performance
# - Single training step timing
# - Memory usage (GPU memory)
# - Batch sizes that work

# Identify bottlenecks:
# - Memory transfers (H2D, D2H)
# - Kernel launch overhead
# - Synchronization points
```

### Phase 3: Optimization Targets
Based on Phase 2 profiling, optimize:

1. **Memory Transfers**
   - Fuse kernel launches to reduce D2H transfers
   - Use in-place operations where possible
   - Batch multiple operations

2. **Kernel Fusion** 
   - RichardsGlu forward: fuse Linear + GLU + Activation
   - Attention: fuse QKV projection + attention + output

3. **GPU Memory Management**
   - Pre-allocate buffers at startup
   - Reuse buffers across iterations
   - Implement memory pooling

## Files to Monitor

### Core GPU Implementations
- src/domain/richards/richards_glu.rs (forward_gpu at line 151)
- src/domain/attention/poly_attention.rs (forward_gpu at line 1615)
- src/domain/layers/components/attention_gpu_kernel.rs
- src/domain/layers/components/gpu_gemm_kernels.rs

### GPU Infrastructure
- src/domain/compute/gpu_device.rs (auto_detect at line 171)
- src/domain/compute/unified_gpu_executor.rs
- src/domain/compute/unified_gpu_buffer_pool.rs

### Build Configuration
- Cargo.toml (gpu-wgpu, gpu-cuda, gpu-metal features)
- AGENTS.md (build commands)

## Known Working Features

✅ Automatic GPU detection (GpuDevice::auto_detect)
✅ Vulkan backend initialization
✅ Device capability querying
✅ Buffer allocation/deallocation
✅ Strict no-fallback semantics
✅ Component GPU attachment (set_gpu_device)

## Potential Issues to Watch

⚠️ **GPU Memory Leaks**: Monitor pool allocations
⚠️ **Synchronization**: Ensure proper GPU->CPU sync before reading results
⚠️ **Feature Flags**: Keep gpu-wgpu/gpu-cuda/gpu-metal consistent
⚠️ **Fallback Code**: Currently errors on missing GPU - make sure this is tested
⚠️ **Batch Size Scaling**: GPU memory is limited - batch size affects speed

## Optimization Opportunities

1. **High Priority**
   - Implement kernel fusion for RichardsGlu
   - Optimize GEMM parameters for GPU
   - Pre-allocate workspace buffers

2. **Medium Priority**
   - Implement persistent GPU memory pool
   - Fuse attention QKV projections
   - Optimize MoE routing on GPU

3. **Low Priority**
   - Implement async GPU operations
   - Support multi-GPU training
   - Implement gradient accumulation on GPU

## Success Criteria

✅ Training completes 100 iterations with GPU enabled
✅ No crashes, panics, or fallback to CPU
✅ GPU memory stays within bounds (< 6GB for small model)
✅ Training iterations < 100ms (baseline for optimization)
✅ Loss values reasonable and decreasing

## Next Immediate Action

**RUN QUICK TRAINING TEST**:
```bash
cargo run --bin main --features gpu-wgpu --release 2>&1 | head -500
```

**SUCCESS INDICATORS**:
- No compilation errors
- No GPU initialization errors
- Training loop executes without panics
- GPU kernels execute (no fallback messages)
- Memory allocation succeeds

**FAILURE INDICATORS** (troubleshoot):
- Missing GPU device error → Vulkan not installed
- Kernel compilation error → WGPU shader issue
- Memory allocation failed → GPU memory too small
- Fallback to CPU → Expected behavior if GPU not available
