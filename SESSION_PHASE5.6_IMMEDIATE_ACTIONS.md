# Session: Phase 5.6 Immediate Actions

**Start Time**: 2026-02-15  
**Thread**: @T-019c6409-39ff-73a9-a641-8b3d1bccb44b  
**Objective**: Begin GPU backend consolidation with RichardsGLU fused kernel implementation

---

## Priority Order (Do in This Order)

### 1. IMMEDIATE: Implement RichardsGLU Fused Kernel Execute Function

**File**: `src/domain/layers/components/fused_kernels_module.rs`  
**Function**: `richards_glu_fused::execute()` (lines 108-123)

**What to do**:
- Replace the TODO placeholder with the full implementation
- Follow the step-by-step guide in `PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md`
- Key steps:
  1. Lock GPU device
  2. Validate input dimensions
  3. Allocate GPU buffers for input, weights, intermediates
  4. Pass 1: Compute x1 @ W1, x2 @ W2, apply Richards + sigmoid, element-wise multiply
  5. Pass 2: Compute gated @ W_out, add residual
  6. Download result and cleanup

**Testing**:
```bash
cargo test --test gpu_shared_components_phase56 richards_glu_fused --release
```

**Success Criteria**:
- Compilation succeeds with all GPU features
- Output shape matches input shape
- Values are finite (no NaN/Inf)
- No GPU memory leaks

---

### 2. NEXT: Implement forward_gpu() in SharedFeedforward

**File**: `src/domain/layers/components/feedforward.rs`  
**Method**: Add `forward_gpu()` implementation

**What to do**:
- Extract RichardsGlu weights and parameters
- Build `RichardsGluFusedKernelParams` from component config
- Call `richards_glu_fused::execute()`
- Handle MoE case with TODO for now

**Code Template**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let gpu_device = require_gpu_or_error(&self.gpu_device, "forward_gpu")?;
    
    match &self.feedforward {
        FeedForwardVariant::RichardsGlu(richards) => {
            // Extract params and call fused kernel
            // (See PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md for details)
        }
        FeedForwardVariant::MixtureOfExperts(_) => {
            Err(ModelError::NotImplemented {
                message: "GPU MoE not yet implemented".to_string(),
            })
        }
    }
}
```

**Testing**:
```bash
cargo test --lib shared_feedforward::tests::test_forward_gpu
```

---

### 3. VERIFY: Check GPU Device Auto-Detection

**File**: `src/domain/compute/gpu_device.rs`  
**Method**: `GpuDevice::auto_detect()` (lines 155-182)

**Status**: Already implemented ✓

**What to verify**:
- Auto-detection priority: CUDA > Metal > Vulkan (verify in code)
- Strict no-fallback: Returns error if no GPU (not CPU fallback)
- Test on your machine:

```bash
cargo test --lib gpu_device::tests::gpu_device_strict_no_fallback --release
```

**Expected Output**:
```
GPU detected: <backend_name> (CUDA|Metal|Vulkan)
```

OR (if no GPU):
```
No GPU available (expected on CPU-only systems): ...
```

---

### 4. CONSOLIDATION: Review Shared Component Trait Integration

**Files**:
- `src/domain/layers/components/attention_context.rs` - lines 58-62
- `src/domain/layers/components/temporal_processing.rs` - lines 47-51
- `src/domain/layers/components/feedforward.rs` - lines 57-61

**What to verify**:
- All three components have `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- All components should implement `GpuComponent` trait via macro
- GPU device can be attached via `enable_gpu_auto_detect()`

**Check existing implementations**:
```bash
cargo test --lib gpu_component --release
```

---

### 5. OPTIMIZATION: Add CPU/GPU Activation Kernels (Optional for Phase 5.6.3)

**For Later**: GPU-side Richards curve and Sigmoid kernels

**Files to create** (not in this session):
- `src/domain/compute/cuda/richards_kernel.cu`
- `src/domain/compute/metal/richards_kernel.metal`
- `src/domain/compute/wgpu_ops.rs` - Add compute shader

---

## Build Commands

### Build with GPU Features

```bash
# WGPU (Vulkan/Metal/DX12 on Windows)
cargo build --release --features gpu-wgpu

# CUDA (NVIDIA GPU on Linux/Windows)
cargo build --release --features gpu-cuda

# Metal (Apple Silicon/Intel Mac)
cargo build --release --features gpu-metal

# All GPU backends
cargo build --release --features gpu-all
```

### Run Tests

```bash
# All GPU tests (requires GPU)
cargo test --release --features gpu-wgpu -- --nocapture

# Specific test
cargo test --lib gpu_device::tests --release

# GPU shared components (if exists)
cargo test --test gpu_shared_components_phase56 --release
```

---

## Compilation Verification

Before starting implementation:

```bash
# Check compilation with all features
cargo check --all-features

# Build just the target modules
cargo build --release --features gpu-wgpu \
  -p rustgpt

# If any errors, fix immediately before proceeding
```

---

## File Status Check

### Files That Should Already Exist

```bash
# Verify these files exist and have content
ls -la src/domain/compute/gpu_device.rs      # ✓ Auto-detection
ls -la src/domain/compute/gpu_component.rs   # ✓ Trait definition
ls -la src/domain/layers/components/unified_gpu_backend.rs  # ✓ Entry point
ls -la src/domain/layers/components/fused_kernels_module.rs # ✓ Stubs exist
ls -la src/domain/layers/components/unified_buffer_pool.rs  # ✓ Memory management
```

### Files You'll Be Editing

```bash
# These will be edited in this session
src/domain/layers/components/fused_kernels_module.rs      # Implement execute()
src/domain/layers/components/feedforward.rs              # Add forward_gpu()
```

---

## Session Milestones

✓ **Milestone 1: Planning Complete**
- [x] Review GPU backend architecture
- [x] Understand auto-detection flow
- [x] Understand shared component integration
- [x] Create consolidation plan document

**Milestone 2: RichardsGLU Implementation** (Current)
- [ ] Implement `fused_kernels_module.rs::execute()`
- [ ] Add `SharedFeedforward::forward_gpu()`
- [ ] Verify compilation with all GPU features
- [ ] Run basic GPU tests

**Milestone 3: Testing & Validation** (Next)
- [ ] Unit test for correctness
- [ ] Benchmark against target (2ms for 1K batch)
- [ ] Memory efficiency validation
- [ ] Integration test with full component

**Milestone 4: Consolidation** (Follow-up)
- [ ] PolyAttention fused kernel
- [ ] Mamba selective scan kernel
- [ ] AttentionContext GPU ops
- [ ] Full component integration

---

## Debugging Quick Reference

### GPU Not Detected
```bash
# Verify your system has GPU
nvidia-smi         # NVIDIA CUDA
metal --version    # Apple Metal
vulkan-info        # Vulkan (Windows/Linux)

# Recompile with correct feature flag
cargo clean
cargo build --release --features gpu-cuda  # Replace gpu-cuda with your GPU type
```

### Compilation Errors
```bash
# Clear build cache
cargo clean

# Check what went wrong
cargo check --release 2>&1 | head -20

# Check specific module
cargo check -p rustgpt --release

# If it's a linking issue, check GPU libraries are installed
# - CUDA: nvcc --version
# - Metal: Already available on macOS
# - Vulkan: vulkansdk
```

### Runtime Errors in Tests
```bash
# Run with backtrace for more details
RUST_BACKTRACE=1 cargo test --release --lib gpu_device -- --nocapture

# Check error message for specific backend issue
# Error message will indicate: "CUDA not available", "Metal not supported", etc.
```

---

## Documentation Links

- **Consolidation Plan**: `PHASE5.6_GPU_BACKENDS_CONSOLIDATION_PLAN.md`
- **RichardsGLU Guide**: `PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md`
- **GPU Device Code**: `src/domain/compute/gpu_device.rs` (lines 155-182)
- **GpuComponent Trait**: `src/domain/compute/gpu_component.rs` (lines 41-87)
- **Original Thread**: https://ampcode.com/threads/T-019c6409-39ff-73a9-a641-8b3d1bccb44b

---

## Success Criteria for This Session

1. ✓ RichardsGLU fused kernel `execute()` function fully implemented
2. ✓ `SharedFeedforward::forward_gpu()` method added
3. ✓ Compilation succeeds with all GPU features
4. ✓ GPU auto-detection works (CUDA > Metal > Vulkan)
5. ✓ Basic tests pass (no crashes, shapes correct)
6. ✓ Memory managed correctly (no leaks)
7. ✓ Strict no-fallback enforced (GPU errors if unavailable)

---

## Time Estimates

- **RichardsGLU Implementation**: 45 minutes
- **SharedFeedforward Integration**: 30 minutes
- **Compilation & Testing**: 30 minutes
- **Documentation & Cleanup**: 15 minutes

**Total**: ~2 hours for this session

---

## Next Session Quickstart

If continuing in next session:

```bash
# Check status
git status

# Build and test
cargo build --release --features gpu-all
cargo test --release

# If everything passes, move to PolyAttention fused kernel
# File: src/domain/layers/components/fused_kernels_module.rs (lines 135+)
```

---

## Notes

- **GPU Detection**: Strict no-fallback means GPU operations error explicitly if no GPU available (good for troubleshooting)
- **Buffer Management**: All buffers should be deallocated at end of operation (check for leaks with valgrind/metal debugger)
- **Data Transfer**: Minimize CPU↔GPU transfers; keep data on GPU between operations
- **Feature Flags**: Build with correct GPU feature for your hardware
  - Windows: `gpu-wgpu` (usually works for most systems)
  - Linux: `gpu-cuda` (if NVIDIA) or `gpu-wgpu` (Vulkan)
  - macOS: `gpu-metal` (native) or `gpu-wgpu` (fallback)

---

**Ready to implement? Start with Step 1 above!**
