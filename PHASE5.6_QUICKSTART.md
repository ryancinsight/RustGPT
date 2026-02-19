# Phase 5.6: GPU Consolidation - Quick Start

**Status**: Ready to Begin  
**Entry Point**: Follow the steps below in order  
**Time Estimate**: 2-3 hours for this session

---

## Prerequisites Check

Before starting, verify your system and environment:

```bash
# Check Rust version (1.85+ required)
rustc --version

# Check cargo works
cargo --version

# Check GPU drivers (if available)
nvidia-smi          # NVIDIA CUDA
metal --version     # Apple Metal
vulkan-info         # Vulkan
```

---

## Step 1: Read Documentation (15 min)

Read these in order to understand what we're building:

1. **This file** (`PHASE5.6_QUICKSTART.md`) - You're reading it
2. `SESSION_PHASE5.6_CONSOLIDATION_SUMMARY.md` - Overview of Phase 5.6
3. `PHASE5.6_GPU_BACKENDS_CONSOLIDATION_PLAN.md` - Full consolidation strategy
4. `SESSION_PHASE5.6_IMMEDIATE_ACTIONS.md` - Priority action items

---

## Step 2: Understand Current Architecture (15 min)

Read the source code to understand existing structure:

```bash
# GPU Device (auto-detection logic)
cat src/domain/compute/gpu_device.rs | head -200

# GpuComponent Trait (all components implement this)
cat src/domain/compute/gpu_component.rs | head -150

# Fused Kernels Stubs (what we're implementing)
cat src/domain/layers/components/fused_kernels_module.rs | head -200

# Unified GPU Backend (entry point)
cat src/domain/layers/components/unified_gpu_backend.rs | head -150
```

---

## Step 3: Verify Compilation (10 min)

Test that your environment can compile with GPU features:

```bash
# Check compilation
cargo check --all-features

# If errors, identify and fix them
# Most likely issues:
# 1. GPU library not installed (CUDA, Metal, Vulkan SDK)
# 2. Feature flags not matching your system
# 3. Rust edition mismatch (need 1.85+)
```

**Troubleshooting**:
```bash
# If CUDA error, install CUDA toolkit
# If Metal error, you're not on macOS (use gpu-wgpu instead)
# If Vulkan error, install Vulkan SDK

# Try WGPU (most portable)
cargo check --features gpu-wgpu
```

---

## Step 4: Build with GPU Features (10 min)

Choose the right feature flag for your system:

```bash
# Option A: WGPU (Works on Windows/Linux/Mac with Vulkan)
cargo build --release --features gpu-wgpu

# Option B: CUDA (NVIDIA GPU, Linux/Windows)
cargo build --release --features gpu-cuda

# Option C: Metal (Apple Silicon/Intel Mac)
cargo build --release --features gpu-metal

# Option D: All GPU backends (if all dependencies available)
cargo build --release --features gpu-all
```

**Success**: Build completes with no errors.

---

## Step 5: Run GPU Detection Test (5 min)

Verify GPU auto-detection works:

```bash
cargo test --lib gpu_device::tests::gpu_device_strict_no_fallback --release

# Expected output (one of these):
# "GPU detected: CUDA (cuda)" ← Success on NVIDIA
# "GPU detected: Metal (metal)" ← Success on Apple Silicon/Intel Mac
# "GPU detected: Vulkan (vulkan)" ← Success with Vulkan drivers
# "No GPU available" ← Expected on CPU-only systems
```

---

## Step 6: Begin Implementation

Now follow `SESSION_PHASE5.6_IMMEDIATE_ACTIONS.md` in priority order:

### Priority 1: RichardsGLU Fused Kernel (45 min)

**File**: `src/domain/layers/components/fused_kernels_module.rs`

**What**: Replace lines 108-123 (the TODO) with full implementation

**Guide**: Read `PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md` for detailed steps

**Quick checklist**:
- [ ] Lock GPU device
- [ ] Validate input dimensions
- [ ] Allocate GPU buffers (input, weights, intermediates, output)
- [ ] Pass 1: Compute x1=x@W1, x2=x@W2, apply Richards curve, sigmoid, element-wise mult
- [ ] Pass 2: Compute output=gated@W_out, add residual
- [ ] Download result and deallocate buffers

**Test**:
```bash
cargo build --release --features gpu-wgpu
cargo test --lib richards_glu_fused --release
```

### Priority 2: SharedFeedforward GPU Support (30 min)

**File**: `src/domain/layers/components/feedforward.rs`

**What**: Add `forward_gpu()` method that extracts RichardsGlu params and calls fused kernel

**Quick template**:
```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let gpu_device = require_gpu_or_error(&self.gpu_device, "forward_gpu")?;
    
    match &self.feedforward {
        FeedForwardVariant::RichardsGlu(richards) => {
            // Extract weights, build RichardsGluFusedKernelParams
            // Call richards_glu_fused::execute()
            // Return result
        }
        FeedForwardVariant::MixtureOfExperts(_) => {
            Err(ModelError::NotImplemented {
                message: "GPU MoE not yet implemented".to_string(),
            })
        }
    }
}
```

**Test**:
```bash
cargo test --lib shared_feedforward --release
```

### Priority 3: Verify & Test (30 min)

**Compilation test**:
```bash
cargo check --release --features gpu-all
cargo build --release --features gpu-all
```

**Unit tests**:
```bash
cargo test --release --lib gpu_device
cargo test --release --lib gpu_component
cargo test --release --lib richardson_glu_fused
```

**Integration test** (if exists):
```bash
cargo test --release --test gpu_shared_components_phase56
```

---

## Success Criteria

✓ **This Session Complete When**:
1. RichardsGLU fused kernel implemented
2. SharedFeedforward::forward_gpu() method added
3. Compilation succeeds with all GPU features
4. GPU auto-detection tests pass
5. No compilation warnings
6. Documentation updated

---

## If You Get Stuck

### Compilation Error?
1. Check error message carefully
2. Verify GPU library installed (`nvidia-smi`, `metal --version`, `vulkan-info`)
3. Try simpler feature flag: `cargo build --features gpu-wgpu`
4. Check Rust version: `rustc --version` (need 1.85+)

### GPU Not Detected?
1. Verify your system has GPU
2. Check drivers are installed
3. Recompile with correct feature flag for your GPU

### Tests Failing?
1. Run with backtrace: `RUST_BACKTRACE=1 cargo test --release`
2. Check error message for specific GPU issue
3. Verify input dimensions match kernel expectations
4. Check GPU memory isn't exhausted

### Implementation Questions?
1. See `PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md` for step-by-step details
2. Check existing kernel code in `src/domain/compute/cuda/` or `src/domain/compute/metal/`
3. Review `gpu_device.rs` for similar operations

---

## Files You'll Work With

### Will Modify
- `src/domain/layers/components/fused_kernels_module.rs` - RichardsGLU kernel
- `src/domain/layers/components/feedforward.rs` - Add forward_gpu()

### Will Reference (Don't modify yet)
- `src/domain/compute/gpu_device.rs` - Already implemented ✓
- `src/domain/compute/gpu_component.rs` - Already implemented ✓
- `src/domain/layers/components/unified_gpu_backend.rs` - Already implemented ✓
- `src/domain/layers/components/unified_buffer_pool.rs` - Already implemented ✓

---

## Command Cheatsheet

```bash
# Build with GPU features
cargo build --release --features gpu-wgpu

# Check compilation
cargo check --features gpu-wgpu

# Run all tests
cargo test --release --features gpu-wgpu

# Run specific test
cargo test --lib gpu_device --release

# Format code
cargo fmt

# Lint code
cargo clippy --all-targets

# Clean build
cargo clean
```

---

## Documentation Map

```
d:/RustGPT/
├── PHASE5.6_QUICKSTART.md ← You are here
├── PHASE5.6_GPU_BACKENDS_CONSOLIDATION_PLAN.md ← Architecture & strategy
├── PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md ← Implementation details
├── SESSION_PHASE5.6_IMMEDIATE_ACTIONS.md ← Priority action items
├── SESSION_PHASE5.6_CONSOLIDATION_SUMMARY.md ← Full overview
└── src/domain/
    ├── compute/
    │   ├── gpu_device.rs ← Auto-detection (done)
    │   ├── gpu_component.rs ← Trait definition (done)
    │   └── gpu_ops.rs ← Matrix operations interface
    └── layers/components/
        ├── fused_kernels_module.rs ← TODO: Implement here
        ├── feedforward.rs ← TODO: Add forward_gpu() here
        ├── unified_gpu_backend.rs ← Entry point (done)
        └── unified_buffer_pool.rs ← Memory management (done)
```

---

## Time Breakdown

- **Documentation**: 15 min (already done for you)
- **Architecture Understanding**: 15 min
- **Compilation Verification**: 10 min
- **GPU Detection Test**: 5 min
- **RichardsGLU Implementation**: 45 min
- **SharedFeedforward Integration**: 30 min
- **Testing & Verification**: 30 min
- **Documentation & Cleanup**: 15 min

**Total**: ~2.5 hours

---

## Next Session

After this session completes successfully:

1. **PolyAttention Fused Kernel** (30x speedup)
2. **Mamba Selective Scan Kernel** (20x speedup)
3. **AttentionContext GPU Ops** (30x speedup)
4. **Full Component Integration & Testing**
5. **Cleanup & Consolidation**

---

## Support

If you need help:
1. Check the detailed guide: `PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md`
2. Review auto-generated code patterns in existing GPU implementations
3. Check error messages (they include helpful suggestions)
4. Search for similar operations in `gpu_device.rs` or `gpu_ops.rs`

---

## Ready?

```bash
# 1. Verify GPU detection
cargo test --lib gpu_device::tests --release

# 2. Build with GPU features
cargo build --release --features gpu-wgpu

# 3. Open the implementation guide
# Read: PHASE5.6_RICHARDS_GLU_FUSED_KERNEL_GUIDE.md

# 4. Start implementing
# File: src/domain/layers/components/fused_kernels_module.rs
# Function: richards_glu_fused::execute() at line 108

echo "Ready to start Phase 5.6 GPU Consolidation!"
```

**Begin with Step 1 in `SESSION_PHASE5.6_IMMEDIATE_ACTIONS.md`**
