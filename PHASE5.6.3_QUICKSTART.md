# Phase 5.6.3: GPU Backend Consolidation - Quick Start

**Status**: Ready to begin GPU kernel implementation  
**Date**: Feb 16, 2026

## Current State

### ✅ Completed
- GPU device auto-detection (CUDA > Metal > WGPU priority)
- Unified `GpuMatrixOps` trait across all backends
- WGPU WGSL shaders for core operations (GEMM, element-wise, norm, softmax)
- CUDA/Metal stub implementations with informative error messages
- GPU memory pool abstraction
- Unified GPU kernels dispatcher

### ❌ Missing / Needs Work
- **CRITICAL**: RichardsGLU two-pass fused kernel (WGPU + CUDA + Metal)
- PolyAttention kernels (if using PolyAttention layers)
- CUDA `.cu` kernel implementations
- Metal `.metal` shader implementations
- Comprehensive GPU integration tests
- Performance benchmarks with profiling

## Quick Commands

### Build & Test
```bash
# Build with WGPU (default cross-platform GPU)
cargo build --release --features gpu-wgpu

# Build with all GPU backends
cargo build --release --features gpu-all

# Run GPU kernel tests
cargo test --lib gpu_device --features gpu-all

# Run GPU integration tests
cargo test --test gpu_shared_components_phase56 --features gpu-all

# Run benchmarks
cargo bench --bench gpu_performance --features gpu-wgpu
```

### Debug GPU Detection
```bash
# Test backend detection
cargo test --test gpu_backend_detection -- --nocapture --features gpu-all

# Expected output:
# Detected GPU backends: ["cuda", "wgpu", ...] (or subset)
```

## Implementation Roadmap (This Session)

### 1. Verify WGPU Implementation (30 min)
```bash
# Quick sanity check
cargo test --lib gpu_ops --features wgpu

# Check for compilation errors in shaders
cargo check --features wgpu

# Run a simple kernel test
cargo test --test gpu_shared_components_phase56 --features wgpu
```

### 2. Implement RichardsGLU Two-Pass Kernel (2 hours)

The critical bottleneck for performance optimization.

**Location**: `src/domain/compute/wgpu_ops.rs`

**Current state**: Individual kernels exist, need fusion
- ✅ GEMM kernel (Pass 2 output projection)
- ✅ Element-wise ops (mul, add_scaled, scale)
- ❌ RichardsGLU fusion wrapper

**Implementation**:
```rust
// In wgpu_ops.rs
fn richards_glu_fused(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,           // [batch, input_dim]
    w_g1: &GpuBuffer,            // [input_dim, hidden_dim]
    w_g2: &GpuBuffer,            // [input_dim, hidden_dim]
    w_out: &GpuBuffer,           // [hidden_dim, output_dim]
    output: &mut GpuBuffer,      // [batch, output_dim]
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<()> {
    // Pass 1: Activation
    // - Allocate [batch, hidden_dim] intermediate buffer
    // - Compute input @ w_g1 → value
    // - Compute input @ w_g2 → gate_logits
    // - Compute gated = gate_logits * value
    
    // Pass 2: Projection
    // - Compute gated @ w_out → output
    // - Return
    
    // Key: Keep intermediate on GPU (zero-copy)
}
```

**Dispatch strategy**:
```
Pass 1 Kernel Dispatch:
  - Workgroups: (batch_size * hidden_dim + 255) / 256 with group size 256

Pass 2 Kernel Dispatch:
  - Use existing GEMM kernel with [batch, hidden_dim] @ [hidden_dim, output_dim]
  - Workgroups: ((batch_size * output_dim + 255) / 256) with 256 threads
```

### 3. Create CUDA Kernel Templates (1 hour)

Create `.cu` template files for developers to implement later.

**Files to create**:
```
src/domain/compute/cuda/kernels/
  ├── element_wise.cu      # All element-wise ops in one kernel
  ├── richards_curve.cu    # Stable Richards curve computation
  ├── richards_glu_fused.cu # Two-pass GLU kernel
  ├── activation.cu        # Softmax, layer norm
  └── Makefile.cu          # CUDA compilation template
```

**Example template** (`element_wise.cu`):
```cuda
// CUDA Element-Wise Operations Kernel Template
// Implement the following functions:

__global__ void kernel_relu(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = max(0.0f, input[idx]);
    }
}

// TODO: Implement other element-wise ops similarly
```

### 4. Test & Verify (1 hour)

```bash
# Verify WGPU kernels
cargo test --test gpu_shared_components_phase56 \
    --features wgpu -- --nocapture

# Expected: All tests pass, shows kernel execution times

# Profile dispatch overhead
cargo bench --bench gpu_performance --features wgpu \
    -- richards_glu

# Expected: Should show 2 kernel launches instead of 5+
```

## Debugging Tips

### WGSL Shader Compilation Errors
```rust
// If you see "shader compilation error" at runtime:
// 1. Check the shader_source string around line XX
// 2. Validate WGSL syntax: https://gpuweb.github.io/gpuweb/wgsl/
// 3. Common issues:
//    - Missing type annotations
//    - Incorrect uniform buffer alignment (must be 16-byte aligned)
//    - Out-of-bounds workgroup reads
```

### GPU Memory Issues
```rust
// If you see "GPU memory pool exhausted":
// 1. Check workspace capacity settings
// 2. Verify buffers are deallocated after use
// 3. Use memory_stats() to track allocation count:
let stats = pool.memory_stats();
println!("Used: {} bytes in {} allocations", 
    stats.used_bytes, stats.allocation_count);
```

### Auto-Detection Failures
```bash
# If GPU detection fails:

# 1. Check what's available
cargo test --test gpu_backend_detection -- --nocapture

# 2. Force specific backend
RUST_LOG=debug cargo test --test gpu_backend_detection \
    --features gpu-cuda -- --nocapture

# 3. Verify environment
echo $CUDA_HOME        # Should point to CUDA toolkit
echo $CUDA_PATH        # Alternative CUDA path
```

## Key Files to Modify

```
Primary:
  src/domain/compute/wgpu_ops.rs
    ├─ Add richards_glu_fused() kernel
    └─ Add poly_attention kernels (if needed)

Secondary:
  src/domain/compute/gpu_ops.rs
    ├─ May need to add new trait methods
    └─ Update GpuMatrixOps trait if needed

Testing:
  tests/gpu_shared_components_phase56.rs
    └─ Ensure all GPU kernels tested with all backends
```

## Success Checklist

- [ ] WGPU builds without errors
- [ ] All existing GPU tests pass with `--features wgpu`
- [ ] RichardsGLU two-pass kernel implemented and verified
- [ ] Kernel dispatch reduced from 5+ to 2 launches (verified via profiling)
- [ ] CUDA kernel templates created
- [ ] GPU memory pool tests pass
- [ ] Auto-detection prioritizes CUDA > Metal > WGPU
- [ ] No fallback to CPU in GPU paths

## Next Session Handoff

If not completed this session:

1. **Priority**: Finish RichardsGLU fused kernel
   - Blocks performance optimization goals
   - Foundation for other fused kernels

2. **Then**: Implement CUDA kernels
   - Start with element-wise operations (simpler)
   - Move to GEMM with cuBLAS
   - Finally custom kernels (RichardsGLU, etc.)

3. **Finally**: Metal implementation
   - Lower priority for cross-platform
   - Can use MPS for BLAS
   - Custom kernels in `.metal` files

## References

- **Consolidation Plan**: PHASE5.6.3_IMPLEMENTATION_PLAN.md
- **GPU Kernel Status**: PHASE5.6.3_GPU_KERNEL_STATUS.md
- **WGSL Documentation**: https://gpuweb.github.io/gpuweb/wgsl/
- **cuBLAS Integration**: https://docs.rs/cudarc/latest/cudarc/
- **Metal Shading Language**: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf

---

**Ready to begin Phase 5.6.3 GPU consolidation. Start with WGPU verification, then RichardsGLU kernel implementation.**
