# Quick Start: Phase 4 GPU Kernel Implementation
**Status**: Ready to begin  
**Focus**: Implement actual GPU kernels (dispatch is now active)  
**Priority**: AttentionContext kernel (30x target)

---

## Current State Summary

✅ **GPU Dispatch Active**: All 3 shared components routing to GPU when available  
✅ **GpuComponent Trait**: Implemented in all components  
✅ **Buffer Management**: UnifiedBufferPool ready for allocations  
✅ **Auto-Detection**: CUDA > Metal > Vulkan > WGPU priority order  
✅ **Graceful Fallback**: CPU path active as safety net  

❌ **GPU Kernels**: Stubs exist, awaiting implementation

---

## What's Already Done (Don't Re-do)

```
✅ SharedAttentionContext
  ├── apply_context() → GPU dispatch wired
  ├── apply_context_into() → GPU dispatch wired
  └── GpuComponent trait: implemented

✅ SharedFeedforward
  ├── forward() → ComputeBackend.is_gpu()
  ├── forward_into() → ComputeBackend dispatch
  └── GpuComponent trait: implemented

✅ SharedTemporalProcessing
  ├── forward() → ComputeBackend.is_gpu()
  ├── forward_with_causal() → ComputeBackend dispatch
  ├── forward_into() → ComputeBackend dispatch
  └── GpuComponent trait: implemented
```

---

## What Needs to Be Done (Phase 4)

### 1. AttentionContext GPU Kernel (Start Here)

**File**: `src/domain/layers/components/unified_gpu_kernels.rs`

**Method to Implement**:
```rust
pub fn forward_attention_context(
    &mut self,
    input: &Array2<f32>,       // (batch, embed_dim)
    context: &Array2<f32>,     // (embed_dim, embed_dim)
    strength: f32,
) -> Result<Array2<f32>> {
    // TODO: Implement matrix multiplication + element-wise mixing
    // Current: Returns input unchanged (stub)
    // Target: 30x speedup
}
```

**What It Needs to Do**:
```
1. Transfer input to GPU
2. Compute: matrix_mult = input @ context
3. Mix: output = input + (strength * matrix_mult)
4. Transfer output back to CPU
```

**Backend-Specific Implementations**:
- **CUDA**: Use cuBLAS for matrix multiply
- **WGPU**: Compute shader with tiling (64x64)
- **Metal**: Metal Performance Shaders
- **Vulkan**: Compute shader (generic)

---

### 2. RichardsGLU Fused Kernel

**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Method to Create**:
```rust
pub fn richards_glu_fused(
    input: &Array2<f32>,
    w1: &Array2<f32>,
    w2: &Array2<f32>,
    w_out: &Array2<f32>,
    backend: &mut UnifiedGpuBackend,
) -> Result<Array2<f32>> {
    // TODO: Fused kernel combining:
    // x1 = input @ W1 → richards_activation(x1)
    // x2 = input @ W2 → sigmoid(x2)
    // gated = x1 * sigmoid(x2)
    // output = gated @ W_out
}
```

---

### 3. PolyAttention Fused Kernel

**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Method to Create**:
```rust
pub fn poly_attention_fused(
    input: &Array2<f32>,
    w_q: &Array2<f32>,
    w_k: &Array2<f32>,
    w_v: &Array2<f32>,
    w_o: &Array2<f32>,
    num_heads: usize,
    backend: &mut UnifiedGpuBackend,
) -> Result<Array2<f32>> {
    // TODO: Single-pass attention kernel
    // Q = input @ W_Q
    // K = input @ W_K  
    // V = input @ W_V
    // scores = (Q @ K.T) / sqrt(embed_dim)
    // attn = softmax(scores)
    // output = attn @ V @ W_O
}
```

---

### 4. Mamba Scan Kernel

**File**: `src/domain/layers/components/fused_kernels_module.rs`

**Method to Create**:
```rust
pub fn mamba_scan_kernel(
    input: &Array2<f32>,
    a: &Array2<f32>,
    b: &Array2<f32>,
    c: &Array2<f32>,
    backend: &mut UnifiedGpuBackend,
) -> Result<Array2<f32>> {
    // TODO: Vectorized state space scan
    // h[t] = A[t] * h[t-1] + B[t] * x[t]
    // y[t] = C[t] * h[t]
}
```

---

## Code Pattern: WGPU Compute Shader Template

All GPU kernels follow this pattern:

```rust
#[cfg(feature = "gpu-wgpu")]
pub fn forward_attention_context_wgpu(
    &mut self,
    input: &Array2<f32>,
    context: &Array2<f32>,
    strength: f32,
) -> Result<Array2<f32>> {
    let batch = input.nrows();
    let embed_dim = input.ncols();
    
    // 1. Upload to GPU
    let input_buf = self.buffer_pool.allocate(batch * embed_dim)?;
    self.upload_f32(&input, &input_buf)?;
    
    // 2. Allocate output buffer
    let output_buf = self.buffer_pool.allocate(batch * embed_dim)?;
    
    // 3. Dispatch kernel
    let kernel = self.compute_pipeline("attention_context")?;
    let mut encoder = self.device.create_command_encoder()?;
    let mut pass = encoder.begin_compute_pass();
    pass.set_pipeline(&kernel);
    pass.set_bind_group(0, &self.create_bind_group(
        &input_buf,
        &context,
        &output_buf,
    )?);
    pass.dispatch_workgroups(
        (batch as u32 + 63) / 64,    // tile width
        (embed_dim as u32 + 63) / 64, // tile height
        1,
    );
    drop(pass);
    
    // 4. Download from GPU
    let mut result = Array2::zeros((batch, embed_dim));
    self.download_f32(&output_buf, &mut result)?;
    
    // 5. Cleanup
    self.buffer_pool.recycle(input_buf)?;
    self.buffer_pool.recycle(output_buf)?;
    
    Ok(result)
}
```

---

## Testing Pattern

```rust
#[test]
#[cfg(feature = "gpu-wgpu")]
fn test_attention_context_kernel_correctness() {
    // 1. Create test data
    let input = Array2::random((128, 64));
    let context = Array2::random((64, 64));
    
    // 2. CPU reference
    let cpu_result = cpu_attention_context(&input, &context, 1.0);
    
    // 3. GPU computation
    let mut backend = UnifiedGpuBackend::auto_detect().unwrap();
    let gpu_result = backend.forward_attention_context(&input, &context, 1.0)
        .expect("GPU kernel failed");
    
    // 4. Compare (tolerance: 1e-4)
    for (cpu_val, gpu_val) in cpu_result.iter().zip(gpu_result.iter()) {
        let diff = (cpu_val - gpu_val).abs();
        assert!(diff < 1e-4, "Mismatch: CPU={}, GPU={}", cpu_val, gpu_val);
    }
}
```

---

## Benchmark Pattern

```rust
#[bench]
fn bench_attention_context_gpu_1k_batch(b: &mut Bencher) {
    let input = Array2::zeros((1024, 64));
    let context = Array2::zeros((64, 64));
    let mut backend = UnifiedGpuBackend::auto_detect().unwrap();
    
    b.iter(|| {
        backend.forward_attention_context(&input, &context, 1.0)
            .expect("Benchmark failed")
    });
}

// Expected: 0.5ms (30x speedup from 15ms CPU)
```

---

## Build & Test Commands

```bash
# Check only (fast)
cargo check --lib --features gpu-wgpu

# Run tests with GPU
cargo test --lib --features gpu-wgpu gpu_kernels

# Run specific test
cargo test --lib forward_attention_context_wgpu -- --exact --nocapture

# Benchmark
cargo bench --bench gpu_kernels_phase56 --features gpu-wgpu -- --measurement-time=10

# Build release (use this for benchmarks)
cargo build --release --features gpu-wgpu
./target/release/[binary] [args]
```

---

## File Organization

```
src/domain/layers/components/
├── unified_gpu_kernels.rs          ← AttentionContext kernel
├── fused_kernels_module.rs         ← RichardsGLU, PolyAttention, Mamba
├── gpu_kernels_attention_context.rs ← (Optional: backend-specific code)
├── gpu_kernels_richardsglu.rs      ← (Optional: backend-specific code)
├── attention_context.rs            ✅ Already wired
├── feedforward.rs                  ✅ Already wired
└── temporal_processing.rs          ✅ Already wired

tests/
├── gpu_kernels_phase56_verification.rs  ← Integration tests
└── [existing tests still work]

benches/
├── gpu_kernels_phase56.rs          ← Benchmarks
└── [existing benchmarks]
```

---

## Performance Targets Checklist

Before marking a kernel as complete:

```
AttentionContext:
☐ Implemented for CUDA
☐ Implemented for WGPU
☐ Implemented for Metal
☐ CPU vs GPU matches to 1e-4 tolerance
☐ Benchmark shows ≥30x speedup
☐ Tests pass with --features gpu-wgpu
☐ Tests pass with --features gpu-cuda
☐ Tests pass with --features gpu-metal

RichardsGLU:
☐ Fused kernel combines 5 launches → 2 passes
☐ 25x speedup target verified
☐ All backends working
☐ Numerical accuracy within 1e-4

PolyAttention:
☐ Single-pass kernel (3+ launches → 1)
☐ Multi-head support verified
☐ 30x speedup target verified
☐ All backends working

Mamba Scan:
☐ Vectorized scan implementation
☐ State propagation correct
☐ Selective masking working
☐ 20x speedup target verified
```

---

## Known Constraints

### GPU Memory Budget
- Pre-allocate with power-of-2 sizing in UnifiedBufferPool
- Don't exceed GPU memory (typically 2-8GB for NVIDIA/Apple)
- Use buffer recycling to minimize allocations

### Kernel Execution
- All kernels must be deterministic (same input → same output)
- Handle NaN/Inf gracefully (clamp to reasonable values)
- Support variable batch sizes (1 to 1024)

### Testing Environment
- WGPU: Works on all platforms (software fallback if no GPU)
- CUDA: Requires NVIDIA GPU + CUDA Toolkit 12.4+
- Metal: macOS only (Apple Silicon or Intel GPU)
- Vulkan: Linux/Windows (requires Vulkan SDK)

---

## Debugging Tips

### GPU Errors
```rust
if let Err(e) = backend.forward_attention_context(...) {
    eprintln!("GPU kernel failed: {}", e);
    eprintln!("Falling back to CPU");
    return cpu_attention_context(...);
}
```

### Memory Leaks
- Use `valgrind` or GPU memory profiler
- Check buffer pool is recycling correctly
- Verify no dangling GPU buffers

### Numerical Differences
- Use `cargo test -- --nocapture` to see values
- Check accumulation order (matrix multiply is non-associative)
- Validate with extended precision CPU reference

### Performance Profiling
- **NVIDIA**: `nvidia-smi` for utilization, `nsight` for kernel profiling
- **Apple**: Xcode GPU profiler with Metal debugger
- **WGPU**: Chrome DevTools if using web backend

---

## Next Steps

1. **Today**: Pick AttentionContext kernel
2. **First**: Implement WGPU compute shader
3. **Then**: Add CUDA/Metal wrappers
4. **Finally**: Verify speedup target

---

## References

### Files That Know How to Do This
- `attention_context_gpu.rs` - Shows how apply_incoming_context_gpu() works
- `unified_gpu_backend.rs` - GPU backend trait definitions
- `unified_gpu_kernels.rs` - Kernel dispatcher location
- `feedforward_gpu.rs` - Example GPU dispatch pattern

### Commands to Know
```bash
cargo check --lib --features gpu-wgpu    # Fast check
cargo test --lib --features gpu-wgpu     # Run tests
cargo bench --bench gpu_kernels_phase56  # Benchmarks
cargo clippy --all-targets                # Lint
```

---

**Ready to Code**: GPU dispatch infrastructure complete. Start with AttentionContext kernel implementation.

Time Estimate: 30 min setup + 1-2 hours coding + 30 min testing = 2-3 hours per kernel
