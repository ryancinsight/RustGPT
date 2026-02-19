# GPU Backend Quick Start - Phase 5.3

## Immediate Compilation Check

```bash
# Verify build succeeds
cargo build --release

# Run tests (subset for quick feedback)
cargo test --lib test_gpu --nocapture 2>&1 | head -100
```

---

## Using GPU in Your Code

### 1. Enable GPU for a Component

```rust
use crate::domain::layers::components::GpuSharedOpsContext;

let mut ctx = GpuSharedOpsContext::new();

// Auto-detect GPU (strict - errors if unavailable)
ctx.enable_gpu_auto_detect()?;

// Ensure buffers are ready
ctx.ensure_capacity(batch_size, embed_dim, seq_len)?;

// Check if ready
if ctx.is_gpu_ready() {
    println!("GPU backend: {}", ctx.backend_name().unwrap_or("unknown"));
}
```

### 2. Run GPU Operations

```rust
// For SharedFeedforward
let result = feedforward.forward_gpu(&input, &mut ctx, &mut ops)?;

// For SharedAttentionContext
let result = attention.apply_incoming_context_gpu(&input, &mut ctx, &mut ops)?;

// For SharedTemporalProcessing (all mixing types)
let result = temporal.forward_gpu_dispatch(&input, &mut ctx, &mut ops)?;
```

### 3. Error Handling (No Silent Fallback)

```rust
// GPU operations explicitly return Result
match ctx.enable_gpu_auto_detect() {
    Ok(()) => {
        // GPU is definitely available
        let output = component.forward_gpu(&input)?;
    }
    Err(e) => {
        // GPU is NOT available - must handle explicitly
        eprintln!("GPU unavailable: {}", e);
        // Option 1: Use CPU fallback (explicitly)
        let output = component.forward(&input);
        // Option 2: Error out
        return Err(e);
    }
}
```

---

## Architecture Quick View

```
Component Layer          GPU Infrastructure         GPU Hardware
──────────────────────────────────────────────────────────────

SharedFeedforward  →  GpuSharedOpsContext  →  CUDA / Metal / Vulkan
SharedAttention    →      (auto-detect)    →  (strict, no fallback)
SharedTemporal     →   (buffer pool)       →  GpuDevice::auto_detect()
```

---

## Building with GPU Support

```bash
# CPU only (default)
cargo build --release

# With Vulkan/WGPU support (cross-platform)
cargo build --release --features gpu-wgpu

# With CUDA support (NVIDIA GPUs)
cargo build --release --features gpu-cuda

# With Metal support (macOS)
cargo build --release --features gpu-metal

# All GPU backends
cargo build --release --features gpu-all
```

---

## Testing GPU Functionality

### Check if GPU is available
```bash
cargo test --lib test_gpu_auto_detect -- --exact --nocapture
```

### Output on GPU-available machine
```
test gpu_shared_ops_context_creation ... ok
test gpu_auto_detect_no_fallback ... ok (GPU found and initialized)
```

### Output on CPU-only machine
```
test gpu_auto_detect_no_fallback ... ok (GPU unavailable, error handling verified)
```

---

## Common Patterns

### Pattern 1: Batch Processing with GPU
```rust
let mut executor = GpuBatchExecutor::new_auto_detect(batch_size, embed_dim, seq_len)?;

for batch in data.chunks(batch_size) {
    // Update capacity if needed
    executor.ensure_capacity(batch.len(), embed_dim, seq_len)?;
    
    // Process on GPU (or error if not available)
    let output = component.forward_gpu(&batch)?;
    results.push(output);
}
```

### Pattern 2: Fallback to CPU (Explicit)
```rust
match component.set_compute_backend_checked(ComputeBackend::AutoGpu) {
    Ok(()) => {
        // GPU is available
        output = component.forward(&input);
    }
    Err(_) => {
        // GPU not available - explicitly use CPU
        component.set_compute_backend(ComputeBackend::Cpu);
        output = component.forward(&input);
    }
}
```

### Pattern 3: Conditional GPU Usage
```rust
let mut ctx = GpuSharedOpsContext::new();
let gpu_available = ctx.enable_gpu_auto_detect().is_ok();

if gpu_available {
    ctx.ensure_capacity(batch, embed, seq)?;
    output = feedforward.forward_gpu(&input, &mut ctx, &mut ops)?;
} else {
    output = feedforward.forward(&input);
}
```

---

## Debugging GPU Issues

### Enable verbose output
```rust
if let Some(backend_name) = ctx.backend_name() {
    println!("Using GPU backend: {}", backend_name);
    let (batch, embed, seq, ready) = ctx.capacity_info();
    println!("GPU capacity: batch={}, embed={}, seq={}, ready={}", 
             batch, embed, seq, ready);
}
```

### Check GPU device directly
```rust
if let Some(device_arc) = ctx.device() {
    match device_arc.lock() {
        Ok(device) => {
            println!("GPU device name: {}", device.name());
            println!("GPU backend: {}", device.backend().as_str());
        }
        Err(_) => eprintln!("GPU device lock failed"),
    }
}
```

---

## Current Implementation Status

| Component | CPU | GPU | Status |
|-----------|-----|-----|--------|
| SharedFeedforward | ✅ | 🔶 | Dispatch ready, kernels pending |
| SharedAttentionContext | ✅ | 🔶 | Dispatch ready, kernels pending |
| SharedTemporalProcessing | ✅ | 🔶 | Dispatch ready, kernels pending |
| PolyAttention | ✅ | 🔶 | Auto-detect ready, kernels pending |

Legend: ✅ Complete | 🔶 Partial/Framework | ❌ Not implemented

---

## Next Phase Work (Phase 5.4+)

GPU kernel implementation in:
- `src/domain/layers/components/feedforward_gpu.rs` (RichardsGLU, MoE kernels)
- `src/domain/layers/components/attention_context_gpu.rs` (context modulation kernels)
- `src/domain/layers/components/temporal_processing_gpu.rs` (attention, Mamba, RG-LRU kernels)

**Kernel Technologies**:
- WGPU compute shaders for cross-platform GPU support
- CUDA kernels for NVIDIA-specific optimization
- Metal Performance Shaders for Apple Silicon

---

## Support Matrix

| Hardware | Supported | Feature Flag | Status |
|----------|-----------|--------------|--------|
| NVIDIA GPU | ✅ | `gpu-cuda` | Compiled |
| AMD GPU | ✅ | `gpu-wgpu` | Via Vulkan |
| Apple Silicon | ✅ | `gpu-metal` | Native Metal |
| Intel GPU | ✅ | `gpu-wgpu` | Via Vulkan |
| CPU only | ✅ | (default) | Works (no GPU) |

---

## Quick Diagnostics

```bash
# Check if GPU backend is detecting hardware
cargo run --release --example gpu_diagnostics

# Build with GPU support to verify feature flags
cargo build --release --features gpu-wgpu --verbose

# Run GPU-specific tests
cargo test --lib test_shared_component_gpu_manager -- --nocapture
```

---

## File Navigation

**Core GPU Infrastructure**:
- `src/domain/layers/components/gpu_shared_ops.rs` - Unified context & buffer management
- `src/domain/compute/gpu_device.rs` - GPU device abstraction
- `src/domain/compute/gpu_ops.rs` - GPU operation interfaces

**Component GPU Implementations**:
- `src/domain/layers/components/feedforward_gpu.rs`
- `src/domain/layers/components/attention_context_gpu.rs`
- `src/domain/layers/components/temporal_processing_gpu.rs`

**Shared Components** (updated for GPU):
- `src/domain/layers/components/feedforward.rs`
- `src/domain/layers/components/attention_context.rs`
- `src/domain/layers/components/temporal_processing.rs`

**PolyAttention GPU**:
- `src/domain/attention/poly_attention_gpu.rs` - GPU context helpers
- `src/domain/attention/poly_attention.rs` - GPU forward/backward (placeholders)

---

## Common Issues & Solutions

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| "GPU device not attached" | `enable_gpu_auto_detect()` not called | Call it before GPU operations |
| "GPU not ready" | `ensure_capacity()` not called | Call after attaching GPU |
| "No supported GPU backend" | GPU not detected at runtime | Install GPU drivers, use `cpu` |
| "Feature flag error" | Feature not enabled at compile time | Rebuild with `--features gpu-cuda` |
| "Device lock failed" | GPU device sync issue | Check for concurrent access |

---

## Performance Tips

1. **Batch Processing**: Larger batches = better GPU utilization
2. **Power-of-2 Sizing**: Buffers auto-sized to power-of-2 for GPU alignment
3. **Pre-allocation**: Call `ensure_capacity()` once before processing loop
4. **Transfer Cost**: GPU best for compute-heavy operations (avoid excessive H2D/D2H transfers)

---

## Session Checklist

- [x] Build succeeds with no errors
- [x] GPU detection implemented (strict, no fallback)
- [x] Buffer management framework in place
- [x] Integration with all shared components
- [x] Placeholder kernels ready for implementation
- [x] Tests framework in place
- [ ] Full GPU kernel implementation (Phase 5.4)
- [ ] Performance benchmarking
- [ ] Multi-GPU support
- [ ] Gradient computation on GPU
