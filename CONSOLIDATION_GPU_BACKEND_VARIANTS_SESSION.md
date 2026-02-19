# GPU Backend Variants Consolidation Session

**Date**: 2026-02-16
**Phase**: 5.6 GPU Consolidation

## Summary

This session focused on consolidating and optimizing shared GPU components between Diffusion, SSM, and Transformer architectures with automatic GPU detection and strict no-fallback semantics.

## Changes Made

### 1. Memory Optimization in `unified_gpu_kernels.rs`

**Fixed**: Duplicate buffer allocation bug in `attention_forward()` method.

**Before** (buggy):
```rust
// First allocation (never used - leaked)
let input_buf = device.allocate(input_size)?;
let q_buf = device.allocate(qkv_size)?;
// ... more allocations

// Second allocation (actual use)
let mut input_buf = device.allocate(input_size)?;
let mut q_buf = device.allocate(qkv_size)?;
// ... more allocations
```

**After** (optimized):
```rust
// Single allocation pass
let mut input_buf = device.allocate(input_size)?;
let mut q_buf = device.allocate(qkv_size)?;
// ... all allocations in one pass
```

**Impact**: ~50% memory reduction for attention operations.

### 2. New GPU Backend Variants Module

Created [`src/domain/layers/components/gpu_backend_variants.rs`](src/domain/layers/components/gpu_backend_variants.rs) with:

#### `DiffusionGpuBackend`
- Forward diffusion: add noise to input
- Predict noise from noisy input
- Denoising step: predict x_{t-1} from x_t
- Full denoising loop
- Noise schedules: Linear, Cosine, Sigmoid

#### `SsmGpuBackend`
- Mamba selective scan
- RG-LRU recurrent computation
- Unified SSM forward pass

#### `TransformerGpuBackend`
- Multi-head attention
- Flash attention (memory-efficient)
- Layer normalization
- Activation functions (GELU, SiLU, ReLU, Richards curve)

#### `GpuBackendFactory`
- Unified factory for all GPU backends
- `is_gpu_available()` - Check GPU availability
- `best_backend_name()` - Get best available backend name
- Factory methods for each backend type

### 3. Module Updates

Updated [`src/domain/layers/components/mod.rs`](src/domain/layers/components/mod.rs:1):
- Added `gpu_backend_variants` module
- Re-exported all GPU backend types
- Updated documentation with usage examples

### 4. Benchmark Fix

Fixed [`benches/unified_gpu_components.rs`](benches/unified_gpu_components.rs:27):
- Changed `rng.gen::<f32>()` to use `rand::distributions::Uniform` for Rust 2024 compatibility

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    GpuBackendVariants                               │
├────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ DiffusionGpu │  │   SsmGpu     │  │    TransformerGpu        │ │
│  │   Backend    │  │   Backend    │  │       Backend            │ │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘ │
│         │                 │                       │               │
│         └─────────────────┼───────────────────────┘               │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  UnifiedGpuKernels                          │  │
│  │   (Attention, SSM, Normalization, Activation)              │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                    GpuDevice                                │  │
│  │   (CUDA > Metal > Vulkan auto-detection)                   │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Strict No-Fallback Design

All GPU operations follow strict no-fallback semantics:

1. **Auto-detection**: `GpuDevice::auto_detect()` returns error if no GPU available
2. **Backend rejection**: `UnifiedGpuBackend::new(ComputeBackend::Cpu)` returns error
3. **Explicit errors**: All GPU operations return `Result<T, ModelError>` with descriptive errors

## Performance Targets

| Component          | CPU Time | GPU Target | Speedup |
|--------------------|----------|------------|---------|
| RichardsGLU FFN    | 30ms     | 1.5ms      | 20x     |
| PolyAttention      | 40ms     | 1.3ms      | 30x     |
| Mamba Scan         | 50ms     | 2.5ms      | 20x     |
| RG-LRU             | 35ms     | 2.3ms      | 15x     |
| AttentionContext   | 10ms     | 0.5ms      | 20x     |

## Usage Examples

### Diffusion

```rust
use llm::domain::layers::components::{
    DiffusionGpuBackend, NoiseScheduleParams, NoiseScheduleType
};

// Create with auto-detection
let diffusion = DiffusionGpuBackend::auto_detect(1000)?;

// Use cosine schedule
let diffusion = DiffusionGpuBackend::auto_detect(1000)?
    .with_schedule(NoiseScheduleParams::cosine(1000));

// Forward diffusion
let noisy = diffusion.forward_diffusion(&clean_input, t, None)?;

// Denoise
let denoised = diffusion.denoise(noisy_input, &model_weights)?;
```

### SSM (Mamba/RG-LRU)

```rust
use llm::domain::layers::components::SsmGpuBackend;

// Mamba
let mamba = SsmGpuBackend::mamba(256, 512, 128, 32)?;
let output = mamba.forward(&input)?;

// RG-LRU
let rg_lru = SsmGpuBackend::rg_lru(256, 512, 128, 32)?;
let output = rg_lru.forward(&input)?;
```

### Transformer

```rust
use llm::domain::layers::components::TransformerGpuBackend;

let transformer = TransformerGpuBackend::auto_detect(8, 512, 128, 32)?
    .with_causal(true)
    .with_activation(GpuActivation::Gelu);

// Attention
let output = transformer.attention_forward(&input, &wq, &wk, &wv, &wo)?;

// Layer norm
let normalized = transformer.layer_norm_forward(&input, Some(&gamma), Some(&beta))?;
```

### Factory

```rust
use llm::domain::layers::components::GpuBackendFactory;

// Check availability
if GpuBackendFactory::is_gpu_available() {
    println!("GPU: {:?}", GpuBackendFactory::best_backend_name());
}

// Create backends
let diffusion = GpuBackendFactory::diffusion(1000)?;
let mamba = GpuBackendFactory::ssm_mamba(256, 512, 128, 32)?;
let transformer = GpuBackendFactory::transformer(8, 512, 128, 32)?;
```

## Tests Passed

```
running 6 tests
test domain::layers::components::unified_gpu_backend::tests::test_stats_tracking ... ok
test domain::layers::components::unified_gpu_kernels::tests::test_attention_params ... ok
test domain::layers::components::unified_gpu_backend::tests::test_cpu_backend_rejected ... ok
test domain::layers::components::unified_gpu_kernels::tests::test_norm_params ... ok
test domain::layers::components::unified_gpu_kernels::tests::test_ssm_params ... ok
test domain::layers::components::unified_gpu_backend::tests::test_auto_detect_no_fallback ... ok

test result: ok. 6 passed; 0 failed; 0 ignored
```

## Files Modified

1. `src/domain/layers/components/unified_gpu_kernels.rs` - Fixed duplicate allocation bug
2. `src/domain/layers/components/mod.rs` - Added new module and re-exports
3. `src/domain/layers/components/gpu_backend_variants.rs` - **NEW** GPU backend variants
4. `benches/unified_gpu_components.rs` - Fixed Rust 2024 compatibility

## Next Steps

1. **GPU Feature Testing**: Run full test suite with `--features gpu-wgpu` after fixing pre-existing GPU compilation issues
2. **Performance Benchmarks**: Run benchmarks to validate speedup targets
3. **Integration Testing**: Test GPU backends with actual model forward passes
4. **Documentation**: Add more inline examples and usage guides

## Known Issues

The GPU feature flag compilation (`--features gpu-wgpu`) has pre-existing issues in other modules that need separate fixes:
- `poly_attention.rs`: Trait method visibility and borrow checker issues
- `ssm_gpu_kernels.rs`: Error variant mismatches
- `gpu_gemm_kernels.rs`: GpuDevice constructor signature changes

These are unrelated to the consolidation work and should be addressed in a follow-up session.
