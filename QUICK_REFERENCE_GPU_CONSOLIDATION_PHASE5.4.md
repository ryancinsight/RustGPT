# GPU Consolidation Phase 5.4 - Quick Reference Card

**Use this**: For quick lookups, API reference, and troubleshooting  
**See full docs**: `SESSION_CONSOLIDATION_GPU_PHASE5.4_SUMMARY.md`  
**Implementation guide**: `PHASE5.4_GPU_FORWARD_IMPLEMENTATION_GUIDE.md`  
**Migration guide**: `GPU_CONSOLIDATION_MIGRATION_GUIDE.md`

---

## New Consolidated API (Phase 5.4)

### GpuComponent Trait
```rust
use crate::domain::compute::GpuComponent;

pub trait GpuComponent: Sized {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>);
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;
    fn is_gpu_ready(&self) -> bool;
    fn gpu_backend_name(&self) -> Option<&'static str>;
    fn ensure_capacity(&mut self, batch_size: usize, embed_dim: usize, seq_len: usize) -> Result<()>;
}
```

### Helper Function
```rust
use crate::domain::compute::require_gpu_device;

// Validates GPU is attached, errors with clear message if not
require_gpu_device(&device, "operation_name")?;
```

---

## Quick Implementation Checklist

### For New GPU Components
- [ ] Add `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field
- [ ] Implement `GpuComponent` trait
- [ ] Create `forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>>` method
- [ ] Call `require_gpu_device()` at start of GPU method
- [ ] Upload input to GPU, execute kernels, download output
- [ ] Test GPU path and compare vs CPU reference (should be <1e-4 difference)

### For Existing GPU Components (Migration)
- [ ] Replace imports from `shared_gpu_manager` with imports from `domain::compute`
- [ ] Replace imports from `gpu_shared_ops` with `domain::compute` APIs
- [ ] Remove GPU context parameters from method signatures
- [ ] Implement `GpuComponent` trait
- [ ] Update tests to use new API
- [ ] Remove deprecated manager structs from fields

---

## Key APIs at a Glance

| API | Purpose | Location |
|-----|---------|----------|
| `GpuComponent` | Interface for GPU-capable components | `domain::compute` |
| `UnifiedGpuBufferPool` | GPU memory management | `domain::compute` |
| `GpuDevice::auto_detect()` | Auto-detect GPU (strict: errors if none) | `domain::compute` |
| `require_gpu_device()` | Validate GPU is attached | `domain::compute` |
| `UnifiedLayerWorkspace` | CPU/GPU buffer management | `layers::components` |

---

## Error Handling Pattern

```rust
// ✅ Correct: Explicit error if no GPU
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    require_gpu_device(&self.gpu_device, "forward_gpu")?;
    
    let (batch_size, embed_dim) = (input.nrows(), input.ncols());
    self.ensure_capacity(batch_size, embed_dim, 1)?;
    
    // ... GPU computation ...
    
    Ok(output)
}

// Caller chooses fallback strategy
match component.forward_gpu(&input) {
    Ok(output) => output,
    Err(e) if e.contains("GPU") => {
        println!("GPU unavailable, using CPU");
        component.forward_cpu(&input)?
    }
    Err(e) => return Err(e),
}
```

---

## GPU Kernels Reference

### Available WGSL Kernels
- **GEMM**: Matrix multiply `C = alpha*A@B + beta*C`
- **Softmax**: Row-wise softmax normalization
- **LayerNorm**: Layer normalization with learnable parameters
- **AXPY**: Vector operation `y = a*x + b*y`
- **Richards Curve**: Custom activation function (Phase 5.1+)
- **PolyAttention**: Polynomial attention scoring (Phase 5.2+)
- **Mamba Selective Scan**: SSM recurrence (Phase 5.4 - WIP)
- **RG-LRU Recurrence**: RG-LRU recurrence (Phase 5.4 - WIP)

### Using Kernels
```rust
// Access through device execution context
let (pool, ops) = device.execution_context();

// GEMM: output = alpha * A @ B + beta * output
ops.gemm_f32(1.0, &a_buf, &b_buf, 0.0, &mut output, m, n, k)?;

// Softmax: row-wise normalization
ops.softmax(&input, &mut output, rows, cols)?;

// LayerNorm: with gamma, beta parameters
ops.layer_norm(&input, &gamma, &beta, &mut output, batch, features, eps)?;
```

---

## GPU Consolidation Timeline

| Phase | Timeline | Status | Focus |
|-------|----------|--------|-------|
| 5.4.1 | Feb 14 | ✅ DONE | Consolidate GPU managers, unified API |
| 5.4.2 | Feb 15-16 | TODO | GPU forward implementations (DiffusionBlock, Mamba, RG-LRU) |
| 5.4.3 | Feb 16-17 | TODO | Memory optimization, kernel fusion |
| 5.4.4 | Feb 17 | TODO | Verification, testing, benchmarking |
| 5.5+ | Mar+ | Future | Mixed precision, async execution, CUDA backend |

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "GPU device not attached" | Call `enable_gpu_auto_detect()` first |
| "No supported GPU backend was detected" | No GPU on system OR compile with `--features gpu-wgpu` |
| GPU output differs from CPU | Numerical tolerance (expect <1e-4 difference), check kernel implementation |
| Slow GPU performance | Verify data transfers aren't bottleneck, check kernel fusion opportunities |
| Build takes forever | Build queue backed up, try `cargo check` or work on other files |

---

## Strict No-Fallback Mode Explained

### What It Means
GPU operations **error explicitly** if GPU unavailable. They never silently fall back to CPU.

### Why It's Better
```
Old (Silent Fallback):                    New (Strict No-Fallback):
forward_gpu()                             forward_gpu() -> Result
  ↓                                         ↓
Is GPU ready? ← No ← Silently use CPU    Is GPU ready? ← No ← Error!
                ↑                                            ↑
             Hidden!                              Explicit, traceable

Risk: Unpredictable performance            Benefit: Predictable, debuggable
Silent regressions if GPU breaks           Clear failure path
```

### For Developers
```rust
// You decide what to do if GPU fails
if block.is_gpu_ready() {
    output = block.forward_gpu(&input)?
} else {
    output = block.forward_cpu(&input)?  // Explicit choice
}
```

---

## File Locations

### Core GPU Infrastructure
- `src/domain/compute/gpu_device.rs` - GPU device abstraction
- `src/domain/compute/gpu_ops.rs` - GPU operation trait definition
- `src/domain/compute/gpu_memory.rs` - GPU memory management
- `src/domain/compute/unified_gpu_buffer_pool.rs` - Buffer pool + consolidation APIs
- `src/domain/compute/unified_gpu_executor.rs` - GPU kernel dispatcher
- `src/domain/compute/wgpu_ops.rs` - WGPU backend + WGSL shaders

### Shared Components
- `src/domain/layers/components/shared_gpu_manager.rs` - **DEPRECATED** (Phase 5.4)
- `src/domain/layers/components/gpu_shared_ops.rs` - **DEPRECATED** (Phase 5.4)
- `src/domain/layers/components/attention_context_gpu.rs` - Attention GPU path
- `src/domain/layers/components/feedforward_gpu.rs` - Feedforward GPU path
- `src/domain/layers/components/temporal_processing_gpu.rs` - Temporal GPU dispatch

### Block Implementations
- `src/domain/blocks/transformer_block.rs` - Has GPU variant (verify)
- `src/domain/blocks/diffusion_block.rs` - CPU only (TODO: add GPU)
- `src/domain/temporal/mamba.rs` - CPU path (TODO: GPU kernel)
- `src/domain/temporal/rg_lru.rs` - CPU path (TODO: GPU kernel)

---

## Performance Expectations

### Expected Speedups (When Fully Implemented)
- **GEMM operations**: 10-50x faster (depending on matrix size)
- **Attention**: 5-20x faster (with kernel fusion)
- **Feedforward**: 10-30x faster (with in-place operations)
- **SSM operations**: 2-10x faster (with optimized scan kernel)
- **Full block**: 5-15x faster (end-to-end)

### Memory Efficiency Improvements (Phase 5.4.3)
- **With power-of-2 sizing**: 20-30% reduction in GPU allocations
- **With in-place operations**: 10-15% reduction in intermediate buffers
- **With kernel fusion**: 15-25% reduction in memory bandwidth

---

## Import Templates

### New Code (Phase 5.4+)
```rust
use crate::domain::compute::{
    GpuComponent,
    GpuDevice,
    UnifiedGpuBufferPool,
    require_gpu_device,
};
use std::sync::{Arc, Mutex};
```

### Old Code (Deprecated)
```rust
// ❌ Don't use
use crate::domain::layers::components::{
    SharedComponentGpuManager,
    GpuSharedOpsContext,
};
```

---

## Test Template

```rust
#[test]
fn test_component_gpu_forward() {
    let mut component = MyComponent::new();
    
    // Try to enable GPU (may not be available)
    if component.enable_gpu_auto_detect().is_err() {
        println!("No GPU available, skipping GPU test");
        return;
    }
    
    let input = Array2::random((2, 64));
    let output = component.forward_gpu(&input).unwrap();
    
    // Verify output shape and correctness
    assert_eq!(output.shape(), input.shape());
    
    // Optional: Compare with CPU reference
    let output_cpu = component.forward_cpu(&input).unwrap();
    let max_diff = (&output - &output_cpu).mapv(f32::abs).max();
    assert!(max_diff < 1e-4, "GPU output differs from CPU by {}", max_diff);
}
```

---

## Related Documentation Index

| Document | Purpose | Length |
|----------|---------|--------|
| SESSION_CONSOLIDATION_GPU_PHASE5.4_SUMMARY.md | Overall session summary | 300 lines |
| SESSION_CONSOLIDATION_GPU_PHASE5.4_PLAN.md | Strategy & roadmap | 150 lines |
| SESSION_CONSOLIDATION_GPU_PHASE5.4_PROGRESS.md | Progress tracker | 200 lines |
| PHASE5.4_GPU_FORWARD_IMPLEMENTATION_GUIDE.md | Technical implementation details | 400+ lines |
| GPU_CONSOLIDATION_MIGRATION_GUIDE.md | Migration patterns & examples | 400+ lines |
| GPU_BACKEND_IMPLEMENTATION_STATUS.md | Phase 5.3 GPU status | 200 lines |

---

*Quick Reference for Phase 5.4 GPU Consolidation*  
*Last Updated: February 14, 2026*  
*For detailed information, see referenced documentation files*
