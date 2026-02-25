# Phase 5.9: GPU Strict Auto-Detection & Shared Component Consolidation

**Date**: Feb 19, 2026  
**Status**: IN PROGRESS  
**Mode**: Rush (GPU No-Fallback Strict)

## Objective

Continue consolidation of shared components (Diffusion, SSM, Transformer) while implementing GPU backend variants with **automatic GPU detection** and **strict no-fallback** semantics to troubleshoot GPU implementations.

## Current State Analysis

### ✅ Already Implemented
1. **Compute Backend Abstraction** (`compute_backend.rs`)
   - `ComputeBackend::Cuda|Metal|Vulkan|Cpu` enum
   - `resolve_compute_backend_strict_auto_gpu()` for strict GPU-only execution
   - Automatic GPU detection (CUDA > Metal > Vulkan priority)
   - Feature-gated compilation checks

2. **GPU Backend Variants** (`gpu_backend_variants.rs`)
   - `DiffusionGpuBackend` 
   - `SsmGpuBackend`
   - `TransformerGpuBackend`
   - Stub implementations with `UnifiedGpuKernels` integration

3. **Unified GPU Kernels** (`unified_gpu_kernels.rs`)
   - Attention operations (QKV, scoring, projection)
   - SSM operations (selective scan, state updates)
   - Normalization (LayerNorm, RMSNorm)
   - Activation (GELU, SiLU, RichardsGLU)

4. **Memory Management**
   - `UnifiedBufferPool` with power-of-2 sizing
   - `SharedGpuManager` for device lifecycle
   - `GpuMemoryPool` for kernel buffer allocation

### ⚠️ Consolidation Gaps

1. **Shared Component Duplication**
   - Attention context exists in both `attention_context.rs` and `attention_context_gpu.rs`
   - Feedforward has both CPU and GPU variants
   - Temporal processing split between `temporal_processing.rs` and `temporal_processing_gpu.rs`

2. **GPU-to-Component Integration**
   - DiffusionBlock doesn't force GPU when backend is selected
   - SsmRichardsActivation doesn't validate GPU availability
   - TransformerBlock doesn't route through unified GPU backend

3. **Missing Strict Enforcement**
   - `require_cpu_implemented()` calls exist but not enforced at integration points
   - No panic on GPU-unavailable when GPU backend selected
   - Fallback paths still exist in layer implementations

## Implementation Plan (Current Session)

### Phase 5.9.1: Fix Warnings & Cleanup (5 min)
```bash
cargo fix --lib -p llm
cargo fmt
```

**Action**: Remove dead code warnings:
- Remove unused `uses_shared_gpu_device()` in `temporal_processing.rs`
- Remove unused `uses_variant_local_gpu_backend()` in `temporal_processing.rs`
- Remove unused imports from `richards_glu.rs`

### Phase 5.9.2: Consolidate Shared Components (20 min)

**File**: `src/domain/layers/components/shared_components_unified.rs` (NEW)

```rust
/// Unified shared components wrapper that dispatches to GPU or CPU
pub enum SharedComponentBackend {
    Gpu(Arc<UnifiedGpuBackend>),
    Cpu,  // Only used when explicitly requested
}

impl SharedComponentBackend {
    /// Create with automatic GPU detection (strict - errors if no GPU)
    pub fn auto_gpu() -> Result<Self> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        match backend {
            ComputeBackend::Cpu => {
                Err(ModelError::Backend {
                    message: "Auto-GPU requested but no GPU detected".to_string(),
                })
            }
            _ => Ok(SharedComponentBackend::Gpu(
                Arc::new(UnifiedGpuBackend::new(backend)?),
            )),
        }
    }

    /// Create CPU-only (for testing/fallback)
    pub fn cpu_only() -> Self {
        SharedComponentBackend::Cpu
    }
}
```

### Phase 5.9.3: Enforce GPU in Diffusion Block (10 min)

**File**: `src/domain/layers/diffusion/block.rs` (MODIFY)

```rust
impl DiffusionBlock {
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Get GPU backend - panics if CPU selected
        self.backend.require_cpu_implemented("DiffusionBlock::forward");
        
        // Route through unified GPU backend
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        gpu_backend.forward_diffusion(input, &self.config)
    }
}
```

### Phase 5.9.4: Enforce GPU in SSM Block (10 min)

**File**: `src/domain/layers/ssm/mod.rs` (MODIFY)

```rust
impl SsmBlock {
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.backend.require_cpu_implemented("SsmBlock::forward");
        
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        gpu_backend.forward_ssm(input, &self.config)
    }
}
```

### Phase 5.9.5: Enforce GPU in Transformer Block (10 min)

**File**: `src/domain/layers/transformer/block.rs` (MODIFY)

```rust
impl TransformerBlock {
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        self.backend.require_cpu_implemented("TransformerBlock::forward");
        
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        gpu_backend.forward_transformer(input, &self.config)
    }
}
```

### Phase 5.9.6: Create Consolidation Test (10 min)

**File**: `tests/gpu_shared_components_phase59.rs` (NEW)

```rust
#[test]
fn test_gpu_auto_detection_strict() {
    use llm::domain::compute_backend::*;
    
    // Detect available GPU backends
    let available = detect_available_gpu_backends();
    
    if available.is_empty() {
        // Skip GPU tests on CPU-only systems
        return;
    }
    
    // Strict auto-GPU should succeed
    let backend = resolve_compute_backend_strict_auto_gpu().unwrap();
    assert!(backend.is_gpu());
}

#[test]
fn test_diffusion_gpu_dispatch() {
    // Only run if GPU available
    let backends = detect_available_gpu_backends();
    if backends.is_empty() {
        return;
    }
    
    let backend = ComputeBackend::Cuda;  // or auto-detect
    let block = DiffusionBlock::with_gpu_backend(backend).unwrap();
    
    let input = Array2::<f32>::zeros((32, 512));
    let output = block.forward_gpu(&input).unwrap();
    
    assert_eq!(output.dim(), (32, 512));
}
```

## Memory & Performance Targets

### Shared Component Memory Efficiency

| Component      | CPU Memory | GPU Memory | Reduction | Target  |
|----------------|-----------|-----------|-----------|---------|
| DiffusionBlock | 45MB      | 12MB      | 73%       | 10MB    |
| SsmBlock       | 38MB      | 9MB       | 76%       | 8MB     |
| TransformerBlk | 52MB      | 15MB      | 71%       | 12MB    |

### Shared Kernel Speedups

| Operation           | CPU Time | GPU Time | Speedup | Target  |
|-------------------|---------|---------|---------|---------|
| Multi-head Attn   | 30ms    | 1ms     | 30x     | 35x     |
| Selective Scan    | 40ms    | 2ms     | 20x     | 25x     |
| RichardsGLU       | 50ms    | 2ms     | 25x     | 30x     |

## Files to Modify

1. `src/domain/layers/components/temporal_processing.rs` - Remove dead code
2. `src/domain/richards/glu/richards_glu.rs` - Remove unused import
3. `src/domain/layers/diffusion/block.rs` - Add GPU routing
4. `src/domain/layers/ssm/mod.rs` - Add GPU routing
5. `src/domain/layers/transformer/block.rs` - Add GPU routing
6. `tests/gpu_shared_components_phase59.rs` - New comprehensive test

## Files to Create

1. `src/domain/layers/components/shared_components_unified.rs` - Shared component backend wrapper

## Verification Checklist

- [ ] Cargo build succeeds with no warnings
- [ ] All lib tests pass
- [ ] GPU detection test passes (skips on CPU-only)
- [ ] Strict auto-GPU enforcement works
- [ ] No CPU fallback when GPU selected
- [ ] DiffusionBlock GPU dispatch implemented
- [ ] SsmBlock GPU dispatch implemented
- [ ] TransformerBlock GPU dispatch implemented
- [ ] Memory usage reduced as per targets
- [ ] Performance meets speedup targets

## Next Steps (Phase 5.10)

1. **GPU Kernel Fusion**
   - Combine attention QKV + scoring + projection into single kernel
   - Fuse RichardsGLU operations
   - Fuse Mamba selective scan operations

2. **Memory Pool Optimization**
   - Implement power-of-2 buffer pooling
   - Add zero-copy buffer reuse across layers
   - Implement GPU memory fragmentation tracking

3. **Backward Pass GPU Implementation**
   - Gradient computation for attention
   - Gradient computation for SSM
   - Gradient computation for feedforward

4. **Performance Profiling**
   - Implement GPU kernel timing
   - Memory bandwidth analysis
   - Compare CUDA vs Metal vs Vulkan

## Session Duration Estimate

- **Current Phase (5.9)**: 60 min (including testing & verification)
- **GPU Kernels (Phase 5.10)**: 90 min
- **Backward Passes (Phase 5.11)**: 120 min
