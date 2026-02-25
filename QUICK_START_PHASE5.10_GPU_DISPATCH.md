# Quick Start: Phase 5.10 - GPU Dispatch Integration

**Status**: Planning complete, ready for implementation  
**Expected Duration**: 60 minutes  
**Prerequisites**: Phase 5.9 complete (Unified shared component backend)

## Objective

Integrate GPU backend dispatch into Diffusion, SSM, and Transformer blocks with automatic GPU detection and strict enforcement.

## Implementation Pattern

Each block follows the same pattern:

```rust
// In block's forward_gpu() method
pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Step 1: Enforce GPU (panic if CPU selected)
    self.backend.require_cpu_implemented("BlockName::forward");
    
    // Step 2: Get GPU backend with strict auto-detection
    let gpu_backend = UnifiedGpuBackend::auto_detect()?;
    
    // Step 3: Route through unified GPU kernels
    gpu_backend.forward_operation(input, &self.config)
}
```

## Phase 5.10.1: DiffusionBlock GPU Dispatch (20 min)

**File**: `src/domain/layers/diffusion/block.rs`

```rust
impl DiffusionBlock {
    /// Forward pass through GPU with strict no-fallback.
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Validate GPU availability
        self.backend.require_cpu_implemented("DiffusionBlock::forward_gpu");
        
        // Get unified GPU backend
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        
        // Route through GPU kernels
        gpu_backend.forward_diffusion(
            input,
            &self.config,
            &self.noise_schedule_params,
        )
    }

    /// Forward pass with automatic backend selection (CPU or GPU).
    pub fn forward_auto(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.backend.is_gpu() {
            self.forward_gpu(input)
        } else {
            self.forward_cpu(input)  // Existing CPU implementation
        }
    }
}
```

**Test Locations**:
- Add test in existing `tests/diffusion_block_*` files
- Create new integration test for GPU forward pass

## Phase 5.10.2: SsmBlock GPU Dispatch (20 min)

**File**: `src/domain/layers/ssm/mod.rs`

```rust
impl SsmBlock {
    /// Forward pass through GPU with strict no-fallback.
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Validate GPU availability
        self.backend.require_cpu_implemented("SsmBlock::forward_gpu");
        
        // Get unified GPU backend
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        
        // Route through GPU kernels (Mamba or RG-LRU)
        match self.ssm_type {
            SsmType::Mamba => gpu_backend.forward_mamba(input, &self.config),
            SsmType::RgLru => gpu_backend.forward_rg_lru(input, &self.config),
            // ... other variants
        }
    }

    /// Forward pass with automatic backend selection.
    pub fn forward_auto(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.backend.is_gpu() {
            self.forward_gpu(input)
        } else {
            self.forward_cpu(input)  // Existing CPU implementation
        }
    }
}
```

**Note**: SSM has multiple variants (Mamba, RG-LRU, etc.) - each routes to appropriate GPU kernel.

## Phase 5.10.3: TransformerBlock GPU Dispatch (20 min)

**File**: `src/domain/layers/transformer/block.rs`

```rust
impl TransformerBlock {
    /// Forward pass through GPU with strict no-fallback.
    pub fn forward_gpu(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Validate GPU availability
        self.backend.require_cpu_implemented("TransformerBlock::forward_gpu");
        
        // Get unified GPU backend
        let gpu_backend = UnifiedGpuBackend::auto_detect()?;
        
        // Route through GPU kernels
        gpu_backend.forward_transformer(
            input,
            &self.attention_config,
            &self.feedforward_config,
        )
    }

    /// Forward pass with automatic backend selection.
    pub fn forward_auto(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        if self.backend.is_gpu() {
            self.forward_gpu(input)
        } else {
            self.forward_cpu(input)  // Existing CPU implementation
        }
    }
}
```

## Verification Checklist (Per Block)

- [ ] `forward_gpu()` method compiles
- [ ] `require_cpu_implemented()` validation in place
- [ ] `UnifiedGpuBackend::auto_detect()` called
- [ ] GPU kernels routed correctly
- [ ] Error handling for GPU initialization failure
- [ ] GPU test added (check GPU forward matches CPU semantics)
- [ ] No warnings on compile
- [ ] All existing tests still pass

## Test Template for GPU Forward Pass

```rust
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
#[test]
fn test_diffusion_block_gpu_forward() {
    use ndarray::Array2;
    use llm::domain::layers::diffusion::DiffusionBlock;
    use llm::domain::compute_backend::resolve_compute_backend_strict_auto_gpu;

    // Skip if GPU not available
    let backends = llm::domain::compute_backend::detect_available_gpu_backends();
    if backends.is_empty() {
        return;
    }

    // Create block with GPU backend
    let backend = resolve_compute_backend_strict_auto_gpu().unwrap();
    let mut block = DiffusionBlock::new(...);
    block.set_compute_backend(backend);

    // Test GPU forward
    let input = Array2::<f32>::zeros((32, 512));
    let output = block.forward_gpu(&input).expect("GPU forward failed");

    // Verify output shape
    assert_eq!(output.dim(), (32, 512));

    // TODO: Add numerical validation vs CPU version
    // let cpu_output = block.forward_cpu(&input).unwrap();
    // assert!(output.approx_eq(&cpu_output, 1e-4));
}
```

## Common Issues & Solutions

### Issue 1: `UnifiedGpuBackend::auto_detect() not found`
**Solution**: Ensure GPU features are compiled: `cargo build --release --features gpu-cuda`

### Issue 2: `forward_diffusion() doesn't exist`
**Solution**: Add stub methods to `UnifiedGpuBackend` if missing
```rust
pub fn forward_diffusion(&self, input: &Array2<f32>, config: &DiffusionConfig, ...) -> Result<Array2<f32>> {
    todo!("Implement GPU diffusion forward")
}
```

### Issue 3: Tests panic on `require_cpu_implemented()`
**Solution**: This is correct behavior! Use `forward_gpu()` only when GPU is actually selected.

### Issue 4: GPU forward output doesn't match CPU
**Solution**: Check:
- Data layout (row-major vs column-major)
- Floating point precision (use approx equality)
- Numerical stability (initialization, activation functions)

## Integration Flow

```
Model::train()
  ├─ resolve_compute_backend(preference)?  // Get backend
  │
  └─ For each batch:
      ├─ DiffusionBlock::forward_auto()
      │   ├─ If GPU: forward_gpu() → UnifiedGpuBackend
      │   └─ If CPU: forward_cpu() → existing implementation
      │
      ├─ TransformerBlock::forward_auto()
      │   ├─ If GPU: forward_gpu() → UnifiedGpuBackend
      │   └─ If CPU: forward_cpu() → existing implementation
      │
      └─ SsmBlock::forward_auto()
          ├─ If GPU: forward_gpu() → UnifiedGpuBackend
          └─ If CPU: forward_cpu() → existing implementation
```

## Phase 5.10 Deliverables

1. ✅ DiffusionBlock GPU dispatch
2. ✅ SsmBlock GPU dispatch  
3. ✅ TransformerBlock GPU dispatch
4. ✅ GPU forward tests (3 test functions)
5. ✅ Integration test (forward_auto selection)
6. ✅ All lib tests still passing
7. ✅ Build succeeds with/without GPU features

## Timeline

| Milestone | Est. Time | Status |
|-----------|-----------|--------|
| DiffusionBlock dispatch | 20 min | 📋 Pending |
| SsmBlock dispatch | 20 min | 📋 Pending |
| TransformerBlock dispatch | 20 min | 📋 Pending |
| Tests & verification | 10 min | 📋 Pending |
| **Total** | **~70 min** | 📋 **Ready** |

## Success Criteria

- ✅ `cargo build --release` succeeds (0 warnings)
- ✅ `cargo test --lib` passes (600+ tests)
- ✅ GPU dispatch methods added to all 3 blocks
- ✅ GPU forward tests pass (or skip gracefully on CPU-only systems)
- ✅ CPU/GPU auto-selection working
- ✅ No silent CPU fallback when GPU selected
- ✅ Clear error messages on GPU issues

## Next Phase (5.11)

After 5.10 completes:

1. **GPU Kernel Fusion**
   - Combine attention QKV + scoring + V into 1 kernel
   - Combine RichardsGLU operations into 1 kernel
   - Combine Mamba scan + projection into 1 kernel

2. **Backward Pass** (5.11.5+)
   - Gradient computation for attention
   - Gradient computation for feedforward
   - Gradient computation for temporal ops

3. **Memory Optimization** (5.12)
   - Power-of-2 buffer pooling
   - Zero-copy buffer reuse
   - GPU memory fragmentation tracking

4. **Performance Profiling** (5.13)
   - GPU kernel timing
   - Memory bandwidth analysis
   - Backend comparison benchmarks

---

**Ready to start Phase 5.10**: Begin with DiffusionBlock integration
