# Phase 5.6 GPU Consolidation - Quick Start Guide

**Target Audience**: Developers continuing Phase 5.6 GPU kernel implementation  
**Status**: Foundation complete, ready for GPU kernel work  
**Duration to Complete Phase**: ~1 week of full-time work

---

## What's Ready (This Session - Feb 15)

### ✅ Planning & Documentation
- Comprehensive 320-line implementation plan
- 400-line GPU implementation patterns reference
- 250-line session progress tracker
- Complete architecture documentation

### ✅ Code & Infrastructure
- RichardsGLU GPU kernel (fully functional, no changes needed)
- MixtureOfExperts GPU skeleton (placeholder)
- Test suite foundation with passing GPU auto-detection test
- Updated AGENTS.md with GPU commands

### ✅ Compilation
```bash
cargo check --lib  # ✅ Clean
cargo test --test gpu_shared_components_phase56 gpu_auto_detection_no_fallback  # ✅ PASS
```

---

## What You Need to Do (Next Steps)

### Phase 5.6.1a: RichardsGLU Validation (1-2 hours)
**Goal**: Verify existing RichardsGLU GPU kernel produces correct results

1. **Run validation test**:
```bash
cargo test shared_feedforward_cpu_vs_gpu_richardsglu --test gpu_shared_components_phase56 -- --nocapture
```

2. **If test fails**: Check `GPU_IMPLEMENTATION_PATTERNS.md` for numerical validation pattern

3. **Success criteria**:
- Test passes with max element-wise diff ≤ 1e-4
- Tests with batch sizes: [1, 16, 32, 64, 128]
- Tests with embed dims: [256, 512, 768, 1024]

---

### Phase 5.6.1b: MixtureOfExperts GPU Kernel (2-3 hours)
**Goal**: Implement full GPU path for MixtureOfExperts

1. **Edit file**: `src/domain/mixtures/moe.rs`

2. **Replace the placeholder** in `MixtureOfExperts::forward_gpu()`:
```rust
// CURRENT (placeholder):
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    Ok(self.forward(input))  // CPU fallback
}

// IMPLEMENT (following RichardsGLU pattern):
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // 1. Get GPU device
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU device not set for MoE".to_string(),
        })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // 2. Upload input
    let input_slice = input.as_slice()?;
    let input_buf = pool.upload(input_slice)?;
    
    // 3. Execute MoE kernel (to be implemented)
    self.forward_moe_gpu_kernel(pool, ops, &input_buf, ...)?;
    
    // 4. Download output
    // ... (see RichardsGLU pattern)
    
    Ok(output)
}
```

3. **Reference**: Copy kernel structure from `src/domain/richards/richards_glu.rs::forward_gpu_kernel`

4. **Steps in kernel**:
   - Allocate GPU buffers
   - Router GEMM: `input @ W_router` → routing_logits
   - Softmax on routing_logits
   - For each active expert:
     - Expert GEMM + activation
     - Accumulate with gate weight
   - Final output assembly

5. **Add unit test** to `tests/gpu_shared_components_phase56.rs`:
```rust
#[test]
fn shared_feedforward_cpu_vs_gpu_mixtureofexperts() {
    // Follow pattern from RichardsGLU test
    // Compare CPU vs GPU forward outputs
    // Assert max_diff ≤ 1e-4
}
```

---

### Phase 5.6.2: SharedTemporalProcessing (Feb 18-22)

#### 5.6.2a: PolyAttention GPU Kernel
**File**: `src/domain/attention/poly_attention.rs`  
**Implementation Steps**:
1. Add `forward_gpu()` method
2. Implement polynomial basis computation (kernel fusion candidate)
3. Implement gating mechanism
4. Add numerical validation tests

#### 5.6.2b: Mamba/RG-LRU GPU Kernel
**Files**: `src/domain/layers/ssm/mamba.rs`, `src/domain/layers/ssm/rg_lru.rs`  
**Implementation Steps**:
1. Implement recurrent scan on GPU
2. Fuse state update with activation
3. Add numerical validation tests

#### 5.6.2c: TransformerAttention GPU Kernel
**File**: `src/domain/attention/transformer_attention.rs`  
**Implementation Steps**:
1. QKV projection (3 GEMMs or 1 fused)
2. Attention scores (GEMM + softmax)
3. Context aggregation
4. Output projection
5. Add numerical validation tests

---

### Phase 5.6.3: SharedAttentionContext (Feb 23-24)

**File**: `src/domain/layers/components/attention_context.rs`  
**Implementation**: Fused context modulation kernel
1. Similarity computation: `activation @ activation^T`
2. Softmax normalization
3. Context modulation: `input + (strength / embed_dim) * similarity`
4. Numerical validation tests

---

### Phase 5.6.4: Integration & Benchmarks (Feb 25-28)

**Goal**: Full pipeline testing and performance validation

1. **Integration test**: Full forward pass (Feedforward → Temporal → Attention)
2. **Training loop test**: Forward + backward with GPU
3. **Benchmark suite**: CPU vs GPU performance comparison
4. **Memory tracking**: Verify AllocationStats (reuse_count, resize_count)
5. **Batch size variation**: Test with multiple batch sizes

---

## Quick Reference Commands

### Build & Check
```bash
cargo check --lib                           # Fast check
cargo build --release --features gpu-wgpu  # Full build with GPU
```

### Testing
```bash
# Run single GPU test
cargo test --test gpu_shared_components_phase56 gpu_auto_detection_no_fallback

# Run all GPU tests
cargo test --test gpu_shared_components_phase56

# Run with output
cargo test --test gpu_shared_components_phase56 -- --nocapture

# Run specific test (single-threaded for debugging)
cargo test --test gpu_shared_components_phase56 shared_feedforward_cpu_vs_gpu_richardsglu -- --nocapture --test-threads=1
```

### Formatting
```bash
cargo fmt                    # Format code
cargo fmt -- --check        # Check format
cargo clippy --all-targets  # Lint
```

---

## Key Files Reference

### Documentation (Read These First)
1. **`PHASE5.6_GPU_CONSOLIDATION_IMPLEMENTATION.md`** - Main roadmap
2. **`GPU_IMPLEMENTATION_PATTERNS.md`** - Code patterns and templates
3. **`SESSION_PROGRESS_PHASE5.6_GPU_CONSOLIDATION.md`** - Session details

### Code to Modify
1. **`src/domain/mixtures/moe.rs`** - Add MoE GPU kernel
2. **`src/domain/attention/poly_attention.rs`** - Add PolyAttention GPU (Phase 5.6.2)
3. **`src/domain/layers/ssm/mamba.rs`** - Add Mamba GPU (Phase 5.6.2)
4. **`src/domain/attention/transformer_attention.rs`** - Add Transformer GPU (Phase 5.6.2)
5. **`src/domain/layers/components/attention_context.rs`** - Add context GPU (Phase 5.6.3)

### Tests to Extend
1. **`tests/gpu_shared_components_phase56.rs`** - Main test file
   - Uncomment and implement placeholder tests
   - Add new tests for each component

### Reference Implementations
1. **`src/domain/richards/richards_glu.rs::forward_gpu()`** - Reference GPU kernel
2. **`src/domain/compute/gpu_device.rs`** - GPU device API
3. **`src/domain/compute/gpu_ops.rs`** - GPU operations (GEMM, softmax, etc.)

---

## Common Patterns

### GPU Forward Pass Template
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // 1. Get device
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend { /* ... */ })?
        .clone();
    
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // 2. Upload
    let input_buf = pool.upload(input.as_slice()?)?;
    
    // 3. Allocate output
    let mut output_buf = pool.allocate(output_size)?;
    
    // 4. Execute kernel
    self.forward_gpu_kernel(pool, ops, &input_buf, &mut output_buf)?;
    
    // 5. Download
    let mut result = Array2::zeros(output_dim);
    pool.download(&output_buf, result.as_slice_mut()?)?;
    
    Ok(result)
}
```

### CPU vs GPU Validation Test Template
```rust
#[test]
fn component_gpu_vs_cpu_accuracy() {
    // Create component and set GPU device
    let mut component = Component::new(...);
    let device = GpuDevice::auto_detect().expect("GPU required");
    component.set_gpu_device(Arc::new(Mutex::new(device)));
    
    // Create test input
    let input = create_test_input(batch_size, embed_dim);
    
    // CPU forward
    let cpu_result = component.forward(&input);
    
    // GPU forward
    let gpu_result = component.forward_gpu(&input).expect("GPU forward failed");
    
    // Validate accuracy
    let max_diff = max_element_diff(&gpu_result, &cpu_result);
    assert!(max_diff <= 1e-4, "Max diff: {:.2e}", max_diff);
}
```

---

## Troubleshooting

### "GPU device not set" error
**Problem**: GPU kernel called but device not attached  
**Solution**: Set GPU device before calling GPU forward:
```rust
let device = GpuDevice::auto_detect()?;
component.set_gpu_device(Arc::new(Mutex::new(device)));
```

### Numerical accuracy issues (max_diff > 1e-4)
**Problem**: GPU results don't match CPU  
**Solution**:
1. Check for data type mismatches (f32 vs f64)
2. Verify kernel parameters (temperature, scale, etc.)
3. Check kernel implementation against CPU reference
4. See `GPU_IMPLEMENTATION_PATTERNS.md` for validation patterns

### Compilation errors
**Problem**: Types not found, traits not implemented  
**Solution**:
1. Check imports: `use crate::domain::compute::GpuDevice;`
2. Implement `forward_gpu()` with correct signature
3. Ensure component has `gpu_device: Option<Arc<Mutex<GpuDevice>>>` field

---

## Success Indicators

### After Each Phase
- ✅ All code compiles without errors
- ✅ New GPU tests added and passing
- ✅ CPU vs GPU numerical comparison within tolerance
- ✅ Memory tracking shows positive reuse_count
- ✅ Batch size variation tests pass

### End of Phase 5.6
- ✅ All shared components have GPU kernels
- ✅ Full pipeline test passes (Feedforward → Temporal → Attention)
- ✅ Training loop works with GPU
- ✅ Performance improvements measured and documented
- ✅ All tests passing
- ✅ Zero compilation warnings

---

## Timeline Estimate

| Phase | Time | Deadline |
|-------|------|----------|
| 5.6.1a | 1-2 hrs | Today (Feb 15) |
| 5.6.1b | 2-3 hrs | Tomorrow (Feb 16) |
| 5.6.2a-c | 6-8 hrs | Feb 18-22 |
| 5.6.3 | 2-3 hrs | Feb 23-24 |
| 5.6.4 | 3-4 hrs | Feb 25-28 |
| **Total** | **~15-20 hrs** | **Feb 28** |

---

## Need Help?

1. **Implementation patterns**: See `GPU_IMPLEMENTATION_PATTERNS.md`
2. **Code examples**: Check `src/domain/richards/richards_glu.rs`
3. **Testing patterns**: See `tests/gpu_shared_components_phase56.rs`
4. **Architecture questions**: Read `PHASE5.6_GPU_CONSOLIDATION_IMPLEMENTATION.md`
5. **Progress tracking**: Check `SESSION_PROGRESS_PHASE5.6_GPU_CONSOLIDATION.md`

---

## Good Luck! 🚀

You have everything needed to complete Phase 5.6. Follow the patterns, implement one component at a time, and validate each step. The architecture is solid, and success is within reach.

**Start with**: RichardsGLU validation test (1 hour, quick win)  
**Then continue with**: MixtureOfExperts GPU kernel (2-3 hours)
