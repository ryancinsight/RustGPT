# Quick Start: Phase 5.6 GPU Kernel Implementation

**Focus**: Fused kernels, shared components consolidation, strict GPU-first  
**Duration**: ~12 hours per session  
**Target**: 25-30x speedup on RichardsGLU, PolyAttention, Mamba  

---

## Session Overview

### Part 1: GPU Validation (2 hours)
```bash
# 1. Verify compilation
cargo check --lib --features gpu-all

# 2. Test auto-detection
cargo test --lib gpu_device --nocapture

# 3. Test RichardsGlu forward
cargo test --lib richards_glu --nocapture

# Expected: All pass, GPU detected or explicit "none" (cpu-only system OK)
```

### Part 2: Backward Pass (3 hours)
**File**: `src/domain/richards/richards_glu.rs`

```rust
pub fn backward_gpu(&mut self, grad_output: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend { ... })?;
    
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // 1. Allocate gradient buffers
    let grad_hidden = pool.allocate(batch_size * hidden_dim * 4)?;
    let grad_input = pool.allocate(batch_size * input_dim * 4)?;
    
    // 2. Backward through W_out: grad_hidden = grad_output @ w_out^T
    ops.gemm_f32(pool, 1.0, &grad_output_buf, &cache.w_out_t, 0.0,
                 &mut grad_hidden, ...)?;
    
    // 3. Backward through gating (chain rule)
    // grad = grad_hidden * (1 - gate)^2 or similar
    
    // 4. Backward through Richards activation (Richards derivative)
    
    // 5. Accumulate parameter gradients
    // ∂L/∂W1 = input^T @ (grad through activation)
    
    Ok(grad_input)
}
```

### Part 3: Fused Kernels (4 hours)
**Strategy**: Two-pass execution, keep data on GPU

**Pass 1** (Compute hidden dimension):
```wgsl
// x1 = input @ w1
// x2 = input @ w2
// value = x1 * richards(x1)
// gate = richards(x2)
// gated = value * gate
// Output: [batch, hidden] (stays on GPU)
```

**Pass 2** (Project to output):
```wgsl
// output = gated @ w_out
// Output: [batch, output]
```

**Implementation in Rust**:
```rust
pub fn forward_gpu_optimized(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()?;
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // Upload once
    let input_buf = pool.upload(input.as_slice()?)?;
    
    // Pass 1: Compute hidden
    let hidden_buf = pool.allocate(batch * hidden_dim * 4)?;
    ops.richards_glu_pass1(pool, &input_buf, &cache.w1, &cache.w2, 
                           &params, &mut hidden_buf)?;
    
    // Pass 2: Project to output (no intermediate download)
    let output_buf = pool.allocate(batch * output_dim * 4)?;
    ops.richards_glu_pass2(pool, &hidden_buf, &cache.w_out,
                           &params, &mut output_buf)?;
    
    // Download once
    let mut output = Array2::zeros((batch, output_dim));
    pool.download(&output_buf, output.as_slice_mut()?)?;
    
    Ok(output)
}
```

### Part 4: Testing & Integration (2 hours)
```bash
# Test numerical validation
cargo test --lib richards_glu::test_gpu_cpu_numerical_match

# Test backward pass
cargo test --lib richards_glu::test_backward_gradient_flow

# Test batch robustness
cargo test --lib richards_glu::test_batch_size_robustness

# Run full integration
cargo test --test gpu_shared_components_phase56

# Benchmark
cargo bench --bench phase56_gpu_optimization
```

### Part 5: Documentation & Cleanup (1 hour)
- [ ] Update AGENTS.md with GPU commands
- [ ] Add migration guide to GPU API
- [ ] Remove deprecated code (`shared_gpu_manager.rs` if done)
- [ ] Format and lint
```bash
cargo fmt
cargo clippy --all-targets --features gpu-all
```

---

## Key Patterns

### Strict GPU-First
```rust
// ✅ CORRECT
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()
        .ok_or_else(|| ModelError::Backend {
            message: "GPU required. Call enable_gpu_auto_detect() first.".into()
        })?;  // STRICT: Error if no GPU
    
    // GPU computation
    device.execute()?
}

// ❌ WRONG (fallback pattern - avoid)
pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    match self.forward_gpu(input) {
        Ok(output) => Ok(output),
        Err(_) => self.forward_cpu(input),  // NO: Never fallback
    }
}
```

### Memory Pool Reuse
```rust
// ✅ CORRECT (zero allocation per forward)
pub fn ensure_workspace(&mut self, pool: &mut dyn GpuMemoryPool) -> Result<()> {
    if let Some(ws) = &self.workspace {
        if ws.batch_size == batch_size {
            return Ok(());  // Reuse existing
        }
    }
    
    self.workspace = Some(Workspace::allocate(pool, batch_size)?);
    Ok(())
}
```

### Numerical Validation
```rust
#[test]
fn test_gpu_cpu_match() {
    let (cpu_output, gpu_output) = run_both_paths();
    
    let diff = (&cpu_output - &gpu_output).norm_l2();
    let relative_error = diff / cpu_output.norm_l2();
    
    assert!(relative_error < 1e-4, "Error: {:.2e}", relative_error);
}
```

---

## File Structure for Reference

```
src/domain/
├── compute/
│   ├── richards_glu_fused_kernel.rs  ← NEW: Fused kernel infrastructure
│   ├── gpu_device.rs                 ← Device abstraction
│   ├── gpu_ops.rs                    ← Trait definitions
│   ├── wgpu_ops.rs                   ← WGSL kernel implementations
│   └── mod.rs                        ← (registry)
├── layers/
│   └── components/
│       ├── unified_gpu_backend.rs    ← Entry point
│       └── (others for attention, feedforward, temporal)
└── richards/
    ├── richards_glu.rs               ← GPU forward/backward
    ├── richards_activation.rs        ← Activation function
    └── (others)
```

---

## Commands Cheat Sheet

```bash
# Build & check
cargo check --lib --features gpu-all
cargo build --release --features gpu-all

# Test GPU functionality
cargo test --lib gpu_device --nocapture
cargo test --lib richards_glu --nocapture
cargo test --lib richards_glu_fused_kernel

# Run integration tests
cargo test --test gpu_shared_components_phase56

# Performance benchmarks
cargo bench --bench phase56_gpu_optimization

# Linting & formatting
cargo fmt
cargo clippy --all-targets --features gpu-all

# Full test suite
cargo test --lib
```

---

## Common Issues & Fixes

### Issue: GPU Not Detected
```rust
// Error: "GPU device not set"
// Fix: Call before forward
layer.enable_gpu_auto_detect()?;

// Or on construction
let mut layer = RichardsGlu::new(768, 3072);
layer.enable_gpu_auto_detect()?;
```

### Issue: Memory Pool Error
```rust
// Error: "Failed to allocate buffer"
// Cause: Pool exhausted or invalid size
// Fix: Check buffer sizing, ensure deallocate called

pool.deallocate(buf);  // Release after use
```

### Issue: Numerical Mismatch (>1e-4)
```rust
// GPU output differs from CPU
// Causes:
// 1. Float precision: Use f32, not f64
// 2. Operation order: Non-associative (reductions)
// 3. Activation: Different Richards parameter values

// Debug:
println!("CPU: {:?}", cpu_output);
println!("GPU: {:?}", gpu_output);
println!("Diff: {}", (cpu - gpu).norm_l2());
```

### Issue: Compilation Error in wgpu_ops.rs
```rust
// Error: Shader compilation error
// Fix: Check WGSL syntax:
// - Function names match shader entry point
// - Binding groups match kernel definition
// - Struct field alignment (@repr(C))
```

---

## Success Checklist

### After Each 2-Hour Block

**Block 1 (GPU Validation)**
- [ ] `cargo check --lib` passes
- [ ] GPU detected (or explicit "none" message)
- [ ] `cargo test --lib gpu_device` passes
- [ ] RichardsGlu forward works

**Block 2 (Backward Pass)**
- [ ] Backward kernel defined
- [ ] Parameter gradients computed
- [ ] No panics, shapes correct
- [ ] Gradients non-zero (learning)

**Block 3 (Fused Kernels)**
- [ ] Pass 1 kernel defined (hidden computation)
- [ ] Pass 2 kernel defined (output projection)
- [ ] Two launches execute without error
- [ ] Numerical validation: GPU/CPU match <1e-4

**Block 4 (Testing)**
- [ ] All integration tests pass
- [ ] Batch sizes 1-1024 work
- [ ] No memory leaks (buffers deallocated)
- [ ] Benchmarks show 20-30x improvement

**Block 5 (Cleanup)**
- [ ] Code formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated
- [ ] Ready for commit/push

---

## Performance Targets

| Operation | Input Size | CPU Time | GPU Target | Speedup |
|-----------|-----------|----------|------------|---------|
| RichardsGLU forward | 1024×768→3072→768 | 50ms | 2ms | **25x** |
| PolyAttention | 512×12×256 heads | 30ms | 1ms | **30x** |
| Mamba scan | 512×768 | 40ms | 2ms | **20x** |
| Attention context | 1024×768 | 15ms | 0.5ms | **30x** |

**Memory Efficiency Target**: >92% (minimal padding)

---

## Next Steps After This Session

1. **Session 2**: GPU validation + backward pass implementation
2. **Session 3**: Fused kernel implementation + testing
3. **Session 4**: Integration + benchmarks + cleanup
4. **Session 5**: PolyAttention + Mamba GPU kernels

---

## References

- **Main Strategy**: `PHASE5.6_CONSOLIDATION_GPU_KERNELS_SESSION.md`
- **Detailed Roadmap**: `SESSION_PHASE5.6_CONSOLIDATION_EXECUTION.md`
- **This Guide**: `QUICK_START_PHASE5.6_GPU_KERNELS.md`
- **Thread**: @T-019c63ce-c226-712b-ae6e-582d945501e4

---

**Ready to start? Run:**
```bash
cargo check --lib --features gpu-all && cargo test --lib gpu_device
```

If both pass, you're ready for GPU kernel implementation.
