# Session Execution Plan: Phase 5.6 GPU Consolidation

**Current Status**: Phase 5.6 Initialization  
**Objective**: Implement fused kernels and consolidate shared GPU components  
**Approach**: Strict GPU-first with automatic detection, no fallback  
**Expected Outcome**: 25-30x speedup on core components  

---

## Part 1: Foundation & Validation (HOURS 1-3)

### 1.1 Verify GPU Detection Works
**Status**: ⏳ NEXT

```bash
# Test automatic GPU detection
cargo test --lib gpu_device::tests::test_auto_detect -- --exact --nocapture

# Expected output:
# - GPU detected: [WGPU|CUDA|Metal|Vulkan]
# - Backend initialized successfully
```

**Success Criteria**:
- ✅ System reports detected GPU (or "none" on CPU-only systems)
- ✅ No fallback to CPU (strict error policy working)
- ✅ Device initialization succeeds

### 1.2 Test RichardsGlu GPU Forward Pass
**Status**: ⏳ NEXT

```bash
cargo test --lib richards_glu::tests::test_forward_gpu_basic -- --exact --nocapture
```

**Test File**: `src/domain/richards/richards_glu.rs`  
**What to verify**:
- `enable_gpu_auto_detect()` initializes GPU
- `forward_gpu()` executes without panic
- Output shape is correct: (batch_size, output_dim)
- No CPU fallback attempted

### 1.3 Numerical Validation: CPU vs GPU
**Status**: ⏳ NEXT

Create test in `src/domain/richards/richards_glu.rs`:

```rust
#[test]
fn test_gpu_cpu_numerical_match() {
    let mut layer = RichardsGlu::new(768, 3072);
    let input = Array2::random((8, 768), Normal::new(0.0, 0.1).unwrap());
    
    // CPU forward
    let output_cpu = layer.forward(&input);
    
    // GPU forward
    layer.enable_gpu_auto_detect().expect("GPU required");
    let output_gpu = layer.forward_gpu(&input).expect("GPU forward failed");
    
    // Verify match: L2 distance ≤ 1e-4
    let diff = (&output_cpu - &output_gpu).norm_l2();
    assert!(diff < 1e-4 * output_cpu.norm_l2(), 
            "GPU/CPU mismatch: {} (relative error > 0.01%)", diff);
}
```

**Expected Results**:
- GPU output matches CPU within ±1e-4 relative error
- Numerical stability confirmed across batch sizes

---

## Part 2: Optimization Strategy & Backward Pass (HOURS 3-6)

### 2.1 Implement Backward Pass Integration
**Status**: ⏳ TODO

**Location**: `src/domain/richards/richards_glu.rs`

**Implementation**:
```rust
/// Backward pass through GPU operations
pub fn backward_gpu(&mut self, grad_output: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()?;
    let mut device = device_arc.lock().unwrap();
    let (pool, ops) = device.execution_context();
    
    // Backward: gradient flows from output to input
    // grad_output @ w_out^T → grad_hidden
    // grad_hidden uses chain rule through Richards activation
    
    let mut grad_input = pool.allocate(input_size)?;
    
    // Compute: grad_hidden = grad_output @ w_out^T
    ops.gemm_f32(
        pool,
        1.0,
        &grad_output_buf,
        &cache.w_out_t,  // Transposed W_out
        0.0,
        &mut grad_hidden_buf,
        ...
    )?;
    
    // Apply chain rule through Richards activation
    ops.richards_curve_backward(pool, ...)?;
    
    // Accumulate parameter gradients
    self.optimizer_w1.apply_gradient(...)?;
    self.optimizer_w2.apply_gradient(...)?;
    self.optimizer_w_out.apply_gradient(...)?;
    
    Ok(grad_input)
}
```

**Tests to Add**:
1. Gradient shapes correct
2. Gradients non-zero (parameters learning)
3. Gradient accumulation working
4. Parameter updates moving in loss-reducing direction

### 2.2 Implement Memory Pool Reuse
**Status**: ⏳ TODO

**Goal**: Zero allocation per forward pass (reuse buffers)

**Pattern**:
```rust
pub struct RichardsGluWorkspace {
    // Intermediate buffers reused across forward passes
    pub x1: GpuBuffer,
    pub x2: GpuBuffer,
    pub value: GpuBuffer,
    pub gate: GpuBuffer,
    pub gated: GpuBuffer,
    pub output: GpuBuffer,
    
    // Cached sizes for reallocation trigger
    cached_batch_size: usize,
    cached_hidden_dim: usize,
}

impl RichardsGlu {
    pub fn ensure_workspace(&mut self, pool: &mut dyn GpuMemoryPool, 
                            batch_size: usize) -> Result<()> {
        if let Some(ws) = &self.gpu_workspace {
            if ws.cached_batch_size == batch_size {
                return Ok(()); // Reuse existing buffers
            }
        }
        // Reallocate only if size changes
        self.gpu_workspace = Some(RichardsGluWorkspace::allocate(
            pool, batch_size, self.w1.ncols())?);
        Ok(())
    }
}
```

---

## Part 3: Fused Kernel Implementation (HOURS 6-9)

### 3.1 WGSL Fused Kernel (Correct Implementation)
**Status**: ⏳ TODO

**File**: `src/domain/compute/wgpu_ops.rs`

**Better Approach**: Two-pass with streaming (no intermediate download)

```wgsl
// PASS 1: Compute hidden dimension
// x1 = input @ w1, x2 = input @ w2, value, gate, gated
// Kernel 1: [batch, hidden] → [batch, hidden] intermediate
// Launch: (batch_size, hidden_dim)

// PASS 2: Project to output
// output = gated @ w_out
// Kernel 2: [batch, hidden] → [batch, output]
// Launch: (batch_size, output_dim)

// Combined via GPU memory pool (keep gated on GPU between passes)
```

**Register implementation in GPU ops**:
```rust
pub fn richards_glu_fused_pass1(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    input: &GpuBuffer,
    w1: &GpuBuffer,
    w2: &GpuBuffer,
    params: &RichardsGluParams,
    output_hidden: &mut GpuBuffer,  // [batch, hidden]
) -> Result<()> { ... }

pub fn richards_glu_fused_pass2(
    &mut self,
    pool: &mut dyn GpuMemoryPool,
    hidden: &GpuBuffer,
    w_out: &GpuBuffer,
    params: &RichardsGluParams,
    output: &mut GpuBuffer,  // [batch, output]
) -> Result<()> { ... }
```

### 3.2 Integration with RichardsGlu
**Status**: ⏳ TODO

```rust
impl RichardsGlu {
    pub fn forward_gpu_optimized(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let device_arc = self.gpu_device.as_ref()?;
        let mut device = device_arc.lock().unwrap();
        let (pool, ops) = device.execution_context();
        
        // 1. Upload input once
        let input_buf = pool.upload(input.as_slice()?)?;
        
        // 2. Ensure workspace (zero-alloc reuse)
        self.ensure_workspace(pool, input.nrows())?;
        
        // 3. Execute Pass 1 (hidden computation)
        ops.richards_glu_fused_pass1(
            pool, &input_buf, &cache.w1, &cache.w2,
            &params, &mut hidden_buf
        )?;
        
        // 4. Execute Pass 2 (output projection)
        ops.richards_glu_fused_pass2(
            pool, &hidden_buf, &cache.w_out,
            &params, &mut output_buf
        )?;
        
        // 5. Download once
        let mut output = Array2::zeros((batch_size, output_dim));
        pool.download(&output_buf, output.as_slice_mut()?)?;
        
        // Stats: 2 GPU launches, 2 transfers (input up, output down)
        // vs 5+ launches and 10+ transfers in naive approach
        
        Ok(output)
    }
}
```

---

## Part 4: Integration & Testing (HOURS 9-12)

### 4.1 Full Integration Test
**Status**: ⏳ TODO

```bash
cargo test --test gpu_shared_components_phase56 -- --nocapture
```

**Test Coverage**:
- [ ] RichardsGlu forward/backward
- [ ] Poly Attention forward/backward
- [ ] Mamba Selective Scan forward
- [ ] Attention Context GPU ops
- [ ] Zero-allocation reuse verified
- [ ] Numerical validation (<1e-4 error)
- [ ] Batch robustness (1-1024)
- [ ] No CPU fallback (strict errors)

### 4.2 Performance Benchmarks
**Status**: ⏳ TODO

Create `benches/phase56_gpu_optimization.rs`:

```rust
#[bench]
fn bench_richards_glu_gpu_vs_cpu(b: &mut Bencher) {
    let mut layer = RichardsGlu::new(768, 3072);
    let input = Array2::random((1024, 768), Normal::new(0.0, 0.1).unwrap());
    
    // Enable GPU
    layer.enable_gpu_auto_detect().expect("GPU required");
    
    b.iter(|| {
        layer.forward_gpu(&input).expect("GPU forward failed")
    });
}
```

**Expected Results**:
- RichardsGLU: 25x speedup target
- PolyAttention: 30x speedup target
- Mamba: 20x speedup target

---

## Part 5: Consolidation & Cleanup (HOURS 12+)

### 5.1 Remove Deprecated Code
**Status**: TODO

- [ ] Remove `shared_gpu_manager.rs` (deprecated module)
- [ ] Consolidate `shared_gpu_ops.rs` into unified backend
- [ ] Remove duplicate shader definitions in `wgpu_ops.rs`
- [ ] Clean up feature flags (gpu-wgpu vs wgpu)

### 5.2 Documentation Updates
**Status**: TODO

- [ ] Update AGENTS.md with GPU consolidation commands
- [ ] Document fused kernel strategy in code comments
- [ ] Create migration guide for new GPU API
- [ ] Add troubleshooting guide for GPU detection failures

### 5.3 Final Integration
**Status**: TODO

**Verify all components use unified backend**:
- `SharedAttentionContext` → uses `UnifiedGpuBackend`
- `SharedFeedforward` → uses `UnifiedGpuBackend`
- `SharedTemporalProcessing` → uses `UnifiedGpuBackend`
- Diffusion blocks → GPU-enabled
- SSM variants → GPU-enabled
- Transformer blocks → GPU-enabled

---

## Metrics & Success Criteria

### Phase Completion
- [ ] GPU detection works (auto_detect returns device or error, never CPU)
- [ ] RichardsGlu GPU forward matches CPU output ±1e-4
- [ ] RichardsGlu GPU backward pass integrates
- [ ] Fused kernels reduce launches (5 → 2 for RichardsGLU)
- [ ] Zero-allocation workspace reuse working
- [ ] All numerical tests pass
- [ ] Batch robustness (1-1024) verified
- [ ] No deprecated code in use
- [ ] Documentation complete

### Performance Metrics
| Component | CPU Time | GPU Time | Speedup | Status |
|-----------|----------|----------|---------|--------|
| RichardsGLU (1K batch) | 50ms | 2ms | 25x | ⏳ |
| PolyAttention (512 batch) | 30ms | 1ms | 30x | ⏳ |
| Mamba (512 batch) | 40ms | 2ms | 20x | ⏳ |
| AttentionContext (1K batch) | 15ms | 0.5ms | 30x | ⏳ |

---

## Commands for This Session

```bash
# Build with all GPU features
cargo build --release --features gpu-all

# Run tests with GPU
cargo test --lib gpu_device::tests::test_auto_detect -- --exact --nocapture
cargo test --lib richards_glu::tests::test_forward_gpu_basic -- --exact --nocapture
cargo test --lib richards_glu::tests::test_gpu_cpu_numerical_match -- --exact --nocapture

# Run full integration test suite
cargo test --test gpu_shared_components_phase56 -- --nocapture

# Benchmark performance
cargo bench --bench phase56_gpu_optimization

# Check compilation (quick validation)
cargo check --lib --features gpu-all

# Format before commit
cargo fmt
cargo clippy --all-targets --features gpu-all
```

---

## References

**Previous Threads**:
- @T-019c63d7-7061-76bc-941d-256b8ec4020f - Consolidation strategy

**Implementation Guides**:
- Fused Kernel Pattern: Combine N operations into 1 GPU launch
- Memory Pool Reuse: Power-of-2 sizing, zero-alloc per forward
- Numerical Stability: Log-sum-exp trick for softmax, clamp exponents
- Strict GPU-First: Error on no GPU, never fallback to CPU

**Target Algorithms**:
- Richards Activation: σ(x) = 1 / (1 + (k*m)^(1/m) * exp(-β*(x-ν)))
- Poly Attention: Polynomial basis approximation of attention
- Mamba Selective Scan: Parallelizable scan with input-dependent transitions
