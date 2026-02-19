# Phase 5.6.3: Quick Reference Card

## 🎯 Project: Fused GPU Kernels for Shared Components

**Status**: Foundation complete, ready for kernel implementation
**Next**: Implement 4 fused kernels with performance targets

---

## 📊 Performance Targets

| Component | Operation | Launches | CPU | GPU | Speedup |
|-----------|-----------|----------|-----|-----|---------|
| **RichardsGLU** | fused | 2 | 50ms | 2ms | **25x** |
| **PolyAttention** | fused | 1 | 30ms | 1ms | **30x** |
| **Mamba Scan** | kernel | 1 | 40ms | 2ms | **20x** |
| **AttentionContext** | GEMM | 1 | 15ms | 0.5ms | **30x** |

---

## 🏗️ Module Structure

```
src/domain/layers/components/fused_kernels_module.rs
├── richards_glu_fused            [PRIORITY 1] Two-pass fused kernel
├── poly_attention_fused          [PRIORITY 2] Single-pass fused kernel  
├── mamba_scan_kernel             [PRIORITY 3] Recurrent optimization
└── attention_context_ops         [PRIORITY 4] GPU GEMM wrapper
```

---

## 🔧 Implementation Checklist (Per Kernel)

### Priority 1: RichardsGLU Fused

- [ ] **Design** (DONE)
  - Two-pass strategy: Pass 1 (projection+activation+gating) + Pass 2 (output proj)
  - Parameters: batch_size, input_dim, hidden_dim, output_dim, Richards curve params

- [ ] **Implement Kernels**
  - [ ] WGSL: `src/domain/compute/gpu_kernels/richards_glu_fused.wgsl`
  - [ ] CUDA: `src/domain/compute/gpu_kernels/richards_glu_fused.cu`
  - [ ] Metal: `src/domain/compute/gpu_kernels/richards_glu_fused.metal`

- [ ] **Rust Integration**
  - [ ] Update `execute()` in `fused_kernels_module.rs`
  - [ ] Wire into `RichardsGlu::forward_gpu_fused()`
  - [ ] Update `feedforward_gpu.rs` dispatch

- [ ] **Testing**
  - [ ] Create `tests/gpu_richardson_glu_fused.rs`
  - [ ] Numerical accuracy test (ε ≤ 1e-4)
  - [ ] Kernel launch count test (exactly 2)
  - [ ] Performance benchmark (target 25x)

### Priority 2: PolyAttention Fused
- [ ] Design two-pass or single-pass strategy
- [ ] Implement WGSL/CUDA/Metal kernels
- [ ] Rust integration in `temporal_processing_gpu.rs`
- [ ] Testing (ε ≤ 1e-3, exactly 1 launch, 30x speedup)

### Priority 3: Mamba Selective Scan
- [ ] Design recurrent scan kernel
- [ ] Implement WGSL/CUDA/Metal kernels
- [ ] Rust integration in `temporal_processing_gpu.rs`
- [ ] Testing (ε ≤ 1e-3, 20x speedup)

### Priority 4: AttentionContext GPU Ops
- [ ] Use existing GEMM operations
- [ ] Implement in `attention_context_gpu.rs`
- [ ] Testing (ε ≤ 1e-4, 30x speedup)

---

## 📝 Key Implementation Pattern

### Rust Wrapper Template
```rust
pub fn execute(
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &Array2<f32>,
    // ... weights, params ...
) -> Result<Array2<f32>> {
    let mut dev = device.lock()?;
    let (pool, ops) = dev.execution_context();
    
    // 1. Upload inputs
    let input_buf = pool.upload(input.as_slice().unwrap())?;
    // ... upload weights ...
    
    // 2. Allocate intermediates & output
    let mut gated_buf = pool.allocate(size)?;
    let mut output_buf = pool.allocate(size)?;
    
    // 3. Execute Pass 1 kernel
    ops.fused_kernel_pass1(pool, ...)?;
    
    // 4. Execute Pass 2 kernel (or standard GEMM)
    ops.gemm_f32(pool, ...)?;
    
    // 5. Download result
    let mut output = Array2::zeros((batch_size, output_dim));
    pool.download(&output_buf, output.as_slice_mut().unwrap())?;
    
    Ok(output)
}
```

---

## 🧪 Testing Pattern

```rust
#[test]
fn test_kernel_execution() {
    if let Ok(device) = GpuDevice::auto_detect() {
        // Create test data
        let input = Array2::random(...);
        let params = KernelParams::new(...);
        
        // Execute kernel
        let result = kernel_module::execute(&device, &input, &params)?;
        
        // Verify shape, accuracy, launch count
        assert_eq!(result.dim(), expected_shape);
        assert!(max_diff <= tolerance);
    }
}
```

---

## 📚 Documentation Files

1. **`PHASE5.6_CONSOLIDATION_ACTION_PLAN_FEB15.md`**
   - Complete roadmap and priorities
   - Phase 5.6.4 and 5.6.5 plans
   - Success metrics

2. **`PHASE5.6.3_FUSED_KERNELS_IMPLEMENTATION_GUIDE.md`**
   - RichardsGLU detailed walkthrough
   - WGSL pseudocode
   - Integration patterns
   - Common pitfalls

3. **`SESSION_PHASE5.6_CONSOLIDATION_KICKOFF_FEB15.md`**
   - Session summary
   - What was accomplished
   - Next steps

---

## ✅ Success Criteria

Each kernel must meet:
- ✅ Compile without errors/warnings
- ✅ Numerical accuracy ≤ tolerance (1e-4 for attention, 1e-3 for others)
- ✅ Correct kernel launch count (2 for RichardsGLU, 1 for others)
- ✅ Tests pass on auto-detected GPU
- ✅ Performance target met (25x, 30x, 20x, 30x)
- ✅ Zero fallback to CPU (strict GPU-only)

---

## 🚀 Build & Test Commands

```bash
# Check compilation
cargo check --lib

# Run GPU tests
cargo test --test gpu_richardson_glu_fused
cargo test --test gpu_shared_components_phase56

# Format & lint
cargo fmt
cargo clippy --all-targets

# Benchmark (once implemented)
cargo bench --bench gpu_fused_kernels
```

---

## 🔗 Integration Points

### RichardsGLU
- Entry: `RichardsGlu::forward_gpu()` → `forward_gpu_fused()`
- Module: `fused_kernels_module::richards_glu_fused::execute()`
- Call site: `src/domain/richardson/richards_glu.rs` line ~307

### PolyAttention
- Entry: `TemporalMixingLayer::forward_gpu_dispatch()`
- Module: `fused_kernels_module::poly_attention_fused::execute()`
- Call site: `src/domain/layers/components/temporal_processing_gpu.rs` line ~64

### Mamba Scan
- Entry: `TemporalMixingLayer::forward_gpu_mamba()`
- Module: `fused_kernels_module::mamba_scan_kernel::execute()`
- Call site: `src/domain/layers/components/temporal_processing_gpu.rs` line ~92

### AttentionContext
- Entry: `SharedAttentionContext` GPU methods
- Module: `fused_kernels_module::attention_context_ops`
- Call site: `src/domain/layers/components/attention_context_gpu.rs`

---

## 🎓 WGSL Kernel Structure

```wgsl
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Kernel computation
}
```

---

## 📦 Files Modified This Session

✅ `src/domain/layers/components/unified_gpu_backend.rs` - Unused import removed
✅ `src/domain/layers/components/mod.rs` - Added fused_kernels_module
✅ `src/domain/richardson/richards_glu.rs` - Fixed unused variable warnings
✅ Created `src/domain/layers/components/fused_kernels_module.rs` - 240 lines

---

## 💡 Pro Tips

1. **Start with RichardsGLU** - Most straightforward two-pass pattern
2. **Use WGSL for reference** - Easier to debug than CUDA/Metal
3. **Test single GPU backend first** - Easier debugging vs multi-backend
4. **Validate intermediate buffers** - Check Pass 1 output before Pass 2
5. **Benchmark early** - Establish baselines immediately
6. **Document assumptions** - Thread block sizes, shared memory layout

---

## 🆘 Troubleshooting

**Compilation fails on GPU kernel methods**
→ Ensure `GpuMatrixOps` trait has required methods

**Tests fail with numerical errors**
→ Check epsilon values, tensor layout, and weight transposition

**Kernel launches showing wrong count**
→ Verify each operation calls `ops.method()` exactly once per pass

**Performance not matching target**
→ Profile with GPU profiler, check for stalls/occupancy issues

---

## 📅 Timeline

- **Now**: Foundation complete, module structure ready
- **Next 2 hours**: RichardsGLU kernel design & WGSL skeleton
- **Next 4 hours**: RichardsGLU CUDA/Metal & Rust integration
- **Next 6 hours**: Testing, validation, benchmarking
- **Week 2**: PolyAttention + Mamba implementations
- **Week 3**: AttentionContext + Zero-copy pipeline
- **Week 4**: Optimization & Phase 5.6 completion

---

**Last Updated**: Feb 15, 2026
**Author**: GPU Consolidation Task Force
**Status**: Ready for Implementation Phase
