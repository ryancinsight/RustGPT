# Phase 5.5 Quick Start - GPU Backend Implementation
**Goal**: Implement WGPU BLAS operations and integrate with components  
**Duration**: 3 sessions (~60 hours)  
**Status**: Ready to begin

---

## Phase Overview

```
Phase 5.5: GPU Consolidation & Implementation
├── Session 1: WGPU BLAS Foundation
│   ├── GEMM (General Matrix Multiply)
│   ├── Element-wise operations
│   ├── Normalization (layer_norm, softmax)
│   └── PolyAttention kernels
│
├── Session 2: Component GPU Paths
│   ├── Diffusion → forward_gpu()
│   ├── SSM → temporal GPU kernels
│   ├── Transformer → unified GPU path
│   ├── PolyAttention → fused GPU ops
│   └── Shared components consolidation
│
└── Session 3: Validation & Benchmarking
    ├── Numerical validation (all ops vs CPU)
    ├── Performance benchmarks
    ├── Component integration tests
    └── Finalize & document
```

---

## Pre-Session Checklist

- [ ] Clone/pull latest: `git pull origin main`
- [ ] Run baseline tests: `cargo test --lib` (expect 539 passing)
- [ ] Check GPU: `nvidia-smi` or `Metal` device availability
- [ ] Read: `WGPU_BLAS_IMPLEMENTATION_GUIDE.md` (sections 1-6)
- [ ] Read: `GPU_CONSOLIDATION_PHASE5.5_ACTION_PLAN.md` (sections 1-3)

---

## Session 1 Agenda: WGPU BLAS Foundation

### Time Allocation
- **Setup** (30 min): Create files, project structure
- **GEMM** (3 hours): Shader + Rust wrapper + tests
- **Element-wise** (2 hours): 5-6 basic operations
- **Normalization** (1.5 hours): Layer norm + softmax
- **Attention kernels** (1 hour): Stub/planning
- **Validation** (1 hour): Test harness setup

### Task 1.1: GEMM Implementation (PRIMARY GOAL)

**Milestone**: GEMM shader compiles and passes 50+ test cases vs CPU reference

#### Step 1: Create Project Structure
```bash
mkdir -p src/domain/compute/wgpu_shaders
touch src/domain/compute/wgpu_shaders/{gemm,activation,softmax,attention}.wgsl
touch src/domain/compute/wgpu_ops_blas.rs
```

#### Step 2: Implement GEMM Shader
**File**: `src/domain/compute/wgpu_shaders/gemm.wgsl`

Template (from WGPU guide):
```wgsl
@compute @workgroup_size(16, 16)
fn gemm_f32(...) {
    // Tile-based matrix multiply with shared memory
    // See WGPU_BLAS_IMPLEMENTATION_GUIDE.md section 3
}
```

**Checklist**:
- [ ] Shader syntax correct (use `naga check gemm.wgsl`)
- [ ] Parameters: alpha, beta, M, N, K, transpose flags
- [ ] Tile size: 16x16 workgroup (balance compute vs shared memory)
- [ ] Barrier calls correct (`workgroupBarrier()`)

#### Step 3: Rust Wrapper
**File**: `src/domain/compute/wgpu_ops.rs` → add to `WgpuMatrixOps`

```rust
impl WgpuMatrixOps {
    pub fn gemm_f32(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        alpha: f32,
        a: &GpuBuffer,
        b: &GpuBuffer,
        beta: f32,
        output: &mut GpuBuffer,
        m: usize, n: usize, k: usize,
        trans_a: bool, trans_b: bool,
    ) -> Result<()> {
        // See WGPU_BLAS_IMPLEMENTATION_GUIDE.md section 6
    }
}
```

**Checklist**:
- [ ] Compile shader to pipeline
- [ ] Create bind groups
- [ ] Dispatch workgroups: ceil(M/16) × ceil(N/16) × 1
- [ ] Synchronize queue

#### Step 4: Test Harness
**File**: `tests/gpu_blas_gemm.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_gemm_f32_small_identity() {
        // 4×4 @ 4×4 identity matrix
    }
    
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_gemm_f32_medium_random() {
        // 128×128 @ 128×128 random matrices
    }
    
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_gemm_f32_large_ones() {
        // 512×512 @ 512×512 matrices of ones
    }
    
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_gemm_f32_vs_cpu_reference() {
        // Random matrices, tolerance ε ≤ 1e-4
    }
}
```

**Validation**:
- [ ] CPU reference: `ndarray` GEMM or manual implementation
- [ ] 10 test cases with different shapes
- [ ] Tolerance check: `(gpu_result - cpu_result).abs() <= 1e-4 * cpu_result.abs()`
- [ ] Test transpose flags (NN, NT, TN, TT)

#### Step 5: Performance Benchmark
```bash
cargo bench --bench gpu_ops_benchmark -- gemm
```

**Target**: 
- Small (64×64): <1ms
- Medium (256×256): 5-20ms
- Large (1024×1024): 20-100ms

---

### Task 1.2: Element-Wise Operations

**Target Operations** (in priority order):
1. `fill_f32` - Initialize buffer
2. `scale` - Multiply by scalar
3. `mul` - Element-wise multiply
4. `relu` - Max(0, x)
5. `sigmoid` - 1/(1+exp(-x))
6. `gelu` - x * Φ(x)

**Per-operation checklist**:
- [ ] WGSL shader compiled
- [ ] Rust wrapper (1-2 hours per op)
- [ ] Test harness (3-5 test cases per op)
- [ ] CPU reference validation
- [ ] Numerical tolerance check

---

### Task 1.3: Normalization

**Target Operations**:
1. `softmax` - exp(x)/sum(exp(x)) row-wise
2. `layer_norm` - (x - mean) / sqrt(var + eps)

**Implementation notes**:
- Softmax: Use log-sum-exp trick (see section 4C.2 in guide)
- Layer norm: Two passes (mean→variance, then normalize)
- Use shared memory for reduction operations

---

### Task 1.4: PolyAttention Kernels (Stub for Session 1)

**For Session 1**: Create empty shell
```rust
impl WgpuMatrixOps {
    pub fn moh_gate_activation(...) -> Result<()> {
        Err(ModelError::Backend {
            message: "TODO: Implement MoH gate in Session 1.4".to_string(),
        })
    }
    
    // ... other PolyAttention stubs
}
```

**For Session 2**: Full implementation

---

## Session 1 End Criteria

```
✓ GEMM working: 50+ tests passing
✓ Element-wise ops: 6/6 implemented
✓ Normalization: 2/2 implemented
✓ Tests: 100+ new GPU operation tests
✓ Compilation: cargo test --lib --features wgpu (all passing)
✓ Benchmarks: Results documented
✓ Commit: "feat: WGPU BLAS Phase 1 - matrix ops and element-wise"
```

---

## Session 2 Agenda: Component Integration

### Task 2.1: Diffusion GPU Path
**File**: `src/domain/diffusion/diffusion_block.rs`

Add method:
```rust
pub fn forward_gpu(
    &self,
    input: &GpuBuffer,
    step: u32,
    device: &mut GpuDevice,
) -> Result<GpuBuffer> {
    // Use device.gemm_f32(), device.relu(), etc.
    // Pipeline: Embedding → Denoising → Output
    // Zero CPU↔GPU transfers
}
```

### Task 2.2: SSM GPU Kernels
**File**: `src/domain/ssm/temporal_mixing_layer.rs`

Add GPU path with recurrent kernels in WGSL.

### Task 2.3: Transformer GPU Path
**File**: `src/domain/transformer/block.rs`

Add method:
```rust
pub fn forward_gpu(
    &self,
    input: &GpuBuffer,
    device: &mut GpuDevice,
) -> Result<GpuBuffer> {
    // Attention → Temporal → Feedforward (all on GPU)
}
```

### Task 2.4: PolyAttention Implementation
**File**: `src/domain/attention/poly_attention.rs`

Complete the PolyAttention GPU kernels from Task 1.4 stubs.

---

## Session 3 Agenda: Validation & Benchmarking

### Task 3.1: Numerical Validation
- [ ] All GPU ops validated vs CPU reference
- [ ] 1000+ random test cases
- [ ] Edge cases: zeros, NaN, Inf, denormalized
- [ ] Tolerance: ε ≤ 1e-4

### Task 3.2: Performance Benchmarking
- [ ] BLAS operations: TFLOPS
- [ ] Component forward passes: Speedup factor
- [ ] End-to-end training step: Speedup factor
- [ ] Memory usage: GPU peak memory, CPU↔GPU transfers

### Task 3.3: Integration Testing
- [ ] Component GPU paths work with rest of pipeline
- [ ] Training loop with GPU components
- [ ] Inference with GPU components

---

## Commands Cheat Sheet

```bash
# Setup
cargo test --lib                    # Baseline (should pass 539)
cargo build --features wgpu         # Build with WGPU

# Testing
cargo test --lib --features wgpu    # All tests with WGPU
cargo test --test gpu_blas_gemm     # Specific test file
cargo test --lib wgpu_gemm_ --      # Specific test function

# Benchmarking
cargo bench --bench gpu_ops_benchmark
cargo bench --bench gpu_ops_benchmark -- gemm

# Validation
naga check src/domain/compute/wgpu_shaders/gemm.wgsl
cargo clippy --all-targets --features wgpu

# Build profiles
cargo build --release --features wgpu
cargo build --features wgpu --target wasm32-unknown-unknown
```

---

## Debugging Tips

### Shader Compilation Errors
```rust
// Check shader source:
println!("{}", shader_source);

// Enable validation:
let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
    flags: wgpu::InstanceFlags::VALIDATION,
    ..Default::default()
});
```

### Numerical Correctness
```rust
// Print first 10 elements:
for (i, &v) in result.iter().take(10).enumerate() {
    println!("result[{}] = {} (expected {})", i, v, cpu_result[i]);
}
```

### Memory Issues
```rust
// Check buffer sizes:
assert_eq!(a.size_bytes(), m * k * std::mem::size_of::<f32>());
```

---

## File Locations Quick Reference

| Component | Path | Entry Point |
|-----------|------|-------------|
| Error types | `src/common/errors.rs` | `ModelError::GpuDeviceNotFound` |
| GPU device | `src/domain/compute/gpu_device.rs` | `GpuDevice::auto_detect()` |
| BLAS trait | `src/domain/compute/gpu_ops.rs` | `trait GpuMatrixOps` |
| WGPU impl | `src/domain/compute/wgpu_ops.rs` | `struct WgpuMatrixOps` |
| Shaders | `src/domain/compute/wgpu_shaders/` | `.wgsl` files |
| Tests | `tests/gpu_blas_*.rs` | Test functions |
| Benches | `benches/gpu_ops_benchmark.rs` | Benchmark harness |

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "GPU device not found" | GPU not available or feature not enabled. Use `cargo build --features wgpu` |
| Shader compilation error | Check syntax with `naga check shader.wgsl` |
| Numerical accuracy low | Check tolerance formula; GPU f32 ≈ CPU f32 ±1e-4 |
| Tests timeout | Reduce problem size or enable release mode benchmarks |
| Memory allocation fails | Reduce buffer sizes or check available GPU RAM |

---

## Success Criteria Summary

### Session 1 (BLAS Foundation)
- [x] Project structure created
- [x] GEMM working (50+ tests)
- [x] Element-wise ops (6/6)
- [x] Normalization (2/2)
- [x] 100+ GPU tests passing
- [x] Benchmarks recorded

### Session 2 (Component Integration)  
- [ ] Diffusion GPU path
- [ ] SSM GPU kernels
- [ ] Transformer GPU path
- [ ] PolyAttention implementation
- [ ] Shared component consolidation
- [ ] 50+ integration tests

### Session 3 (Validation & Benchmarking)
- [ ] All ops validated (1000+ cases)
- [ ] Performance benchmarks
- [ ] Integration tests
- [ ] Documentation complete
- [ ] 550+ tests total passing

---

## Next Actions

1. **Before Session 1 starts**:
   ```bash
   git pull origin main
   cargo test --lib  # Verify 539 tests pass
   ```

2. **Session 1, Hour 1**:
   - Read: Section 1-2 of `WGPU_BLAS_IMPLEMENTATION_GUIDE.md`
   - Create directory structure
   - Set up build system

3. **Session 1, Hour 2**:
   - Implement GEMM shader
   - Compile and validate with `naga`

4. **Session 1, Hour 3-4**:
   - Complete Rust wrapper
   - Create test harness
   - Run validation tests

---

**Prepared by**: Amp  
**Date**: February 14, 2026  
**Status**: Ready for Session 1 start  
**Estimated completion**: February 18, 2026
