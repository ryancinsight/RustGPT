# GPU Consolidation Phase 5.5 - Action Plan
**Date**: February 14, 2026  
**Status**: Compilation Clean, Ready for GPU Backend Implementation  
**Target**: Unified GPU backend with WGPU BLAS operations and component integration

---

## 1. Current State ✓

### Completed
- [x] Error handling updated: GPU-specific error types added (`GpuDeviceNotFound`, `GpuInitializationError`, `GpuMemoryAllocation`, `GpuShaderCompilation`)
- [x] GPU device auto-detection implemented with strict no-fallback design
- [x] Test compilation warnings resolved (unused imports, unused variables)
- [x] Deprecated `CpuGpuMatrixOps` stub completed with all trait methods
- [x] All 539 unit tests passing (no failures or errors)
- [x] Unified buffer pool and executor frameworks in place

### GPU Trait Hierarchy
```
GpuMatrixOps (trait in gpu_ops.rs)
├── BLAS Level 3: gemm_f32, gemm_batched_f32, gemv_f32
├── Element-wise: relu, gelu, silu, sigmoid, mul, add_scaled, scale, axpy
├── Activation: richards_curve, richards_gate, fill_f32
├── Normalization: layer_norm, softmax
├── PolyAttention: poly_attention_fused, blr_projection, compute_cope_scores, moh_gate_activation
└── Data transfer: upload, download, copy_within_device, permute_4d

Implementations:
├── CpuMatrixOps (fallback for testing)
├── CudaMatrixOps (NVIDIA cuBLAS, feature: gpu-cuda)
├── MetalMatrixOps (Apple MPS, feature: gpu-metal)
└── WgpuMatrixOps (Cross-platform Vulkan/DX12/Metal, feature: wgpu)
```

---

## 2. Immediate Actions (Session 1)

### A. Implement WGPU BLAS Foundation
**Location**: `src/domain/compute/wgpu_ops.rs`

#### Task 2A.1: GEMM (General Matrix Multiply)
- [ ] Implement `gemm_f32` for single matrix multiply
- [ ] Target: Dense matrix multiplication (A @ B)
- [ ] Implement post-scaling with beta factor
- [ ] Support transpose flags for A and B
- [ ] Use WGSL workgroup optimization (e.g., 16x16 tiles)
- [ ] Test against CPU reference with tolerance ε ≤ 1e-4

**Pseudo-implementation**:
```wgsl
// Each workgroup computes 16x16 block of output
@compute @workgroup_size(16, 16)
fn gemm_f32(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Tile-based matrix multiply
    // Accumulate A[i,k] * B[k,j] for all k
    // Store to output[i,j] = alpha * sum + beta * output[i,j]
}
```

#### Task 2A.2: Batched GEMM
- [ ] Implement `gemm_batched_f32` using batched BLAS operations
- [ ] Handle contiguous and strided layouts
- [ ] Support 3 stride parameters: [stride_a, stride_b, stride_c]
- [ ] Optimize for batch_count >> 1

#### Task 2A.3: GEMV (Matrix-Vector Multiply)
- [ ] Implement `gemv_f32` for A @ x multiplication
- [ ] Optimize for single-column RHS (vectorized reduction)
- [ ] Use scan operations for efficient summation

---

### B. Element-Wise Operations in WGPU
**Location**: `src/domain/compute/wgpu_ops.rs`

#### Task 2B.1: Activation Functions
- [ ] `relu`: `output = max(0, input)`
- [ ] `gelu`: `output = input * Φ(input)` (approximation or lookup)
- [ ] `silu`: `output = input * sigmoid(input)`
- [ ] `sigmoid`: `output = 1 / (1 + exp(-input))` (use stable version)
- [ ] `richards_curve`: Custom parametric curve using provided params
- [ ] `richards_gate`: Gating function with per-head alpha/beta

**Implementation approach**:
- One shader per operation
- Thread per element (coalesced global memory access)
- Use uniform buffers for parameters

#### Task 2B.2: Element-Wise Arithmetic
- [ ] `mul`: `output = input1 * input2`
- [ ] `add_scaled`: `output += scale * input`
- [ ] `scale`: `output *= scale`
- [ ] `axpy`: `output = a * input1 + b * input2`
- [ ] `fill_f32`: Fill entire buffer with scalar value

**GPU optimization**: Coalesce all these into parameterized kernels to minimize shader code bloat.

---

### C. Normalization Operations
**Location**: `src/domain/compute/wgpu_ops.rs`

#### Task 2C.1: Layer Normalization
- [ ] Implement `layer_norm` for batch processing
- [ ] Two-pass algorithm:
  1. Compute per-feature mean and variance (parallel reduction)
  2. Normalize and apply gamma/beta scaling
- [ ] Use shared memory for reduction (avoid global memory bottleneck)

#### Task 2C.2: Softmax
- [ ] Implement `softmax` with numerically stable log-sum-exp trick
- [ ] Row-wise softmax for attention matrices
- [ ] Use shared memory for reduction of max and sum

---

### D. PolyAttention GPU Kernels (Task 2D)
**Location**: `src/domain/compute/wgpu_ops.rs` + shader compilation

#### Task 2D.1: MoH Gate Activation
- [ ] Implement `moh_gate_activation` GPU path
- [ ] Fused operation: Richards(alpha * logits + beta)
- [ ] Input: logits (batch, heads)
- [ ] Params: per-head alpha, beta vectors

#### Task 2D.2: BLR Projection
- [ ] Implement `blr_projection` for content/position projections
- [ ] Use GEMM internally for matrix products
- [ ] Apply Richards curve transformation to outputs

#### Task 2D.3: Content + Position Score Fusion
- [ ] Implement `compute_cope_scores` (content + position embedding scores)
- [ ] Efficient elementwise addition with broadcasting
- [ ] `poly_attention_fused`: Complete attention pipeline

---

## 3. Integration Phase (Session 2)

### A. Component-Level GPU Forward Paths
**Scope**: Diffusion, SSM, Transformer, PolyAttention blocks

#### Task 3A.1: DiffusionBlock GPU Path
- [ ] Location: `src/domain/diffusion/diffusion_block.rs`
- [ ] Implement `forward_gpu()` entry point
- [ ] Pipeline: Embedding → Denoising → Output
- [ ] Use `GpuDevice` for all matrix operations
- [ ] **Zero CPU↔GPU transfers during forward pass**
- [ ] Test against CPU reference implementation

#### Task 3A.2: SSM (Mamba/RG-LRU) GPU Path
- [ ] Location: `src/domain/ssm/temporal_mixing_layer.rs`
- [ ] Replace placeholder kernels in `temporal_processing_gpu.rs`
- [ ] Implement WGSL recurrent scan kernels for SSM state updates
- [ ] State initialization, A-matrix diagonalization
- [ ] Numerically stable log-domain operations
- [ ] Benchmark: Expected speedup 10-50x over CPU (sequence-dependent)

#### Task 3A.3: TransformerBlock GPU Path
- [ ] Location: `src/domain/transformer/block.rs`
- [ ] Unified forward pass: Attention → Temporal → Feedforward
- [ ] Verify zero CPU↔GPU transfers
- [ ] Profile memory allocation (target: <90% peak GPU memory)

#### Task 3A.4: PolyAttention GPU Integration
- [ ] Location: `src/domain/attention/poly_attention.rs`
- [ ] Use GPU kernels for fused operations
- [ ] Verify numerical accuracy vs CPU (ε ≤ 1e-4)
- [ ] Profile head-specific gate computation

---

### B. Shared Component GPU Unification
**Scope**: AttentionContext, SharedComponents, Temporal variants

#### Task 3B.1: AttentionContext GPU Consolidation
- [ ] Merge `SharedAttentionContext` and `AttentionContextGpu` implementations
- [ ] Single unified interface with GPU/CPU path selection
- [ ] GPU path uses `GpuDevice::apply_attention_context()`
- [ ] Test: Numerical equivalence across paths

#### Task 3B.2: Temporal Mixing Variants
- [ ] Consolidate Mamba, RG-LRU, and vanilla RNN variants
- [ ] Single `TemporalMixingLayer` with variant selection
- [ ] GPU kernels for all three variants
- [ ] Benchmark suite for each variant

#### Task 3B.3: Feedforward Consolidation
- [ ] Merge dual-dense, swiglu, and gated variants
- [ ] Single GPU kernel for all with parameter selection
- [ ] FusedFeedforward trait implementation
- [ ] Optimize MLP pipeline (dense → activation → dense)

---

## 4. Validation & Benchmarking (Session 3)

### A. Numerical Validation
- [ ] Test each GPU kernel against CPU reference (1000+ random inputs)
- [ ] Verify tolerance: max error ≤ 1e-4 (f32 precision limit ≈ 1e-7)
- [ ] Check edge cases: zeros, large values, denormalized numbers
- [ ] NaN/Inf handling consistency

### B. Performance Benchmarks
**Target baseline**: M1 Mac, RTX 3090, AMD RX 6900 XT

```bash
cargo bench --bench gpu_ops_benchmark
```

Expected performance:
- **GEMM** (f32): 50-100 TFLOPS (modern GPUs)
- **GEMV**: 10-20 TFLOPS (memory-limited)
- **Element-wise**: 100-500 GB/s (bandwidth-limited)
- **Attention forward**: 5-15x speedup vs CPU
- **Memory overhead**: <5% vs CPU equivalent

### C. Component Benchmarks
- [ ] Full diffusion forward: 10-50x speedup
- [ ] SSM forward: 20-100x speedup (sequence-dependent)
- [ ] Transformer block: 5-10x speedup
- [ ] End-to-end training step: 8-15x speedup

---

## 5. Feature Flags & Build Strategy

### Current Feature Matrix
```toml
[features]
default = ["cpu-only"]
cpu-only = []
gpu-cuda = ["cudarc"]
gpu-metal = ["metal", "objc"]
wgpu = ["wgpu"]
all-gpu = ["gpu-cuda", "gpu-metal", "wgpu"]
```

### Build Configurations

| Config | Features | GPU Backend | Use Case |
|--------|----------|-------------|----------|
| `cargo build` | `cpu-only` | CPU only | Development, testing |
| `cargo build --features wgpu` | `wgpu` | WGPU (Vulkan/DX12/Metal) | Cross-platform deployment |
| `cargo build --features gpu-cuda` | `gpu-cuda` | CUDA cuBLAS | NVIDIA GPUs |
| `cargo build --features gpu-metal` | `gpu-metal` | Metal MPS | Apple Silicon |
| `cargo build --features all-gpu` | All GPU features | Auto-detect (priority: CUDA > Metal > WGPU) | Maximum compatibility |

### Compilation Rules
- [ ] CPU-only builds: CpuMatrixOps stub (no GPU at all)
- [ ] Feature-gated backends: Only enabled backends compiled
- [ ] Auto-detection: `GpuDevice::auto_detect()` searches enabled backends in priority order
- [ ] Strict no-fallback: If no GPU enabled/available, error immediately

---

## 6. Code Organization

### Files to Modify/Create

#### New Files
- [ ] `src/domain/compute/wgpu_ops_blas.rs` - BLAS Level 3 kernels
- [ ] `src/domain/compute/wgpu_ops_activation.rs` - Element-wise operations
- [ ] `src/domain/compute/wgpu_ops_attention.rs` - PolyAttention kernels
- [ ] `src/domain/compute/shaders/gemm.wgsl` - Matrix multiply shader
- [ ] `src/domain/compute/shaders/activation.wgsl` - Activation functions
- [ ] `src/domain/compute/shaders/softmax.wgsl` - Normalization
- [ ] `tests/gpu_integration_diffusion.rs` - Diffusion GPU tests
- [ ] `tests/gpu_integration_ssm.rs` - SSM GPU tests
- [ ] `tests/gpu_integration_transformer.rs` - Transformer GPU tests

#### Modified Files
- [ ] `src/domain/compute/wgpu_ops.rs` - Add BLAS/activation/attention implementations
- [ ] `src/domain/diffusion/diffusion_block.rs` - Add `forward_gpu()` method
- [ ] `src/domain/ssm/temporal_mixing_layer.rs` - Add GPU path with kernels
- [ ] `src/domain/transformer/block.rs` - Add GPU path
- [ ] `src/domain/attention/poly_attention.rs` - Use GPU attention kernels
- [ ] `src/domain/layers/components/attention_context_gpu.rs` - GPU consolidation
- [ ] `Cargo.toml` - Add wgpu + shader compilation dependencies

---

## 7. Success Criteria

### Phase 5.5 Completion
- [x] Error handling and auto-detection (COMPLETED)
- [ ] WGPU BLAS operations fully implemented and tested
- [ ] All 3 component types (Diffusion, SSM, Transformer) with GPU paths
- [ ] PolyAttention GPU integration complete
- [ ] Numerical validation: All operations ε ≤ 1e-4 vs CPU
- [ ] Performance benchmarks: 5-50x speedup depending on operation
- [ ] Compilation clean with all warnings addressed
- [ ] All tests passing (target: >550 tests including new GPU tests)
- [ ] Documentation: GPU architecture guide + kernel descriptions

### Metrics
- **Code quality**: 0 clippy warnings in GPU modules
- **Test coverage**: GPU paths covered for all major components
- **Documentation**: SHADER.md describing kernel algorithms
- **Performance**: Benchmarks in `benches/gpu_*.rs`

---

## 8. Dependencies & Resources

### External Dependencies
```toml
wgpu = "0.19"         # Cross-platform GPU API
wgsl-inline = "0.3"   # Shader compilation macros
bytemuck = "1.14"     # GPU data layout
ndarray = "0.15"      # Reference CPU operations
```

### Reference Materials
- **WGPU Compute Shaders**: https://wgpu.rs/
- **GPU BLAS Algorithms**: https://github.com/openai/triton (concepts)
- **Richards Curve Implementation**: Project's existing CPU implementation
- **Attention Mechanics**: Project's CPU poly_attention.rs

---

## 9. Session Checklist

### Before Starting
- [ ] Pull latest changes
- [ ] Verify all tests pass (`cargo test --lib`)
- [ ] Check GPU availability on development machine

### During Session
- [ ] Commit working checkpoints every 1-2 hours
- [ ] Run benchmarks after each major feature
- [ ] Document shader algorithms as code comments
- [ ] Test edge cases (empty tensors, single-element, large sizes)

### End of Session
- [ ] All new tests passing
- [ ] Performance benchmarks recorded
- [ ] Documentation updated
- [ ] Create continuation notes for next session

---

## 10. Next Steps

**Immediate (Next Session)**:
1. Implement WGPU GEMM shader (Task 2A.1)
2. Create BLAS test harness with CPU reference
3. Validate 1000 random test cases
4. Benchmark GEMM vs CPU baseline

**Following Session**:
1. Complete element-wise operations (Task 2B)
2. Implement normalization (Task 2C)
3. Create comprehensive operation test suite

**Phase Target**: 
Complete all GPU backend implementations by end of Session 3 (Feb 18, 2026)
