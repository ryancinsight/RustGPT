# Phase 5.6.3: GPU Kernel Implementation Status

**Date**: Feb 16, 2026  
**Objective**: Track which GPU kernels are implemented vs. stubbed across WGPU, CUDA, Metal

## BLAS Level 3: Matrix Operations

| Kernel | WGPU | CUDA | Metal | Notes |
|--------|------|------|-------|-------|
| `gemm_f32` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Tiled matrix mult (16x16 tiles) |
| `gemm_batched_f32` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Multiple GEMM ops |
| `gemv_f32` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Matrix-vector product |

## Element-Wise Operations

| Kernel | WGPU | CUDA | Metal | Notes |
|--------|------|------|-------|-------|
| `relu` | ✅ WGSL impl | ❌ Stub | ❌ Stub | max(0, x) |
| `gelu` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Approximate: 0.5 * x * (1 + tanh(...)) |
| `silu` | ✅ WGSL impl | ❌ Stub | ❌ Stub | x * sigmoid(x) |
| `sigmoid` | ✅ WGSL impl | ❌ Stub | ❌ Stub | 1 / (1 + exp(-x)) |
| `mul` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Element-wise multiply |
| `add_scaled` | ✅ WGSL impl | ❌ Stub | ❌ Stub | y += scale * x |
| `scale` | ✅ WGSL impl | ❌ Stub | ❌ Stub | y *= scale |
| `axpy` | ✅ WGSL impl | ❌ Stub | ❌ Stub | y = a*x + b*z |

## Specialized Element-Wise Operations

| Kernel | WGPU | CUDA | Metal | Status |
|--------|------|------|-------|--------|
| `richards_curve` | ✅ WGSL impl | ❌ Stub | ❌ Stub | **CRITICAL**: Stable log-space exponent |
| `moh_gate_activation` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Richards-gated per-head activation |

## Normalization Operations

| Kernel | WGPU | CUDA | Metal | Notes |
|--------|------|------|-------|-------|
| `layer_norm` | ✅ WGSL impl | ❌ Stub | ❌ Stub | (x - mean) / sqrt(var + eps) * gamma + beta |
| `softmax` | ✅ WGSL impl | ❌ Stub | ❌ Stub | Numerically stable (log-sum-exp) |

## PolyAttention-Specific Kernels

| Kernel | WGPU | CUDA | Metal | Priority | Notes |
|--------|------|------|-------|----------|-------|
| `poly_attention_fused` | ❌ **MISSING** | ❌ Stub | ❌ Stub | **HIGH** | Content + positional scoring fused |
| `blr_projection` | ❌ **MISSING** | ❌ Stub | ❌ Stub | **HIGH** | BLR low-rank projection |
| `compute_cope_scores` | ❌ **MISSING** | ❌ Stub | ❌ Stub | **MEDIUM** | COPE positional attention |

## Fused Kernels (Performance Critical)

| Kernel | WGPU | CUDA | Metal | Status | Impact |
|--------|------|------|-------|--------|--------|
| `richards_glu_fused` | ❌ **MISSING** | ❌ Stub | ❌ Stub | **CRITICAL** | 2-pass GLU (5+ → 2 launches) |

### RichardsGLU Two-Pass Kernel Breakdown

```
Pass 1: Activation + Gating
  - Input projection: [batch, input_dim] @ W_g1 → [batch, hidden_dim]
  - Gate logits: [batch, input_dim] @ W_g2 → [batch, hidden_dim]
  - Gated output: logits * value → [batch, hidden_dim]
  - Status: Element-wise ops exist, need fusion kernel

Pass 2: Output Projection
  - Matrix multiply: [batch, hidden_dim] @ W_out → [batch, output_dim]
  - Status: GEMM kernel exists, can reuse
```

## Data Transfer Operations

| Kernel | WGPU | CUDA | Metal | Notes |
|--------|------|------|-------|-------|
| `upload` | ✅ Impl | ✅ Impl | ❌ Stub | CPU → GPU |
| `download` | ✅ Impl | ✅ Impl | ❌ Stub | GPU → CPU |
| `copy_within_device` | ✅ Impl | ✅ Impl | ❌ Stub | GPU → GPU |
| `permute_4d` | ✅ Impl | ❌ Stub | ❌ Stub | Tensor reshaping |

## Reduction Operations

| Kernel | WGPU | CUDA | Metal | Notes |
|--------|------|------|-------|-------|
| `sum` | ✅ Impl | ❌ Stub | ❌ Stub | Reduce all elements |
| `mean` | ✅ Impl | ❌ Stub | ❌ Stub | Average across buffer |

## Implementation Priority Matrix

### Phase 5.6.3 Timeline

**Immediate (Feb 17)** - WGPU Completion
1. Verify existing WGSL kernels compile and run correctly
2. Add RichardsGLU two-pass kernel (CRITICAL for performance)
3. Add PolyAttention fused kernels (if using RustGPT with PolyAttention)
4. Test & benchmark all kernels

**Next (Feb 18)** - CUDA Stubs → Real Kernels
1. GEMM with cuBLAS (high priority)
2. Element-wise ops (parallel reduction)
3. Richards curve (stable computation)
4. RichardsGLU two-pass (custom CUDA kernel)

**Later (Feb 19+)** - Metal Implementation
1. Metal Performance Shaders (MPS) for BLAS
2. Metal compute shaders for specialized kernels

## Implementation Checklist

### WGPU Backend (Primary Target)

- [x] GEMM (tiled matrix multiplication)
- [x] GEMV (matrix-vector product)
- [x] Element-wise: ReLU, GELU, SiLU, Sigmoid
- [x] Element-wise: mul, add_scaled, scale, axpy
- [x] Richards curve (stable log-space)
- [x] MOH gate activation
- [x] Layer norm
- [x] Softmax (log-sum-exp)
- [x] Data transfer (upload/download)
- [x] Sum/Mean reductions
- [x] Permute 4D
- [ ] **RichardsGLU fused (TWO PASSES)** ← CRITICAL
- [ ] **PolyAttention fused** ← If needed
- [ ] **BLR projection** ← If using PolyAttention
- [ ] **COPE scores** ← If using PolyAttention

### CUDA Backend

- [ ] GEMM via cuBLAS
- [ ] GEMV via cuBLAS
- [ ] Element-wise kernel (single kernel, all ops)
- [ ] Richards curve (with FP32 stability checks)
- [ ] Data transfer integration
- [ ] Reduction operations (thrust library)
- [ ] RichardsGLU two-pass kernel

### Metal Backend

- [ ] BLAS operations via Metal Performance Shaders
- [ ] Element-wise kernels (`.metal` files)
- [ ] Specialized activation kernels
- [ ] Data transfer

## Testing Strategy

### Verification Tests (Priority Order)

```rust
#[test]
fn test_wgpu_gemm_correctness() { /* Compare vs CPU */ }

#[test]
fn test_wgpu_richards_curve_stability() { /* Large exponents */ }

#[test]
fn test_wgpu_moh_gate_activation() { /* Per-head gating */ }

#[test]
fn test_wgpu_richards_glu_fused() { /* Two-pass kernel */ }

#[test]
fn test_cuda_gemm_correctness() { /* cuBLAS wrapper */ }

#[test]
fn test_cuda_element_wise() { /* All ops in one kernel */ }
```

### Benchmark Targets

```
WGPU Benchmarks:
  - GEMM: [1024x1024] @ [1024x1024] → should see 10x+ vs CPU
  - Softmax: [4096 x 512] → should see 5x+ vs CPU
  - RichardsGLU: [batch=32, input=512, hidden=1024] → 2-pass dispatch

CUDA Benchmarks:
  - Compare cuBLAS GEMM vs WGPU tiled GEMM
  - Element-wise throughput comparison
```

## Shared Component Consolidation Targets

### Diffusion Models
- Attention in diffusion blocks (can reuse transformer kernels)
- Activation functions (unified element-wise ops)
- Normalization (shared layer norm)

### SSM (Mamba, RG-LRU)
- Selective scan (specialized SSM kernel - may need custom)
- Gating via Richards curve (unified richards_curve kernel)
- State updates (can build from element-wise ops)

### Transformer
- Multi-head attention (QKV projection + softmax)
- Feedforward (GEMM + activation)
- Layer norm + residual

## Known Issues & Workarounds

1. **CUDA cuBLAS linking**: Ensure CUDA toolkit is installed and paths configured
2. **Metal on non-macOS**: Will fail detection gracefully, fall back to WGPU/CUDA
3. **WGPU shader compilation**: Errors will appear at runtime, check shader_source strings
4. **Richards curve precision**: Use log-space for large exponents (avoid NaN/Inf)

## References

- **GPU Device Auto-detection**: `src/domain/compute/gpu_device.rs`
- **WGSL Shader Examples**: `src/domain/compute/wgpu_ops.rs#L30-L500`
- **CUDA Stubs**: `src/domain/compute/cuda/ops.rs`
- **Metal Stubs**: `src/domain/compute/metal/ops.rs`
- **Memory Pool**: `src/domain/compute/gpu_memory.rs`

## Next Steps

1. **Validate WGPU kernels** by running integration tests
2. **Profile kernel dispatch** to measure reduction from 5+ to 2 launches
3. **Create CUDA `.cu` kernel templates** for submission
4. **Plan Metal implementation** based on CUDA structure
