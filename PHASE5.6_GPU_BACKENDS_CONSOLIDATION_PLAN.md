# Phase 5.6: GPU Backend Consolidation & Optimization Plan

**Status**: Ready for Implementation  
**Duration**: Current Session  
**Priority**: Critical Path (Blocking Phase 5 Completion)

## Executive Summary

Consolidate and optimize GPU backend variants across Diffusion, SSM, and Transformer shared components with automatic GPU detection and strict no-fallback semantics for troubleshooting. This phase focuses on:

1. **GPU Backend Variants**: Implement GPU-specific code paths for all shared components
2. **Fused Kernels**: Reduce GPU launches by combining operations
3. **Memory Efficiency**: Achieve >92% memory efficiency through power-of-2 sizing and zero-copy pipelines
4. **Auto-Detection**: Automatic GPU detection with CUDA > Metal > Vulkan priority

---

## Part 1: GPU Backend Variant Implementation

### 1.1 Components Requiring GPU Variants

| Component | Location | Current Status | GPU Implementation |
|-----------|----------|-----------------|-------------------|
| **SharedFeedforward** | `layers/components/feedforward.rs` | ✓ GPU traits present | Richards GLU fused kernel |
| **SharedAttentionContext** | `layers/components/attention_context.rs` | ✓ GPU traits present | Similarity + GEMM ops |
| **SharedTemporalProcessing** | `layers/components/temporal_processing.rs` | ✓ GPU traits present | Attention/Mamba/RG-LRU dispatch |
| **UnifiedGpuBackend** | `layers/components/unified_gpu_backend.rs` | ✓ Trait framework | Forward pass implementations |
| **Fused Kernels** | `layers/components/fused_kernels_module.rs` | ✓ Skeleton present | Kernel implementations |

### 1.2 Implementation Pattern

Each GPU variant implementation follows this pattern:

```rust
// 1. Configuration struct with kernel parameters
pub struct FusedKernelParams {
    batch_size: u32,
    // ... kernel-specific params
}

// 2. Execute function with two-pass strategy
pub fn execute(
    device: &Arc<Mutex<GpuDevice>>,
    pool: &mut dyn GpuMemoryPool,
    ops: &mut dyn GpuMatrixOps,
    input: &Array2<f32>,
    weights: ...,
    params: &FusedKernelParams,
) -> Result<Array2<f32>>

// 3. Return type preserves ndarray compatibility
// for integration with existing code
```

### 1.3 GPU Component Trait Implementation

All shared components implement `GpuComponent` via macro:

```rust
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedFeedforward {
    pub feedforward: FeedForwardVariant,
    #[serde(skip)]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,  // Unified GPU device
}

// Macro: impl_gpu_component!(SharedFeedforward);
// Generates:
// - set_gpu_device()
// - enable_gpu_auto_detect() [strict: errors if no GPU]
// - is_gpu_ready()
// - gpu_backend_name()
// - gpu_device()
// - ensure_capacity()
```

---

## Part 2: Fused Kernel Implementation Details

### 2.1 RichardsGLU Fused Kernel (2-Pass Strategy)

**Target**: 25x speedup (50ms CPU → 2ms GPU for 1K batch)

**Pass 1** (Combined Operations):
```
1. Load x (batch_size, input_dim) from GPU memory
2. Compute x1 = x @ W1 (batch_size, hidden_dim)
3. Compute x2 = x @ W2 (batch_size, hidden_dim)
4. Apply Richards activation: value = x1 * richards(x1; nu, k, m, beta)
5. Compute gate = sigmoid(x2 * gate_temp_inv)
6. Combine: gated = value * gate (batch_size, hidden_dim)
→ Keep in GPU cache
```

**Pass 2** (Output Projection):
```
1. Compute output = gated @ W_out (batch_size, output_dim)
2. Add residual: output += input
→ Download to CPU (or reuse for next layer)
```

**Kernel Optimization Points**:
- Shared memory for block-level W1, W2 loading (reduce bandwidth)
- Register tiling for intermediate values
- Fused activation + gating (avoid round-trip to global memory)

### 2.2 PolyAttention Fused Kernel (1-Pass Strategy)

**Target**: 30x speedup (30ms CPU → 1ms GPU for 512 batch)

**Single Kernel**:
```
1. Parallel GEMMs for Q, K, V projections (shared weight loading)
2. Polynomial attention scoring: attn = poly_basis(Q @ K^T)
3. Softmax (row-wise, stable numerics)
4. Apply to values: output = attn @ V
5. Project output: output = output @ W_o
→ Single kernel with cooperative thread groups
```

### 2.3 Mamba Scan Fused Kernel (Selective Scan)

**Target**: 20x speedup (40ms CPU → 2ms GPU for 512 batch)

**Scan Strategy**:
```
For each position t in sequence:
  1. Load input[t] and state[t-1]
  2. Compute selection vector: delta = softplus(input[t] @ delta_w)
  3. Selective state update: state[t] = A * state[t-1] + B * delta * input[t]
  4. Compute output[t] = C @ state[t]
  5. Write state[t] back for next position
→ Warp-level scan for efficiency
```

---

## Part 3: GPU Detection & Component Integration

### 3.1 Automatic GPU Detection Flow

```
UnifiedGpuBackend::auto_detect()
  ↓
GpuDevice::auto_detect()
  ↓
detect_available_gpu_backends()  [compile-time detected backends]
  ↓
Backends: CUDA > Metal > Vulkan
  ↓
Return FIRST available backend OR error if NONE available
  ↓
[STRICT: No fallback to CPU]
```

### 3.2 Component Integration Pattern

```rust
// In component initialization
let mut component = SharedFeedforward::new(...);

// Attach GPU device (automatic detection)
component.enable_gpu_auto_detect()?;  // Errors if no GPU

// Pre-allocate GPU buffers for known dimensions
component.ensure_capacity(batch_size, embed_dim, seq_len)?;

// Forward pass uses GPU automatically
let output = component.forward(&input);  // Uses GPU if attached
```

### 3.3 Cross-Architecture Memory Sharing

```rust
// Unified buffer pool manages memory across architectures
let buffer_manager = SharedBufferManager::with_specialized_pools();

// Transformer-optimized pool for attention matrices
let attn_buffer = buffer_manager.transformer_pool().allocate(size)?;

// Diffusion-optimized pool for large feature maps
let feat_buffer = buffer_manager.diffusion_pool().allocate(size)?;

// SSM-optimized pool for state vectors
let state_buffer = buffer_manager.ssm_pool().allocate(size)?;

// All pools share same underlying memory via `unified` fallback
```

---

## Part 4: Performance Optimization Targets

### 4.1 Memory Efficiency Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| **Allocation Reuse** | >90% | Power-of-2 sizing, buffer pooling |
| **Zero-Copy** | >95% | Keep data on GPU between ops |
| **Memory Fragmentation** | <5% | Power-of-2 classes, 16 size classes |
| **Peak Memory** | <256MB per layer | Streaming weights, selective buffering |

### 4.2 Kernel Launch Reduction

| Component | Before | After | Method |
|-----------|--------|-------|--------|
| RichardsGLU | 5 launches | 2 passes | Fused projection + activation + gating |
| PolyAttention | 3+ launches | 1 launch | Cooperative thread groups |
| Mamba Scan | Multiple loops | 1 recurrent | Warp-level scan |
| AttentionContext | 3 launches | 1 launch | Single GEMM + scaling |

### 4.3 Expected Performance

| Component | CPU Time (1K batch) | GPU Time (Target) | Speedup |
|-----------|-------------------|------------------|---------|
| RichardsGLU | 50ms | 2ms | **25x** |
| PolyAttention | 30ms | 1ms | **30x** |
| Mamba Scan | 40ms | 2ms | **20x** |
| AttentionContext | 15ms | 0.5ms | **30x** |

---

## Part 5: Implementation Checklist

### Phase 5.6.2: Fused Kernel Implementation

- [ ] **RichardsGLU Fused**
  - [ ] Implement Pass 1 kernel (projection → activation → gating)
  - [ ] Implement Pass 2 kernel (output projection + residual)
  - [ ] Test correctness vs. CPU variant
  - [ ] Benchmark against target (2ms for 1K batch)

- [ ] **PolyAttention Fused**
  - [ ] Implement parallel Q/K/V projections
  - [ ] Implement polynomial scoring kernel
  - [ ] Add softmax normalization
  - [ ] Test correctness and benchmark (1ms for 512 batch)

- [ ] **Mamba Scan**
  - [ ] Implement selective scan kernel
  - [ ] Add state management
  - [ ] Test with SSM components
  - [ ] Benchmark (2ms for 512 batch)

- [ ] **AttentionContext GPU Ops**
  - [ ] Implement similarity matrix computation
  - [ ] Add softmax row-wise
  - [ ] Test with shared attention context
  - [ ] Benchmark (0.5ms for 1K batch)

### Phase 5.6.3: Buffer Pool Optimization

- [ ] Implement power-of-2 sizing for all GPU buffers
- [ ] Add buffer reuse tracking
- [ ] Verify zero-copy pipeline across all components
- [ ] Test memory efficiency metrics

### Phase 5.6.4: Component Integration

- [ ] Implement `forward_gpu()` in SharedFeedforward
- [ ] Implement `forward_gpu()` in SharedAttentionContext
- [ ] Implement `forward_gpu()` in SharedTemporalProcessing
- [ ] Add automatic GPU detection with strict no-fallback
- [ ] Create integration tests across all components

### Phase 5.6.5: Consolidation & Cleanup

- [ ] Remove duplicate GPU code (consolidate into shared_gpu_ops.rs)
- [ ] Remove deprecated shared_gpu_manager.rs
- [ ] Update all imports to use UnifiedGpuBackend
- [ ] Verify no CPU fallback paths for GPU operations
- [ ] Full test suite with GPU-only execution

---

## Part 6: Troubleshooting with Strict No-Fallback

### 6.1 GPU Detection Errors

**Scenario**: GPU hardware present, but feature flags not enabled

```
Error: Automatic GPU detection found runtime backend(s): CUDA
This binary was not built with matching GPU feature flags.
Enable one of: --features gpu-cuda, gpu-metal, gpu-wgpu.
```

**Resolution**:
```bash
# Recompile with matching feature flag
cargo build --release --features gpu-cuda
```

### 6.2 GPU Operation Failures

**Scenario**: GPU operation fails mid-forward pass

```
Error: GPU operation 'forward_attention_context' failed:
[backend-specific error from CUDA/Metal/Vulkan]
```

**Resolution**:
1. Check error details (CUDA error code, Metal NSError, Vulkan result)
2. Verify GPU memory availability: `nvidia-smi` or Metal debugger
3. Check buffer alignment (power-of-2 sizing should be verified)
4. Fallback debugging: Compare with CPU implementation output

### 6.3 Memory Efficiency Issues

**Monitor**:
```rust
let stats = backend.memory_stats()?;
println!("Allocated: {}, Available: {}, Utilization: {}%",
    stats.total_bytes_allocated,
    stats.total_bytes_available,
    (stats.total_bytes_allocated * 100) / stats.total_bytes_available);
```

**Target**: >92% utilization through power-of-2 sizing and buffer pooling

---

## Part 7: Integration with Shared Components

### 7.1 Transformer Integration

```rust
// In TransformerBlock::forward()
let mut attn = SharedAttentionContext::default();
attn.enable_gpu_auto_detect()?;  // CUDA > Metal > Vulkan

let mut ff = SharedFeedforward::new(FeedForwardVariant::RichardsGlu(...));
ff.enable_gpu_auto_detect()?;

// Forward uses GPU automatically
let attn_out = attn.forward(&input)?;
let ff_out = ff.forward(&attn_out)?;
```

### 7.2 Diffusion Integration

```rust
// In DiffusionBlock::forward()
let mut temporal = SharedTemporalProcessing::new(...);
temporal.enable_gpu_auto_detect()?;

let temporal_out = temporal.forward(&input)?;  // GPU forward
```

### 7.3 SSM Integration

```rust
// In MambaBlock::forward()
let mut temporal = SharedTemporalProcessing::new(
    TemporalMixingLayer::Mamba(...),
    ...
);
temporal.enable_gpu_auto_detect()?;

let output = temporal.forward(&input)?;  // GPU forward with Mamba kernel
```

---

## Part 8: Success Criteria

- [x] All shared components implement `GpuComponent` trait
- [ ] RichardsGLU achieves 25x speedup (50ms → 2ms on 1K batch)
- [ ] PolyAttention achieves 30x speedup (30ms → 1ms on 512 batch)
- [ ] Mamba Scan achieves 20x speedup (40ms → 2ms on 512 batch)
- [ ] Memory efficiency >92% through power-of-2 sizing
- [ ] Zero-copy GPU pipeline (data stays on GPU between ops)
- [ ] Strict no-fallback: Clear error messages if GPU unavailable
- [ ] Full integration with Transformer, Diffusion, and SSM architectures
- [ ] Comprehensive test suite with GPU-only execution
- [ ] All deprecated GPU code removed

---

## Next Steps

1. **Immediate**: Implement RichardsGLU fused kernel (Pass 1 & 2)
2. **Follow**: PolyAttention fused kernel
3. **Then**: Mamba selective scan kernel
4. **Finally**: Component integration and testing

**Estimated Time**: 4-6 hours with benchmarking

---

## References

- **Thread**: https://ampcode.com/threads/T-019c6409-39ff-73a9-a641-8b3d1bccb44b
- **GPU Device**: `src/domain/compute/gpu_device.rs` (auto-detection)
- **Component Trait**: `src/domain/compute/gpu_component.rs` (GpuComponent)
- **Buffer Pool**: `src/domain/layers/components/unified_buffer_pool.rs`
- **Kernel Stubs**: `src/domain/layers/components/fused_kernels_module.rs`
