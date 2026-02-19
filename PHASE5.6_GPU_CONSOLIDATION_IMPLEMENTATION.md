# Phase 5.6: GPU Consolidation Implementation Plan
## Automatic GPU Detection with Strict No-Fallback

**Date**: February 15, 2026  
**Status**: IN PROGRESS  
**Focus**: Consolidation + GPU implementation for shared components (Diffusion, SSM, Transformer)

---

## 1. Architecture Overview

### Unified GPU Management Stack
```
┌─────────────────────────────────────────────────────────────┐
│ Shared Components (Diffusion, SSM, Transformer)             │
│ - SharedFeedforward (RichardsGLU, MoE)                      │
│ - SharedTemporalProcessing (PolyAttention, Mamba, Attention)│
│ - SharedAttentionContext (context modulation)               │
└────────────────────┬────────────────────────────────────────┘
                     │ implements GpuComponent
┌────────────────────▼────────────────────────────────────────┐
│ UnifiedGpuBufferPool (Phase 5.3)                            │
│ - Power-of-2 buffer sizing                                  │
│ - AllocationStats tracking (reuse_count, resize_count)      │
│ - Zero-copy reuse across forward passes                     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ GpuDevice (Automatic Detection)                             │
│ - GpuDevice::auto_detect() - NO FALLBACK                    │
│ - Feature-gated: gpu-wgpu (focus), gpu-cuda, gpu-metal      │
│ - Strict error handling: returns Result, never silently CPU │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┼───────────┬────────────┐
         ▼           ▼           ▼            ▼
      WGPU        CUDA        Metal      (Error if none)
    (Vulkan)   (Phase 5.7)  (Phase 5.8)
```

---

## 2. Implementation Tasks

### Phase 5.6.1: SharedFeedforward GPU Implementation
**File**: `src/domain/layers/components/feedforward_gpu.rs`

#### RichardsGLU GPU Path
1. **Upload input** to GPU buffer
2. **Linear split** (2 GEMMs): `[W1 @ x || W2 @ x]`
3. **Richards activation** on x2 using fused kernel
4. **Element-wise multiply**: `x1 * activated(x2)`
5. **Download output**

**Key Components**:
- Use `UnifiedGpuBufferPool::allocate()` for pre-sized buffers
- Call `GpuDevice::gemm_f32()` for split computation
- Implement `richards_gpu_activate_fused()` kernel
- Kernel fusion pattern: GEMM + Richards + Multiply in single pass

#### MixtureOfExperts GPU Path
1. **Router GEMM**: `input @ W_router` → routing_logits
2. **Softmax**: Normalize routing logits
3. **Expert computation**: Parallel expert GEMMs
4. **Weighted sum**: Combine expert outputs using gates

**Target**: Zero CPU↔GPU transfers during forward pass

---

### Phase 5.6.2: SharedTemporalProcessing GPU Implementation
**File**: `src/domain/layers/components/temporal_processing_gpu.rs`

#### PolyAttention GPU Kernel
1. **Polynomial basis** computation (fused with input projection)
2. **Gating mechanism** (learnable gates per basis term)
3. **Weighted sum** of polynomial terms
4. **Output projection**

**Optimization**: Fuse polynomial basis + projection into single kernel to minimize memory roundtrips

#### Mamba/RG-LRU GPU Kernel
1. **Recurrent scan** (sequential, but SIMD-friendly on GPU)
2. **Fused multiplicative state update**
3. **Output projection**

**Target**: Maintain numerical stability (ε ≤ 1e-4 vs CPU)

#### TransformerAttention GPU Kernel
1. **QKV projection** (3 GEMMs fused or separate)
2. **Attention scores** (GEMM + softmax fusion)
3. **Context aggregation** (weighted sum of values)
4. **Output projection**

---

### Phase 5.6.3: SharedAttentionContext GPU Implementation
**File**: `src/domain/layers/components/attention_context_gpu.rs`

1. **Context similarity** computation: `activation @ activation^T`
2. **Softmax** normalization row-wise
3. **Context modulation**: `input + (strength / embed_dim) * (input @ context)`

**Fused Kernel Pattern**:
```rust
// Single kernel (not 3 separate ops)
fn fused_context_modulation(
    input: &GpuBuffer,      // (batch, embed_dim)
    context: &GpuBuffer,    // (embed_dim, embed_dim)
    output: &mut GpuBuffer, // (batch, embed_dim)
    strength: f32,
    // ... other params
)
```

---

## 3. Strict No-Fallback Strategy

### Auto-Detection Implementation
```rust
// In src/domain/compute/gpu_device.rs (already implemented)
pub fn auto_detect() -> Result<Self> {
    let detected = detect_available_gpu_backends();
    if let Some(&backend) = detected.first() {
        Self::new(backend)
    } else {
        Err(ModelError::Backend {
            message: "No supported GPU backend detected".to_string(),
        })
    }
}
```

### Component-Level Enforcement
- `SharedFeedforward::forward_into()`: 
  - If `compute_backend.is_gpu()` → GPU path ONLY
  - Else → CPU path ONLY (no automatic GPU selection)
  - Never silently fall back

- `SharedTemporalProcessing::forward_into()`:
  - Same pattern: explicit backend selection
  - Error if GPU was requested but unavailable

- Tests must verify: **No code path silently falls back to CPU**

---

## 4. Performance Targets

| Component | Operation | CPU Time | GPU Target | Efficiency Gain |
|-----------|-----------|----------|-----------|-----------------|
| SharedFeedforward | RichardsGLU 1K batch | ~50ms | ~2ms | 25× |
| SharedFeedforward | MoE (8 experts) | ~100ms | ~5ms | 20× |
| SharedTemporal | PolyAttention | ~30ms | ~1ms | 30× |
| SharedTemporal | Mamba scan | ~40ms | ~2ms | 20× |
| SharedTemporal | Transformer QKV | ~25ms | ~1ms | 25× |
| SharedAttention | Context modulation | ~15ms | ~0.5ms | 30× |

**Numerical Accuracy Target**: ε ≤ 1e-4 between GPU and CPU implementations

---

## 5. Memory Efficiency Improvements

### Power-of-2 Sizing (Already in UnifiedGpuBufferPool)
- Reduces fragmentation
- Enables efficient reuse
- Example: 1001 elements → allocate 1024, track padding in AllocationStats

### Zero-Allocation Forward Pass
- Pre-allocate buffers once (training init)
- Reuse across all iterations
- AllocationStats tracks:
  - `total_allocated`: Total bytes allocated
  - `total_wasted_padding`: Power-of-2 padding waste
  - `reuse_count`: Buffer reuses (target: high)
  - `resize_count`: Reallocations (target: 0 after init)

### Example Workspace Setup
```rust
// During LLMModel initialization (GPU mode)
let mut pool = UnifiedGpuBufferPool::new(device)?;

// Pre-allocate for batch_size=32, embed_dim=768
let feedforward_buf = pool.allocate_f32(32 * 768)?;  // 1024 sized
let temporal_buf = pool.allocate_f32(32 * 768)?;     // reuse same size
let attention_buf = pool.allocate_f32(32 * 768)?;    // reuse same size

// Forward pass: reuse_count += 3 per iteration
// resize_count stays at 0 (assuming batch size constant)
```

---

## 6. Testing Strategy

### Unit Tests (All Components)
1. **GPU availability check**: Skip gracefully if no GPU
2. **Numerical accuracy**: Compare GPU vs CPU outputs (ε ≤ 1e-4)
3. **Memory tracking**: Verify AllocationStats (reuse_count > 0, resize_count = 0)
4. **Strict no-fallback**: Verify error when GPU requested but unavailable

### Integration Tests
1. **Full forward pass** (Feedforward → Temporal → Attention)
2. **Training loop** (forward + gradient computation)
3. **Batch size variation** (16, 32, 64, 128) to test workspace resizing
4. **Multi-GPU scenarios** (future)

### Benchmark Suite
- Compare RichardsGLU: CPU vs WGPU vs CUDA
- Compare Mamba scan: CPU vs WGPU vs CUDA
- Compare Transformer attention: CPU vs WGPU vs CUDA
- Track memory usage and AllocationStats

---

## 7. Fallback Plan (If GPU Unavailable)

**DO NOT IMPLEMENT FALLBACK**.

Instead:
- Provide clear error messages
- Document GPU requirements in README
- Suggest: `cargo build --features gpu-wgpu` (for WGPU)
- For development: CPU mode available via explicit feature flag (future)

---

## 8. Feature Flags & Compilation

### Current Structure
```toml
[features]
wgpu = ["dep:wgpu", "dep:bytemuck"]  # Vulkan/WebGPU
gpu-cuda = ["dep:cudarc"]              # CUDA
gpu-metal = ["dep:objc-sys"]           # Metal (macOS)
```

### Enforcement
- `auto_detect()` checks compile-time feature flags
- Returns error if runtime backend doesn't match compile-time flags
- Example: System has CUDA, but compiled with `--features gpu-wgpu` → error (not silent fallback)

---

## 9. Implementation Checklist

### Week 1 (Feb 15-21)
- [ ] **5.6.1a**: Implement `forward_gpu_richards()` with GEMM + activation fusion
- [ ] **5.6.1b**: Implement `forward_gpu_moe()` with router + expert parallelization
- [ ] **5.6.1c**: Unit tests + numerical verification for SharedFeedforward GPU
- [ ] **5.6.2a**: Implement `forward_gpu_poly_attention()`
- [ ] **5.6.2b**: Implement `forward_gpu_mamba()`

### Week 2 (Feb 22-28)
- [ ] **5.6.2c**: Implement `forward_gpu_transformer_attention()`
- [ ] **5.6.3a**: Implement fused context modulation kernel
- [ ] **5.6.3b**: Unit tests + numerical verification for SharedAttentionContext GPU
- [ ] **5.6.4**: Integration tests (full forward pass)
- [ ] **5.6.5**: Benchmark suite + performance regression detection

### Week 3 (Mar 1-7)
- [ ] **5.6.6**: Memory efficiency verification (AllocationStats)
- [ ] **5.6.7**: Documentation + usage guide
- [ ] **5.6.8**: CI/CD setup (GPU tests on CUDA + WGPU runners)

---

## 10. Code Patterns Reference

### GpuComponent Trait Pattern
```rust
impl GpuComponent for SharedFeedforward {
    fn attach_gpu(&mut self, device: Arc<Mutex<GpuDevice>>) -> Result<()> {
        self.gpu_device = Some(device);
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }
    
    fn clear_gpu(&mut self) {
        self.gpu_device = None;
    }
}
```

### Zero-Copy Forward Pattern
```rust
pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
    if !self.is_gpu_ready() {
        return Err(ModelError::Backend {
            message: "GPU not attached for SharedFeedforward".to_string(),
        });
    }
    
    let mut device = self.gpu_device.lock()?;
    let pool = device.buffer_pool();
    
    // 1. Get buffers from pool (pre-sized)
    let gpu_input = pool.allocate_f32(input.len())?;
    let gpu_output = pool.allocate_f32(output.len())?;
    
    // 2. Upload
    device.upload(&input.iter().copied().collect::<Vec<_>>()[..], &gpu_input)?;
    
    // 3. Compute
    match &self.feedforward {
        FeedForwardVariant::RichardsGlu { .. } => {
            device.forward_gpu_richards(&gpu_input, &gpu_output)?;
        }
        _ => { /* ... */ }
    }
    
    // 4. Download
    let mut cpu_output = vec![0.0; output.len()];
    device.download(&gpu_output, &mut cpu_output)?;
    
    // 5. Copy to output buffer
    output.assign(&Array2::from_shape_vec((out_shape.0, out_shape.1), cpu_output)?);
    
    // 6. Deallocate (optional; pool may reuse)
    pool.deallocate(gpu_input);
    pool.deallocate(gpu_output);
    
    Ok(())
}
```

---

## Current Status

✅ **Completed (Phase 5.3-5.5)**:
- GpuDevice with auto_detect()
- UnifiedGpuBufferPool with AllocationStats
- GpuComponent trait
- WGPU backend integration
- CPU→GPU data transfer operations

🔄 **In Progress (Phase 5.6)**:
- SharedFeedforward GPU kernels (RichardsGLU, MoE)
- SharedTemporalProcessing GPU kernels (PolyAttention, Mamba, Attention)
- SharedAttentionContext GPU kernels
- Comprehensive testing & benchmarking

⏳ **Future (Phase 5.7-5.8)**:
- CUDA kernel implementations
- Metal kernel implementations (macOS)
- Multi-GPU coordination

---

## Notes

- **No CPU fallback**: This is intentional. GPU failures surface immediately, forcing explicit handling.
- **Numerical accuracy**: Maintain ε ≤ 1e-4 vs CPU reference to catch precision issues early.
- **Memory tracking**: AllocationStats enable proactive detection of memory leaks and inefficiencies.
- **Feature flags**: Ensure feature-gated code compiles correctly in all feature combinations.
