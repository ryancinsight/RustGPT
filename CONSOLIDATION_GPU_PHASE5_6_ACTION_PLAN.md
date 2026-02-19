# GPU Consolidation & Cleanup Phase 5.6 - Action Plan
**Date**: Feb 16, 2026  
**Status**: Implementation Kickoff  
**Thread**: @T-019c675f-91bb-7058-b594-cbc0e38d5091

## Overview

Consolidate shared components (Diffusion, SSM, Transformer) with unified GPU backend variants and strict no-fallback GPU execution semantics. All GPU operations will **fail fast** if backend is unavailable, ensuring predictable performance.

---

## Phase Structure: 3 Concurrent Workstreams

### 🔴 **Workstream 1: GPU Backend Unification** (Priority 1)
Merge duplicated GPU initialization and kernel dispatch logic into single source of truth.

**Tasks**:
1. **Consolidate `GpuDevice` initialization paths**
   - Location: `src/domain/compute/gpu_device.rs`
   - Merge CUDA, Metal, WGPU backend detection
   - Implement strict error path (no CPU fallback)
   - Add auto-detection with feature flag priority: CUDA > Metal > Vulkan > WGPU

2. **Unify `GpuMatrixOps` interface**
   - Location: `src/domain/compute/gpu_ops.rs`
   - Ensure all backends implement identical GEMM, activation, softmax signatures
   - Add pool-based allocation for consistent memory management
   - Add test suite for cross-backend numerical consistency (< 1e-4 tolerance)

3. **Consolidate Memory Pool implementations**
   - Location: `src/domain/compute/unified_gpu_buffer_pool.rs`
   - Merge CUDA, Metal, WGPU pool implementations
   - Standardize power-of-2 sizing and capacity tracking
   - Add telemetry: allocations, reallocations, total bytes

**Success Criteria**:
- ✓ Single `GpuDevice::auto_detect()` entry point works on all platforms
- ✓ Zero code duplication between CUDA/Metal/WGPU backends
- ✓ Memory pool reuse rate > 99% (reallocations only when capacity exceeded)
- ✓ All tests pass with `--features gpu-all`

---

### 🟡 **Workstream 2: Shared Component GPU Variants** (Priority 2)
Implement GPU kernels for the 3 core shared components.

**2A: SharedAttentionContext → `attention_context_gpu.rs` (70% complete)**

Location: `src/domain/layers/components/attention_context_gpu.rs`

Remaining tasks:
- [ ] Implement `AttentionContextParams` struct (dims, strengths, update_rate)
- [ ] GPU forward: similarity matrix computation (QKV projection → softmax → output)
- [ ] GPU backward: gradient computation for all parameters
- [ ] Integration with `attention_context.rs` (route GPU calls)
- [ ] Test: numerical equivalence with CPU implementation

**Implementation Pattern**:
```rust
pub fn forward_gpu(
    &mut self,
    input: &Array2<f32>,
    context_matrix: &Array2<f32>,
    strength: f32,
) -> Result<Array2<f32>> {
    // 1. Upload to GPU
    // 2. Call device.{gemm_f32, softmax, etc.}
    // 3. Download result
    // 4. Update stats
}
```

---

**2B: SharedFeedforward → `feedforward_gpu.rs` (40% complete)**

Location: `src/domain/layers/components/feedforward_gpu.rs`

Remaining tasks:
- [ ] RichardsGLU fused kernel (2-pass: activation → projection)
  - Pass 1: `input @ W1` → Richards activation → output
  - Pass 2: `output @ W2` → final output
  - **Target**: 25x speedup (50ms → 2ms on 1K batch)
- [ ] MixtureOfExperts kernel (expert selection + aggregation)
- [ ] Bias addition and activation on GPU (not CPU post-download)
- [ ] Memory pool integration for workspace buffers

**Key Optimization**: RichardsGLU fused kernel
```
Input: [batch, input_dim]
├─ Projection: @ W1 → [batch, hidden_dim]
├─ Richards: tanh activation → [batch, hidden_dim]
├─ Projection: @ W2 → [batch, output_dim]
└─ Bias add + activation
Output: [batch, output_dim]
```

---

**2C: SharedTemporalProcessing → `temporal_processing_gpu.rs` (20% complete)**

Location: `src/domain/layers/components/temporal_processing_gpu.rs`

Remaining tasks:
- [ ] Attention GPU kernel (Q/K/V projections, scaled softmax, output)
  - **Target**: 30x speedup (30ms → 1ms on 512 batch)
  - Causal masking support
  - Sliding window attention support
- [ ] Mamba selective scan GPU kernel (state update, projection)
  - **Target**: 20x speedup (40ms → 2ms on 512 batch)
- [ ] RG-LRU recurrent kernel (state recurrence, projection)
  - **Target**: 15x speedup (30ms → 2ms on 512 batch)
- [ ] Workspace management for state tensors (persistent allocation)

---

### 🟢 **Workstream 3: Testing & Validation** (Priority 3)
Ensure numerical accuracy and performance targets across all consolidations.

**3A: Numerical Validation Tests**
- Location: `tests/gpu_shared_components_phase56.rs`
- For each component (Attention, Feedforward, Temporal):
  - Compare GPU output vs CPU reference output
  - Tolerance: < 1e-4 relative error
  - Test corner cases: batch=1, batch=512, seq_len=1, seq_len=2048

**3B: Performance Benchmarks**
- Location: `benches/gpu_kernels_bench.rs`
- Measure latency for each kernel at various batch sizes
- Verify targets met (25x-30x speedups)
- Profile memory usage vs CPU baseline

**3C: Integration Tests**
- Verify shared components work together (Attention → Feedforward → Temporal)
- Test GPU device attach/detach
- Test auto-detection on systems with/without GPU

---

## Implementation Patterns

### Pattern 1: GPU Component with Workspace
```rust
pub struct MyComponent {
    // ... existing fields ...
    
    #[serde(skip)]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
    
    #[serde(skip)]
    workspace: Option<ComponentWorkspace>,
}

impl GpuComponent for MyComponent {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }
    
    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        Ok(())
    }
    
    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }
}

pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    if self.gpu_device.is_some() {
        return self.forward_gpu(input).expect("GPU forward failed - no fallback");
    }
    self.forward_cpu(input)
}
```

### Pattern 2: Fused Kernel Implementation
```rust
fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let mut device = self.gpu_device.lock().map_err(|_| ModelError::Backend {
        message: "Failed to acquire GPU device lock".to_string(),
    })?;
    
    let (batch_size, input_dim) = input.dim();
    let mut workspace = self.workspace.as_mut()
        .ok_or_else(|| ModelError::Backend {
            message: "Workspace not initialized".to_string(),
        })?;
    
    // Ensure workspace has capacity
    workspace.ensure_capacity(&mut device, batch_size, self.hidden_dim)?;
    
    // Pass 1: W1 projection
    device.upload(input.as_slice().unwrap(), &mut workspace.buf_input)?;
    device.gemm_f32(
        1.0, &workspace.buf_input, &workspace.w1,
        0.0, &mut workspace.buf_hidden,
        batch_size, self.hidden_dim, input_dim, false, false
    )?;
    
    // Pass 2: Richards activation + W2 projection (fused)
    device.richards_glu_fused(
        &workspace.buf_hidden,
        &workspace.w2,
        &mut workspace.buf_output,
        batch_size, self.hidden_dim, self.output_dim
    )?;
    
    // Download result
    let mut output = vec![0.0f32; batch_size * self.output_dim];
    device.download(&workspace.buf_output, &mut output)?;
    
    Ok(Array2::from_shape_vec((batch_size, self.output_dim), output)?)
}
```

### Pattern 3: Strict No-Fallback Initialization
```rust
pub fn auto_detect() -> Result<Self> {
    let device = GpuDevice::auto_detect()?;  // Errors if no GPU
    Ok(Self {
        device: Arc::new(Mutex::new(device)),
        // ...
    })
}

pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // No CPU fallback - errors propagate
    let mut device = self.device.lock()
        .map_err(|_| ModelError::Backend {
            message: "GPU device lock failed".to_string(),
        })?;
    
    // GPU operations required - return error if not available
    device.gemm_f32(...)?;  // .? propagates error
    
    Ok(result)
}
```

---

## File Organization

```
src/domain/
├── compute/
│   ├── gpu_device.rs                    [Unified auto-detect]
│   ├── gpu_ops.rs                       [GpuMatrixOps interface]
│   ├── unified_gpu_buffer_pool.rs       [Consolidated memory pool]
│   ├── cuda/
│   │   ├── ops.rs                       [CUDA-specific impl]
│   │   └── memory.rs
│   ├── metal/
│   │   ├── ops.rs                       [Metal-specific impl]
│   │   └── memory.rs
│   ├── wgpu/
│   │   ├── ops.rs                       [WGPU-specific impl]
│   │   └── memory.rs
│   └── gpu_component.rs                 [GpuComponent trait]
│
└── layers/components/
    ├── attention_context.rs              [CPU implementation]
    ├── attention_context_gpu.rs          [GPU variant + kernels] ← IMPLEMENT
    ├── feedforward.rs                    [CPU implementation]
    ├── feedforward_gpu.rs                [GPU variant + RichardsGLU] ← IMPLEMENT
    ├── temporal_processing.rs            [CPU implementation]
    ├── temporal_processing_gpu.rs        [GPU variant + Attention/Mamba/RG-LRU] ← IMPLEMENT
    ├── unified_gpu_kernels.rs            [Kernel dispatcher]
    └── unified_gpu_backend.rs            [Backend dispatcher]
```

---

## Testing Strategy

### Unit Tests (Per Component)
- `tests/gpu_shared_components_phase56.rs`
  - Test CPU vs GPU numerical equivalence
  - Test workspace capacity management
  - Test auto-detect error handling

### Integration Tests
- `tests/gpu_shared_component_integration.rs`
  - Test SharedAttentionContext → SharedFeedforward → SharedTemporalProcessing pipeline
  - Test GPU device attachment/detachment
  - Test workspace reset between operations

### Benchmarks
- `benches/gpu_kernels_bench.rs`
  - Measure latency for each component at various batch sizes
  - Verify 25x-30x speedup targets
  - Profile memory usage

---

## Success Metrics

### Code Quality
- ✓ Zero code duplication between CUDA/Metal/WGPU backends
- ✓ 100% of GPU operations use workspace pools (no ad-hoc allocations)
- ✓ All GPU operations follow strict no-fallback pattern

### Performance
- ✓ SharedAttentionContext GPU: 20x speedup vs CPU (batch=512)
- ✓ SharedFeedforward (RichardsGLU) GPU: 25x speedup vs CPU (batch=1K)
- ✓ SharedTemporalProcessing (Attention) GPU: 30x speedup vs CPU (batch=512)
- ✓ SharedTemporalProcessing (Mamba) GPU: 20x speedup vs CPU (batch=512)
- ✓ Memory pool reuse rate > 99%

### Numerical Accuracy
- ✓ GPU vs CPU output difference < 1e-4 (relative error)
- ✓ Backward pass gradients match within tolerance
- ✓ All tests pass with `cargo test --lib`

### Debugging & Observability
- ✓ GPU backend auto-detection works on all platforms
- ✓ Kernel launch counts tracked and reported
- ✓ Memory usage (upload/download bytes) tracked
- ✓ Error messages clearly indicate GPU backend failures

---

## Next Session Priorities

1. **Immediate** (Next 2 hours):
   - [ ] Implement `GpuDevice::auto_detect()` consolidation
   - [ ] Merge CUDA/Metal/WGPU memory pool implementations
   - [ ] Add unified error handling (no CPU fallback)

2. **Short-term** (Next 4 hours):
   - [ ] Implement SharedAttentionContext GPU kernels
   - [ ] Implement SharedFeedforward RichardsGLU fused kernel
   - [ ] Begin SharedTemporalProcessing attention kernel

3. **Medium-term** (Session completion):
   - [ ] Complete all GPU kernel implementations
   - [ ] Full test suite with numerical validation
   - [ ] Benchmark suite with performance targets
   - [ ] Documentation and integration guide

---

## Key References

- **Thread**: https://ampcode.com/threads/T-019c675f-91bb-7058-b594-cbc0e38d5091
- **AGENTS.md**: Build commands, architecture guidelines
- **Phase 5.6 Docs**: Kernel implementation patterns, memory management
- **GPU Kernel Consolidation**: Fused kernel strategies, workspace pooling

