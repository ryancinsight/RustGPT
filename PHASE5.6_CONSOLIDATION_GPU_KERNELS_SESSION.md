# Phase 5.6 Consolidation: GPU Kernel Implementation & Shared Components Optimization

**Thread**: @T-019c63ce-c226-712b-ae6e-582d945501e4  
**Status**: In Progress  
**Focus**: Implement fused kernels, eliminate duplicate code, optimize memory efficiency  
**Approach**: Strict GPU-first, automatic detection with no fallback  

---

## 1. Shared Components Architecture

### A. Components Identified for Consolidation
1. **RichardsGlu** - Feedforward with learnable Richards curves
2. **SharedAttentionContext** - Similarity-based context modulation  
3. **SharedTemporalProcessing** - Unified temporal mixing (Attention, SSM, RG-LRU)
4. **SharedFeedforward** - Unified interface for RichardsGlu and MoE

### B. Current State
- ✅ `GpuComponent` trait implemented (foundation)
- ✅ `UnifiedGpuBackend` provides entry point
- ✅ Basic GEMM operations in `wgpu_ops.rs`
- ✅ RichardsGlu has CPU forward/backward
- ✅ GPU device abstraction working (WGPU, CUDA, Metal)

### C. Critical Gaps
1. **No fused kernels** - Operations execute separately (high global memory traffic)
2. **CPU-side activation/normalization** - Data downloaded, processed on CPU, uploaded
3. **Duplicate shader definitions** - `wgpu_ops.rs` has structural issues
4. **No backward pass integration** - GPU gradients not flowing back to parameters

---

## 2. Fused Kernel Strategy

### A. RichardsGlu Fused Kernel (Target: 5 ops → 1 kernel)

**Current Flow** (5 GPU launches + CPU overhead):
```
GPU: W1 projection (GEMM)
GPU: W2 projection (GEMM)
CPU: Richards activation (swish + gate)
GPU: Gating multiply
GPU: W_out projection (GEMM)
CPU: Download + bias add
```

**Fused Flow** (1 GPU launch):
```
GPU: [W1 proj → Richards Act → W2 proj → Gate mul → W_out proj]
     - Load: input, W1, W2, W_out, gate params
     - Compute: x1 = input @ W1; x2 = input @ W2; act = richards(x1)
     - Compute: gated = act * gate(x2); output = gated @ W_out
     - Store: output
```

**Memory Efficiency**:
- Before: 5 × (upload + download) = 10 transfers
- After: 2 transfers (input upload, output download)
- **Reduction: 80% fewer global memory round-trips**

### B. Poly Attention Fused Kernel

**Current** (3 GEMM + separate softmax + normalization):
- Q @ K → Attention weights
- Softmax (separate)
- Weights @ V → Output
- Normalize (separate)

**Fused** (1 kernel):
- Q @ K → softmax (log-sum-exp) → weights @ V → normalize in one pass
- **Target: 30x speedup on 512 batch size**

### C. Mamba Selective Scan Fused Kernel

**Current** (Multiple passes for scan, projection, activation):

**Fused** (Single kernel):
- Load: input, A, B, C matrices
- Compute: selective scan with gates
- Output: fused scan + projection result

---

## 3. Implementation Roadmap

### Phase 5.6.1: Foundation & Validation (HIGH PRIORITY)
**Objective**: Implement and validate one fused kernel with automatic GPU detection

**Tasks**:
1. ✅ Verify auto_detect() works with strict no-fallback
2. ✅ Confirm RichardsGlu::forward_gpu() executes
3. ⏳ **[NEXT]** Create WGSL fused kernel for RichardsGlu
4. ⏳ **[NEXT]** Implement backward pass for RichardsGlu GPU
5. ⏳ **[NEXT]** Numerical validation suite (batch sizes 1-1024)

**Expected Outcome**:
- RichardsGlu forward/backward fully GPU-accelerated
- 25x speedup target achieved
- Zero memory allocation per forward pass

### Phase 5.6.2: Consolidation & Integration
**Objective**: Apply fused kernel pattern to remaining components

**Tasks**:
1. Implement Poly Attention fused kernel
2. Implement Mamba Selective Scan fused kernel
3. Unify attention context GPU operations
4. Create GPU backward pass for attention

**Expected Outcome**:
- All 4 shared components GPU-optimized
- 20-30x speedup across the board
- >92% memory efficiency

### Phase 5.6.3: Cleanup & Production
**Objective**: Finalize, remove deprecated code, full integration

**Tasks**:
1. Remove `shared_gpu_manager.rs` (deprecated)
2. Consolidate `gpu_shared_ops.rs` into unified backend
3. Full integration with diffusion, SSM, transformer blocks
4. Performance benchmarks (throughput, memory, latency)

---

## 4. Automatic GPU Detection (Strict No-Fallback)

### Implementation Pattern

```rust
// ✅ CORRECT: Strict GPU-first
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device_arc = self.gpu_device.as_ref()
        .ok_or(ModelError::Backend {
            message: "GPU device not initialized. Call enable_gpu_auto_detect() first.".to_string(),
        })?;
    
    // GPU computation or error - NO fallback to CPU
    device.gemm(...)?
}

// ✅ CORRECT: Auto-detection entry point
pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
    let device = GpuDevice::auto_detect()?;  // Errors if no GPU
    self.gpu_device = Some(Arc::new(Mutex::new(device)));
    Ok(())
}
```

### Detection Priority Order
1. **CUDA** (highest performance on NVIDIA)
2. **Metal** (native on macOS)
3. **Vulkan** (cross-platform, lowest overhead)
4. **WGPU** (web compatibility, universal fallback)
5. **None** → Error (strict policy)

---

## 5. Performance Targets

| Component | Operation | CPU Ref | GPU Target | Speedup |
|-----------|-----------|---------|------------|---------|
| **RichardsGLU** | 1K batch | 50ms | 2ms | **25x** |
| **PolyAttention** | 512 batch | 30ms | 1ms | **30x** |
| **Mamba Scan** | 512 batch | 40ms | 2ms | **20x** |
| **AttentionContext** | 1K batch | 15ms | 0.5ms | **30x** |

**Memory Targets**:
- >92% allocation efficiency (reduce padding waste)
- <1% fragmentation (power-of-2 sizing)
- Zero reallocation per forward pass

---

## 6. Testing Strategy

### Numerical Validation
```rust
#[test]
fn test_richards_glu_gpu_numerical_validation() {
    // CPU forward
    let output_cpu = layer.forward(&input);
    
    // GPU forward
    layer.enable_gpu_auto_detect()?;
    let output_gpu = layer.forward_gpu(&input)?;
    
    // Compare: L2 distance ≤ 1e-4
    assert_gpu_cpu_close(&output_gpu, &output_cpu, 1e-4);
}

#[test]
fn test_backward_gradient_flow() {
    // Ensure gradients flow through GPU operations
    let mut layer = RichardsGlu::new(768, 3072);
    layer.enable_gpu_auto_detect()?;
    
    let output = layer.forward_gpu(&input)?;
    let grads = compute_gradients(&output);
    
    // Verify parameter gradients updated
    assert!(layer.optimizer_w1.has_momentum());
}

#[test]
fn test_batch_size_robustness() {
    for batch_size in &[1, 8, 16, 32, 64, 128, 256, 512, 1024] {
        let input = random_array2((*batch_size, 768));
        let output = layer.forward_gpu(&input)?;
        assert_eq!(output.dim(), (*batch_size, 768));
    }
}
```

---

## 7. Integration Checklist

- [ ] RichardsGlu GPU forward pass finalized
- [ ] RichardsGlu GPU backward pass implemented
- [ ] Poly Attention GPU forward/backward
- [ ] Mamba Selective Scan GPU forward/backward
- [ ] Attention Context GPU operations
- [ ] All components in `impl_gpu_component!` macro
- [ ] Numerical validation <1e-4 error
- [ ] Batch robustness (1-1024)
- [ ] Memory efficiency >92%
- [ ] Performance targets achieved
- [ ] Zero-copy forward pipeline working
- [ ] Deprecated code removed
- [ ] Full integration tests passing
- [ ] Benchmarks documented

---

## 8. Next Immediate Actions

### Session Target (TODAY)
1. **Verify GPU detection works** → Test auto_detect() on system
2. **Implement fused WGSL kernel** → RichardsGlu [W1, Richards, W2, Gate, W_out]
3. **Backward pass skeleton** → Parameter gradients flow from GPU
4. **Initial numerical validation** → Compare CPU vs GPU outputs

### Success Criteria
- ✅ `cargo test --test gpu_shared_components_phase56` passes
- ✅ RichardsGlu forward_gpu() matches CPU output ±1e-4
- ✅ Backward pass integrates without panics
- ✅ No fallback to CPU (strict error handling)

---

## References
- **Fused kernels**: Zero-Copy Pipeline pattern (reduce global memory traffic)
- **Power-of-2 sizing**: Alignment for SIMD and cache efficiency
- **Log-sum-exp trick**: Numerical stability in softmax
- **Selective scan**: Mamba SSM core algorithm (parallelizable via scan primitive)
