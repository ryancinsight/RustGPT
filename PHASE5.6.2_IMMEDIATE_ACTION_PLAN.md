# Phase 5.6.2: Fused GPU Kernels - Immediate Action Plan

**Status**: Ready for Execution
**Date**: February 15, 2026
**Focus**: Implement fused GPU kernels for shared components with automatic GPU detection and strict no-fallback semantics.

## Executive Summary

Phase 5.6.1 established the GPU component infrastructure. Phase 5.6.2 focuses on implementing high-performance fused GPU kernels that combine multiple operations into single kernel launches, minimizing global memory roundtrips and maximizing throughput.

### Key Principles
1. **Automatic GPU Detection**: Use `enable_gpu_auto_detect()` without CPU fallback
2. **Fused Kernels**: Combine 2-5 operations into single kernel
3. **Zero-Copy Pipeline**: Keep data on GPU for entire forward pass
4. **Strict Error Handling**: Errors, not silent fallbacks
5. **Numerical Validation**: Tolerance ε ≤ 1e-4 vs CPU reference

---

## Workstream 1: RichardsGLU Fused Kernel (PRIORITY 1)

### Current State
- RichardsGLU has basic GPU component wrapper
- Operations are dispatched to individual GPU kernels
- No kernel fusion implemented yet

### Target: Single-Pass Kernel
Combine these operations into one GPU kernel:
```rust
// CPU version (5 separate operations):
1. x1 = input @ W1          // GEMM
2. x2 = input @ W2          // GEMM
3. x1 = richards_curve(x1)  // Activation
4. x1 = x1 * x2             // Element-wise multiply
5. output = x1 @ W_out      // GEMM

// GPU fused version (1 kernel):
output = richards_glu_fused(input, W1, W2, W_out)
```

### Implementation Steps

#### Step 1.1: WGPU Fused Kernel (src/domain/compute/wgpu_ops.rs)
- Add `richardson_glu_fused` method to `WgpuMatrixOps`
- Shader: Combine projection + Richards + multiply + output
- Minimize global memory writes (1 instead of 5)

#### Step 1.2: CUDA Fused Kernel (src/domain/compute/cuda/kernels/)
- Create `richards_glu_fused.cu` kernel
- Use shared memory for intermediate values
- Optimize for warp-level operations

#### Step 1.3: Backward Pass for RichardsGLU
- Implement gradient computation on GPU
- Compute: ∂L/∂input, ∂L/∂W1, ∂L/∂W2, ∂L/∂W_out
- Use automatic differentiation or manual gradient kernels

#### Step 1.4: Validation & Benchmarking
- Numerical accuracy test: |GPU_output - CPU_output| ≤ 1e-4
- Batch sizes: 1, 32, 128, 512
- Performance target: 25x speedup on 1K batch

### Files to Create/Modify
```
src/domain/compute/wgpu_ops.rs                    (add fused kernel method)
src/domain/compute/cuda/kernels/richards_glu.cu   (new CUDA kernel)
src/domain/richards/richards_glu.rs               (integrate GPU forward/backward)
tests/gpu_richards_glu_fused.rs                   (test validation)
```

---

## Workstream 2: PolyAttention GPU Kernels (PRIORITY 2)

### Current State
- Basic GPU component structure in place
- No actual kernel implementation

### Target: Polynomial Attention Pipeline
```
Q = input @ W_q              (GEMM)
K = input @ W_k              (GEMM)
poly_scores = poly_basis(Q @ K.T)  (Custom kernel)
gates = softmax(poly_scores)        (GPU softmax)
output = gates @ V                  (GEMM)
```

### Implementation Steps

#### Step 2.1: Polynomial Basis Computation Kernel
- Compute polynomial basis: b_i(x) = c_i * x^i + c_{i-1} * x^{i-1} + ... + c_0
- Support learnable degrees (3-7 polynomials typically)
- Input: raw attention scores (batch * seq_len * seq_len)
- Output: polynomial activation matrix

#### Step 2.2: Learnable Degree Adaptation
- Per-head polynomial degree selection
- Gradient computation for degree parameters

#### Step 2.3: Full Forward + Backward
- Forward pass: polynomial basis → softmax → weighted sum
- Backward pass: gradient through polynomial operations

#### Step 2.4: Validation
- Compare with CPU poly attention reference
- Test degree adaptation learning

### Files to Create/Modify
```
src/domain/compute/wgpu_ops.rs                          (poly basis kernel)
src/domain/compute/cuda/kernels/poly_attention.cu       (CUDA kernel)
src/domain/attention/poly_attention_gpu.rs              (GPU integration)
tests/gpu_poly_attention_kernels.rs                     (validation)
```

---

## Workstream 3: Memory Efficiency & Zero-Copy Pipeline (PRIORITY 3)

### Current State
- Individual component GPU support
- Buffer pool exists but not fully integrated

### Target: Full Forward Pass on GPU
```
Input
  ↓ (upload once)
GPU ← SharedTemporalProcessing (attention/SSM)
  ↓ (stays on GPU)
GPU ← SharedFeedforward (RichardsGLU)
  ↓ (stays on GPU)
GPU ← SharedAttentionContext
  ↓ (download once)
Output
```

### Implementation Steps

#### Step 3.1: Zero-Allocation Reuse Pattern
- Pre-allocate all buffers for batch_size
- Reuse same buffers across 1000+ forward passes
- Track allocation stats (reuse_count, resize_count)

#### Step 3.2: Power-of-2 Buffer Sizing
- Allocation sizes: 64, 128, 256, 512, 1024, 2048, ...
- Reduces reallocation when growing by small amounts
- Efficiency target: >95% buffer utilization

#### Step 3.3: Memory Profiling
- Track total_allocated, total_wasted_padding
- Efficiency metric: (total_allocated - wasted) / total_allocated
- Target: >92% efficiency

#### Step 3.4: Integration with SharedComponentWorkspace
- Coordinate CPU and GPU buffer management
- Unified capacity tracking

### Files to Create/Modify
```
src/domain/layers/components/shared_gpu_manager.rs      (enhanced)
src/domain/compute/unified_gpu_buffer_pool.rs           (capacity/stats)
tests/gpu_memory_efficiency.rs                          (profiling)
```

---

## Workstream 4: GPU Automatic Detection with No Fallback (PRIORITY 1)

### Current State
- `GpuComponent::enable_gpu_auto_detect()` defined
- Implementation varies by component

### Goal: Unified, Strict Behavior
- All components use same detection code
- Error clearly if GPU unavailable
- No silent fallback to CPU

### Implementation Steps

#### Step 4.1: Unify Detection Code
- Create `GpuDetectionStrategy` trait
- Implement: auto_detect(), validate_backend(), report_capabilities()
- Centralize in `gpu_device.rs`

#### Step 4.2: Component Integration
- All shared components call unified detection
- Each component reports status on startup
- Clear error messages with remediation steps

#### Step 4.3: Testing
- Test with GPU present (should work)
- Test with GPU absent (should error clearly)
- Test with GPU feature flag mismatch

### Files to Create/Modify
```
src/domain/compute/gpu_device.rs                        (detection strategy)
src/domain/layers/components/shared_gpu_manager.rs      (integration)
tests/gpu_detection.rs                                  (validation)
```

---

## Immediate Next Steps (Next 4 Hours)

### Priority Order
1. ✅ **Review current GPU infrastructure** (DONE)
2. 🔄 **Implement RichardsGLU fused kernel** (WGPU first, then CUDA)
3. 🔄 **Add GPU backward pass for RichardsGLU**
4. 🔄 **Create validation test suite**
5. 🔄 **Implement PolyAttention GPU kernels**
6. 🔄 **Integrate zero-copy pipeline**
7. 🔄 **Benchmark and optimize**

### Success Criteria
- RichardsGLU forward + backward GPU working
- 25x speedup on 1K batch
- Numerical accuracy ε ≤ 1e-4
- Zero allocation overhead after warmup
- No CPU fallback paths

---

## Technical Reference: GPU Kernel Structure

### WGPU Kernel Template
```rust
pub fn richards_glu_fused(
    &mut self,
    input: &GpuBuffer,
    w1: &GpuBuffer,
    w2: &GpuBuffer,
    w_out: &GpuBuffer,
    output: &mut GpuBuffer,
    batch_size: usize,
    input_dim: usize,
    hidden_dim: usize,
) -> Result<()> {
    // 1. Create compute shader
    // 2. Bind buffers
    // 3. Dispatch compute (workgroups)
    // 4. Synchronize
    Ok(())
}
```

### CUDA Kernel Template
```cuda
__global__ void richards_glu_fused_kernel(
    const float* input,
    const float* w1, const float* w2, const float* w_out,
    float* output,
    int batch_size, int input_dim, int hidden_dim
) {
    // 1. Load input tile into shared memory
    // 2. Compute x1 = input @ W1
    // 3. Compute x2 = input @ W2
    // 4. Apply Richards activation: x1 = richards(x1)
    // 5. Element-wise multiply: x1 *= x2
    // 6. Project to output: output = x1 @ W_out
    // 7. Store to global memory
}
```

---

## Dependencies & Integration Points

### GPU Features Required
```
[features]
gpu-wgpu = ["wgpu", "bytemuck"]
gpu-cuda = ["cudarc", "cuda-kernels"]
gpu-metal = ["metal", "metal-rs"]
gpu-all = ["gpu-wgpu", "gpu-cuda", "gpu-metal"]
```

### Compilation Commands
```bash
# Build with WGPU
cargo build --release --features gpu-wgpu

# Build with CUDA
cargo build --release --features gpu-cuda

# Build with all GPU backends
cargo build --release --features gpu-all
```

### Testing Commands
```bash
# Test GPU RichardsGLU kernels
cargo test --lib gpu_richards_glu_fused

# Test GPU PolyAttention kernels
cargo test --lib gpu_poly_attention_kernels

# Test GPU memory efficiency
cargo test --lib gpu_memory_efficiency

# All GPU tests
cargo test --lib --features gpu-all | grep gpu
```

---

## Performance Targets

| Component | Operation | Current | GPU Target | Speedup |
|-----------|-----------|---------|-----------|---------|
| RichardsGLU | 1K batch | 50ms | 2ms | 25x |
| PolyAttention | 512 batch | 30ms | 1ms | 30x |
| Mamba | 512 batch | 40ms | 2ms | 20x |
| AttentionContext | 1K batch | 15ms | 0.5ms | 30x |

---

## Risk Mitigation

### Risk: GPU Kernel Bugs
- **Mitigation**: Extensive numerical validation against CPU reference
- **Tolerance**: ε ≤ 1e-4 absolute difference
- **Testing**: Random seed reproducibility

### Risk: Memory Fragmentation
- **Mitigation**: Power-of-2 sizing + pre-allocation
- **Monitoring**: Track reuse_count, resize_count
- **Threshold**: Alert if resize_count > 10 per 1000 passes

### Risk: Feature Flag Mismatches
- **Mitigation**: Clear error messages during auto_detect()
- **Testing**: Explicit feature flag tests
- **Documentation**: Setup guides for each backend

---

## Session Handoff

Next session should:
1. Review this plan
2. Start with RichardsGLU fused kernel (WGPU)
3. Validate numerical accuracy
4. Profile memory usage
5. Move to PolyAttention kernels if time allows

**Estimated Completion**: 2-3 sessions (~12-15 hours)
