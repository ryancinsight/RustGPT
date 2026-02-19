# Phase 5.7: Full GPU Kernel Implementation - Kickoff

**Status**: 🚀 **STARTING**  
**Date**: Feb 18, 2026  
**Previous Phase**: 5.6.4d (Complete - GPU Training Loops Validated)

---

## Phase 5.7 Mission

Complete GPU acceleration by implementing full-GPU kernels for:
1. **MoE Router** - Softmax, Richards, Reduction kernels
2. **RichardsGlu** - Activation derivative kernels (eliminate CPU fallback)
3. **Attention** - Q, K, V gradient kernels
4. **SSM** - State transition gradient kernels

**Expected Outcome**: 30-40% speedup in backward pass, full GPU pipeline without CPU fallback.

---

## Implementation Roadmap

### Priority 1: MoE Router GPU Kernels (Days 1-3)

```
Target: src/domain/mixtures/moe.rs → backward_gpu()

Current State (CPU):
  ExpertSelector::backward_gpu() {
      // Uses CPU for:
      // 1. Softmax gradient
      // 2. Richards derivative  
      // 3. Bias reduction
  }

Target State (GPU):
  ExpertSelector::backward_gpu() {
      // GPU Kernels:
      // 1. softmax_gradient_kernel()
      // 2. richards_derivative_kernel()
      // 3. reduction_kernel() for bias
  }
```

**Files to Create**:
1. `src/domain/compute/gpu_moe_kernels.rs` - MoE-specific GPU kernels
2. `src/domain/compute/gpu_softmax_kernel.rs` - Softmax gradient kernel
3. `src/domain/compute/gpu_reduction_kernels.rs` - Bias accumulation kernels

**Estimated Kernels**: 3-4 shaders, ~200 lines GPU code

---

### Priority 2: RichardsGlu Derivative Kernels (Days 4-5)

```
Target: src/domain/richards/richards_glu.rs → backward_gpu()

Current State (CPU):
  RichardsGlu::backward_gpu() {
      // GPU: GEMM operations (w1, w2, w_out)
      // CPU: Richards activation derivative (Rayon)
      // CPU: RichardsGate derivative (Rayon)
  }

Target State (Full GPU):
  RichardsGlu::backward_gpu() {
      // All GPU:
      // GPU GEMM: w1, w2, w_out gradients
      // GPU: Richards derivative kernel
      // GPU: Gate derivative kernel
      // GPU: Bias accumulation
  }
```

**Files to Create**:
1. `src/domain/compute/gpu_richards_derivative_kernel.rs`
2. `src/domain/compute/gpu_gate_derivative_kernel.rs`

**Estimated Kernels**: 2 shaders, ~150 lines GPU code

---

### Priority 3: Attention Backward Kernels (Days 6-8)

```
Target: src/domain/attention/poly_attention.rs → backward_gpu()

Current State (Partial GPU):
  PolyAttention::backward_gpu() {
      // Some GPU operations but not complete
  }

Target State (Full GPU):
  PolyAttention::backward_gpu() {
      // GPU Kernels:
      // 1. dQ gradient computation
      // 2. dK gradient computation
      // 3. dV gradient computation
      // 4. dScale gradient (poly scale param)
      // 5. Attention weight gradient
  }
```

**Files to Create**:
1. `src/domain/compute/gpu_attention_kernels.rs` - Q, K, V gradients
2. `src/domain/layers/components/attention_backward_gpu.rs` - Orchestration

**Estimated Kernels**: 5-6 shaders, ~300 lines GPU code

---

### Priority 4: SSM Backward Kernels (Days 9-10)

```
Target: src/domain/layers/ssm/mamba.rs → backward_gpu()

Current State (CPU only):
  Mamba::backward() uses CPU for all gradient computation

Target State (GPU):
  Mamba::backward_gpu() {
      // GPU Kernels:
      // 1. State derivative kernel
      // 2. Input gradient kernel
      // 3. Weight gradient accumulation
      // 4. Scan gradient (reverse pass)
  }
```

**Files to Create**:
1. `src/domain/compute/gpu_ssm_kernels.rs` - SSM-specific kernels

**Estimated Kernels**: 4 shaders, ~200 lines GPU code

---

### Priority 5: Kernel Fusion & Optimization (Days 11-15)

```
Fuse consecutive operations:
  - GEMM + Activation + Gating → Single fused kernel
  - Softmax + Reduction → Single kernel
  - Attention weight + scaling → Single kernel
```

**Target Performance Gains**:
- Memory bandwidth: 20-30% reduction
- Launch overhead: 50% reduction via fusion
- Overall: 30-40% speedup expected

---

## Detailed Task Breakdown

### Day 1: Softmax Gradient Kernel

**File**: `src/domain/compute/gpu_softmax_kernel.rs`

```rust
// Kernel: softmax_backward
// Input: softmax(logits), grad_output, sequence_length
// Output: grad_logits
// 
// Algorithm:
// grad_logits[i] = softmax[i] * (grad_output[i] - sum(softmax * grad_output))

#[compute]
fn softmax_backward(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // For each position i:
    // 1. Load softmax[i] and grad_output[i]
    // 2. Compute sum via shared memory reduction
    // 3. grad_logits[i] = softmax[i] * (grad[i] - sum)
}
```

**Tests**:
```rust
#[test]
fn test_softmax_gradient_kernel() {
    // Create softmax output and gradient
    // Compare GPU kernel vs CPU reference implementation
    // Assert max error < 1e-5
}
```

---

### Day 2: Richards Derivative Kernel

**File**: `src/domain/compute/gpu_richards_derivative_kernel.rs`

```rust
// Kernel: richards_derivative
// Input: x (input to Richards function)
// Output: grad (derivative of Richards(x))
//
// Algorithm:
// d/dx[x * Richards(x)] = Richards(x) + x * dRichards/dx(x)
// Where dRichards/dx ≈ d/dx[curve_point + alpha * (1 - curve_point/max_val)]

#[compute]
fn richards_derivative(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // For each element:
    // 1. Load x and Richards parameters (alpha, curve_point, max_val)
    // 2. Compute d/dx[Richards(x)]
    // 3. Store gradient for backward accumulation
}
```

**Tests**:
```rust
#[test]
fn test_richards_derivative_kernel_numerical_gradient() {
    // Compute analytical gradient via kernel
    // Compute numerical gradient via finite differences
    // Assert relative error < 1e-4
}
```

---

### Day 3: Reduction Kernel

**File**: `src/domain/compute/gpu_reduction_kernels.rs`

```rust
// Kernel: bias_reduction (sum-reduce for bias gradients)
// Input: grad_buffer (batch, features)
// Output: grad_bias (features,)
//
// Algorithm:
// grad_bias[j] = sum_over_batch(grad_buffer[:, j])

#[compute]
fn bias_reduction(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    // For each feature j:
    // 1. Load all grad_buffer[:, j] into shared memory
    // 2. Tree reduction (sum)
    // 3. Atomic add to grad_bias[j]
}
```

**Optimizations**:
- Use workgroup shared memory for fast reduction
- Atomic operations for bias accumulation
- Handle variable batch sizes

---

### Days 4-5: RichardsGlu Kernels

**File**: `src/domain/richards/richards_glu.rs`

Update `backward_gpu()` to use GPU kernels instead of Rayon:

```rust
impl RichardsGlu {
    pub async fn backward_gpu(
        &mut self,
        grad_output: &Array2<f32>,
        device: &Arc<Mutex<GpuDevice>>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        // 1. Upload grad_output to GPU
        // 2. GEMM: grad_w_out = gated.T @ grad_output
        // 3. GEMM: grad_gated = grad_output @ w_out.T
        // 4. GPU Kernel: Richards derivative
        // 5. GPU Kernel: Gate derivative
        // 6. GEMM: grad_w1 = input.T @ grad_x1
        // 7. GEMM: grad_w2 = input.T @ grad_x2
        // 8. GPU Kernel: Bias reduction
        // 9. Download gradients
    }
}
```

---

### Days 6-8: Attention Backward Kernels

**File**: `src/domain/compute/gpu_attention_kernels.rs`

```rust
// Kernel: attention_q_gradient
// Inputs: query, key, value, attention_weights, grad_output, scale
// Output: grad_query

// Kernel: attention_k_gradient
// Inputs: query, key, value, attention_weights, grad_output, scale
// Output: grad_key

// Kernel: attention_v_gradient
// Inputs: query, key, value, attention_weights, grad_output, scale
// Output: grad_value

// Kernel: attention_weight_gradient
// Inputs: attention_weights, grad_output, value
// Output: grad_attention_logits
```

**Mathematical Reference**:
```
Forward:
  scores = (Q @ K.T) / sqrt(d) + scale_param
  weights = softmax(scores)
  output = weights @ V

Backward:
  grad_weights = grad_output @ V.T
  grad_scores = weights * (grad_weights - sum(weights * grad_weights))
  grad_K = Q.T @ (grad_scores / sqrt(d))
  grad_Q = grad_scores @ K / sqrt(d)
  grad_V = weights.T @ grad_output
  grad_scale = sum(grad_scores) per batch
```

---

### Days 9-10: SSM Backward Kernels

**File**: `src/domain/compute/gpu_ssm_kernels.rs`

```rust
// Kernel: ssm_state_gradient
// Inputs: hidden_state, grad_hidden, A, B, C matrices
// Output: grad_input, grad_state

// Kernel: ssm_scan_gradient
// Implements reverse-mode automatic differentiation through
// the selective scan operation (complex!)
```

**Challenge**: SSM backward involves reverse-scan which is complex on GPU.

---

### Days 11-15: Kernel Fusion

**File**: `src/domain/compute/gpu_kernel_fusion.rs`

```rust
// Fused Kernels:

// 1. gemm_activate_gate_fusion
//    Input: A (m×k), B (k×n), x (m×n)
//    Output: (A @ B + activation(x) * gate(x))
//    Replaces: 3 kernel calls → 1

// 2. softmax_reduce_fusion
//    Input: logits, grad_output
//    Output: grad_input, grad_bias
//    Replaces: 2 kernel calls → 1

// 3. attention_fused_backward
//    Input: Q, K, V, weights, grad_out
//    Output: grad_Q, grad_K, grad_V
//    Replaces: 5 kernel calls → 1
```

---

## Testing Strategy

### Unit Tests (Each Kernel)
```rust
#[test]
fn test_kernel_name_correctness() {
    // Numerical validation against CPU reference
    // Max error < 1e-4
}

#[test]
fn test_kernel_name_numerics() {
    // Numerical gradient checking
    // Analytical vs finite-difference gradients
}
```

### Integration Tests
```rust
#[test]
fn test_full_backward_pass_moe() {
    // MoE backward with all GPU kernels
    // Validate gradients match CPU reference
}

#[test]
fn test_full_training_loop_gpu() {
    // Run 5 epochs of training
    // Validate convergence behavior
    // Check memory stability
}
```

### Performance Tests
```rust
#[bench]
fn bench_softmax_kernel_vs_cpu() {
    // Measure speedup over CPU implementation
    // Target: >2x speedup
}

#[bench]
fn bench_fused_kernels() {
    // Measure fusion benefit
    // Target: 20-30% reduction in total time
}
```

---

## Build Commands

```bash
# Build with GPU support
cargo build --release --features gpu-wgpu

# Run GPU kernel tests
cargo test --features gpu-wgpu gpu_softmax_kernel
cargo test --features gpu-wgpu gpu_richards_derivative_kernel
cargo test --features gpu-wgpu gpu_reduction_kernels

# Run integration tests
cargo test --features gpu-wgpu test_full_backward_pass

# Benchmark
cargo bench --bench gpu_kernel_fusion_benchmarks --features gpu-wgpu
```

---

## Success Criteria

✅ **Phase 5.7 Complete When**:
1. All MoE router kernels GPU-accelerated (3 new kernels)
2. All RichardsGlu derivatives GPU-accelerated (2 new kernels)
3. Attention backward GPU kernels (5-6 new kernels)
4. SSM backward GPU kernels (2-3 new kernels)
5. Kernel fusion implemented (3+ fusions)
6. All tests passing with <1% numerical error
7. 30-40% backward pass speedup validated
8. Zero CPU fallback in critical path
9. Memory pool stable over 100+ iterations

**Estimated Timeline**: 15 working days
**Target Completion**: ~1 week with parallel work

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Complex SSM backward | Start with simpler kernels (Softmax, Richards); SSM can be hybrid |
| Numerical stability | Numerical gradient checking for each kernel; validate against CPU |
| GPU memory pressure | Stream-based processing; gradient checkpointing fallback |
| Kernel debugging | Use wgpu debugging tools; add detailed tracing |

---

## Next Immediate Actions

1. **Read current GPU kernel structure**
   ```bash
   find src/domain/compute -name "*kernel*.rs" | head -10
   ```

2. **Examine existing shader patterns**
   ```bash
   grep -r "#\[compute\]" src/domain/compute/
   ```

3. **Start Day 1: Softmax Kernel**
   - Create `gpu_softmax_kernel.rs`
   - Implement softmax_backward shader
   - Write numerical validation test
   - Validate against CPU reference

4. **Create test infrastructure**
   - Reference softmax CPU implementation
   - Numerical gradient checking utility
   - GPU kernel test harness

---

## Documentation to Maintain

- `PHASE5_7_GPU_KERNEL_IMPLEMENTATION_KICKOFF.md` (this file)
- `PHASE5_7_GPU_KERNELS_STATUS.md` (daily progress)
- Inline shader documentation
- Test result logs

---

## Rollout Plan (After Phase 5.7)

Once Phase 5.7 complete:
1. Deploy to staging
2. Run 100-epoch training validation
3. Benchmark against CPU baseline
4. Document performance metrics
5. Prepare for Phase 6.0 (multi-GPU support)

---

Ready to begin Phase 5.7. **Start with Day 1: Softmax Gradient Kernel.**

