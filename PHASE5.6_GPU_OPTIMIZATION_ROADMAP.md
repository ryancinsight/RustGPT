# Phase 5.6 GPU Optimization Roadmap

## Current State: Foundation Complete ✅

- ✅ All GPU kernels implemented
- ✅ Automatic detection working  
- ✅ 615 tests passing
- ✅ Zero compilation errors
- ✅ Training binary ready

---

## Optimization Phases

### PHASE A: Baseline Metrics (1 session)

**Goal**: Establish performance baseline for optimization decisions

#### A1: Build & Run Test
```bash
cargo build --release --features gpu-wgpu
./target/release/main.exe  # Run 50 iterations
```

**Measure**:
- [ ] Time per iteration (ms)
- [ ] GPU utilization (%)
- [ ] Memory usage (MB)
- [ ] GPU memory (MB)
- [ ] Batch throughput (tokens/sec)

#### A2: Profile GPU Operations
```bash
# Linux/Mac: use nvidia-smi or similar
# Windows: Use GPU-Z or similar tools
```

**Identify bottlenecks**:
- [ ] CPU-GPU transfer time
- [ ] Kernel execution time
- [ ] Memory allocation overhead
- [ ] Synchronization points

#### A3: Compare CPU vs GPU
```bash
cargo run --release  # CPU only
./target/release/main.exe

cargo run --release --features gpu-wgpu  # GPU
./target/release/main.exe
```

**Target**: GPU should be 3-5x faster for forward pass

---

### PHASE B: Memory Optimization (1-2 sessions)

**Goal**: Reduce GPU memory pressure and transfers

#### B1: Identify Hot Paths
Profile GPU memory operations:
- [ ] Input/output transfers
- [ ] Activation buffer allocations
- [ ] Gradient buffer allocations

**Files to examine**:
- src/domain/richard/richards_glu.rs (forward_gpu line 167-188)
- src/domain/attention/poly_attention.rs (forward_gpu line 1639-1659)
- src/domain/layers/components/gpu_gemm_kernels.rs

#### B2: Implement Buffer Pooling
```rust
// Pre-allocate buffers at startup instead of per-iteration
pub struct GpuBufferPool {
    input_buffers: Vec<GpuBuffer>,
    output_buffers: Vec<GpuBuffer>,
    temp_buffers: Vec<GpuBuffer>,
}

impl GpuBufferPool {
    pub fn allocate_power_of_two(size: usize) -> GpuBuffer {
        // Use power-of-2 sizing to match workspace pattern
    }
}
```

**Expected improvement**: 5-10% speedup from reduced allocation overhead

#### B3: Reduce Transfers
Identify opportunities to:
- [ ] Keep intermediate results on GPU
- [ ] Batch multiple operations
- [ ] Use in-place operations where possible
- [ ] Reduce host-device sync points

---

### PHASE C: Kernel Fusion (2-3 sessions)

**Goal**: Reduce kernel launch overhead and bandwidth

#### C1: RichardsGlu Fusion

**Current**: Separate kernels
```
Input → Linear(w_gate) → Sigmoid
     → Linear(w_value) → Activation
     → Multiply (GLU)
     → Linear(w_out) → Output
```

**Target**: Fused kernel
```
Input → [Linear + Sigmoid + Linear + Activation + GLU + Linear] → Output
```

**Implementation location**: src/domain/compute/richardson_glu_fused_kernel.rs

**Expected improvement**: 15-20% speedup + 20% bandwidth reduction

#### C2: Attention QKV Projection Fusion

**Current**: 3 separate GEMMs
```
Q = Input @ W_Q.T
K = Input @ W_K.T  
V = Input @ W_V.T
```

**Target**: Single fused GEMM
```
[Q, K, V] = Input @ [W_Q, W_K, W_V].T (single operation)
```

**Implementation location**: src/domain/layers/components/gpu_gemm_kernels.rs

**Expected improvement**: 10-15% speedup

#### C3: Attention Output Projection Fusion

**Current**: 
```
Attention → Linear(W_O) → Output
```

**Target**: 
```
Attention → [Linear + potential activation] → Output (fused)
```

---

### PHASE D: Batch Optimization (1-2 sessions)

**Goal**: Maximize GPU throughput with appropriate batch sizes

#### D1: Determine Optimal Batch Size
```bash
# Test with varying batch sizes
for batch in 4 8 16 32 64 128; do
  ./main --batch-size $batch 2>&1 | grep "throughput\|memory"
done
```

**Track**:
- [ ] GPU memory usage
- [ ] Tokens/sec throughput
- [ ] Power consumption
- [ ] Thermal limits

#### D2: Gradient Accumulation on GPU
```rust
// Keep gradients on GPU between batches
pub struct GpuGradientAccumulator {
    grad_buffers: Vec<GpuBuffer>,
    accumulation_steps: usize,
}
```

**Expected benefit**: Larger effective batch without host transfers

---

### PHASE E: Advanced Optimizations (Optional)

#### E1: Asynchronous GPU Operations
```rust
pub async fn forward_gpu_async(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Don't wait for result immediately
    // Continue CPU work while GPU executes
}
```

#### E2: Multi-GPU Support
```rust
pub struct MultiGpuExecutor {
    devices: Vec<GpuDevice>,
    load_balancer: LoadBalancer,
}
```

#### E3: Mixed Precision (FP16)
```rust
// Use float16 on GPU, float32 on CPU where beneficial
pub fn forward_gpu_mixed_precision(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    // Convert to FP16 on GPU
    // Execute kernels
    // Convert back to FP32
}
```

---

## Implementation Order

### Must-Have (Critical Path)
1. **Baseline metrics** (1 session) - Everything else depends on this
2. **Buffer pooling** (1 session) - Quick win, 5-10% speedup
3. **RichardsGlu fusion** (2 sessions) - 15-20% speedup
4. **QKV projection fusion** (1 session) - 10-15% speedup

### Should-Have (High Value)
5. **Batch optimization** (1 session) - Maximize hardware utilization
6. **Attention output fusion** (1 session) - 5-10% speedup

### Nice-to-Have (Advanced)
7. Async GPU operations
8. Multi-GPU support
9. Mixed precision

---

## Performance Targets

### By Phase
| Phase | Expected Speedup | Session Count |
|-------|------------------|---------------|
| Current | 1.0x (baseline) | 0 |
| After A | 1.0x (measured) | 1 |
| After B | 1.05-1.1x | 2 |
| After C | 1.2-1.4x | 5 |
| After D | 1.3-1.5x | 6 |

### Final Target
- **3-5x faster than CPU** (typical for GPU acceleration)
- **50-75% GPU utilization** (realistic for inference)
- **<6GB GPU memory** (for small models)
- **<100ms iteration** (for realistic batch size)

---

## Testing Strategy

### Unit Tests
```bash
# Test individual kernels
cargo test --lib kernel --features gpu-wgpu
```

### Integration Tests
```bash
# Test end-to-end training
cargo test --test transformer_block_verification --features gpu-wgpu
```

### Performance Tests
```bash
# Benchmark specific components
cargo bench --bench gpu_forward_pass
```

### Regression Prevention
```bash
# Run full test suite before/after each optimization
cargo test --lib --features gpu-wgpu
```

---

## Metrics to Track

### Performance
- [ ] Iterations per second
- [ ] Tokens processed per second
- [ ] Loss convergence rate
- [ ] Gradient update frequency

### Resource Usage
- [ ] GPU memory peak
- [ ] GPU memory average
- [ ] GPU utilization %
- [ ] Power consumption (W)
- [ ] Temperature (°C)

### Quality
- [ ] Test pass rate (615/615)
- [ ] Numerical accuracy vs CPU
- [ ] Gradient correctness
- [ ] Memory leak detection

---

## Success Criteria

### Phase Completion
- ✅ Baseline established
- ✅ All tests still passing
- ✅ Performance improved by target amount
- ✅ No memory leaks detected
- ✅ No accuracy degradation

### Final Success
- ✅ 3-5x GPU speedup achieved
- ✅ All 615 tests passing
- ✅ Production-ready code
- ✅ Optimization complete

---

## Key Locations for Implementation

| Component | File | Key Functions |
|-----------|------|---------------|
| RichardsGlu | src/domain/richards/richards_glu.rs | forward_gpu (L151), forward_gpu_kernel (L193) |
| PolyAttention | src/domain/attention/poly_attention.rs | forward_gpu (L1615), backward_gpu (L1695) |
| GEMM Kernels | src/domain/layers/components/gpu_gemm_kernels.rs | wgpu_gemm functions |
| Buffer Pool | src/domain/compute/unified_gpu_buffer_pool.rs | allocation logic |
| GPU Device | src/domain/compute/gpu_device.rs | execution_context() |
| Workspace | src/domain/layers/components/unified_layer_workspace.rs | buffer management |

---

## Risk Mitigation

### Risk: Performance regression
**Mitigation**: Run full test suite + benchmarks before/after each change

### Risk: Memory leaks
**Mitigation**: Monitor GPU memory over long runs, use Valgrind if available

### Risk: Numerical inaccuracy
**Mitigation**: Compare GPU results with CPU reference on test data

### Risk: Hardware-specific issues
**Mitigation**: Test on multiple systems (NVIDIA preferred, Intel/AMD if available)

---

## Conclusion

The GPU foundation is solid. Optimization phases are ordered by impact and complexity.

**Recommended approach**: Complete Phase A immediately, use results to prioritize Phases B-D.

**Timeline estimate**: 4-6 weeks for full optimization cycle (3-5x speedup goal).

Start with Phase A baseline - it's quick and informs everything else.
