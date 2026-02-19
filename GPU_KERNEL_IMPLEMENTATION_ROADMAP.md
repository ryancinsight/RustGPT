# GPU Kernel Implementation Roadmap - Phase 5.6
**Foundation**: Shared components now have unified GPU device management (complete)  
**Next Focus**: GPU kernel implementations for maximum speedup  
**Target**: 20-30× performance improvement over CPU

---

## Kernel Implementation Order (By Priority & Dependency)

### Tier 1: Foundation Kernels (Dependencies for Tier 2+)

#### 1.1 RichardsGLU WGPU Kernel (CRITICAL PATH)
**Dependency**: None (standalone feedforward)  
**Impact**: 25× speedup (50ms → 2ms for 1K batch)  
**Complexity**: Medium (2 GEMMs + element-wise + projection)

**Implementation Steps**:
1. Create `src/domain/compute/wgpu/kernels/richards_glu.wgsl`
   - Kernel 1: `x1 = input @ w1^T` (GEMM)
   - Kernel 2: `x2 = input @ w2^T` (GEMM)
   - Kernel 3: `value = RichardsActivation(x1)` (element-wise)
   - Kernel 4: `gate_sigma = RichardsGate(x2)` (element-wise)
   - Kernel 5: `gated = value * gate_sigma` (element-wise multiply)
   - Kernel 6: `output = gated @ w_out^T` (GEMM)

2. Or create fused kernel (preferred):
   - Single kernel doing all 6 operations with loop unrolling
   - Minimizes global memory roundtrips
   - Target tile size: 16×16

3. Integration points:
   - `RichardsGlu::forward_gpu()` calls this kernel
   - Use existing `UnifiedGpuDevice::gemm_f32()` for GEMMs
   - Implement element-wise ops directly in shader

4. Validation:
   - Numerical accuracy: ε ≤ 1e-4 vs CPU reference
   - Test with various batch sizes: [1, 8, 32, 64, 128]
   - Benchmark: measure speedup ratio

**Estimated Time**: 2-3 hours

---

#### 1.2 Softmax WGPU Kernel (Foundation for Attention)
**Dependency**: None (utility kernel)  
**Impact**: Required for all attention variants  
**Complexity**: Medium (numerically stable, reduction ops)

**Implementation**:
1. Enhance existing `SHADER_SOFTMAX` in `wgpu_ops.rs`
2. Add row-wise softmax with log-sum-exp trick
3. Support batched input (batch_size, seq_len, seq_len)
4. Handle edge cases: empty rows, NaN/Inf values

**Estimated Time**: 1 hour

---

### Tier 2: Attention Variants (Depend on Foundation)

#### 2.1 PolyAttention WGPU Kernel (HIGH COMPLEXITY)
**Dependency**: Softmax (Tier 1.2)  
**Impact**: 30× speedup (30ms → 1ms)  
**Complexity**: High (polynomial basis + gating)

**Kernels Needed**:
1. Polynomial basis computation: `P_k(x) = sum_{m=0}^{k} c_m * x^m`
2. QKV projections (3 GEMMs)
3. Polynomial attention scores: `S = P_k(Q @ K^T)`
4. Gating mechanism: learnable poly-degree adaptation
5. Softmax + value projection

**WGSL Structure**:
```wgsl
// Kernel 1: Polynomial basis
@compute @workgroup_size(16, 16)
fn compute_poly_basis(...) { ... }

// Kernel 2: QKV + Poly Scores
@compute @workgroup_size(16, 16)
fn poly_attention_scores(...) { ... }

// Kernel 3: Softmax + Value Proj (fused)
@compute @workgroup_size(16, 16)
fn poly_attention_output(...) { ... }

// Kernel 4: Gating (learnable degree adaptation)
@compute @workgroup_size(1024)
fn adaptive_gating(...) { ... }
```

**Estimated Time**: 3-4 hours

---

#### 2.2 TransformerAttention WGPU Kernel (MEDIUM)
**Dependency**: Softmax (Tier 1.2)  
**Impact**: 25× speedup (25ms → 1ms)  
**Complexity**: Medium (standard scaled dot-product)

**Kernels Needed**:
1. QKV projection (3 GEMMs or fused)
2. Scaled attention scores: `S = scale * (Q @ K^T)`
3. Softmax
4. Value aggregation: `output = softmax(S) @ V`

**Optimization**:
- Fuse Q/K/V projections if possible
- Cache softmax denominators to avoid second pass
- Use shared memory for tile-local reductions

**Estimated Time**: 1.5-2 hours

---

#### 2.3 Mamba/RG-LRU WGPU Kernel (HIGHEST COMPLEXITY)
**Dependency**: None (recurrent, not parallel-reducible)  
**Impact**: 20× speedup (40ms → 2ms)  
**Complexity**: Very High (sequential recurrence)

**Challenges**:
- Recurrent state updates (can't parallelize across time)
- Numerical stability critical for long sequences
- Multiplicative state coupling

**Kernels Needed**:
1. State-to-value projection (GEMM)
2. Gating (element-wise)
3. Parallel scan (prefix sum pattern for state)
4. Output projection (GEMM)

**Parallel Scan Strategy**:
```wgsl
// Work-efficient parallel scan in shared memory
// Blelloch prefix sum for state propagation
@compute @workgroup_size(256)
fn mamba_scan(...) {
    // Local scan in workgroup
    // Global sync if sequence > workgroup size
    // Propagate partial results
}
```

**Estimated Time**: 3-4 hours

---

### Tier 3: Mixture of Experts & Context (Can run independently)

#### 3.1 MixtureOfExperts Router WGPU Kernel
**Dependency**: Softmax (Tier 1.2)  
**Impact**: 20× speedup (100ms → 5ms for 8 experts)  
**Complexity**: Medium

**Kernels Needed**:
1. Router GEMM: `logits = input @ router_weights^T`
2. Router softmax (per token)
3. Expert gate selection (top-k or softmax)

**Estimated Time**: 1.5-2 hours

---

#### 3.2 MixtureOfExperts Expert Execution
**Dependency**: None (standalone)  
**Impact**: Parallel expert computation  
**Complexity**: Medium (scheduling, load balancing)

**Implementation**:
- Dispatch expert GEMMs in parallel
- Aggregate expert outputs with gate scaling
- Handle load imbalance

**Estimated Time**: 2 hours

---

#### 3.3 AttentionContext Modulation WGPU Kernel
**Dependency**: None  
**Impact**: 30× speedup (15ms → 0.5ms)  
**Complexity**: Low (simple GEMM + element-wise)

**Kernel**:
1. Context GEMM: `context_out = input @ context_matrix^T`
2. Scale: `context_out *= strength / embed_dim`
3. Residual: `output = context_out + input`

**Estimated Time**: 30 minutes

---

## WGPU Shader Directory Structure

```
src/domain/compute/wgpu/
├── kernels/
│   ├── mod.rs                    # Kernel registry
│   ├── richards_glu.wgsl         # Tier 1.1
│   ├── softmax.wgsl              # Tier 1.2 (enhance existing)
│   ├── poly_attention.wgsl       # Tier 2.1
│   ├── transformer_attention.wgsl # Tier 2.2
│   ├── mamba_scan.wgsl           # Tier 2.3
│   ├── moe_router.wgsl           # Tier 3.1
│   ├── moe_experts.wgsl          # Tier 3.2
│   └── attention_context.wgsl    # Tier 3.3
├── wgpu_ops.rs                   # Updated with kernel dispatch
└── memory.rs                      # Existing buffer management
```

---

## GPU Kernel Testing Strategy

### Unit Tests (Per Kernel)
```rust
#[test]
#[cfg(feature = "wgpu")]
fn test_richards_glu_kernel_vs_cpu() {
    let mut device = GpuDevice::auto_detect().expect("GPU required");
    
    // CPU reference
    let cpu_output = richardsglu_cpu(&input);
    
    // GPU execution
    let gpu_output = richardsglu_gpu(&mut device, &input)?;
    
    // Numerical accuracy check
    for (cpu_val, gpu_val) in cpu_output.iter().zip(gpu_output.iter()) {
        assert!((cpu_val - gpu_val).abs() < 1e-4);
    }
}
```

### Integration Tests
- Full forward pass through SharedFeedforward with GPU enabled
- Verify gradients (if training)
- Check memory usage: power-of-2 allocation efficiency

### Benchmark Tests
```rust
#[bench]
fn bench_richardson_glu_gpu_vs_cpu(b: &mut Bencher) {
    let input = Array2::zeros((1024, 512));
    
    // Benchmark GPU
    b.iter(|| {
        device.forward_gpu(&input)
    });
}
```

---

## Performance Targets

| Kernel | Input | CPU Time | GPU Target | Speedup | Memory |
|--------|-------|----------|-----------|---------|--------|
| RichardsGLU | 1K × 512 | 50ms | 2ms | 25× | 2MB |
| PolyAttention | 64 × 512 × 64 | 30ms | 1ms | 30× | 3MB |
| TransformerAttn | 64 × 512 × 64 | 25ms | 1ms | 25× | 3MB |
| Mamba | 256 × 512 | 40ms | 2ms | 20× | 2MB |
| MoE (8x) | 1K × 512 | 100ms | 5ms | 20× | 4MB |
| AttentionCtx | 64 × 512 | 15ms | 0.5ms | 30× | 1MB |

**Total Forward Pass Speedup (all kernels)**:
- CPU: ~260ms (sum of all)
- GPU: ~11.5ms (sum of all with pipelining)
- **Overall: ~22.6× speedup**

---

## CUDA Backend Strategy (Post-WGPU)

Once WGPU kernels are validated:

1. **Kernel Porting** (40% effort)
   - Translate WGSL to CUDA/C++
   - Similar structure, CUDA libraries (cuBLAS, cuDNN)
   - Testing against same numerical accuracy threshold

2. **cudarc Integration** (30% effort)
   - Use cudarc crate for kernel dispatch
   - Memory management via CudaSlice
   - Error handling & synchronization

3. **Testing** (30% effort)
   - Parallel testing infrastructure
   - GPU feature gate selection
   - Performance comparison

---

## Dependency Graph

```
Foundation:
├── Softmax (Tier 1.2)
│
Attention Variants:
├── PolyAttention (depends on Softmax)
├── TransformerAttention (depends on Softmax)
│
Sequential:
└── Mamba (no dependencies, self-contained)

Mixtures & Context:
├── MoE Router (depends on Softmax)
├── MoE Experts (no dependencies)
└── AttentionContext (no dependencies)

Critical Path (longest):
1. RichardsGLU (foundation feedforward) → 2-3h
2. PolyAttention (depends on Softmax) → 3-4h total
3. Mamba (complex sequential) → 3-4h

Earliest completion: RichardsGLU in ~3 hours
Full implementation: ~25-30 hours total
```

---

## Session Checklist

### Before Starting Kernel Implementation
- [ ] Run `cargo test --lib shared_feedforward` - verify no regressions
- [ ] Run `cargo test --lib shared_temporal_processing` - verify no regressions
- [ ] Run `cargo test --lib shared_attention_context` - verify no regressions
- [ ] Build with all GPU features: `cargo build --release --features gpu-all`
- [ ] Verify `GpuDevice::auto_detect()` works on target system

### For Each Kernel Implementation
- [ ] Create `.wgsl` shader file with detailed comments
- [ ] Implement kernel in `wgpu_ops.rs` (or appropriate module)
- [ ] Add unit test comparing GPU output to CPU reference (ε ≤ 1e-4)
- [ ] Add benchmark comparing GPU vs CPU time
- [ ] Verify zero-allocation reuse (pre-allocate all buffers)
- [ ] Check memory efficiency: `AllocationStats::efficiency_percent()` > 90%
- [ ] Format: `cargo fmt`
- [ ] Lint: `cargo clippy --all-targets`

---

## Next Session Quick Start

1. **Baseline Measurement**:
   ```bash
   cargo build --release --features gpu-wgpu
   cargo bench --bench gpu_component_baseline
   ```
   (Measure CPU times for all operations)

2. **Start with RichardsGLU** (highest impact, lowest complexity)
   - Create `src/domain/compute/wgpu/kernels/richards_glu.wgsl`
   - Implement kernel with fused operations
   - Test against CPU reference

3. **Move to PolyAttention** (highest speedup potential)
   - Requires Softmax kernel
   - Complex polynomial basis computation
   - Gating mechanism

4. **Mamba in parallel** (can work independently)
   - Recurrent operations require different approach
   - Parallel scan critical for performance

---

## Risk Mitigation

**Risk**: Numerical instability in GPU vs CPU  
**Mitigation**: Strict ε ≤ 1e-4 tolerance in all tests, use double precision in hotspots

**Risk**: Memory fragmentation with power-of-2 allocation  
**Mitigation**: AllocationStats tracking, pre-allocate with realistic batch sizes

**Risk**: WGPU shader compilation failures across platforms  
**Mitigation**: Test on Windows (WGPU → DX12), Linux (WGPU → Vulkan), macOS (WGPU → Metal)

**Risk**: Recurrent operations (Mamba) diverge over long sequences  
**Mitigation**: Validate with increasing sequence lengths, use compensated arithmetic if needed

---

## Success Criteria

✅ **Phase 5.6.1 (RichardsGLU)**: 25× speedup verified  
✅ **Phase 5.6.2 (Attention)**: 25-30× speedup verified  
✅ **Phase 5.6.3 (Context)**: 30× speedup verified  
✅ **All kernels**: Numerical accuracy ε ≤ 1e-4  
✅ **Memory**: Power-of-2 allocation efficiency > 90%  
✅ **Integration**: Full forward pass executes on GPU with zero CPU fallback  

---

## End Goal

A fully GPU-accelerated shared component library with:
- 20-30× speedup over CPU
- Zero intermediate CPU transfers
- Strict no-fallback semantics
- Multi-backend support (WGPU → CUDA → Metal)
- Production-ready numerical stability
