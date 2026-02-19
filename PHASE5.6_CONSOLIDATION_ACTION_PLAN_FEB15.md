# Phase 5.6: GPU Consolidation & Optimization Action Plan (Feb 15, 2026)

## Current Status
- **Overall**: ~50% complete (Phase 5.6.1 validation done, Phase 5.6.2 backward pass foundation laid)
- **Completed**: Auto-detection, forward pass testing, numerical validation framework
- **In Progress**: Fused kernel implementation, component GPU variants
- **Deprecated**: `shared_gpu_manager.rs` → consolidate into `unified_gpu_buffer_pool.rs`

---

## Phase 5.6.3: Fused Kernels & Performance Optimization

### Priority 1: RichardsGLU Fused Kernel (HIGH IMPACT)
**Target**: Reduce 5 GPU launches → 2 passes; 50ms → 2ms (1K batch)

**Current State**:
- `richards_glu.rs`: Has `forward_gpu()` but executes separate kernels
  - W1 projection (GEMM)
  - Richards activation (custom kernel)
  - W2 projection (GEMM)
  - Gating (element-wise)
  - W_out projection (GEMM)

**Implementation Path**:
1. **Two-Pass Strategy**:
   - **Pass 1**: W1 → Richards + W2 gate computation (fused kernel in `unified_gpu_kernels.rs`)
   - **Pass 2**: Gated hidden → W_out projection (standard GEMM)
   
2. **Kernel Implementation**:
   - Create `RichardsGluFusedKernel` struct in `unified_gpu_kernels.rs`
   - WGSL/CUDA: Combine W1, Richards curve, gating into single kernel
   - Use shared memory for intermediate values
   
3. **Integration**:
   - Modify `RichardsGlu::forward_gpu()` to use fused kernel
   - Update `feedforward_gpu.rs` dispatch
   - Add benchmarks to track speedup

**Acceptance Criteria**:
- ✓ Compile without errors/warnings
- ✓ Numerical accuracy ≤ 1e-4 vs unfused CPU
- ✓ 2 GPU launches (verified via stats)
- ✓ Test passes on auto-detected GPU

---

### Priority 2: PolyAttention GPU Kernel (HIGH IMPACT)
**Target**: Reduce 3+ GPU launches → 1 pass; 30ms → 1ms (512 batch)

**Current State**:
- `temporal_processing_gpu.rs`: `forward_gpu_poly_attention()` is stub
- Uses CPU fallback for polynomial scoring + softmax + projection

**Implementation Path**:
1. **Single-Pass Kernel**:
   - Q projection: input @ W_q (GEMM)
   - K projection: input @ W_k (GEMM)
   - Polynomial scores: Q @ K^T (custom polynomial basis kernel)
   - Softmax: GPU softmax kernel
   - Output: scores @ V @ W_o (GEMM)

2. **Kernel Implementation**:
   - Create `PolyAttentionFusedKernel` in `unified_gpu_kernels.rs`
   - Polynomial basis computation (configurable degree)
   - Numerically stable softmax

3. **Integration**:
   - Route through `TemporalMixingLayer::forward_gpu_dispatch()`
   - Validate against `PolyAttention::forward_cpu()`

**Acceptance Criteria**:
- ✓ Compile without errors
- ✓ Numerical accuracy ≤ 1e-3 vs CPU (polynomial basis may have small differences)
- ✓ Single GPU kernel launch
- ✓ Test passes

---

### Priority 3: Mamba Selective Scan GPU Kernel (MEDIUM IMPACT)
**Target**: 20x speedup; 40ms → 2ms (512 batch)

**Current State**:
- `temporal_processing_gpu.rs`: `forward_gpu_mamba()` is stub
- CPU implementation in `src/domain/layers/ssm/`

**Implementation Path**:
1. **Selective Scan Kernel**:
   - Recurrent computation of state updates
   - Can't be fully fused (recurrent dependency)
   - GPU: Parallel scan with shared state

2. **Kernel Implementation**:
   - Create `MambaScanKernel` in `unified_gpu_kernels.rs`
   - Sequential state evolution (inherently recurrent)
   - Use warp-level reductions for efficiency

3. **Integration**:
   - Route through `TemporalMixingLayer::forward_gpu_dispatch()`
   - Validate against Mamba/Mamba2 CPU impl

**Acceptance Criteria**:
- ✓ Numerical accuracy ≤ 1e-3
- ✓ >10x speedup vs CPU
- ✓ Handles variable batch/seq lengths

---

### Priority 4: AttentionContext GPU Ops (MEDIUM IMPACT)
**Target**: 30x speedup; 15ms → 0.5ms (1K batch)

**Current State**:
- `attention_context_gpu.rs`: Trait-only, no GPU implementation
- Similarity matrix computation + modulation

**Implementation Path**:
1. **Operations**:
   - `apply_incoming_context_gpu()`: input @ context_strength
   - `update_outgoing_context_gpu()`: (input^T @ output) / batch_size

2. **Kernel Implementation**:
   - Use GPU GEMM directly (already optimized in GpuDevice)
   - No custom kernel needed - leverage existing ops

3. **Integration**:
   - Implement trait methods in `attention_context_gpu.rs`
   - Wire into `SharedAttentionContext`

**Acceptance Criteria**:
- ✓ Compile without errors
- ✓ Numerical accuracy ≤ 1e-4
- ✓ Uses GpuDevice GEMM operations

---

## Phase 5.6.4: Zero-Copy Forward Pipeline

### Goal
Keep data on GPU across full forward pass: Attention → Temporal → Feedforward

**Implementation**:
1. **Data Lifetime Management**:
   - Allocate buffers once per batch (not per operation)
   - Reuse through `UnifiedBufferPool` with power-of-2 sizing
   - Return buffers to pool only at end of forward pass

2. **Integration Points**:
   - `UnifiedGpuBackend`: Manages unified buffer pool
   - `SharedAttentionContext`: Use GPU-allocated buffers
   - `SharedTemporalProcessing`: Receive buffers, output buffers
   - `SharedFeedforward`: Consume buffers, return final output

3. **Validation**:
   - Track buffer allocations (target: 1 per forward pass)
   - Memory efficiency >92% (actual_used / total_allocated)

**Acceptance Criteria**:
- ✓ Zero CPU ↔ GPU transfers between operations
- ✓ Single buffer pool allocation per forward pass
- ✓ >92% memory efficiency
- ✓ Test validates full pipeline

---

## Phase 5.6.5: Shared Component Consolidation

### Components Needing GPU Variants
1. **PolyAttention**: GPU kernel (Priority 2)
2. **Mamba/Mamba2**: Selective scan kernel (Priority 3)
3. **RG-LRU**: Recurrent computation kernel
4. **SharedAttentionContext**: GPU ops (Priority 4)
5. **SharedFeedforward**: RichardsGLU fused (Priority 1)

### Consolidation Tasks
1. **Remove CPU-only paths** from GPU-enabled components
2. **Implement missing GPU dispatch** methods
3. **Add comprehensive tests** for each variant
4. **Profile & optimize** hot paths

---

## Strict No-Fallback GPU Detection

### Enforcement Strategy
- ✓ `UnifiedGpuBackend::auto_detect()` returns error if no GPU
- ✓ `UnifiedGpuBackend::new(ComputeBackend::Cpu)` returns error
- ✓ All GPU operations error if device not attached
- ✓ Tests skip if GPU unavailable (don't fall back)

### Validation
Run tests with GPU detection:
```bash
cargo test --test gpu_shared_components_phase56 2>&1 | grep -E "GPU|Backend|detected"
cargo test --test gpu_richards_verification 2>&1
cargo test --test gpu_component_integration 2>&1
```

---

## Development Sequence

1. **Clean warnings** (5 min)
   - Remove unused imports from gpu_backend, richards_glu, tests
   
2. **Implement RichardsGLU fused kernel** (90 min)
   - Create kernel structure
   - Implement WGSL/CUDA variants
   - Integrate into forward_gpu path
   - Validate numerically

3. **Implement PolyAttention GPU kernel** (90 min)
   - Create kernel structure
   - Polynomial scoring computation
   - Full forward pass fusing
   - Validate

4. **Implement Mamba selective scan** (60 min)
   - Create kernel structure
   - Sequential scan implementation
   - Validate accuracy

5. **Implement AttentionContext GPU ops** (30 min)
   - Wire trait methods
   - Use existing GEMM operations
   - Test integration

6. **Zero-copy pipeline** (60 min)
   - Integrate buffer pool with forward pass
   - Add memory tracking
   - Validate efficiency

7. **Testing & optimization** (60 min)
   - Run full test suite
   - Profile hot paths
   - Document performance gains

---

## Code Organization

### Files to Create
- `src/domain/layers/components/fused_kernels/` (new directory)
  - `mod.rs`
  - `richards_glu_fused.rs`
  - `poly_attention_fused.rs`
  - `mamba_scan_fused.rs`

### Files to Modify
- `src/domain/layers/components/unified_gpu_kernels.rs` (add fused kernel types)
- `src/domain/layers/components/unified_gpu_backend.rs` (refine dispatch)
- `src/domain/richardson/richards_glu.rs` (use fused kernel)
- `src/domain/layers/components/temporal_processing_gpu.rs` (implement stubs)
- `src/domain/layers/components/attention_context_gpu.rs` (implement ops)

### Files to Remove/Deprecate
- `src/domain/layers/components/shared_gpu_manager.rs` (mark deprecated)

---

## Performance Targets (Phase 5.6 Completion)

| Component | Operation | CPU Ref | GPU Target | Speedup |
|-----------|-----------|---------|------------|---------|
| RichardsGLU | 1K batch | 50ms | 2ms | 25x |
| PolyAttention | 512 batch | 30ms | 1ms | 30x |
| Mamba Scan | 512 batch | 40ms | 2ms | 20x |
| AttentionContext | 1K batch | 15ms | 0.5ms | 30x |
| Full forward pass | 1K tokens | 200ms | 10ms | 20x |

---

## Success Metrics

- ✅ All 5 components have GPU implementations
- ✅ Fused kernels reduce launches by 50%+
- ✅ Numerical accuracy ≤ 1e-3 (ε ≤ 1e-4 for attention)
- ✅ >92% memory efficiency
- ✅ All tests pass on auto-detected GPU
- ✅ Strict no-fallback enforcement verified
- ✅ Performance gains documented & benchmarked
