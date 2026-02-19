# Phase 5.6 GPU Backend Variants Consolidation Plan
**Date**: February 15, 2026  
**Focus**: Consolidation & GPU Backend Implementation with Strict No-Fallback  
**Status**: IN PROGRESS

## 1. Overview
This phase unifies GPU execution across diffusion, SSM, and transformer shared components with strict GPU-only execution (no CPU fallback). Implementation starts with WGPU backend with explicit error on GPU unavailability.

---

## 2. Current State Assessment

### 2.1 Infrastructure (READY)
- ✅ `GpuDevice` - Unified GPU device abstraction with auto-detection
- ✅ `UnifiedGpuBufferPool` - Power-of-2 buffer management with reuse tracking
- ✅ `GpuComponent` trait - Unified interface for all GPU-capable components
- ✅ `gpu_device.rs`, `gpu_ops.rs`, `wgpu_ops.rs` - Core GPU primitives

### 2.2 Shared Components (PARTIAL)
| Component | CPU | GPU | Status |
|-----------|-----|-----|--------|
| **SharedFeedforward** | ✅ Complete | 🔄 In-Progress | RichardsGLU GPU kernel complete; MoE placeholder |
| **SharedTemporalProcessing** | ✅ Complete | ⚠️ Partial | PolyAttention needs WGPU kernel |
| **SharedAttentionContext** | ✅ Complete | ❌ Not Started | Needs GEMM + softmax kernel fusion |

### 2.3 GPU Backends Status
| Backend | Feature | Compute | Memory | Ops | Status |
|---------|---------|---------|--------|-----|--------|
| **WGPU** | `wgpu` | ✅ | ✅ | ✅ | **Primary** |
| **CUDA** | `gpu-cuda` | ⚠️ Skeleton | ⚠️ Skeleton | ⚠️ Skeleton | Secondary |
| **Metal** | `gpu-metal` | ⚠️ Skeleton | ⚠️ Skeleton | ⚠️ Skeleton | Tertiary |

---

## 3. Phase 5.6 Implementation Roadmap

### **Phase 5.6.1: SharedFeedforward GPU Variants (Week 1)**

#### 5.6.1a: RichardsGLU WGPU Optimization
- **Status**: Kernel exists, needs integration
- **Location**: `src/domain/richards/richards_glu.rs` + WGPU variant
- **Tasks**:
  1. Create `src/domain/compute/wgpu/kernels/richards_glu.wgsl` (shader)
  2. Implement fused kernel: 2 GEMMs + Richards curve + gate multiply + output proj
  3. Benchmark: Target 25× speedup (50ms CPU → 2ms GPU for 1K batch)
  4. Test tolerance: ε ≤ 1e-4 vs CPU reference

#### 5.6.1b: MixtureOfExperts WGPU Kernel
- **Status**: Placeholder skeleton only
- **Location**: `src/domain/mixtures/moe.rs`
- **Tasks**:
  1. Create `src/domain/compute/wgpu/kernels/moe_router.wgsl` (router GEMM + softmax)
  2. Create `src/domain/compute/wgpu/kernels/moe_weighted_sum.wgsl` (weighted accumulation)
  3. Implement parallel expert execution
  4. Benchmark: Target 20× speedup (100ms CPU → 5ms GPU for 8 experts)

#### 5.6.1c: FeedforwardGpu Integration
- **Status**: `feedforward_gpu.rs` exists but needs unification
- **Tasks**:
  1. Consolidate `feedforward_gpu.rs` with `SharedFeedforward`
  2. Implement `GpuComponent` trait for `SharedFeedforward`
  3. Add GPU memory pre-allocation in `forward_gpu()`
  4. Add GPU-specific error messages (no silent CPU fallback)

---

### **Phase 5.6.2: SharedTemporalProcessing GPU Variants (Week 2)**

#### 5.6.2a: PolyAttention WGPU Kernel
- **Status**: Needs custom kernel for polynomial basis + gating
- **Location**: `src/domain/attention/poly_attention.rs`
- **Tasks**:
  1. Create `src/domain/compute/wgpu/kernels/poly_attention.wgsl`
  2. Implement: basis computation + QKV projection + attention scores + gating
  3. Benchmark: Target 30× speedup (30ms CPU → 1ms GPU)
  4. Test polynomial degree adaptation on GPU

#### 5.6.2b: Mamba/RG-LRU WGPU Kernel
- **Status**: Needs recurrent scan operations
- **Location**: `src/domain/ssm/` (modular state space)
- **Tasks**:
  1. Create `src/domain/compute/wgpu/kernels/mamba_scan.wgsl` (parallel recurrent scan)
  2. Implement fused multiplicative state updates
  3. Benchmark: Target 20× speedup (40ms CPU → 2ms GPU)
  4. Verify stability of recurrent computations on GPU

#### 5.6.2c: TransformerAttention WGPU Kernel
- **Status**: Standard attention, partial GPU support exists
- **Location**: `src/domain/attention/` (standard transformer)
- **Tasks**:
  1. Optimize QKV projection for WGPU
  2. Fuse softmax into attention score computation
  3. Benchmark: Target 25× speedup (25ms CPU → 1ms GPU)

---

### **Phase 5.6.3: SharedAttentionContext GPU Variant (Week 3)**

#### 5.6.3a: WGPU Kernel for Context Modulation
- **Status**: Not started
- **Location**: `src/domain/layers/components/attention_context_gpu.rs`
- **Tasks**:
  1. Create `src/domain/compute/wgpu/kernels/attention_context.wgsl`
  2. Implement: context GEMM + scaling + residual add
  3. Benchmark: Target 30× speedup (15ms CPU → 0.5ms GPU)

#### 5.6.3b: Consolidation
- **Tasks**:
  1. Implement `GpuComponent` trait for `SharedAttentionContext`
  2. Unify CPU and GPU code paths
  3. Test zero-allocation reuse

---

### **Phase 5.6.4: GPU Backend Variant Abstraction (Week 4)**

#### 5.6.4a: Create GpuBackendVariant Trait
```rust
pub trait GpuBackendVariant {
    fn backend_name(&self) -> &'static str;
    fn is_available() -> bool;
    fn initialize() -> Result<Self>;
}
```

#### 5.6.4b: WGPU Variant Implementation
- Consolidate all WGPU kernels under `src/domain/compute/wgpu/`
- Implement `GpuBackendVariant` for WGPU
- Full integration test suite

#### 5.6.4c: CUDA Skeleton → Foundation
- Create CUDA equivalents for all WGPU kernels
- Use cudarc for kernel dispatch
- Parallel testing with WGPU

#### 5.6.4d: Metal Skeleton → Foundation
- Create Metal equivalents (macOS only)
- Use metal-rs for kernel dispatch
- Feature-gated compilation

---

## 4. Strict No-Fallback Implementation Strategy

### 4.1 Error Handling Pattern
```rust
pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let device = require_gpu_or_error(&self.gpu_device, "forward_gpu")?;
    // ... GPU operations ...
    Ok(output)
}
```

### 4.2 Compilation Mode
| Mode | Behavior |
|------|----------|
| `--features gpu-wgpu` | WGPU only; error if not available |
| `--features gpu-cuda` | CUDA only; error if not available |
| `--features gpu-all` | Priority: CUDA > Metal > Vulkan; error if none |
| (no GPU features) | Compilation succeeds, GPU ops return errors at runtime |

### 4.3 Error Messages
All GPU operation failures include:
- The operation name that failed
- The required backend
- How to enable it (e.g., `--features gpu-wgpu`)

---

## 5. Performance Optimization Targets

| Component | Operation | CPU | GPU Target | Gain |
|-----------|-----------|-----|-----------|------|
| SharedFeedforward | RichardsGLU 1K batch | 50ms | 2ms | 25× |
| SharedFeedforward | MoE (8 experts) | 100ms | 5ms | 20× |
| SharedTemporal | PolyAttention | 30ms | 1ms | 30× |
| SharedTemporal | Mamba scan | 40ms | 2ms | 20× |
| SharedTemporal | Transformer QKV | 25ms | 1ms | 25× |
| SharedAttention | Context modulation | 15ms | 0.5ms | 30× |

---

## 6. Consolidation Checklist

### 6.1 Code Deduplication
- [ ] Merge `feedforward_gpu.rs` into `SharedFeedforward`
- [ ] Merge `temporal_processing_gpu.rs` into `SharedTemporalProcessing`
- [ ] Merge `attention_context_gpu.rs` into `SharedAttentionContext`
- [ ] Remove deprecated `CpuGpuMatrixOps` references

### 6.2 GPU Trait Implementation
- [ ] `SharedFeedforward` implements `GpuComponent`
- [ ] `SharedTemporalProcessing` implements `GpuComponent`
- [ ] `SharedAttentionContext` implements `GpuComponent`
- [ ] All implementations test with auto-detection

### 6.3 Memory Efficiency
- [ ] Power-of-2 buffer sizing across all components
- [ ] Zero-allocation reuse verification
- [ ] AllocationStats tracking enabled
- [ ] Benchmark memory overhead (target: < 10%)

### 6.4 Testing
- [ ] Unit tests: Each kernel with CPU tolerance ε ≤ 1e-4
- [ ] Integration tests: Full forward pass across layers
- [ ] Performance benchmarks: Measure speedup against CPU
- [ ] Error handling: No-fallback semantics verified

---

## 7. Implementation Order (Priority)

1. **SharedFeedforward RichardsGLU** (Highest impact, ~25× speedup)
2. **SharedTemporalProcessing PolyAttention** (Complex kernel, new pattern)
3. **SharedFeedforward MoE** (Larger codebase, parallel ops)
4. **SharedTemporalProcessing Mamba** (Recurrent ops, stability critical)
5. **SharedAttentionContext** (Simpler ops, more fusion opportunities)
6. **CUDA/Metal variants** (After WGPU validation)

---

## 8. Dependencies & Integration Points

```
GPU Infrastructure (READY)
├── GpuDevice (auto_detect, no fallback)
├── UnifiedGpuBufferPool (power-of-2, reuse tracking)
└── GpuComponent trait (unified interface)
    │
    ├── SharedFeedforward (GPU variants)
    │   ├── RichardsGLU.wgpu kernel
    │   └── MoE.wgpu kernels
    │
    ├── SharedTemporalProcessing (GPU variants)
    │   ├── PolyAttention.wgpu kernel
    │   ├── Mamba.wgpu kernel
    │   └── TransformerAttn.wgpu kernel
    │
    └── SharedAttentionContext (GPU variant)
        └── AttentionContext.wgpu kernel
```

---

## 9. Session Deliverables

### This Session
- [ ] Phase 5.6.1a complete: RichardsGLU WGPU integration
- [ ] Shared components implement `GpuComponent` trait
- [ ] Auto-detection tested with strict no-fallback
- [ ] Benchmarks for 25× speedup target

### Next Session
- [ ] Phase 5.6.1b: MoE kernels
- [ ] Phase 5.6.2a: PolyAttention kernels
- [ ] Consolidate feedforward_gpu.rs into SharedFeedforward

---

## 10. Files to Create/Modify

### Create
```
src/domain/compute/wgpu/kernels/
├── richards_glu.wgsl
├── moe_router.wgsl
├── moe_weighted_sum.wgsl
├── poly_attention.wgsl
├── mamba_scan.wgsl
├── attention_context.wgsl
└── mod.rs (kernel registration)
```

### Modify
```
src/domain/layers/components/
├── feedforward.rs (GpuComponent impl)
├── feedforward_gpu.rs (merge/deprecate)
├── temporal_processing.rs (GpuComponent impl)
├── temporal_processing_gpu.rs (merge/deprecate)
├── attention_context.rs (GpuComponent impl)
└── attention_context_gpu.rs (merge/deprecate)
```

---

## End Goal
A unified, consolidated GPU-enabled LLM training framework where:
- All shared components support GPU with strict no-fallback semantics
- WGPU backend provides 20-30× speedups over CPU
- Power-of-2 buffer allocation minimizes memory waste
- Clear error messages guide troubleshooting
- CUDA/Metal backends available as drop-in alternates
