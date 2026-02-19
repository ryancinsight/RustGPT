# Consolidation Session - Feb 14 Immediate Actions

## Current Status Summary
- **Build**: Broken (14 errors + 8 warnings)
- **Test Status**: Unknown (build must pass first)
- **Phase**: Continuation of Phase 5.5 GPU Backend Consolidation

## Priority Consolidation Targets

### TIER 1: CRITICAL BLOCKING ERRORS (Must fix first)

#### 1. PolyAttention GPU Cleanup (poly_attention.rs)
**Status**: 6 compilation errors blocking all tests
- Missing import: `anyhow`
- Missing field: `low_rank_query_gate` in 4 ForwardContext initializers
- Missing argument in function call (6 vs 7 params)
- Missing method: `shape_3d` on GpuBuffer

**Action**: Fix ForwardContext initialization and method signatures

#### 2. Attention Context GPU (attention_context_gpu.rs)
**Status**: 3 errors + 4 warnings
- Missing type: `CpuGpuMatrixOps` (obsolete type)
- Missing import: `SharedAttentionContext`
- Unused variables in stubs

**Action**: Replace CpuGpuMatrixOps with UnifiedGpuExecutor approach

#### 3. Feedforward GPU (feedforward_gpu.rs)
**Status**: 2 errors + field mismatch
- Missing type: `CpuGpuMatrixOps`
- RichardsGlu variant has incompatible field structure
- Accessing non-existent fields (w1, w2, w3, w_gate, gamma, beta, etc.)

**Action**: Verify RichardsGlu struct definition and update field access

#### 4. Temporal Processing GPU (temporal_processing_gpu.rs)
**Status**: 2 errors + 2 missing PolyAttention::default() calls
- Missing type: `CpuGpuMatrixOps`
- PolyAttention doesn't impl Default

**Action**: Implement proper initialization without Default trait

#### 5. Common Layer Wrapper (common.rs)
**Status**: 2 errors
- TemporalMixing variant (PolyAttention) doesn't have forward_gpu method
- Trait dispatch mismatch

**Action**: Implement GpuTemporalOps for PolyAttention

### TIER 2: DESIGN ISSUES (Consolidation)

#### 1. Duplicate GPU Manager Pattern
- `SharedComponentGpuManager` (in shared components)
- `UnifiedGpuExecutor` (in compute backend)
- `UnifiedGpuBufferPool` (new unified design)

**Consolidation Goal**: 
- Keep UnifiedGpuBufferPool as source of truth
- Update all components to use GpuComponent trait
- Deprecate SharedComponentGpuManager

#### 2. GPU Operation Abstraction Mismatch
- Old pattern: Each component has own GPU methods
- New pattern: Centralized UnifiedGpuExecutor with component dispatch

**Migration Path**:
- Convert all `forward_gpu()` to use UnifiedGpuExecutor
- Implement GpuComponent trait uniformly
- Remove component-specific GPU manager fields

### TIER 3: OPTIMIZATION OPPORTUNITIES

#### 1. Kernel Fusion Points
- Attention + Activation (GELU/SiLU)
- Feedforward + Gating (Richards)
- Temporal + Projection

#### 2. Memory Bandwidth Optimization
- Review buffer allocation patterns in UnifiedGpuBufferPool
- Implement LRU cache for temporal buffers
- Verify power-of-2 sizing is working

#### 3. SSM Recurrent Scan Implementation
- Placeholder kernels in temporal_processing_gpu.rs
- Need real WGSL/CUDA kernel implementations for Mamba/RG-LRU

## Immediate Fix Sequence

### Phase 1: Enable Build (Day 1)
1. Fix PolyAttention ForwardContext initializers
2. Remove CpuGpuMatrixOps references (replace with UnifiedGpuExecutor)
3. Fix RichardsGlu field access
4. Implement PolyAttention::Default or fix initialization
5. Add missing GpuComponent trait impl for PolyAttention

### Phase 2: Consolidate GPU Architecture (Day 2)
1. Audit all GPU-related trait implementations
2. Verify GpuComponent trait uniformity
3. Consolidate manager classes
4. Test GPU device auto-detection with strict no-fallback

### Phase 3: Shared Component Optimization (Day 3)
1. Verify Diffusion block GPU path
2. Verify SSM/Mamba GPU path (with placeholder kernels)
3. Verify Transformer block end-to-end GPU path
4. Measure memory transfer overhead

## Expected Outcomes

| Metric | Target | Method |
|--------|--------|--------|
| Build Status | Pass | Fix compilation errors (Tier 1) |
| Test Coverage | 183+ tests | Verify tests run after build fix |
| GPU Device Auto-Detection | No CPU fallback | Strict error handling |
| Shared Component Trait Unification | 100% | All components impl GpuComponent |
| Memory Reuse | Zero-allocation after warmup | Verify pool implementation |
| Numerical Accuracy | ε ≤ 1e-4 | Compare against CPU reference |

## Branch Strategy

- Work on main branch
- Incremental commits after each fix category
- Test suite validation after Phase 1

## Next Actions

1. Start with PolyAttention ForwardContext fixes
2. Run `cargo check` to verify incremental progress
3. Address one component at a time (Attention -> Feedforward -> Temporal)
4. Test GPU device detection immediately after build passes
