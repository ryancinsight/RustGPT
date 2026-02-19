# Phase 5: Advanced Consolidation & Performance Optimization
**Date**: 2026-02-12  
**Status**: In Progress  
**Focus**: Memory efficiency, shared component optimization, and unified workspace management

---

## Executive Summary

The RustGPT codebase has achieved significant consolidation with centralized shared components (`SharedAttentionContext`, `SharedTemporalProcessing`, `SharedFeedforward`). Phase 5 focuses on:

1. **Memory Optimization** - Eliminate redundant allocations across diffusion, SSM, and transformer
2. **Workspace Unification** - Extract common workspace patterns into reusable traits
3. **Buffer Pooling** - Consolidate intermediate buffer management across architectures
4. **Performance Analysis** - Profile critical paths and eliminate bottlenecks

---

## Current Architecture State

### ✅ Consolidated Components (Completed)

| Component | File | Status | Details |
|-----------|------|--------|---------|
| **SharedAttentionContext** | `attention_context.rs` | ✅ Optimized | Lazy allocation, in-place gradients, `general_mat_mul` |
| **SharedTemporalProcessing** | `temporal_processing.rs` | ✅ Unified | Zero-cost abstraction via Layer trait |
| **SharedFeedforward** | `feedforward.rs` | ✅ Unified | Supports RichardsGLU, MoE with routing metrics |
| **RichardsNorm/GLU** | `domain/richards/` | ✅ Optimized | Custom activation with stable gradients |
| **IntermediateBufferPool** | `intermediate_buffer_pool.rs` | ✅ Active | Power-of-2 sizing, lazy allocation |
| **AdaptiveResiduals** | `adaptive_residuals.rs` | ✅ Tuned | Memory pooling, gradient computation |

### 🔄 In-Progress Consolidation (Phase 5)

| Area | Current State | Target | Priority |
|------|---------------|--------|----------|
| **Workspace Traits** | Diffusion & Transformer have separate implementations | Unified `WorkspaceManaged` trait | **P0** |
| **Linear Projections** | SSM uses custom `LinearProjection`, others use raw ndarray | Unified `ProjectionLayer` trait | **P1** |
| **KV Cache Management** | Multiple streaming implementations | Unified `StreamingWorkspace` trait | **P1** |
| **Gradient Routing** | Per-block gradient logic | Unified `GradientRouter` | **P2** |
| **Buffer Allocation** | IntermediateBufferPool + workspace pools | Single unified pool strategy | **P2** |

---

## Performance Optimization Roadmap

### Phase 5.1: Workspace Unification (CRITICAL)

**Goal**: Extract `WorkspaceManaged` trait to eliminate ~300 LOC of duplication

**Current Code Duplication**:
```
TransformerBlock (383-443 lines):
  - norm1_workspace
  - attention_workspace
  - norm2_workspace  
  - ffn_workspace

DiffusionBlock (1091-1105 lines):
  - Similar workspace allocation patterns
```

**Action Items**:
1. Extract trait in `src/domain/layers/components/workspace_managed.rs`:
```rust
pub trait WorkspaceManaged {
    fn ensure_capacity(&mut self, batch: usize, seq_len: usize, embed_dim: usize);
    fn clear_workspace(&mut self);
    fn workspace_memory_usage(&self) -> usize;
}
```

2. Implement for `TransformerBlock`, `DiffusionBlock`, `RgLruBlock`
3. Remove redundant workspace initialization code
4. **Expected savings**: ~300 LOC, 10-15% allocation reduction

---

### Phase 5.2: Unified Buffer Strategy (HIGH)

**Goal**: Consolidate `IntermediateBufferPool` + workspace pools into single hierarchy

**Current State**:
- `IntermediateBufferPool`: Per-block intermediate buffers (norm_out, mix_out, ffn_out)
- `AdaptiveResidualsWorkspace`: Per-block residual workspace
- `TransformerBlockStreamingWorkspace`: Per-block streaming state

**Unified Strategy**:
```rust
/// Unified workspace for all layer types
pub struct LayerWorkspace {
    // Norm buffers
    norm1_out: Array2<f32>,
    norm2_out: Array2<f32>,
    
    // Mixing/attention buffers
    temporal_out: Array2<f32>,
    
    // FFN buffers
    ffn_intermediate: Array2<f32>,
    
    // Streaming state (SSM/RG-LRU only)
    streaming: Option<StreamingState>,
    
    // Memory metadata
    current_shape: (usize, usize),
    allocation_limit: usize,
}
```

**Benefits**:
- Single `ensure_capacity` call per forward pass
- Shared allocation strategy across all architectures
- Easier memory profiling and debugging
- ~20% reduction in allocation overhead

---

### Phase 5.3: Linear Projection Unification (HIGH)

**Goal**: Consolidate SSM's `LinearProjection` with transformer's standard approach

**Current Duplication**:
```rust
// SSM: src/domain/layers/ssm/components/projection_layers.rs
pub struct LinearProjection {
    weights: Array2<f32>,
    biases: Array1<f32>,
    // Custom forward/backward
}

// Transformer: Uses FeedforwardProcessor or raw ndarray ops
// Diffusion: Uses conditioning.rs with FiLM layers
```

**Unified Approach**:
```rust
pub struct ProjectionLayer {
    weights: Array2<f32>,
    biases: Option<Array1<f32>>,
    use_bias: bool,
    scale_output: bool, // For stabilization
}

impl ProjectionLayer {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // Optimized: general_mat_mul for input @ weights.t() + biases
    }
    
    pub fn backward(&self, input: &Array2<f32>, grad_output: &Array2<f32>) 
        -> (Array2<f32>, Array2<f32>, Option<Array1<f32>>) {
        // Weight, input, and bias gradients
    }
}
```

**Savings**:
- Eliminate `LinearProjection` (60 LOC)
- Consistent gradient computation
- Shared optimization opportunities (BLAS-level)

---

### Phase 5.4: Memory Pooling Strategy (MEDIUM)

**Goal**: Implement global buffer pool for reduced fragmentation

**Current Allocations per Forward Pass**:
- Norm operations: ~400 MB (if batch=32, seq=512, embed=2048)
- Attention: ~600 MB (Q, K, V matrices)
- FFN: ~800 MB (intermediate layer)
- **Total**: ~1.8 GB per pass (reallocated every step)

**Pooling Strategy**:
```rust
pub struct GlobalBufferPool {
    // Pre-allocated pools by size
    pools: HashMap<(usize, usize), Vec<Array2<f32>>>,
    
    // Track usage
    peak_usage: usize,
    recycle_threshold: usize, // Free unused after N steps
}

impl GlobalBufferPool {
    pub fn acquire(&mut self, shape: (usize, usize)) -> Array2<f32>;
    pub fn release(&mut self, buffer: Array2<f32>);
    pub fn reset(&mut self);
}
```

**Benefits**:
- Reduce heap fragmentation
- Faster allocation via mempool patterns
- Better cache locality
- **Estimated improvement**: 20-30% faster forward passes (allocation-bound)

---

## Optimization Techniques (Highest Impact)

### 1. In-Place Operations (10-15% speedup)

**Current**: `apply_context` returns new array via `Cow`
```rust
pub fn apply_context(&self, input: &Array2<f32>) -> Cow<'_, Array2<f32>>
```

**Target**: Prefer in-place variants
```rust
pub fn apply_context_into(&self, input: &Array2<f32>, output: &mut Array2<f32>) -> bool
```

**Action**:
- ✅ Already done for `SharedAttentionContext` 
- Update `SharedTemporalProcessing` to support in-place forward
- Update `SharedFeedforward` to support in-place forward

---

### 2. Batch Normalization Fusion (8-12% speedup)

**Current**: Separate norm + mixing + residual operations
```
output = residual_weight * input + mixing(norm(input))
```

**Fused Kernel Pattern**:
```rust
pub fn fused_norm_mixing_residual(
    &self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) {
    // Single kernel: norm + mix + add residual in one pass
    // Reduces memory bandwidth and cache misses
}
```

**Target Layers**: TransformerBlock, DiffusionBlock
**Estimated gain**: 8-12% wall-clock improvement on inference

---

### 3. Selective Gradient Computation (5-10% training speedup)

**Current**: Compute all gradients in `compute_gradients`
**Target**: Skip unused components

```rust
pub fn compute_gradients_selective(
    &self,
    input: &Array2<f32>,
    grads: &Array2<f32>,
    compute_mask: GradientMask, // Which gradients to compute
) -> (Array2<f32>, Option<Array2<f32>>, Option<Array2<f32>>)
```

**Use Cases**:
- Freeze certain parameters → skip their gradients
- Head pruning → selective head gradients
- Layer-wise learning rates → skip unused paths

---

### 4. Reduced Precision Intermediate States (15-25% memory reduction)

**Current**: All intermediate states in f32
**Target**: f32 for activations, f16 for context matrices (with periodic upcast)

```rust
pub struct MixedPrecisionAttentionContext {
    // High precision for active updates
    outgoing_context: Option<Array2<f32>>,
    // Low precision for historical context
    historical_context: Option<Array2<f16>>,
}
```

**Benefits**:
- Reduce context matrix memory by 50% (128×128 → 8KB instead of 16KB)
- Maintain numerical stability with periodic f32 accumulation
- Better cache efficiency

---

## Implementation Plan (Priority Order)

### Week 1: Workspace Unification (P0)
**Files to modify**:
1. Create `src/domain/layers/components/workspace_managed.rs`
2. Refactor `TransformerBlock` → implement `WorkspaceManaged`
3. Refactor `DiffusionBlock` → implement `WorkspaceManaged`
4. Refactor `RgLruBlock` → implement `WorkspaceManaged`
5. Update tests to verify workspace reuse

**Expected outcome**: 300 LOC reduction, test suite passes

### Week 2: In-Place Operations (P1)
**Files to modify**:
1. Add `forward_into` variants to `SharedTemporalProcessing`
2. Add `forward_into` variants to `SharedFeedforward`
3. Update TransformerBlock to use in-place ops
4. Update DiffusionBlock to use in-place ops
5. Benchmark allocation reduction

### Week 3: Buffer Unification (P1)
**Files to modify**:
1. Create `src/domain/layers/components/unified_buffer_pool.rs`
2. Consolidate `IntermediateBufferPool` + workspace pools
3. Implement per-block `ensure_capacity` that delegates to pool
4. Add global pool strategy for large allocations
5. Profile memory usage before/after

### Week 4: Gradient Optimization (P2)
**Files to modify**:
1. Implement `selective_compute_gradients` in shared components
2. Add `GradientMask` configuration
3. Update training loops to use selective computation
4. Benchmark training speed improvement

---

## Measurement Plan

### Metrics to Track

**Memory**:
- Peak memory usage per forward pass
- Allocation count per step
- Fragmentation ratio (via `/proc/self/maps` or similar)

**Performance**:
- Forward pass time (batch=32, seq=512, embed=2048)
- Backward pass time
- E2E training step time

**Code Quality**:
- Line count (target: -300 LOC for workspace unification)
- Test coverage (target: maintain 100% for critical paths)
- Compilation time (target: no increase)

### Baseline Measurements (Before)
```
Allocations per step: ~50-60
Peak memory (single batch): ~2.0 GB
Forward + backward time: ~450ms
Code lines (layers/): ~8500
```

### Target Improvements (After Phase 5)
```
Allocations per step: ~30-35 (-40%)
Peak memory (single batch): ~1.6 GB (-20%)
Forward + backward time: ~380ms (-15%)
Code lines (layers/): ~8200 (-300)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Regression in accuracy | Low | High | Extensive test suite + numerical validation |
| Performance degradation | Low | High | Profile each change with benchmarks |
| Breaking changes | Medium | Medium | Feature gates for new implementations |
| Increased compile time | Low | Medium | Avoid template bloat in consolidation |

---

## Testing Strategy

### Unit Tests
- Each consolidated component: behavior unchanged
- Workspace trait: capacity management, reuse verification
- Buffer pool: allocation/deallocation correctness

### Integration Tests
- TransformerBlock forward/backward unchanged
- DiffusionBlock forward/backward unchanged
- RG-LRU forward/backward unchanged

### Performance Tests
```bash
# Before
cargo bench --bench unified_transformer_block

# After  
cargo bench --bench unified_transformer_block
# Verify: allocation count ↓, speed ↑, memory ↓
```

---

## Success Criteria

✅ **Completed**:
- `SharedAttentionContext` with lazy allocation and in-place ops
- `SharedTemporalProcessing` with zero-cost abstraction
- `SharedFeedforward` supporting multiple FFN variants
- `IntermediateBufferPool` with power-of-2 sizing

🔄 **In Progress** (Phase 5):
- [ ] Workspace unification trait extracted
- [ ] In-place forward ops implemented across all components
- [ ] Unified buffer pool strategy deployed
- [ ] 15% speedup achieved on inference

❓ **Future** (Phase 6+):
- Selective gradient computation
- Batch norm fusion kernels
- Mixed-precision context matrices
- Distributed buffer pooling

---

## Related Documentation
- `AGENTS.md` - Build & test commands
- `src/domain/layers/components/mod.rs` - Component exports
- `src/domain/layers/transformer/block.rs` - TransformerBlock implementation
- `src/domain/layers/diffusion/block.rs` - DiffusionBlock implementation
