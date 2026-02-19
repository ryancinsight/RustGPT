# Consolidation & Optimization: Phase 5 Continuation (Feb 13, 2026)

**Status**: Active consolidation with targeted memory & performance optimizations  
**Test Suite**: 476 tests passing (0 failures)  
**Architecture**: Unified shared components across Diffusion, Transformer, SSM

---

## Current State Assessment

### ✅ Completed (Phase 4 & Earlier)
1. **SharedAttentionContext** - Lazy allocation, zero-copy matrix ops
2. **UnifiedLayerWorkspace** - Consolidated buffer management
3. **WorkspaceManaged Trait** - Unified interface for all blocks
4. **SharedFilmModulation** - Power-of-2 scratch buffer sizing
5. **AdaptiveResiduals** - Memory pooling with head activity
6. **Shared components**: Common, BlockCore, TemporalProcessing, Feedforward

### 📊 Current Memory Profile
| Component | Memory/Step | Optimization |
|-----------|------------|--------------|
| Core Buffers (12 layers) | 60 KB | Power-of-2 sizing ✓ |
| Streaming State | 8 KB | Lazy allocation ✓ |
| Context Matrices | 12 KB | On-demand ✓ |
| FiLM Parameters | 24 KB | Cached capacity ✓ |
| **Total (Optimized)** | **129 KB** | **83% reduction** |

---

## Priority Optimization Opportunities

### 1. **In-Place Forward Operations** (Medium Priority, High Impact)
**Target**: SharedTemporalProcessing, SharedFeedforward  
**Goal**: Eliminate intermediate allocations in forward pass

#### Current Pattern (Allocates)
```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // Multiple allocations inside temporal mixing
    let attention_out = self.attention.forward(&input);
    let mixed = self.apply_mixing(&attention_out);
    mixed
}
```

#### Optimized Pattern (Zero-Alloc)
```rust
pub fn forward_into(
    &mut self,
    input: &Array2<f32>,
    output: &mut Array2<f32>,
) -> Result<()> {
    // Reuse provided output buffer, no intermediate allocations
    self.attention.forward_into(input, output)?;
    self.apply_mixing_inplace(output);
    Ok(())
}
```

**Expected Impact**:
- 10-15% speedup on inference
- 40 KB/step memory reduction per layer
- Better cache locality

**Implementation Plan**:
1. Add `forward_into()` variants to `PolyAttention`, `Mamba`, `RingAttention`, `SlidingWindowAttention`
2. Update `SharedTemporalProcessing` to expose `forward_into()`
3. Add `forward_into()` variants to `RichardsGLU`, `MixtureOfExperts`
4. Integrate into TransformerBlock and DiffusionBlock forward paths

---

### 2. **Global Buffer Pool Consolidation** (Medium Priority, Medium Impact)
**Target**: Cross-component buffer reuse  
**Goal**: Reduce fragmentation, improve cache hit rate

#### Current State
- Each block has its own UnifiedLayerWorkspace
- No reuse between blocks at different stages
- Power-of-2 sizing adds 20-30% padding

#### Proposed GlobalBufferPool
```rust
pub struct GlobalBufferPool {
    // Size-class -> Vec<Buffer>
    pools: HashMap<usize, Vec<Array2<f32>>>,
    // Track allocations per size class
    stats: HashMap<usize, (u32, usize)>, // (count, total_bytes)
}

impl GlobalBufferPool {
    /// Acquire a buffer of at least `size` elements
    pub fn acquire(&mut self, size: usize) -> Array2<f32> { }
    
    /// Return buffer to pool
    pub fn release(&mut self, buffer: Array2<f32>) { }
    
    /// Statistics for memory profiling
    pub fn stats(&self) -> PoolStats { }
}
```

**Expected Impact**:
- 15-20% reduction in heap fragmentation
- Better NUMA locality
- Easier memory tracking

**Implementation Plan**:
1. Create `src/domain/layers/components/global_buffer_pool.rs`
2. Add size-class calculation (power-of-2 buckets)
3. Implement acquire/release with stats tracking
4. Integrate into LLMModel initialization
5. Add pool reset at epoch boundaries

---

### 3. **Selective Gradient Computation** (Low Priority, High Impact)
**Target**: Frozen or low-activity layers  
**Goal**: Skip gradient computation in non-trainable sections

#### Pattern
```rust
pub struct GradientComputeMask {
    layer_masks: Vec<bool>, // false = skip gradients
}

impl SharedBlockCore {
    pub fn backward_with_mask(
        &self,
        gradients: &Array2<f32>,
        compute_mask: Option<&GradientComputeMask>,
    ) -> BackwardResult { }
}
```

**Expected Impact**:
- 20-30% backward pass speedup (if 50% layers frozen)
- Zero overhead when all layers trainable

---

### 4. **Batch Norm / Normalization Fusion** (Low Priority, Low Impact)
**Target**: RichardsNorm layer stack  
**Goal**: Fuse normalization + mixing + residual

#### Current (3 kernels)
```
norm1 -> temporal_mixing -> residual_add -> norm2
```

#### Fused (1 kernel)
```
fused_norm_mixing_residual -> fused_norm2
```

**Expected Impact**:
- 5% speedup
- 8 KB/step memory reduction

---

### 5. **Mixed Precision Cache** (Medium Priority, Low Impact)
**Target**: Historical attention matrices  
**Goal**: Store KV-cache in f16, compute in f32

#### Pattern
```rust
pub struct MixedPrecisionCache {
    k_cache_f16: Array2<f16>, // Storage
    v_cache_f16: Array2<f16>,
    // Promoted to f32 during forward
}
```

**Expected Impact**:
- 50% memory reduction for KV-cache
- 2-3% compute overhead
- Good for long context sequences

---

## Consolidation Milestones

### Phase 5.1: In-Place Operations (Week 1)
- [ ] Add `forward_into()` to all temporal mixing layers
- [ ] Implement `forward_into()` in SharedFeedforward
- [ ] Update block-level forward calls
- [ ] Benchmark and validate 10-15% improvement
- [ ] Test compatibility with existing code

### Phase 5.2: Global Buffer Pool (Week 2)
- [ ] Create GlobalBufferPool component
- [ ] Integrate with LLMModel
- [ ] Add pool statistics tracking
- [ ] Validate fragmentation reduction
- [ ] Add reset/cleanup mechanisms

### Phase 5.3: Advanced Optimizations (Week 3)
- [ ] Implement GradientComputeMask (optional)
- [ ] Explore batch norm fusion (profiling first)
- [ ] Consider mixed precision for long sequences
- [ ] Profile and identify remaining bottlenecks

### Phase 5.4: Validation & Documentation (Week 4)
- [ ] Full integration tests
- [ ] Performance benchmarks
- [ ] Memory profiling reports
- [ ] Update architecture documentation

---

## Code Quality Metrics

### Current Status
- **Tests**: 476 passing, 0 failures, 0 ignored
- **Warnings**: 4 (unused variables, dead code in tests)
- **Code Coverage**: ~85% (integration tests)
- **Performance**: Baseline established

### Phase 5.1 Target Metrics
- **Tests**: 490+ passing (add in-place operation tests)
- **Warnings**: < 3
- **Performance**: 10-15% inference speedup
- **Memory**: 40 KB/step reduction

---

## Technical Design Notes

### In-Place Operations Design
```rust
// Trait for layers supporting in-place forward
pub trait ForwardIntoLayer {
    fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()>;
}

// Used by unified blocks
pub fn forward_with_workspace(
    &mut self,
    input: &Array2<f32>,
    workspace: &mut UnifiedLayerWorkspace,
) {
    // 1. Norm + Temporal (in-place into norm1_out)
    self.pre_norm.forward_into(input, workspace.norm1_out_mut().unwrap());
    self.temporal_mixing.forward_into(
        workspace.norm1_out().unwrap(),
        workspace.temporal_out_mut().unwrap(),
    );
    
    // 2. Residual + norm2 (in-place)
    add_residual_inplace(workspace.norm1_out(), workspace.temporal_out_mut());
    
    // 3. FFN (in-place)
    self.pre_ffn_norm.forward_into(workspace.temporal_out(), workspace.norm2_out_mut().unwrap());
    self.feedforward.forward_into(workspace.norm2_out(), workspace.ffn_out_mut().unwrap());
    
    // Final: copy to output
    output.assign(workspace.ffn_out().unwrap());
}
```

### GlobalBufferPool Integration
```rust
// In LLMModel
pub struct LLMModel {
    layers: Vec<TransformerBlock>,
    buffer_pool: Option<Arc<Mutex<GlobalBufferPool>>>,
}

// Blocks can access shared pool
impl TransformerBlock {
    pub fn forward_with_pool(
        &mut self,
        input: &Array2<f32>,
        pool: Option<&Arc<Mutex<GlobalBufferPool>>>,
    ) -> Array2<f32> { }
}
```

---

## Risk Assessment

### Low Risk
- In-place operations (backward compatible via Cow return type)
- Global buffer pool (optional, can be disabled)

### Medium Risk
- Gradient computation mask (requires testing all freeze scenarios)
- Batch norm fusion (needs extensive validation)

### Mitigation
- Maintain both old and new code paths during transition
- Extensive integration testing before merge
- Benchmark suite with regression detection

---

## Success Criteria

### Phase 5.1 Complete
- ✓ All in-place operations implemented
- ✓ 10-15% measured speedup
- ✓ All 476 tests still passing
- ✓ No performance regressions
- ✓ Memory reduction validated

### Overall Phase 5 Complete
- ✓ 25-30% total speedup
- ✓ 50% memory reduction from baseline
- ✓ 500+ integration tests
- ✓ Production-ready code
- ✓ Comprehensive documentation

---

## References

- **Baseline Architecture**: CONSOLIDATION_COMPONENTS_MANIFEST.md
- **Previous Session**: T-019c5596-85e3-710a-a671-a28d6bf3db20
- **Build System**: AGENTS.md
- **Performance Guide**: OPTIMIZATION_PATTERNS_GUIDE.md

---

## Next Actions

1. ✅ Review current component state ← **DONE (this assessment)**
2. ⏭️ Start Phase 5.1 - In-Place Operations
   - Analyze SharedTemporalProcessing for allocation hotspots
   - Design forward_into() interface
   - Implement for PolyAttention first
3. Create detailed implementation PRs
4. Add comprehensive benchmarks
5. Profile memory and CPU efficiency
