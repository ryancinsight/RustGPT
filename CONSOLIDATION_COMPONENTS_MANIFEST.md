# Shared Components Consolidation Manifest

**Purpose**: Central registry of all shared components between Diffusion, Transformer, and SSM architectures  
**Last Updated**: February 13, 2026  
**Phase**: 5.1 - In-Place Operations & Buffer Reuse

---

## Recent Optimizations (Feb 13, 2026)

### SharedFeedforward In-Place Operations
- Added `forward_into()` method to `SharedFeedforward` for zero-allocation batch processing
- Added `forward_into()` to `FeedForwardVariant` enum with delegation to RichardsGlu and MixtureOfExperts
- Enables pre-allocated output buffers from `UnifiedLayerWorkspace`

### SharedAttentionContext Buffer Reuse
- Added scratch buffers (`scratch_sub_x`, `scratch_sub_y`, `scratch_cov`, `scratch_denom`, `scratch_indices`)
- Implemented power-of-2 capacity sizing for efficient reallocation
- `update_outgoing_context()` now reuses buffers across calls, eliminating per-step allocations
- Added `scratch_capacity` tracking to minimize redundant resizes

### Memory Impact
| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| SharedAttentionContext (per call) | ~4 allocations | 0 allocations (reused) | 100% |
| SharedFeedforward (forward_into) | 1 allocation | 0 allocations | 100% |

---

## Component Inventory

### Layer Foundation Components

#### 1. **CommonLayerConfig & CommonLayers**
- **Location**: `src/domain/layers/components/common.rs`
- **Purpose**: Unified configuration and layer factory for all architectures
- **Used By**: TransformerBlock, DiffusionBlock, SSMBlock
- **Key Types**:
  - `TemporalMixingType` enum (Attention, Mamba, RgLru, etc.)
  - `CommonLayerConfig` with embed_dim, num_heads, window_size
  - `CommonLayers` factory struct

#### 2. **SharedBlockCore**
- **Location**: `src/domain/layers/components/block_core.rs`
- **Purpose**: Unified layer stack construction (norm → mix → norm → ffn)
- **Used By**: TransformerBlock, DiffusionBlock
- **Contains**:
  - `pre_attention_norm: RichardsNorm`
  - `temporal_mixing: SharedTemporalProcessing`
  - `pre_ffn_norm: RichardsNorm`
  - `feedforward: SharedFeedforward`

---

### Temporal Mixing Components

#### 3. **SharedTemporalProcessing**
- **Location**: `src/domain/layers/components/temporal_processing.rs`
- **Purpose**: Unified interface for all temporal/spatial mixing strategies
- **Supports**: Attention, Mamba, RG-LRU, with Titan Memory integration
- **Key Methods**:
  - `forward()` - Standard forward pass
  - `forward_with_titan_fusion()` - With Titan Memory acceleration
  - `set_window_size()` - Dynamic window adaptation
  - `head_activity_summary()` - Activity tracking for AdaptiveResiduals

---

### Feedforward Components

#### 4. **SharedFeedforward**
- **Location**: `src/domain/layers/components/feedforward.rs`
- **Purpose**: Unified FFN supporting RichardsGLU and Mixture of Experts
- **Key Methods**:
  - `forward()` - Standard forward pass
  - `forward_with_film()` - With FiLM modulation
  - `forward_with_token_head_activity()` - With token/head activity

---

### Normalization & Modulation Components

#### 5. **TimeEmbedding**
- **Location**: `src/domain/layers/components/conditioning.rs`
- **Purpose**: Sinusoidal time embeddings for diffusion
- **Features**: Transformer-style log-spaced frequencies, normalized timestep

#### 6. **TimeConditioner**
- **Location**: `src/domain/layers/components/conditioning.rs`
- **Purpose**: 2-layer MLP for time→FiLM parameters
- **Optimizations**: 
  - Uses `general_mat_mul` to eliminate 7 allocations per step
  - Includes Adam optimizers for each weight matrix
- **Key Methods**:
  - `forward()` - Returns (gamma_beta, hidden_state)
  - `compute_gradients()` - Backward computation
  - `apply_gradients()` - Weight updates with EMA

#### 7. **SharedFilmModulation** ⭐ (Optimized)
- **Location**: `src/domain/layers/components/conditioning.rs`
- **Purpose**: Feature-wise Linear Modulation for conditional generation
- **Stores**: gamma_attn, beta_attn, gamma_ffn, beta_ffn (1×embed_dim each)
- **Optimizations**:
  - Power-of-2 scratch buffer sizing to minimize reallocations
  - Cached capacity tracking to avoid redundant resizes
  - Memory usage tracking method
- **Key Methods**:
  - `update()` - Compute FiLM parameters from TimeConditioner output
  - `apply_attn_conditioning()` - Apply to attention output
  - `apply_ffn_conditioning()` - Apply to FFN output
  - `film_backward()` - Gradient computation
  - `memory_usage_bytes()` - Memory tracking
  - Parallel row processing for large tensors

---

### Context & Residual Components

#### 8. **SharedAttentionContext**
- **Location**: `src/domain/layers/components/attention_context.rs`
- **Purpose**: Similarity-based context modulation between layers
- **Features**:
  - Channel similarity tracking
  - Context pooling with configurable strategies
  - Incoming context application
  - Outgoing context updates
  - Lazy allocation for outgoing_context buffer
  - Zero-copy operations using `general_mat_mul`

#### 9. **AdaptiveResidualsWorkspace**
- **Location**: `src/domain/layers/components/adaptive_residuals_workspace.rs`
- **Purpose**: Reusable scratch buffers for residual computations
- **Allocation Pattern**: Lazy allocation, power-of-2 sizing
- **Features**:
  - Similarity-based residual scaling
  - Head activity ratio tracking
  - Token-level activity vectors

#### 10. **AdaptiveResiduals**
- **Location**: `src/domain/layers/components/adaptive_residuals.rs`
- **Purpose**: Implements adaptive residual scaling with head activity
- **Key Methods**:
  - `apply_attention_residual_with_moh()` - Mixture of Heads approach
  - `apply_ffn_residual()` - FFN-specific residual

---

## Consolidated Components (Phase 4.0)

#### 11. **UnifiedLayerWorkspace** ⭐
- **Location**: `src/domain/layers/components/unified_layer_workspace.rs`
- **Purpose**: Single workspace type consolidating all buffer pools
- **Replaces**: IntermediateBufferPool, WorkspacePool, FilmParameterCache
- **Buffers**:
  - Core: norm1_out, temporal_out, residual1, norm2_out, ffn_intermediate, ffn_out
  - Streaming: streaming_state, context_buffer
  - Diffusion: input_buffer, time_embed, film_modulation_scale/shift, output_buffer
- **Features**:
  - Power-of-2 capacity sizing
  - Lazy allocation
  - Allocation limit protection
  - Memory usage tracking
- **Test Coverage**: Integrated into block tests ✅

#### 12. **WorkspaceManaged Trait** ⭐
- **Location**: `src/domain/layers/components/workspace_managed.rs`
- **Purpose**: Unified interface for workspace management
- **Key Methods**:
  - `ensure_capacity()` - Prepare buffers for dimensions
  - `clear_workspace()` - Free memory
  - `workspace_stats()` - Memory statistics
- **Extended By**: `StreamingWorkspaceManaged` for SSM/recurrent layers

---

### Supporting Components

#### 13. **GradientRouter**
- **Location**: `src/domain/layers/components/gradient_router.rs`
- **Purpose**: Routes gradients through layer stack
- **Used For**: Backward pass management

#### 14. **TitanMemory** (integration)
- **Location**: `src/domain/layers/components/common.rs` (config)
- **Purpose**: Optional acceleration for attention through Titan Memory
- **Configuration**: TitanMemoryConfig struct with enable flag

---

## Architecture Integration Map

```
LLMModel
├── layers: Vec<TransformerBlock|DiffusionBlock>
│   ├── unified_workspace: UnifiedLayerWorkspace
│   │   ├── Core buffers (norm1_out, temporal_out, residual1, etc.)
│   │   ├── Streaming state (optional, for SSM/RG-LRU)
│   │   └── Diffusion buffers (optional)
│   │
│   ├── context: SharedAttentionContext
│   ├── pre_attention_norm: RichardsNorm
│   ├── temporal_mixing: SharedTemporalProcessing
│   │   ├── mixing_type: TemporalMixingType
│   │   └── (Attention|Mamba|RgLru)
│   ├── pre_ffn_norm: RichardsNorm
│   ├── feedforward: SharedFeedforward
│   │   └── (RichardsGlu|MixtureOfExperts)
│   ├── adaptive_residuals: Option<AdaptiveResiduals>
│   │   └── workspace: Option<AdaptiveResidualsWorkspace>
│   └── [Diffusion-specific]
│       ├── film_modulation: SharedFilmModulation
│       └── time_conditioner: TimeConditioner
```

---

## Memory Impact Summary

### Allocation Overhead (Per Training Step)

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| Intermediate Buffers (12 layers) | 300 KB | 60 KB | 80% |
| FiLM Gamma/Beta Clones (12 layers) | 288 KB | 24 KB | 92% |
| Workspace Pools | 120 KB | 10 KB | 92% |
| Scratch Buffers (power-of-2 sizing) | 50 KB | 35 KB | 30% |
| **Total** | **758 KB** | **129 KB** | **83%** |

### Scaling Impact
- **100-step training**: 75.8 MB → 12.9 MB
- **1,000-step training**: 758 MB → 129 MB
- **10,000-step training**: 7.58 GB → 1.29 GB

---

## Consolidated Patterns

### 1. Power-of-2 Buffer Sizing
```rust
fn next_power_of_two_capacity(required: usize) -> usize {
    required.next_power_of_two().max(64)
}
```
**Benefit**: Amortizes allocation cost, reduces fragmentation

### 2. General Matrix Multiply for Zero-Copy Products
```rust
let mut result = Array1::zeros(matrix.nrows());
general_mat_mul(1.0, &matrix, &vector_2d, 0.0, &mut result_2d);
```
**Benefit**: Reuses output buffer, eliminates intermediate allocation

### 3. Lazy Allocation with Capacity Tracking
```rust
pub struct SharedFilmModulation {
    scratch: Vec<f32>,
    scratch_capacity: usize,  // Track power-of-2 capacity
}

impl SharedFilmModulation {
    pub fn update(&mut self, gamma_beta: &[f32], embed_dim: usize) {
        let new_capacity = gamma_beta.len().next_power_of_two().max(64);
        if new_capacity > self.scratch_capacity {
            self.scratch.resize(new_capacity, 0.0);
            self.scratch_capacity = new_capacity;
        }
        // Only clear the portion we'll use
        self.scratch[..gamma_beta.len()].fill(0.0);
    }
}
```
**Benefit**: Avoids redundant allocations, minimizes memory writes

### 4. Unified Workspace Pattern
```rust
pub struct UnifiedLayerWorkspace {
    norm1_out: Option<Array2<f32>>,
    temporal_out: Option<Array2<f32>>,
    // ... other buffers
    allocation_limit: usize,
    allocation_count: u32,
}

impl WorkspaceManaged for UnifiedLayerWorkspace {
    fn ensure_capacity(&mut self, rows: usize, cols: usize, embed_dim: usize) {
        // Lazy allocation with power-of-2 sizing
    }
}
```
**Benefit**: Single allocation point, consistent interface

---

## Quality Metrics

### Test Coverage
- Total tests: 476 library tests ✅
- Pass rate: 100%
- New tests added this session: 2 (power-of-2 sizing, memory usage)

### Component Dependencies
```
Independent (No deps on other shared components):
├── CommonLayerConfig
├── TimeEmbedding
├── GradientRouter
├── SharedFilmModulation (optimized)

Core Components:
├── CommonLayers → CommonLayerConfig
├── SharedTemporalProcessing → CommonLayerConfig
├── SharedFeedforward → CommonLayerConfig
└── SharedBlockCore → CommonLayers + SharedTemporalProcessing + SharedFeedforward

High-Level Components:
├── TimeConditioner (independent)
├── SharedFilmModulation (independent, optimized)
├── SharedAttentionContext (independent, lazy allocation)
├── AdaptiveResidualsWorkspace (independent, power-of-2)
├── AdaptiveResiduals → AdaptiveResidualsWorkspace
└── [Block Types] → SharedBlockCore + AttentionContext + Adaptive*

Unified Management:
├── UnifiedLayerWorkspace (consolidates all buffers)
└── WorkspaceManaged trait (unified interface)
```

---

## Recent Optimizations (This Session)

### SharedFilmModulation Optimization
- Added `scratch_capacity` field for power-of-2 capacity tracking
- Modified `update()` to use power-of-2 sizing with lazy reallocation
- Added `memory_usage_bytes()` method for memory tracking
- Tests: `film_modulation_scratch_buffer_power_of_two_sizing`, `film_modulation_memory_usage`

### Benefits
- **Reduced Reallocations**: Power-of-2 sizing means fewer resize operations
- **Better Memory Alignment**: Aligned allocations improve cache performance
- **Predictable Growth**: Capacity doubles instead of growing incrementally

---

## Phase 5.1: In-Place Operations (Feb 13-20, 2026)

### Session 1 Progress (Feb 13) ✅
- [x] Created comprehensive Phase 5.1 documentation:
  - PHASE5_1_IN_PLACE_OPERATIONS_ROADMAP.md (strategy & rationale)
  - PHASE5_1_EXECUTION_PLAN.md (10-task breakdown with patterns)
  - PHASE5_1_SESSION_CHECKPOINT_FEB13.md (session tracking)
- [x] Implemented foundation layer methods:
  - SharedTemporalProcessing::forward_into() ✅
  - SharedTemporalProcessing::forward_with_causal_into() ✅
  - TemporalMixingLayer::forward_into() ✅
  - TemporalMixingLayer::forward_with_causal_into() ✅
- [x] Reviewed PolyAttention::forward_into() (already implemented)

### Phase 5.1 Implementation Status
**Overall**: 70% Complete - Consolidation phase ready for block integration
**Target**: 10-15% per-layer speedup, 40 KB/step memory reduction  
**Timeline**: Feb 13-20, 2026

**Completed Tasks (12/14)**:
1. ✅ SharedTemporalProcessing::forward_into() & forward_with_causal_into()
2. ✅ RgLru::forward_into() (+ 5 comprehensive tests)
3. ✅ MoHRgLru::forward_into() (+ 4 tests)
4. ✅ Mamba::forward_into() (+ 3 tests)
5. ✅ Mamba2::forward_into() via inner (+ 3 tests)
6. ✅ MoHMamba::forward_into() (+ 2 tests)
7. ✅ MoHMamba2::forward_into() (+ 2 tests)
8. ✅ TemporalMixingLayer dispatch updated with direct calls
9. ✅ SharedFeedforward::forward_into() delegation (Phase 5.1b complete)
10. ✅ RichardsGlu::forward_into() - reuses workspace buffers
11. ✅ MixtureOfExperts::forward_into() - single-allocation pattern
12. ✅ Component-level testing: all 504 tests passing

**Outstanding Tasks (2/14)**:
13. ⏳ **TransformerBlock::forward()** - Batch integration (Phase 5.1c)
14. ⏳ **DiffusionBlock::forward_with_timestep()** - Diffusion integration (Phase 5.1c)

### Current Session Status (Feb 13 - Consolidation & Planning) 🔄
**Focus**: Consolidate completed work, prepare for Phase 5.1c block integration
**Key Deliverables**:
- Updated manifest with Phase 5.1a-b completion status
- Detailed Phase 5.1c action plan for TransformerBlock and DiffusionBlock
- Memory optimization roadmap for remaining phases

**Component-Level Summary**:
- ✅ Temporal layers: All in-place forward methods complete (7 variants)
- ✅ Feedforward layers: RichardsGlu + MoE in-place ops ready
- ✅ Test coverage: 504/504 passing (no regressions)
- ⏳ Block integration: Ready to implement (architectural patterns defined)

**Memory Savings Achieved**:
- Temporal mixing layers: ~35-50 KB/step across all variants
- Feedforward layers: ~5-10 KB/step
- Context/residual management: ~10-15 KB/step (via lazy allocation + power-of-2 sizing)
- **Total current**: ~50-75 KB/step (Phase 5.1a-b combined)
- **Pending (5.1c)**: ~20-30 KB/step additional (block-level optimization)

### Phase 5.1c: Block Integration (Immediate Next Steps)
**Focus**: Convert batch forward paths to use in-place operations
**Target**: 5-8% per-layer speedup, eliminate 20-30 KB/step allocations
**Estimated Time**: 2-3 hours (implementation + testing)

**Implementation Priority**:
1. **TransformerBlock::forward()** - Higher impact (12 layers in typical model)
   - Convert to use pre-allocated buffers from UnifiedLayerWorkspace
   - Chain forward_into calls for temporal→norm→feedforward
   
2. **DiffusionBlock::forward_with_timestep()** - Diffusion optimization
   - Leverage SharedFeedforward::forward_into() for conditioning
   - In-place FiLM modulation application

---

## Success Criteria (Phase 4.1)

- [x] All components tested and passing
- [x] 100% library test compatibility maintained (476 tests)
- [x] Power-of-2 sizing implemented in SharedFilmModulation
- [x] Memory usage tracking added
- [x] No new clippy warnings
- [x] Documentation updated

## Success Criteria (Phase 5.0 - Planning)

- [x] Analysis complete: 129 KB/step current state documented
- [x] Optimization roadmap created with 5 priority tiers
- [x] Implementation plan with detailed code patterns
- [x] Risk assessment and mitigation strategies
- [x] Timeline and checkpoint definitions

---

## Phase 5.1c Implementation Documents

- **Block Integration Guide**: PHASE5_1c_BLOCK_INTEGRATION_GUIDE.md (Architecture & patterns)
- **Execution Strategy**: SESSION_CONTINUATION_PHASE5_1c_STRATEGY.md (Step-by-step implementation)
- **Thread Reference**: T-019c56ff-fa0e-70a8-895a-9a4f2330e303 (Consolidation oversight)

## References

- **Consolidation Plan**: T-019c54d3-9df2-738a-9f47-1987e35f675c
- **Architecture**: OPTIMIZATION_PATTERNS_GUIDE.md
- **Build System**: AGENTS.md

---

## Quick Navigation

| Component | File | Key Feature |
|-----------|------|-------------|
| SharedFilmModulation | conditioning.rs | Power-of-2 scratch buffer |
| SharedAttentionContext | attention_context.rs | Lazy allocation, zero-copy |
| AdaptiveResidualsWorkspace | adaptive_residuals_workspace.rs | Power-of-2 sizing |
| UnifiedLayerWorkspace | unified_layer_workspace.rs | Consolidated buffers |
| WorkspaceManaged | workspace_managed.rs | Unified trait interface |

