# Mixture-of-Heads (MoH): Parameter-Neutral Implementation Plan

**Date**: 2025-01-17  
**Status**: Phase 1 Complete - Ready for Implementation  
**Goal**: Implement MoH with ≤2% parameter increase and 5-8% speedup

---

## Phase 1: Parameter Analysis (COMPLETE)

### Current Baseline

**Total Parameters**: 573,440

**Configuration:**
- EMBEDDING_DIM: 128
- HIDDEN_DIM: 256
- MAX_SEQ_LEN: 80
- num_layers: 3
- num_heads: 8
- num_kv_heads: 4 (GQA)
- head_dim: 16 (128 / 8)
- use_rms_norm: true
- use_swiglu: true
- positional_encoding: CoPE { max_pos: 64 }

**Parameter Breakdown Per Layer:**
- Attention (GQA): 32,768 params
  * Q heads (8): 8 × (128 × 16) = 16,384
  * K heads (4): 4 × (128 × 16) = 8,192
  * V heads (4): 4 × (128 × 16) = 8,192
- CoPE: 8 × (64 × 16) = 8,192
- RMSNorm (2×): 2 × 128 = 256
- SwiGLU: 3 × (128 × 256) = 98,304
- **Total per layer**: 139,520
- **3 layers**: 418,560
- **Embeddings + Output**: 154,880

### Parameter-Neutral MoH Configuration

**Target**: Keep total within ±2% of 573,440
- Acceptable range: 561,971 to 584,909
- Router budget: ~11,469 parameters (2%)

**Selected Configuration:**
- **num_shared_heads**: 2 (25% of 8 heads, always active)
- **num_routed_heads**: 6 (75% of 8 heads, Top-K selection)
- **num_active_routed_heads**: 4 (67% of routed heads)
- **Total active per token**: 2 + 4 = 6 heads (75%)
- **load_balance_weight**: 0.01 (standard from MoH paper)

**Router Parameters Per Layer:**
- W_shared: 2 × 128 = 256 params
- W_routed: 6 × 128 = 768 params
- W_head_type: 2 × 128 = 256 params
- **Total per layer**: 1,280 params
- **3 layers**: 3,840 params

**New Total**: 573,440 + 3,840 = **577,280 parameters**
- Increase: +3,840 (+0.67%)
- **Status**: ✅ Well within 2% budget

### Expected Performance

**Compute Savings:**
- Active heads: 6/8 = 75%
- Attention compute reduction: 25%
- Attention is ~30% of total compute
- **Gross savings**: 25% × 30% = 7.5%

**Router Overhead:**
- 3 matrix multiplications per layer
- 2 softmax operations per layer
- Top-K selection per layer
- **Estimated overhead**: 1-2% of total compute

**Net Speedup:**
- Gross savings: 7.5%
- Router overhead: 1-2%
- **Target net speedup**: 5-8% ✅

**Memory Impact:**
- Router params: 3,840 / 573,440 = 0.67%
- Runtime (routing scores): batch × seq_len × num_heads × 4 bytes
- Example: 1 × 80 × 8 × 4 = 2,560 bytes = 2.5KB
- **Total memory overhead**: <1% ✅

---

## Phase 2: Implementation Plan

### File Structure

**NEW Files:**
1. `src/routing.rs` - Shared routing utilities (for MoH and future MoE)
2. `src/head_router.rs` - MoH-specific router implementation
3. `tests/adaptive_heads_test.rs` - Comprehensive unit tests
4. `tests/adaptive_heads_integration_test.rs` - Integration tests

**MODIFIED Files:**
1. `src/model_config.rs` - Add HeadSelectionStrategy enum
2. `src/self_attention.rs` - Integrate router, modify forward pass
3. `src/llm.rs` - Add load balance loss
4. `src/model_builder.rs` - Initialize router, display stats
5. `src/lib.rs` - Export new types
6. `src/main.rs` - Add MoH configuration

### Modular Design for MoE Reusability

**Shared Routing Utilities (`src/routing.rs`):**
```rust
pub mod routing {
    // Generic Top-K selection (reusable for MoH and MoE)
    pub fn top_k_indices(scores: &Array2<f32>, k: usize) -> Vec<Vec<usize>>;
    
    // Generic load balance loss (reusable for MoH and MoE)
    pub fn compute_load_balance_loss(
        routing_scores: &Array2<f32>,
        activation_mask: &Array2<bool>,
    ) -> f32;
    
    // Straight-through estimator for gradient flow
    pub fn straight_through_estimator(
        forward_output: &Array2<bool>,
        backward_gradient: &Array2<f32>,
    ) -> Array2<f32>;
    
    // Numerically stable softmax
    pub fn softmax(x: &Array2<f32>) -> Array2<f32>;
}
```

**Benefits:**
- ✅ DRY principle - no code duplication
- ✅ Easier maintenance - fix bugs in one place
- ✅ Consistent behavior - MoH and MoE use same logic
- ✅ Easier testing - test utilities once
- ✅ Future-proof - MoE implementation straightforward

### HeadSelectionStrategy Enum

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    /// All heads always active (standard multi-head attention)
    /// - Zero overhead
    /// - Backward compatible
    AllHeads,
    
    /// Mixture-of-Heads: dynamic head selection per token
    MixtureOfHeads {
        num_shared_heads: usize,
        num_active_routed_heads: usize,
        load_balance_weight: f32,
    },
    
    /// Static pruning: first K heads (for ablation studies)
    StaticPruning {
        num_active_heads: usize,
    },
}
```

### HeadRouter Structure

```rust
pub struct HeadRouter {
    num_shared_heads: usize,
    num_routed_heads: usize,
    num_active_routed_heads: usize,
    
    // Router weights
    w_shared: Array2<f32>,      // (num_shared, embedding_dim)
    w_routed: Array2<f32>,      // (num_routed, embedding_dim)
    w_head_type: Array2<f32>,   // (2, embedding_dim)
    
    // Optimizers
    optimizer_shared: Adam,
    optimizer_routed: Adam,
    optimizer_head_type: Adam,
    
    // Cached for backward pass
    cached_routing_scores: Option<Array2<f32>>,
    cached_activation_mask: Option<Array2<bool>>,
    
    load_balance_weight: f32,
}
```

---

## Phase 3: Testing Strategy

### Unit Tests (`tests/adaptive_heads_test.rs`)

1. **Router Creation**
   - Test initialization
   - Test parameter count
   - Test weight shapes

2. **Routing Logic**
   - Test shared head scores
   - Test routed head scores
   - Test Top-K selection correctness
   - Test head type balancing

3. **Load Balance Loss**
   - Test loss computation
   - Test loss is 0 when balanced
   - Test loss increases with imbalance

4. **Gradient Flow**
   - Test straight-through estimator
   - Test router gradient computation
   - Test gradient magnitudes

### Integration Tests (`tests/adaptive_heads_integration_test.rs`)

1. **Feature Combinations**
   - MoH + GQA (all configurations)
   - MoH + CoPE
   - MoH + RoPE
   - MoH + Learned embeddings
   - MoH + Sliding Window
   - MoH + Adaptive Window
   - MoH + RMSNorm
   - MoH + SwiGLU

2. **Correctness**
   - Output shape preservation
   - No NaN/Inf values
   - Gradient flow correctness
   - KV head activation (GQA)

3. **Performance**
   - Parameter count verification
   - Memory usage profiling
   - Compute time measurement

---

## Phase 4: Configuration

### Default Configuration (`src/main.rs`)

```rust
// ============================================================================
// ADAPTIVE ATTENTION HEADS CONFIGURATION
// ============================================================================
// Choose head selection strategy:
//   - AllHeads: All heads active (backward compatible, baseline)
//   - MixtureOfHeads: Dynamic head selection (recommended, 5-8% speedup)
//   - StaticPruning: First K heads (for comparison)
// ============================================================================

use llm::HeadSelectionStrategy;

let head_selection = HeadSelectionStrategy::MixtureOfHeads {
    num_shared_heads: 2,              // 25% always active
    num_active_routed_heads: 4,       // 67% of routed heads
    load_balance_weight: 0.01,        // Prevent routing collapse
};

// Apply to config
config.head_selection = head_selection;
```

### Backward Compatibility

```rust
// Default: AllHeads (backward compatible)
impl Default for HeadSelectionStrategy {
    fn default() -> Self {
        HeadSelectionStrategy::AllHeads
    }
}
```

---

## Phase 5: Validation

### Success Criteria

**Parameters:**
- ✅ Total: 577,280 (baseline: 573,440)
- ✅ Increase: +0.67% (target: <2%)

**Performance:**
- ⏳ Inference speedup: 5-8% (to be measured)
- ⏳ Training speedup: 3-5% (to be measured)
- ⏳ Memory overhead: <1% (to be measured)

**Quality:**
- ⏳ All tests passing (187+ existing + new tests)
- ⏳ No compiler warnings
- ⏳ No clippy warnings
- ⏳ Code coverage >90%

**Compatibility:**
- ⏳ Works with GQA (all configurations)
- ⏳ Works with CoPE, RoPE, Learned
- ⏳ Works with Adaptive Window
- ⏳ Works with RMSNorm, SwiGLU

---

## Design Principles Compliance

| Principle | Compliance | Notes |
|-----------|-----------|-------|
| **SOLID** | ✅ | Single responsibility (head selection) |
| **CUPID** | ✅ | Composable with all features |
| **GRASP** | ✅ | Information expert (router knows importance) |
| **CLEAN** | ✅ | Clear separation (routing vs attention) |
| **SSOT** | ✅ | Single enum for strategy |
| **SPOT** | ✅ | Single configuration point |
| **Zero-cost** | ✅ | Minimal overhead (<1% memory, 1-2% compute) |

---

## Future MoE Compatibility

**Shared Infrastructure:**
- ✅ `routing::top_k_indices()` - Reusable for expert selection
- ✅ `routing::compute_load_balance_loss()` - Reusable for expert balancing
- ✅ `routing::straight_through_estimator()` - Reusable for gradient flow
- ✅ Router pattern - Same architecture for ExpertRouter

**When MoE is Implemented:**
```rust
// src/expert_router.rs (FUTURE)
pub struct ExpertRouter {
    // Same structure as HeadRouter
    // Uses same routing utilities
    // Different application (FFN experts vs attention heads)
}
```

---

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Planning | Complete | ✅ |
| Phase 2: Implementation | 2-3 hours | ⏳ |
| Phase 3: Testing | 1-2 hours | ⏳ |
| Phase 4: Configuration | 30 min | ⏳ |
| Phase 5: Validation | 1 hour | ⏳ |
| **Total** | **4-6 hours** | ⏳ |

---

## Next Steps

1. ✅ Phase 1 Complete: Parameter analysis done
2. ⏳ Create `src/routing.rs` with shared utilities
3. ⏳ Create `src/head_router.rs` with MoH implementation
4. ⏳ Update `src/model_config.rs` with enum
5. ⏳ Integrate into `src/self_attention.rs`
6. ⏳ Update remaining files
7. ⏳ Create comprehensive tests
8. ⏳ Validate performance and parameters

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-17  
**Status**: ✅ Phase 1 Complete, Ready for Implementation

