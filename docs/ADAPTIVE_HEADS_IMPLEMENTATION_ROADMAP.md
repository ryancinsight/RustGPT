# Adaptive Attention Heads: Implementation Roadmap

**Status**: Planning Phase - DO NOT IMPLEMENT YET  
**Date**: 2025-01-17  
**Estimated Effort**: 6 weeks  
**Risk Level**: LOW

---

## Overview

This document provides a detailed, step-by-step roadmap for implementing Mixture-of-Heads (MoH) adaptive attention in RustGPT. Each phase includes specific tasks, acceptance criteria, and testing requirements.

---

## Phase 1: Core Router Implementation (Week 1)

### Objectives
- Implement `HeadRouter` struct with routing logic
- Implement routing score computation
- Implement Top-K selection
- Implement load balance loss
- Unit tests for all router components

### Tasks

#### Task 1.1: Create `src/head_router.rs`
```rust
pub struct HeadRouter {
    num_shared_heads: usize,
    num_routed_heads: usize,
    num_active_routed_heads: usize,
    w_shared: Array2<f32>,
    w_routed: Array2<f32>,
    w_head_type: Array2<f32>,
    optimizer_shared: Adam,
    optimizer_routed: Adam,
    optimizer_head_type: Adam,
    cached_routing_scores: Option<Array2<f32>>,
    load_balance_weight: f32,
}
```

**Acceptance Criteria:**
- [ ] Struct compiles without errors
- [ ] All fields properly initialized
- [ ] Serialization/deserialization works

#### Task 1.2: Implement `new()` Constructor
```rust
impl HeadRouter {
    pub fn new(
        embedding_dim: usize,
        num_shared_heads: usize,
        num_routed_heads: usize,
        num_active_routed_heads: usize,
        load_balance_weight: f32,
    ) -> Self
}
```

**Acceptance Criteria:**
- [ ] Xavier initialization for all weights
- [ ] Optimizers initialized with correct learning rates
- [ ] Validates num_active_routed_heads ≤ num_routed_heads

#### Task 1.3: Implement `route()` Method
```rust
pub fn route(&mut self, input: &Array2<f32>) -> Array2<bool>
```

**Steps:**
1. Compute shared head scores: `Softmax(W_s @ input^T)`
2. Compute routed head scores: `Softmax(W_r @ input^T)`
3. Select Top-K routed heads per token
4. Compute head type balancing: `Softmax(W_h @ input^T)`
5. Combine into activation mask

**Acceptance Criteria:**
- [ ] Returns correct shape: (seq_len, total_heads)
- [ ] Shared heads always active
- [ ] Exactly K routed heads active per token
- [ ] Routing scores cached for backward pass

#### Task 1.4: Implement `compute_load_balance_loss()`
```rust
pub fn compute_load_balance_loss(&self) -> f32
```

**Formula:**
```
L_b = Σ(i=1 to num_routed) P_i * f_i

where:
  P_i = average routing score for head i
  f_i = fraction of tokens that selected head i
```

**Acceptance Criteria:**
- [ ] Returns scalar loss value
- [ ] Loss is 0 when perfectly balanced
- [ ] Loss increases with imbalance

#### Task 1.5: Implement Helper Methods
- `softmax()`: Numerically stable softmax
- `top_k_indices()`: Select Top-K indices per row
- `straight_through_estimator()`: Gradient flow for discrete selection

**Acceptance Criteria:**
- [ ] Softmax is numerically stable (no NaN/Inf)
- [ ] Top-K returns correct indices
- [ ] STE allows gradient flow

#### Task 1.6: Unit Tests
Create `tests/head_router_test.rs`:
- [ ] Test router creation
- [ ] Test routing score computation
- [ ] Test Top-K selection correctness
- [ ] Test load balance loss computation
- [ ] Test gradient flow through STE
- [ ] Test edge cases (K=0, K=all, single token)

**Deliverables:**
- ✅ `src/head_router.rs` (fully implemented)
- ✅ `tests/head_router_test.rs` (all tests passing)
- ✅ Documentation for all public methods

---

## Phase 2: Configuration and Integration (Week 2)

### Objectives
- Add `HeadSelectionStrategy` enum to `ModelConfig`
- Integrate router into `SelfAttention`
- Modify forward pass to use head selection
- Basic integration tests

### Tasks

#### Task 2.1: Update `src/model_config.rs`
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    AllHeads,
    MixtureOfHeads {
        num_shared_heads: usize,
        num_active_routed_heads: usize,
        load_balance_weight: f32,
    },
    StaticPruning {
        num_active_heads: usize,
    },
}

pub struct ModelConfig {
    // ... existing fields ...
    pub head_selection: HeadSelectionStrategy,
}
```

**Acceptance Criteria:**
- [ ] Enum compiles and serializes correctly
- [ ] All constructors updated with default `AllHeads`
- [ ] Documentation for each variant

#### Task 2.2: Update `src/self_attention.rs`
Add fields:
```rust
pub struct SelfAttention {
    // ... existing fields ...
    head_selection: HeadSelectionStrategy,
    router: Option<HeadRouter>,
    head_activation_mask: Option<Array2<bool>>,
}
```

**Acceptance Criteria:**
- [ ] Fields added without breaking existing code
- [ ] Router initialized only for MixtureOfHeads
- [ ] Backward compatibility maintained

#### Task 2.3: Modify `forward()` Method
```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    // 1. Compute head activation mask
    let mask = self.compute_head_mask(input);
    
    // 2. Compute attention for active heads only
    let outputs = self.compute_active_heads(input, &mask);
    
    // 3. Aggregate with weighted summation
    self.aggregate_outputs(&outputs, &mask)
}
```

**Acceptance Criteria:**
- [ ] AllHeads: All heads active (backward compatible)
- [ ] MixtureOfHeads: Dynamic routing works
- [ ] StaticPruning: First K heads active
- [ ] Output shape unchanged
- [ ] Gradients flow correctly

#### Task 2.4: Implement Helper Methods
- `compute_head_mask()`: Compute activation mask
- `compute_active_heads()`: Run attention for active heads
- `aggregate_outputs()`: Weighted summation

**Acceptance Criteria:**
- [ ] Methods are modular and testable
- [ ] Clear separation of concerns
- [ ] Efficient (no unnecessary allocations)

#### Task 2.5: Update `src/lib.rs`
```rust
pub mod head_router;
pub use head_router::HeadRouter;
pub use model_config::HeadSelectionStrategy;
```

**Acceptance Criteria:**
- [ ] Exports compile correctly
- [ ] Public API is clean

#### Task 2.6: Integration Tests
Create `tests/adaptive_heads_integration_test.rs`:
- [ ] Test AllHeads strategy
- [ ] Test MixtureOfHeads strategy
- [ ] Test StaticPruning strategy
- [ ] Test forward pass shape preservation
- [ ] Test backward pass gradient flow
- [ ] Test with different num_heads configurations

**Deliverables:**
- ✅ Updated `src/model_config.rs`
- ✅ Updated `src/self_attention.rs`
- ✅ Updated `src/lib.rs`
- ✅ Integration tests passing

---

## Phase 3: GQA Compatibility (Week 3)

### Objectives
- Implement KV head activation masking for GQA
- Test all GQA configurations
- Verify correctness with grouped heads

### Tasks

#### Task 3.1: Implement `compute_kv_head_mask()`
```rust
fn compute_kv_head_mask(
    &self,
    q_head_mask: &Array2<bool>
) -> Array2<bool>
```

**Logic:**
```
For each KV head group:
  Activate KV head if ANY Q head in group is active
  
Example: 8 Q heads, 2 KV heads (4 Q per KV)
  Q heads active: [0, 2, 5, 7]
  Q head 0, 2 → KV head 0 (activate)
  Q head 5, 7 → KV head 1 (activate)
  Result: Both KV heads active
```

**Acceptance Criteria:**
- [ ] Correct KV head activation for all GQA configs
- [ ] Handles MHA (num_kv_heads = num_heads)
- [ ] Handles MQA (num_kv_heads = 1)
- [ ] Efficient implementation

#### Task 3.2: Update Attention Computation
Modify attention to use KV head mask:
```rust
fn compute_attention_with_gqa(
    &self,
    q_mask: &Array2<bool>,
    kv_mask: &Array2<bool>
) -> Array2<f32>
```

**Acceptance Criteria:**
- [ ] Only active KV heads computed
- [ ] Q heads correctly grouped to KV heads
- [ ] Output shape correct

#### Task 3.3: GQA Tests
Create `tests/adaptive_heads_gqa_test.rs`:
- [ ] Test MHA (8 Q, 8 KV)
- [ ] Test GQA (8 Q, 4 KV)
- [ ] Test GQA (8 Q, 2 KV)
- [ ] Test MQA (8 Q, 1 KV)
- [ ] Test with MixtureOfHeads (50%, 75%, 90% active)
- [ ] Verify KV head activation correctness
- [ ] Verify output correctness

**Deliverables:**
- ✅ KV head masking implemented
- ✅ All GQA tests passing
- ✅ Documentation updated

---

## Phase 4: Comprehensive Testing (Week 4)

### Objectives
- Test all feature combinations
- Performance profiling
- Memory profiling
- Gradient flow verification

### Tasks

#### Task 4.1: Feature Combination Tests
Test MoH with:
- [ ] CoPE (all max_pos values)
- [ ] RoPE
- [ ] Learned positional embeddings
- [ ] Sliding Window (all window sizes)
- [ ] Adaptive Window (all strategies)
- [ ] RMSNorm
- [ ] SwiGLU
- [ ] All combinations of above

**Acceptance Criteria:**
- [ ] All combinations work correctly
- [ ] No crashes or NaN values
- [ ] Output shapes correct
- [ ] Gradients flow correctly

#### Task 4.2: Performance Profiling
Profile:
- [ ] Router overhead (forward pass)
- [ ] Router overhead (backward pass)
- [ ] Attention compute savings
- [ ] Net speedup (inference)
- [ ] Net speedup (training)

**Target Metrics:**
- Router overhead: <2%
- Attention savings: 10-50%
- Net speedup: 5-8%

#### Task 4.3: Memory Profiling
Profile:
- [ ] Router parameter memory
- [ ] Routing score memory
- [ ] Peak memory usage
- [ ] Memory leaks (valgrind/miri)

**Target Metrics:**
- Router params: <1% of model
- Runtime overhead: <1%
- No memory leaks

#### Task 4.4: Gradient Flow Verification
Verify:
- [ ] Router gradients are non-zero
- [ ] Router gradients are finite
- [ ] Head gradients flow correctly
- [ ] Load balance loss gradients correct
- [ ] No gradient explosion/vanishing

**Deliverables:**
- ✅ All feature tests passing
- ✅ Performance benchmarks documented
- ✅ Memory profiling results
- ✅ Gradient flow verified

---

## Phase 5: Documentation (Week 5)

### Objectives
- Write comprehensive usage guide
- Create examples
- Document performance characteristics
- Migration guide

### Tasks

#### Task 5.1: Create `docs/ADAPTIVE_HEADS_USAGE.md`
Sections:
- [ ] Quick start guide
- [ ] Configuration options
- [ ] Best practices
- [ ] Troubleshooting
- [ ] FAQ

#### Task 5.2: Create Examples
- [ ] Basic MoH usage
- [ ] MoH + GQA
- [ ] MoH + CoPE
- [ ] MoH + Adaptive Window
- [ ] Continue-tuning with MoH

#### Task 5.3: Update `src/main.rs`
Add configuration section:
```rust
// ============================================================================
// ADAPTIVE ATTENTION HEADS CONFIGURATION
// ============================================================================
// Choose head selection strategy:
//   - AllHeads: All heads active (backward compatible)
//   - MixtureOfHeads: Dynamic head selection (recommended)
//   - StaticPruning: First K heads (for comparison)
// ============================================================================

config.head_selection = HeadSelectionStrategy::MixtureOfHeads {
    num_shared_heads: 2,              // 25% of 8 heads
    num_active_routed_heads: 4,       // 67% of 6 routed heads
    load_balance_weight: 0.01,        // Prevent routing collapse
};
```

#### Task 5.4: Update Architecture Summary
Modify `src/model_builder.rs` to display:
- [ ] Head selection strategy
- [ ] Number of shared/routed heads
- [ ] Activation rate
- [ ] Router parameter count

**Deliverables:**
- ✅ Usage guide complete
- ✅ Examples working
- ✅ Configuration documented
- ✅ Architecture summary updated

---

## Phase 6: Optimization (Week 6)

### Objectives
- Profile bottlenecks
- Vectorize operations
- Optimize memory usage
- Final benchmarks

### Tasks

#### Task 6.1: Profile Bottlenecks
Use profiler to identify:
- [ ] Hotspots in router
- [ ] Hotspots in attention
- [ ] Memory allocation patterns
- [ ] Cache misses

#### Task 6.2: Optimize Router
- [ ] Vectorize matrix operations
- [ ] Fuse operations where possible
- [ ] Reduce allocations
- [ ] Optimize Top-K selection

#### Task 6.3: Optimize Attention
- [ ] Skip inactive heads efficiently
- [ ] Reuse buffers
- [ ] Optimize aggregation

#### Task 6.4: Final Benchmarks
Compare:
- [ ] AllHeads vs. MixtureOfHeads (50%, 75%, 90%)
- [ ] With/without GQA
- [ ] With/without Adaptive Window
- [ ] Memory usage
- [ ] Training time
- [ ] Inference time
- [ ] Accuracy

**Target Metrics:**
- Inference speedup: 5-8%
- Training speedup: 3-5%
- Accuracy: Same or +1-2%
- Memory: +0.5-1%

**Deliverables:**
- ✅ Optimizations implemented
- ✅ Benchmarks documented
- ✅ Performance targets met

---

## Acceptance Criteria (Overall)

### Functionality
- [ ] All three strategies work correctly
- [ ] Compatible with all existing features
- [ ] No regressions in existing tests
- [ ] Gradients flow correctly
- [ ] No NaN/Inf values

### Performance
- [ ] Inference speedup: 5-8%
- [ ] Training speedup: 3-5%
- [ ] Memory overhead: <1%
- [ ] Router overhead: <2%

### Quality
- [ ] All tests passing (unit + integration)
- [ ] Code coverage >90%
- [ ] Documentation complete
- [ ] Examples working
- [ ] No compiler warnings

### Design
- [ ] Follows SOLID/CUPID/GRASP principles
- [ ] Enum-based architecture
- [ ] Backward compatible
- [ ] Extensible for future variants

---

## Risk Mitigation Checklist

- [ ] Load balance loss prevents routing collapse
- [ ] Warm-up period for training stability
- [ ] GQA compatibility thoroughly tested
- [ ] Gradient flow verified
- [ ] Performance profiled and optimized
- [ ] Memory leaks checked
- [ ] Edge cases tested

---

## Success Metrics

| Metric | Target | Measured |
|--------|--------|----------|
| Inference Speedup | 5-8% | TBD |
| Training Speedup | 3-5% | TBD |
| Accuracy Change | 0% to +2% | TBD |
| Memory Overhead | <1% | TBD |
| Router Overhead | <2% | TBD |
| Test Coverage | >90% | TBD |
| All Tests Pass | 100% | TBD |

---

## Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Core Router | 1 week | Week 1 | Week 1 |
| Phase 2: Integration | 1 week | Week 2 | Week 2 |
| Phase 3: GQA | 1 week | Week 3 | Week 3 |
| Phase 4: Testing | 1 week | Week 4 | Week 4 |
| Phase 5: Documentation | 1 week | Week 5 | Week 5 |
| Phase 6: Optimization | 1 week | Week 6 | Week 6 |
| **Total** | **6 weeks** | | |

---

## Next Steps

**Before Starting:**
1. ✅ Review this roadmap
2. ⏳ Get team approval
3. ⏳ Set up development branch
4. ⏳ Set up benchmarking infrastructure
5. ⏳ Schedule weekly check-ins

**When Approved:**
1. Create `feature/adaptive-heads` branch
2. Start Phase 1: Core Router
3. Daily commits with tests
4. Weekly progress reviews
5. Continuous integration testing

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-17  
**Status**: ✅ Ready for Implementation

