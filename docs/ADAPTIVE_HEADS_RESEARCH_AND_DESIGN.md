# Adaptive Attention Heads: Research and Design Document

**Status**: Research and Planning Phase (Implementation NOT started)  
**Date**: 2025-01-17  
**Author**: RustGPT Development Team

---

## Executive Summary

This document presents comprehensive research on adaptive attention head mechanisms and proposes a detailed design for integrating **Mixture-of-Heads (MoH)** into RustGPT. MoH dynamically selects which attention heads to activate per token, achieving 5-8% inference speedup and 0-2% accuracy improvement while maintaining full compatibility with existing features (GQA, CoPE, Sliding Window, Adaptive Window).

**Key Findings:**
- MoH achieves same/better performance with 50-90% of attention heads
- Successfully continue-tuned LLaMA3-8B with only 3% of original training data
- Complementary to Adaptive Window (spatial vs. channel efficiency)
- Proven stable training with load balance loss
- Minimal memory overhead (<1%)

---

## 1. Research Phase: Academic Literature Review

### 1.1 Primary Paper: MoH (Mixture-of-Heads)

**Paper**: "MoH: Multi-Head Attention as Mixture-of-Head Attention"  
**Authors**: Peng Jin, Bo Zhu, Li Yuan, Shuicheng Yan (Skywork AI)  
**Published**: arXiv:2410.11842, October 2024  
**Code**: https://github.com/SkyworkAI/MoH

#### Mathematical Formulation

**Standard Multi-Head Attention (Summation Form):**
```
MultiHead(X, X') = Σ(i=1 to h) H^i W_O^i
```

**Mixture-of-Heads Attention:**
```
MoH(X, X') = Σ(i=1 to h) g_i H^i W_O^i
```

Where `g_i` is the routing score (non-zero only for activated heads).

**Two-Stage Routing:**

1. **Shared Heads** (always active):
   ```
   g_i = α_1 * Softmax(W_s x_t)_i    for 1 ≤ i ≤ h_s
   ```

2. **Routed Heads** (Top-K selection):
   ```
   g_i = α_2 * Softmax(W_r x_t)_{i-h_s}    if Head i is in Top-K
   g_i = 0                                  otherwise
   ```

3. **Head Type Balancing:**
   ```
   [α_1, α_2] = Softmax(W_h x_t)
   ```

**Load Balance Loss:**
```
L_b = Σ(i=h_s+1 to h) P_i * f_i

where:
  P_i = (1/T) Σ(t=1 to T) Softmax(W_r x_t)_{i-h_s}  (avg routing score)
  f_i = (1/T) Σ(t=1 to T) 𝟙(token x_t selects Head i)  (activation frequency)
```

#### Key Results

| Model | Heads Active | Performance | Speedup |
|-------|-------------|-------------|---------|
| MoH-ViT-B | 75% | 84.9% (vs 84.8%) | ~1.3× |
| MoH-ViT-B | 50% | 84.7% (vs 84.8%) | ~2.0× |
| MoH-LLaMA3-8B | 75% | 64.0% (vs 61.6%) | ~1.3× |
| MoH-DiT-XL/2 | 90% | Better FID | ~1.1× |

**Key Advantages:**
1. **Efficiency**: 25-50% reduction in attention compute
2. **Flexibility**: Weighted summation vs. standard summation
3. **Specialization**: Heads learn distinct features (lower similarity)
4. **Compatibility**: Works with ViT, DiT, LLMs
5. **Continue-tuning**: Can upgrade pre-trained models

### 1.2 Related Work

**Attention Head Pruning:**
- Voita et al. (2019): Quantify head importance, prune redundant heads
- Michel et al. (2019): Extensive pruning without accuracy loss
- Bhattacharyya et al. (2023): Reduce redundancy in vision models

**Mixture-of-Experts:**
- Shazeer et al. (2017): MoE layers between LSTM
- Fedus et al. (2022): Switch Transformer (Top-1 expert)
- Lepikhin et al. (2021): Gshard (Top-2 expert routing)

**Head Specialization:**
- Wu et al. (2024): Retrieval heads in long-context models
- Fu et al. (2024): KV cache compression via head selection
- Xiao et al. (2024): DuoAttention (different head types)

### 1.3 Why MoH Over Alternatives?

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Static Pruning** | Simple, no overhead | Fixed heads, no adaptation | ❌ Too rigid |
| **Gradient-based Pruning** | Principled selection | One-time, not dynamic | ❌ Not adaptive |
| **Entropy-based Selection** | Interpretable | Heuristic, not learned | ❌ Suboptimal |
| **MoH (Chosen)** | Learned, dynamic, proven | Routing overhead | ✅ Best trade-off |

---

## 2. Analysis Phase: RustGPT Architecture Compatibility

### 2.1 Current Architecture

**SelfAttention Structure:**
```rust
pub struct SelfAttention {
    pub num_heads: usize,              // Fixed number of heads
    pub num_kv_heads: usize,           // For GQA
    heads: Vec<AttentionHead>,         // All heads always active
    positional_encoding: PositionalEncodingVariant,  // Learned/RoPE/CoPE
    window_size: Option<usize>,        // Sliding window
    use_adaptive_window: bool,         // Dynamic window sizing
    // ... other fields
}
```

### 2.2 Compatibility Analysis

| Feature | Compatibility | Notes |
|---------|--------------|-------|
| **GQA** | ✅ Compatible | Activate KV head if ANY Q head in group is active |
| **CoPE** | ✅ Compatible | Each selected head uses its own CoPE instance |
| **RoPE** | ✅ Compatible | Apply RoPE to selected heads only |
| **Sliding Window** | ✅ Compatible | Selected heads use window masking |
| **Adaptive Window** | ✅ Compatible | Orthogonal mechanisms (spatial vs. channel) |
| **RMSNorm** | ✅ Compatible | No interaction |
| **SwiGLU** | ✅ Compatible | No interaction |

**GQA + MoH Interaction:**
```
Example: 8 Q heads, 2 KV heads (4 Q heads per KV head)
- Route 4 Q heads: [0, 2, 5, 7]
- Q head 0, 2 → KV head 0 (activate)
- Q head 5, 7 → KV head 1 (activate)
- Result: Both KV heads active (efficient)
```

### 2.3 Adaptive Window vs. Adaptive Heads

| Aspect | Adaptive Window | Adaptive Heads |
|--------|----------------|----------------|
| **Dimension** | Spatial (which tokens) | Channel (which heads) |
| **Granularity** | Per-layer | Per-token |
| **Mechanism** | Window size adjustment | Head selection |
| **Overhead** | ~0% (masking only) | ~1-2% (routing) |
| **Savings** | 0-50% (context-dependent) | 10-50% (activation rate) |
| **Training** | No extra loss | Load balance loss |
| **Complementary?** | **YES** - Orthogonal | **YES** - Orthogonal |

**Synergy:** Combined speedup is multiplicative!
- Adaptive Window: 50% tokens → 2× speedup
- Adaptive Heads: 75% heads → 1.33× speedup
- Combined: 2× × 1.33× = **2.66× speedup**

---

## 3. Design Phase: Proposed Architecture

### 3.1 Enum-Based Configuration

Following RustGPT's `PositionalEncodingType` pattern:

```rust
/// Strategy for selecting which attention heads to activate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    /// All heads always active (standard multi-head attention)
    /// - Zero overhead
    /// - Backward compatible
    /// - Use for baseline comparisons
    AllHeads,
    
    /// Mixture-of-Heads: dynamic head selection per token
    /// - Learned routing via router network
    /// - Shared heads (always active) + routed heads (Top-K)
    /// - Weighted summation for flexibility
    /// - Load balance loss to prevent routing collapse
    MixtureOfHeads {
        /// Number of shared heads (always active, capture common knowledge)
        /// Recommended: 25% of total heads (e.g., 2 out of 8)
        num_shared_heads: usize,
        
        /// Number of routed heads to activate per token (Top-K)
        /// Recommended: 50-75% of routed heads
        /// Example: 8 total, 2 shared, 6 routed → activate 3-4 routed
        num_active_routed_heads: usize,
        
        /// Weight for load balance loss (β in paper)
        /// Recommended: 0.01 (prevents routing collapse)
        load_balance_weight: f32,
    },
    
    /// Static head pruning: use only first K heads (for ablation studies)
    /// - No routing overhead
    /// - Fixed head selection
    /// - Useful for comparing against dynamic routing
    StaticPruning {
        /// Number of heads to keep active
        num_active_heads: usize,
    },
}
```

### 3.2 Router Architecture

```rust
/// Router network for Mixture-of-Heads attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeadRouter {
    /// Number of shared heads
    num_shared_heads: usize,
    
    /// Number of routed heads
    num_routed_heads: usize,
    
    /// Number of routed heads to activate (Top-K)
    num_active_routed_heads: usize,
    
    /// Router for shared heads: W_s ∈ R^(num_shared × embedding_dim)
    w_shared: Array2<f32>,
    
    /// Router for routed heads: W_r ∈ R^(num_routed × embedding_dim)
    w_routed: Array2<f32>,
    
    /// Head type balancing: W_h ∈ R^(2 × embedding_dim)
    w_head_type: Array2<f32>,
    
    /// Optimizers for router weights
    optimizer_shared: Adam,
    optimizer_routed: Adam,
    optimizer_head_type: Adam,
    
    /// Cached routing scores for backward pass
    #[serde(skip)]
    cached_routing_scores: Option<Array2<f32>>,
    
    /// Load balance loss weight
    load_balance_weight: f32,
}
```

### 3.3 Integration with SelfAttention

```rust
pub struct SelfAttention {
    // ... existing fields ...
    
    /// Head selection strategy
    head_selection: HeadSelectionStrategy,
    
    /// Router network (only for MixtureOfHeads)
    router: Option<HeadRouter>,
    
    /// Cached head activation mask for current forward pass
    #[serde(skip)]
    head_activation_mask: Option<Array2<bool>>,  // (seq_len, num_heads)
}
```

### 3.4 Forward Pass Algorithm

```rust
fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
    let (seq_len, emb_dim) = input.shape();
    
    // Step 1: Compute head activation mask
    let head_mask = match &self.head_selection {
        HeadSelectionStrategy::AllHeads => {
            // All heads active
            Array2::from_elem((seq_len, self.num_heads), true)
        }
        HeadSelectionStrategy::MixtureOfHeads { .. } => {
            // Route heads dynamically
            self.router.as_mut().unwrap().route(input)
        }
        HeadSelectionStrategy::StaticPruning { num_active_heads } => {
            // First K heads active
            self.compute_static_mask(seq_len, *num_active_heads)
        }
    };
    
    self.head_activation_mask = Some(head_mask.clone());
    
    // Step 2: Compute attention for active heads only
    let mut outputs = Vec::new();
    for (head_idx, head) in self.heads.iter().enumerate() {
        // Check if this head is active for any token
        if head_mask.column(head_idx).iter().any(|&active| active) {
            let head_output = head.forward(input);
            outputs.push((head_idx, head_output));
        }
    }
    
    // Step 3: Weighted summation (for MoH) or standard summation
    self.aggregate_head_outputs(&outputs, &head_mask)
}
```

---

## 4. Implementation Plan

### 4.1 File-by-File Changes

**NEW: `src/head_router.rs`** (Primary implementation)
- `HeadRouter` struct with router weights
- `new()`: Initialize router with Xavier initialization
- `route()`: Compute routing scores and select Top-K heads
- `compute_load_balance_loss()`: Implement L_b from paper
- `backward()`: Update router weights
- Helper methods: `softmax()`, `top_k_indices()`, `straight_through_estimator()`

**MODIFY: `src/model_config.rs`** (Configuration)
- Add `HeadSelectionStrategy` enum (3 variants)
- Add `head_selection: HeadSelectionStrategy` field to `ModelConfig`
- Update constructors with default `AllHeads`
- Add builder methods for MoH configuration

**MODIFY: `src/self_attention.rs`** (Integration)
- Add `router: Option<HeadRouter>` field
- Add `head_selection: HeadSelectionStrategy` field
- Update constructors to initialize router
- Modify `forward()` to use head selection
- Add `compute_head_activation_mask()` method
- Add `compute_kv_head_mask_for_gqa()` method
- Store routing scores for backward pass

**MODIFY: `src/llm.rs`** (Loss computation)
- Add load balance loss accumulation
- Modify `backward()` to include L_b
- Add `get_total_loss()` combining task loss + load balance loss

**MODIFY: `src/model_builder.rs`** (Network construction)
- Initialize router if MoH is enabled
- Update architecture summary to display head selection
- Add head activation statistics

**MODIFY: `src/lib.rs`** (Exports)
- Add `pub mod head_router;`
- Export `HeadSelectionStrategy`

**MODIFY: `src/main.rs`** (Configuration examples)
- Add head selection configuration section
- Provide examples for all three strategies
- Document trade-offs

**NEW: `tests/adaptive_heads_test.rs`** (Testing)
- Test router creation and initialization
- Test routing score computation
- Test Top-K selection correctness
- Test load balance loss
- Test GQA compatibility
- Test with all PE types (CoPE, RoPE, Learned)
- Test with sliding window
- Test with adaptive window
- Test backward pass and gradient flow

**NEW: `docs/ADAPTIVE_HEADS_IMPLEMENTATION.md`** (Documentation)
- Usage guide with examples
- Performance benchmarks
- Troubleshooting guide
- Migration from AllHeads to MoH

### 4.2 Implementation Phases

**Phase 1: Core Router (Week 1)**
- Implement `HeadRouter` struct
- Routing score computation
- Top-K selection
- Load balance loss
- Unit tests for router

**Phase 2: Integration (Week 2)**
- Add enum to `ModelConfig`
- Integrate router into `SelfAttention`
- Modify forward pass
- Basic integration tests

**Phase 3: GQA Compatibility (Week 3)**
- Implement KV head masking
- Test all GQA configurations
- Verify correctness

**Phase 4: Comprehensive Testing (Week 4)**
- Test all feature combinations
- Performance profiling
- Memory profiling
- Gradient flow verification

**Phase 5: Documentation (Week 5)**
- Write usage guide
- Create examples
- Document performance characteristics
- Migration guide

**Phase 6: Optimization (Week 6)**
- Profile bottlenecks
- Vectorize operations
- Optimize memory usage
- Final benchmarks

---

## 5. Performance Analysis

### 5.1 Memory Impact

**Router Parameters:**
```
W_shared: num_shared_heads × embedding_dim
W_routed: num_routed_heads × embedding_dim
W_head_type: 2 × embedding_dim

Example (8 heads, 2 shared, 512 dim):
  (2 + 6 + 2) × 512 = 5,120 params = 20KB
  
Negligible compared to attention weights (millions of params)
```

**Runtime Memory:**
```
Routing scores: batch_size × seq_len × num_heads × 4 bytes

Example (batch=2, seq=1024, heads=8):
  2 × 1024 × 8 × 4 = 64KB (temporary)
  
Total overhead: <1% of model memory
```

### 5.2 Compute Impact

**Router Overhead:**
- 3 matrix multiplications: O(seq_len × embedding_dim × num_heads)
- 2 softmax operations: O(seq_len × num_heads)
- Top-K selection: O(seq_len × num_heads × log(K))
- **Total: ~1-2% of attention compute**

**Attention Savings:**
- Standard: 100% of heads active
- MoH (75% active): 25% compute reduction
- Attention is ~30-40% of total model compute
- **Net savings: 7-10% of total compute**

**Expected Speedup:**
- Inference: 5-8% (net of routing overhead)
- Training: 3-5% (net of routing + load balance)

### 5.3 Accuracy Impact

Based on MoH paper results:
- **75% heads active**: +0% to +2% accuracy
- **50% heads active**: -0.1% to +0.5% accuracy
- **Weighted summation**: Adds flexibility, can improve accuracy

---

## 6. Risk Assessment and Mitigation

### 6.1 High-Priority Risks

**Risk 1: Routing Collapse**
- **Description**: All tokens route to same few heads
- **Impact**: Loss of diversity, degraded performance
- **Probability**: Medium
- **Mitigation**:
  1. Load balance loss (β = 0.01)
  2. Monitor head utilization during training
  3. Initialize router with small random weights
  4. Adjust β if imbalance detected
- **Contingency**: Fall back to AllHeads

### 6.2 Medium-Priority Risks

**Risk 2: Training Instability**
- **Description**: Routing introduces non-stationarity
- **Impact**: Slower convergence
- **Probability**: Low
- **Mitigation**:
  1. Warm-up period (first 1000-5000 steps with AllHeads)
  2. Lower learning rate for router (0.1× main LR)
  3. Gradient clipping for router weights
- **Contingency**: Extend warm-up period

**Risk 3: GQA Incompatibility**
- **Description**: Routed Q heads misalign with grouped KV heads
- **Impact**: Incorrect attention
- **Probability**: Low
- **Mitigation**:
  1. Proper KV head activation masking
  2. Comprehensive GQA tests
  3. Clear documentation
- **Contingency**: Disable MoH with GQA (conservative)

### 6.3 Low-Priority Risks

**Risk 4: Gradient Flow Issues**
- **Mitigation**: Straight-through estimator (proven technique)
- **Contingency**: Gumbel-Softmax (differentiable alternative)

**Risk 5: Overhead Exceeds Savings**
- **Mitigation**: Profile carefully, optimize router
- **Contingency**: Use StaticPruning (no routing overhead)

---

## 7. Comparison with Adaptive Window

| Feature | Adaptive Window | Adaptive Heads |
|---------|----------------|----------------|
| **Mechanism** | Adjust window size | Select heads |
| **Dimension** | Spatial (tokens) | Channel (heads) |
| **Overhead** | ~0% | ~1-2% |
| **Savings** | 0-50% | 10-50% |
| **Learned?** | Heuristic | Yes (router) |
| **Complementary?** | **YES** | **YES** |

**Key Insight**: These are orthogonal optimizations that can be combined for multiplicative speedup!

---

## 8. Recommendations

### 8.1 Implementation Decision

**✅ RECOMMENDED: Implement MoH as primary adaptive heads mechanism**

**Rationale:**
1. Strong empirical evidence (MoH paper, multiple domains)
2. Proven stable training
3. Compatible with all existing features
4. Minimal overhead (<2%)
5. Significant speedup (5-8%)
6. Potential accuracy improvement (0-2%)

### 8.2 Configuration Defaults

```rust
// Conservative defaults for initial release
HeadSelectionStrategy::MixtureOfHeads {
    num_shared_heads: 2,              // 25% of 8 heads
    num_active_routed_heads: 4,       // 67% of 6 routed heads
    load_balance_weight: 0.01,        // Standard from paper
}
```

### 8.3 Migration Path

1. **Phase 1**: AllHeads (current behavior, zero risk)
2. **Phase 2**: StaticPruning (validate savings, no routing)
3. **Phase 3**: MixtureOfHeads (full dynamic routing)

### 8.4 Testing Strategy

1. Unit tests for router components
2. Integration tests for all feature combinations
3. Performance benchmarks (memory, compute, accuracy)
4. Ablation studies (shared vs. routed heads, activation rates)
5. Long-running stability tests

---

## 9. References

### Primary Paper
- Jin, P., Zhu, B., Yuan, L., & Yan, S. (2024). MoH: Multi-Head Attention as Mixture-of-Head Attention. arXiv:2410.11842.

### Related Work
- Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.
- Michel, P., Levy, O., & Neubig, G. (2019). Are Sixteen Heads Really Better than One?
- Fedus, W., et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.
- Lepikhin, D., et al. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.

---

## 10. Next Steps

**DO NOT IMPLEMENT YET** - This is a research and planning document.

**Before implementation:**
1. Review this design with team
2. Validate assumptions with small-scale experiments
3. Finalize configuration API
4. Create detailed test plan
5. Set up performance benchmarking infrastructure

**When ready to implement:**
1. Start with Phase 1 (Core Router)
2. Incremental development with continuous testing
3. Regular performance profiling
4. Document learnings and adjustments

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-17  
**Status**: ✅ Research Complete, Ready for Review

