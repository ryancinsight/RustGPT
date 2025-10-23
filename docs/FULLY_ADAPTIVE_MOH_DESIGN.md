# Fully Adaptive Mixture-of-Heads (MoH) Design

**Date**: 2025-10-19  
**Status**: Design Phase  
**Goal**: Eliminate hardcoded shared/routed head splits, enable full adaptivity based on input complexity

---

## Executive Summary

Current MoH implementation has **hardcoded architecture**:
- Fixed `num_shared_heads` (e.g., 2 always active)
- Fixed `num_routed_heads` (e.g., 6 candidates for routing)
- Adaptive only in **which** routed heads to select, not **how many** total heads

**Proposed**: **Fully Adaptive MoH** where:
- ✅ **No hardcoded shared heads** - all heads are candidates for routing
- ✅ **Complexity-aware head count** - simple inputs use 1-3 heads, complex inputs use 5-8 heads
- ✅ **Learned head selection** - router learns which heads AND how many heads per token
- ✅ **Applicable to Transformer** - unified architecture

---

## Current MoH Architecture Analysis

### Current Implementation (src/head_router.rs)

**Architecture**:
```rust
pub struct HeadRouter {
    num_shared_heads: usize,      // HARDCODED: e.g., 2 (always active)
    num_routed_heads: usize,      // HARDCODED: e.g., 6 (routing candidates)
    
    w_shared: Array2<f32>,        // Router for shared heads
    w_routed: Array2<f32>,        // Router for routed heads
    w_head_type: Array2<f32>,     // Balancing between shared/routed
    
    threshold_predictor: Option<ThresholdPredictor>,  // Per-token threshold
}
```

**Routing Algorithm** (lines 471-567):
```text
1. Shared heads: ALWAYS activate first num_shared_heads (e.g., heads 0-1)
2. Routed heads: Compute routing scores for remaining heads (e.g., heads 2-7)
3. Adaptive Top-P: Select routed heads until cumulative probability ≥ threshold
4. Result: 2 shared + 1-6 routed = 3-8 total heads active
```

**Limitations**:
- ❌ **Shared heads always active** - wastes computation on simple inputs
- ❌ **Fixed split** - can't adapt shared/routed ratio per layer or task
- ❌ **Complexity ignored** - doesn't consider input difficulty
- ❌ **Suboptimal for recursive models** - recursive depth amplifies inefficiency

---

## Fully Adaptive MoH Design

### Core Concept: Complexity-Aware Dynamic Head Selection

**Key Insight**: Input complexity should determine head count, not architecture

**Examples**:
- **Simple input**: "The cat sat" → 1-2 heads sufficient
- **Medium input**: "Explain photosynthesis" → 3-4 heads needed
- **Complex input**: "Analyze quantum entanglement implications" → 6-8 heads required

### Architecture: Unified Head Router

```rust
pub struct FullyAdaptiveHeadRouter {
    /// Total number of heads (all are routing candidates)
    num_heads: usize,
    
    /// Router weights: (num_heads, embedding_dim)
    /// Computes routing scores for ALL heads (no shared/routed split)
    w_router: Array2<f32>,
    
    /// Complexity predictor: (embedding_dim, 1)
    /// Predicts input complexity → determines target head count
    w_complexity: Array2<f32>,
    complexity_bias: f32,
    
    /// Threshold predictor: (embedding_dim, 1)
    /// Predicts per-token threshold for top-p selection
    w_threshold: Array2<f32>,
    threshold_bias: f32,
    
    /// Minimum heads to activate (safety: at least 1)
    min_heads: usize,
    
    /// Maximum heads to activate (efficiency: at most num_heads)
    max_heads: usize,
    
    /// Load balance weight (prevent routing collapse)
    load_balance_weight: f32,
    
    /// Complexity-aware loss weight (encourage efficient head usage)
    complexity_loss_weight: f32,
}
```

### Routing Algorithm: Complexity-Driven Top-P

```text
Input: x ∈ R^(seq_len × embedding_dim)

1. COMPLEXITY PREDICTION (per token):
   complexity_logits = x @ w_complexity + complexity_bias
   complexity_scores = sigmoid(complexity_logits)  # Range [0, 1]
   target_heads = min_heads + complexity_scores * (max_heads - min_heads)
   # Example: complexity=0.2 → target=1.6 heads, complexity=0.8 → target=6.4 heads

2. THRESHOLD PREDICTION (per token):
   threshold_logits = x @ w_threshold + threshold_bias
   thresholds = sigmoid(threshold_logits) * 0.5 + 0.3  # Range [0.3, 0.8]

3. HEAD ROUTING (per token):
   routing_logits = x @ w_router^T  # (seq_len, num_heads)
   routing_probs = softmax(routing_logits, dim=-1)
   
4. ADAPTIVE TOP-P SELECTION (per token):
   For each token i:
     a. Sort heads by routing_probs[i] (descending)
     b. Select heads until:
        - Cumulative probability ≥ thresholds[i], OR
        - Number of heads ≥ target_heads[i], OR
        - All heads selected (fallback)
     c. Activate selected heads
     
5. OUTPUT:
   mask ∈ {0,1}^(seq_len × num_heads)  # Boolean mask of active heads
```

### Loss Functions

#### 1. Load Balance Loss (prevent routing collapse)
```text
L_balance = Σ(i=1 to num_heads) P_i × f_i

Where:
- P_i = average routing probability for head i across all tokens
- f_i = fraction of tokens that selected head i

Goal: Encourage uniform head usage across tokens
```

#### 2. Complexity Loss (encourage efficient head usage)
```text
L_complexity = λ × |avg_active_heads - avg_target_heads|

Where:
- avg_active_heads = actual average heads activated per token
- avg_target_heads = predicted target heads from complexity predictor
- λ = complexity_loss_weight (e.g., 0.01)

Goal: Align actual head usage with predicted complexity
```

#### 3. Sparsity Loss (encourage minimal head usage)
```text
L_sparsity = μ × (avg_active_heads / num_heads)

Where:
- μ = sparsity_weight (e.g., 0.001)

Goal: Minimize head usage while maintaining quality
```

#### Total Auxiliary Loss
```text
L_aux = L_balance + L_complexity + L_sparsity
```

---

## Advantages Over Current MoH

| Feature | Current MoH | Fully Adaptive MoH |
|---------|-------------|-------------------|
| **Shared Heads** | 2 always active (25%) | 0 (all heads routed) |
| **Min Active Heads** | 3 (2 shared + 1 routed) | 1 (complexity-driven) |
| **Max Active Heads** | 8 (2 shared + 6 routed) | 8 (all heads) |
| **Complexity Awareness** | ❌ No | ✅ Yes (learned predictor) |
| **Simple Input Efficiency** | 3 heads (37.5%) | 1-2 heads (12-25%) |
| **Complex Input Capacity** | 8 heads (100%) | 8 heads (100%) |
| **Adaptivity** | Which routed heads | Which heads + how many |
| **Recursive Model Compatibility** | Not tested | Designed for recursive models |
| **Parameter Overhead** | 1280 (for 8 heads, 128 dim) | 1152 (10% fewer) |

**Expected Gains**:
- **15-25% speedup** on simple inputs (vs current MoH's 5-8%)
- **Better gradient flow** in recursive models (fewer heads = less gradient splitting)
- **Adaptive capacity** (use more heads only when needed)
- **Unified architecture** (same code for different architectures)

---

## Implementation Plan

### Phase 1: Core Implementation (6 hours)

#### 1.1 Add FullyAdaptiveMoH to HeadSelectionStrategy (1 hour)

**File**: `src/model_config.rs`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    AllHeads,
    MixtureOfHeads { ... },  // Keep for backward compatibility
    
    /// Fully adaptive MoH: no hardcoded shared/routed split
    /// All heads are routing candidates, head count determined by complexity
    FullyAdaptiveMoH {
        /// Minimum heads to activate (safety)
        min_heads: usize,
        
        /// Maximum heads to activate (efficiency)
        max_heads: usize,
        
        /// Weight for load balance loss
        load_balance_weight: f32,
        
        /// Weight for complexity alignment loss
        complexity_loss_weight: f32,
        
        /// Weight for sparsity loss
        sparsity_weight: f32,
        
        /// Layer index (for layer-wise adaptation)
        layer_idx: usize,
    },
}
```

#### 1.2 Implement FullyAdaptiveHeadRouter (3 hours)

**File**: `src/head_router.rs` (add new struct)

```rust
pub struct FullyAdaptiveHeadRouter {
    // Core routing
    num_heads: usize,
    embedding_dim: usize,
    w_router: Array2<f32>,
    
    // Complexity prediction
    w_complexity: Array2<f32>,
    complexity_bias: f32,
    complexity_optimizer: Adam,
    
    // Threshold prediction
    w_threshold: Array2<f32>,
    threshold_bias: f32,
    threshold_optimizer: Adam,
    
    // Configuration
    min_heads: usize,
    max_heads: usize,
    load_balance_weight: f32,
    complexity_loss_weight: f32,
    sparsity_weight: f32,
    
    // Optimizers
    router_optimizer: Adam,
    
    // Caching
    cached_routing_probs: Option<Array2<f32>>,
    cached_complexity_scores: Option<Array1<f32>>,
    cached_thresholds: Option<Array1<f32>>,
    cached_mask: Option<Array2<bool>>,
    cached_target_heads: Option<Array1<f32>>,
}

impl FullyAdaptiveHeadRouter {
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        min_heads: usize,
        max_heads: usize,
        load_balance_weight: f32,
        complexity_loss_weight: f32,
        sparsity_weight: f32,
    ) -> Self { ... }
    
    pub fn route(&mut self, input: &Array2<f32>) -> Array2<bool> { ... }
    
    pub fn compute_load_balance_loss(&self) -> f32 { ... }
    
    pub fn compute_complexity_loss(&self) -> f32 { ... }
    
    pub fn compute_sparsity_loss(&self) -> f32 { ... }
    
    pub fn backward(&mut self, grad_output: &Array2<f32>, lr: f32) { ... }
}
```

#### 1.3 Integrate into SelfAttention (1 hour)

**File**: `src/self_attention.rs`

Update `set_head_selection()` to handle `FullyAdaptiveMoH`:

```rust
pub fn set_head_selection(&mut self, strategy: HeadSelectionStrategy, layer_idx: usize) {
    self.head_selection = strategy.clone();
    
    match &strategy {
        HeadSelectionStrategy::FullyAdaptiveMoH {
            min_heads,
            max_heads,
            load_balance_weight,
            complexity_loss_weight,
            sparsity_weight,
            ..
        } => {
            self.router = Some(HeadRouter::FullyAdaptive(
                FullyAdaptiveHeadRouter::new(
                    self.embedding_dim,
                    self.num_heads,
                    *min_heads,
                    *max_heads,
                    *load_balance_weight,
                    *complexity_loss_weight,
                    *sparsity_weight,
                )
            ));
        }
        // ... other strategies
    }
}
```

#### 1.4 Update main.rs Configuration (1 hour)

**File**: `src/main.rs`

```rust
// Fully Adaptive MoH configuration
let head_selection = HeadSelectionStrategy::FullyAdaptiveMoH {
    min_heads: 1,                      // Allow single head for simple inputs
    max_heads: 8,                      // All heads available for complex inputs
    load_balance_weight: 0.01,         // Prevent routing collapse
    complexity_loss_weight: 0.01,      // Align head usage with complexity
    sparsity_weight: 0.001,            // Encourage minimal head usage
    layer_idx: 0,                      // Set per layer in model builder
};
```

### Phase 2: Architecture Integration (3 hours)

#### 2.1 Add Fully Adaptive MoH to Architecture (2 hours)

**File**: `src/trm.rs`

Update `TinyRecursiveModel::new()` to accept `head_selection`:

```rust
pub fn new(
    embedding_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: Option<usize>,
    recursive_depth: usize,
    use_swiglu: bool,
    max_seq_len: usize,
    head_selection: HeadSelectionStrategy,  // NEW PARAMETER
) -> Self {
    let kv_heads = num_kv_heads.unwrap_or(num_heads);
    let mut attention = SelfAttention::new_with_gqa(
        embedding_dim,
        num_heads,
        kv_heads,
        false,
        max_seq_len,
        None,
    );
    
    // Enable fully adaptive MoH
    attention.set_head_selection(head_selection, 0);
    
    // ... rest of initialization
}
```

#### 2.2 Handle Router Gradients in Recursive Backward Pass (1 hour)

**Challenge**: Router gradients must accumulate across D recursive steps

**Solution**: Track router gradients separately per step, then sum:

```rust
// In TinyRecursiveModel::compute_gradients()
let mut router_grad_accumulator = vec![Array2::zeros((embedding_dim, num_heads)); recursive_depth];

for step in (0..recursive_depth).rev() {
    // ... existing gradient computation
    
    // Accumulate router gradients
    if let Some(router_grad) = attention.get_router_gradients() {
        router_grad_accumulator[step] = router_grad;
    }
}

// Sum router gradients across steps (with depth scaling)
let total_router_grad = router_grad_accumulator.iter()
    .fold(Array2::zeros((embedding_dim, num_heads)), |acc, grad| acc + grad)
    * (1.0 / recursive_depth as f32);
```

---

## Testing Strategy

### Baseline Comparisons

| Configuration | Expected Loss | Expected Grad Norm | Expected Avg Heads |
|---------------|---------------|-------------------|-------------------|
| AllHeads (baseline) | 0.436 | 1.41 | 8.0 |
| Current MoH | ~0.40 | ~2.0 | 5.5 |
| Fully Adaptive MoH | **~0.35** | **~1.5** | **3.5** |

### Test Cases

1. **Simple Input**: "The cat sat"
   - Expected: 1-2 heads active
   - Complexity score: 0.1-0.3

2. **Medium Input**: "Explain photosynthesis process"
   - Expected: 3-4 heads active
   - Complexity score: 0.4-0.6

3. **Complex Input**: "Analyze quantum entanglement implications for computing"
   - Expected: 6-8 heads active
   - Complexity score: 0.7-0.9

### Success Criteria

- ✅ **Loss**: ≤ 0.35 (20% better than baseline)
- ✅ **Gradient Stability**: ≤ 2.0 (maintain stability)
- ✅ **Efficiency**: Avg 3-4 heads (50% reduction vs AllHeads)
- ✅ **Adaptivity**: Complexity scores correlate with input difficulty
- ✅ **Output Quality**: Correct instruction-following maintained

---

## Next Steps

1. **Review & Approve Design** (User decision)
2. **Implement Phase 1** (6 hours)
3. **Test with Transformer** (2 hours)
4. **Implement Phase 2** (3 hours)
5. **Test with Architecture** (2 hours)
6. **Benchmark & Compare** (2 hours)

**Total Estimated Time**: 15 hours

