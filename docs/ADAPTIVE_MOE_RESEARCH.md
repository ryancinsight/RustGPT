# Adaptive MoE Research & Implementation Plan

## Executive Summary

Based on user insight: **Hardcoded top-k routing may be incompatible with our adaptive architecture (MoH, sliding window)**. Research shows dynamic/adaptive routing significantly outperforms fixed top-k routing.

**Key Finding**: "Harder Tasks Need More Experts: Dynamic Routing in MoE Models" (Huang et al., 2024) demonstrates:
- **0.7% average improvement** over top-2 routing
- **<90% activated parameters** (more efficient)
- **Dynamic allocation**: Complex tasks get more experts, simple tasks get fewer
- **Layer-wise variation**: Bottom layers need more experts, top layers need fewer

## Research Papers

### 1. Dynamic Routing MoE (Huang et al., 2024) ⭐ PRIMARY
**Paper**: "Harder Tasks Need More Experts: Dynamic Routing in MoE Models"
**URL**: https://arxiv.org/abs/2403.07652
**Key Contributions**:
- **Top-P Routing**: Select experts until cumulative probability exceeds threshold p
- **Dynamic Loss**: Minimize entropy to encourage sparse expert selection
- **Load Balance Loss**: Ensure uniform expert utilization
- **Results**: 0.7% improvement with <2 experts activated on average

**Mathematical Formulation**:
```
1. Compute routing probabilities: P = Softmax(W_r · x^T)
2. Sort P in descending order → sorted indices I
3. Find minimum k where: Σ(P_i,j for j≤k) ≥ p
4. Activate experts: S = {e_I1, e_I2, ..., e_Ik}
5. Output: MoE(x) = Σ(P_i * e_i(x)) for e_i ∈ S
```

**Loss Function**:
```
Loss = Loss_lm + α*Loss_balance + β*Loss_dynamic

Loss_dynamic = -Σ(P_i * log(P_i))  # Minimize entropy
Loss_balance = N * Σ(f_i * Q_i)     # Balance expert utilization

where:
- f_i = fraction of tokens choosing expert i
- Q_i = fraction of router probability for expert i
- α = 1e-2 (load balance weight)
- β = 1e-4 (dynamic loss weight)
```

**Hyperparameters**:
- **Threshold p**: 0.4 (controls confidence level)
- **Range**: 0.1-0.7 (0.4 is optimal)
- **Effect**: Higher p → more experts activated

**Key Insights**:
1. **Task-dependent**: BBH (reasoning) uses 1.87 experts, Hellaswag uses 1.72
2. **Layer-dependent**: L0 uses ~4 experts, L23 uses ~1 expert
3. **Token-dependent**: Ambiguous tokens (subwords) use more experts
4. **Training dynamics**: Starts with >2 experts, converges to <2 after 60B tokens

### 2. Expert Choice Routing (Zhou et al., 2022)
**Key Idea**: Inverted routing - experts choose tokens instead of tokens choosing experts
**Benefit**: Perfect load balancing, no auxiliary loss needed
**Trade-off**: Fixed FLOPS per layer (less flexible than dynamic routing)

### 3. Adaptive Top-K (Various, 2024)
**Key Idea**: Learn k per token/layer instead of fixing k globally
**Approaches**:
- Learned gating for k selection
- Context-aware k prediction
- Reinforcement learning for k optimization

## Why Our MoE Failed: Root Cause Analysis

### Hypothesis: Hardcoded Top-2 Incompatible with Adaptive Architecture

**Evidence**:
1. **MoH Success**: Mixture-of-Heads uses adaptive attention weights per token
2. **Sliding Window Success**: Adaptive context window based on position
3. **MoE Failure**: Hardcoded top-2 forces rigid routing regardless of token complexity

**Specific Issues**:
1. **Gradient Accumulation**: 2 experts always active → gradients always accumulate from 2 sources
2. **Routing Rigidity**: Simple tokens waste computation, complex tokens lack capacity
3. **Layer Mismatch**: Early layers need more experts (4+), late layers need fewer (1)
4. **Training Instability**: Fixed routing prevents model from learning optimal expert allocation

**Validation**:
- Baseline (no MoE): Loss 0.387 ✅
- MoE top-2: Loss 3.12 ❌ (10x worse)
- Gradient explosion despite LARS
- Output collapse

## Proposed Solution: Adaptive Dynamic Routing

### Design Principles

1. **Adaptive Expert Selection**: Let model decide how many experts per token
2. **Confidence-Based Routing**: Use routing probability as confidence signal
3. **Layer-Wise Variation**: Allow different expert counts per layer
4. **Gradient Stability**: Integrate with bidirectional LARS
5. **Incremental Implementation**: Start simple, add complexity gradually

### Implementation Plan

#### Phase 1: Top-P Dynamic Routing (Recommended First Step)

**Algorithm**:
```rust
fn route_dynamic(&self, input: &Array2<f32>, threshold_p: f32) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let logits = self.router_weights.dot(&input.t());
    let probs = softmax(&logits, axis=0);  // Shape: [num_experts, seq_len]
    
    let mut expert_indices = Vec::new();
    let mut expert_weights = Vec::new();
    
    for token_idx in 0..seq_len {
        let token_probs = probs.column(token_idx);
        
        // Sort probabilities in descending order
        let mut sorted_indices: Vec<usize> = (0..num_experts).collect();
        sorted_indices.sort_by(|&a, &b| token_probs[b].partial_cmp(&token_probs[a]).unwrap());
        
        // Find minimum k where cumulative probability ≥ threshold_p
        let mut cumulative_prob = 0.0;
        let mut selected_experts = Vec::new();
        let mut selected_weights = Vec::new();
        
        for &expert_idx in &sorted_indices {
            cumulative_prob += token_probs[expert_idx];
            selected_experts.push(expert_idx);
            selected_weights.push(token_probs[expert_idx]);
            
            if cumulative_prob >= threshold_p {
                break;
            }
        }
        
        // Normalize weights (optional - paper doesn't normalize)
        // let sum: f32 = selected_weights.iter().sum();
        // selected_weights.iter_mut().for_each(|w| *w /= sum);
        
        expert_indices.push(selected_experts);
        expert_weights.push(selected_weights);
    }
    
    (expert_indices, expert_weights)
}
```

**Loss Functions**:
```rust
// Dynamic loss: Minimize entropy to encourage sparse selection
fn compute_dynamic_loss(&self, probs: &Array2<f32>) -> f32 {
    let entropy = -probs * probs.mapv(|p| if p > 1e-10 { p.ln() } else { 0.0 });
    entropy.sum() / (probs.nrows() * probs.ncols()) as f32
}

// Load balance loss: Encourage uniform expert utilization
fn compute_load_balance_loss(&self, expert_indices: &Vec<Vec<usize>>, probs: &Array2<f32>) -> f32 {
    let num_experts = probs.nrows();
    let num_tokens = expert_indices.len();
    
    // f_i: fraction of tokens choosing expert i
    let mut expert_counts = vec![0.0; num_experts];
    for token_experts in expert_indices {
        for &expert_idx in token_experts {
            expert_counts[expert_idx] += 1.0;
        }
    }
    let f: Vec<f32> = expert_counts.iter().map(|&c| c / num_tokens as f32).collect();
    
    // Q_i: fraction of router probability for expert i
    let q: Vec<f32> = (0..num_experts)
        .map(|i| probs.row(i).sum() / num_tokens as f32)
        .collect();
    
    // Load balance loss: N * Σ(f_i * Q_i)
    num_experts as f32 * f.iter().zip(q.iter()).map(|(fi, qi)| fi * qi).sum::<f32>()
}
```

**Configuration**:
```rust
// In main.rs
let use_moe: bool = true;
let moe_routing_type: &str = "dynamic";  // "top-k" or "dynamic"
let moe_threshold_p: f32 = 0.4;  // Confidence threshold (0.1-0.7)
let moe_dynamic_loss_weight: f32 = 1e-4;  // β
let moe_load_balance_weight: f32 = 1e-2;  // α

config.moe_routing_type = moe_routing_type.to_string();
config.moe_threshold_p = moe_threshold_p;
config.moe_dynamic_loss_weight = moe_dynamic_loss_weight;
```

#### Phase 2: Layer-Wise Adaptive Thresholds (Future Enhancement)

**Observation**: Bottom layers need more experts (p=0.6), top layers need fewer (p=0.2)

**Implementation**:
```rust
// Different threshold per layer
let layer_thresholds = vec![
    0.6, 0.6, 0.5, 0.5,  // L0-L3: More experts for shallow representations
    0.4, 0.4, 0.4, 0.4,  // L4-L7: Medium
    0.3, 0.3, 0.3, 0.3,  // L8-L11: Fewer experts
    0.2, 0.2, 0.2,       // L12-L14: Minimal experts (avoid overthinking)
];
```

#### Phase 3: Learned Threshold (Advanced)

**Idea**: Learn threshold p per token/layer instead of fixing it

**Implementation**:
```rust
// Add learnable threshold predictor
struct ThresholdPredictor {
    weights: Array2<f32>,  // [hidden_dim, 1]
    bias: f32,
}

impl ThresholdPredictor {
    fn predict(&self, input: &Array2<f32>) -> Array1<f32> {
        let logits = input.dot(&self.weights) + self.bias;
        logits.mapv(|x| sigmoid(x) * 0.6 + 0.1)  // Range: [0.1, 0.7]
    }
}
```

## Integration with Existing Architecture

### Compatibility Checklist

✅ **Bidirectional LARS**: Dynamic routing compatible - each expert gets adaptive LR
✅ **MoH (Mixture-of-Heads)**: Both use adaptive selection mechanisms
✅ **Sliding Window**: Both adapt based on context
✅ **Pre-LN + RMSNorm**: No conflicts
✅ **SwiGLU Experts**: Works with any expert architecture
✅ **CoPE**: Position encoding independent of routing
✅ **GQA**: Attention mechanism independent of FFN routing

### Gradient Flow Considerations

**Key Insight**: Dynamic routing changes gradient accumulation pattern

**Solution**: Scale gradients by number of active experts
```rust
// In backward pass
let num_active = expert_indices[token_idx].len();
let gradient_scale = 2.0 / num_active as f32;  // Normalize to expected 2 experts
let scaled_grad = grad * gradient_scale;
```

## Expected Outcomes

### Performance Targets

| Metric | Baseline | MoE Top-2 (Failed) | MoE Dynamic (Target) |
|--------|----------|-------------------|---------------------|
| **Loss** | 0.387 | 3.12 | **≤0.35** |
| **Output Quality** | Excellent | Collapsed | **Excellent** |
| **L0 Gradient** | 7-11 | 10-25 | **7-12** |
| **Avg Experts** | N/A | 2.0 (fixed) | **1.5-1.8** |
| **Parameters** | 197K | 393K | **~295K** (1.5x) |

### Success Criteria

**Minimum Viable Product (MVP)**:
- ✅ Code compiles
- ✅ Training completes 100 epochs
- ✅ Loss ≤ 0.40 (maintain baseline)
- ✅ Output quality: Coherent text
- ✅ Gradient stability: L0 in 7-15 range
- ✅ Average experts: 1.5-2.0 (adaptive)

**Stretch Goals**:
- Loss < 0.35 (improve baseline by 10%)
- Expert specialization: Different experts for different token types
- Layer-wise variation: L0 uses 2-3 experts, L14 uses 1 expert
- Inference speedup: 25% faster than top-2 (fewer experts)

## Implementation Timeline

### Session 1: Core Dynamic Routing (2-3 hours)
1. **Implement top-p routing** (60 min)
   - Modify Router::route() to support dynamic selection
   - Add threshold_p parameter
   - Test with fixed threshold

2. **Implement dynamic loss** (30 min)
   - Add entropy minimization loss
   - Integrate with existing loss computation

3. **Test and validate** (60 min)
   - Run training with p=0.4
   - Monitor average experts per token
   - Check gradient stability
   - Verify output quality

### Session 2: Optimization (1-2 hours)
1. **Tune threshold p** (30 min)
   - Test p ∈ {0.3, 0.4, 0.5}
   - Find optimal value

2. **Add layer-wise thresholds** (30 min)
   - Implement per-layer threshold array
   - Test with graduated thresholds

3. **Gradient scaling** (30 min)
   - Normalize gradients by num_active_experts
   - Verify stability

### Session 3: Advanced Features (Optional, 2-3 hours)
1. **Learned threshold predictor**
2. **Expert utilization tracking**
3. **Routing pattern visualization**

## Risk Mitigation

### Potential Issues

1. **Gradient Instability**: Dynamic expert count → variable gradient magnitude
   - **Mitigation**: Scale gradients by num_active_experts
   - **Fallback**: Clip gradients within MoE layer

2. **Routing Collapse**: All tokens select same expert
   - **Mitigation**: Load balance loss + dynamic loss
   - **Monitoring**: Track expert utilization per epoch

3. **Threshold Sensitivity**: Performance depends heavily on p
   - **Mitigation**: Start with p=0.4 (proven optimal)
   - **Tuning**: Grid search p ∈ {0.3, 0.4, 0.5}

4. **Training Instability**: Dynamic routing may destabilize early training
   - **Mitigation**: Warmup period with fixed top-2, then switch to dynamic
   - **Alternative**: Gradually decrease p from 0.7 → 0.4 over training

## References

1. Huang et al., "Harder Tasks Need More Experts: Dynamic Routing in MoE Models", 2024
   - https://arxiv.org/abs/2403.07652
   - https://github.com/ZhenweiAn/Dynamic_MoE

2. Zhou et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS 2022

3. Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", 2021

4. Jiang et al., "Mixtral of Experts", 2024

