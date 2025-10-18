# Mixture of Experts (MoE) Implementation Plan

## Research Summary

### Key Papers (2020-2024)

1. **Switch Transformers** (Fedus et al., Google, 2021)
   - Simplified MoE with top-1 routing (single expert per token)
   - Load balancing via auxiliary loss
   - Scales to 1.6T parameters with sparse activation
   - Key insight: Simplicity works - top-1 is sufficient

2. **Mixtral 8x7B** (Mistral AI, 2024)
   - Top-2 routing (2 experts per token)
   - 8 experts, each 7B parameters
   - Outperforms Llama 2 70B while using 5x less compute
   - Production-proven architecture

3. **Expert Choice Routing** (Zhou et al., Google, 2022)
   - Inverts routing: experts choose tokens (not tokens choose experts)
   - Perfect load balancing by design
   - Better gradient flow

4. **ST-MoE** (Zoph et al., Google, 2022)
   - Stability techniques for MoE training
   - Router z-loss for preventing routing collapse
   - Gradient clipping specific to MoE components

### Design Decisions

**1. Number of Experts: 4**
- Rationale: Maintain parameter count while increasing capacity
- Current FFN: 128×256 + 256×128 = 65,536 params (SwiGLU: 3×65,536 = 196,608)
- With 4 experts: Each expert = 128×64 + 64×128 = 16,384 params (SwiGLU: 3×16,384 = 49,152)
- Total MoE: 4×49,152 + router = 196,608 + ~512 = 197,120 params (≈same as current)

**2. Routing Strategy: Top-2**
- More stable than top-1 (Switch Transformers)
- Proven in production (Mixtral)
- Better gradient flow than top-1
- Load balancing easier than top-k (k>2)

**3. Integration Point: Replace SwiGLU/FeedForward layers**
- MoE operates on FFN layer (not attention)
- Complementary to MoH (which operates on attention)
- Each transformer layer: Attention (with MoH) → Norm → MoE → Norm

**4. Load Balancing: Auxiliary Loss + Router Z-Loss**
- Auxiliary loss: Encourages uniform expert utilization
- Router z-loss: Prevents routing collapse (all tokens → same expert)
- Combined weight: 0.01 (standard in literature)

**5. Expert Architecture: SwiGLU-based**
- Each expert is a smaller SwiGLU network
- Maintains current activation function benefits
- Better gradient flow than ReLU-based experts

## Architecture Design

### MoE Layer Structure

```
Input (seq_len, embedding_dim)
  ↓
Router Network (learned gating)
  ├─> Expert Selection (Top-2 per token)
  ├─> Routing Weights (softmax over selected experts)
  └─> Load Balance Loss
  ↓
Expert Networks (4× SwiGLU)
  ├─> Expert 0: SwiGLU(128, 64)
  ├─> Expert 1: SwiGLU(128, 64)
  ├─> Expert 2: SwiGLU(128, 64)
  └─> Expert 3: SwiGLU(128, 64)
  ↓
Weighted Combination
  ↓
Residual Connection
  ↓
Output (seq_len, embedding_dim)
```

### Router Network

```rust
Router {
    w_gate: Array2<f32>,  // (embedding_dim, num_experts)
    optimizer: Adam,
}

fn route(input: &Array2<f32>) -> (Vec<usize>, Vec<f32>, f32) {
    // 1. Compute logits: input @ w_gate
    let logits = input.dot(&self.w_gate);
    
    // 2. Top-2 selection per token
    let (expert_indices, expert_weights) = top_k_routing(&logits, 2);
    
    // 3. Compute auxiliary losses
    let load_balance_loss = compute_load_balance_loss(&logits);
    let router_z_loss = compute_router_z_loss(&logits);
    
    (expert_indices, expert_weights, load_balance_loss + router_z_loss)
}
```

### Load Balance Loss

```
L_balance = α × Σ(i=1 to N) f_i × P_i

where:
- f_i = fraction of tokens routed to expert i
- P_i = average routing probability for expert i
- α = 0.01 (load balance weight)
```

### Router Z-Loss

```
L_z = β × Σ(tokens) log²(Σ(experts) exp(logit_i))

where:
- β = 0.001 (router z-loss weight)
- Encourages router logits to stay small
- Prevents numerical instability
```

## Implementation Plan

### Phase 1: Core MoE Components (src/moe.rs)

1. **Expert Network** (~100 lines)
   - Wrapper around SwiGLU with smaller hidden_dim
   - Forward/backward passes
   - Parameter counting

2. **Router Network** (~150 lines)
   - Gating network with learned weights
   - Top-k routing logic
   - Auxiliary loss computation
   - Gradient computation for router

3. **MoE Layer** (~200 lines)
   - Combines router + experts
   - Token dispatching to experts
   - Weighted combination of expert outputs
   - Residual connection
   - Full backward pass

### Phase 2: Integration with Existing Architecture

1. **Update LayerEnum** (src/llm.rs)
   - Add `MoE(Box<MoELayer>)` variant
   - Implement Layer trait for MoE

2. **Update ModelBuilder** (src/model_builder.rs)
   - Add `use_moe` configuration flag
   - Replace SwiGLU/FeedForward with MoE when enabled
   - Maintain parameter count balance

3. **Update ModelConfig** (src/model_config.rs)
   - Add MoE configuration fields:
     * `use_moe: bool`
     * `num_experts: usize`
     * `num_active_experts: usize` (k in top-k)
     * `expert_hidden_dim: usize`
     * `load_balance_weight: f32`

### Phase 3: Training Loop Integration

1. **Auxiliary Loss Tracking** (src/llm.rs)
   - Collect MoE auxiliary losses during forward pass
   - Add to total loss: `total_loss = task_loss + moe_aux_loss`
   - Log expert utilization metrics

2. **Gradient Monitoring** (src/llm.rs)
   - Track router gradient norms
   - Track expert gradient norms
   - Ensure bidirectional LARS applies to MoE components

### Phase 4: Validation and Optimization

1. **Parameter Count Verification**
   - Ensure total params ≈ current (±2%)
   - Document parameter distribution

2. **Training Stability**
   - Monitor loss convergence
   - Check gradient flow through router and experts
   - Verify load balancing (expert utilization should be ~25% each)

3. **Performance Metrics**
   - Compare final loss vs baseline
   - Measure expert specialization
   - Track routing patterns

## Expected Outcomes

### Parameter Count (Target: ~573K)

**Current (3 layers, SwiGLU):**
- Embeddings: 78,464
- 3× (Attention + Norm + SwiGLU + Norm): 3×(49,152 + 256 + 196,608 + 256) = 738,816
- Final Norm: 256
- Output Projection: 68,224
- **Total: ~886K**

**With MoE (3 layers, 4 experts, top-2):**
- Embeddings: 78,464
- 3× (Attention + Norm + MoE + Norm): 3×(49,152 + 256 + 197,120 + 256) = 741,168
- Final Norm: 256
- Output Projection: 68,224
- **Total: ~888K** (≈same)

### Training Metrics (Target)

- Final loss: ≤ 0.317 (maintain or improve)
- Gradient stability: L0-L2 < 20, balanced flow
- Expert utilization: 20-30% per expert (balanced)
- Router gradient norm: < 10 (stable routing)

### Quality Metrics

- Output coherence: Maintain current quality
- Expert specialization: Measure via routing patterns
- Inference efficiency: Potential 2x speedup (only 2/4 experts active)

## Risk Mitigation

1. **Routing Collapse**: Auxiliary losses + monitoring
2. **Gradient Instability**: Bidirectional LARS + router-specific clipping
3. **Load Imbalance**: Tunable load balance weight
4. **Quality Degradation**: Gradual rollout, A/B testing

## References

1. Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models", 2021
2. Jiang et al., "Mixtral of Experts", 2024
3. Zhou et al., "Mixture-of-Experts with Expert Choice Routing", 2022
4. Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022
5. Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer", 2017

