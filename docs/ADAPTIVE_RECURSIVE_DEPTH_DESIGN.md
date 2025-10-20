# Adaptive Recursive Depth for TRM: Design Document

**Date**: 2025-10-20  
**Status**: Phase 1 - Research and Design Complete  
**Objective**: Design mechanism for TRM to dynamically learn optimal recursive depth per input

---

## Executive Summary

**Proposal**: Implement **Adaptive Computation Time (ACT)** mechanism for TRM to dynamically adjust recursive depth based on input complexity.

**Key Benefits**:
- ✅ **Efficiency**: Simple inputs use fewer steps (e.g., 2-3 instead of 5)
- ✅ **Quality**: Complex inputs can use more steps (e.g., 7-10 instead of 5)
- ✅ **Adaptivity**: Per-sequence depth based on learned complexity
- ✅ **Differentiability**: Maintains excellent gradient flow (current: 2.13)

**Recommended Approach**: **Option A - Halting Predictor with ACT**

---

## Phase 1: Research Analysis

### 1.1 Existing Approaches

#### **Universal Transformers (Dehghani et al., 2018) - ACT**

**Mechanism**:
```python
# At each recursive step t:
p_halt[t] = sigmoid(W_halt · hidden_state[t] + b_halt)
cumulative_p[t] = sum(p_halt[0:t+1])

# Stop when cumulative probability exceeds threshold
if cumulative_p[t] >= threshold (e.g., 0.95):
    stop_recursion()

# Ponder loss (auxiliary): Penalize excessive depth
ponder_loss = (actual_steps / max_steps) * ponder_weight
```

**Gradient Flow**:
- Uses **weighted sum** of all steps for differentiability
- Each step contributes proportionally to its halting probability
- All steps must be computed during training (for gradients)
- At inference: Can actually stop early (no gradient needed)

**Pros**:
- ✅ Proven approach (used in Universal Transformers)
- ✅ Fully differentiable
- ✅ Simple to implement
- ✅ Works well with recursive architectures

**Cons**:
- ⚠️ Training requires computing all steps (even if halting early)
- ⚠️ Inference can stop early (efficiency gain only at inference)

#### **PonderNet (Banino et al., 2021)**

**Mechanism**:
```python
# Learn probability distribution over number of steps
p(n_steps) = geometric_distribution(lambda_learned)

# Regularization: KL divergence with prior
kl_loss = KL(p_learned || p_prior)

# More principled probabilistic framework
```

**Pros**:
- ✅ More principled probabilistic approach
- ✅ Better theoretical foundation
- ✅ Explicit distribution over depths

**Cons**:
- ⚠️ More complex to implement
- ⚠️ Requires geometric prior design
- ⚠️ May be overkill for TRM

#### **Early Exit Mechanisms**

**Mechanism**:
```python
# Add classifier at each step
confidence[t] = classifier(hidden_state[t])

# Exit when confidence exceeds threshold
if confidence[t] > threshold:
    return output[t]
```

**Pros**:
- ✅ Simple conceptually
- ✅ True early stopping (efficiency at training and inference)

**Cons**:
- ❌ Discrete decision breaks gradient flow
- ❌ Requires straight-through estimators (gradient approximation)
- ❌ Less stable training

### 1.2 Recommendation: ACT-based Halting Predictor

**Chosen Approach**: **Option A - Halting Predictor with ACT**

**Rationale**:
1. ✅ **Proven**: Used successfully in Universal Transformers
2. ✅ **Simple**: Minimal code changes to TRM
3. ✅ **Differentiable**: Maintains gradient flow quality
4. ✅ **Compatible**: Works seamlessly with existing MoH mechanism
5. ✅ **Stable**: No discrete decisions, smooth optimization

**Rejected Alternatives**:
- ❌ **Option B (Depth Predictor)**: Less adaptive, predicts depth upfront without seeing intermediate states
- ❌ **Option C (Per-Token Adaptive)**: Too complex, batching challenges, may not be needed for sequence-level tasks

---

## Phase 1: Design Specification

### 2.1 Architecture Changes

#### **Add to `TinyRecursiveModel` struct** (`src/trm.rs`):

```rust
pub struct TinyRecursiveModel {
    // ... existing fields ...
    
    // Adaptive depth components
    adaptive_depth_enabled: bool,
    max_recursive_depth: usize,  // Maximum allowed depth (e.g., 10)
    halt_threshold: f32,         // Cumulative probability threshold (e.g., 0.95)
    ponder_loss_weight: f32,     // Weight for ponder loss (e.g., 0.01)
    
    // Halting predictor parameters
    w_halt: Array2<f32>,         // (embedding_dim, 1)
    b_halt: f32,                 // Scalar bias
    halt_optimizer: Adam,        // Optimizer for halting predictor
    
    // Statistics tracking
    actual_depths: Vec<usize>,   // Track actual depths used per sequence
    avg_depth: f32,              // Running average depth
}
```

#### **Constructor Changes**:

```rust
pub fn new(
    embedding_dim: usize,
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: Option<usize>,
    recursive_depth: usize,      // Now becomes max_recursive_depth if adaptive
    use_swiglu: bool,
    max_seq_len: usize,
    head_selection: HeadSelectionStrategy,
    adaptive_depth_config: Option<AdaptiveDepthConfig>,  // NEW
) -> Self {
    // ... existing initialization ...
    
    let (adaptive_enabled, max_depth, halt_thresh, ponder_weight, w_halt, b_halt, halt_opt) = 
        if let Some(config) = adaptive_depth_config {
            let w = Array2::from_shape_fn((embedding_dim, 1), |_| 
                rng.gen_range(-0.01..0.01));
            let b = 0.0;  // Initialize bias to 0 (sigmoid(0) = 0.5)
            let opt = Adam::new((embedding_dim, 1));
            (true, config.max_depth, config.halt_threshold, config.ponder_weight, w, b, opt)
        } else {
            // Fixed depth mode (current behavior)
            let w = Array2::zeros((embedding_dim, 1));
            let b = 0.0;
            let opt = Adam::new((embedding_dim, 1));
            (false, recursive_depth, 0.95, 0.0, w, b, opt)
        };
    
    Self {
        // ... existing fields ...
        adaptive_depth_enabled: adaptive_enabled,
        max_recursive_depth: max_depth,
        halt_threshold: halt_thresh,
        ponder_loss_weight: ponder_weight,
        w_halt: w,
        b_halt: b,
        halt_optimizer: halt_opt,
        actual_depths: Vec::new(),
        avg_depth: 0.0,
    }
}
```

#### **Configuration Struct**:

```rust
#[derive(Clone, Debug)]
pub struct AdaptiveDepthConfig {
    pub max_depth: usize,           // Maximum recursive depth (e.g., 10)
    pub halt_threshold: f32,        // Cumulative p threshold (e.g., 0.95)
    pub ponder_weight: f32,         // Ponder loss weight (e.g., 0.01)
}
```

### 2.2 Forward Pass Changes

#### **Modified `forward()` method**:

```rust
pub fn forward(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
    let (batch_size, seq_len, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    
    // Initialize state
    let mut x = input.clone();
    
    // Track halting probabilities and cumulative probabilities
    let mut halt_probs = Vec::new();      // p_halt at each step
    let mut cumulative_probs = Array1::zeros(batch_size);  // Per-sequence cumulative p
    let mut active_mask = Array1::from_elem(batch_size, true);  // Which sequences are still active
    
    // Determine actual recursive depth
    let actual_depth = if self.adaptive_depth_enabled {
        self.max_recursive_depth
    } else {
        self.recursive_depth  // Fixed depth (current behavior)
    };
    
    // Recursive loop
    for step in 0..actual_depth {
        // Check if all sequences have halted
        if self.adaptive_depth_enabled && !active_mask.iter().any(|&a| a) {
            break;  // All sequences halted, stop early
        }
        
        // Attention sublayer (Pre-LN)
        let normed = self.norm1.forward(&x)?;
        let attn_out = self.attention.forward(&normed)?;
        let attn_scale = self.attention_step_scales[step];
        x = &x + &(attn_out * attn_scale);
        
        // FFN sublayer (Pre-LN)
        let normed = self.norm2.forward(&x)?;
        let ffn_out = if self.use_swiglu {
            self.feed_forward_swiglu.as_ref().unwrap().forward(&normed)?
        } else {
            self.feed_forward_standard.as_ref().unwrap().forward(&normed)?
        };
        let ffn_scale = self.ffn_step_scales[step];
        x = &x + &(ffn_out * ffn_scale);
        
        // Compute halting probability (if adaptive depth enabled)
        if self.adaptive_depth_enabled {
            // Pool sequence dimension: mean over seq_len
            let pooled = x.mean_axis(Axis(1)).unwrap();  // (batch_size, embedding_dim)
            
            // Compute halting logits: W_halt · pooled + b_halt
            let halt_logits = pooled.dot(&self.w_halt).into_shape(batch_size).unwrap();
            let halt_logits = halt_logits.mapv(|x| x + self.b_halt);
            
            // Apply sigmoid: p_halt = sigmoid(logits)
            let p_halt = halt_logits.mapv(|x| 1.0 / (1.0 + (-x).exp()));
            
            // Update cumulative probabilities (only for active sequences)
            for i in 0..batch_size {
                if active_mask[i] {
                    cumulative_probs[i] += p_halt[i];
                    
                    // Check if sequence should halt
                    if cumulative_probs[i] >= self.halt_threshold {
                        active_mask[i] = false;
                    }
                }
            }
            
            halt_probs.push(p_halt);
        }
        
        // Cache states for backward pass
        self.cached_states.push(x.clone());
        // ... cache attention and FFN outputs ...
    }
    
    // Track actual depth used (for statistics)
    if self.adaptive_depth_enabled {
        let depths: Vec<usize> = active_mask.iter()
            .enumerate()
            .map(|(i, &active)| {
                if active {
                    actual_depth  // Reached max depth
                } else {
                    // Find first step where cumulative_p >= threshold
                    halt_probs.iter()
                        .scan(0.0, |cum, p| {
                            *cum += p[i];
                            Some(*cum)
                        })
                        .position(|cum| cum >= self.halt_threshold)
                        .unwrap_or(actual_depth) + 1
                }
            })
            .collect();
        
        self.actual_depths = depths;
        self.avg_depth = self.actual_depths.iter().sum::<usize>() as f32 / batch_size as f32;
    }
    
    Ok(x)
}
```

### 2.3 Backward Pass Changes

#### **Modified `backward()` method**:

```rust
pub fn backward(&mut self, grad_output: &Array2<f32>, learning_rate: f32) -> Result<()> {
    // ... existing backward pass logic ...
    
    // Compute ponder loss gradient (if adaptive depth enabled)
    if self.adaptive_depth_enabled {
        // Ponder loss: L_ponder = (avg_depth / max_depth) * ponder_weight
        // Gradient: dL/d(avg_depth) = ponder_weight / max_depth
        
        let ponder_grad_scale = self.ponder_loss_weight / self.max_recursive_depth as f32;
        
        // Backpropagate through halting predictor
        // For each step, compute gradient of halting probability
        // This is complex - simplified version:
        
        let mut w_halt_grad = Array2::zeros(self.w_halt.dim());
        let mut b_halt_grad = 0.0;
        
        for (step, halt_prob) in self.cached_halt_probs.iter().enumerate() {
            // Gradient of sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            let sigmoid_grad = halt_prob.mapv(|p| p * (1.0 - p));
            
            // Gradient from ponder loss
            let ponder_contrib = sigmoid_grad.mapv(|g| g * ponder_grad_scale);
            
            // Accumulate gradients
            let pooled_state = self.cached_pooled_states[step].clone();
            for i in 0..pooled_state.shape()[0] {
                let state_vec = pooled_state.row(i);
                let grad_scalar = ponder_contrib[i];
                
                for j in 0..state_vec.len() {
                    w_halt_grad[[j, 0]] += state_vec[j] * grad_scalar;
                }
                b_halt_grad += grad_scalar;
            }
        }
        
        // Update halting predictor parameters
        let w_halt_grad_2d = w_halt_grad;
        self.halt_optimizer.step(&mut self.w_halt, &w_halt_grad_2d, learning_rate);
        self.b_halt -= learning_rate * b_halt_grad;
    }
    
    Ok(())
}
```

### 2.4 Auxiliary Loss

#### **Ponder Loss Computation**:

```rust
pub fn get_ponder_loss(&self) -> f32 {
    if self.adaptive_depth_enabled && !self.actual_depths.is_empty() {
        // Ponder loss: Penalize average depth to encourage efficiency
        // L_ponder = (avg_depth / max_depth) * ponder_weight
        (self.avg_depth / self.max_recursive_depth as f32) * self.ponder_loss_weight
    } else {
        0.0
    }
}
```

#### **Integration into Training Loop** (`src/llm.rs`):

```rust
// Add ponder loss from TRM blocks
for layer in &self.network {
    if let Some(trm_block) = layer.as_trm_block() {
        let ponder_loss = trm_block.get_ponder_loss();
        batch_loss += ponder_loss;
    }
}
```

---

## Phase 1: Gradient Flow Analysis

### 3.1 Gradient Flow Through Variable Depth

**Challenge**: How to backpropagate through variable-length computation?

**Solution**: **Weighted Sum Approach** (from ACT paper)

```rust
// During forward pass, compute weighted output
let mut weighted_output = Array2::zeros(output_shape);
let mut remainder_weights = Array1::ones(batch_size);

for step in 0..max_depth {
    let step_output = compute_step(input);
    let halt_prob = compute_halt_prob(step_output);
    
    // Weight this step's output by its halting probability
    let step_weight = halt_prob.min(&remainder_weights);  // Element-wise min
    weighted_output += &(step_output * step_weight);
    
    // Update remainder
    remainder_weights -= &step_weight;
    
    // Stop if all remainder is consumed
    if remainder_weights.sum() < 1e-6 {
        break;
    }
}

// Final output is weighted sum of all steps
return weighted_output;
```

**Gradient Flow**:
- Each step contributes to final output proportionally to its weight
- Gradients flow back through all computed steps
- Halting predictor receives gradients from all steps it influenced
- **Fully differentiable** - no discrete decisions

### 3.2 Batching with Different Depths

**Challenge**: Different sequences may need different depths. How to batch efficiently?

**Solution**: **Compute all steps, mask inactive sequences**

```rust
// All sequences compute all steps (up to max_depth)
// Use active_mask to zero out gradients for halted sequences

for step in 0..max_depth {
    // Compute step for all sequences
    let step_output = compute_step(input);
    
    // Mask inactive sequences (already halted)
    let masked_output = step_output * active_mask.broadcast();
    
    // Update state
    input = masked_output;
}
```

**Trade-off**:
- ✅ Simple batching (all sequences same shape)
- ✅ Efficient GPU utilization (no dynamic shapes)
- ⚠️ Training computes all steps (efficiency gain only at inference)
- ✅ Inference can stop early (true efficiency gain)

---

## Phase 1: Expected Outcomes

### 4.1 Depth Distribution Hypothesis

**Simple Inputs** (e.g., "What is 2+2?"):
- Expected depth: 2-3 steps
- Rationale: Simple pattern matching, no complex reasoning needed

**Complex Inputs** (e.g., "Explain quantum entanglement"):
- Expected depth: 7-10 steps
- Rationale: Requires multi-step reasoning, refinement

**Correlation with MoH Complexity**:
- Hypothesis: Adaptive depth should correlate with MoH complexity predictor
- Low complexity → fewer steps
- High complexity → more steps

### 4.2 Performance Predictions

**Baseline (Fixed Depth D=5)**:
- Loss: 0.568
- Avg depth: 5.0 (always)
- Output quality: Perfect

**Adaptive Depth (Max D=7, ponder_weight=0.01)**:
- Loss: 0.50-0.55 (expected improvement)
- Avg depth: 4.5-5.5 (should vary)
- Output quality: Perfect or better
- Efficiency: 10-20% fewer steps on average

**Adaptive Depth (Max D=10, ponder_weight=0.05)**:
- Loss: 0.45-0.52 (expected improvement)
- Avg depth: 5.0-7.0 (should vary more)
- Output quality: Perfect or better
- Efficiency: May use more steps for complex inputs

---

## Phase 1: Implementation Plan

### 5.1 Code Changes Summary

**Files to Modify**:
1. `src/trm.rs` - Add adaptive depth mechanism
2. `src/model_config.rs` - Add `AdaptiveDepthConfig`
3. `src/model_builder.rs` - Pass adaptive depth config to TRM
4. `src/llm.rs` - Collect ponder loss, log depth statistics
5. `src/main.rs` - Configure adaptive depth experiments

**Estimated Lines of Code**:
- `src/trm.rs`: +150 lines (halting predictor, forward/backward changes)
- `src/model_config.rs`: +10 lines (config struct)
- `src/model_builder.rs`: +5 lines (pass config)
- `src/llm.rs`: +30 lines (ponder loss, statistics)
- `src/main.rs`: +20 lines (configuration)
- **Total**: ~215 lines

### 5.2 Testing Strategy

**Unit Tests**:
- Test halting predictor forward pass
- Test gradient computation for halting predictor
- Test ponder loss calculation
- Test depth tracking statistics

**Integration Tests**:
- Test TRM with adaptive depth enabled
- Test TRM with adaptive depth disabled (backward compatibility)
- Test batching with different depths

**Validation Experiments**:
- Baseline: Fixed depth D=5 (100 epochs)
- Experiment 1: Adaptive max_depth=7, ponder=0.01 (100 epochs)
- Experiment 2: Adaptive max_depth=10, ponder=0.05 (100 epochs)

---

## Phase 1: Success Criteria

✅ **Design Complete**:
- [x] Analyzed existing approaches (ACT, PonderNet, early exit)
- [x] Chose optimal mechanism (ACT-based halting predictor)
- [x] Designed architecture changes (halting predictor, forward/backward)
- [x] Solved gradient flow challenges (weighted sum, masking)
- [x] Planned implementation (code changes, testing strategy)

**Ready for Phase 2**: Implementation

---

## Next Steps

1. **Phase 2: Implementation** (2-3 hours)
   - Implement halting predictor in `src/trm.rs`
   - Add configuration structs
   - Update training loop for ponder loss
   - Add depth statistics logging

2. **Phase 3: Validation** (1-2 hours)
   - Run baseline experiment (fixed depth D=5)
   - Run adaptive depth experiments (max_depth=7, 10)
   - Analyze depth distributions
   - Compare performance metrics

3. **Phase 4: Documentation** (30 minutes)
   - Document results in `docs/ADAPTIVE_RECURSIVE_DEPTH_RESULTS.md`
   - Update README with adaptive depth feature
   - Create visualization of depth distributions

