# Learned Adaptive Temperature Investigation

## Executive Summary

**Recommendation**: **Implement per-token learned temperature** as it offers the best balance of flexibility, theoretical soundness, and practical benefits.

**Key Findings**:
- ✅ **Feasible**: Pattern already exists (complexity/threshold predictors)
- ✅ **Beneficial**: Could improve loss convergence and gradient stability
- ✅ **Low Risk**: Can be implemented with safeguards (bounds, initialization)
- ✅ **Theoretically Sound**: Temperature should adapt to input complexity

---

## Current State Analysis

### Temperature in Soft Routing

**Current Implementation** (`src/head_router.rs:1362`):
```rust
let gating = 1.0 / (1.0 + (-(cumulative_prob - token_threshold) * self.temperature).exp());
```

**Current Value**: `temperature: 5.0` (hardcoded, line 1281)

**Role**: Controls sharpness of sigmoid gating
- **High temperature (>5)**: Sharp transitions → more discrete-like → stronger head selection
- **Low temperature (<5)**: Smooth transitions → more continuous → softer head selection

**Current Results**:
- Loss: 0.301 (excellent)
- Gradient norm: 18.1 (stable but high)
- Output: Perfect
- Efficiency: 46% gain

---

## Design Options

### Option 1: Per-Token Learned Temperature (RECOMMENDED)

**Architecture**:
```rust
// Add to FullyAdaptiveHeadRouter
w_temperature: Array2<f32>,  // (embedding_dim, 1)
temperature_bias: f32,
temperature_optimizer: Adam,

// Predict per-token temperature
let temp_logits = input.dot(&self.w_temperature).into_shape(seq_len).unwrap();
let temperatures = temp_logits.mapv(|x| {
    let sigmoid = 1.0 / (1.0 + (-(x + self.temperature_bias)).exp());
    1.0 + sigmoid * 9.0  // Map [0, 1] → [1.0, 10.0]
});

// Use per-token temperature in gating
let gating = 1.0 / (1.0 + (-(cumulative_prob - token_threshold) * temperatures[token_idx]).exp());
```

**Advantages**:
- ✅ Maximum flexibility: Each token gets optimal temperature
- ✅ Consistent with existing design (complexity/threshold are per-token)
- ✅ Theoretically sound: Complex inputs may need different sharpness
- ✅ Differentiable: Gradients flow through temperature prediction

**Disadvantages**:
- ⚠️ More parameters: +embedding_dim parameters
- ⚠️ Slightly more computation: One extra matrix multiplication

**Expected Benefits**:
- Simple inputs → lower temperature → smoother routing → better gradient flow
- Complex inputs → higher temperature → sharper routing → clearer head selection
- Could reduce gradient norm from 18 → closer to target 2.5

---

### Option 2: Per-Layer Learned Temperature

**Architecture**:
```rust
// Add to FullyAdaptiveHeadRouter
temperature: f32,  // Learned parameter (not hardcoded)
temperature_optimizer: Adam,

// In backward pass, compute temperature gradient
let temp_grad = compute_temperature_gradient();
self.temperature_optimizer.step_scalar(&mut self.temperature, temp_grad, lr);
```

**Advantages**:
- ✅ Simple: Only 1 parameter per router (per layer)
- ✅ Low overhead: No extra forward computation
- ✅ Layer-wise adaptation: Early layers may need different temp than late layers

**Disadvantages**:
- ⚠️ Less flexible: All tokens in a layer share same temperature
- ⚠️ Requires scalar optimizer: Need to implement scalar Adam step
- ⚠️ Less theoretically motivated: Why should all tokens have same temperature?

---

### Option 3: Global Learned Temperature

**Architecture**:
```rust
// Single global temperature shared across all layers
static mut GLOBAL_TEMPERATURE: f32 = 5.0;
```

**Advantages**:
- ✅ Simplest: Only 1 parameter total
- ✅ Minimal overhead

**Disadvantages**:
- ❌ Too inflexible: All layers and tokens share same temperature
- ❌ Not recommended: Loses layer-wise and token-wise adaptation

---

## Implementation Plan (Option 1: Per-Token)

### Step 1: Add Temperature Predictor Fields

```rust
// In FullyAdaptiveHeadRouter struct (around line 1175)
/// Temperature predictor weights: (embedding_dim, 1)
w_temperature: Array2<f32>,

/// Temperature predictor bias
temperature_bias: f32,

/// Optimizer for temperature predictor
temperature_optimizer: Adam,

/// Cached temperatures for backward pass
#[serde(skip)]
cached_temperatures: Option<Array1<f32>>,
```

### Step 2: Initialize in Constructor

```rust
// In new() method (around line 1268)
let w_temperature = Array2::from_shape_fn((embedding_dim, 1), |(i, _)| {
    rng.gen_range(-0.1..0.1)
});
let temperature_bias = 0.0;

Self {
    // ... existing fields ...
    w_temperature,
    temperature_bias,
    temperature_optimizer: Adam::new((embedding_dim, 1)),
    cached_temperatures: None,
    // Remove: temperature: 5.0,
}
```

### Step 3: Predict Temperature in Forward Pass

```rust
// In route() method, after threshold prediction (around line 1333)
// 3.5. Predict per-token temperatures [1.0, 10.0]
let temp_logits = input.dot(&self.w_temperature).into_shape(seq_len).unwrap();
let temperatures = temp_logits.mapv(|x| {
    let sigmoid = 1.0 / (1.0 + (-(x + self.temperature_bias)).exp());
    1.0 + sigmoid * 9.0  // Map [0, 1] → [1.0, 10.0]
});

// Cache for backward pass
self.cached_temperatures = Some(temperatures.clone());
```

### Step 4: Use Per-Token Temperature in Gating

```rust
// In soft routing loop (around line 1362)
let token_temperature = temperatures[token_idx];
let gating = 1.0 / (1.0 + (-(cumulative_prob - token_threshold) * token_temperature).exp());
```

### Step 5: Compute Temperature Gradients in Backward Pass

```rust
// In backward() method (around line 1563)
// 4. Temperature predictor gradients
let mut temperature_grads = Array2::zeros((self.embedding_dim, 1));
if let Some(temperatures) = &self.cached_temperatures {
    for token_idx in 0..seq_len {
        let temp = temperatures[token_idx];
        let soft_heads: f32 = soft_weights.row(token_idx).sum();
        let target = target_heads[token_idx];
        
        // Gradient heuristic: push temperature to improve head selection
        // If using too many heads (soft_heads > target): increase temp (sharper)
        // If using too few heads (soft_heads < target): decrease temp (smoother)
        let grad_scale = (soft_heads - target) * 0.1;
        
        for dim_idx in 0..self.embedding_dim {
            temperature_grads[[dim_idx, 0]] += grad_scale * input[[token_idx, dim_idx]];
        }
    }
}
temperature_grads /= seq_len as f32;

// Update temperature predictor
self.temperature_optimizer.step(&mut self.w_temperature, &temperature_grads, lr * 0.1);
```

### Step 6: Update num_parameters()

```rust
// In num_parameters() method (around line 1575)
pub fn num_parameters(&self) -> usize {
    self.num_heads * self.embedding_dim  // w_router
        + self.embedding_dim + 1          // w_complexity + bias
        + self.embedding_dim + 1          // w_threshold + bias
        + self.embedding_dim + 1          // w_temperature + bias (NEW)
}
```

---

## Risk Analysis

### Potential Issues

1. **Temperature Collapse**
   - **Risk**: Temperature could collapse to extreme values (0 or ∞)
   - **Mitigation**: Bounded range [1.0, 10.0] via sigmoid mapping
   - **Severity**: Low (bounds prevent collapse)

2. **Training Instability**
   - **Risk**: Adding new learned parameter could destabilize training
   - **Mitigation**: 
     - Initialize near current value (5.0 → sigmoid(0) * 9 + 1 = 5.5)
     - Use slower learning rate (lr * 0.1)
     - Gradual warm-up possible
   - **Severity**: Low (similar to complexity/threshold predictors)

3. **Gradient Noise**
   - **Risk**: Temperature gradients could be noisy
   - **Mitigation**: Adam optimizer smooths gradients
   - **Severity**: Low (Adam handles noise well)

4. **Overfitting**
   - **Risk**: Per-token temperature could overfit to training data
   - **Mitigation**: Small number of parameters (embedding_dim + 1)
   - **Severity**: Low (same as complexity/threshold)

### Safety Measures

1. **Bounded Range**: [1.0, 10.0] prevents extreme values
2. **Slow Learning**: lr * 0.1 for temperature (same as complexity/threshold)
3. **Initialization**: Start near current working value (5.0)
4. **Monitoring**: Log temperature statistics (min, max, avg)

---

## Expected Benefits

### 1. Improved Loss Convergence

**Hypothesis**: Adaptive temperature could improve loss beyond 0.301

**Mechanism**:
- Simple tokens → lower temp → smoother gradients → better optimization
- Complex tokens → higher temp → sharper selection → clearer signal

**Expected**: Loss 0.301 → 0.25-0.28 (10-15% improvement)

### 2. Better Gradient Stability

**Hypothesis**: Per-token temperature could reduce gradient norm

**Mechanism**:
- Current: All tokens use temp=5.0 (one-size-fits-all)
- Learned: Each token gets optimal temp → less gradient conflict

**Expected**: Gradient norm 18.1 → 10-15 (closer to target 2.5)

### 3. Enhanced Adaptivity

**Hypothesis**: Temperature should correlate with complexity

**Mechanism**:
- Low complexity → low temp → smooth routing (fewer heads)
- High complexity → high temp → sharp routing (more heads)

**Expected**: Temperature range [2-8] with complexity correlation

### 4. Layer-Wise Specialization

**Hypothesis**: Different layers may learn different temperature patterns

**Mechanism**:
- Early layers: Process raw features → may need higher temp
- Late layers: Process abstract features → may need lower temp

**Expected**: Layer-wise temperature variation

---

## Testing Plan

### Phase 1: Implementation (1-2 hours)

1. Add temperature predictor fields to `FullyAdaptiveHeadRouter`
2. Implement temperature prediction in `route()`
3. Update gating to use per-token temperature
4. Implement temperature gradients in `backward()`
5. Update `num_parameters()`

### Phase 2: Quick Validation (10 epochs, ~5 minutes)

**Success Criteria**:
- ✅ No compilation errors
- ✅ No runtime errors (NaN, panic)
- ✅ Loss converges (≤ 0.50)
- ✅ Output quality correct (not gibberish)
- ✅ Temperature values reasonable (1.0-10.0)

**Monitoring**:
```rust
// Add to logging
let (avg_temp, min_temp, max_temp) = compute_temperature_stats();
info!("Temp: {:.2} [{:.2}-{:.2}]", avg_temp, min_temp, max_temp);
```

### Phase 3: Full Training (100 epochs, ~30 minutes)

**Success Criteria**:
- ✅ Loss ≤ 0.301 (match or beat current)
- ✅ Gradient norm ≤ 18.1 (match or improve)
- ✅ Output quality perfect
- ✅ Temperature shows meaningful variation

**Analysis**:
- Plot temperature vs complexity correlation
- Compare layer-wise temperature patterns
- Measure loss improvement vs fixed temperature

---

## Theoretical Justification

### Why Per-Token Temperature Makes Sense

1. **Gumbel-Softmax Analogy**
   - Gumbel-Softmax uses temperature to control discreteness
   - Temperature is often annealed during training (high → low)
   - Per-token temperature extends this: each token gets optimal discreteness

2. **Complexity-Temperature Relationship**
   - **Simple inputs**: Low complexity → fewer heads needed → lower temp (smoother)
   - **Complex inputs**: High complexity → more heads needed → higher temp (sharper)
   - This creates a natural coupling between complexity and routing sharpness

3. **Gradient Flow Optimization**
   - Different tokens may have different gradient flow requirements
   - Adaptive temperature allows per-token gradient optimization
   - Could reduce gradient conflicts across tokens

4. **Existing Pattern**
   - Complexity predictor: Per-token complexity scores
   - Threshold predictor: Per-token thresholds
   - Temperature predictor: Per-token temperatures (consistent design)

---

## Alternative: Temperature Annealing

**Simpler Alternative**: Instead of learned temperature, use scheduled annealing

```rust
// Anneal temperature during training
let progress = epoch as f32 / max_epochs as f32;
let temperature = 10.0 - progress * 5.0;  // 10.0 → 5.0 over training
```

**Pros**:
- ✅ No new parameters
- ✅ Simple to implement
- ✅ Proven technique (Gumbel-Softmax)

**Cons**:
- ❌ Not per-token adaptive
- ❌ Fixed schedule (not data-driven)
- ❌ Less flexible than learned

**Recommendation**: Try learned temperature first; fall back to annealing if issues arise

---

## Conclusion

### Final Recommendation: **IMPLEMENT PER-TOKEN LEARNED TEMPERATURE**

**Rationale**:
1. ✅ **Feasible**: Pattern exists (complexity/threshold predictors)
2. ✅ **Low Risk**: Bounded, slow learning, similar to existing predictors
3. ✅ **High Potential**: Could improve loss, gradients, and adaptivity
4. ✅ **Theoretically Sound**: Temperature should adapt to input complexity
5. ✅ **Consistent Design**: Matches existing per-token prediction pattern

**Expected Outcomes**:
- **Best Case**: Loss 0.25-0.28, gradient norm 10-15, enhanced adaptivity
- **Likely Case**: Loss 0.28-0.30, gradient norm 15-18, meaningful temperature variation
- **Worst Case**: Loss 0.30-0.32, gradient norm 18-20, revert to fixed temperature

**Next Steps**:
1. Implement per-token temperature predictor
2. Run 10-epoch quick test to validate stability
3. Run 100-epoch full training to measure benefits
4. Analyze temperature patterns and correlations
5. Compare results vs fixed temperature baseline

**Fallback Plan**:
- If learned temperature causes instability → revert to fixed temperature
- If no benefit observed → consider temperature annealing instead
- If gradient issues persist → investigate other causes (learning rate, architecture)

