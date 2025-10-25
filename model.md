# Model Architecture and Algorithms

This document provides mathematical descriptions of all algorithms implemented in the codebase, organized by forward and backward passes.

## Forward Pass

### Embeddings

**Token Embeddings:**
```
E ∈ ℝ^{V × D}
e_i = E[token_i] ∈ ℝ^D
```

**Positional Embeddings (CoPE):**
```
P ∈ ℝ^{M × D}
p_i = P[pos_i] ∈ ℝ^D
```

**Combined Embedding:**
```
x_i = e_i + p_i ∈ ℝ^D
```

### DynamicTanhNorm

**Normalization:**
```
y = tanh(α · x) ⊙ γ + β
```
where:
- α ∈ ℝ (learnable nonlinearity scale)
- γ ∈ ℝ^D (per-feature scale)
- β ∈ ℝ^D (per-feature bias)

### PolyAttention

**Polynomial Attention (Degree p=3):**

**Query/Key/Value Projections:**
```
Q = X · W_Q ∈ ℝ^{N × D_h}
K = X · W_K ∈ ℝ^{N × D_h}
V = X · W_V ∈ ℝ^{N × D_h}
```

**Attention Scores with CoPE:**
```
s_{ij} = (q_i · k_j) / √D_h + q_i · p_{i-j}  (for j ≤ i, sliding window)
```

**Polynomial Activation:**
```
φ(s) = scale · (a · s^p + b)
```

**Gated Attention (Mixture-of-Heads):**
```
g_h = φ_poly(α_g · (X · w_g) + β_g) ∈ ℝ^{N × 1}
m_h = sigmoid(α_τ · (X · w_τ) + β_τ) ∈ ℝ^{N × 1}
eff_h = g_h · m_h
```

**Head Output:**
```
y_h = ∑_{j=0}^{i} φ(s_{ij}) · v_j · eff_h[i]
```

**Multi-Head Concatenation:**
```
Y = concat([y_1, ..., y_H]) · W_O + X
```

### SwiGLU

**Gated Linear Unit:**
```
x1 = X · W1 ∈ ℝ^{N × D_hidden}
x2 = X · W2 ∈ ℝ^{N × D_hidden}
swish = x1 ⊙ φ_poly(α_swish · x1) ∈ ℝ^{N × D_hidden}
gate = φ_poly(α_gate · x2 + β_gate) ∈ ℝ^{N × D_hidden}
gated = swish ⊙ gate ∈ ℝ^{N × D_hidden}
y = gated · W_out + X ∈ ℝ^{N × D}
```

### Output Projection

**Logits:**
```
logits = Y · W_out ∈ ℝ^{N × V}
```

### Softmax and Sampling

**Probability Distribution:**
```
p_i = softmax(logits_i) = exp(logits_i) / ∑_j exp(logits_j)
```

**Greedy Decoding:**
```
next_token = argmax(p)
```

## Backward Pass

### Cross-Entropy Loss

**Loss:**
```
L = -∑_{i=1}^N log(p_{i,target_i})
```

**Gradient w.r.t. logits:**
```
∂L/∂logits_{i,j} = p_{i,j} - δ_{j,target_i}
```

### Output Projection

**Gradients:**
```
∂L/∂W_out = Y^T · ∂L/∂logits
∂L/∂Y = ∂L/∂logits · W_out^T
```

### SwiGLU

**Gradient Flow:**
```
∂L/∂gated = ∂L/∂y · W_out^T
∂L/∂swish = ∂L/∂gated ⊙ gate
∂L/∂gate = ∂L/∂gated ⊙ swish
∂L/∂x1 = ∂L/∂swish ⊙ φ_poly'(α_swish · x1) ⊙ α_swish
∂L/∂x2 = ∂L/∂gate ⊙ φ_poly'(α_gate · x2 + β_gate) ⊙ α_gate
∂L/∂W_out = gated^T · ∂L/∂y
∂L/∂W1 = X^T · ∂L/∂x1
∂L/∂W2 = X^T · ∂L/∂x2
```

**Polynomial Gate Gradients:**
```
∂L/∂w_poly = ∑ ∂L/∂φ · (c·z)^k for k in weights
```

### PolyAttention

**Attention Gradients:**
```
∂L/∂φ_{ij} = ∂L/∂y_h[i] · v_j · eff_h[i]
∂L/∂s_{ij} = ∂L/∂φ_{ij} · scale · a · p · s^{p-1}
∂L/∂q_i += ∑_j ∂L/∂s_{ij} · k_j / √D_h
∂L/∂k_j += ∑_i ∂L/∂s_{ij} · q_i / √D_h
∂L/∂v_j += φ(s_{ij}) · ∂L/∂y_h[i] · eff_h[i]
```

**Gating Gradients:**
```
∂L/∂g_h = ∂L/∂y_h ⊙ y_pre_h ⊙ m_h
∂L/∂m_h = ∂L/∂y_h ⊙ y_pre_h ⊙ g_h
∂L/∂z_h = ∂L/∂g_h ⊙ φ_poly'(z_h)
∂L/∂w_g += X^T · ∂L/∂z_h ⊙ α_g
∂L/∂α_g += (∂L/∂z_h ⊙ X·w_g).sum()
∂L/∂β_g += ∂L/∂z_h.sum()
```

**Threshold Gradients (MoH):**
```
∂L/∂τ = ∂L/∂m_h ⊙ sigmoid'(α_τ·y + β_τ)
∂L/∂w_τ += X^T · ∂L/∂τ ⊙ sigmoid'(α_τ·y + β_τ) ⊙ α_τ
∂L/∂α_τ += ∂L/∂τ ⊙ sigmoid'(α_τ·y + β_τ) ⊙ y
∂L/∂β_τ += ∂L/∂τ ⊙ sigmoid'(α_τ·y + β_τ)
```

### DynamicTanhNorm

**Gradients:**
```
∂L/∂α = ∑ ∂L/∂y ⊙ sech²(α·x) ⊙ x ⊙ γ
∂L/∂γ = ∂L/∂y ⊙ tanh(α·x)
∂L/∂β = ∂L/∂y
∂L/∂x = ∂L/∂y ⊙ γ ⊙ sech²(α·x) ⊙ α
```

### Embeddings

**Token Gradients:**
```
∂L/∂E[token_i] += ∂L/∂x_i
```

**Positional Gradients:**
```
∂L/∂P[pos_i] += ∂L/∂x_i
```

## Gradient Instability Analysis

### Potential Instability Sources

1. **Polynomial Attention:**
   - High-degree polynomials (p=3) can cause gradient explosion in attention scores
   - CoPE positional encoding adds unbounded terms to attention logits
   - Mixture-of-Heads gating introduces additional nonlinearity

2. **SwiGLU Gates:**
   - Polynomial approximations to sigmoid may not be numerically stable
   - Learned polynomial weights can diverge during training

3. **DynamicTanhNorm:**
   - Learnable α parameter can cause tanh saturation or explosion
   - Per-feature γ/β parameters may lead to feature collapse

4. **Sliding Window Attention:**
   - Abrupt attention cutoff at window boundaries
   - No gradient flow beyond window size

### Recommendations for Stability

1. **Gradient Clipping:**
   - Implement global gradient norm clipping (threshold: 2000.0 as in code)
   - Per-layer gradient monitoring

2. **Polynomial Regularization:**
   - Add L2 regularization to polynomial weights
   - Constrain polynomial degrees to prevent overfitting

3. **Adaptive Learning Rates:**
   - Use layer-wise adaptive LR scaling (LARS) as implemented
   - AMSGrad variant for better convergence guarantees

4. **Numerical Stability:**
   - Safe softmax with max subtraction
   - Gradient anomaly detection and early stopping

## Polynomial Flexibility Enhancements

### Current Polynomial Usage

1. **Attention Polynomials:** Degree-3 approximation to attention nonlinearity
2. **SwiGLU Gates:** Cubic polynomial sigmoid approximations
3. **Gating Functions:** Learnable polynomials for head selection

### Potential Improvements

1. **Higher-Order Polynomials:**
   - Increase degree p in PolyAttention for better approximation
   - Adaptive degree selection based on sequence complexity

2. **Chebyshev Polynomials:**
   - Use Chebyshev basis for better numerical stability
   - Orthogonal polynomials reduce conditioning issues

3. **Adaptive Polynomials:**
   - Learnable polynomial degrees per layer/head
   - Context-dependent polynomial selection

4. **Spline Approximations:**
   - Piecewise polynomial approximations for better local fit
   - Reduced global polynomial degree requirements

### Implementation Suggestions

1. **Polynomial Attention Variants:**
   ```
   φ(s) = ∑_{k=0}^p w_k · T_k(s/max_s)
   ```
   where T_k are Chebyshev polynomials

2. **Gated Polynomial Networks:**
   ```
   y = ∑_{k=0}^p g_k · P_k(x)
   ```
   where P_k are orthogonal polynomials and g_k are learned gates

3. **Adaptive Polynomial Degrees:**
   - Per-token degree selection based on complexity predictors
   - Hierarchical polynomial expansion for long contexts