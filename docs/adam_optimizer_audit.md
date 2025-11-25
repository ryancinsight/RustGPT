# Adam Optimizer Audit and Enhancement Plan

## Date: 2024-11-24

## Status: IN PROGRESS

## Executive Summary

This audit examines the current Adam optimizer implementation, identifies the root cause of non-deterministic training results, and proposes incremental enhancements inspired by modern optimizers (Muon, MADGRAD, Mirror Descent).

---

## Issue Analysis: Non-Deterministic Training Results

### Observed Behavior

Running the same training twice produces different outputs:
- First run: `"Assistant : : : : : : : : :"`
- Second run: `"</s>"`

### Root Cause: Unseeded Random Number Generation

The codebase uses `rand::rng()` without a fixed seed in multiple locations:

```rust
// Found throughout the codebase:
let mut rng = rand::rng();  // No seed = non-deterministic
```

**Affected Areas:**
1. Weight initialization (embedding layers, attention, FFN)
2. Dropout during training
3. Data shuffling
4. Speculative sampling

**Solution:** Add a `--seed` CLI option and propagate a seeded RNG throughout initialization.

---

## Current Adam Implementation Audit

### File: `src/adam.rs`

### Strengths ✓

| Feature | Status | Notes |
|---------|--------|-------|
| Bias correction | ✓ | Proper m̂ and v̂ computation |
| AMSGrad variant | ✓ | Optional v_hat_max tracking |
| AdamW (decoupled WD) | ✓ | Proper weight decay handling |
| Shape validation | ✓ | Prevents runtime panics |
| State reset | ✓ | `reset()` method available |

### Current Algorithm

```
Input: learning rate η, betas (β₁, β₂), epsilon ε, weight decay λ
Initialize: m₀ = 0, v₀ = 0, t = 0

For each step:
    t ← t + 1
    
    # AdamW: decoupled weight decay
    if decoupled_wd and λ > 0:
        θ ← θ × (1 - λη)
        g ← gradient
    else if λ > 0:
        g ← gradient + λθ  # L2 regularization
    else:
        g ← gradient
    
    # Momentum update
    m ← β₁m + (1 - β₁)g
    v ← β₂v + (1 - β₂)g²
    
    # Bias correction
    m̂ ← m / (1 - β₁ᵗ)
    v̂ ← v / (1 - β₂ᵗ)
    
    # AMSGrad (optional)
    if amsgrad:
        v̂_max ← max(v̂_max, v̂)
        v̂_used ← v̂_max
    else:
        v̂_used ← v̂
    
    # Parameter update
    θ ← θ - η × m̂ / (√v̂_used + ε)
```

### Issues Found

1. **No gradient clipping integration** - Clipping is done externally in `llm.rs`
2. **No warmup built-in** - Warmup is handled at training loop level
3. **Per-parameter instances** - Each layer creates its own Adam, no global state
4. **No learning rate scheduling** - Cosine annealing done externally

---

## Enhancement Plan: Incremental Improvements

### Phase 1: Deterministic Training (Priority: HIGH)

Add seed support for reproducible results.

**Changes:**
1. Add `--seed <u64>` CLI option
2. Create seeded RNG at startup
3. Propagate to all initialization functions

### Phase 2: Gradient Orthogonalization (Muon-inspired)

**Key Insight from Muon:** Orthogonalizing the momentum update improves training by:
- Balancing updates across all singular directions
- Preventing updates from being dominated by a few directions
- Improving conditioning of the update matrix

**Newton-Schulz Iteration (5 steps, bfloat16-stable):**
```rust
fn newton_schulz5(g: &Array2<f32>, steps: usize) -> Array2<f32> {
    let (a, b, c) = (3.4445, -4.7750, 2.0315);
    let mut x = g.clone();
    let norm = x.iter().map(|&v| v * v).sum::<f32>().sqrt() + 1e-7;
    x /= norm;
    
    // Transpose if tall matrix
    let transposed = g.nrows() > g.ncols();
    if transposed {
        x = x.t().to_owned();
    }
    
    for _ in 0..steps {
        let a_mat = x.dot(&x.t());
        let b_mat = &a_mat * b + a_mat.dot(&a_mat) * c;
        x = &x * a + b_mat.dot(&x);
    }
    
    if transposed { x.t().to_owned() } else { x }
}
```

**Hybrid Approach:** Apply orthogonalization to 2D parameters only (hidden layers), use standard Adam for embeddings/output.

### Phase 3: MADGRAD-inspired Dual Averaging

**Key Insight from MADGRAD:** Uses dual averaging instead of exponential moving average for better theoretical convergence.

```rust
// MADGRAD-style gradient accumulation
s_k = s_{k-1} + λ_k * g_k          // Sum of weighted gradients
z_k = z_{k-1} + λ_k * g_k²         // Sum of weighted squared gradients
x_k = x_0 - s_k / (z_k^(1/3) + ε)  // Cubic root scaling
```

**Advantage:** Better for sparse gradients, improved convergence on noisy objectives.

### Phase 4: Adaptive Learning Rate Scaling

**Spectral Norm Scaling:** Scale learning rate by inverse spectral norm of layer weights.

```rust
fn spectral_norm_estimate(w: &Array2<f32>, iters: usize) -> f32 {
    // Power iteration for largest singular value
    let mut v = Array1::ones(w.ncols());
    for _ in 0..iters {
        let u = w.dot(&v);
        let u_norm = u.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let u = &u / u_norm.max(1e-8);
        v = w.t().dot(&u);
        let v_norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        v = &v / v_norm.max(1e-8);
    }
    let sigma = w.dot(&v).iter().map(|&x| x * x).sum::<f32>().sqrt();
    sigma
}

// Use: lr_effective = lr / spectral_norm_estimate(weights, 5)
```

### Phase 5: Mirror Descent Integration

**Key Insight:** Mirror descent generalizes gradient descent using Bregman divergences.

For neural networks, use **matrix entropy** as the mirror map:
```
Φ(W) = Tr(W log W - W)  // Matrix entropy
∇Φ(W) = log W           // Gradient of mirror map
```

**Simplified Integration:** Use log-space updates for attention weights.

---

## Implementation Roadmap

| Phase | Enhancement | Complexity | Impact | Files to Modify |
|-------|-------------|------------|--------|-----------------|
| 1 | Seed support | Low | HIGH | cli.rs, main.rs, embeddings.rs, attention/* |
| 2 | Newton-Schulz orthogonalization | Medium | HIGH | adam.rs (new method) |
| 3 | MADGRAD-style accumulation | Medium | Medium | adam.rs (new variant) |
| 4 | Spectral norm scaling | Low | Medium | adam.rs, transformer_block.rs |
| 5 | Mirror descent (experimental) | High | Unknown | New file |

---

## Proposed New API

```rust
pub enum OptimizerVariant {
    Adam,           // Standard Adam
    AdamW,          // Decoupled weight decay
    AMSGrad,        // AMSGrad variant
    Muon,           // Orthogonalized momentum (2D params only)
    MADGRAD,        // Dual averaging style
}

pub struct UnifiedOptimizer {
    variant: OptimizerVariant,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    // State
    m: Array2<f32>,
    v: Array2<f32>,
    s: Option<Array2<f32>>,  // MADGRAD sum
    v_hat_max: Option<Array2<f32>>,  // AMSGrad
    timestep: usize,
}

impl UnifiedOptimizer {
    pub fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>, lr: f32) {
        match self.variant {
            OptimizerVariant::Muon => self.step_muon(params, grads, lr),
            OptimizerVariant::MADGRAD => self.step_madgrad(params, grads, lr),
            _ => self.step_adam(params, grads, lr),
        }
    }
}
```

---

## Testing Strategy

1. **Determinism Test:** Run training twice with same seed, verify identical outputs
2. **Convergence Test:** Compare loss curves: Adam vs AdamW vs Muon
3. **Benchmark:** Measure wall-clock time per epoch for each variant
4. **Quality Test:** Evaluate generation quality (BLEU, perplexity) across variants

---

## References

1. **Adam:** Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
2. **AdamW:** Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
3. **AMSGrad:** Reddi et al., "On the Convergence of Adam and Beyond" (2018)
4. **Muon:** Jordan et al., "Muon: An optimizer for hidden layers" (2024)
5. **MADGRAD:** Defazio & Jelassi, "Adaptivity without Compromise" (2021)
6. **Shampoo:** Gupta et al., "Preconditioned Stochastic Tensor Optimization" (2018)

---

## Next Steps

1. [ ] Implement Phase 1: Add `--seed` CLI option
2. [ ] Test determinism with fixed seed
3. [ ] Implement Newton-Schulz orthogonalization
4. [ ] Benchmark Muon-style updates vs standard Adam
5. [ ] Document results and iterate
