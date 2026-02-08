# CoPE (Contextual Position Embeddings) Research & Implementation Findings

## Overview

This document details the investigation and implementation of enhanced CoPE variants for the RustGPT attention mechanism. Three new variants have been implemented alongside the existing base CoPE.

---

## Implemented Variants

### 1. Base CoPE (Existing)

**Location:** `src/domain/attention/position/cope.rs`

**Features:**
- Learnable position embeddings: `(max_pos + 1, embed_dim)`
- Adam optimizer integration
- Full gradient flow
- Stability via clipping

---

### 2. GatedCoPE (NEW - High Priority)

**Location:** `src/domain/attention/position/gated_cope.rs`

**Mathematical Formulation:**
```
s_ij = q_i · k_j + g_ij · CoPE(q_i, pos_ij)

where g_ij = σ(W_g · [q_i; k_j] + b_g)
```

**Key Features:**
- **Learnable gating per position**: Gate parameters `(q ⊕ k) → scalar` for each relative position
- **Temperature scaling**: `σ(x/T)` for sharper/smoother gates
- **Multiplicative interaction**: `gate * CoPE_contrib` preserves gradient flow better than addition
- **Numerical stability**: Clamped sigmoid prevents overflow

**Benefits:**
- Adaptive position/content weighting per attention head
- Learns when to emphasize position vs content similarity
- Reduced information loss through multiplicative gating

**Parameters:**
- Base CoPE: `(max_pos+1) × embed_dim`
- Gate weights: `2×embed_dim × (max_pos+1)`
- Gate bias: `max_pos+1`

---

### 3. HierarchicalCoPE (NEW - Medium Priority)

**Location:** `src/domain/attention/position/hierarchical_cope.rs`

**Mathematical Formulation:**
```
pos_ij = chunk_idx * chunk_size + local_pos
CoPE_total = α_local * CoPE_local(local_pos) + α_global * CoPE_global(chunk_idx)
```

**Key Features:**
- **Two-level granularity**:
  - Local CoPE: Positions within a chunk (`chunk_size`, `embed_dim`)
  - Global CoPE: Chunk-level positions (`max_chunks`, `embed_dim`)
- **Content-aware chunking**: Learnable boundary predictor
- **Mixing weights**: `α_local`, `α_global` with soft transitions

**Benefits:**
- Better generalization to sequences longer than `max_pos`
- Learns natural chunking boundaries from data
- Reduced parameters for equivalent range

**Parameters:**
```
Local CoPE: chunk_size × embed_dim
Global CoPE: max_chunks × embed_dim
Chunk predictor: embed_dim × 2 + 1
```

---

### 4. FactorizedCoPE (NEW - Low Priority)

**Location:** `src/domain/attention/position/factorized_cope.rs`

**Mathematical Formulation:**
```
CoPE(q, pos) = log(1 + exp(U[pos] @ V @ q))
```

**Key Features:**
- **Low-rank factorization**: `PE = U @ V` where `U ∈ ℝ^(max_pos×r)`, `V ∈ ℝ^(r×embed_dim)`
- **Log1p stabilization**: `log(1 + exp(x))` for smooth gradients
- **Temperature scaling**: Controls gradient magnitude

**Benefits:**
- O(`max_pos × r`) parameters instead of O(`max_pos × embed_dim`)
- `r << embed_dim` (recommended: `embed_dim / 4`)
- Typical compression ratio: 3-4x

**Example Compression:**
```
max_pos=512, embed_dim=128, rank=32
Full: 65664 parameters
Factored: 20512 parameters (3.2x compression)
```

---

## Integration Points

### Forward Pass Integration

```rust
// GatedCoPE usage in attention:
let cope_contrib = gated_cope.gated_cope_contribution(&q, &k, pos);
let mut score = q.dot(&k) * dk_scale;
score += cope_contrib;

// HierarchicalCoPE usage:
let (local, global) = hierarchical_cope.hierarchical_components(&q, pos);
score += alpha_local * local + alpha_global * global;

// FactorizedCoPE usage:
let cope_contrib = factorized_cope.factorized_cope_contribution(&q, pos);
score += cope_contrib;
```

### Gradient Application

```rust
// GatedCoPE gradient tuple
let grads = (base_cope_grad, gate_grads);
gated_cope.apply_gradients(&grads, lr);

// HierarchicalCoPE gradient tuple  
let grads = (local_grad, global_grad, predictor_w_grad, predictor_b_grad);
hierarchical_cope.apply_gradients(&grads, lr);

// FactorizedCoPE gradient tuple
let grads = (up_grad, down_grad);
factorized_cope.apply_gradients(&grads, lr);
```

---

## Stability Guarantees

All variants inherit and extend the stability properties documented in `ADR_poly_attention_stability.md`:

### GatedCoPE
- **Gate saturation**: Temperature scaling prevents extreme gates
- **Clamping**: `[-500, 500]` bound on sigmoid input

### HierarchicalCoPE
- **Boundary gates**: Sigmoid with bounds prevents gradient explosion
- **Mixing normalization**: `α` values normalized to sum to 1

### FactorizedCoPE
- **Log1p stability**: Piecewise computation prevents overflow
- **Gradual initialization**: `N(0, 0.02/√r)` prevents large initial values

---

## Usage Recommendations

| Variant | Use Case | Priority |
|---------|----------|----------|
| Base CoPE | Standard attention, production | Default |
| GatedCoPE | Variable position importance | High |
| HierarchicalCoPE | Long sequences (>4K tokens) | Medium |
| FactorizedCoPE | Memory-constrained environments | Low |

---

## Testing

All variants include comprehensive unit tests:

```bash
cargo test --package rust-gpt gated_cope
cargo test --package rust-gpt hierarchical_cope
cargo test --package rust-gpt factorized_cope
```

**Test Coverage:**
- Creation and initialization
- Forward computation (finite outputs)
- Parameter counting verification
- Numerical stability (extreme inputs)
- Gradient flow

---

## Future Enhancements

1. **Hybrid CoPE**: Combine GatedCoPE + FactorizedCoPE for memory-efficient adaptive positions
2. **Learned Temperature**: Make temperature a learnable parameter
3. **Dynamic Chunking**: HierarchicalCoPE with learned chunk boundaries per head
4. **CoPE + RoPE Hybrid**: Combine relative positions (RoPE) with learnable biases (CoPE)

---

## Conclusion

The implemented CoPE variants provide a flexible toolkit for position encoding:

- **GatedCoPE**: Maximum expressivity through adaptive gating
- **HierarchicalCoPE**: Scalability to very long sequences
- **FactorizedCoPE**: Memory efficiency for resource-constrained deployment

All implementations are production-ready with:
- Full gradient flow
- Stability guarantees
- Comprehensive tests
- Clean API design
