# Transformer Module

This module provides reusable transformer block components that can be used across different transformer architectures (standard, hierarchical, recurrent, etc.).

## Overview

The core component is the `TransformerBlock` which encapsulates the standard transformer block pattern:

- **Pre-attention normalization**: Layer normalization before attention
- **Attention mechanism**: PolyAttention with CoPE positional encoding
- **Pre-feedforward normalization**: Layer normalization before feedforward
- **Feedforward network**: RichardsGlu (Mixture-of-Experts is available but typically disabled)
- **Residual connections**: Proper residual connections around attention and feedforward

## Architecture

```
Input
  │
  ├─► Pre-Attention Norm ──► Attention ──┐
  │                                     │
  └─────────────────────────────────────┼─► Residual ─► Pre-FFN Norm ──► Feedforward ──┐
                                        │                                           │
                                        └─────────────────────────────────────────────┼─► Residual ─► Output
```

## Components

- `TransformerBlock`: Complete transformer block with all components
- `TransformerBlockConfig`: Configuration for transformer block parameters
- `FeedForwardVariant`: Enum for different feedforward network types

## Usage

```rust
use crate::transformer::TransformerBlock;

// Create from configuration
let config = TransformerBlockConfig {
    embed_dim: 128,
    hidden_dim: 256,
    num_heads: 8,
    // ... other config
};
let mut block = TransformerBlock::new(config);

// Or create from ModelConfig
let block = TransformerBlock::from_model_config(&model_config, layer_idx);

// Use in forward/backward pass
let output = block.forward(&input);
let input_grads = block.backward(&output_grads, learning_rate);
```

## Benefits

- **Modular**: Clean separation of concerns
- **Reusable**: Same block can be used for different architectures (HRM, LRM, standard transformer)
- **Configurable**: Support for different attention mechanisms and feedforward networks
- **Tested**: Comprehensive unit tests ensure correctness
- **Serializable**: Full serde support for model persistence

## Future Extensions

This module provides the foundation for implementing:
- **Hierarchical Transformer (HRM)**: Multi-scale hierarchical processing
- **Latent Recursive Model (LRM)**: Recurrent transformer variants
- **Other transformer variants**: Any architecture that uses the standard transformer block pattern




## Mathematical Specification

Let `X ∈ ℝ^{T×d}` denote the input sequence with length `T` and embedding dimension `d`.

- Pre-attention normalization: `Z₁ = LN₁(X)` where for each timestep `t`,
  `LN₁(x_t) = γ₁ ⊙ ((x_t − μ_t) / σ_t) + β₁` with learnable `γ₁, β₁ ∈ ℝ^{d}`.
- Attention: `A = Attn(Z₁)` with `A ∈ ℝ^{T×d}` produced by multi-head Polynomial Attention of degree `p` with CoPE positional encoding and head selection strategy.
- First residual: `R₁ = X + A`.
- Pre-FFN normalization: `Z₂ = LN₂(R₁)` where `LN₂` has learnable `γ₂, β₂ ∈ ℝ^{d}`.
- Feedforward:
  - RichardsGlu: `F = W₂ · φ(W₁ · Z₂ + b₁) + b₂`, with `W₁ ∈ ℝ^{d×h}`, `W₂ ∈ ℝ^{h×d}`, `b₁ ∈ ℝ^{h}`, `b₂ ∈ ℝ^{d}`, and nonlinearity `φ` induced by the Richards GLU.
- Output: `Y = R₁ + F`.

These equations correspond exactly to the implementation: pre-norm → attention + residual → pre-norm → feedforward + residual (`src/transformer/transformer_block.rs:226-252`).

### Gradient Invariants

With upstream gradient `∂L/∂Y`:
- Split at output residual: `∂L/∂R₁ = ∂L/∂Y`, `∂L/∂F = ∂L/∂Y`.
- Through FFN: `(∂L/∂Z₂, ∂L/∂θ_ffn) = FFN.backward(Z₂, ∂L/∂F)`.
- Through pre-FFN norm: `(∂L/∂R₁)_from_ffn = LN₂.backward(R₁, ∂L/∂Z₂)`.
- Combine residual-1 gradients: `G₁ = ∂L/∂R₁ + (∂L/∂R₁)_from_ffn`.
- Split at residual-1: `∂L/∂X_direct = G₁`, `∂L/∂A = G₁`.
- Through attention: `(∂L/∂Z₁, ∂L/∂θ_attn) = Attn.backward(Z₁, ∂L/∂A)`.
- Through pre-attention norm: `∂L/∂X_norm = LN₁.backward(X, ∂L/∂Z₁)`.
- Final input gradients: `∂L/∂X = ∂L/∂X_direct + ∂L/∂X_norm`.

This matches the analytical gradient routing implemented in `compute_gradients` (`src/transformer/transformer_block.rs:268-347`).

## Complexity Analysis

Define parameters: `T` (sequence length), `d` (embed dim), `h` (hidden dim, typically `≥ d`), `H` (heads), `p` (polynomial degree).

- Attention FLOPs (PolyAttention, degree `p`, `H` heads): `≈ 4 · T · d · H · p`.
- Feedforward FLOPs (two affine transforms): `≈ 2 · T · d · h`.
- Layer norms FLOPs (two norms): `≈ 4 · T · d`.
- Total forward FLOPs: `≈ T · [4 d H p + 2 d h + 4 d]`.

Bytes moved (approx., 4 bytes per `f32`):
- Activations: `≈ 4 · T · d`.
- Hidden activations in FFN: `≈ 4 · T · h`.
- Total forward bytes: `≈ 4 · T · (d + h)`.

Asymptotics:
- Time: `O(T · d · (H · p + h + 1))`.
- Memory (activations): `O(T · (d + h))`.

These estimates are implemented in `metrics::perf::estimate_transformer_block` (`src/metrics/perf.rs:18-33`) and used across benchmarking and performance reporting.

### Notes
- Sliding window attention (`window_size`) limits effective receptive field while maintaining the above per-token arithmetic cost structure; overall throughput benefits from kernel locality even as the core complexity remains linear in `T` for PolyAttention.
- Mixture-of-Experts is not used in the current configuration; FFN is RichardsGlu-only.

## Locality and Adaptation Enhancements

- Dynamic window sizing: The attention window is set per-forward, clamped to sequence length. When adaptive mode is enabled, window size follows strategy:
  - SequenceLengthBased: `w = clamp(seq_len/2, min_window, max_window)`
  - AttentionEntropy: EMA of `τ` span and gating RMS maps linearly to `[min_window, max_window]`
  - Fixed/PerplexityBased: uses configured base window. See `src/transformer/transformer_block.rs:260-287`.
- Adaptive polynomial degree: PolyAttention starts at `p=1` and adapts based on forward metrics (gating `τ` span and predictor RMS). Applied after each forward call. See `src/attention/poly_attention.rs:499-513` and `src/attention/poly_attention.rs:379-441`.

### Configuration
- Fields: `use_adaptive_window`, `min_window_size`, `max_window_size`, `window_adaptation_strategy`, `entropy_ema_alpha` in `TransformerBlockConfig` (`src/transformer/transformer_block.rs:71-99`).
- Populated from `ModelConfig` in `from_model_config` (`src/transformer/transformer_block.rs:168-196`).

### Optimization Routes (Maintaining Accuracy)
- Reduce `p` and active heads only when accuracy thresholds are met; `p` increases when gating is diffuse and predictor RMS is high, decreases when both are low.
- Sequence-length-aware windows: `dynamic_w = min(seq_len, base_window)` increases cache locality and reduces memory bandwidth while preserving the receptive field up to the configured bound.
- Buffer reuse: Prefer in-place updates and contiguous layouts in attention and FFN to reduce clones and allocations.

### Implementation References
- Window setting before attention: `src/transformer/transformer_block.rs:229-234`
- Degree adaptation hook: `src/attention/poly_attention.rs:... adapt_degree_from_forward_metrics` and call in `forward_impl`.

## Current Implementation Status

- Attention: `PolyAttention` with head selection strategy applied during construction (`src/transformer/transformer_block.rs:133-134`).
- Residual and pre-norm ordering: `forward` uses pre-norm → attention → residual → pre-norm → feedforward → residual (`src/transformer/transformer_block.rs:226-239`).
- Configuration sourcing: `from_model_config` sets `max_pos` and `window_size` appropriately (`src/transformer/transformer_block.rs:172-193`).
- Gradients: analytical routing with deterministic partition capture (`src/transformer/transformer_block.rs:268-347`).
- Gradient application: global clipping and LARS-style adaptive scaling per submodule (`src/transformer/transformer_block.rs:374-391`, `src/transformer/transformer_block.rs:410-461`, `src/transformer/transformer_block.rs:463-477`).

## Optimization Routes

### Reduce Complexity (without accuracy loss)
- Head selection (`HeadSelectionStrategy`): activate a subset of heads to reduce effective `H` while maintaining representational coverage. Configured at attention construction (`src/transformer/transformer_block.rs:133-134`).
- Polynomial degree `p`: choose minimal degree that preserves target accuracy; reduces attention flops linearly (`src/metrics/perf.rs:18-33`).
- Sliding window (`window_size`): constrain local context to improve cache locality and throughput; estimator remains linear in `T` for PolyAttention, but memory access patterns improve (`src/transformer/transformer_block.rs:178-187`).

### Improve Performance
- Fuse normalization + residual adds on hot paths to reduce memory traffic, preserving mathematical equivalence of forward equations.
- Precompute and reuse CoPE positional components across steps where `max_pos` is fixed.
- Use contiguous `Array2` layouts and avoid unnecessary clones in forward; reuse buffers for `norm` and `ffn` intermediates.
- Exploit head selection to schedule active heads densely for better SIMD/cache utilization.

### Memory Efficiency
- Activation checkpointing on attention/FFN to trade compute for memory during training while preserving analytical gradient correctness.
- Reduce intermediate retention: cache only required tensors for `compute_gradients`; avoid cloning large arrays when slices suffice (`src/transformer/transformer_block.rs:241-249`).
- Right-size hidden dimension `h` to `≥ d` but avoid oversized values; FLOPs and bytes scale linearly with `h`.

### Accuracy Preservation
- Maintain pre-norm ordering and residual gradient splits exactly as implemented (`src/transformer/transformer_block.rs:268-317`).
- Keep LARS-style scaling during `apply_gradients` to balance learning rates across submodules (`src/transformer/transformer_block.rs:410-447`).
- Validate head selection and `p` reductions against held-out metrics; only accept changes that satisfy target accuracy thresholds.





