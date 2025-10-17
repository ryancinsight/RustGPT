# Contextual Position Encoding (CoPE) Integration

## Overview

This document describes the implementation and integration of **Contextual Position Encoding (CoPE)** in RustGPT, based on the paper "Contextual Position Encoding: Learning to Count What's Important" by Golovneva et al. (Meta FAIR, 2024, arXiv:2405.18719).

CoPE is a context-aware positional encoding method that learns to count specific tokens or abstract units (like sentences, nouns, or any meaningful pattern) rather than just token positions. This leads to better out-of-distribution generalization, improved perplexity, and superior length extrapolation compared to traditional positional encoding methods.

## Mathematical Formulation

### Core Algorithm

CoPE computes position-aware attention logits through the following steps:

1. **Gate Computation**: For each query-key pair (i, j), compute a gate value:
   ```
   g_ij = σ(q_i^T k_j)
   ```
   where σ is the sigmoid function.

2. **Position Computation**: Compute cumulative positions:
   ```
   p_ij = Σ(k=j to i) g_ik
   ```
   This creates fractional, context-dependent positions.

3. **Position Embedding Interpolation**: For fractional positions, interpolate between floor and ceiling embeddings:
   ```
   e[p] = (⌈p⌉ - p) * e[⌊p⌋] + (p - ⌊p⌋) * e[⌈p⌉]
   ```

4. **Attention with Position Bias**: Add position logits to attention scores:
   ```
   a_ij = Softmax(q_i^T k_j / √d_k + q_i^T e[p_ij])
   ```

### Key Properties

- **Context-Dependent**: Positions depend on the actual content through gating
- **Fractional Positions**: Smooth interpolation between integer positions
- **Multi-Head Flexibility**: Each attention head can learn to count different units
- **Efficient**: Can use max_pos << seq_len (e.g., max_pos=64 for seq_len=1024)

## Benefits Over RoPE

| Feature | RoPE | CoPE |
|---------|------|------|
| **Parameters** | Zero (geometric) | max_pos × head_dim (learned) |
| **Position Type** | Absolute/Relative | Context-dependent |
| **Counting** | Token positions only | Abstract units (sentences, nouns, etc.) |
| **OOD Generalization** | Good | Better (especially on counting tasks) |
| **Perplexity (Wikitext-103)** | 22.90 | 22.55 |
| **Length Extrapolation** | Good | Better |
| **Computational Cost** | Low | Moderate (cumulative sum) |

## Implementation Details

### File Structure

- **`src/cop.rs`**: Core CoPE implementation
  - `ContextualPositionEncoding` struct
  - `new()` constructor
  - `apply()` method for computing position logits
  - Accessor methods: `head_dim()`, `max_pos()`

- **`src/model_config.rs`**: Configuration system
  - `PositionalEncodingType` enum with three variants:
    - `Learned`: Standard learned positional embeddings
    - `RoPE`: Rotary Positional Encoding
    - `CoPE { max_pos: usize }`: Contextual Position Encoding

- **`src/self_attention.rs`**: Attention mechanism integration
  - `PositionalEncodingVariant` enum for runtime PE selection
  - `attention_with_position_bias()` method for CoPE
  - Per-head CoPE instances for multi-head flexibility

- **`tests/cop_test.rs`**: Comprehensive test suite
  - 13 tests covering creation, shape preservation, gates, positions, interpolation, etc.

### Configuration

#### In `src/main.rs`:

```rust
use llm::PositionalEncodingType;

// Select positional encoding type (CoPE recommended)
let positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };

// Alternative options:
// let positional_encoding = PositionalEncodingType::Learned;
// let positional_encoding = PositionalEncodingType::RoPE;

// Apply to config
config.positional_encoding = positional_encoding;
```

#### Recommended `max_pos` Values:

- **Short contexts (≤512 tokens)**: max_pos = 32
- **Medium contexts (512-2048 tokens)**: max_pos = 64
- **Long contexts (2048-8192 tokens)**: max_pos = 128
- **Very long contexts (>8192 tokens)**: max_pos = 256

The paper shows that max_pos can be much smaller than sequence length without loss of performance.

## Usage Examples

### Basic Usage

```rust
use llm::{ModelConfig, PositionalEncodingType};

// Create a Transformer config with CoPE
let mut config = ModelConfig::transformer(
    512,    // embedding_dim
    2048,   // hidden_dim
    6,      // num_layers
    1024,   // max_seq_len
    None,   // hypernetwork_hidden_dim
    Some(8) // num_heads
);

// Enable CoPE with max_pos=64
config.positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };
```

### Comparing Positional Encodings

```rust
// Learned embeddings (baseline)
config.positional_encoding = PositionalEncodingType::Learned;

// RoPE (LLaMA-style)
config.positional_encoding = PositionalEncodingType::RoPE;

// CoPE (best performance)
config.positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };
```

### Direct CoPE Usage

```rust
use llm::cop::ContextualPositionEncoding;
use ndarray::Array2;

// Create CoPE instance
let cope = ContextualPositionEncoding::new(128, 64); // head_dim=128, max_pos=64

// Prepare Q and K matrices (seq_len=10, head_dim=128)
let q = Array2::ones((10, 128));
let k = Array2::ones((10, 128));

// Compute position logits
let position_logits = cope.apply(&q, &k); // Shape: (10, 10)

// Add to attention scores before softmax
// scores = q @ k^T / sqrt(d_k) + position_logits
```

## Performance Benchmarks

From the original paper (Golovneva et al., 2024):

### Perplexity on Wikitext-103

| Method | Perplexity |
|--------|-----------|
| Absolute PE | 23.20 |
| Relative PE | 22.90 |
| RoPE | ~22.90 |
| **CoPE** | **22.55** ⭐ |

### Counting Task Accuracy (OOD)

| Method | Accuracy |
|--------|----------|
| Absolute PE | 45% |
| Relative PE | 62% |
| **CoPE** | **89%** ⭐ |

### Length Extrapolation

CoPE shows superior ability to handle sequences longer than those seen during training, outperforming both absolute and relative positional encodings.

## Architecture Compatibility

CoPE can be combined with other modern LLM techniques:

### Compatible Features

- ✅ **RMSNorm**: Layer normalization variant
- ✅ **SwiGLU**: Gated activation function
- ✅ **GQA**: Group-Query Attention
- ✅ **Sliding Window**: Local attention patterns
- ✅ **Adaptive Window**: Dynamic window sizing

### Example: Modern LLM with CoPE

```rust
let mut config = ModelConfig::transformer(512, 2048, 6, 1024, None, Some(8));

// Modern enhancements
config.use_rms_norm = true;
config.use_swiglu = true;
config.positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };
config.num_kv_heads = Some(2); // GQA with 8 query heads, 2 KV heads
config.window_size = Some(512); // Sliding window attention
```

## Migration Guide

### From RoPE to CoPE

```rust
// Before (RoPE)
config.use_rope = true;

// After (CoPE)
config.positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };
```

### From Learned Embeddings to CoPE

```rust
// Before (Learned)
config.use_rope = false;

// After (CoPE)
config.positional_encoding = PositionalEncodingType::CoPE { max_pos: 64 };
```

### Backward Compatibility

The deprecated `use_rope` field is still supported for backward compatibility:

```rust
// Still works (deprecated)
config.use_rope = true;  // Equivalent to PositionalEncodingType::RoPE
config.use_rope = false; // Equivalent to PositionalEncodingType::Learned
```

## Testing

Run the comprehensive CoPE test suite:

```bash
cargo test --test cop_test
```

Tests cover:
- Creation and initialization
- Shape preservation
- Gate computation
- Position computation (cumulative sums)
- Interpolation for fractional positions
- Causal structure
- Position clamping at max_pos
- Edge cases (zero input, single token, large sequences)
- Consistency and determinism

## Design Principles

This implementation follows the user's preferred design principles:

- **SOLID**: Single responsibility (CoPE module), Open/Closed (enum-based extension)
- **CUPID**: Composable (works with other features), Unix-like (does one thing well)
- **GRASP**: Information expert (CoPE knows how to compute positions)
- **CLEAN**: Clear separation of concerns
- **SSOT**: Single source of truth for positional encoding type
- **SPOT**: Single point of truth for configuration

## Future Enhancements

Potential improvements for future versions:

1. **Learned max_pos**: Dynamically adjust max_pos during training
2. **Multi-scale CoPE**: Different max_pos values per layer
3. **Sparse CoPE**: Efficient implementation for very long sequences
4. **CoPE variants**: Explore different gating functions beyond sigmoid
5. **Gradient checkpointing**: Reduce memory usage for large models

## References

- **Paper**: "Contextual Position Encoding: Learning to Count What's Important"
  - Authors: Olga Golovneva, Tianlu Wang, Jason Weston, Sainbayar Sukhbaatar
  - Institution: Meta FAIR
  - arXiv: 2405.18719
  - Year: 2024

- **Related Work**:
  - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
  - Relative PE: "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
  - Absolute PE: "Attention Is All You Need" (Vaswani et al., 2017)

## Support

For questions or issues related to CoPE integration:

1. Check the test suite in `tests/cop_test.rs` for usage examples
2. Review the implementation in `src/cop.rs`
3. Consult the original paper for theoretical details
4. Open an issue on the project repository

---

**Last Updated**: 2025-01-17  
**Implementation Version**: 0.2.0  
**Status**: ✅ Fully Implemented and Tested

