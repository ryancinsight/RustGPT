# Transformer Module

This module provides reusable transformer block components that can be used across different transformer architectures (standard, hierarchical, recurrent, etc.).

## Overview

The core component is the `TransformerBlock` which encapsulates the standard transformer block pattern:

- **Pre-attention normalization**: Layer normalization before attention
- **Attention mechanism**: PolyAttention with CoPE positional encoding
- **Pre-feedforward normalization**: Layer normalization before feedforward
- **Feedforward network**: RichardsGlu or Mixture-of-Experts
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
- **Reusable**: Same block can be used for different architectures (HRM, TRM, standard transformer)
- **Configurable**: Support for different attention mechanisms and feedforward networks
- **Tested**: Comprehensive unit tests ensure correctness
- **Serializable**: Full serde support for model persistence

## Future Extensions

This module provides the foundation for implementing:
- **Hierarchical Transformer (HRM)**: Multi-scale hierarchical processing
- **Transformer Recurrent Model (TRM)**: Recurrent transformer variants
- **Other transformer variants**: Any architecture that uses the standard transformer block pattern





