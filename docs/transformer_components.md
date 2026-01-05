# Transformer Components Documentation

## Overview

The transformer architecture has been refactored into modular components for improved flexibility and composition. This document describes the new component-based architecture and speculative sampling implementation.

## Modular Transformer Components

### 1. AttentionContext

**Purpose**: Manages attention context and similarity representations for cross-layer conditioning.

**Key Features**:
- Maintains activation similarity matrices between layers
- Provides similarity-based context signals for next-layer conditioning
- Supports learned similarity context strength
- Enables cross-layer information flow

**Mathematical Formulation**:
```
S_t = X_t · X_t^T / embed_dim          // Activation similarity matrix
X'_t = X_t + (strength / embed_dim) * X_t · S_{t-1}  // Context-conditioned input
```

### 2. FeedforwardProcessor

**Purpose**: Encapsulates feedforward network processing with support for multiple variants.

**Supported Variants**:
- **RichardsGLU**: Gated linear unit with Richards curve activation
- **MixtureOfExperts**: Sparse expert routing with load balancing
- **SwiGLU**: Swish-gated linear unit

**Key Features**:
- Unified interface for different feedforward architectures
- Automatic gradient routing to appropriate parameters
- Performance monitoring and metrics collection

### 3. NormalizationLayer

**Purpose**: Provides flexible normalization with Richards-based dynamic normalization.

**Key Features**:
- Dynamic Tanh normalization with learnable parameters
- Layer normalization with learned scale and bias
- Gradient-safe normalization operations
- Configurable normalization strength

**Mathematical Formulation**:
```
y = tanh(α · (x - μ) / σ) ⊙ γ + β
```

### 4. ResidualConnection

**Purpose**: Manages residual connections with adaptive scaling and gradient handling.

**Key Features**:
- Adaptive residual scaling based on gradient norms
- Pre-norm vs post-norm configuration support
- Gradient accumulation and routing
- Numerical stability checks

### 5. TemporalMixingWrapper

**Purpose**: Abstract wrapper for temporal mixing mechanisms (attention or RG-LRU).

**Key Features**:
- Unified interface for different temporal mixing strategies
- Automatic dispatch to appropriate implementation
- Performance monitoring and metrics
- Gradient routing to underlying mechanism

### 6. WindowAdaptation

**Purpose**: Dynamic window size adaptation for attention mechanisms.

**Key Features**:
- Adaptive window sizing based on sequence complexity
- Entropy-based window adjustment
- Performance vs quality tradeoff management
- Gradient-aware adaptation

## Transformer Block Architecture

The new `TransformerBlock` uses these components in a modular composition:

```
TransformerBlock {
    pre_attention_norm: NormalizationLayer,
    temporal_mixing: TemporalMixingWrapper,  // Attention or RG-LRU
    pre_ffn_norm: NormalizationLayer,
    feedforward: FeedforwardProcessor,       // GLU, MoE, etc.
    attention_context: AttentionContext,
    residual_connections: [ResidualConnection; 2],
    window_adaptation: WindowAdaptation,
}
```

### Forward Pass

```rust
fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
    // Pre-attention normalization
    let norm1 = self.pre_attention_norm.forward(input.clone());
    
    // Temporal mixing (attention or RG-LRU)
    let attn_out = self.temporal_mixing.forward(norm1);
    
    // Residual connection 1
    let residual1 = self.residual_connections[0].combine(input, attn_out);
    
    // Pre-FFN normalization
    let norm2 = self.pre_ffn_norm.forward(residual1.clone());
    
    // Feedforward processing
    let ffn_out = self.feedforward.forward(norm2);
    
    // Residual connection 2
    let output = self.residual_connections[1].combine(residual1, ffn_out);
    
    // Update attention context
    self.attention_context.update(&input, &output);
    
    output
}
```

## Speculative Sampling

### Overview

Speculative sampling is a decoding acceleration technique that uses a draft model to propose multiple tokens, which are then verified by the full model. This reduces the number of full model evaluations required.

### Implementation

The speculative sampling system supports two modes:

#### 1. Transformer Mode

**Key Features**:
- Draft model: Reduced-layer transformer (configurable depth)
- Verification model: Full transformer
- Gamma (γ): Number of speculative steps
- Tau (τ): Acceptance threshold (probability-based)

**Algorithm**:
```
1. Draft model generates γ candidate tokens
2. Full model evaluates all γ candidates in parallel
3. Accept tokens where verification probability > τ
4. Reject and regenerate tokens where probability ≤ τ
5. Advance by number of accepted tokens
```

#### 2. Diffusion Mode

**Key Features**:
- Draft model: Simplified diffusion process
- Verification model: Full diffusion model
- Gamma (γ): Number of denoising steps to speculate
- Tau (τ): Acceptance threshold (MSE-based)

**Algorithm**:
```
1. Draft model performs γ denoising steps
2. Full model evaluates the speculated denoising trajectory
3. Accept steps where MSE < τ
4. Reject and re-denoise steps where MSE ≥ τ
5. Continue from last accepted state
```

### Configuration

```rust
pub struct SpeculativeSamplingConfig {
    pub gamma: usize,          // Number of speculative steps (4-8 typical)
    pub tau: f32,             // Acceptance threshold (0.01-0.1 typical)
    pub draft_layers: usize,   // Depth of draft model (2-4 typical)
    pub temperature: f32,      // Sampling temperature (1.0 = no modification)
    pub top_p: f32,            // Nucleus sampling threshold (1.0 = disable)
}
```

### Performance Characteristics

**Speedup**: Typically 2-4x decoding speed improvement
**Memory**: Additional memory for draft model states
**Quality**: Minimal impact on output quality when properly tuned

### Tuning Guidelines

1. **Gamma (γ)**: Start with 4, increase for longer sequences
2. **Tau (τ)**: Start with 0.01, adjust based on acceptance rate (target 70-90%)
3. **Draft Layers**: 2-4 layers typically sufficient for good draft quality
4. **Temperature**: Use 0.8-1.0 for balanced diversity vs quality

## Integration with Transformer Block

The speculative sampling system integrates at the decoding level:

```
// Standard decoding
let next_token = transformer.decode(current_state);

// Speculative decoding
let (accepted_tokens, new_state) = speculative_sampler.decode(
    current_state, 
    &transformer, 
    &draft_transformer
);
```

## Performance Optimizations

### 1. Cached Intermediates

- Zero-copy sharing of input tensors using `Arc<Array2<f32>>`
- Eliminates O(seq_len × embed_dim) clones per forward pass
- Thread-safe access for parallel gradient computation

### 2. Gradient Partitioning

- Pre-computed parameter partition sizes
- Efficient gradient routing to appropriate optimizers
- Reduces gradient application overhead

### 3. Similarity Context

- Learned similarity context strength
- Cross-layer information flow without additional parameters
- Improves convergence in deep networks

### 4. Window Adaptation

- Dynamic window sizing based on sequence entropy
- Reduces computation for simple sequences
- Maintains quality for complex patterns

## Benchmarking

### Attention Performance

```bash
# Run attention comparison benchmark
cargo run --release --bin bench_attention_compare
```

### Transformer Throughput

```bash
# Run transformer throughput benchmark
cargo run --release --bin bench_transformer
```

### Speculative Sampling Evaluation

```bash
# Evaluate speculative sampling speedup
cargo run --release -- --speculative --speculative-mode transformer --eval-only
```

## Future Enhancements

### 1. Mixed Precision Support
- Feature-flagged f16/bf16 storage for key parameters
- Reduced memory bandwidth and storage requirements

### 2. Kernel Fusion
- Fuse score computation and polynomial evaluation
- Reduce memory traffic in attention hot paths

### 3. Adaptive Architecture
- Dynamic selection between attention and RG-LRU based on sequence characteristics
- Per-layer architecture specialization

### 4. Advanced Caching
- Thread-local storage buffers for attention intermediates
- Reuse allocations across multiple forward passes

## References

- **Transformer Architecture**: Vaswani et al., "Attention is All You Need" (2017)
- **Speculative Sampling**: Leviathan et al., "Fast Inference from Transformers" (2022)
- **RG-LRU**: Orvieto et al., "Resurrecting Recurrent Neural Networks" (2023)
- **Modular Design Patterns**: Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software"

## API Documentation

For detailed API documentation, see the Rustdoc-generated documentation:

```bash
cargo doc --open
```

Then navigate to the `layers::transformer` module for complete component documentation.