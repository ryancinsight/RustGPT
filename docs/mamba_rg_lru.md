# Mamba and RG-LRU Documentation

## Overview

This document provides comprehensive documentation for the Mamba and RG-LRU (Real-Gated Linear Recurrent Unit) implementations in RustGPT. These state-space models provide efficient alternatives to transformer attention for sequence processing.

## Mamba Architecture

### Core Concepts

Mamba is a selective state-space model that combines:
1. **Input-dependent parameterization** for dynamic adaptation
2. **Selective scan mechanism** for efficient sequence processing
3. **Causal convolution** for local context integration
4. **Hardware-aware parallel scans** for efficient computation

### Mathematical Formulation

#### Input Projection

```
// Combined projection to (u, gate) space
[U, G] = X · W_in + b_in
where:
- U ∈ ℝ^{T×D} : input-dependent projection
- G ∈ ℝ^{T×D} : gating signal
- W_in ∈ ℝ^{D×2D} : input projection matrix
```

#### Causal Convolution

```
// Depthwise convolution on U
U_conv = DepthwiseConv1D(U, W_conv)
where:
- W_conv ∈ ℝ^{K×D} : convolution kernel
- K : kernel size (typically 3-5)
```

#### State-Space Parameters

```
// Input-dependent parameters
Δ = softplus(X · W_Δ + b_Δ)  // Time step
B = X · W_B + b_B            // Input projection
C = X · W_C + b_C            // Output projection

// Fixed diagonal state matrix
A = -softplus(A_log)         // Stable diagonal matrix
```

#### Selective Scan

```
// Discretized state-space representation
Ã = exp(Δ · A) ∈ ℝ^{D×D}
B̃ = (Δ · B) · inv(Δ · A) ∈ ℝ^{D×D}

// Recurrent state update
H_t = Ã · H_{t-1} + B̃ · U_conv[t]

// Gated output
Y_t = C · H_t ⊙ σ(G_t)
```

#### Final Projection

```
// Output projection with skip connection
Y = [Y_1, Y_2, ..., Y_T] · W_out + D · X
where:
- W_out ∈ ℝ^{D×D} : output projection
- D ∈ ℝ^{D} : learned skip coefficients
```

### Implementation Details

#### Memory Layout

```rust
struct Mamba {
    embed_dim: usize,
    conv_kernel: usize,
    
    // Projection weights
    w_in: Array2<f32>,      // [D, 2D]
    b_in: Array2<f32>,      // [1, 2D]
    
    // State-space parameters
    w_dt: Array2<f32>,      // [D, D]
    b_dt: Array2<f32>,      // [1, D]
    w_b: Array2<f32>,       // [D, D]
    b_b: Array2<f32>,       // [1, D]
    w_c: Array2<f32>,       // [D, D]
    b_c: Array2<f32>,       // [1, D]
    
    // Diagonal state matrix
    a_log: Array2<f32>,     // [1, D]
    
    // Skip connection
    d_skip: Array2<f32>,    // [1, D]
    
    // Convolution
    conv_w: Array2<f32>,    // [K, D]
    conv_b: Array2<f32>,    // [1, D]
    
    // Output projection
    w_out: Array2<f32>,     // [D, D]
    b_out: Array2<f32>,     // [1, D]
}
```

#### Forward Pass

```rust
fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
    // Cache input for gradient computation
    self.cached_input = Some(input.clone());
    
    // Input projection: [T, D] -> [T, 2D]
    let proj = input.dot(&self.w_in) + &self.b_in;
    let (u_pre, gate_logits) = proj.split_at(Axis(1), self.embed_dim);
    
    // Causal convolution
    let u_conv = self.apply_conv(u_pre);
    
    // State-space parameters
    let (dt, b, c) = self.compute_ssm_params(&input);
    let a = self.compute_a();
    
    // Selective scan
    let h = self.selective_scan(&u_conv, &dt, &a, &b);
    
    // Gated output
    let gate = sigmoid_f32(&gate_logits);
    let y = h * gate;
    
    // Final projection
    let output = y.dot(&self.w_out) + &self.b_out;
    
    // Skip connection
    output + &self.d_skip * &input
}
```

### Performance Characteristics

#### Time Complexity
- **Convolution**: O(T × K × D)
- **Parameter Computation**: O(T × D²)
- **Selective Scan**: O(T × D²) with parallel scan optimization
- **Overall**: O(T × D²) - linear in sequence length

#### Memory Usage
- **Parameters**: ~12D² (input proj + conv + SSM params + output proj)
- **Activation Memory**: O(T × D) for intermediate states
- **Cache**: O(T × D) for gradient computation

#### Hardware Efficiency
- **Parallel Scan**: GPU-friendly parallel prefix sum implementation
- **Memory Access**: Sequential memory patterns for good cache utilization
- **FLOPs**: ~24D² per token (competitive with attention)

### Training Considerations

#### Initialization
- **A_log**: Initialize to small positive values for stable A matrix
- **Projection weights**: Xavier initialization for balanced gradients
- **Convolution**: Small random initialization to avoid oversmoothing

#### Gradient Flow
- **Skip connection**: Ensures gradient flow through depth
- **Gating**: Provides nonlinearity while maintaining gradient magnitude
- **State matrix**: Stable gradients due to diagonal structure

#### Regularization
- **Weight decay**: Apply to all parameters except A_log
- **Gradient clipping**: Essential for stable training
- **Learning rate**: Typically 1-3× higher than transformers

## RG-LRU Architecture

### Core Concepts

RG-LRU (Real-Gated Linear Recurrent Unit) is a simplified recurrent architecture that:
1. Uses **diagonal recurrence** for stability and efficiency
2. Incorporates **learnable gating** for dynamic control
3. Maintains **linear complexity** in sequence length
4. Provides **trainable temporal mixing** as alternative to attention

### Mathematical Formulation

#### Gating Mechanism

```
// Reset and input gates
r_t = σ(X_t · W_a + b_a) ∈ ℝ^D
i_t = σ(X_t · W_x + b_x) ∈ ℝ^D

// Diagonal recurrence parameter
a_t = σ(λ) ∈ ℝ^D
```

#### Recurrent State Update

```
// Gated recurrence relation
H_t = a_t ⊙ H_{t-1} + (1 - a_t) ⊙ (r_t ⊙ H_{t-1} + i_t ⊙ X_t)

// Simplified form
H_t = (a_t + (1 - a_t) ⊙ r_t) ⊙ H_{t-1} + (1 - a_t) ⊙ i_t ⊙ X_t
```

#### Output Projection

```
// Final output with optional projection
Y = H_T · W_out
```

### Implementation Details

#### Memory Layout

```rust
struct RgLru {
    embed_dim: usize,
    
    // Gate parameters
    w_a: Array2<f32>,      // [D, D] - reset gate weights
    b_a: Array2<f32>,      // [1, D] - reset gate bias
    w_x: Array2<f32>,      // [D, D] - input gate weights
    b_x: Array2<f32>,      // [1, D] - input gate bias
    
    // Diagonal recurrence
    lambda: Array2<f32>,   // [1, D] - recurrence parameter
    
    // Output projection (optional)
    w_out: Array2<f32>,    // [D, D]
}
```

#### Forward Pass

```rust
fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
    let (t, d) = input.dim();
    
    // Cache input
    self.cached_input = Some(input.clone());
    
    // Compute gates
    let r = sigmoid_f32(&(input.dot(&self.w_a) + &self.b_a));
    let i = sigmoid_f32(&(input.dot(&self.w_x) + &self.b_x));
    
    // Compute diagonal recurrence
    let a = sigmoid_f32(&self.lambda);
    
    // Initialize hidden state
    let mut h_prev = Array2::zeros((1, d));
    let mut h_sequence = Vec::with_capacity(t);
    
    // Recurrent processing
    for t_idx in 0..t {
        let x_t = input.row(t_idx);
        let r_t = r.row(t_idx);
        let i_t = i.row(t_idx);
        
        // State update
        let h_t = &a * &h_prev * &r_t + (&Array2::ones(a.raw_dim()) - &a) * &i_t * &x_t;
        
        h_sequence.push(h_t.clone());
        h_prev = h_t;
    }
    
    // Stack and project
    let h_stacked = ndarray::stack(Axis(0), &h_sequence).unwrap();
    h_stacked.dot(&self.w_out)
}
```

### Multi-head RG-LRU (MoH-RG-LRU)

#### Architecture

```
// Split input into heads
X_h = split(X, num_heads) for h = 1..H

// Per-head processing
H_h = RG-LRU_h(X_h) for h = 1..H

// MoH gating
E = MoHGating(X) ∈ ℝ^H

// Weighted combination
Y = ∑_{h=1}^H E_h ⊙ H_h
```

#### Implementation

```rust
struct MoHRgLru {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    
    moh: MoHGating,          // Mixture-of-Heads gating
    heads: Vec<RgLru>,       // Per-head RG-LRU layers
    
    // Activity tracking
    last_avg_active_heads: Option<f32>,
    last_head_activity_vec: Option<Vec<f32>>,
}
```

#### Forward Pass

```rust
fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
    // Cache input
    self.cached_input = Some(input.clone());
    
    // Compute MoH gating
    let eff_weights = self.moh.forward(&input);
    self.cached_eff = Some(eff_weights.clone());
    
    // Split input into heads
    let head_inputs = self.split_input(&input);
    
    // Process each head
    let mut head_outputs = Vec::new();
    for (h_idx, head) in self.heads.iter_mut().enumerate() {
        let head_input = &head_inputs[h_idx];
        let head_out = head.forward(head_input.clone());
        head_outputs.push(head_out);
    }
    
    self.cached_head_out = Some(head_outputs.clone());
    
    // Weighted combination
    self.combine_heads(&head_outputs, &eff_weights)
}
```

### Performance Characteristics

#### Time Complexity
- **Single-head RG-LRU**: O(T × D²)
- **MoH-RG-LRU**: O(T × D²) (same as single-head, but with head parallelism)
- **Memory**: O(T × D) for recurrent states

#### Advantages over Attention
- **Linear complexity**: O(T) vs O(T²) for attention
- **Stable gradients**: Diagonal recurrence prevents exploding gradients
- **Memory efficiency**: No attention matrix storage
- **Parallelism**: Head-level parallelism in MoH variant

#### Tradeoffs
- **Context mixing**: Less expressive than full attention
- **Long-range dependencies**: May require deeper stacking
- **Parameter efficiency**: Fewer parameters than attention

## Integration with Transformer Architecture

### Temporal Mixing Wrapper

The `TemporalMixingWrapper` enum allows seamless integration:

```rust
enum TemporalMixingLayer {
    Attention(PolyAttention),
    RgLru(RgLru),
    MoHRgLru(MoHRgLru),
    Mamba(Mamba),
}
```

### Transformer Block Usage

```rust
struct TransformerBlock {
    pre_attention_norm: RichardsNorm,
    temporal_mixing: TemporalMixingLayer,  // Can be attention or RG-LRU
    pre_ffn_norm: RichardsNorm,
    feedforward: FeedForwardVariant,
    // ... other components
}
```

### Configuration

```rust
// Choose temporal mixing in config
let config = TransformerBlockConfig {
    temporal_mixing: TemporalMixingType::RgLru,
    // or TemporalMixingType::Mamba
    // or TemporalMixingType::MoHRgLru { num_heads: 4 }
    // ... other config
};
```

## Benchmarking and Performance

### Attention vs RG-LRU Comparison

```bash
# Benchmark attention performance
cargo run --release --bin bench_attention_compare

# Benchmark transformer with RG-LRU
cargo run --release --bin bench_transformer -- --architecture rg-lru
```

### Expected Performance

| Architecture | Time Complexity | Memory | Parameters | Best For |
|--------------|----------------|--------|------------|----------|
| Attention | O(T²D) | High | High | Complex patterns, long-range dependencies |
| RG-LRU | O(TD²) | Medium | Medium | Efficient processing, stable training |
| MoH-RG-LRU | O(TD²) | Medium | High | Balanced efficiency and capacity |
| Mamba | O(TD²) | High | High | Hardware-efficient, high-quality outputs |

### Training Recommendations

#### RG-LRU Specific
- **Learning rate**: 1-2× higher than attention (better gradient flow)
- **Batch size**: Can be larger due to memory efficiency
- **Sequence length**: Works well with longer sequences (1024+)
- **Depth**: May need more layers for same capacity as attention

#### Mamba Specific
- **Initialization**: Critical for stable training
- **Gradient clipping**: Essential (clip norm ~1.0-2.0)
- **Warmup**: Longer warmup period recommended
- **Regularization**: Moderate weight decay (1e-4 to 1e-3)

## Future Enhancements

### RG-LRU Improvements

1. **Parallel Scan Implementation**: GPU-friendly parallel recurrence
2. **Mixed Precision**: FP16/bfloat16 support for parameters
3. **Adaptive Gating**: Learnable gate combinations
4. **Hierarchical RG-LRU**: Multi-scale temporal processing

### Mamba Improvements

1. **Block-diagonal A**: More expressive state mixing
2. **Multi-dimensional gating**: Enhanced control
3. **Memory-efficient scan**: Reduced activation memory
4. **Fused operations**: Kernel fusion for better performance

### Hybrid Architectures

1. **Attention + RG-LRU**: Combine strengths of both approaches
2. **Adaptive mixing**: Dynamic selection based on input characteristics
3. **Layer-wise specialization**: Different mechanisms per layer
4. **Progressive refinement**: RG-LRU draft + attention verification

## References

### Mamba
- **Original Paper**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- **Key Insight**: Hardware-aware parallel scan for efficient SSM computation
- **Implementation**: Reference CPU-friendly implementation with causal convolution

### RG-LRU
- **Original Paper**: Orvieto et al., "Resurrecting Recurrent Neural Networks for Long Sequences" (2023)
- **Key Insight**: Diagonal recurrence with learned gating for stable training
- **Advantages**: Linear complexity with transformer-comparable quality

### State-Space Models
- **Foundations**: HiPPO theory for continuous-time sequence modeling
- **Discretization**: Zero-order hold (ZOH) for stable discretization
- **Selective Mechanisms**: Input-dependent parameterization for adaptivity

## API Documentation

For detailed API documentation, see the Rustdoc-generated documentation:

```bash
cargo doc --open
```

Navigate to:
- `layers::ssm::mamba` for Mamba implementation
- `layers::ssm::rg_lru` for RG-LRU implementation
- `layers::transformer::block` for integration details

## Example Usage

### Using RG-LRU in Transformer

```rust
use crate::layers::ssm::rg_lru::RgLru;
use crate::layers::transformer::block::TransformerBlock;

// Create RG-LRU layer
let rg_lru = RgLru::new(256); // 256-dimensional

// Create transformer block with RG-LRU
let config = TransformerBlockConfig {
    embed_dim: 256,
    temporal_mixing: TemporalMixingType::RgLru,
    // ... other config
};

let block = TransformerBlock::new(&config);

// Forward pass
let input = Array2::zeros((128, 256)); // 128 tokens, 256 dim
let output = block.forward(input);
```

### Using Mamba

```rust
use crate::layers::ssm::mamba::Mamba;

// Create Mamba layer
let mamba = Mamba::new(256, 3); // 256 dim, kernel size 3

// Forward pass
let input = Array2::zeros((128, 256));
let output = mamba.forward(input);
```

### Using MoH-RG-LRU

```rust
use crate::layers::ssm::rg_lru::MoHRgLru;
use crate::mixtures::HeadSelectionStrategy;

// Create multi-head RG-LRU
let moh_rg_lru = MoHRgLru::new(
    256,           // embed_dim
    4,             // num_heads
    &HeadSelectionStrategy::Learned, // gating strategy
);

// Forward pass
let input = Array2::zeros((128, 256));
let output = moh_rg_lru.forward(input);
```

## Troubleshooting

### Common Issues

#### Training Instability
- **Symptom**: NaN gradients or exploding loss
- **Solution**: Reduce learning rate, enable gradient clipping, check initialization

#### Poor Convergence
- **Symptom**: Slow learning or plateauing loss
- **Solution**: Increase learning rate, try different initialization, add more layers

#### Memory Issues
- **Symptom**: Out of memory errors
- **Solution**: Reduce batch size, use smaller embed_dim, enable gradient checkpointing

#### Performance Issues
- **Symptom**: Slow training or inference
- **Solution**: Enable release mode, check for unnecessary allocations, profile hot paths

### Debugging Tips

1. **Gradient Monitoring**: Check gradient norms during training
2. **Activation Analysis**: Monitor activation distributions
3. **Memory Profiling**: Use `heaptrack` or similar tools
4. **Performance Profiling**: Use `perf` or `vtune` for hot spot analysis

## Conclusion

Mamba and RG-LRU provide powerful alternatives to transformer attention, offering:
- **Linear time complexity** for efficient sequence processing
- **Stable training** with good gradient properties
- **Hardware efficiency** with parallel-friendly operations
- **Flexible integration** with existing transformer architecture

These models enable efficient scaling to longer sequences and larger models while maintaining high quality outputs.