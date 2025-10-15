# Quick Start: Using HyperMixer Architecture

## TL;DR

To switch from Transformer to HyperMixer, change one line in `src/main_hyper.rs`:

```rust
// Line ~30
let architecture = ArchitectureType::HyperMixer; // Change from Transformer
```

That's it! The rest is automatic.

## What is HyperMixer?

HyperMixer is an MLP-based alternative to Transformers that:
- Uses **hypernetworks** to generate token-mixing weights dynamically
- Has **linear complexity** O(n) instead of quadratic O(n²)
- Is more **parameter efficient** than standard Transformers
- Works better than static MLPMixer because weights adapt to input

## When to Use HyperMixer vs Transformer

### Use HyperMixer When:
- ✅ You need to process **long sequences** (where sequence length > embedding dimension)
- ✅ You want **lower memory usage**
- ✅ You need **faster inference** on long sequences
- ✅ You want to **experiment** with MLP-based architectures

### Use Transformer When:
- ✅ You need **proven performance** on standard benchmarks
- ✅ You're working with **short sequences**
- ✅ You want to leverage **pre-trained models**
- ✅ You need **attention visualizations**

## Step-by-Step Guide

### 1. Choose Your Architecture

Edit `src/main_hyper.rs` around line 30:

```rust
// Option A: Use Transformer (default)
let architecture = ArchitectureType::Transformer;

// Option B: Use HyperMixer
let architecture = ArchitectureType::HyperMixer;
```

### 2. Configure Hyperparameters

The configuration is created automatically based on your choice:

```rust
let config = match architecture {
    ArchitectureType::Transformer => {
        ModelConfig::transformer(
            EMBEDDING_DIM,  // e.g., 128
            HIDDEN_DIM,     // e.g., 256
            3,              // num_layers
            MAX_SEQ_LEN,    // e.g., 80
        )
    }
    ArchitectureType::HyperMixer => {
        ModelConfig::hypermixer(
            EMBEDDING_DIM,  // e.g., 128
            HIDDEN_DIM,     // e.g., 256
            3,              // num_layers
            MAX_SEQ_LEN,    // e.g., 80
            None,           // hypernetwork_hidden_dim (None = auto)
        )
    }
};
```

### 3. Run Your Code

```bash
# Build and run
cargo run --release --bin llm

# Or just build
cargo build --release
```

### 4. Compare Results

The model will print architecture information:

```
=== Model Architecture Summary ===
Architecture Type: HyperMixer
Embedding Dimension: 128
Hidden Dimension: 256
Number of Layers: 3
Max Sequence Length: 80
Hypernetwork Hidden Dim: 32

Layer Stack:
  0: Embeddings
  1: HyperMixerBlock
  2: HyperMixerBlock
  3: HyperMixerBlock
  4: OutputProjection

Total Parameters: XXXXX
==================================
```

## Advanced Configuration

### Tuning Hypernetwork Size

The hypernetwork size affects model capacity and speed:

```rust
// Small hypernetwork (faster, less expressive)
ModelConfig::hypermixer(128, 256, 3, 80, Some(16))

// Medium hypernetwork (balanced) - DEFAULT
ModelConfig::hypermixer(128, 256, 3, 80, None) // Auto: embedding_dim / 4

// Large hypernetwork (slower, more expressive)
ModelConfig::hypermixer(128, 256, 3, 80, Some(64))
```

**Rule of thumb**: 
- Small: `embedding_dim / 8`
- Medium: `embedding_dim / 4` (default)
- Large: `embedding_dim / 2`

### Adjusting Layer Count

```rust
// Shallow model (faster training)
ModelConfig::hypermixer(128, 256, 2, 80, None)

// Medium model (balanced)
ModelConfig::hypermixer(128, 256, 3, 80, None)

// Deep model (more capacity)
ModelConfig::hypermixer(128, 256, 6, 80, None)
```

### Sequence Length Considerations

HyperMixer becomes more efficient as sequence length increases:

```rust
// Short sequences (< 50 tokens): Transformer may be better
ModelConfig::hypermixer(128, 256, 3, 50, None)

// Medium sequences (50-200 tokens): Both work well
ModelConfig::hypermixer(128, 256, 3, 100, None)

// Long sequences (> 200 tokens): HyperMixer shines
ModelConfig::hypermixer(128, 256, 3, 512, None)
```

## Performance Tips

### Memory Optimization
- HyperMixer uses less memory than Transformer
- For very long sequences, reduce `hidden_dim` instead of `embedding_dim`

### Training Speed
- HyperMixer is faster for sequences where `seq_len > embedding_dim`
- Use smaller hypernetwork for faster training
- Batch size can often be larger with HyperMixer

### Learning Rate
- Start with same learning rate as Transformer
- HyperMixer may benefit from slightly higher learning rates
- Use learning rate warmup for stability

## Troubleshooting

### Model Not Learning
- **Check**: Is your hypernetwork too small? Try increasing it
- **Check**: Is your learning rate appropriate? Try 0.001 for pre-training
- **Check**: Are gradients flowing? Add gradient clipping if needed

### Out of Memory
- **Reduce**: `hidden_dim` (e.g., from 512 to 256)
- **Reduce**: `num_layers` (e.g., from 6 to 3)
- **Reduce**: batch size
- **Note**: HyperMixer should use less memory than Transformer

### Slow Training
- **Reduce**: `hypernetwork_hidden_dim`
- **Reduce**: `hidden_dim`
- **Use**: Smaller batch size with gradient accumulation

## Comparison Checklist

When comparing Transformer vs HyperMixer:

- [ ] Use **same hyperparameters** (embedding_dim, hidden_dim, num_layers)
- [ ] Train for **same number of epochs**
- [ ] Use **same learning rate schedule**
- [ ] Test on **same dataset**
- [ ] Measure **training time**
- [ ] Measure **inference time**
- [ ] Measure **memory usage**
- [ ] Compare **final accuracy/loss**

## Example Configurations

### Small Model (Fast Experimentation)
```rust
// Transformer
ModelConfig::transformer(64, 128, 2, 50)

// HyperMixer
ModelConfig::hypermixer(64, 128, 2, 50, Some(16))
```

### Medium Model (Balanced)
```rust
// Transformer
ModelConfig::transformer(128, 256, 3, 80)

// HyperMixer
ModelConfig::hypermixer(128, 256, 3, 80, None)
```

### Large Model (High Capacity)
```rust
// Transformer
ModelConfig::transformer(256, 512, 6, 128)

// HyperMixer
ModelConfig::hypermixer(256, 512, 6, 128, Some(64))
```

## Next Steps

1. **Experiment**: Try both architectures on your task
2. **Benchmark**: Measure speed and memory usage
3. **Tune**: Adjust hyperparameters for your use case
4. **Document**: Record which architecture works better for your data

## Resources

- **Paper**: [HyperMixer: An MLP-based Low Cost Alternative to Transformers](https://arxiv.org/abs/2203.03691)
- **Full Documentation**: See `docs/HYPERMIXER_ARCHITECTURE.md`
- **Implementation Details**: See `docs/REFACTORING_SUMMARY.md`

## Getting Help

If you encounter issues:
1. Check the documentation in `docs/`
2. Review the code comments in `src/hypermixer.rs`
3. Run tests: `cargo test`
4. Check compilation: `cargo check`

## Contributing

Found a bug or have an improvement? Contributions welcome!
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass: `cargo test`

