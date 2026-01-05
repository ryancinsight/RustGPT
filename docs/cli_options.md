# CLI Options and Training Features Documentation

## Overview

This document provides comprehensive documentation for all CLI options and training features available in RustGPT, including the latest additions for architecture selection, speculative sampling, and deterministic training.

## CLI Structure

The CLI is built using `clap` and provides a structured interface for configuring training runs. The main entry point is in `src/cli.rs`.

## Basic Usage

```bash
# Show help
cargo run --release -- --help

# Basic training
cargo run --release

# Training with specific architecture
cargo run --release -- --architecture transformer
```

## Architecture Selection

### `--architecture` Option

Selects the base architecture for the model.

**Options**:
- `transformer` (default): Standard transformer architecture
- `trm`: Transformer-Recurrent Mixture
- `diffusion`: Diffusion model (transformer-based)
- `mamba`: Mamba state-space model
- `rg-lru`: RG-LRU recurrent architecture
- `moh-rg-lru`: Multi-head RG-LRU with Mixture-of-Heads

**Important Compatibility Notes**:
- **Diffusion training (`--diffusion`) only works with transformer-based architectures**
- **Mamba/RG-LRU architectures are not compatible with diffusion training**
- For SSM + diffusion, use: `--architecture transformer --temporal-mixing mamba --diffusion`

**Examples**:
```bash
# Use Mamba architecture (pure SSM, no diffusion)
cargo run --release -- --architecture mamba

# Use RG-LRU architecture (pure recurrent, no diffusion)
cargo run --release -- --architecture rg-lru

# Use Multi-head RG-LRU
cargo run --release -- --architecture moh-rg-lru

# Use diffusion with transformer
cargo run --release -- --architecture diffusion

# Use transformer with Mamba temporal mixing + diffusion
cargo run --release -- --architecture transformer --temporal-mixing mamba --diffusion
```

### `--temporal-mixing` Option

Configures the temporal mixing mechanism within transformer blocks.

**Options**:
- `attention` (default): Standard polynomial attention
- `rg-lru`: RG-LRU temporal mixing
- `mamba`: Mamba temporal mixing

**Examples**:
```bash
# Use RG-LRU temporal mixing in transformer blocks
cargo run --release -- --temporal-mixing rg-lru

# Use Mamba temporal mixing
cargo run --release -- --temporal-mixing mamba
```

## Speculative Sampling

### `--speculative` Flag

Enables speculative sampling for accelerated decoding.

**Behavior**:
- Creates a draft model with reduced depth
- Uses draft model to propose multiple tokens
- Verifies proposals with full model
- Accepts/rejects based on threshold

**Examples**:
```bash
# Enable speculative sampling (default mode)
cargo run --release -- --speculative

# Disable speculative sampling
cargo run --release  # no --speculative flag
```

### `--speculative-mode` Option

Selects the speculative sampling mode.

**Options**:
- `diffusion` (default): Speculative sampling for diffusion models
- `transformer`: Speculative sampling for transformer models

**Examples**:
```bash
# Transformer speculative sampling
cargo run --release -- --speculative --speculative-mode transformer

# Diffusion speculative sampling
cargo run --release -- --speculative --speculative-mode diffusion
```

### Speculative Sampling Configuration

Additional options for fine-tuning speculative sampling:

```bash
# Configure speculative sampling parameters
cargo run --release -- --speculative --gamma 4 --tau 0.01 --draft-layers 2
```

**Parameters**:
- `--gamma`: Number of speculative steps (default: 4)
- `--tau`: Acceptance threshold (default: 0.01)
- `--draft-layers`: Depth of draft model (default: 2)

## Training Configuration

### `--epochs` Option

Sets the number of training epochs.

**Default**: 100
**Range**: 1-1000

**Examples**:
```bash
# Train for 50 epochs
cargo run --release -- --epochs 50

# Train for 200 epochs
cargo run --release -- --epochs 200
```

### `--batch-size` Option

Sets the batch size for training.

**Default**: 32
**Range**: 1-256

**Examples**:
```bash
# Use batch size 64
cargo run --release -- --batch-size 64

# Use batch size 16
cargo run --release -- --batch-size 16
```

### `--learning-rate` Option

Sets the base learning rate.

**Default**: 0.001
**Range**: 0.0001-0.1

**Examples**:
```bash
# Use learning rate 0.0005
cargo run --release -- --learning-rate 0.0005

# Use learning rate 0.002
cargo run --release -- --learning-rate 0.002
```

### `--seed` Option

Sets a fixed random seed for reproducible training.

**Behavior**:
- Seeds all RNG instances
- Forces single-threaded Rayon pool for deterministic parallel execution
- Ensures reproducible results across runs

**Examples**:
```bash
# Deterministic training with seed 42
cargo run --release -- --seed 42

# Deterministic training with seed 123
cargo run --release -- --seed 123
```

**Note**: When using `--seed`, training will be single-threaded to ensure complete determinism, which may impact performance.

## Model Persistence

### `--continue-from` Option

Loads a model from disk to continue training.

**Examples**:
```bash
# Continue training from saved model
cargo run --release -- --continue-from models/rustgpt.bin

# Continue from specific path
cargo run --release -- --continue-from path/to/model.bin
```

### `--save-every` Option

Configures how often to save model checkpoints.

**Default**: 10 (save every 10 epochs)
**Range**: 1-100

**Examples**:
```bash
# Save every 5 epochs
cargo run --release -- --save-every 5

# Save every 20 epochs
cargo run --release -- --save-every 20
```

## Evaluation and Interactive Mode

### `--interactive` Flag

Enables interactive mode after training for manual testing.

**Behavior**:
- Trains the model normally
- Enters interactive prompt loop after training
- Allows manual input and model responses

**Examples**:
```bash
# Train and enter interactive mode
cargo run --release -- --interactive

# Train with specific config and enter interactive mode
cargo run --release -- --architecture mamba --interactive
```

### `--eval-only` Flag

Runs evaluation without training.

**Behavior**:
- Loads model if `--continue-from` is specified
- Runs evaluation metrics
- Exits without training

**Examples**:
```bash
# Evaluate saved model
cargo run --release -- --continue-from models/rustgpt.bin --eval-only

# Evaluate with speculative sampling
cargo run --release -- --continue-from models/rustgpt.bin --speculative --eval-only
```

## Advanced Configuration

### `--embed-dim` Option

Sets the embedding dimension.

**Default**: 128
**Range**: 64-512
**Must be divisible by**: number of heads

**Examples**:
```bash
# Use 256-dimensional embeddings
cargo run --release -- --embed-dim 256

# Use 64-dimensional embeddings
cargo run --release -- --embed-dim 64
```

### `--hidden-dim` Option

Sets the hidden dimension for feedforward networks.

**Default**: 256
**Range**: 128-1024

**Examples**:
```bash
# Use 512-dimensional hidden layer
cargo run --release -- --hidden-dim 512

# Use 128-dimensional hidden layer
cargo run --release -- --hidden-dim 128
```

### `--num-heads` Option

Sets the number of attention heads.

**Default**: 8
**Range**: 1-16
**Constraint**: Must divide embed-dim evenly

**Examples**:
```bash
# Use 4 attention heads
cargo run --release -- --num-heads 4

# Use 16 attention heads
cargo run --release -- --num-heads 16
```

### `--num-layers` Option

Sets the number of transformer layers.

**Default**: 6
**Range**: 1-24

**Examples**:
```bash
# Use 12 transformer layers
cargo run --release -- --num-layers 12

# Use 3 transformer layers
cargo run --release -- --num-layers 3
```

## Mixture of Experts Configuration

### `--use-moe` Flag

Enables Mixture of Experts in feedforward networks.

**Examples**:
```bash
# Enable MoE
cargo run --release -- --use-moe

# Enable MoE with specific architecture
cargo run --release -- --architecture transformer --use-moe
```

### `--num-experts` Option

Sets the number of experts in MoE.

**Default**: 4
**Range**: 2-16

**Examples**:
```bash
# Use 8 experts
cargo run --release -- --use-moe --num-experts 8

# Use 16 experts
cargo run --release -- --use-moe --num-experts 16
```

### `--expert-capacity` Option

Sets the capacity factor for MoE routing.

**Default**: 1.0
**Range**: 0.5-2.0

**Examples**:
```bash
# Use capacity factor 1.5
cargo run --release -- --use-moe --expert-capacity 1.5

# Use capacity factor 0.8
cargo run --release -- --use-moe --expert-capacity 0.8
```

## Training Features

### Adaptive Learning Rate

The training system supports adaptive learning rate scheduling:

- **Warmup**: Linear warmup over first 10% of training
- **Cosine Decay**: Cosine annealing after warmup
- **Layer-wise Scaling**: Automatic learning rate scaling per layer

**Configuration**:
```bash
# Configure learning rate schedule
cargo run --release -- --learning-rate 0.001 --warmup-steps 1000 --min-lr 1e-5
```

### Gradient Clipping

Automatic gradient clipping is enabled by default:

- **Threshold**: 2000.0 (global norm)
- **Behavior**: Clips gradients to prevent exploding updates
- **Configuration**: Adjustable via config

### Mixed Precision Training

**Status**: Experimental (feature flag)

**Enable**:
```bash
# Enable mixed precision (when available)
cargo run --release --features mixed-precision
```

## Observability and Logging

### Logging Configuration

The system uses `tracing` for structured logging:

**Environment Variables**:
```bash
# Set log level
RUST_LOG=debug cargo run --release
RUST_LOG=info cargo run --release   # Default
RUST_LOG=warn cargo run --release   # Warnings only
RUST_LOG=error cargo run --release   # Errors only
```

**Log Directives**:
```bash
# Specific module logging
RUST_LOG=llm::training=debug,llm::attention=info cargo run --release
```

### Training Metrics

**Logged Metrics**:
- Epoch number and progress
- Training loss (cross-entropy)
- Gradient norms (global and per-layer)
- Learning rate (current value)
- Timing information (epoch duration)
- Memory usage (when available)

**Example Output**:
```
INFO  llm::training: Starting pre-training phase
INFO  llm::training: Epoch 1/100 - loss: 2.3456, grad_norm: 0.1234, lr: 0.0008
INFO  llm::training: Epoch 2/100 - loss: 2.1234, grad_norm: 0.0987, lr: 0.0012
INFO  llm::training: Transitioning to instruction tuning phase
```

## Configuration Files

### Model Configuration

The system uses a builder pattern for model configuration:

**Key Components**:
- `ModelConfig`: Top-level configuration
- `TransformerBlockConfig`: Per-block configuration
- `TrainingConfig`: Training hyperparameters

**Example Configuration**:
```rust
let config = ModelConfig {
    architecture: ArchitectureType::Transformer,
    embed_dim: 256,
    hidden_dim: 512,
    num_heads: 8,
    num_layers: 6,
    temporal_mixing: TemporalMixingType::Attention,
    use_moe: true,
    moe_config: Some(ExpertRouterConfig {
        num_experts: 4,
        capacity_factor: 1.0,
    }),
    speculative_config: Some(SpeculativeSamplingConfig {
        gamma: 4,
        tau: 0.01,
        draft_layers: 2,
    }),
};
```

## Best Practices

### Training Stability

1. **Start with smaller models**: Test with `--embed-dim 64 --num-layers 3` before scaling up
2. **Use gradient clipping**: Always enabled by default
3. **Monitor gradient norms**: Watch for exploding gradients
4. **Learning rate tuning**: Start with 0.001 and adjust based on loss curves

### Architecture Selection

| Use Case | Recommended Architecture | Diffusion Compatible? |
|----------|--------------------------|----------------------|
| General purpose | `transformer` | ✅ Yes |
| Efficient processing | `rg-lru` or `moh-rg-lru` | ❌ No |
| Long sequences | `mamba` | ❌ No |
| High quality | `transformer` with MoE | ✅ Yes |
| Experimental | `diffusion` or `trm` | ✅/⚠️ Yes/Experimental |
| SSM + Diffusion | `transformer --temporal-mixing mamba` | ✅ Yes |

### Architecture + Diffusion Compatibility

**✅ Compatible Combinations:**
```bash
# Pure diffusion transformer
cargo run --release -- --architecture diffusion

# Transformer with diffusion training
cargo run --release -- --architecture transformer --diffusion

# Transformer with Mamba temporal mixing + diffusion
cargo run --release -- --architecture transformer --temporal-mixing mamba --diffusion
```

**❌ Incompatible Combinations:**
```bash
# These will fail or produce unexpected results:
cargo run --release -- --architecture mamba --diffusion          # ❌ Mamba != Diffusion
cargo run --release -- --architecture rg-lru --diffusion        # ❌ RG-LRU != Diffusion
cargo run --release -- --architecture moh-rg-lru --diffusion    # ❌ MoH-RG-LRU != Diffusion
```

**⚠️ Experimental Combinations:**
```bash
# May work but not officially supported:
cargo run --release -- --architecture trm --diffusion           # ⚠️ Experimental
```

### Performance Optimization

1. **Batch size**: Larger batches for better GPU utilization
2. **Sequence length**: Match to your data characteristics
3. **Architecture**: Balance quality and efficiency
4. **Speculative sampling**: Enable for faster inference

## Troubleshooting

### Common Issues

#### Out of Memory
- **Solution**: Reduce `--batch-size`, `--embed-dim`, or `--num-layers`
- **Alternative**: Enable gradient checkpointing (when available)

#### Training Divergence
- **Solution**: Reduce `--learning-rate`, enable `--seed` for debugging
- **Check**: Gradient norms in logs

#### Slow Training
- **Solution**: Use `--batch-size 64` or higher, ensure release mode
- **Check**: `RUST_LOG=info` for timing information

#### Poor Quality
- **Solution**: Increase model size, try different architecture
- **Check**: Loss curves and gradient norms

### Debugging Commands

```bash
# Verbose logging for debugging
RUST_LOG=debug cargo run --release -- --seed 42

# Check gradient norms
RUST_LOG=llm::training=debug cargo run --release

# Profile performance
cargo run --release -- --epochs 1  # Single epoch for quick test
```

## Configuration Reference

### Complete Option List

```bash
cargo run --release -- --help
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `RUST_LOG` | Logging level | `RUST_LOG=debug` |
| `RAYON_NUM_THREADS` | Thread pool size | `RAYON_NUM_THREADS=4` |
| `RUST_BACKTRACE` | Backtrace on panic | `RUST_BACKTRACE=1` |

## Examples

### Basic Training

```bash
# Default transformer training
cargo run --release

# With specific seed for reproducibility
cargo run --release -- --seed 42
```

### Architecture Comparison

```bash
# Train transformer
cargo run --release -- --architecture transformer --epochs 50

# Train RG-LRU
cargo run --release -- --architecture rg-lru --epochs 50

# Train Mamba
cargo run --release -- --architecture mamba --epochs 50
```

### Speculative Sampling Evaluation

```bash
# Evaluate speculative sampling speedup
cargo run --release -- --speculative --speculative-mode transformer --eval-only --continue-from models/transformer.bin

# Compare with baseline
cargo run --release -- --eval-only --continue-from models/transformer.bin
```

### MoE Training

```bash
# Train with Mixture of Experts
cargo run --release -- --use-moe --num-experts 8 --expert-capacity 1.5

# Large MoE model
cargo run --release -- --embed-dim 256 --num-layers 12 --use-moe --num-experts 16
```

## Advanced Usage

### Custom Configuration

For advanced users, configuration can be specified programmatically:

```rust
use crate::config_builder::build_model_config;
use crate::cli::Args;

let args = Args {
    architecture: "mamba".to_string(),
    embed_dim: 256,
    hidden_dim: 512,
    num_heads: 8,
    num_layers: 6,
    use_moe: false,
    speculative: true,
    speculative_mode: "transformer".to_string(),
    gamma: 4,
    tau: 0.01,
    draft_layers: 2,
    seed: Some(42),
    // ... other fields
};

let config = build_model_config(&args);
```

### Training Monitoring

For detailed training monitoring:

```bash
# Detailed training logs
RUST_LOG=llm::training=debug,llm::attention=info cargo run --release

# Monitor specific components
RUST_LOG=llm::training=debug,llm::mixtures=debug cargo run --release -- --use-moe
```

## Conclusion

The RustGPT CLI provides a flexible and powerful interface for training and evaluating various neural network architectures. Key features include:

- **Multiple architectures**: Transformer, Mamba, RG-LRU, Diffusion, TRM
- **Speculative sampling**: Accelerated decoding for faster inference
- **Deterministic training**: Reproducible results with `--seed`
- **Modular configuration**: Fine-grained control over model parameters
- **Comprehensive logging**: Detailed observability for debugging and optimization

For the latest options and features, always check:
```bash
cargo run --release -- --help
```