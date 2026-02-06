# E-prop: Eligibility Propagation with ES-D-RTRL

This module implements the Optimized Eligibility Propagation (e-prop) framework enhanced with Exponentially Smoothed Diagonal Approximated Real-Time Recurrent Learning (ES-D-RTRL) for scalable spiking neural networks.

## Overview

ES-D-RTRL achieves **O(N) time and memory complexity** while maintaining 90-99% gradient fidelity to full Backpropagation Through Time (BPTT), making it suitable for training brain-scale models (125k+ neurons).

### Key Features

- **Linear Complexity**: O(N) per timestep vs O(N²) for standard e-prop
- **Biological Plausibility**: Local eligibility traces + global learning signals
- **Online Learning**: Forward-only gradient computation (no backward pass required)
- **SNN Optimized**: Leverages spike sparsity and signed-input properties
- **Scalable**: Supports large-scale neuromorphic models

### Complexity Comparison

| Algorithm   | Memory | Time/Step | BPTT Fidelity |
|-------------|--------|-----------|---------------|
| Full RTRL   | O(N³)  | O(N³)     | 100%          |
| e-prop      | O(N²)  | O(N²)     | 95-98%        |
| **ES-D-RTRL** | **O(N)** | **O(N)** | **90-95%** |
| BPTT        | O(TN²) | O(TN²)    | 100%          |

## Module Structure

The implementation is organized into focused modules for separation of concerns:

```
src/eprop/
├── mod.rs          # Module definition and re-exports
├── config.rs       # Configuration structures (NeuronConfig, EPropConfig)
├── neuron.rs       # Neuron dynamics (LIF/ALIF models)
├── traces.rs       # ES-D-RTRL eligibility trace computation
├── trainer.rs      # Main training engine
└── utils.rs        # Utility functions (outer product, etc.)
```

### Module Responsibilities

- **config**: All configuration parameters for neurons, traces, and training
- **neuron**: Spiking neuron dynamics (LIF/ALIF) with surrogate gradients
- **traces**: ES-D-RTRL implementation (EligibilityTraces, TraceUpdater)
- **trainer**: Complete training loop orchestration and ES-D-RTRL integration
- **utils**: Linear algebra utilities (outer products, clipping, metrics)

## Quick Start

### Basic Usage

```rust
use eprop::{EPropTrainer, EPropConfig, NeuronModel, NeuronConfig};
use ndarray::Array1;

// Configure trainer
let config = EPropConfig {
    num_neurons: 128,
    input_dim: 64,
    output_dim: 10,
    neuron_config: NeuronConfig::lif(), // or NeuronConfig::alif()
    learning_rate: 1e-3,
    num_cycles: 3,
    ..Default::default()
};

// Create trainer
let mut trainer = EPropTrainer::new(config)?;

// Training loop
for (input, target) in dataset {
    let loss = trainer.train_step(&input.view(), &target.view())?;
    println!("Loss: {:.4}", loss);
}

// Check statistics
let stats = trainer.stats();
println!("Avg firing rate: {:.2}%", stats.avg_firing_rate * 100.0);
```

### Advanced Configuration

```rust
use eprop::{EPropConfig, NeuronConfig, NeuronModel};

// ALIF neurons with custom adaptation
let neuron_config = NeuronConfig {
    model: NeuronModel::ALIF,
    alpha: 0.9,        // Membrane decay
    rho: 0.99,         // Adaptation decay
    beta: 0.2,         // Adaptation strength
    v_threshold: 1.0,
    gamma_pd: 0.3,
};

let config = EPropConfig {
    num_neurons: 256,
    input_dim: 128,
    output_dim: 20,
    neuron_config,
    alpha_smooth: 0.9,      // Trace smoothing
    learning_rate: 5e-4,
    grad_clip: Some(5.0),   // Gradient clipping
    sparsity_threshold: Some(0.01), // Weight pruning
    num_cycles: 5,          // Recurrent cycles
    init_scale: 0.5,        // Weight init scale
};
```

## Theoretical Foundation

The algorithm is based on three core theorems:

### Theorem 1: Gradient Decomposition
For gradient w.r.t. weight W^{ji}:
```
∂E/∂W^{ji} = Σ_t L_t^j · e_t^{ji}
```
where:
- `L_t^j = ∂E/∂z_t^j` is the learning signal (global)
- `e_t^{ji}` is the eligibility trace (local)

### Theorem 2: Diagonal Jacobian Approximation
Full Jacobian `J_t = D_t + K_t` is approximated by diagonal `D_t`:
```
cos(vec(J_t), vec(D_t)) > 0.99  (for firing rates < 12 Hz)
```
Reduces complexity from O(N³) to O(N²).

### Theorem 3: Rank-One Exponential Smoothing
Diagonal trace is approximated as rank-one product:
```
ε_t ≈ ε_t^f ⊗ ε_t^x

ε_t^x = α·ε_{t-1}^x + x_t                    (presynaptic)
ε_t^f = α·(D_t ∘ ε_{t-1}^f) + (1-α)·D_t^f    (postsynaptic)
```
Achieves O(N) complexity with 90-95% BPTT fidelity.

## Neuron Models

### Leaky Integrate-and-Fire (LIF)

Basic spiking neuron model:
```
v_{t+1} = α·v_t + I_t - z_t·v_th
z_t = H(v_t - v_th)
```

### Adaptive LIF (ALIF)

LIF with spike-frequency adaptation:
```
v_{t+1} = α·v_t + I_t - z_t·v_th
A_t = v_th + β·a_t
z_t = H(v_t - A_t)
a_{t+1} = ρ·a_t + z_t
```

Parameters:
- `α`: Membrane decay (exp(-Δt/τ_m))
- `ρ`: Adaptation decay (exp(-Δt/τ_a))
- `β`: Adaptation strength
- `v_th`: Spike threshold

## Training Statistics

The trainer tracks comprehensive statistics:

```rust
let stats = trainer.stats();

// Number of gradient updates
println!("Updates: {}", stats.num_updates);

// Average firing rate
println!("Firing rate: {:.2}%", stats.avg_firing_rate * 100.0);

// Gradient norms (last 100)
if let Some(avg_norm) = stats.avg_grad_norm() {
    println!("Avg gradient norm: {:.4}", avg_norm);
}

// Loss history
if let Some(avg_loss) = stats.avg_loss(10) {
    println!("Recent loss (last 10): {:.4}", avg_loss);
}
```

## Weight Management

### Export Weights
```rust
let weights = trainer.export_weights();
// HashMap with keys: "W_in", "W_rec", "W_out"
```

### Import Weights
```rust
trainer.import_weights(weights)?;
```

## Testing

The module includes 61 comprehensive tests covering:
- Configuration validation
- Neuron dynamics (LIF/ALIF)
- Trace updates and decay
- Gradient computation
- Training steps
- Weight export/import

Run tests:
```bash
cargo test --lib eprop
```

## Performance Considerations

### Memory Usage
- **Input weights**: O(N × I)
- **Recurrent weights**: O(N²) but can be sparse
- **Traces**: O(N + I) (rank-one representation)
- **State**: O(N)

### Computational Cost
- **Forward pass**: O(N² + N×I) dominated by matmuls
- **Trace update**: O(N + I) per timestep
- **Gradient computation**: O(N×I + N²) rank-one updates

### Optimization Tips
1. Use `sparsity_threshold` to prune small weights
2. Reduce `num_cycles` for faster training
3. Adjust `alpha_smooth` to balance trace memory
4. Use smaller `learning_rate` for stable convergence
5. Enable `grad_clip` to prevent explosion

## Examples

### Temporal Sequence Learning
```rust
let config = EPropConfig::for_scale(256, 128, 10);
let mut trainer = EPropTrainer::new(config)?;

for epoch in 0..100 {
    for (sequence, target) in dataset {
        trainer.reset_state(); // Reset between sequences
        
        // Process sequence
        for input in sequence {
            trainer.forward(&input)?;
        }
        
        // Compute output and train
        let output = trainer.compute_output();
        let loss = mse(&output, &target);
        
        // Apply gradients
        let learning_signal = compute_signal(&output, &target);
        trainer.apply_update(&learning_signal)?;
    }
}
```

### Pattern Classification
```rust
let config = EPropConfig {
    neuron_config: NeuronConfig::alif(), // Use adaptation
    num_cycles: 5, // Multiple processing cycles
    ..Default::default()
};
let mut trainer = EPropTrainer::new(config)?;

for (pattern, label) in dataset {
    let loss = trainer.train_step(&pattern.view(), &label.view())?;
}
```

## References

1. **Bellec et al. (2020)**: "A solution to the learning dilemma for recurrent networks of spiking neurons"
2. **Yin et al. (2025)**: "ES-D-RTRL: Diagonal Approximated RTRL with Exponential Smoothing"

## Integration with RustGPT

This module can be integrated as an alternative training method for temporal sequences, complementing the existing Transformer-based architecture with neuromorphic SNN capabilities.

```rust
// In your main training loop
use eprop::EPropTrainer;

match training_mode {
    TrainingMode::Standard => standard_train(),
    TrainingMode::EProp => eprop_train(),
}
```

## License

Same as parent project (see LICENSE.txt).
