# GPU-Accelerated Adam Optimizer

## Overview

The `GpuAdam` optimizer provides a fully GPU-resident implementation of the Adam optimizer, eliminating CPU-GPU data transfers during training. This implementation supports:

- **Standard Adam**: Adaptive Moment Estimation with bias correction
- **AdamW**: Decoupled weight decay for better generalization
- **AMSGrad**: Maximum of past squared gradients for improved convergence

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      GpuAdam                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  m_buffer   │  │  v_buffer   │  │  v_max_buffer (opt) │  │
│  │  (GPU)      │  │  (GPU)      │  │  (GPU)              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              GpuAdamConfig                            │   │
│  │  • beta1: 0.9 (first moment decay)                   │   │
│  │  • beta2: 0.999 (second moment decay)                │   │
│  │  • epsilon: 1e-8 (numerical stability)               │   │
│  │  • weight_decay: 0.0 (L2 regularization)             │   │
│  │  • use_decoupled_wd: false (AdamW)                   │   │
│  │  • use_amsgrad: false (AMSGrad)                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    WGSL Compute Shader                      │
│                                                             │
│  @compute @workgroup_size(256)                              │
│  fn main() {                                                │
│    1. Apply weight decay (if L2)                           │
│    2. Update m: m = β₁·m + (1-β₁)·g                        │
│    3. Update v: v = β₂·v + (1-β₂)·g²                       │
│    4. Bias correction: m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)       │
│    5. AMSGrad: v_max = max(v_max, v̂)                      │
│    6. AdamW: θ = θ·(1 - ηλ)                                │
│    7. Update: θ = θ - η·m̂/(√v̂ + ε)                        │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```rust
use rust_gpt::infrastructure::optimizer::{GpuAdam, GpuAdamConfig};
use rust_gpt::domain::compute::GpuDevice;
use std::sync::{Arc, Mutex};

// Create GPU device
let device = Arc::new(Mutex::new(GpuDevice::new(ComputeBackend::Vulkan)?));

// Create optimizer for 1M parameters
let mut optimizer = GpuAdam::new(device.clone(), 1_000_000)?;

// Training loop
for batch in training_data {
    // Forward pass, backward pass (all on GPU)
    // ... compute gradients into grads_buffer ...
    
    // Optimization step (no CPU-GPU transfer!)
    optimizer.step(&mut params_buffer, &grads_buffer, 0.001)?;
}
```

### AdamW (Decoupled Weight Decay)

```rust
// Create AdamW optimizer with weight decay 0.01
let mut optimizer = GpuAdam::new_adamw(device, param_count, 0.01)?;

// Or use config:
let config = GpuAdamConfig::adamw(0.01);
let mut optimizer = GpuAdam::with_config(device, param_count, config)?;
```

### AMSGrad

```rust
// Create AMSGrad optimizer
let mut optimizer = GpuAdam::new_amsgrad(device, param_count)?;

// Or use config:
let config = GpuAdamConfig::amsgrad();
let mut optimizer = GpuAdam::with_config(device, param_count, config)?;
```

### Custom Configuration

```rust
let config = GpuAdamConfig {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weight_decay: 0.001,
    use_decoupled_wd: true,  // AdamW style
    use_amsgrad: false,
};

let mut optimizer = GpuAdam::with_config(device, param_count, config)?;
```

### Checkpointing

```rust
// Save optimizer state
let m_state = optimizer.download_m()?;
let v_state = optimizer.download_v()?;
let timestep = optimizer.timestep();

// Save to disk...
// std::fs::write("optimizer_m.bin", bytemuck::cast_slice(&m_state))?;

// Restore optimizer state
optimizer.upload_m(&m_state)?;
optimizer.upload_v(&v_state)?;
// Note: timestep is tracked internally, reset if needed
```

## Performance

### Memory Layout

| Buffer | Size | Purpose |
|--------|------|---------|
| `params` | N × 4 bytes | Model parameters (read-write) |
| `grads` | N × 4 bytes | Gradients (read-only) |
| `m` | N × 4 bytes | First moment estimate |
| `v` | N × 4 bytes | Second moment estimate |
| `v_max` | N × 4 bytes | AMSGrad max (optional) |

Total memory: **4N to 5N** bytes (16-20 MB per 1M parameters)

### GPU Utilization

- **Workgroup size**: 256 threads
- **Parallelism**: All N parameters updated simultaneously
- **Memory bandwidth**: ~4 reads + 3-4 writes per parameter

### Comparison with CPU Adam

| Operation | CPU Adam | GPU Adam |
|-----------|----------|----------|
| Gradient transfer | Required | Not needed |
| Parameter transfer | Required | Not needed |
| Update computation | Sequential | Parallel |
| Total round-trips | 2 per step | 0 per step |

## Mathematical Details

### Adam Algorithm

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²

m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

θ_t = θ_{t-1} - η · m̂_t / (√v̂_t + ε)
```

### AMSGrad Variant

```
v̂_{max,t} = max(v̂_{max,t-1}, v̂_t)
θ_t = θ_{t-1} - η · m̂_t / (√v̂_{max,t} + ε)
```

### AdamW (Decoupled Weight Decay)

```
θ_t = θ_{t-1} · (1 - λη) - η · m̂_t / (√v̂_t + ε)
```

Where λ is the weight decay coefficient.

## Integration with Training Loop

The GPU Adam optimizer integrates seamlessly with the existing GPU training infrastructure:

```rust
// In your training loop
loop {
    // 1. Forward pass (GPU)
    gpu.forward(&input_buffer, &mut output_buffer, &weights_buffer)?;
    
    // 2. Compute loss (GPU)
    gpu.compute_loss(&output_buffer, &target_buffer, &mut loss_buffer)?;
    
    // 3. Backward pass (GPU)
    gpu.backward(&grad_output_buffer, &mut grad_weights_buffer)?;
    
    // 4. Optimizer step (GPU) - NO TRANSFER!
    adam.step(&mut weights_buffer, &grad_weights_buffer, learning_rate)?;
}
```

## Error Handling

The optimizer returns `Result<T, ModelError>` for all operations:

```rust
match optimizer.step(&mut params, &grads, lr) {
    Ok(()) => { /* success */ },
    Err(ModelError::Backend { message }) => {
        eprintln!("GPU error: {}", message);
    },
    Err(e) => {
        eprintln!("Other error: {}", e);
    },
}
```

## Testing

Run the integration tests:

```bash
# Run GPU Adam tests
cargo test --test gpu_adam_optimizer --features gpu-wgpu

# Run all optimizer tests
cargo test --lib optimizer --features gpu-wgpu
```

## Future Improvements

1. **Fused Adam + Gradient Computation**: Combine backward pass with Adam update
2. **Multi-tensor Adam**: Update multiple parameter tensors in single kernel
3. **Gradient Accumulation**: Support for larger effective batch sizes
4. **Learning Rate Scheduling**: GPU-resident LR schedule state
5. **Mixed Precision**: FP16/BF16 support for memory efficiency

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) (AMSGrad)
