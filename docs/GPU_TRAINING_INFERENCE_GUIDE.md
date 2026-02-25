# GPU Training and Inference Guide

This document describes the GPU-native training and inference implementation in RustGPT, which enables all operations to run on GPU without CPU-GPU data transfer.

## Overview

The GPU implementation provides:
- **Zero-copy training**: Forward pass, loss computation, backward pass, and optimizer steps all run on GPU
- **GPU-native inference**: Token generation with KV-cache entirely on GPU
- **Automatic backend detection**: Supports CUDA, Metal, Vulkan, and WGPU backends
- **Memory-efficient workspace**: Pre-allocated buffers for intermediate activations

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GpuLLMModel                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  token_embeddings: GpuBuffer [vocab_size, embedding_dim]                    │
│  position_embeddings: Option<GpuBuffer> [max_seq_len, embedding_dim]        │
│  layers: Vec<GpuLayer>                                                       │
│  ├── GpuLayer::Transformer(GpuTransformerLayer)                             │
│  ├── GpuLayer::SSM(GpuSSMLayer)                                              │
│  └── GpuLayer::MoE(GpuMoELayer)                                              │
│  output_projection: GpuBuffer [embedding_dim, vocab_size]                    │
│  final_ln_gamma/beta: GpuBuffer [embedding_dim]                              │
│  workspace: GpuModelWorkspace                                                │
│  kv_cache: Option<GpuKVCache>                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Feature Flags

Enable GPU support with Cargo features:

```toml
# WGPU backend (cross-platform)
cargo build --release --features gpu-wgpu

# CUDA backend (NVIDIA GPUs)
cargo build --release --features gpu-cuda

# Metal backend (Apple GPUs)
cargo build --release --features gpu-metal

# All backends
cargo build --release --features gpu-all
```

## Usage

### GPU-Native Training

```rust
use llm::domain::models::llm::LLM;
use llm::domain::compute::GpuDevice;

// Load or create model
let mut llm = LLM::new(vocab, network);

// Train a batch entirely on GPU
let input_ids: Vec<usize> = vec![/* token IDs */];
let target_ids: Vec<usize> = vec![/* target token IDs */];
let batch_size = 4;
let seq_len = 128;
let learning_rate = 0.001;

let loss = llm.train_batch_gpu(
    &input_ids,
    &target_ids,
    batch_size,
    seq_len,
    learning_rate
)?;
println!("Training loss: {}", loss);
```

### GPU-Native Inference

```rust
// Generate tokens on GPU
let prompt_ids: Vec<usize> = vec![/* prompt tokens */];
let max_new_tokens = 100;
let temperature = 0.8;

let generated = llm.generate_gpu(
    &prompt_ids,
    max_new_tokens,
    temperature
)?;
println!("Generated tokens: {:?}", generated);
```

### Low-Level GPU Model

For more control, use `GpuLLMModel` directly:

```rust
use std::sync::{Arc, Mutex};
use llm::domain::compute::GpuDevice;
use llm::domain::models::{GpuLLMModel, GpuLayer, GpuTransformerLayer};
use llm::application::training::{GpuTrainingPipeline, GpuTrainingConfig};

// Create GPU device
let device = Arc::new(Mutex::new(GpuDevice::auto_detect()?));

// Create GPU model (from CPU model or from scratch)
let gpu_model = GpuLLMModel::new(
    device.clone(),
    config,
    vocab_size,
    token_embeddings,
    position_embeddings,
    layers,
    output_projection,
    output_bias,
    final_ln_gamma,
    final_ln_beta,
)?;

// Forward pass
let logits = gpu_model.forward(&input_buffer, batch_size, seq_len)?;

// Training step
let config = GpuTrainingConfig {
    learning_rate: 0.001,
    weight_decay: 0.01,
    ..Default::default()
};
let mut pipeline = GpuTrainingPipeline::new(device, gpu_model.param_count(), config)?;
let loss = gpu_model.train_step(&input, &targets, batch_size, seq_len, &mut pipeline)?;

// Inference with KV-cache
let output = gpu_model.generate_gpu(&prompt, prompt_len, max_tokens, temperature)?;
```

## Components

### GpuLLMModel

The main GPU-resident model wrapper. Holds all model parameters on GPU and provides:
- `forward()` - Forward pass returning logits
- `train_step()` - Complete training step with loss and gradient computation
- `generate_gpu()` - Autoregressive token generation with KV-cache

### GpuTransformerLayer

GPU-resident transformer layer with:
- QKV projection weights
- Attention output projection
- FFN up/down projections
- Layer normalization parameters

### GpuKVCache

Key-Value cache for efficient inference:
- Pre-allocated memory for all layers
- O(1) append operation for new tokens
- Automatic cache management

### GpuTrainingPipeline

Manages training state and optimization:
- Adam/AdamW optimizer
- Learning rate scheduling (warmup + cosine decay)
- Gradient clipping and normalization
- EMA tracking for adaptive methods

### GpuLossWorkspace

Workspace for loss computation:
- Cross-entropy loss
- Symmetric cross-entropy loss
- Softmax workspace buffers

## Memory Management

### Workspace Pre-allocation

The `GpuModelWorkspace` pre-allocates buffers for:
- Hidden states: `[batch, seq, embed]`
- Attention scores: `[batch, heads, seq, seq]`
- QKV buffer: `[batch, seq, 3 * embed]`
- FFN intermediate: `[batch, seq, hidden]`
- Logits: `[batch, seq, vocab]`
- Gradient buffer: `[max(hidden_size, logits_size)]`

### Buffer Pool

For dynamic allocations, use `GpuMemoryPool`:
```rust
let mut pool = GpuMemoryPool::new(device, 1024 * 1024 * 1024); // 1GB pool
let buffer = pool.acquire(size)?;
// ... use buffer ...
pool.release(buffer);
```

## Performance Considerations

1. **Batch Size**: Larger batch sizes better utilize GPU parallelism
2. **Sequence Length**: Pre-allocate workspace for max sequence length
3. **KV-Cache**: Enable for autoregressive generation to avoid recomputation
4. **Gradient Accumulation**: Use for effective larger batch sizes

## Backend-Specific Notes

### CUDA
- Requires NVIDIA GPU with compute capability 7.0+
- Best performance with cuBLAS and cuDNN

### Metal
- Requires Apple Silicon or AMD GPU on macOS
- Uses Metal Performance Shaders

### WGPU
- Cross-platform fallback
- Works on any GPU with Vulkan/Metal/DX12 support

## Troubleshooting

### GPU Not Detected
```
Error: Backend { message: "No GPU device found" }
```
Solution: Ensure GPU drivers are installed and `RUSTGPT_GPU_STRICT_NO_FALLBACK` is not set.

### Out of Memory
```
Error: Backend { message: "Failed to allocate buffer" }
```
Solution: Reduce batch size or sequence length, or use gradient accumulation.

### Slow Performance
- Check that GPU is actually being used (enable logging)
- Ensure workspace is pre-allocated, not re-allocated each step
- Use appropriate batch size for your GPU

## Future Work

- [ ] Flash Attention 2 integration
- [ ] FP16/BF16 mixed precision training
- [ ] Model parallelism for large models
- [ ] Kernel fusion for improved performance
