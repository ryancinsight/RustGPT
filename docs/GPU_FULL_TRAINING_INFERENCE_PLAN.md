# GPU Full Training and Inference Implementation Plan

## Executive Summary

This document outlines the implementation plan for complete GPU-native training and inference in RustGPT. The goal is to eliminate CPU-GPU data transfer during training and inference, keeping all tensors on GPU throughout the entire pipeline.

## Current State Analysis

### Already Implemented (Phase 5.3-5.9)

1. **GPU Backend Abstraction** (`src/domain/compute/`)
   - `GpuDevice` - Unified device context for CUDA, Metal, WGPU
   - `GpuBuffer`, `GpuMemoryPool` - Memory management
   - `GpuMatrixOps` trait - Matrix operations interface
   - Auto-detection for GPU backends with strict no-fallback

2. **GPU Training Pipeline** (`src/application/training/gpu_training.rs`)
   - `GpuTrainingPipeline` - End-to-end GPU training framework
   - `GpuParameterGroup` - GPU-resident parameters with gradient accumulation
   - `GpuAdam` optimizer (`src/infrastructure/optimizer/gpu_adam.rs`)
   - Learning rate schedulers (constant, warmup cosine, warmup linear, exponential)

3. **GPU Kernels** (`src/domain/layers/components/`)
   - `unified_gpu_kernels.rs` - Attention, SSM operations
   - `ssm_gpu_kernels.rs` - Mamba/RG-LRU kernels
   - `attention_gpu_kernel.rs` - Attention operations
   - `feedforward_gpu.rs` - Feedforward operations
   - `gpu_gemm_kernels.rs` - GEMM operations

4. **GPU Operations Available on GpuDevice**
   - GEMM (batched and single)
   - Layer normalization
   - Softmax (forward and backward)
   - Activations (ReLU, GELU, SiLU, Sigmoid)
   - Richards curve (forward and backward)
   - Selective scan (SSM forward)
   - PolyAttention fused operations
   - MoH gate activation
   - Element-wise operations (mul, add_scaled, scale, axpy)
   - Reductions (sum, mean)
   - Data transfer (upload, download, copy_within_device)

### Missing Components

1. **GPU-resident LLMModel** - Current `LLMModel` uses CPU `ndarray`
2. **GPU Loss Functions** - Cross-entropy, symmetric CE
3. **GPU Embedding Layer** - Token embeddings
4. **GPU Output Projection** - Final projection layer
5. **GPU KV-Cache** - For inference
6. **GPU Backward Pass for All Layers** - Complete gradient computation
7. **GPU Data Pipeline** - Keep training data on GPU

## Implementation Plan

### Phase 1: GPU Model Wrapper (Week 1)

#### 1.1 GpuLLMModel Structure

```rust
/// GPU-resident LLM model with all parameters on GPU
pub struct GpuLLMModel {
    /// GPU device
    device: Arc<Mutex<GpuDevice>>,
    /// Model configuration
    config: ModelConfig,
    /// Token embedding table [vocab_size, embed_dim]
    token_embeddings: GpuBuffer,
    /// Position embedding table [max_seq_len, embed_dim]
    position_embeddings: Option<GpuBuffer>,
    /// Transformer layers (GPU-resident parameters)
    layers: Vec<GpuLayer>,
    /// Output projection [embed_dim, vocab_size]
    output_projection: GpuBuffer,
    /// Output bias [vocab_size]
    output_bias: Option<GpuBuffer>,
    /// Layer norm gamma/beta for final normalization
    final_ln_gamma: GpuBuffer,
    final_ln_beta: GpuBuffer,
}
```

#### 1.2 GpuLayer Enum

```rust
/// GPU-resident layer with all parameters on GPU
pub enum GpuLayer {
    Transformer(GpuTransformerBlock),
    Diffusion(GpuDiffusionBlock),
    SSM(GpuSSMBlock),
    MoE(GpuMoEBlock),
}
```

#### 1.3 GpuTransformerBlock

```rust
pub struct GpuTransformerBlock {
    /// Attention QKV projection [embed_dim, 3 * embed_dim]
    qkv_weight: GpuBuffer,
    /// Attention output projection [embed_dim, embed_dim]
    attn_out_weight: GpuBuffer,
    /// Attention output bias
    attn_out_bias: Option<GpuBuffer>,
    /// FFN up projection [embed_dim, ffn_dim]
    ffn_up_weight: GpuBuffer,
    /// FFN down projection [ffn_dim, embed_dim]
    ffn_down_weight: GpuBuffer,
    /// Layer norm parameters
    ln1_gamma: GpuBuffer,
    ln1_beta: GpuBuffer,
    ln2_gamma: GpuBuffer,
    ln2_beta: GpuBuffer,
    /// Workspace buffers (reused across forward/backward)
    workspace: GpuTransformerWorkspace,
}
```

### Phase 2: GPU Forward Pass (Week 1-2)

#### 2.1 Embedding Lookup on GPU

```rust
impl GpuLLMModel {
    /// Embed token IDs on GPU
    /// input_ids: [batch_size, seq_len] token IDs
    /// output: [batch_size, seq_len, embed_dim] embeddings
    pub fn embed_tokens(&mut self, input_ids: &GpuBuffer, output: &mut GpuBuffer) -> Result<()> {
        // Use GPU gather/scatter or custom kernel
        // No CPU transfer
    }
}
```

#### 2.2 Attention Forward on GPU

```rust
impl GpuTransformerBlock {
    /// Forward pass entirely on GPU
    pub fn forward(
        &mut self,
        device: &mut GpuDevice,
        input: &GpuBuffer,      // [batch, seq, embed]
        attention_mask: Option<&GpuBuffer>,
        kv_cache: Option<&mut GpuKVCache>,
        output: &mut GpuBuffer, // [batch, seq, embed]
    ) -> Result<()> {
        // 1. Layer norm
        // 2. QKV projection (GEMM)
        // 3. Attention scoring (softmax)
        // 4. Attention output (GEMM)
        // 5. Residual add
        // 6. FFN (GEMM + activation)
        // 7. Residual add
    }
}
```

#### 2.3 SSM Forward on GPU

Already partially implemented in `ssm_gpu_kernels.rs`. Need to integrate with GpuLLMModel.

### Phase 3: GPU Loss Functions (Week 2)

#### 3.1 Cross-Entropy Loss on GPU

```rust
/// GPU cross-entropy loss
pub fn gpu_cross_entropy_loss(
    device: &mut GpuDevice,
    logits: &GpuBuffer,    // [batch, seq, vocab]
    targets: &GpuBuffer,   // [batch, seq] token IDs
    loss: &mut GpuBuffer,  // scalar loss
    grad: &mut GpuBuffer,  // [batch, seq, vocab] gradients
) -> Result<()> {
    // 1. Softmax over vocabulary
    // 2. Gather target probabilities
    // 3. Compute -log(prob)
    // 4. Reduce mean
    // 5. Compute gradients (softmax - one_hot)
}
```

#### 3.2 Symmetric Cross-Entropy on GPU

```rust
/// GPU symmetric cross-entropy (forward + backward)
pub fn gpu_symmetric_cross_entropy(
    device: &mut GpuDevice,
    logits: &GpuBuffer,
    targets: &GpuBuffer,
    alpha: f32,
    beta: f32,
    epsilon: f32,
    loss: &mut GpuBuffer,
    grad: &mut GpuBuffer,
) -> Result<()>;
```

### Phase 4: GPU Backward Pass (Week 2-3)

#### 4.1 Backward Pass Architecture

```rust
/// GPU backward pass state
pub struct GpuBackwardState {
    /// Gradients for each layer
    layer_grads: Vec<GpuLayerGrads>,
    /// Intermediate activations (saved during forward)
    saved_activations: Vec<GpuSavedActivation>,
    /// Gradient accumulation buffers
    grad_accumulators: Vec<GpuBuffer>,
}

/// Saved activation for backward pass
pub struct GpuSavedActivation {
    /// Input to layer
    input: GpuBuffer,
    /// Attention scores (for attention backward)
    attention_scores: Option<GpuBuffer>,
    /// FFN intermediate (for FFN backward)
    ffn_intermediate: Option<GpuBuffer>,
}
```

#### 4.2 Attention Backward on GPU

```rust
impl GpuTransformerBlock {
    /// Backward pass entirely on GPU
    pub fn backward(
        &mut self,
        device: &mut GpuDevice,
        grad_output: &GpuBuffer,
        saved: &GpuSavedActivation,
        grad_input: &mut GpuBuffer,
        param_grads: &mut GpuLayerGrads,
    ) -> Result<()> {
        // 1. Backward through FFN
        // 2. Backward through residual
        // 3. Backward through attention
        // 4. Backward through layer norm
        // 5. Accumulate parameter gradients
    }
}
```

### Phase 5: GPU Training Loop Integration (Week 3)

#### 5.1 Complete GPU Training Step

```rust
impl GpuLLMModel {
    /// Single training step entirely on GPU
    pub fn train_step(
        &mut self,
        input_ids: &GpuBuffer,
        target_ids: &GpuBuffer,
        pipeline: &mut GpuTrainingPipeline,
    ) -> Result<f32> {
        // 1. Forward pass (GPU)
        let logits = self.forward_gpu(input_ids)?;
        
        // 2. Loss computation (GPU)
        let (loss, grad) = self.compute_loss_gpu(&logits, target_ids)?;
        
        // 3. Backward pass (GPU)
        self.backward_gpu(&grad)?;
        
        // 4. Optimizer step (GPU)
        pipeline.step(&mut self.parameters)?;
        
        Ok(loss)
    }
}
```

### Phase 6: GPU Inference with KV-Cache (Week 3-4)

#### 6.1 GPU KV-Cache

```rust
/// GPU-resident KV-cache for inference
pub struct GpuKVCache {
    /// Key cache [batch, num_heads, max_seq, head_dim]
    key_cache: GpuBuffer,
    /// Value cache [batch, num_heads, max_seq, head_dim]
    value_cache: GpuBuffer,
    /// Current sequence position
    current_pos: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl GpuKVCache {
    /// Append new KV to cache
    pub fn append(
        &mut self,
        device: &mut GpuDevice,
        new_key: &GpuBuffer,
        new_value: &GpuBuffer,
    ) -> Result<()>;
    
    /// Get cached KV for attention
    pub fn get_cached_kv(&self) -> (&GpuBuffer, &GpuBuffer);
}
```

#### 6.2 GPU Inference

```rust
impl GpuLLMModel {
    /// Generate tokens entirely on GPU
    pub fn generate_gpu(
        &mut self,
        prompt_ids: &GpuBuffer,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<GpuBuffer> {
        // 1. Process prompt with KV-cache
        // 2. Sample next token (GPU)
        // 3. Append to sequence
        // 4. Continue until max_tokens or EOS
    }
    
    /// Sample next token on GPU
    fn sample_token_gpu(
        &mut self,
        logits: &GpuBuffer,
        temperature: f32,
    ) -> Result<u32>;
}
```

### Phase 7: GPU Memory Management (Week 4)

#### 7.1 GPU Buffer Pool for Model

```rust
/// GPU buffer pool for model workspace
pub struct GpuModelWorkspace {
    /// Forward pass workspace
    forward_workspace: Vec<GpuBuffer>,
    /// Backward pass workspace
    backward_workspace: Vec<GpuBuffer>,
    /// Temporary buffers (reused)
    temp_buffers: Vec<GpuBuffer>,
    /// Buffer pool for dynamic allocation
    pool: UnifiedGpuBufferPool,
}

impl GpuModelWorkspace {
    /// Get or allocate a buffer of given size
    pub fn get_buffer(&mut self, size: usize) -> &mut GpuBuffer;
    
    /// Return buffer to pool
    pub fn return_buffer(&mut self, buffer: GpuBuffer);
}
```

#### 7.2 Memory-Efficient Training

```rust
/// Memory-efficient training configuration
pub struct GpuMemoryConfig {
    /// Use gradient checkpointing
    pub checkpoint_layers: bool,
    /// Activation recomputation strategy
    pub recompute_activations: bool,
    /// Micro-batch size for gradient accumulation
    pub micro_batch_size: usize,
    /// Maximum GPU memory to use (bytes)
    pub max_memory_bytes: usize,
}
```

## File Structure

```
src/
├── domain/
│   ├── models/
│   │   ├── gpu_llm.rs          # GpuLLMModel implementation
│   │   ├── gpu_layer.rs        # GpuLayer enum and implementations
│   │   └── gpu_kv_cache.rs     # GPU KV-cache
│   ├── loss/
│   │   └── gpu_loss.rs         # GPU loss functions
│   └── layers/
│       └── components/
│           ├── gpu_backward.rs # GPU backward pass utilities
│           └── gpu_workspace.rs # GPU workspace management
├── application/
│   └── training/
│       └── gpu_training.rs     # (existing) extend with model integration
└── infrastructure/
    └── optimizer/
        └── gpu_adam.rs         # (existing) GPU Adam optimizer
```

## Performance Targets

| Operation | CPU Time | GPU Target | Speedup |
|-----------|----------|------------|---------|
| Forward pass (512 batch) | 30ms | 1ms | 30x |
| Backward pass (512 batch) | 60ms | 2ms | 30x |
| Optimizer step | 10ms | 0.5ms | 20x |
| Full training step | 100ms | 4ms | 25x |
| Inference (per token) | 5ms | 0.2ms | 25x |

## Testing Strategy

1. **Unit Tests**: Each GPU kernel tested against CPU reference
2. **Integration Tests**: Full forward/backward pass numerical gradient checking
3. **Performance Tests**: Benchmark against CPU baseline
4. **Memory Tests**: Verify no memory leaks, proper buffer reuse

## Migration Path

1. **Phase 1**: Add GPU model alongside CPU model (feature flag)
2. **Phase 2**: Add GPU training option to CLI
3. **Phase 3**: Benchmark and optimize
4. **Phase 4**: Make GPU the default when available

## Dependencies

- Existing GPU infrastructure (Phase 5.3-5.9)
- `wgpu` or `cudarc` for GPU operations
- No new external dependencies required

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| GPU memory overflow | Gradient checkpointing, micro-batching |
| Numerical precision differences | FP32 throughout, careful kernel implementation |
| Backend-specific bugs | Comprehensive test suite across backends |
| Performance regression | Continuous benchmarking |

## Success Criteria

1. All training operations execute on GPU without CPU transfer
2. All inference operations execute on GPU without CPU transfer
3. Performance targets met (25x+ speedup)
4. Memory usage bounded and predictable
5. All tests pass on CUDA, Metal, and WGPU backends
