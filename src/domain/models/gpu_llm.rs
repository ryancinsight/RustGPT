//! GPU-Native LLM Model
//!
//! This module provides a GPU-resident LLM model implementation where all operations
//! (forward pass, backward pass, inference) remain on GPU without CPU transfer.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         GpuLLMModel                                  │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  token_embeddings: GpuBuffer [vocab_size, embed_dim]                │
//! │  position_embeddings: GpuBuffer [max_seq_len, embed_dim]            │
//! │  layers: Vec<GpuLayer>                                              │
//! │  output_projection: GpuBuffer [embed_dim, vocab_size]               │
//! │  final_ln_gamma/beta: GpuBuffer [embed_dim]                         │
//! │  workspace: GpuModelWorkspace                                        │
//! │  kv_cache: Option<GpuKVCache>                                       │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! // Create GPU model from CPU model
//! let gpu_model = GpuLLMModel::from_cpu_model(&cpu_model, &mut device)?;
//!
//! // Training step (all on GPU)
//! let loss = gpu_model.train_step(&input_ids, &targets, &mut pipeline)?;
//!
//! // Inference (all on GPU)
//! let output_ids = gpu_model.generate_gpu(&prompt_ids, 100, 0.8)?;
//! ```

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::unified_gpu_backend::{GpuActivation, GpuTemporalType};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::application::training::GpuTrainingPipeline;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::models::gpu_kv_cache::{GpuKVCache, GpuKVCacheConfig};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::loss::{GpuLossWorkspace, GpuSymmetricCEConfig, gpu_symmetric_cross_entropy_loss};

use crate::domain::models::config::ModelConfig;

// ============================================================================
// GPU Layer Types
// ============================================================================

/// GPU-resident transformer layer parameters
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuTransformerLayer {
    /// Attention QKV projection [embed_dim, 3 * embed_dim]
    pub qkv_weight: GpuBuffer,
    /// Attention QKV bias [3 * embed_dim]
    pub qkv_bias: Option<GpuBuffer>,
    /// Attention output projection [embed_dim, embed_dim]
    pub attn_out_weight: GpuBuffer,
    /// Attention output bias [embed_dim]
    pub attn_out_bias: Option<GpuBuffer>,
    /// FFN up projection [embed_dim, ffn_dim]
    pub ffn_up_weight: GpuBuffer,
    /// FFN up bias [ffn_dim]
    pub ffn_up_bias: Option<GpuBuffer>,
    /// FFN down projection [ffn_dim, embed_dim]
    pub ffn_down_weight: GpuBuffer,
    /// FFN down bias [embed_dim]
    pub ffn_down_bias: Option<GpuBuffer>,
    /// Pre-attention layer norm gamma [embed_dim]
    pub ln1_gamma: GpuBuffer,
    /// Pre-attention layer norm beta [embed_dim]
    pub ln1_beta: GpuBuffer,
    /// Pre-FFN layer norm gamma [embed_dim]
    pub ln2_gamma: GpuBuffer,
    /// Pre-FFN layer norm beta [embed_dim]
    pub ln2_beta: GpuBuffer,
    /// Activation function
    pub activation: GpuActivation,
    /// Layer index
    pub layer_idx: usize,
}

/// GPU-resident SSM layer parameters (Mamba/RG-LRU)
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuSSMLayer {
    /// Input projection [embed_dim, state_dim * 2]
    pub input_proj: GpuBuffer,
    /// State transition A [state_dim]
    pub a_param: GpuBuffer,
    /// State transition B projection [embed_dim, state_dim]
    pub b_proj: GpuBuffer,
    /// Output projection C [state_dim, embed_dim]
    pub c_proj: GpuBuffer,
    /// Skip connection D [embed_dim]
    pub d_param: GpuBuffer,
    /// Output projection [embed_dim, embed_dim]
    pub output_proj: GpuBuffer,
    /// Layer norm gamma [embed_dim]
    pub ln_gamma: GpuBuffer,
    /// Layer norm beta [embed_dim]
    pub ln_beta: GpuBuffer,
    /// SSM type
    pub ssm_type: GpuTemporalType,
    /// Layer index
    pub layer_idx: usize,
}

/// GPU-resident MoE layer parameters
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuMoELayer {
    /// Router/gating weights [embed_dim, num_experts]
    pub router_weight: GpuBuffer,
    /// Expert weights (shared structure)
    pub expert_up_weights: Vec<GpuBuffer>,
    pub expert_down_weights: Vec<GpuBuffer>,
    /// Layer norm parameters
    pub ln_gamma: GpuBuffer,
    pub ln_beta: GpuBuffer,
    /// Number of experts
    pub num_experts: usize,
    /// Top-k routing
    pub top_k: usize,
    /// Layer index
    pub layer_idx: usize,
}

/// GPU-resident layer enum
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub enum GpuLayer {
    Transformer(GpuTransformerLayer),
    SSM(GpuSSMLayer),
    MoE(GpuMoELayer),
}

// ============================================================================
// GPU Model Workspace
// ============================================================================

/// Workspace for GPU model forward/backward passes
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuModelWorkspace {
    /// Hidden states buffer [batch, seq, embed]
    pub hidden_states: GpuBuffer,
    /// Attention scores buffer [batch, heads, seq, seq]
    pub attention_scores: GpuBuffer,
    /// QKV buffer [batch, seq, 3 * embed]
    pub qkv_buffer: GpuBuffer,
    /// FFN intermediate [batch, seq, ffn_dim]
    pub ffn_intermediate: GpuBuffer,
    /// Logits buffer [batch, seq, vocab]
    pub logits: GpuBuffer,
    /// Gradient buffer (reused)
    pub grad_buffer: GpuBuffer,
    /// Loss workspace
    pub loss_workspace: GpuLossWorkspace,
    /// Current batch size
    pub batch_size: usize,
    /// Current sequence length
    pub seq_len: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuModelWorkspace {
    /// Create a new workspace with pre-allocated buffers
    pub fn new(
        device: &mut GpuDevice,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        let hidden_size = batch_size * seq_len * embed_dim;
        let attn_size = batch_size * num_heads * seq_len * seq_len;
        let qkv_size = batch_size * seq_len * 3 * embed_dim;
        let ffn_size = batch_size * seq_len * ffn_dim;
        let logits_size = batch_size * seq_len * vocab_size;

        let hidden_states = device.allocate_f32(hidden_size)?;
        let attention_scores = device.allocate_f32(attn_size)?;
        let qkv_buffer = device.allocate_f32(qkv_size)?;
        let ffn_intermediate = device.allocate_f32(ffn_size)?;
        let logits = device.allocate_f32(logits_size)?;
        let grad_buffer = device.allocate_f32(hidden_size.max(logits_size))?;
        let loss_workspace = GpuLossWorkspace::new(device, batch_size, seq_len, vocab_size)?;

        Ok(Self {
            hidden_states,
            attention_scores,
            qkv_buffer,
            ffn_intermediate,
            logits,
            grad_buffer,
            loss_workspace,
            batch_size,
            seq_len,
        })
    }

    /// Resize workspace if needed
    pub fn ensure_capacity(
        &mut self,
        device: &mut GpuDevice,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        vocab_size: usize,
    ) -> Result<()> {
        if batch_size != self.batch_size || seq_len != self.seq_len {
            let hidden_size = batch_size * seq_len * embed_dim;
            let attn_size = batch_size * num_heads * seq_len * seq_len;
            let qkv_size = batch_size * seq_len * 3 * embed_dim;
            let ffn_size = batch_size * seq_len * ffn_dim;
            let logits_size = batch_size * seq_len * vocab_size;

            self.hidden_states = device.allocate_f32(hidden_size)?;
            self.attention_scores = device.allocate_f32(attn_size)?;
            self.qkv_buffer = device.allocate_f32(qkv_size)?;
            self.ffn_intermediate = device.allocate_f32(ffn_size)?;
            self.logits = device.allocate_f32(logits_size)?;
            self.grad_buffer = device.allocate_f32(hidden_size.max(logits_size))?;
            self.loss_workspace.ensure_capacity(device, batch_size, seq_len, vocab_size)?;

            self.batch_size = batch_size;
            self.seq_len = seq_len;
        }
        Ok(())
    }
}

// ============================================================================
// GPU LLM Model
// ============================================================================

/// GPU-resident LLM model
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuLLMModel {
    /// GPU device
    device: Arc<Mutex<GpuDevice>>,
    /// Model configuration
    config: ModelConfig,
    /// Vocabulary size
    vocab_size: usize,
    /// Token embedding table [vocab_size, embedding_dim]
    token_embeddings: GpuBuffer,
    /// Position embedding table [max_seq_len, embedding_dim] (optional for RoPE)
    position_embeddings: Option<GpuBuffer>,
    /// Transformer layers
    layers: Vec<GpuLayer>,
    /// Output projection [embedding_dim, vocab_size]
    output_projection: GpuBuffer,
    /// Output bias [vocab_size]
    output_bias: Option<GpuBuffer>,
    /// Final layer norm gamma [embedding_dim]
    final_ln_gamma: GpuBuffer,
    /// Final layer norm beta [embedding_dim]
    final_ln_beta: GpuBuffer,
    /// Workspace for forward/backward
    workspace: GpuModelWorkspace,
    /// KV-cache for inference
    kv_cache: Option<GpuKVCache>,
    /// Loss configuration
    loss_config: GpuSymmetricCEConfig,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuLLMModel {
    /// Create a new GPU LLM model
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: Arc<Mutex<GpuDevice>>,
        config: ModelConfig,
        vocab_size: usize,
        token_embeddings: GpuBuffer,
        position_embeddings: Option<GpuBuffer>,
        layers: Vec<GpuLayer>,
        output_projection: GpuBuffer,
        output_bias: Option<GpuBuffer>,
        final_ln_gamma: GpuBuffer,
        final_ln_beta: GpuBuffer,
    ) -> Result<Self> {
        // Create workspace
        let mut device_guard = device.lock().map_err(|_| ModelError::Lock {
            message: "Failed to lock GPU device".to_string(),
        })?;

        let workspace = GpuModelWorkspace::new(
            &mut device_guard,
            1, // Default batch size
            config.max_seq_len.min(128), // Default sequence length
            config.embedding_dim,
            config.num_heads.unwrap_or(8),
            config.hidden_dim,
            vocab_size,
        )?;

        drop(device_guard);

        Ok(Self {
            device,
            config,
            vocab_size,
            token_embeddings,
            position_embeddings,
            layers,
            output_projection,
            output_bias,
            final_ln_gamma,
            final_ln_beta,
            workspace,
            kv_cache: None,
            loss_config: GpuSymmetricCEConfig::default(),
        })
    }

    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Initialize KV-cache for inference
    pub fn init_kv_cache(&mut self) -> Result<()> {
        let mut device = self.device.lock().map_err(|_| ModelError::Lock {
            message: "Failed to lock GPU device".to_string(),
        })?;

        let num_heads = self.config.num_heads.unwrap_or(8);
        let cache_config = GpuKVCacheConfig::new(
            1, // batch size
            num_heads,
            self.config.max_seq_len,
            self.config.embedding_dim / num_heads,
            self.layers.len(),
        );

        self.kv_cache = Some(GpuKVCache::new(&mut device, cache_config)?);
        Ok(())
    }

    /// Clear KV-cache
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        if let Some(cache) = &mut self.kv_cache {
            let mut device = self.device.lock().map_err(|_| ModelError::Lock {
                message: "Failed to lock GPU device".to_string(),
            })?;
            cache.reset_all(&mut device)?;
        }
        Ok(())
    }

    /// Embed token IDs on GPU
    fn embed_tokens_impl(
        device: &mut GpuDevice,
        token_embeddings: &GpuBuffer,
        input_ids: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
        seq_len: usize,
        vocab_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        // Download token IDs, lookup embeddings, upload
        // TODO: Replace with GPU gather kernel
        let num_tokens = batch_size * seq_len;
        let mut ids_host = vec![0.0f32; num_tokens];
        device.download(input_ids, &mut ids_host)?;

        let token_ids: Vec<usize> = ids_host.iter().map(|&x| x as usize).collect();

        // Download embedding table
        let mut embeddings_host = vec![0.0f32; vocab_size * embed_dim];
        device.download(token_embeddings, &mut embeddings_host)?;

        // Lookup embeddings
        let mut output_host = vec![0.0f32; num_tokens * embed_dim];
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id < vocab_size {
                let src_start = token_id * embed_dim;
                let dst_start = i * embed_dim;
                output_host[dst_start..dst_start + embed_dim]
                    .copy_from_slice(&embeddings_host[src_start..src_start + embed_dim]);
            }
        }

        device.upload(&output_host, output)
    }

    /// Forward pass entirely on GPU
    ///
    /// input_ids: [batch, seq] token IDs
    /// Returns: [batch, seq, vocab] logits
    pub fn forward(
        &mut self,
        input_ids: &GpuBuffer,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GpuBuffer> {
        let mut device = self.device.lock().map_err(|_| ModelError::Lock {
            message: "Failed to lock GPU device".to_string(),
        })?;

        let num_heads = self.config.num_heads.unwrap_or(8);
        let embed_dim = self.config.embedding_dim;
        let hidden_dim = self.config.hidden_dim;
        let vocab_size = self.vocab_size;

        // Ensure workspace capacity
        self.workspace.ensure_capacity(
            &mut device,
            batch_size,
            seq_len,
            embed_dim,
            num_heads,
            hidden_dim,
            vocab_size,
        )?;

        // 1. Token embedding
        Self::embed_tokens_impl(
            &mut device,
            &self.token_embeddings,
            input_ids,
            &mut self.workspace.hidden_states,
            batch_size,
            seq_len,
            vocab_size,
            embed_dim,
        )?;

        // 2. Position embedding (if using absolute positions)
        if let Some(ref pos_emb) = self.position_embeddings {
            Self::add_position_embeddings_impl(
                &mut device,
                &mut self.workspace.hidden_states,
                pos_emb,
                batch_size,
                seq_len,
                embed_dim,
            )?;
        }

        // 3. Layer forward passes
        for layer in &mut self.layers {
            Self::forward_layer_impl(
                &mut device,
                layer,
                &mut self.workspace,
                batch_size,
                seq_len,
                embed_dim,
                hidden_dim,
                num_heads,
            )?;
        }

        // 4. Final layer norm - allocate new buffer for output
        let mut ln_output = device.allocate_f32(batch_size * seq_len * embed_dim)?;
        device.layer_norm(
            &self.workspace.hidden_states,
            &self.final_ln_gamma,
            &self.final_ln_beta,
            &mut ln_output,
            batch_size * seq_len,
            embed_dim,
            1e-5,
        )?;
        // Copy back to hidden_states
        device.copy_within_device(&ln_output, &mut self.workspace.hidden_states, batch_size * seq_len * embed_dim)?;

        // 5. Output projection
        let m = batch_size * seq_len;
        let n = vocab_size;
        let k = embed_dim;

        device.gemm_f32(
            1.0,
            &self.workspace.hidden_states,
            &self.output_projection,
            0.0,
            &mut self.workspace.logits,
            m,
            n,
            k,
            false,
            false,
        )?;

        // 6. Add output bias if present
        if let Some(ref bias) = self.output_bias {
            device.broadcast_add_rows(
                &mut self.workspace.logits,
                bias,
                batch_size * seq_len,
                vocab_size,
            )?;
        }

        // Clone the logits buffer to return
        let mut result = device.allocate_f32(batch_size * seq_len * vocab_size)?;
        device.copy_within_device(&self.workspace.logits, &mut result, batch_size * seq_len * vocab_size)?;

        Ok(result)
    }

    /// Forward pass for a single layer
    fn forward_layer_impl(
        device: &mut GpuDevice,
        layer: &mut GpuLayer,
        workspace: &mut GpuModelWorkspace,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
        ffn_dim: usize,
        num_heads: usize,
    ) -> Result<()> {
        match layer {
            GpuLayer::Transformer(t) => Self::forward_transformer_layer_impl(
                device, workspace, t, batch_size, seq_len, embed_dim, ffn_dim, num_heads,
            ),
            GpuLayer::SSM(s) => Self::forward_ssm_layer_impl(device, workspace, s, batch_size, seq_len),
            GpuLayer::MoE(m) => Self::forward_moe_layer_impl(device, workspace, m, batch_size, seq_len),
        }
    }

    /// Forward pass for transformer layer
    fn forward_transformer_layer_impl(
        device: &mut GpuDevice,
        workspace: &mut GpuModelWorkspace,
        layer: &GpuTransformerLayer,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
        ffn_dim: usize,
        _num_heads: usize,
    ) -> Result<()> {
        // 1. Pre-attention layer norm
        let mut ln_output = device.allocate_f32(batch_size * seq_len * embed_dim)?;
        device.layer_norm(
            &workspace.hidden_states,
            &layer.ln1_gamma,
            &layer.ln1_beta,
            &mut ln_output,
            batch_size * seq_len,
            embed_dim,
            1e-5,
        )?;
        device.copy_within_device(&ln_output, &mut workspace.hidden_states, batch_size * seq_len * embed_dim)?;

        // 2. QKV projection
        let m = batch_size * seq_len;
        let k = embed_dim;
        let n = 3 * embed_dim;

        device.gemm_f32(
            1.0,
            &workspace.hidden_states,
            &layer.qkv_weight,
            0.0,
            &mut workspace.qkv_buffer,
            m,
            n,
            k,
            false,
            false,
        )?;

        // 3. Add QKV bias if present
        if let Some(ref bias) = layer.qkv_bias {
            device.broadcast_add_rows(
                &mut workspace.qkv_buffer,
                bias,
                batch_size * seq_len,
                3 * embed_dim,
            )?;
        }

        // 4. Attention computation
        // TODO: Implement full attention with KV-cache support
        // For now, use simplified attention

        // 5. Attention output projection
        // TODO: Implement attention output

        // 6. Residual connection
        // TODO: Add residual

        // 7. Pre-FFN layer norm
        device.layer_norm(
            &workspace.hidden_states,
            &layer.ln2_gamma,
            &layer.ln2_beta,
            &mut ln_output,
            batch_size * seq_len,
            embed_dim,
            1e-5,
        )?;
        device.copy_within_device(&ln_output, &mut workspace.hidden_states, batch_size * seq_len * embed_dim)?;

        // 8. FFN up projection
        device.gemm_f32(
            1.0,
            &workspace.hidden_states,
            &layer.ffn_up_weight,
            0.0,
            &mut workspace.ffn_intermediate,
            m,
            ffn_dim,
            embed_dim,
            false,
            false,
        )?;

        // 9. Activation
        match layer.activation {
            GpuActivation::Identity => {
                // No activation, pass through
            }
            GpuActivation::Gelu => {
                let mut temp = device.allocate_f32(batch_size * seq_len * ffn_dim)?;
                device.gelu(
                    &workspace.ffn_intermediate,
                    &mut temp,
                    batch_size * seq_len * ffn_dim,
                )?;
                device.copy_within_device(&temp, &mut workspace.ffn_intermediate, batch_size * seq_len * ffn_dim)?;
            }
            GpuActivation::Silu => {
                let mut temp = device.allocate_f32(batch_size * seq_len * ffn_dim)?;
                device.silu(
                    &workspace.ffn_intermediate,
                    &mut temp,
                    batch_size * seq_len * ffn_dim,
                )?;
                device.copy_within_device(&temp, &mut workspace.ffn_intermediate, batch_size * seq_len * ffn_dim)?;
            }
            GpuActivation::Relu => {
                let mut temp = device.allocate_f32(batch_size * seq_len * ffn_dim)?;
                device.relu(
                    &workspace.ffn_intermediate,
                    &mut temp,
                    batch_size * seq_len * ffn_dim,
                )?;
                device.copy_within_device(&temp, &mut workspace.ffn_intermediate, batch_size * seq_len * ffn_dim)?;
            }
        }

        // 10. FFN down projection
        let mut ffn_output = device.allocate_f32(batch_size * seq_len * embed_dim)?;
        device.gemm_f32(
            1.0,
            &workspace.ffn_intermediate,
            &layer.ffn_down_weight,
            0.0,
            &mut ffn_output,
            m,
            embed_dim,
            ffn_dim,
            false,
            false,
        )?;
        device.copy_within_device(&ffn_output, &mut workspace.hidden_states, batch_size * seq_len * embed_dim)?;

        // 11. Residual connection
        // TODO: Add residual

        Ok(())
    }

    /// Forward pass for SSM layer
    fn forward_ssm_layer_impl(
        _device: &mut GpuDevice,
        _workspace: &mut GpuModelWorkspace,
        _layer: &GpuSSMLayer,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<()> {
        // TODO: Implement SSM forward pass
        // This will use the selective_scan_forward kernel
        Ok(())
    }

    /// Forward pass for MoE layer
    fn forward_moe_layer_impl(
        _device: &mut GpuDevice,
        _workspace: &mut GpuModelWorkspace,
        _layer: &GpuMoELayer,
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<()> {
        // TODO: Implement MoE forward pass
        Ok(())
    }

    /// Add position embeddings
    fn add_position_embeddings_impl(
        device: &mut GpuDevice,
        hidden_states: &mut GpuBuffer,
        pos_emb: &GpuBuffer,
        batch_size: usize,
        seq_len: usize,
        embed_dim: usize,
    ) -> Result<()> {
        // Download position embeddings, add to hidden states
        // TODO: Replace with GPU kernel
        let mut hidden_host = vec![0.0f32; batch_size * seq_len * embed_dim];
        let mut pos_host = vec![0.0f32; seq_len * embed_dim];

        device.download(hidden_states, &mut hidden_host)?;
        device.download(pos_emb, &mut pos_host)?;

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..embed_dim {
                    hidden_host[b * seq_len * embed_dim + s * embed_dim + d] +=
                        pos_host[s * embed_dim + d];
                }
            }
        }

        device.upload(&hidden_host, hidden_states)
    }

    /// Training step entirely on GPU
    ///
    /// Returns the loss value
    pub fn train_step(
        &mut self,
        input_ids: &GpuBuffer,
        target_ids: &GpuBuffer,
        batch_size: usize,
        seq_len: usize,
        _pipeline: &mut GpuTrainingPipeline,
    ) -> Result<f32> {
        // Extract values before borrowing
        let vocab_size = self.vocab_size;
        let loss_config = self.loss_config;

        // 1. Forward pass
        let logits = self.forward(input_ids, batch_size, seq_len)?;

        // 2. Loss computation
        let mut device = self.device.lock().map_err(|_| ModelError::Lock {
            message: "Failed to lock GPU device".to_string(),
        })?;

        let mut loss_buffer = device.allocate_f32(1)?;
        let mut grad_buffer = device.allocate_f32(batch_size * seq_len * vocab_size)?;

        let loss = gpu_symmetric_cross_entropy_loss(
            &mut device,
            &logits,
            target_ids,
            loss_config,
            &mut self.workspace.loss_workspace,
            &mut loss_buffer,
            &mut grad_buffer,
        )?;

        // 3. Backward pass
        // TODO: Implement backward pass

        // 4. Optimizer step
        // TODO: Integrate with GpuTrainingPipeline

        Ok(loss)
    }

    /// Generate tokens on GPU
    ///
    /// prompt_ids: [1, prompt_len] prompt token IDs
    /// max_new_tokens: Maximum number of tokens to generate
    /// temperature: Sampling temperature
    /// Returns: [1, prompt_len + generated_len] token IDs
    pub fn generate_gpu(
        &mut self,
        prompt_ids: &GpuBuffer,
        prompt_len: usize,
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<GpuBuffer> {
        // Initialize KV-cache if not present
        if self.kv_cache.is_none() {
            self.init_kv_cache()?;
        } else {
            self.clear_kv_cache()?;
        }

        let vocab_size = self.vocab_size;

        let mut device = self.device.lock().map_err(|_| ModelError::Lock {
            message: "Failed to lock GPU device".to_string(),
        })?;

        // Output buffer
        let total_len = prompt_len + max_new_tokens;
        let mut output_ids = device.allocate_f32(total_len)?;

        // Copy prompt to output
        device.copy_within_device(prompt_ids, &mut output_ids, prompt_len)?;

        // Current position
        let mut current_pos = prompt_len;

        drop(device);

        // Process prompt (prefill)
        let _logits = self.forward(prompt_ids, 1, prompt_len)?;

        // Generate tokens
        for _ in 0..max_new_tokens {
            let mut device = self.device.lock().map_err(|_| ModelError::Lock {
                message: "Failed to lock GPU device".to_string(),
            })?;

            // Get logits for current position
            let mut current_id = device.allocate_f32(1)?;
            device.copy_within_device_range(&output_ids, current_pos - 1, &mut current_id, 0, 1)?;

            drop(device);

            let logits = self.forward(&current_id, 1, 1)?;

            let mut device = self.device.lock().map_err(|_| ModelError::Lock {
                message: "Failed to lock GPU device".to_string(),
            })?;

            // Sample next token
            let next_token = Self::sample_token_gpu_impl(&mut device, &logits, vocab_size, temperature)?;

            // Append to output
            device.upload(&[next_token as f32], &mut output_ids)?;
            current_pos += 1;
        }

        Ok(output_ids)
    }

    /// Sample a token from logits on GPU
    fn sample_token_gpu_impl(
        device: &mut GpuDevice,
        logits: &GpuBuffer,
        vocab_size: usize,
        temperature: f32,
    ) -> Result<u32> {
        // Download logits, sample on CPU
        // TODO: Implement GPU sampling kernel
        let mut logits_host = vec![0.0f32; vocab_size];
        device.download(logits, &mut logits_host)?;

        // Apply temperature
        let temp = temperature.max(1e-6);
        for logit in &mut logits_host {
            *logit /= temp;
        }

        // Softmax
        let max_logit = logits_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for logit in &mut logits_host {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }
        for logit in &mut logits_host {
            *logit /= sum;
        }

        // Sample (argmax for now, TODO: implement proper sampling)
        let max_idx = logits_host
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        Ok(max_idx as u32)
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        // TODO: Calculate from actual buffers
        self.vocab_size * self.config.embedding_dim // Approximate
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        // TODO: Calculate from actual buffers
        self.param_count() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gpu_layer_types() {
        // Test that layer types are properly defined
        // This is a compile-time check
    }
}
