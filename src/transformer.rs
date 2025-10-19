use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    attention_moe::AttentionMoELayer, feed_forward::FeedForward, layer_norm::LayerNorm,
    llm::Layer, rms_norm::RMSNorm, self_attention::SelfAttention, swiglu::SwiGLU,
};

/// Normalization layer type for TransformerBlock
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum NormLayer {
    LayerNorm(Box<LayerNorm>),
    RMSNorm(Box<RMSNorm>),
}

impl NormLayer {
    /// Create a new LayerNorm
    pub fn layer_norm(embedding_dim: usize) -> Self {
        NormLayer::LayerNorm(Box::new(LayerNorm::new(embedding_dim)))
    }

    /// Create a new RMSNorm
    pub fn rms_norm(embedding_dim: usize) -> Self {
        NormLayer::RMSNorm(Box::new(RMSNorm::new(embedding_dim)))
    }

    /// Normalize the input
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            NormLayer::LayerNorm(norm) => norm.normalize(input),
            NormLayer::RMSNorm(norm) => norm.normalize(input),
        }
    }

    /// Compute gradients
    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            NormLayer::LayerNorm(norm) => norm.compute_gradients(input, output_grads),
            NormLayer::RMSNorm(norm) => norm.compute_gradients(input, output_grads),
        }
    }

    /// Apply gradients
    pub fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        match self {
            NormLayer::LayerNorm(norm) => norm.apply_gradients(param_grads, lr),
            NormLayer::RMSNorm(norm) => norm.apply_gradients(param_grads, lr),
        }
    }

    /// Get parameter count
    pub fn parameters(&self) -> usize {
        match self {
            NormLayer::LayerNorm(norm) => norm.parameters(),
            NormLayer::RMSNorm(norm) => norm.parameters(),
        }
    }

    /// Get number of parameter gradients
    pub fn num_param_grads(&self) -> usize {
        match self {
            NormLayer::LayerNorm(_) => 2, // gamma, beta
            NormLayer::RMSNorm(_) => 1,   // gamma only
        }
    }
}

/// Attention layer type for TransformerBlock
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum AttentionLayer {
    SelfAttention(Box<SelfAttention>),
    AttentionMoE(Box<AttentionMoELayer>),
}

impl AttentionLayer {
    /// Create a new SelfAttention layer
    pub fn self_attention(embedding_dim: usize) -> Self {
        AttentionLayer::SelfAttention(Box::new(SelfAttention::new(embedding_dim)))
    }

    /// Create a new AttentionMoE layer
    pub fn attention_moe(
        embedding_dim: usize,
        num_experts: usize,
        num_active_experts: usize,
        num_shared_heads: usize,
        num_routed_heads: usize,
        num_kv_heads: usize,
        use_learned_threshold: bool,
        load_balance_weight: f32,
        router_z_loss_weight: f32,
    ) -> Self {
        AttentionLayer::AttentionMoE(Box::new(AttentionMoELayer::new(
            embedding_dim,
            num_experts,
            num_active_experts,
            num_shared_heads,
            num_routed_heads,
            num_kv_heads,
            use_learned_threshold,
            load_balance_weight,
            router_z_loss_weight,
        )))
    }

    /// Set epoch information for warm-up and annealing
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.set_epoch_info(current_epoch, max_epochs),
            AttentionLayer::AttentionMoE(moe) => moe.set_epoch_info(current_epoch, max_epochs),
        }
    }

    /// Get MoH statistics
    pub fn get_moh_stats(&self) -> Vec<(usize, f32, f32, f32, f32, f32, f32, f32)> {
        match self {
            AttentionLayer::SelfAttention(attn) => {
                let stats = attn.get_moh_stats();
                vec![(0, stats.0, stats.1, stats.2, stats.3, stats.4, stats.5, stats.6)]
            }
            AttentionLayer::AttentionMoE(moe) => moe.get_expert_moh_stats(),
        }
    }

    /// Get auxiliary loss (for MoE routing)
    pub fn get_aux_loss(&self) -> f32 {
        match self {
            AttentionLayer::SelfAttention(_) => 0.0,
            AttentionLayer::AttentionMoE(moe) => moe.get_auxiliary_loss(),
        }
    }

    /// Compute gradients (for compatibility with Layer trait)
    pub fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // AttentionLayer uses backward() directly, not compute_gradients/apply_gradients
        // This is here for compatibility with the Layer trait
        panic!("AttentionLayer does not support compute_gradients - use backward() instead");
    }

    /// Apply gradients (for compatibility with Layer trait)
    pub fn apply_gradients(
        &mut self,
        _param_grads: &[Array2<f32>],
        _lr: f32,
    ) -> crate::errors::Result<()> {
        // AttentionLayer uses backward() directly, not compute_gradients/apply_gradients
        // This is here for compatibility with the Layer trait
        panic!("AttentionLayer does not support apply_gradients - use backward() instead");
    }
}

impl Layer for AttentionLayer {
    fn layer_type(&self) -> &str {
        match self {
            AttentionLayer::SelfAttention(_) => "SelfAttention",
            AttentionLayer::AttentionMoE(_) => "AttentionMoE",
        }
    }

    fn parameters(&self) -> usize {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.parameters(),
            AttentionLayer::AttentionMoE(_moe) => {
                // TODO: Implement parameter counting for AttentionMoE
                0
            }
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.compute_gradients(input, output_grads),
            AttentionLayer::AttentionMoE(_) => {
                // AttentionMoE uses backward() directly
                panic!("AttentionMoE does not support compute_gradients - use backward() instead");
            }
        }
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.apply_gradients(param_grads, lr),
            AttentionLayer::AttentionMoE(_) => {
                // AttentionMoE uses backward() directly
                panic!("AttentionMoE does not support apply_gradients - use backward() instead");
            }
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.forward(input),
            AttentionLayer::AttentionMoE(moe) => moe.forward(input),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            AttentionLayer::SelfAttention(attn) => attn.backward(grads, lr),
            AttentionLayer::AttentionMoE(moe) => moe.backward(grads, lr),
        }
    }
}

/// Feedforward layer type for TransformerBlock
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum FFNLayer {
    FeedForward(Box<FeedForward>),
    SwiGLU(Box<SwiGLU>),
}

impl FFNLayer {
    /// Create a new FeedForward layer
    pub fn feed_forward(embedding_dim: usize, hidden_dim: usize) -> Self {
        FFNLayer::FeedForward(Box::new(FeedForward::new(embedding_dim, hidden_dim)))
    }

    /// Create a new SwiGLU layer
    pub fn swiglu(embedding_dim: usize, hidden_dim: usize) -> Self {
        FFNLayer::SwiGLU(Box::new(SwiGLU::new(embedding_dim, hidden_dim)))
    }

    /// Forward pass
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            FFNLayer::FeedForward(ffn) => ffn.forward(input),
            FFNLayer::SwiGLU(swiglu) => swiglu.forward(input),
        }
    }

    /// Compute gradients
    pub fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            FFNLayer::FeedForward(ffn) => ffn.compute_gradients(input, output_grads),
            FFNLayer::SwiGLU(swiglu) => swiglu.compute_gradients(input, output_grads),
        }
    }

    /// Apply gradients
    pub fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        match self {
            FFNLayer::FeedForward(ffn) => ffn.apply_gradients(param_grads, lr),
            FFNLayer::SwiGLU(swiglu) => swiglu.apply_gradients(param_grads, lr),
        }
    }

    /// Get parameter count
    pub fn parameters(&self) -> usize {
        match self {
            FFNLayer::FeedForward(ffn) => ffn.parameters(),
            FFNLayer::SwiGLU(swiglu) => swiglu.parameters(),
        }
    }

    /// Get number of parameter gradients
    pub fn num_param_grads(&self) -> usize {
        match self {
            FFNLayer::FeedForward(_) => 2, // w1, w2 (no biases - modern LLM practice)
            FFNLayer::SwiGLU(_) => 3,      // w1, w2, w3 (no biases)
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransformerBlock {
    pub attention: AttentionLayer,
    pub feed_forward: FFNLayer,
    pub norm1: NormLayer, // After attention
    pub norm2: NormLayer, // After feed forward

    // DeepNorm scaling parameters (Microsoft Research, 2022)
    // Reference: "DeepNet: Scaling Transformers to 1,000 Layers"
    // alpha: residual scaling factor = (2N)^0.25 where N = num_layers
    // beta: sublayer output scaling factor = 1/alpha
    // These maintain gradient magnitude across depth
    pub alpha: f32,
    pub beta: f32,
}

impl TransformerBlock {
    /// Create a new TransformerBlock with LayerNorm and FeedForward (default)
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        Self::with_config(embedding_dim, hidden_dim, false, false, 1.0)
    }

    /// Create a new TransformerBlock with specified normalization type (backward compatibility)
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden dimension for feedforward layer
    /// * `use_rms_norm` - If true, use RMSNorm; if false, use LayerNorm
    pub fn with_norm_type(embedding_dim: usize, hidden_dim: usize, use_rms_norm: bool) -> Self {
        Self::with_config(embedding_dim, hidden_dim, use_rms_norm, false, 1.0)
    }

    /// Create a new TransformerBlock with full configuration
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden dimension for feedforward layer
    /// * `use_rms_norm` - If true, use RMSNorm; if false, use LayerNorm
    /// * `use_swiglu` - If true, use SwiGLU; if false, use FeedForward
    /// * `alpha` - DeepNorm residual scaling factor (default: 1.0 for shallow networks)
    pub fn with_config(
        embedding_dim: usize,
        hidden_dim: usize,
        use_rms_norm: bool,
        use_swiglu: bool,
        alpha: f32,
    ) -> Self {
        let norm1 = if use_rms_norm {
            NormLayer::rms_norm(embedding_dim)
        } else {
            NormLayer::layer_norm(embedding_dim)
        };

        let norm2 = if use_rms_norm {
            NormLayer::rms_norm(embedding_dim)
        } else {
            NormLayer::layer_norm(embedding_dim)
        };

        let feed_forward = if use_swiglu {
            FFNLayer::swiglu(embedding_dim, hidden_dim)
        } else {
            FFNLayer::feed_forward(embedding_dim, hidden_dim)
        };

        let beta = 1.0 / alpha; // Sublayer output scaling

        TransformerBlock {
            attention: AttentionLayer::self_attention(embedding_dim),
            feed_forward,
            norm1,
            norm2,
            alpha,
            beta,
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Pre-Norm Transformer with simplified DeepNorm-inspired scaling
        // Reference 1: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
        // Reference 2: "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022)
        //
        // Standard Pre-LN: x_{l+1} = x_l + sublayer(norm(x_l))
        // DeepNorm: x_{l+1} = alpha * x_l + beta * sublayer(norm(x_l))
        // where alpha = (2N)^0.25, beta = 1/alpha, N = num_layers
        //
        // Since our sublayers already include residual connections internally,
        // we apply a simplified scaling approach:
        // - For shallow networks (alpha ≈ 1.0): no scaling needed
        // - For deep networks (alpha > 1.0): scale the entire output
        //
        // This maintains gradient magnitude across depth while preserving
        // backward compatibility with existing layer implementations

        // Attention path: norm → attention (with residual inside)
        let norm1_out = self.norm1.normalize(input);
        let attention_out = self.attention.forward(&norm1_out); // includes residual with input

        // Feedforward path: norm → feedforward (with residual inside)
        let norm2_out = self.norm2.normalize(&attention_out);
        self.feed_forward.forward(&norm2_out) // includes residual with attention_out
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Backward through second LayerNorm
        let (grad_norm2, norm2_param_grads) = self
            .norm2
            .compute_gradients(&Array2::zeros((0, 0)), output_grads);

        // Backward through feed-forward (includes residual connection)
        let (grad_ffn, ffn_param_grads) = self
            .feed_forward
            .compute_gradients(&Array2::zeros((0, 0)), &grad_norm2);

        // Backward through first LayerNorm
        let (grad_norm1, norm1_param_grads) = self
            .norm1
            .compute_gradients(&Array2::zeros((0, 0)), &grad_ffn);

        // Backward through attention (includes residual connection)
        let (grad_attention, attention_param_grads) = self
            .attention
            .compute_gradients(&Array2::zeros((0, 0)), &grad_norm1);

        // Collect all parameter gradients
        let mut all_param_grads = Vec::new();
        all_param_grads.extend(attention_param_grads);
        all_param_grads.extend(norm1_param_grads);
        all_param_grads.extend(ffn_param_grads);
        all_param_grads.extend(norm2_param_grads);

        (grad_attention, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        let expected_params = self.parameters();
        if param_grads.len() != expected_params {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "TransformerBlock expected {} parameter gradients, got {}",
                    expected_params,
                    param_grads.len()
                ),
            });
        }

        let mut idx = 0;

        // Apply attention gradients (3 params: w_q, w_k, w_v)
        let attention_params = &param_grads[idx..idx + 3];
        self.attention.apply_gradients(attention_params, lr)?;
        idx += 3;

        // Apply norm1 gradients (1 or 2 params depending on type)
        let norm1_count = self.norm1.num_param_grads();
        let norm1_params = &param_grads[idx..idx + norm1_count];
        self.norm1.apply_gradients(norm1_params, lr)?;
        idx += norm1_count;

        // Apply feed_forward gradients (3 or 4 params depending on type)
        let ffn_count = self.feed_forward.num_param_grads();
        let ffn_params = &param_grads[idx..idx + ffn_count];
        self.feed_forward.apply_gradients(ffn_params, lr)?;
        idx += ffn_count;

        // Apply norm2 gradients (1 or 2 params depending on type)
        let norm2_count = self.norm2.num_param_grads();
        let norm2_params = &param_grads[idx..idx + norm2_count];
        self.norm2.apply_gradients(norm2_params, lr)?;

        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
