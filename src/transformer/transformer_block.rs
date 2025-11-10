use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    attention::poly_attention::PolyAttention,
    errors::{ModelError, Result},
    llm::Layer,
    mixtures::HeadSelectionStrategy,
    model_config::ModelConfig,
    richards::{RichardsGlu, RichardsNorm},
    mixtures::moe::{MixtureOfExperts, ExpertRouterConfig},
};

/// A complete transformer block containing attention and feedforward components
///
/// This encapsulates the standard transformer block pattern:
/// - Pre-attention normalization
/// - Attention mechanism (with residual connection)
/// - Pre-feedforward normalization
/// - Feedforward network (with residual connection)
#[derive(Serialize, Deserialize, Debug)]
pub struct TransformerBlock {
    /// Pre-attention layer normalization
    pub pre_attention_norm: RichardsNorm,

    /// Attention mechanism (PolyAttention, SelfAttention, etc.)
    pub attention: PolyAttention,

    /// Pre-feedforward layer normalization
    pub pre_ffn_norm: RichardsNorm,

    /// Feedforward network (RichardsGlu or MixtureOfExperts)
    pub feedforward: FeedForwardVariant,

    /// Configuration for this block
    config: TransformerBlockConfig,

    /// Cached intermediate states from forward pass (for gradient computation)
    /// (input, norm1_out, attn_out, residual1, norm2_out, ffn_out)
    #[serde(skip_serializing, skip_deserializing)]
    cached_intermediates: Option<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)>,
}

/// Configuration for a transformer block
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransformerBlockConfig {
    /// Embedding dimension
    pub embed_dim: usize,

    /// Hidden dimension for feedforward
    pub hidden_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Polynomial degree for PolyAttention
    pub poly_degree: usize,

    /// Maximum position for CoPE
    pub max_pos: usize,

    /// Sliding window size (None for full attention)
    pub window_size: Option<usize>,

    /// Whether to use Mixture-of-Experts for feedforward
    pub use_moe: bool,

    /// MoE router configuration (if using MoE)
    pub moe_config: Option<ExpertRouterConfig>,

    /// Head selection strategy for attention
    pub head_selection: HeadSelectionStrategy,
}

/// Feedforward network variants
#[derive(Serialize, Deserialize, Debug)]
pub enum FeedForwardVariant {
    /// Standard RichardsGlu feedforward
    RichardsGlu(Box<RichardsGlu>),

    /// Mixture-of-Experts feedforward
    MixtureOfExperts(Box<MixtureOfExperts>),
}

impl TransformerBlock {
    /// Create a new transformer block with the given configuration
    pub fn new(config: TransformerBlockConfig) -> Self {
        // Create pre-attention normalization
        let pre_attention_norm = RichardsNorm::new(config.embed_dim);

        // Create attention layer
        let mut attention = PolyAttention::new(
            config.embed_dim,
            config.num_heads,
            config.poly_degree,
            config.max_pos,
            config.window_size,
        );
        attention.set_head_selection_config(&config.head_selection);

        // Create pre-FFN normalization
        let pre_ffn_norm = RichardsNorm::new(config.embed_dim);

        // Create feedforward layer
        let feedforward = if config.use_moe {
            if let Some(moe_config) = &config.moe_config {
                let moe_layer = MixtureOfExperts::new(
                    config.embed_dim,
                    (config.embed_dim / 4).max(32), // Router hidden dim
                    moe_config.clone(),
                );
                FeedForwardVariant::MixtureOfExperts(Box::new(moe_layer))
            } else {
                // Fallback to RichardsGlu if MoE config is missing
                let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
                FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
            }
        } else {
            let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
            FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
        };

        Self {
            pre_attention_norm,
            attention,
            pre_ffn_norm,
            feedforward,
            config,
            cached_intermediates: None,
        }
    }

    /// Create a transformer block from a model configuration
    ///
    /// This extracts the relevant parameters from a ModelConfig to create
    /// a transformer block with appropriate settings.
    pub fn from_model_config(config: &ModelConfig, layer_idx: usize) -> Self {
        let block_config = TransformerBlockConfig {
            embed_dim: config.embedding_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.get_num_heads(),
            poly_degree: config.get_poly_degree_p(),
            max_pos: if config.use_adaptive_window {
                config.max_window_size
            } else if let Some(w) = config.window_size {
                w
            } else {
                config.max_seq_len
            }.saturating_sub(1), // CoPE max_pos = window_size - 1
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: config.moe_router.as_ref().map(|router| {
                ExpertRouterConfig::from_router(router)
            }),
            head_selection: config.head_selection.clone(),
        };

        Self::new(block_config)
    }

    /// Get the total number of parameters in this transformer block
    pub fn parameter_count(&self) -> usize {
        self.pre_attention_norm.parameters() +
        self.attention.parameters() +
        self.pre_ffn_norm.parameters() +
        self.feedforward.parameters()
    }

    /// Get the weight norm (Frobenius norm) for LARS adaptive learning rates
    pub fn weight_norm(&self) -> f32 {
        self.pre_attention_norm.weight_norm() +
        self.attention.weight_norm() +
        self.pre_ffn_norm.weight_norm() +
        self.feedforward.weight_norm()
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Pre-attention normalization
        let norm1_out = self.pre_attention_norm.forward(input);

        // Attention with residual connection
        let attn_out = self.attention.forward(&norm1_out);
        let residual1 = input + &attn_out; // Residual: x + attn(x)

        // Pre-feedforward normalization
        let norm2_out = self.pre_ffn_norm.forward(&residual1);

        // Feedforward with residual connection
        let ffn_out = self.feedforward.forward(&norm2_out);
        let output = &residual1 + &ffn_out; // Residual: attn_out + ffn(attn_out)

        // Cache intermediate states for gradient computation
        self.cached_intermediates = Some((
            input.clone(),
            norm1_out,
            attn_out,
            residual1,
            norm2_out,
            ffn_out,
        ));

        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backward through feedforward residual connection
        let ffn_grads = self.feedforward.backward(grads, lr);
        let residual1_grads = grads + &ffn_grads; // Gradient from residual

        // Backward through pre-FFN normalization
        let norm2_grads = self.pre_ffn_norm.backward(&residual1_grads, lr);

        // Backward through attention residual connection
        let attn_grads = self.attention.backward(&norm2_grads, lr);
        let input_grads = &norm2_grads + &attn_grads; // Gradient from residual

        // Backward through pre-attention normalization
        self.pre_attention_norm.backward(&input_grads, lr)
    }

    fn parameters(&self) -> usize {
        self.parameter_count()
    }

    fn weight_norm(&self) -> f32 {
        self.weight_norm()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut all_param_grads = Vec::new();

        if let Some((input, norm1_out, attn_out, residual1, norm2_out, ffn_out)) = &self.cached_intermediates {
            // Compute gradients through the transformer block layers

            // Output = residual1 + ffn_out, so gradients split between residual1 and ffn_out
            let ffn_grads = output_grads.clone();
            let residual1_grads = output_grads.clone();

            // Get feedforward gradients
            let (ffn_input_grad, ffn_param_grads) = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => layer.compute_gradients(norm2_out, &ffn_grads),
                FeedForwardVariant::MixtureOfExperts(layer) => layer.compute_gradients(norm2_out, &ffn_grads),
            };

            // Get pre-FFN norm gradients (stateless)
            let residual1_from_ffn = ffn_input_grad;

            // Combine residual gradients
            let residual1_total_grads = residual1_grads + residual1_from_ffn;

            // residual1 = input + attn_out, so gradients split between input and attn_out
            let input_grads = &residual1_total_grads * 0.5;
            let attn_out_grads = &residual1_total_grads * 0.5;

            // Get attention gradients
            let (attn_input_grad, attn_param_grads) = self.attention.compute_gradients(norm1_out, &attn_out_grads);

            // Get pre-attention norm gradients (stateless)
            let norm1_input_grad = attn_input_grad;

            // The final input gradients are the gradients w.r.t. the transformer input
            // (combining gradients from residual and attention path)
            let final_input_grads = &input_grads + &norm1_input_grad;

            // Collect all parameter gradients
            all_param_grads.extend(attn_param_grads);
            all_param_grads.extend(ffn_param_grads);

            // Note: Norms don't have learnable parameters, so no gradients for them

            (final_input_grads, all_param_grads)
        } else {
            // No cached intermediates - return pass-through gradients and empty parameter gradients
            tracing::warn!("TransformerBlock::compute_gradients called without cached intermediates. Call forward() first.");
            (output_grads.clone(), Vec::new())
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if param_grads.is_empty() {
            return Ok(());
        }

        // Split parameter gradients between attention and feedforward components
        let attention_param_count = self.attention.parameters();
        let feedforward_param_count = match &self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.parameters(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.parameters(),
        };

        // Apply attention gradients
        if param_grads.len() >= attention_param_count {
            let attention_grads = &param_grads[0..attention_param_count];
            self.attention.apply_gradients(attention_grads, lr)?;
        }

        // Apply feedforward gradients
        let feedforward_start = attention_param_count;
        if param_grads.len() >= feedforward_start + feedforward_param_count {
            let feedforward_grads = &param_grads[feedforward_start..feedforward_start + feedforward_param_count];
            match &mut self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(feedforward_grads, lr)?,
                FeedForwardVariant::MixtureOfExperts(layer) => layer.apply_gradients(feedforward_grads, lr)?,
            }
        }

        Ok(())
    }
}

impl FeedForwardVariant {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward(input),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.backward(grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.backward(grads, lr),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(param_grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.parameters(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.parameters(),
        }
    }

    fn weight_norm(&self) -> f32 {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.weight_norm(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.weight_norm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::ModelConfig;

    #[test]
    fn test_transformer_block_creation() {
        let config = TransformerBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 1023,
            window_size: Some(4096),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::SoftTopP {
                top_p: 0.9,
                soft_top_p_alpha: 15.0,
            },
        };

        let block = TransformerBlock::new(config);
        assert_eq!(block.layer_type(), "TransformerBlock");
        assert!(block.parameter_count() > 0);
    }

    #[test]
    fn test_transformer_block_from_model_config() {
        let model_config = ModelConfig::transformer(128, 256, 3, 80, None, Some(8));
        let block = TransformerBlock::from_model_config(&model_config, 0);

        assert_eq!(block.layer_type(), "TransformerBlock");
        assert!(block.parameter_count() > 0);
    }

    #[test]
    fn test_transformer_block_forward_backward() {
        let embed_dim = 128;
        let seq_len = 10;
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 79, // max_seq_len - 1
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::SoftTopP {
                top_p: 0.9,
                soft_top_p_alpha: 15.0,
            },
        };

        let mut block = TransformerBlock::new(config);

        // Test forward pass
        let input = Array2::zeros((seq_len, embed_dim)); // seq_len, embed_dim
        let output = block.forward(&input);
        assert_eq!(output.shape(), input.shape());

        // Test backward pass
        let grads = Array2::ones((seq_len, embed_dim));
        let input_grads = block.backward(&grads, 0.001);
        assert_eq!(input_grads.shape(), input.shape());
    }
}
