use std::sync::RwLock;

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    attention::poly_attention::PolyAttention,
    errors::Result,
    llm::Layer,
    mixtures::{
        HeadSelectionStrategy,
        moe::{ExpertRouterConfig, MixtureOfExperts},
    },
    model_config::ModelConfig,
    richards::{RichardsGlu, RichardsNorm},
};

/// Type alias for cached transformer block intermediates to improve readability
type CachedIntermediates = (
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
);

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
    cached_intermediates: Option<CachedIntermediates>,

    /// Cached gradient partition sizes so apply_gradients can route slices correctly
    #[serde(skip_serializing, skip_deserializing)]
    param_partitions: RwLock<Option<ParamPartitions>>,
}

#[derive(Clone, Debug, Default)]
struct ParamPartitions {
    attention: usize,
    feedforward: usize,
    pre_ffn_norm: usize,
    pre_attn_norm: usize,
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
    /// Analytical gradient invariants:
    /// - Residual splits: output = residual1 + ffn_out → d_residual1 and d_ffn_out both receive
    ///   upstream grads
    /// - Norm chain: d_residual1_from_ffn = pre_ffn_norm.backward(d_norm2_out)
    /// - Residual combine: d_residual1_total = d_output + d_residual1_from_ffn
    /// - Attention split: residual1 = input + attn_out → d_input_direct and d_attn_out both receive
    ///   d_residual1_total
    /// - Final input grads: d_input = d_input_direct + pre_attention_norm.backward(d_norm1_out)
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
            param_partitions: RwLock::new(None),
        }
    }

    /// Create a transformer block from a model configuration
    ///
    /// This extracts the relevant parameters from a ModelConfig to create
    /// a transformer block with appropriate settings.
    pub fn from_model_config(config: &ModelConfig, _layer_idx: usize) -> Self {
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
            }
            .saturating_sub(1), // CoPE max_pos = window_size - 1
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: config
                .moe_router
                .as_ref()
                .map(|router| ExpertRouterConfig::from_router(router)),
            head_selection: config.head_selection.clone(),
        };

        Self::new(block_config)
    }

    /// Get the total number of parameters in this transformer block
    pub fn parameter_count(&self) -> usize {
        self.pre_attention_norm.parameters()
            + self.attention.parameters()
            + self.pre_ffn_norm.parameters()
            + self.feedforward.parameters()
    }

    /// Get the weight norm (Frobenius norm) for LARS adaptive learning rates
    pub fn weight_norm(&self) -> f32 {
        self.pre_attention_norm.weight_norm()
            + self.attention.weight_norm()
            + self.pre_ffn_norm.weight_norm()
            + self.feedforward.weight_norm()
    }
}

impl ParamPartitions {
    fn total(&self) -> usize {
        self.attention + self.feedforward + self.pre_ffn_norm + self.pre_attn_norm
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
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.parameter_count()
    }

    fn weight_norm(&self) -> f32 {
        self.weight_norm()
    }

    /// Compute analytical gradients using cached forward intermediates
    /// Ensures full-gradient propagation across residual connections
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let mut all_param_grads = Vec::new();

        if let Some((input_cached, norm1_out, _attn_out, residual1, norm2_out, _ffn_out)) =
            &self.cached_intermediates
        {
            // Compute gradients through the transformer block layers

            // Output = residual1 + ffn_out, so gradients split between residual1 and ffn_out
            let ffn_grads = output_grads.clone();
            let residual1_grads = output_grads.clone();

            // Get feedforward gradients
            let (ffn_input_grad, ffn_param_grads) = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => {
                    layer.compute_gradients(norm2_out, &ffn_grads)
                }
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.compute_gradients(norm2_out, &ffn_grads)
                }
            };

            let (residual1_from_ffn, pre_ffn_param_grads) = self
                .pre_ffn_norm
                .compute_gradients(residual1, &ffn_input_grad);

            // Combine residual gradients
            let residual1_total_grads = residual1_grads + residual1_from_ffn;

            // residual1 = input + attn_out: propagate full upstream gradient to both branches
            let input_grads = residual1_total_grads.clone();
            let attn_out_grads = residual1_total_grads.clone();

            // Get attention gradients
            let (attn_input_grad, attn_param_grads) =
                self.attention.compute_gradients(norm1_out, &attn_out_grads);

            let (norm1_input_grad, pre_attn_param_grads) = self
                .pre_attention_norm
                .compute_gradients(input_cached, &attn_input_grad);

            // The final input gradients are the gradients w.r.t. the transformer input
            // (combining gradients from residual and attention path)
            let final_input_grads = &input_grads + &norm1_input_grad;

            // Capture gradient partition sizes so apply_gradients can re-slice accurately later
            let partitions = ParamPartitions {
                attention: attn_param_grads.len(),
                feedforward: ffn_param_grads.len(),
                pre_ffn_norm: pre_ffn_param_grads.len(),
                pre_attn_norm: pre_attn_param_grads.len(),
            };
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(partitions);
            }

            // Collect all parameter gradients in deterministic order
            all_param_grads.extend(attn_param_grads);
            all_param_grads.extend(ffn_param_grads);
            all_param_grads.extend(pre_ffn_param_grads);
            all_param_grads.extend(pre_attn_param_grads);

            (final_input_grads, all_param_grads)
        } else {
            // No cached intermediates - return pass-through gradients and empty parameter gradients
            tracing::warn!(
                "TransformerBlock::compute_gradients called without cached intermediates. Call forward() first."
            );
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = None;
            }
            (output_grads.clone(), Vec::new())
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if param_grads.is_empty() {
            return Ok(());
        }

        let cached_partitions = self
            .param_partitions
            .read()
            .map(|guard| guard.clone())
            .unwrap_or(None);

        let partitions = cached_partitions
            .or_else(|| {
                if !param_grads.is_empty() {
                    tracing::warn!(
                        "TransformerBlock::apply_gradients missing partition metadata; falling back to legacy routing"
                    );
                }
                None
            })
            .unwrap_or_else(|| ParamPartitions {
            attention: param_grads.len(),
            ..ParamPartitions::default()
            });

        // Sanitize and globally clip gradients
        let mut sanitized: Vec<Array2<f32>> = Vec::with_capacity(param_grads.len());
        let mut norm_sq: f32 = 0.0;
        for g in param_grads {
            let mut gg = g.clone();
            gg.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
            norm_sq += gg.iter().map(|&x| x * x).sum::<f32>();
            sanitized.push(gg);
        }
        let clip = 5.0f32;
        let nrm = norm_sq.sqrt();
        if nrm.is_finite() && nrm > clip && nrm > 0.0 {
            let scale = clip / nrm;
            for gg in &mut sanitized {
                gg.mapv_inplace(|x| x * scale);
            }
        }

        let mut idx = 0usize;
        let total_expected = partitions.total();
        if total_expected != sanitized.len() {
            tracing::warn!(
                expected = total_expected,
                actual = sanitized.len(),
                "TransformerBlock::apply_gradients received unexpected gradient count"
            );
        }

        let mut next_range = |count: usize| {
            let available = sanitized.len().saturating_sub(idx);
            let len = count.min(available);
            let start = idx;
            idx += len;
            start..idx
        };

        // Apply attention gradients with adaptive scaling (LARS-style)
        let attn_range = next_range(partitions.attention);
        let attention_grads = &sanitized[attn_range];
        if !attention_grads.is_empty() {
            let gnorm_attn: f32 = attention_grads
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_attn = self.attention.weight_norm().max(1e-6);
            let scale_attn = (wnorm_attn / (gnorm_attn.max(1e-6))).clamp(0.5, 2.0);
            let scaled: Vec<Array2<f32>> = attention_grads
                .iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale_attn);
                    gg
                })
                .collect();
            self.attention.apply_gradients(&scaled, lr)?;
        }

        // Apply feedforward gradients with adaptive scaling
        let ffn_range = next_range(partitions.feedforward);
        let feedforward_grads = &sanitized[ffn_range];
        if !feedforward_grads.is_empty() {
            let gnorm_ffn: f32 = feedforward_grads
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_ffn = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(l) => l.weight_norm(),
                FeedForwardVariant::MixtureOfExperts(l) => l.weight_norm(),
            }
            .max(1e-6);
            let scale_ffn = (wnorm_ffn / (gnorm_ffn.max(1e-6))).clamp(0.5, 2.0);
            let scaled: Vec<Array2<f32>> = feedforward_grads
                .iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale_ffn);
                    gg
                })
                .collect();
            match &mut self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(&scaled, lr)?,
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.apply_gradients(&scaled, lr)?
                }
            }
        }

        // Apply pre-FFN norm gradients
        let pre_ffn_range = next_range(partitions.pre_ffn_norm);
        let pre_ffn_grads = &sanitized[pre_ffn_range];
        if !pre_ffn_grads.is_empty() {
            self.pre_ffn_norm.apply_gradients(pre_ffn_grads, lr)?;
        }

        // Apply pre-attention norm gradients
        let pre_attn_range = next_range(partitions.pre_attn_norm);
        let pre_attn_grads = &sanitized[pre_attn_range];
        if !pre_attn_grads.is_empty() {
            self.pre_attention_norm.apply_gradients(pre_attn_grads, lr)?;
        }

        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = None;
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

// performance tests are included in other modules or can be added under existing tests
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
        let input_grads = block.backward(&grads, 0.0);
        assert_eq!(input_grads.shape(), input.shape());
    }

    #[test]
    fn test_transformer_block_shape_validation() {
        let embed_dim = 64;
        let seq_len = 5;
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: 128,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 63,
            window_size: Some(32),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        };
        let mut block = TransformerBlock::new(config);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let out = block.forward(&input);
        assert_eq!(out.shape(), input.shape());
        let grads = Array2::<f32>::ones((seq_len, embed_dim));
        let (in_grad, param_grads) = block.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        assert!(param_grads.iter().all(|g| g.ncols() > 0));
    }

    #[test]
    fn test_transformer_block_input_gradients_numeric() {
        let embed_dim = 8;
        let seq_len = 2;
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 15,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
        };
        let mut block = TransformerBlock::new(config);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _out = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));
        let (in_grad, param_grads) = block.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        assert!(in_grad.iter().all(|&x| x.is_finite()));
        let gnorm: f32 = in_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        let onorm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(gnorm <= onorm * 100.0);
        assert!(!param_grads.is_empty());
    }

    #[test]
    fn test_transformer_block_backward_matches_analytical() {
        let embed_dim = 32;
        let seq_len = 6;
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: 64,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 31,
            window_size: Some(16),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        };
        let mut block = TransformerBlock::new(config);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _out = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));

        let (in_grad_analytical, _param_grads) = block.compute_gradients(&input, &grads);
        let in_grad_backward = block.backward(&grads, 0.0);

        assert_eq!(in_grad_backward.shape(), input.shape());
        assert!(in_grad_backward.iter().all(|&x| x.is_finite()));

        let mut diff_sq = 0.0f32;
        for (a, b) in in_grad_analytical.iter().zip(in_grad_backward.iter()) {
            let d = a - b;
            diff_sq += d * d;
        }
        let rmse = (diff_sq / (seq_len * embed_dim) as f32).sqrt();
        assert!(rmse < 1e-3, "RMSE too large: {}", rmse);
    }

    #[test]
    fn test_transformer_block_partitioned_apply_gradients() {
        let embed_dim = 16;
        let seq_len = 4;
        let config = TransformerBlockConfig {
            embed_dim,
            hidden_dim: 32,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 31,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
        };

        let mut block = TransformerBlock::new(config);
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _ = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));
        let (_in_grad, param_grads) = block.compute_gradients(&input, &grads);
        assert!(!param_grads.is_empty());

        // Should apply without panicking and reset partitions afterward
        block.apply_gradients(&param_grads, 1e-3).unwrap();
    }
}
