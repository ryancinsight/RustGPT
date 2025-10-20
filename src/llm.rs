use std::cmp::Ordering;
use std::fs;

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::{
    Embeddings, MAX_SEQ_LEN, Vocab,
    errors::{ModelError, Result},
    gradient_clipping::{AdaptiveClippingConfig, AdaptiveGradientClipping, GradientClipping},
    output_projection::OutputProjection,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum LayerEnum {
    Embeddings(Embeddings),
    SelfAttention(Box<crate::self_attention::SelfAttention>),
    AttentionMoE(Box<crate::attention_moe::AttentionMoELayer>),
    FeedForward(Box<crate::feed_forward::FeedForward>),
    SwiGLU(Box<crate::swiglu::SwiGLU>),
    MoE(Box<crate::moe::MoELayer>),
    LayerNorm(crate::layer_norm::LayerNorm),
    RMSNorm(crate::rms_norm::RMSNorm),
    OutputProjection(OutputProjection),
    HyperMixerBlock(Box<crate::hypermixer::HyperMixerBlock>),
    HRMBlock(Box<crate::hrm::HRMBlock>),
    TRMBlock(Box<crate::trm::TinyRecursiveModel>),
}

impl LayerEnum {
    /// Downcast to SelfAttention layer if this is a SelfAttention variant
    pub fn as_self_attention(&self) -> Option<&crate::self_attention::SelfAttention> {
        match self {
            LayerEnum::SelfAttention(layer) => Some(layer.as_ref()),
            _ => None,
        }
    }

    /// Downcast to mutable SelfAttention layer if this is a SelfAttention variant
    pub fn as_self_attention_mut(&mut self) -> Option<&mut crate::self_attention::SelfAttention> {
        match self {
            LayerEnum::SelfAttention(layer) => Some(layer.as_mut()),
            _ => None,
        }
    }

    /// Downcast to AttentionMoE layer if this is an AttentionMoE variant
    pub fn as_attention_moe(&self) -> Option<&crate::attention_moe::AttentionMoELayer> {
        match self {
            LayerEnum::AttentionMoE(layer) => Some(layer.as_ref()),
            _ => None,
        }
    }

    /// Downcast to mutable AttentionMoE layer if this is an AttentionMoE variant
    pub fn as_attention_moe_mut(&mut self) -> Option<&mut crate::attention_moe::AttentionMoELayer> {
        match self {
            LayerEnum::AttentionMoE(layer) => Some(layer.as_mut()),
            _ => None,
        }
    }

    /// Downcast to TRMBlock layer if this is a TRMBlock variant
    pub fn as_trm_block(&self) -> Option<&crate::trm::TinyRecursiveModel> {
        match self {
            LayerEnum::TRMBlock(layer) => Some(layer.as_ref()),
            _ => None,
        }
    }

    /// Downcast to mutable TRMBlock layer if this is a TRMBlock variant
    pub fn as_trm_block_mut(&mut self) -> Option<&mut crate::trm::TinyRecursiveModel> {
        match self {
            LayerEnum::TRMBlock(layer) => Some(layer.as_mut()),
            _ => None,
        }
    }
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            LayerEnum::Embeddings(layer) => layer.layer_type(),
            LayerEnum::SelfAttention(layer) => layer.layer_type(),
            LayerEnum::AttentionMoE(_) => "AttentionMoE",
            LayerEnum::FeedForward(layer) => layer.layer_type(),
            LayerEnum::SwiGLU(layer) => layer.layer_type(),
            LayerEnum::MoE(layer) => layer.layer_type(),
            LayerEnum::LayerNorm(layer) => layer.layer_type(),
            LayerEnum::RMSNorm(layer) => layer.layer_type(),
            LayerEnum::OutputProjection(layer) => layer.layer_type(),
            LayerEnum::HyperMixerBlock(layer) => layer.layer_type(),
            LayerEnum::HRMBlock(layer) => layer.layer_type(),
            LayerEnum::TRMBlock(layer) => layer.layer_type(),
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.forward(input),
            LayerEnum::SelfAttention(layer) => layer.forward(input),
            LayerEnum::AttentionMoE(layer) => layer.forward(input),
            LayerEnum::FeedForward(layer) => layer.forward(input),
            LayerEnum::SwiGLU(layer) => layer.forward(input),
            LayerEnum::MoE(layer) => layer.forward(input),
            LayerEnum::LayerNorm(layer) => layer.forward(input),
            LayerEnum::RMSNorm(layer) => layer.forward(input),
            LayerEnum::OutputProjection(layer) => layer.forward(input),
            LayerEnum::HyperMixerBlock(layer) => layer.forward(input),
            LayerEnum::HRMBlock(layer) => layer.forward(input),
            LayerEnum::TRMBlock(layer) => layer.forward(input),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            LayerEnum::Embeddings(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::SelfAttention(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::AttentionMoE(_) => {
                panic!("AttentionMoE does not support compute_gradients - use backward() instead");
            }
            LayerEnum::FeedForward(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::SwiGLU(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::MoE(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::LayerNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::RMSNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::OutputProjection(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::HyperMixerBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::HRMBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::TRMBlock(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match self {
            LayerEnum::Embeddings(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::SelfAttention(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::AttentionMoE(_) => {
                panic!("AttentionMoE does not support apply_gradients - use backward() instead");
            }
            LayerEnum::FeedForward(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::SwiGLU(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::MoE(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::LayerNorm(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::RMSNorm(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::OutputProjection(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::HyperMixerBlock(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::HRMBlock(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::TRMBlock(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.backward(grads, lr),
            LayerEnum::SelfAttention(layer) => layer.backward(grads, lr),
            LayerEnum::AttentionMoE(layer) => layer.backward(grads, lr),
            LayerEnum::FeedForward(layer) => layer.backward(grads, lr),
            LayerEnum::SwiGLU(layer) => layer.backward(grads, lr),
            LayerEnum::MoE(layer) => layer.backward(grads, lr),
            LayerEnum::LayerNorm(layer) => layer.backward(grads, lr),
            LayerEnum::RMSNorm(layer) => layer.backward(grads, lr),
            LayerEnum::OutputProjection(layer) => layer.backward(grads, lr),
            LayerEnum::HyperMixerBlock(layer) => layer.backward(grads, lr),
            LayerEnum::HRMBlock(layer) => layer.backward(grads, lr),
            LayerEnum::TRMBlock(layer) => layer.backward(grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            LayerEnum::Embeddings(layer) => layer.parameters(),
            LayerEnum::SelfAttention(layer) => layer.parameters(),
            LayerEnum::AttentionMoE(_) => 0, // TODO: Implement parameter counting
            LayerEnum::FeedForward(layer) => layer.parameters(),
            LayerEnum::SwiGLU(layer) => layer.parameters(),
            LayerEnum::MoE(layer) => layer.parameters(),
            LayerEnum::LayerNorm(layer) => layer.parameters(),
            LayerEnum::RMSNorm(layer) => layer.parameters(),
            LayerEnum::OutputProjection(layer) => layer.parameters(),
            LayerEnum::HyperMixerBlock(layer) => layer.parameters(),
            LayerEnum::TRMBlock(layer) => layer.parameters(),
            LayerEnum::HRMBlock(layer) => layer.parameters(),
        }
    }
}

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>);

    /// Apply gradients to layer parameters
    /// Returns GradientError if param_grads has incorrect length
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()>;
}

#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,
    #[serde(skip)]
    pub gradient_clipper: Option<Box<dyn GradientClipping>>,
}

impl std::fmt::Debug for LLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLM")
            .field("vocab", &self.vocab)
            .field("network", &self.network)
            .field("gradient_clipper", &"Box<dyn GradientClipping>")
            .finish()
    }
}

impl Default for LLM {
    fn default() -> Self {
        use crate::model_builder::build_network;
        use crate::model_config::ModelConfig;

        let config = ModelConfig::default();
        let vocab = Vocab::default();
        let network = build_network(&config, &vocab);

        Self {
            vocab,
            network,
            gradient_clipper: Some(Box::new(AdaptiveGradientClipping::new(
                AdaptiveClippingConfig::default(),
            ))),
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        Self {
            vocab,
            network,
            gradient_clipper: Some(Box::new(AdaptiveGradientClipping::new(
                AdaptiveClippingConfig::default(),
            ))),
        }
    }
}

impl LLM {
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|layer| layer.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        // Sum the parameters across all layers in the network
        self.network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>()
    }

    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);

        // Handle empty output
        if output_tokens.is_empty() {
            return String::new();
        }

        // Convert token_ids to strings
        let token_strs = output_tokens
            .iter()
            .map(|&t| self.vocab.decode.get(&t).unwrap().as_str())
            .collect::<Vec<&str>>();

        token_strs.join(" ")
    }

    fn forward(&mut self, text: &str) -> Vec<usize> {
        // Tokenize the input text
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        // Safety check: ensure we have at least one token
        if tokenized.is_empty() {
            return output_tokens;
        }

        let input_len = tokenized.len();

        // Prevent overflow if input_len >= MAX_SEQ_LEN
        if input_len >= MAX_SEQ_LEN {
            return output_tokens;
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            // let tokenized_clone = tokenized.clone();

            // Check if we're approaching the maximum sequence length
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }

            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            // Safety check: ensure we have at least one token
            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            // Softmax - convert activations of each token to a probability distribution over the
            // vocabulary
            let probs = Self::softmax(&last_logit); // 1 x vocab_size

            // Greedy Decode - Choose the highest probability token for each position
            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        output_tokens
    }

    #[instrument(skip(self, data))]
    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) -> Result<()> {
        self.train_with_batch_size(data, epochs, lr, 1)
    }

    /// Train with configurable batch size for improved performance
    pub fn train_with_batch_size(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        self.train_with_warmup(data, epochs, lr, batch_size, 15) // 15 warmup epochs for better stability
    }

    /// Train with learning rate warmup for stability
    ///
    /// Warmup prevents gradient explosion in early training by gradually increasing
    /// the learning rate from 0 to the target value over warmup_epochs.
    ///
    /// Reference: "Attention is All You Need" (Vaswani et al., 2017)
    pub fn train_with_warmup(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        target_lr: f32,
        batch_size: usize,
        warmup_epochs: usize,
    ) -> Result<()> {
        let tokenized_data = data
            .par_iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut batch_count = 0;

            // Learning rate warmup + cosine annealing
            // Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, 2016)
            let effective_lr = if epoch < warmup_epochs {
                // Linear warmup: gradually increase LR from 0 to target
                target_lr * ((epoch + 1) as f32 / warmup_epochs as f32)
            } else {
                // Cosine annealing after warmup to escape loss plateaus
                // Formula: lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
                let t = (epoch - warmup_epochs) as f32;
                let t_max = (epochs - warmup_epochs) as f32;
                let lr_min = target_lr * 0.10; // Minimum LR is 10% of base LR (gentler decay)
                let lr_max = target_lr;

                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
            };

            // Compute training progress for adaptive MoH
            let training_progress = if epoch < warmup_epochs {
                0.0
            } else {
                (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32
            };

            // Set epoch info for all MoH layers (for warm-up and annealing)
            for layer in self.network.iter_mut() {
                if let Some(attn_layer) = layer.as_self_attention_mut() {
                    attn_layer.set_epoch_info(epoch, epochs);
                } else if let Some(moe_layer) = layer.as_attention_moe_mut() {
                    moe_layer.set_epoch_info(epoch, epochs);
                } else if let Some(trm_block) = layer.as_trm_block_mut() {
                    trm_block.set_epoch_info(epoch, epochs);
                }
            }

            // Process data in batches
            for batch in tokenized_data.chunks(batch_size) {
                let (batch_loss, grad_norm) = self.train_batch(batch, effective_lr, training_progress)?;
                total_loss += batch_loss;
                total_grad_norm += grad_norm;
                batch_count += 1;
            }

            let avg_loss = total_loss / tokenized_data.len() as f32;
            let avg_grad_norm = total_grad_norm / batch_count as f32;

            // NFR-5.2: Training divergence detection
            if avg_loss.is_nan() || avg_loss.is_infinite() {
                return Err(ModelError::Training {
                    message: format!(
                        "Training diverged at epoch {}: loss is {} (NaN or Inf detected)",
                        epoch, avg_loss
                    ),
                });
            }

            if avg_loss > 1e6 {
                return Err(ModelError::Training {
                    message: format!(
                        "Training diverged at epoch {}: loss exceeded threshold (loss = {:.2e} > 1e6)",
                        epoch, avg_loss
                    ),
                });
            }

            // NFR-7.3: Training metrics with gradient norms
            let warmup_status = if epoch < warmup_epochs {
                format!(" (warmup {}/{})", epoch + 1, warmup_epochs)
            } else {
                String::new()
            };

            // Log MoH statistics if enabled (with adaptive metrics)
            let training_progress = if epoch < warmup_epochs {
                0.0
            } else {
                (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32
            };

            // Collect stats from all MoH layers
            let mut moh_layers_stats = Vec::new();
            let mut has_learned_predictor = false;
            let mut threshold_range_min = f32::INFINITY;
            let mut threshold_range_max = f32::NEG_INFINITY;
            let mut total_conf_avg = 0.0;
            let mut total_conf_min = f32::INFINITY;
            let mut total_fallback_pct = 0.0;
            let mut total_pred_norm = 0.0;
            let mut total_complexity_avg = 0.0;
            let mut total_complexity_min = f32::INFINITY;
            let mut total_complexity_max = f32::NEG_INFINITY;
            let mut total_temp_avg = 0.0;
            let mut total_temp_min = f32::INFINITY;
            let mut total_temp_max = f32::NEG_INFINITY;
            let mut has_temperature_stats = false;
            let mut moh_layer_count = 0;

            for (layer_idx, layer) in self.network.iter().enumerate() {
                if let Some(attn_layer) = layer.as_self_attention() {
                    let avg_routed = attn_layer.get_avg_active_routed_heads();
                    if avg_routed > 0.0 {
                        let (min_thresh, max_thresh, mean_thresh, _std_thresh) = attn_layer.get_threshold_stats();
                        let dyn_weight = attn_layer.get_dynamic_loss_weight(training_progress);
                        moh_layers_stats.push((layer_idx, avg_routed, mean_thresh, dyn_weight));

                        // Track threshold range across all layers
                        threshold_range_min = threshold_range_min.min(min_thresh);
                        threshold_range_max = threshold_range_max.max(max_thresh);

                        // Track confidence statistics
                        let (conf_avg, conf_min, fallback_pct) = attn_layer.get_confidence_stats();
                        total_conf_avg += conf_avg;
                        total_conf_min = total_conf_min.min(conf_min);
                        total_fallback_pct += fallback_pct;

                        // Track complexity statistics
                        let (complexity_avg, complexity_min, complexity_max) = attn_layer.get_complexity_stats();
                        total_complexity_avg += complexity_avg;
                        total_complexity_min = total_complexity_min.min(complexity_min);
                        total_complexity_max = total_complexity_max.max(complexity_max);

                        // Track predictor weight norm
                        let pred_norm = attn_layer.get_predictor_weight_norm();
                        total_pred_norm += pred_norm;

                        // Track temperature statistics (for fully adaptive MoH)
                        if let Some((temp_avg, temp_min, temp_max)) = attn_layer.get_temperature_stats() {
                            total_temp_avg += temp_avg;
                            total_temp_min = total_temp_min.min(temp_min);
                            total_temp_max = total_temp_max.max(temp_max);
                            has_temperature_stats = true;
                        }

                        // Check if any layer has learned predictor
                        if attn_layer.has_learned_predictor() {
                            has_learned_predictor = true;
                        }

                        moh_layer_count += 1;
                    }
                } else if let Some(trm_block) = layer.as_trm_block() {
                    // For TRM, get MoH stats from internal attention layer
                    let (avg_routed, mean_thresh, conf_avg, conf_min, fallback_pct,
                         complexity_avg, complexity_min, complexity_max, pred_norm) = trm_block.get_moh_stats();
                    if avg_routed > 0.0 {
                        let dyn_weight = 0.0; // TRM doesn't have dynamic loss weight yet
                        moh_layers_stats.push((layer_idx, avg_routed, mean_thresh, dyn_weight));

                        // Track threshold range (approximate from mean)
                        threshold_range_min = threshold_range_min.min(mean_thresh - 0.1);
                        threshold_range_max = threshold_range_max.max(mean_thresh + 0.1);

                        // Track confidence statistics
                        total_conf_avg += conf_avg;
                        total_conf_min = total_conf_min.min(conf_min);
                        total_fallback_pct += fallback_pct;

                        // Track complexity statistics (now with proper min/max)
                        total_complexity_avg += complexity_avg;
                        total_complexity_min = total_complexity_min.min(complexity_min);
                        total_complexity_max = total_complexity_max.max(complexity_max);

                        // Track predictor weight norm
                        total_pred_norm += pred_norm;

                        // Track temperature statistics (for fully adaptive MoH)
                        if let Some((temp_avg, temp_min, temp_max)) = trm_block.get_temperature_stats() {
                            total_temp_avg += temp_avg;
                            total_temp_min = total_temp_min.min(temp_min);
                            total_temp_max = total_temp_max.max(temp_max);
                            has_temperature_stats = true;
                        }

                        has_learned_predictor = true;
                        moh_layer_count += 1;
                    }
                } else if let Some(moe_layer) = layer.as_attention_moe() {
                    // For AttentionMoE, get per-expert MoH statistics
                    let expert_stats = moe_layer.get_expert_moh_stats();
                    if !expert_stats.is_empty() {
                        // Aggregate statistics across all experts
                        let mut total_avg_routed = 0.0;
                        let mut total_threshold = 0.0;
                        let mut expert_count = 0;

                        for (_expert_idx, avg_heads, threshold_p, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm) in &expert_stats {
                            if *avg_heads > 0.0 {
                                total_avg_routed += avg_heads;
                                total_threshold += threshold_p;
                                total_conf_avg += conf_avg;
                                total_conf_min = total_conf_min.min(*conf_min);
                                total_fallback_pct += fallback_pct;
                                total_complexity_avg += complexity_avg;
                                total_pred_norm += pred_norm;
                                expert_count += 1;
                            }
                        }

                        if expert_count > 0 {
                            let avg_routed = total_avg_routed / expert_count as f32;
                            let mean_thresh = total_threshold / expert_count as f32;
                            let dyn_weight = 0.0; // AttentionMoE doesn't have dynamic loss weight yet

                            moh_layers_stats.push((layer_idx, avg_routed, mean_thresh, dyn_weight));

                            // Track threshold range (approximate from mean)
                            threshold_range_min = threshold_range_min.min(mean_thresh - 0.1);
                            threshold_range_max = threshold_range_max.max(mean_thresh + 0.1);

                            has_learned_predictor = true;
                            moh_layer_count += expert_count;
                        }
                    }
                }
            }

            // Compute averages
            let avg_conf = if moh_layer_count > 0 { total_conf_avg / moh_layer_count as f32 } else { 0.0 };
            let avg_fallback_pct = if moh_layer_count > 0 { total_fallback_pct / moh_layer_count as f32 } else { 0.0 };
            let avg_pred_norm = if moh_layer_count > 0 { total_pred_norm / moh_layer_count as f32 } else { 0.0 };
            let avg_complexity = if moh_layer_count > 0 { total_complexity_avg / moh_layer_count as f32 } else { 0.0 };

            // Format MoH stats: show first, middle, and last layers + confidence + predictor stats
            let mut moh_stats = String::new();
            if !moh_layers_stats.is_empty() {
                let first = &moh_layers_stats[0];
                let last = moh_layers_stats.last().unwrap();

                // Add threshold range if learned predictor is enabled
                let threshold_range_str = if has_learned_predictor && threshold_range_min.is_finite() {
                    format!(" | ThreshRange: [{:.2}-{:.2}]", threshold_range_min, threshold_range_max)
                } else {
                    String::new()
                };

                // Add confidence statistics
                let confidence_str = if avg_conf > 0.0 {
                    format!(" | ConfAvg: {:.2}, ConfMin: {:.2}, Fallback: {:.1}%",
                            avg_conf, total_conf_min, avg_fallback_pct)
                } else {
                    String::new()
                };

                // Add predictor weight norm if learned predictor is enabled
                let predictor_str = if has_learned_predictor && avg_pred_norm > 0.0 {
                    format!(" | PredNorm: {:.4}", avg_pred_norm)
                } else {
                    String::new()
                };

                // Add complexity statistics
                let complexity_str = if avg_complexity > 0.0 && total_complexity_min.is_finite() {
                    format!(" | Complexity: {:.3} [{:.3}-{:.3}]",
                            avg_complexity, total_complexity_min, total_complexity_max)
                } else {
                    String::new()
                };

                // Add temperature statistics (for fully adaptive MoH)
                let avg_temp = if moh_layer_count > 0 { total_temp_avg / moh_layer_count as f32 } else { 0.0 };
                let temperature_str = if has_temperature_stats && total_temp_min.is_finite() {
                    format!(" | Temp: {:.2} [{:.2}-{:.2}]",
                            avg_temp, total_temp_min, total_temp_max)
                } else {
                    String::new()
                };

                // Only show DynW if it's non-zero (Standard MoH only)
                let dyn_weight_str = if first.3 > 1e-10 {
                    format!(" | DynW: {:.2e}", first.3)
                } else {
                    String::new()
                };

                if moh_layers_stats.len() > 2 {
                    let mid = &moh_layers_stats[moh_layers_stats.len() / 2];
                    moh_stats = format!(
                        " | MoH L{}: {:.2}h@{:.2}p | L{}: {:.2}h@{:.2}p | L{}: {:.2}h@{:.2}p{}{}{}{}{}{}",
                        first.0, first.1, first.2,
                        mid.0, mid.1, mid.2,
                        last.0, last.1, last.2,
                        dyn_weight_str,
                        threshold_range_str,
                        confidence_str,
                        predictor_str,
                        complexity_str,
                        temperature_str
                    );
                } else {
                    moh_stats = format!(
                        " | MoH L{}: {:.2}h@{:.2}p | L{}: {:.2}h@{:.2}p{}{}{}{}{}{}",
                        first.0, first.1, first.2,
                        last.0, last.1, last.2,
                        dyn_weight_str,
                        threshold_range_str,
                        confidence_str,
                        predictor_str,
                        complexity_str,
                        temperature_str
                    );
                }
            }

            // Log TRM statistics if present
            let mut trm_stats = String::new();
            for layer in self.network.iter() {
                if let LayerEnum::TRMBlock(trm) = layer {
                    let scales = trm.get_step_scales();

                    // Add depth statistics if adaptive depth is enabled
                    let depth_info = if let Some((avg, min, max)) = trm.get_depth_stats() {
                        format!(" Depth: avg={:.1} [{}-{}]", avg, min, max)
                    } else {
                        String::new()
                    };

                    trm_stats = format!(" | TRM: {}{}", scales, depth_info);
                    break;
                }
            }

            // Log HyperMixer statistics if present
            let mut hypermixer_stats = String::new();
            let mut hypermixer_count = 0;
            for layer in self.network.iter() {
                if let LayerEnum::HyperMixerBlock(hm) = layer {
                    if hypermixer_count == 0 {
                        // Log first block
                        hypermixer_stats = format!(" | HM: {}", hm.get_scales());
                    }
                    hypermixer_count += 1;
                }
            }

            info!(
                epoch = epoch,
                loss = avg_loss,
                grad_norm = avg_grad_norm,
                learning_rate = effective_lr,
                "Training epoch completed{}{}{}{}",
                warmup_status,
                moh_stats,
                trm_stats,
                hypermixer_stats
            );

            // Log detailed AttentionMoE statistics (per-expert metrics)
            for (layer_idx, layer) in self.network.iter().enumerate() {
                if let Some(moe_layer) = layer.as_attention_moe() {
                    moe_layer.log_expert_stats(layer_idx, epoch);
                }
            }
        }

        Ok(())
    }

    /// Train on a single batch of sequences
    /// Returns (batch_loss, gradient_norm)
    fn train_batch(&mut self, batch: &[Vec<usize>], lr: f32, training_progress: f32) -> Result<(f32, f32)> {
        let mut batch_loss = 0.0;
        let mut accumulated_param_grads: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut layer_grad_norms: Vec<f32> = Vec::new(); // Track per-layer gradient norms

        // Initialize accumulated gradients for each layer
        for _ in &self.network {
            accumulated_param_grads.push(Vec::new());
            layer_grad_norms.push(0.0);
        }

        // Process each sequence in the batch
        for training_row in batch {
            if training_row.len() < 2 {
                continue;
            }

            // 1. Slice input and targets
            let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
            let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab.

            // Forward pass with signal propagation variance tracking
            let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
            input
                .row_mut(0)
                .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

            // Track forward pass variance for signal propagation analysis
            // Reference: "Deep Information Propagation" (Schoenholz et al., 2017)
            // Ideal: Var(x_l) ≈ Var(x_0) for all layers (isometry condition)
            let mut layer_variances: Vec<f32> = Vec::new();

            for layer in &mut self.network {
                input = layer.forward(&input);

                // Compute variance of layer output
                let mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
                let variance: f32 = input.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>() / input.len() as f32;
                layer_variances.push(variance);
            }

            let logits = input;
            let probs = Self::softmax(&logits);

            // Compute cross-entropy loss
            batch_loss += Self::cross_entropy_loss_step(&probs, target_ids);

            // Compute gradients w.r.t. logits
            let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

            // Apply gradient clipping
            if let Some(ref mut clipper) = self.gradient_clipper {
                clipper.clip_gradients(&mut grads_output);
            }

            // Backward pass: compute parameter gradients for each layer
            // Note: AttentionMoE layers use backward() directly and are handled separately
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let layer_idx = self.network.len() - 1 - rev_idx;

                // AttentionMoE uses backward() directly, not compute_gradients/apply_gradients
                // We store the gradients and will call backward() on them later
                if matches!(layer, LayerEnum::AttentionMoE(_)) {
                    // Track layer-wise gradient norm for diagnostics
                    let layer_grad_norm: f32 = grads_output.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    layer_grad_norms[layer_idx] += layer_grad_norm;

                    // Store gradients for later backward() call
                    // We use a single-element vec to match the structure
                    accumulated_param_grads[layer_idx] = vec![grads_output.clone()];

                    // For now, pass gradients through unchanged
                    // (AttentionMoE will compute input_grads during backward())
                    continue;
                }

                let (input_grads, param_grads) =
                    layer.compute_gradients(&Array2::zeros((0, 0)), &grads_output);

                // Track layer-wise gradient norm for diagnostics
                let layer_grad_norm: f32 = input_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
                layer_grad_norms[layer_idx] += layer_grad_norm;

                grads_output = input_grads;

                // Accumulate parameter gradients (correct index for reversed iteration)
                if accumulated_param_grads[layer_idx].is_empty() {
                    accumulated_param_grads[layer_idx] = param_grads;
                } else {
                    for (acc_grad, new_grad) in accumulated_param_grads[layer_idx]
                        .iter_mut()
                        .zip(param_grads)
                    {
                        *acc_grad += &new_grad;
                    }
                }
            }
        }

        // Average layer-wise gradient norms
        for norm in &mut layer_grad_norms {
            *norm /= batch.len() as f32;
        }

        // Log layer-wise gradient norms for debugging (only if any exceed threshold)
        let max_layer_grad = layer_grad_norms.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_layer_grad > 10.0 {
            tracing::warn!(
                "Layer-wise gradient norms: {:?}",
                layer_grad_norms.iter()
                    .enumerate()
                    .map(|(i, &norm)| format!("L{}: {:.2}", i, norm))
                    .collect::<Vec<_>>()
            );
        }

        // Add auxiliary losses from MoH routing (load balance + dynamic loss)
        // These are computed once per batch, not per sequence
        for layer in &self.network {
            if let Some(attn_layer) = layer.as_self_attention() {
                // Load balance loss (weighted by load_balance_weight)
                let lb_loss = attn_layer.get_load_balance_loss();
                batch_loss += lb_loss; // Weight is already applied in the loss computation

                // Dynamic loss (weighted by adaptive dynamic_loss_weight)
                let dyn_loss = attn_layer.get_dynamic_loss();
                let dyn_weight = attn_layer.get_dynamic_loss_weight(training_progress);
                batch_loss += dyn_loss * dyn_weight;
            } else if let Some(trm_block) = layer.as_trm_block() {
                // For TRM, get auxiliary losses from internal attention layer
                let lb_loss = trm_block.get_load_balance_loss();
                batch_loss += lb_loss;

                let dyn_loss = trm_block.get_dynamic_loss();
                let dyn_weight = trm_block.get_dynamic_loss_weight(training_progress);
                batch_loss += dyn_loss * dyn_weight;

                // Add ponder loss (adaptive recursive depth)
                let ponder_loss = trm_block.get_ponder_loss();
                batch_loss += ponder_loss;
            } else if let Some(moe_layer) = layer.as_attention_moe() {
                // For AttentionMoE, add auxiliary loss (includes both MoE routing and MoH routing losses)
                let aux_loss = moe_layer.get_auxiliary_loss();
                batch_loss += aux_loss;
            }
        }

        // Prepare averaged gradients and detect anomalies
        let mut averaged_grads_per_layer: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut total_grad_norm_sq = 0.0f32;

        for (layer_idx, param_grads) in accumulated_param_grads.into_iter().enumerate() {
            if !param_grads.is_empty() {
                let averaged_grads: Vec<Array2<f32>> = param_grads
                    .into_iter()
                    .map(|grad| grad / batch.len() as f32)
                    .collect();

                // Detect gradient anomalies (poisoning/training instability)
                if let Err(e) = self.detect_gradient_anomalies(&averaged_grads) {
                    tracing::error!(
                        layer_idx = layer_idx,
                        layer_type = self.network[layer_idx].layer_type(),
                        "Gradient anomaly detected in layer"
                    );
                    return Err(e);
                }

                // Compute L2 norm of gradients for this layer
                for grad in &averaged_grads {
                    total_grad_norm_sq += grad.iter().map(|&x| x * x).sum::<f32>();
                }

                averaged_grads_per_layer.push(averaged_grads);
            } else {
                averaged_grads_per_layer.push(Vec::new());
            }
        }

        // Compute global gradient norm (L2 norm across all parameters)
        let mut grad_norm = total_grad_norm_sq.sqrt();

        // Apply global gradient clipping on parameter gradients
        const GRAD_CLIP_THRESHOLD: f32 = 100.0;
        if grad_norm > GRAD_CLIP_THRESHOLD {
            let scale = GRAD_CLIP_THRESHOLD / grad_norm;
            for grads in &mut averaged_grads_per_layer {
                for grad in grads {
                    grad.mapv_inplace(|x| x * scale);
                }
            }
            // Recompute grad_norm after clipping
            let mut clipped_total_grad_norm_sq = 0.0f32;
            for grads in &averaged_grads_per_layer {
                for grad in grads {
                    clipped_total_grad_norm_sq += grad.iter().map(|&x| x * x).sum::<f32>();
                }
            }
            grad_norm = clipped_total_grad_norm_sq.sqrt();
        }

        // Apply accumulated and averaged gradients with layer-wise adaptive learning rates
        // Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
        // Formula: lr_layer = lr_base * trust_coef * ||W|| / (||∇W|| + weight_decay * ||W|| + ε)
        // This balances gradient flow across layers of different depths

        // Compute adaptive learning rates for all layers first (to avoid borrow checker issues)
        let adaptive_lrs: Vec<f32> = self.network.iter()
            .zip(&averaged_grads_per_layer)
            .enumerate()
            .map(|(layer_idx, (layer, grads))| {
                if grads.is_empty() {
                    lr
                } else {
                    Self::compute_layer_adaptive_lr_static(layer, grads, lr, layer_idx)
                }
            })
            .collect();

        // Apply gradients with computed adaptive learning rates
        for ((layer, averaged_grads), adaptive_lr) in self.network.iter_mut()
            .zip(averaged_grads_per_layer)
            .zip(adaptive_lrs)
        {
            if !averaged_grads.is_empty() {
                // AttentionMoE uses backward() directly instead of apply_gradients()
                if let LayerEnum::AttentionMoE(moe_layer) = layer {
                    // For AttentionMoE, averaged_grads contains the output gradients
                    // Call backward() which updates parameters directly
                    if !averaged_grads.is_empty() {
                        let _input_grads = moe_layer.backward(&averaged_grads[0], adaptive_lr);
                    }
                } else {
                    layer.apply_gradients(&averaged_grads, adaptive_lr)?;
                }
            }
        }

        // Update threshold predictors in MoH layers (if enabled)
        // This happens after parameter updates to use the latest routing statistics
        for layer in &mut self.network {
            if let Some(attn_layer) = layer.as_self_attention_mut() {
                if attn_layer.has_learned_predictor() {
                    // Use a separate learning rate for predictor (slightly higher than base LR)
                    // Predictor needs faster adaptation to track routing efficiency, but not too aggressive
                    // 2x base LR provides faster learning while maintaining stability
                    let predictor_lr = lr * 2.0;
                    attn_layer.update_threshold_predictor(predictor_lr);
                }
            }
        }

        Ok((batch_loss, grad_norm))
    }
    /// Compute layer-wise adaptive learning rate using bidirectional LARS
    /// Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
    ///
    /// Bidirectional approach: Balance gradient flow across all layers
    /// - High-gradient layers (L0-L2): Reduce LR to prevent over-updating
    /// - Low-gradient layers (L6-L14): Increase LR to prevent under-updating
    /// - Target: All layers converge at similar rates
    ///
    /// Formula: lr_layer = lr_base * (target_norm / (grad_norm + ε))^power
    /// where power controls aggressiveness (0.5 = gentle, 1.0 = aggressive)
    fn compute_layer_adaptive_lr_static(
        layer: &LayerEnum,
        grads: &[Array2<f32>],
        base_lr: f32,
        layer_idx: usize,
    ) -> f32 {
        // Skip for layers without gradients
        if grads.is_empty() {
            return base_lr;
        }

        // Compute gradient norm ||∇W||
        let grad_norm: f32 = grads.iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        // Avoid division by zero
        const EPSILON: f32 = 1e-6;
        if grad_norm < EPSILON {
            return base_lr;
        }

        // Bidirectional LARS: Target gradient norm for balanced learning
        // Target chosen based on observed mid-layer gradients (L3-L5: ~2-4)
        const TARGET_GRAD_NORM: f32 = 3.0;
        const POWER: f32 = 0.5; // Gentle adaptation (sqrt scaling)

        // Compute scaling factor
        let scale = (TARGET_GRAD_NORM / (grad_norm + EPSILON)).powf(POWER);

        // Clamp to reasonable range to prevent extreme adjustments
        // 0.3x-3.0x range allows significant adaptation while maintaining stability
        let scale_clamped = scale.clamp(0.3, 3.0);
        let adaptive_lr = base_lr * scale_clamped;

        // Log adaptive LR for debugging (use RUST_LOG=debug to see)
        if layer_idx <= 2 || layer_idx >= 12 {
            tracing::debug!(
                layer_idx = layer_idx,
                layer_type = layer.layer_type(),
                grad_norm = grad_norm,
                base_lr = base_lr,
                adaptive_lr = adaptive_lr,
                scale = scale_clamped,
                "Bidirectional LARS"
            );
        }

        adaptive_lr
    }

    /// Configure gradient clipping strategy
    pub fn set_gradient_clipping(&mut self, clipper: Box<dyn GradientClipping>) {
        self.gradient_clipper = Some(clipper);
    }

    /// Disable gradient clipping
    pub fn disable_gradient_clipping(&mut self) {
        self.gradient_clipper = None;
    }

    /// Detect gradient anomalies that may indicate training instability or poisoning
    fn detect_gradient_anomalies(&self, grads: &[Array2<f32>]) -> Result<()> {
        for (i, grad) in grads.iter().enumerate() {
            let max_grad = grad.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            if max_grad > crate::GRADIENT_ANOMALY_THRESHOLD {
                tracing::warn!(
                    "Gradient anomaly detected in layer {}: max gradient magnitude {}",
                    i,
                    max_grad
                );
                return Err(ModelError::GradientError {
                    message: format!(
                        "Gradient anomaly in layer {}: magnitude {} exceeds threshold {}",
                        i,
                        max_grad,
                        crate::GRADIENT_ANOMALY_THRESHOLD
                    ),
                });
            }

            // Check for NaN/Inf values
            if grad.iter().any(|&x| !x.is_finite()) {
                tracing::error!("Non-finite gradients detected in layer {}", i);
                return Err(ModelError::GradientError {
                    message: format!("Non-finite gradients detected in layer {}", i),
                });
            }
        }
        Ok(())
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Input validation
        let safe_text = if text.len() > crate::MAX_INPUT_LENGTH {
            tracing::warn!(
                "Input text length {} exceeds maximum allowed length {}, truncating",
                text.len(),
                crate::MAX_INPUT_LENGTH
            );
            &text[..crate::MAX_INPUT_LENGTH.min(text.len())]
        } else {
            text
        };

        // Split by whitespace first
        let mut tokens = Vec::new();

        for word in safe_text.split_whitespace() {
            // Special case for end token
            if word == "</s>" {
                if let Some(token_id) = self.vocab.encode(word) {
                    tokens.push(token_id);
                }
                continue;
            }

            let mut current_word = String::new();

            for c in word.chars() {
                if c.is_ascii_punctuation() {
                    // If we have a word before the punctuation, add it
                    if !current_word.is_empty() {
                        if let Some(token_id) = self.vocab.encode(&current_word) {
                            tokens.push(token_id);
                        }
                        current_word.clear();
                    }

                    // Add the punctuation as its own token
                    if let Some(token_id) = self.vocab.encode(&c.to_string()) {
                        tokens.push(token_id);
                    }
                } else {
                    current_word.push(c);
                }
            }

            // Add any remaining word
            if !current_word.is_empty()
                && let Some(token_id) = self.vocab.encode(&current_word)
            {
                tokens.push(token_id);
            }
        }

        tokens
    }

    fn softmax(logits: &Array2<f32>) -> Array2<f32> {
        // logits is seq_len x vocab_size
        let mut result = logits.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            // Calculate exp for each element
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .to_vec()
    }

    fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        let mut loss = 0.0;
        for row_idx in 0..probs.shape()[0] {
            let prob_target = probs[[row_idx, target[row_idx]]]; // Get probability of correct token
            loss -= prob_target.max(1e-15).ln(); // Add numerical stability
        }

        loss / target.len() as f32
    }

    fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone(); // Start with softmax probabilities

        // Defensive check: if shapes mismatch, log warning and return zero gradients
        if probs.shape()[0] != target.len() {
            tracing::error!(
                probs_rows = probs.shape()[0],
                target_len = target.len(),
                "Shape mismatch in gradient computation, returning zero gradients"
            );
            return Array2::zeros(probs.raw_dim());
        }

        let batch_size = target.len() as f32;

        // Compute correct softmax + cross-entropy gradient: softmax - one_hot(target)
        for row_idx in 0..grads.shape()[0] {
            grads[[row_idx, target[row_idx]]] -= 1.0; // Convert to: p - y (where y is one-hot)
        }

        // Normalize by batch size for stable training
        grads.mapv_inplace(|x| x / batch_size);

        grads
    }

    /// Generate text using beam search
    ///
    /// This method uses beam search to generate text, which explores multiple
    /// hypotheses in parallel and can produce higher quality output than greedy decoding.
    ///
    /// # Arguments
    /// * `text` - Input text to condition generation on
    /// * `config` - Beam search configuration
    ///
    /// # Returns
    /// Generated text as a String
    pub fn generate_with_beam_search(
        &mut self,
        text: &str,
        config: &crate::beam_search::BeamSearchConfig,
    ) -> String {
        use crate::beam_search::BeamSearchState;

        // Tokenize the input text
        let initial_tokens = self.tokenize(text);

        // Safety check: ensure we have at least one token
        if initial_tokens.is_empty() {
            return String::new();
        }

        // Initialize beam search state
        let mut state = BeamSearchState::new(initial_tokens.clone(), config.beam_width);

        let end_token = self.vocab.encode("</s>").unwrap();

        // Generate tokens
        for step in 0..config.max_length {
            if state.is_done() {
                break;
            }

            // Collect predictions for all active beams
            let active_beams: Vec<&BeamHypothesis> =
                state.beams.iter().filter(|b| !b.is_complete).collect();
            let num_active = active_beams.len();
            if num_active == 0 {
                break;
            }

            let seq_len = active_beams[0].tokens.len();
            let mut token_matrix = Array2::<f32>::zeros((num_active, seq_len));

            for (i, beam) in active_beams.iter().enumerate() {
                for (j, &token) in beam.tokens.iter().enumerate() {
                    token_matrix[[i, j]] = token as f32;
                }
            }

            let mut input = token_matrix;

            // Forward pass through network (batched)
            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input; // shape (num_active, vocab_size)

            // Extract predictions for each beam
            let predictions: Vec<Array1<f32>> =
                (0..num_active).map(|i| logits.row(i).to_owned()).collect();

            // Compute entropy for adaptive beam width
            if config.use_adaptive_beam && !predictions.is_empty() {
                let entropy = state.compute_entropy(&predictions);
                state.adapt_beam_width(entropy, config);
            }

            // Expand beams with new predictions
            state.expand(&predictions, config, self.vocab.words.len());

            // Mark complete beams
            state.mark_complete(end_token, initial_tokens.len() + step + 1);
        }

        // Get the best hypothesis
        let best = state.get_best(config);

        if let Some(hypothesis) = best {
            // Skip the initial tokens (input) and convert to text
            let output_tokens = &hypothesis.tokens[initial_tokens.len()..];

            if output_tokens.is_empty() {
                return String::new();
            }

            // Convert token_ids to strings
            let token_strs = output_tokens
                .iter()
                .map(|&t| self.vocab.decode.get(&t).unwrap().as_str())
                .collect::<Vec<&str>>();

            token_strs.join(" ")
        } else {
            String::new()
        }
    }

    /// Save model to JSON format (human-readable, larger file size)
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        fs::write(path, json).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load model from JSON format
    pub fn load_json(path: &str) -> Result<Self> {
        let data = fs::read_to_string(path).map_err(ModelError::from)?;
        let llm: LLM = serde_json::from_str(&data).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        Ok(llm)
    }

    /// Save model to binary format (compact, faster, smaller file size)
    pub fn save_binary(&self, path: &str) -> Result<()> {
        let config = bincode::config::standard();
        let encoded =
            bincode::serde::encode_to_vec(self, config).map_err(|e| ModelError::Serialization {
                source: Box::new(e),
            })?;
        fs::write(path, encoded).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load model from binary format
    pub fn load_binary(path: &str) -> Result<Self> {
        let data = fs::read(path).map_err(ModelError::from)?;
        let config = bincode::config::standard();
        let (llm, _): (LLM, usize) =
            bincode::serde::decode_from_slice(&data, config).map_err(|e| {
                ModelError::Serialization {
                    source: Box::new(e),
                }
            })?;
        Ok(llm)
    }

    /// Save model (auto-detects format from extension: .json or .bin)
    pub fn save(&self, path: &str) -> Result<()> {
        if path.ends_with(".json") {
            self.save_json(path)
        } else {
            self.save_binary(path)
        }
    }

    /// Load model (auto-detects format from extension: .json or .bin)
    pub fn load(path: &str) -> Result<Self> {
        if path.ends_with(".json") {
            Self::load_json(path)
        } else {
            Self::load_binary(path)
        }
    }
}

pub use crate::beam_search::*;
