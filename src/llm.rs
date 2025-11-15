use std::fs;

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Distribution;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument, warn};

use crate::{
    MAX_SEQ_LEN, Vocab,
    decoding::GreedyDecoder,
    embeddings::TokenEmbeddings,
    errors::{ModelError, Result},
    metrics::text::corpus_bleu_1_2,
    output_projection::OutputProjection,
    transformer::TransformerBlock,
    trm::TRM,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum LayerEnum {
    TokenEmbeddings(TokenEmbeddings),
    // Removed SelfAttention variant
    // Removed FeedForward variant; RichardsGlu is the only FFN
    RichardsGlu(Box<crate::richards::RichardsGlu>),
    MixtureOfExperts(Box<crate::mixtures::moe::MixtureOfExperts>),

    DynamicTanhNorm(crate::richards::RichardsNorm),
    OutputProjection(OutputProjection),

    // Removed TRMBlock variant
    PolyAttention(Box<crate::attention::poly_attention::PolyAttention>),
    TransformerBlock(Box<TransformerBlock>),
    DiffusionBlock(Box<crate::transformer::diffusion_block::DiffusionBlock>),
    TRM(Box<TRM>),
}

impl LayerEnum {
    // Removed downcast helpers for SelfAttention/TRM to simplify to PolyAttention-only
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.layer_type(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.layer_type(),
            LayerEnum::MixtureOfExperts(layer) => layer.layer_type(),

            LayerEnum::DynamicTanhNorm(layer) => layer.layer_type(),
            LayerEnum::OutputProjection(layer) => layer.layer_type(),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.layer_type(),
            LayerEnum::TransformerBlock(layer) => layer.layer_type(),
            LayerEnum::DiffusionBlock(layer) => layer.layer_type(),
            LayerEnum::TRM(layer) => layer.layer_type(),
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.forward(input),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.forward(input),
            LayerEnum::MixtureOfExperts(layer) => layer.forward(input),

            LayerEnum::DynamicTanhNorm(layer) => layer.forward(input),
            LayerEnum::OutputProjection(layer) => layer.forward(input),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.forward(input),
            LayerEnum::TransformerBlock(layer) => layer.forward(input),
            LayerEnum::DiffusionBlock(layer) => layer.forward(input),
            LayerEnum::TRM(layer) => layer.forward(input),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.compute_gradients(input, output_grads),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::MixtureOfExperts(layer) => layer.compute_gradients(input, output_grads),

            LayerEnum::DynamicTanhNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::OutputProjection(layer) => layer.compute_gradients(input, output_grads),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::TransformerBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::DiffusionBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::TRM(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.apply_gradients(param_grads, lr),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::MixtureOfExperts(layer) => layer.apply_gradients(param_grads, lr),

            LayerEnum::DynamicTanhNorm(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::OutputProjection(layer) => layer.apply_gradients(param_grads, lr),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::TransformerBlock(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::DiffusionBlock(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::TRM(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.backward(grads, lr),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.backward(grads, lr),
            LayerEnum::MixtureOfExperts(layer) => layer.backward(grads, lr),

            LayerEnum::DynamicTanhNorm(layer) => layer.backward(grads, lr),
            LayerEnum::OutputProjection(layer) => layer.backward(grads, lr),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.backward(grads, lr),
            LayerEnum::TransformerBlock(layer) => layer.backward(grads, lr),
            LayerEnum::DiffusionBlock(layer) => layer.backward(grads, lr),
            LayerEnum::TRM(layer) => layer.backward(grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.parameters(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.parameters(),
            LayerEnum::MixtureOfExperts(layer) => layer.parameters(),

            LayerEnum::DynamicTanhNorm(layer) => layer.parameters(),
            LayerEnum::OutputProjection(layer) => layer.parameters(),
            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.parameters(),
            LayerEnum::TransformerBlock(layer) => layer.parameters(),
            LayerEnum::DiffusionBlock(layer) => layer.parameters(),
            LayerEnum::TRM(layer) => layer.parameters(),
        }
    }

    fn weight_norm(&self) -> f32 {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.weight_norm(),
            LayerEnum::RichardsGlu(layer) => layer.weight_norm(),
            LayerEnum::MixtureOfExperts(layer) => layer.weight_norm(),
            LayerEnum::DynamicTanhNorm(layer) => layer.weight_norm(),
            LayerEnum::OutputProjection(layer) => layer.weight_norm(),
            LayerEnum::PolyAttention(layer) => layer.weight_norm(),
            LayerEnum::TransformerBlock(layer) => layer.weight_norm(),
            LayerEnum::DiffusionBlock(layer) => layer.weight_norm(),
            LayerEnum::TRM(layer) => layer.weight_norm(),
        }
    }
}

fn response_span_from_tokens(vocab: &Vocab, tokens: &[usize]) -> Option<(usize, usize)> {
    if tokens.is_empty() {
        return None;
    }
    let mut seen_user_tag = false;
    for (idx, &tid) in tokens.iter().enumerate() {
        let Some(text) = vocab.decode(tid) else {
            continue;
        };
        if text.eq_ignore_ascii_case("user") {
            seen_user_tag = true;
            continue;
        }
        if !seen_user_tag {
            continue;
        }
        if text.eq_ignore_ascii_case("assistant") {
            let colon_after = tokens
                .get(idx + 1)
                .and_then(|&next_id| vocab.decode(next_id))
                .map(|tok| tok == ":")
                .unwrap_or(false);
            if !colon_after {
                continue;
            }
            let mut start = idx + 2; // skip "Assistant" and following ':'
            if start >= tokens.len() {
                return None;
            }
            let mut end = tokens.len();
            if let Some(last_tok) = tokens.last().and_then(|&id| vocab.decode(id)) {
                if last_tok == "</s>" && end > start {
                    end -= 1;
                }
            }
            if start >= end {
                return None;
            }
            return Some((start, end));
        }
    }
    None
}

pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    /// Frobenius norm of all learnable weights in the layer
    /// Used by LARS trust-ratio to balance update magnitude
    fn weight_norm(&self) -> f32;

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>);

    /// Apply gradients to layer parameters
    /// Returns GradientError if param_grads has incorrect length
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()>;
}

#[derive(Serialize, Deserialize, Debug)]
pub enum DecoderType {
    Greedy(GreedyDecoder),
}

impl DecoderType {
    pub fn layer_type(&self) -> &str {
        match self {
            DecoderType::Greedy(_) => "GreedyDecoder",
        }
    }

    pub fn parameters(&self) -> usize {
        match self {
            DecoderType::Greedy(_) => 0, // Greedy has no parameters
        }
    }
}

#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,
    decoder: DecoderType,
    // EMA of median per-layer gradient norm to stabilize adaptive LR balance
    median_grad_ema: Option<f32>,
}

impl std::fmt::Debug for LLM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLM")
            .field("vocab", &self.vocab)
            .field("network", &self.network)
            .finish()
    }
}

impl Default for LLM {
    fn default() -> Self {
        use crate::{model_builder::build_network, model_config::ModelConfig};

        let config = ModelConfig::default();
        let vocab = Vocab::default();
        let network = build_network(&config, &vocab);

        let decoder = DecoderType::Greedy(GreedyDecoder::new());

        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());

        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
        }
    }

    /// Create LLM with GreedyDecoder
    pub fn with_greedy_decoder(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());
        Self {
            vocab,
            network,
            decoder,
            median_grad_ema: None,
        }
    }

    /// Switch to GreedyDecoder
    pub fn enable_greedy(&mut self) {
        let decoder = DecoderType::Greedy(GreedyDecoder::new());
        self.decoder = decoder;
    }
}

impl LLM {
    pub fn network_description(&self) -> String {
        let network_layers = self.network.iter().map(|layer| layer.layer_type()).fold(
            String::new(),
            |mut acc, layer_type| {
                if !acc.is_empty() {
                    acc.push_str(", ");
                }
                acc.push_str(layer_type);
                acc
            },
        );

        // Include decoder type in the description
        format!("{}, {}", network_layers, self.decoder.layer_type())
    }

    pub fn total_parameters(&self) -> usize {
        // Sum the parameters across all layers in the network
        let network_params = self
            .network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>();

        // Add decoder parameters
        network_params + self.decoder.parameters()
    }

    /// Set TRM layers to inference mode for faster prediction
    pub fn set_trm_inference_mode(&mut self) {
        for layer in &mut self.network {
            if let LayerEnum::TRM(trm) = layer {
                trm.set_training_mode(false);
            }
        }
    }

    /// Set TRM layers to training mode for full supervision steps
    pub fn set_trm_training_mode(&mut self) {
        for layer in &mut self.network {
            if let LayerEnum::TRM(trm) = layer {
                trm.set_training_mode(true);
            }
        }
    }

    #[inline]
    pub fn predict(&mut self, text: &str) -> String {
        let output_tokens = self.forward(text);

        // Handle empty output
        if output_tokens.is_empty() {
            return String::new();
        }

        // Convert token_ids to strings
        output_tokens
            .iter()
            .map(|&t| self.vocab.decode(t).unwrap())
            .fold(String::new(), |mut acc, token_str| {
                if !acc.is_empty() {
                    acc.push(' ');
                }
                acc.push_str(token_str);
                acc
            })
    }

    #[inline]
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

            let mut token_input = Array2::zeros((1, tokenized.len()));
            for (i, &token_id) in tokenized.iter().enumerate() {
                token_input[[0, i]] = token_id as f32;
            }
            let mut input = token_input;

            // Forward pass through all layers except output projection to get hidden states
            let network_len = self.network.len();
            let mut hidden_states = input.clone();
            let mut logits = Array2::zeros((1, self.vocab.size()));

            for (i, layer) in self.network.iter_mut().enumerate() {
                input = layer.forward(&input);

                // Capture hidden states before output projection (second-to-last layer)
                if i == network_len - 2 {
                    hidden_states = input.clone();
                }

                // Get logits from output projection (last layer)
                if i == network_len - 1 {
                    logits = input.clone();
                }
            }

            // Safety check: ensure we have at least one token
            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            // Get hidden states for the last position
            let _last_hidden = hidden_states.row(hidden_states.shape()[0] - 1).to_owned();

            let next_token = match &mut self.decoder {
                DecoderType::Greedy(decoder) => {
                    // Simple greedy decoding
                    let probs =
                        crate::softmax::Softmax::new().forward_immutable(&last_logit.view());
                    let tokens = decoder.decode(&probs);
                    tokens[0]
                }
            };

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
        // Set TRM layers to training mode (full supervision steps)
        self.set_trm_training_mode();

        let tokenized_data = data
            .par_iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        // Store previous richards_glu richards weights for delta tracking
        let mut prev_richards_glu_weights: Vec<Vec<f64>> = Vec::new();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut batch_count = 0;
            let mut total_examples = 0usize;
            let mut per_layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

            // Learning rate warmup + cosine annealing
            // Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov &
            // Hutter, 2016)
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
            let _training_progress = if epoch < warmup_epochs {
                0.0
            } else {
                (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32
            };

            // Process data in batches
            for batch in tokenized_data.chunks(batch_size) {
                let (batch_loss, grad_norm, layer_param_grad_norm_sq) =
                    self.train_batch_profiled(batch, effective_lr)?;
                total_loss += batch_loss;
                total_grad_norm += grad_norm;
                batch_count += 1;
                total_examples += batch.len();
                for (i, s) in layer_param_grad_norm_sq.into_iter().enumerate() {
                    per_layer_param_grad_norm_sq[i] += s;
                }
            }

            let avg_loss = total_loss / batch_count as f32;
            let avg_grad_norm = total_grad_norm / batch_count as f32;
            let per_layer_rms: Vec<f32> = per_layer_param_grad_norm_sq
                .iter()
                .map(|&s| (s / (batch_count as f32).max(1.0)).sqrt())
                .collect();

            // Normalize by parameter count so layers with fewer parameters (e.g., RichardsNorm)
            // are not misinterpreted as "dead" purely due to scale differences.
            let layer_param_counts: Vec<usize> = self
                .network
                .iter()
                .map(|layer| layer.parameters().max(1))
                .collect();
            let per_layer_rms_per_param: Vec<f32> = per_layer_rms
                .iter()
                .enumerate()
                .map(|(i, &raw)| {
                    let param_count = layer_param_counts.get(i).copied().unwrap_or(1) as f32;
                    if param_count > 0.0 {
                        raw / param_count.sqrt()
                    } else {
                        raw
                    }
                })
                .collect();

            tracing::info!(
                epoch = epoch,
                per_layer_rms = ?per_layer_rms,
                per_layer_rms_per_param = ?per_layer_rms_per_param,
                layer_param_counts = ?layer_param_counts,
                "Transformer epoch layer param grad RMS"
            );
            let names: Vec<&str> = self.network.iter().map(|l| l.layer_type()).collect();
            tracing::debug!(epoch = epoch, per_layer = ?names, per_layer_rms = ?per_layer_rms, "Layer RMS breakdown");

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

            // Aggregate MoH instrumentation from PolyAttention layers at epoch end
            let mut tau_min_epoch = f32::INFINITY;
            let mut tau_max_epoch = f32::NEG_INFINITY;
            let mut tau_available = false;
            let mut pred_norm_sum = 0.0f32;
            let mut pred_norm_count = 0usize;
            let mut avg_heads_per_token_sum = 0.0f32;
            let mut heads_layers_count = 0usize;
            let mut avg_experts_sum = 0.0f32;
            let mut significant_experts_sum = 0.0f32;
            let mut routing_entropy_sum = 0.0f32;
            let mut experts_layers_count = 0usize;

            for layer in &mut self.network {
                if let LayerEnum::PolyAttention(pa) = layer {
                    if let Some((min_tau, max_tau)) = pa.take_tau_metrics() {
                        tau_available = true;
                        if min_tau < tau_min_epoch {
                            tau_min_epoch = min_tau;
                        }
                        if max_tau > tau_max_epoch {
                            tau_max_epoch = max_tau;
                        }
                    }
                    if let Some(rms_g) = pa.take_pred_norm() {
                        pred_norm_sum += rms_g;
                        pred_norm_count += 1;
                    }
                    let per_head = pa.get_head_metrics_and_reset();
                    if !per_head.is_empty() {
                        let layer_avg_active_heads =
                            per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                        avg_heads_per_token_sum += layer_avg_active_heads;
                        heads_layers_count += 1;
                    }
                }
                if let LayerEnum::TransformerBlock(block) = layer {
                    // Pull through attention metrics from within the block
                    if let Some((min_tau, max_tau)) = block.attention.take_tau_metrics() {
                        tau_available = true;
                        if min_tau < tau_min_epoch {
                            tau_min_epoch = min_tau;
                        }
                        if max_tau > tau_max_epoch {
                            tau_max_epoch = max_tau;
                        }
                    }
                    if let Some(rms_g) = block.attention.take_pred_norm() {
                        pred_norm_sum += rms_g;
                        pred_norm_count += 1;
                    }
                    let per_head = block.attention.get_head_metrics_and_reset();
                    if !per_head.is_empty() {
                        let layer_avg_active_heads =
                            per_head.iter().map(|(avg, _tokens)| avg).sum::<f32>();
                        avg_heads_per_token_sum += layer_avg_active_heads;
                        heads_layers_count += 1;
                    }
                }
                if let LayerEnum::MixtureOfExperts(moe) = layer {
                    let layer_avg_active_experts = moe.config.get_avg_active_experts();
                    let layer_significant_experts = moe.config.get_avg_significant_experts();
                    let layer_routing_entropy = moe.config.get_routing_entropy();
                    avg_experts_sum += layer_avg_active_experts;
                    significant_experts_sum += layer_significant_experts;
                    routing_entropy_sum += layer_routing_entropy;
                    experts_layers_count += 1;
                }
            }

            let tau_min_log = if tau_available {
                tau_min_epoch
            } else {
                f32::NAN
            };
            let tau_max_log = if tau_available {
                tau_max_epoch
            } else {
                f32::NAN
            };
            let tau_range_log = if tau_available {
                tau_max_epoch - tau_min_epoch
            } else {
                f32::NAN
            };
            let pred_norm_rms = if pred_norm_count > 0 {
                pred_norm_sum / pred_norm_count as f32
            } else {
                f32::NAN
            };
            let avg_active_heads = if heads_layers_count > 0 {
                avg_heads_per_token_sum / heads_layers_count as f32
            } else {
                f32::NAN
            };
            let avg_active_experts = if experts_layers_count > 0 {
                avg_experts_sum / experts_layers_count as f32
            } else {
                f32::NAN
            };
            let avg_significant_experts = if experts_layers_count > 0 {
                significant_experts_sum / experts_layers_count as f32
            } else {
                f32::NAN
            };
            let avg_routing_entropy = if experts_layers_count > 0 {
                routing_entropy_sum / experts_layers_count as f32
            } else {
                f32::NAN
            };

            // Collect current richards_glu richards weights for delta tracking
            let mut current_richards_glu_weights: Vec<Vec<f64>> = Vec::new();
            let mut richards_training_status: Vec<bool> = Vec::new();
            for layer in &self.network {
                if let LayerEnum::RichardsGlu(richards_glu) = layer {
                    current_richards_glu_weights.push(richards_glu.gate_curve.weights());
                    richards_training_status.push(richards_glu.gate_curve.has_trained_parameters());
                }
            }

            // Debug: Check if Richards parameters are being trained
            let trained_layers = richards_training_status
                .iter()
                .filter(|&&trained| trained)
                .count();
            if !current_richards_glu_weights.is_empty() {
                tracing::debug!(
                    "RichardsGlu training status: {}/{} layers have trained parameters",
                    trained_layers,
                    current_richards_glu_weights.len()
                );
            }

            // Compute delta changes in richards_glu richards coefficients
            let mut richards_glu_delta_sum = 0.0;
            let mut richards_glu_param_count = 0;
            let mut total_weight_changes = 0;
            let mut significant_changes = 0;

            if !prev_richards_glu_weights.is_empty()
                && current_richards_glu_weights.len() == prev_richards_glu_weights.len()
            {
                for (layer_idx, (prev_layer, curr_layer)) in prev_richards_glu_weights
                    .iter()
                    .zip(current_richards_glu_weights.iter())
                    .enumerate()
                {
                    if prev_layer.len() == curr_layer.len() {
                        for (param_idx, (prev_w, curr_w)) in
                            prev_layer.iter().zip(curr_layer.iter()).enumerate()
                        {
                            let delta = (curr_w - prev_w).abs();
                            richards_glu_delta_sum += delta;
                            richards_glu_param_count += 1;
                            total_weight_changes += 1;

                            // Count significant changes (> 1e-6 relative change)
                            if delta > 1e-6 {
                                significant_changes += 1;
                            }

                            // Debug: Log unusual parameter values
                            if delta > 1.0 {
                                tracing::debug!(
                                    "Large Richards parameter change in layer {} param {}: {:.6} -> {:.6} (delta: {:.6})",
                                    layer_idx,
                                    param_idx,
                                    prev_w,
                                    curr_w,
                                    delta
                                );
                            }
                        }
                    } else {
                        tracing::warn!(
                            "RichardsGlu layer {} weight length mismatch: prev={}, curr={}",
                            layer_idx,
                            prev_layer.len(),
                            curr_layer.len()
                        );
                    }
                }
            } else {
                if prev_richards_glu_weights.is_empty() {
                    tracing::debug!("No previous RichardsGlu weights available (first epoch)");
                } else {
                    tracing::warn!(
                        "RichardsGlu layer count mismatch: prev={}, curr={}",
                        prev_richards_glu_weights.len(),
                        current_richards_glu_weights.len()
                    );
                }
            }

            // Debug: Log parameter change statistics
            if richards_glu_param_count > 0 {
                let avg_delta = richards_glu_delta_sum / richards_glu_param_count as f64;
                let significant_ratio = significant_changes as f64 / total_weight_changes as f64;

                tracing::debug!(
                    "RichardsGlu delta stats: {} params, avg_delta={:.2e}, significant_changes={}/{} ({:.1}%)",
                    richards_glu_param_count,
                    avg_delta,
                    significant_changes,
                    total_weight_changes,
                    significant_ratio * 100.0
                );
            }
            let avg_richards_glu_delta = if richards_glu_param_count > 0 {
                richards_glu_delta_sum / richards_glu_param_count as f64
            } else {
                0.0
            };

            // Update previous weights
            prev_richards_glu_weights = current_richards_glu_weights;

            // NFR-7.3: Training metrics
            let warmup_status = if epoch < warmup_epochs {
                format!(" (warmup {}/{})", epoch + 1, warmup_epochs)
            } else {
                String::new()
            };

            info!(
                epoch = epoch,
                loss = avg_loss,
                grad_norm = avg_grad_norm,
                learning_rate = effective_lr,
                tau_min = tau_min_log,
                tau_max = tau_max_log,
                tau_range = tau_range_log,
                pred_norm_rms = pred_norm_rms,
                avg_active_heads = avg_active_heads,
                avg_active_experts = avg_active_experts,
                avg_significant_experts = avg_significant_experts,
                avg_routing_entropy = avg_routing_entropy,
                richards_glu_richards_delta = avg_richards_glu_delta,
                "Training epoch completed{}",
                warmup_status
            );
        }

        Ok(())
    }

    /// Train TRM layers using autoencoding (pretraining phase)
    /// During autoencoding, the model learns to reconstruct its input through recursive processing
    /// This is the first phase of TRM training before chat-tuning
    #[instrument(skip(self, data))]
    pub fn train_trm_autoencoding(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        // Set TRM layers to training mode (full supervision steps)
        self.set_trm_training_mode();

        let tokenized_data = data
            .par_iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        info!(
            "Starting TRM autoencoding pretraining: {} epochs, {} sequences",
            epochs,
            tokenized_data.len()
        );

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut batch_count = 0;

            // Process data in batches
            for batch in tokenized_data.chunks(batch_size) {
                let (batch_loss, grad_norm) = self.train_batch_trm_autoencoding(batch, lr)?;
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
                        "TRM autoencoding diverged at epoch {}: loss is {} (NaN or Inf detected)",
                        epoch, avg_loss
                    ),
                });
            }

            info!(
                epoch = epoch,
                loss = avg_loss,
                grad_norm = avg_grad_norm,
                "TRM autoencoding epoch completed"
            );
        }

        Ok(())
    }

    /// Complete TRM training pipeline: autoencoding pretraining + chat-tuning
    /// Phase 1: Autoencoding - TRM learns to reconstruct input through recursion
    /// Phase 2: Chat-tuning - Standard next-token prediction on conversational data
    #[instrument(skip(self, pretraining_data, chat_data))]
    pub fn train_trm_complete(
        &mut self,
        pretraining_data: Vec<&str>,
        chat_data: Vec<&str>,
        autoencoding_epochs: usize,
        chat_epochs: usize,
        lr: f32,
        batch_size: usize,
    ) -> Result<()> {
        info!(
            "Starting TRM complete training: {} autoencoding epochs + {} chat-tuning epochs",
            autoencoding_epochs, chat_epochs
        );

        // Phase 1: Autoencoding pretraining
        if autoencoding_epochs > 0 {
            info!("Phase 1: TRM Autoencoding Pretraining");
            self.train_trm_autoencoding(pretraining_data, autoencoding_epochs, lr, batch_size)?;
        }

        // Phase 2: Chat-tuning (standard next-token prediction)
        if chat_epochs > 0 {
            info!("Phase 2: Chat-Tuning (next-token prediction)");
            self.train_with_warmup(chat_data, chat_epochs, lr, batch_size, 15)?;
        }

        info!("TRM training completed successfully");
        Ok(())
    }

    /// Train on a single batch using TRM autoencoding
    /// For autoencoding, the TRM layer learns to reconstruct its embedded input
    fn train_batch_trm_autoencoding(
        &mut self,
        batch: &[Vec<usize>],
        lr: f32,
    ) -> Result<(f32, f32)> {
        let mut batch_loss = 0.0;
        let mut accumulated_param_grads: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut layer_grad_norms: Vec<f32> = Vec::new();

        // Initialize accumulated gradients for each layer
        for _ in &self.network {
            accumulated_param_grads.push(Vec::new());
            layer_grad_norms.push(0.0);
        }

        // Process each sequence in the batch
        for sequence in batch {
            if sequence.is_empty() {
                continue;
            }

            // Convert tokens to embeddings
            let mut input: Array2<f32> = Array2::zeros((1, sequence.len()));
            for (i, &token_id) in sequence.iter().enumerate() {
                input[[0, i]] = token_id as f32;
            }

            // Forward through embedding layer
            input = self.network[0].forward(&input);

            // Forward through remaining layers
            for layer_idx in 1..self.network.len() {
                match &mut self.network[layer_idx] {
                    LayerEnum::TRM(trm) => {
                        // For TRM layers: autoencoding training
                        // The TRM should learn to reconstruct its input (pure autoencoding)
                        let trm_input = input.clone();
                        let (loss, param_grads) =
                            trm.compute_training_gradients(&trm_input, &trm_input)?;
                        batch_loss += loss;

                        // Calculate gradient norm before moving param_grads
                        layer_grad_norms[layer_idx] = param_grads
                            .iter()
                            .map(|g| g.mapv(|x| x * x).sum().sqrt())
                            .sum::<f32>();

                        // Store gradients for this TRM layer
                        accumulated_param_grads[layer_idx] = param_grads;

                        // Update input for next layer (use forward pass, not training)
                        input = trm.forward(&input);
                    }
                    _ => {
                        // For non-TRM layers: standard forward pass
                        input = self.network[layer_idx].forward(&input);
                    }
                }
            }
        }

        // Apply accumulated gradients
        for (layer_idx, param_grads) in accumulated_param_grads.into_iter().enumerate() {
            if !param_grads.is_empty() {
                self.network[layer_idx].apply_gradients(&param_grads, lr)?;
            }
        }

        let total_grad_norm = layer_grad_norms.iter().map(|&x| x * x).sum::<f32>().sqrt();
        Ok((batch_loss, total_grad_norm))
    }

    /// Train on a single batch of sequences
    /// Returns (batch_loss, gradient_norm)
    fn train_batch_profiled(
        &mut self,
        batch: &[Vec<usize>],
        lr: f32,
    ) -> Result<(f32, f32, Vec<f32>)> {
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
            for (i, &token_id) in input_ids.iter().enumerate() {
                input[[0, i]] = token_id as f32;
            }

            // Track forward pass variance for signal propagation analysis
            // Reference: "Deep Information Propagation" (Schoenholz et al., 2017)
            // Ideal: Var(x_l) ≈ Var(x_0) for all layers (isometry condition)
            let mut layer_variances: Vec<f32> = Vec::new();
            let mut layer_inputs: Vec<Array2<f32>> = Vec::with_capacity(self.network.len());

            for layer in &mut self.network {
                layer_inputs.push(input.clone());
                input = layer.forward(&input);

                // Compute variance of layer output in single pass
                let (sum, sum_sq) = input
                    .iter()
                    .fold((0.0, 0.0), |(s, sq), &x| (s + x, sq + x * x));
                let n = input.len() as f32;
                let mean = sum / n;
                let variance = (sum_sq / n) - mean * mean;
                layer_variances.push(variance);
            }

            let logits = input;
            let probs = crate::softmax::Softmax::new().forward_immutable(&logits.view());

            // Symmetric cross-entropy loss and gradients
            let sce_cfg = crate::loss::SymmetricCEConfig::default();
            let sce = crate::loss::symmetric_cross_entropy(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );
            let sce_norm = sce / (target_ids.len().max(1) as f32);
            batch_loss += sce_norm;

            // Compute gradients w.r.t. logits
            let mut grads_output = crate::loss::symmetric_cross_entropy_gradients(
                &probs,
                target_ids,
                sce_cfg.alpha,
                sce_cfg.beta,
                sce_cfg.epsilon,
            );

            // Backward pass: compute parameter gradients for each layer
            // Note: AttentionMoE layers use backward() directly and are handled separately
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let layer_idx = self.network.len() - 1 - rev_idx;

                let (input_grads, param_grads) =
                    layer.compute_gradients(&layer_inputs[layer_idx], &grads_output);

                let layer_grad_norm: f32 = input_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
                layer_grad_norms[layer_idx] += layer_grad_norm;

                grads_output = input_grads;

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
                layer_grad_norms
                    .iter()
                    .enumerate()
                    .map(|(i, &norm)| format!("L{}: {:.2}", i, norm))
                    .collect::<Vec<_>>()
            );
        }

        // PolyAttention-only: no auxiliary routing losses

        // Prepare averaged gradients and detect anomalies
        let mut averaged_grads_per_layer: Vec<Vec<Array2<f32>>> = Vec::new();
        let mut total_grad_norm_sq = 0.0f32;
        let mut layer_param_grad_norm_sq: Vec<f32> = vec![0.0; self.network.len()];

        for (layer_idx, param_grads) in accumulated_param_grads.into_iter().enumerate() {
            if !param_grads.is_empty() {
                let averaged_grads: Vec<Array2<f32>> = param_grads
                    .into_iter()
                    .map(|grad| grad / batch.len() as f32)
                    .collect();

                // Apply mathematically justified gradient clipping based on attention mechanism
                // properties For attention mechanisms, gradients should be bounded
                // by softmax properties and attention score ranges Maximum gradient
                // norm = sqrt(n_params) * max_reasonable_gradient_per_param
                // where max_reasonable_gradient_per_param ≈ 10.0 (based on clamped attention scores
                // [-10, 10])
                let max_reasonable_grad_per_param = 5.0;
                let max_total_grad_norm =
                    (averaged_grads.iter().map(|g| g.len()).sum::<usize>() as f32).sqrt()
                        * max_reasonable_grad_per_param;
                let mut total_layer_grad_norm_sq = 0.0;

                // First pass: compute total gradient norm for this layer
                for grad in &averaged_grads {
                    total_layer_grad_norm_sq += grad.iter().map(|&x| x * x).sum::<f32>();
                }
                let total_layer_grad_norm = total_layer_grad_norm_sq.sqrt();

                // Second pass: clip if needed using mathematically justified threshold
                let scale = if total_layer_grad_norm > max_total_grad_norm {
                    max_total_grad_norm / total_layer_grad_norm
                } else {
                    1.0
                };

                let mut clipped_grads: Vec<Array2<f32>> = if scale < 1.0 {
                    averaged_grads
                        .into_iter()
                        .map(|grad| grad.mapv(|x| x * scale))
                        .collect()
                } else {
                    averaged_grads
                };

                // Sanitize non-finite gradients proactively
                for grad in &mut clipped_grads {
                    grad.iter_mut().for_each(|v| {
                        if !v.is_finite() {
                            *v = 0.0
                        }
                    });
                }

                // Detect gradient anomalies (poisoning/training instability)
                if let Err(e) = self.detect_gradient_anomalies(&clipped_grads) {
                    tracing::error!(
                        layer_idx = layer_idx,
                        layer_type = self.network[layer_idx].layer_type(),
                        "Gradient anomaly detected in layer"
                    );
                    return Err(e);
                }

                // Compute L2 norm of gradients for this layer (after clipping)
                let mut s_layer = 0.0f32;
                for grad in &clipped_grads {
                    let s = grad.iter().map(|&x| x * x).sum::<f32>();
                    total_grad_norm_sq += s;
                    s_layer += s;
                }
                layer_param_grad_norm_sq[layer_idx] += s_layer;

                averaged_grads_per_layer.push(clipped_grads);
            } else {
                averaged_grads_per_layer.push(Vec::new());
            }
        }

        // Compute global gradient norm (L2 norm across all parameters)
        let grad_norm = total_grad_norm_sq.sqrt();

        // Compute per-layer gradient norms (post-clipping)
        let per_layer_grad_norms: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .map(|(_layer, grads)| {
                if grads.is_empty() {
                    0.0
                } else {
                    let mut s = 0.0f32;
                    for g in grads {
                        s += g.iter().map(|&x| x * x).sum::<f32>();
                    }
                    s.sqrt()
                }
            })
            .collect();

        // Median of non-zero per-layer gradient norms as bidirectional target
        let mut nonzero: Vec<f32> = per_layer_grad_norms
            .iter()
            .cloned()
            .filter(|&v| v > 0.0)
            .collect();
        let median_grad_norm = if nonzero.is_empty() {
            grad_norm.max(1e-6)
        } else {
            nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = nonzero.len() / 2;
            if nonzero.len().is_multiple_of(2) {
                (nonzero[mid - 1] + nonzero[mid]) * 0.5
            } else {
                nonzero[mid]
            }
        };

        // EMA-smooth the median to reduce step-to-step volatility
        const EMA_BETA: f32 = 0.9; // 90% memory, gentle smoothing
        let median_smoothed = if let Some(prev) = self.median_grad_ema {
            let sm = EMA_BETA * prev + (1.0 - EMA_BETA) * median_grad_norm;
            self.median_grad_ema = Some(sm);
            sm
        } else {
            self.median_grad_ema = Some(median_grad_norm);
            median_grad_norm
        };

        // Apply accumulated and averaged gradients with layer-wise adaptive learning rates
        // Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
        // Formula: lr_layer = lr_base * trust_coef * ||W|| / (||∇W|| + weight_decay * ||W|| + ε)
        // This balances gradient flow across layers of different depths

        // Compute adaptive learning rates for all layers first (to avoid borrow checker issues)
        let adaptive_lrs: Vec<f32> = self
            .network
            .iter()
            .zip(&averaged_grads_per_layer)
            .enumerate()
            .map(|(layer_idx, (layer, grads))| {
                if grads.is_empty() {
                    lr
                } else {
                    Self::compute_layer_adaptive_lr_static(
                        layer,
                        grads,
                        lr,
                        layer_idx,
                        median_smoothed,
                    )
                }
            })
            .collect();

        // Apply gradients with computed adaptive learning rates
        for ((layer, grads), adaptive_lr) in self
            .network
            .iter_mut()
            .zip(averaged_grads_per_layer)
            .zip(adaptive_lrs)
        {
            if !grads.is_empty() {
                layer.apply_gradients(&grads, adaptive_lr)?;
            }
        }

        // PolyAttention-only: no learned threshold predictors to update

        Ok((batch_loss, grad_norm, layer_param_grad_norm_sq))
    }

    /// Compute layer-wise adaptive learning rate using bidirectional LARS
    /// Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
    ///
    /// Bidirectional approach: Balance gradient flow across all layers
    /// - High-gradient layers (L0-L2): Reduce LR to prevent over-updating
    /// - Low-gradient layers (L6-L14): Increase LR to prevent under-updating
    /// - Target: All layers converge at similar rates
    ///
    /// Formula (trust-ratio + bidirectional balance):
    /// lr_layer = lr_base * clamp( (||W|| / (||∇W|| + ε)) * (median_grad_norm / (||∇W|| +
    /// ε))^power, [min,max] )
    /// - Trust-ratio term encourages proportionate updates relative to parameter scale
    /// - Bidirectional balance aligns layer grad norms towards the batch median
    fn compute_layer_adaptive_lr_static(
        layer: &LayerEnum,
        grads: &[Array2<f32>],
        base_lr: f32,
        layer_idx: usize,
        median_grad_norm: f32,
    ) -> f32 {
        // Skip for layers without gradients
        if grads.is_empty() {
            return base_lr;
        }

        // Compute gradient norm ||∇W||
        let grad_norm: f32 = grads
            .iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        // Avoid division by zero
        const EPSILON: f32 = 1e-6;
        if grad_norm < EPSILON {
            return base_lr;
        }

        // Trust-ratio term: ||W|| / ||∇W||
        let w_norm = layer.weight_norm();
        if w_norm < EPSILON {
            return base_lr;
        }
        let trust_ratio = w_norm / (grad_norm + EPSILON);

        // Bidirectional balance relative to batch median
        const POWER_BALANCE: f32 = 0.5; // Gentle correction
        let balance_scale = (median_grad_norm / (grad_norm + EPSILON)).powf(POWER_BALANCE);

        // Combined scale with conservative clamping
        // Tighter bounds reduce jitter and large swings
        const MIN_SCALE: f32 = 0.8;
        const MAX_SCALE: f32 = 1.2;
        let scale = (trust_ratio * balance_scale).clamp(MIN_SCALE, MAX_SCALE);
        let adaptive_lr = base_lr * scale;

        // Log adaptive LR for debugging (use RUST_LOG=debug to see)
        if layer_idx <= 2 || layer_idx >= 12 {
            tracing::debug!(
                layer_idx = layer_idx,
                layer_type = layer.layer_type(),
                grad_norm = grad_norm,
                base_lr = base_lr,
                adaptive_lr = adaptive_lr,
                scale = scale,
                "Bidirectional LARS"
            );
        }

        adaptive_lr
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
            let nan_count = grad.iter().filter(|&x| x.is_nan()).count();
            let inf_count = grad.iter().filter(|&x| x.is_infinite()).count();
            if nan_count > 0 || inf_count > 0 {
                tracing::error!(
                    "Non-finite gradients detected in layer {}: {} NaN, {} Inf values",
                    i,
                    nan_count,
                    inf_count
                );
                // Log some sample values for debugging
                let first_10: Vec<f32> = grad.iter().take(10).cloned().collect();
                tracing::error!("First 10 gradient values: {:?}", first_10);
                return Err(ModelError::GradientError {
                    message: format!("Non-finite gradients detected in layer {}", i),
                });
            }
        }
        Ok(())
    }

    #[inline]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        self.vocab.tokenize(text)
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

    pub fn total_weight_norm(&self) -> f32 {
        self.network.iter().map(|layer| layer.weight_norm()).sum()
    }

    pub fn train_diffusion_ce(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
        ce_weight: f32,
    ) -> Result<()> {
        let tokenized_data = data
            .par_iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        let response_spans: Vec<Option<(usize, usize)>> = tokenized_data
            .iter()
            .map(|seq| response_span_from_tokens(&self.vocab, seq))
            .collect();

        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        let mut embeddings_idx: Option<usize> = None;
        let mut norm_idx: Option<usize> = None;
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            match layer {
                LayerEnum::TokenEmbeddings(_) => {
                    if embeddings_idx.is_none() {
                        embeddings_idx = Some(i)
                    }
                }
                LayerEnum::DiffusionBlock(_) => diffusion_blocks_idx.push(i),
                LayerEnum::DynamicTanhNorm(_) => norm_idx = Some(i),
                LayerEnum::OutputProjection(_) => out_proj_idx = Some(i),
                _ => {}
            }
        }
        if embeddings_idx.is_none() || diffusion_blocks_idx.is_empty() || out_proj_idx.is_none() {
            return Err(ModelError::Training {
                message: String::from(
                    "Missing required layers for diffusion CE (embeddings/diffusion/output)",
                ),
            });
        }
        let first_block = diffusion_blocks_idx[0];
        let num_timesteps = if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
            b.noise_scheduler.num_timesteps()
        } else {
            1000
        };
        let normal = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        let lambda_ce_schedule = |t: usize| -> f32 {
            let total = num_timesteps.max(1) as f32;
            let center = 0.25 * total;
            let sigma = (0.1 * total).max(1.0);
            let capped_t = t.min(num_timesteps.saturating_sub(1)) as f32;
            let x = (center - capped_t) / sigma;
            1.0 / (1.0 + (-x).exp())
        };
        let log_dir = std::path::Path::new("training_logs");
        let _ = std::fs::create_dir_all(log_dir);
        let ts = format!("{}", chrono::Utc::now().format("%Y%m%d-%H%M%S"));
        let mut log_file =
            std::fs::File::create(log_dir.join(format!("diffusion-{}.csv", ts))).ok();
        if let Some(f) = &mut log_file {
            use std::io::Write;
            let _ = writeln!(f, "epoch,loss,sce,mse,lambda_ce,lr,grad_norm");
        }
        let mut lr_scale = 1.0f32;
        let mut best_loss = f32::INFINITY;
        let mut plateau_epochs = 0usize;
        let plateau_patience = 5usize;
        let plateau_reduce = 0.5f32;
        let min_lr_scale = 0.1f32;
        let effective_batch_size = batch_size.max(1);

        // Warmup epochs default to 15% of total for stability
        let warmup_epochs = ((epochs as f32) * 0.15).ceil() as usize;
        for epoch in 0..epochs {
            // Learning rate warmup + cosine annealing (SGDR)
            let base_lr = if epoch < warmup_epochs {
                lr * ((epoch + 1) as f32 / warmup_epochs as f32)
            } else {
                let t = (epoch - warmup_epochs) as f32;
                let t_max = (epochs - warmup_epochs).max(1) as f32;
                let lr_min = lr * 0.10;
                let lr_max = lr;
                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t / t_max).cos())
            };
            let effective_lr = base_lr * lr_scale;
            let mut total_loss = 0.0f32;
            let mut total_mse = 0.0f32;
            let mut mse_examples = 0usize;
            let mut total_ce = 0.0f32;
            let mut total_lambda_ce = 0.0f32;
            let mut count = 0usize;
            let mut total_grad_norm_sq = 0.0f32;

            let mut batch_start = 0usize;
            while batch_start < tokenized_data.len() {
                let batch_end = (batch_start + effective_batch_size).min(tokenized_data.len());
                let mut grads_per_layer: Vec<Option<Vec<Array2<f32>>>> =
                    vec![None; self.network.len()];
                let mut examples_in_batch = 0usize;
                for seq_idx in batch_start..batch_end {
                    let training_row = &tokenized_data[seq_idx];
                    if training_row.len() < 2 {
                        continue;
                    }
                    examples_in_batch += 1;

                    let response_span = response_spans
                        .get(seq_idx)
                        .copied()
                        .flatten();

                    let input_ids = &training_row[..training_row.len() - 1];
                    let target_ids = &training_row[1..];

                    let mut ids_arr = Array2::<f32>::zeros((1, input_ids.len()));
                    for (i, &tid) in input_ids.iter().enumerate() {
                        ids_arr[[0, i]] = tid as f32;
                    }

                    // x0 via embeddings
                    let emb_idx = embeddings_idx.unwrap();
                    let x0 = match &mut self.network[emb_idx] {
                        LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                        _ => {
                            return Err(ModelError::Training {
                                message: String::from("Embeddings layer missing"),
                            });
                        }
                    };

                    // Decide discrete masked vs continuous path per first diffusion block
                    let is_discrete =
                        if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                            b.is_discrete_masked()
                        } else {
                            false
                        };
                    let mask_id_opt =
                        if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                            b.mask_token_id()
                        } else {
                            None
                        };
                    let mut noise = Array2::<f32>::zeros(x0.raw_dim());
                    for v in noise.iter_mut() {
                        *v = normal.sample(&mut rng) as f32;
                    }
                    // Adaptive timestep sampling (curriculum by epoch + complexity)
                    let complexity = {
                        let unique = training_row
                            .iter()
                            .copied()
                            .collect::<std::collections::BTreeSet<usize>>()
                            .len() as f32;
                        (unique / training_row.len().max(1) as f32).clamp(0.0, 1.0)
                    };
                    let max_t = ((num_timesteps as f32) * ((epoch + 1) as f32 / epochs as f32))
                        .round() as usize; // curriculum
                    let base_t = rng.random_range(0..max_t.max(1));
                    let t = (((1.0 - complexity) * base_t as f32).round() as usize)
                        .min(max_t.max(1) - 1);
                    let (x_t, sqrt_a, sqrt_one_minus_a, discrete_used) = {
                        if is_discrete {
                            let mask_token_id = mask_id_opt
                                .or_else(|| self.vocab.encode("<mask>"))
                                .unwrap_or(self.vocab.encode_or_unknown("<unk>").unwrap_or(0));
                            let ids_masked =
                                if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                                    if let Some(ds) = &b.discrete_scheduler {
                                        if let Some((span_start, span_end)) = response_span {
                                            ds.mask_sequence_span_at_t(
                                                &ids_arr,
                                                mask_token_id,
                                                t,
                                                span_start,
                                                span_end,
                                            )
                                        } else {
                                            ds.mask_sequence_at_t(&ids_arr, mask_token_id, t)
                                        }
                                    } else {
                                        ids_arr.clone()
                                    }
                                } else {
                                    ids_arr.clone()
                                };
                            let x_t_local = match &mut self.network[embeddings_idx.unwrap()] {
                                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_masked),
                                _ => x0.clone(),
                            };
                            (x_t_local, 1.0, 0.0, true)
                        } else {
                            if let LayerEnum::DiffusionBlock(b) = &self.network[first_block] {
                                let x_t_local = b.noise_scheduler.q_sample(&x0, t, &noise);
                                let sa = b.noise_scheduler.sqrt_alpha_cumprod(t);
                                let soa = b.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                                (x_t_local, sa, soa, false)
                            } else {
                                return Err(ModelError::Training {
                                    message: String::from("Diffusion scheduler missing"),
                                });
                            }
                        }
                    };

                    // Predict via full diffusion stack
                    let mut eps_pred = x_t.clone();
                    for &idx in &diffusion_blocks_idx {
                        if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                            b.set_timestep(t);
                            eps_pred = b.forward_with_timestep(&eps_pred, t);
                        }
                    }

                    // Recover x0_hat (continuous) or use predicted embeddings directly (discrete)
                    let x0_hat = if discrete_used {
                        eps_pred.clone()
                    } else {
                        let safe_sqrt_a = sqrt_a.max(1e-6);
                        (&x_t - &(eps_pred.clone() * sqrt_one_minus_a)) / safe_sqrt_a
                    };

                    // Forward through final norm (if present) and output projection
                    let mut hidden = x0_hat.clone();
                    if let Some(nidx) = norm_idx {
                        if let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx] {
                            hidden = norm.forward(&hidden);
                        }
                    }

                    let logits = if let Some(opidx) = out_proj_idx {
                        if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                            op.forward(&hidden)
                        } else {
                            return Err(ModelError::Training {
                                message: String::from("OutputProjection mismatch"),
                            });
                        }
                    } else {
                        return Err(ModelError::Training {
                            message: String::from("OutputProjection missing"),
                        });
                    };

                    // CE loss over next-token targets rows [0..target_len]
                    let probs = crate::softmax::Softmax::new().forward_immutable(&logits.view());
                    let target_len = target_ids.len();
                    let probs_slice = probs.slice(ndarray::s![0..target_len, ..]).to_owned();
                    let lambda_ce = if discrete_used {
                        1.0f32
                    } else {
                        lambda_ce_schedule(t)
                    };
                    let lambda_eps = if discrete_used { 0.0f32 } else { 1.0f32 - lambda_ce };
                    total_lambda_ce += lambda_ce;
                    let sce = crate::loss::symmetric_cross_entropy(
                        &probs_slice,
                        target_ids,
                        ce_weight * lambda_ce,
                        ce_weight * lambda_ce,
                        1e-4,
                    );

                    // CE grads expanded to full logits shape
                    let mut grads_logits = Array2::<f32>::zeros(logits.raw_dim());
                    let sce_grads_slice = crate::loss::symmetric_cross_entropy_gradients(
                        &probs_slice,
                        target_ids,
                        ce_weight * lambda_ce,
                        ce_weight * lambda_ce,
                        1e-4,
                    );
                    grads_logits
                        .slice_mut(ndarray::s![0..target_len, ..])
                        .assign(&sce_grads_slice);

                    // Backward through output projection
                    let (mut grad_hidden, op_param_grads) = if let Some(opidx) = out_proj_idx {
                        if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                            op.compute_gradients(&hidden, &grads_logits)
                        } else {
                            (grads_logits.clone(), Vec::new())
                        }
                    } else {
                        (grads_logits.clone(), Vec::new())
                    };
                    if let Some(opidx) = out_proj_idx {
                        if !op_param_grads.is_empty() {
                            if let Some(slot) = &mut grads_per_layer[opidx] {
                                for (i, g) in op_param_grads.iter().enumerate() {
                                    if i < slot.len() {
                                        slot[i] = &slot[i] + g;
                                    } else {
                                        slot.push(g.clone());
                                    }
                                }
                            } else {
                                grads_per_layer[opidx] = Some(op_param_grads.clone());
                            }
                        }
                    }

                    // Backward through norm to x0_hat
                    if let Some(nidx) = norm_idx {
                        if let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx] {
                            grad_hidden = norm.backward(&grad_hidden, lr);
                        }
                    }

                    // Build gradient for diffusion stack
                    // Build gradient for diffusion stack from mixed objectives
                    let mut grad_eps = if discrete_used {
                        // Discrete masked: CE only path, treat as grad on predicted embeddings
                        grad_hidden.clone()
                    } else {
                        // Chain rule: dL_ce/dε = ( -√(1-ᾱ)/√(ᾱ) ) · dL/dx̂0
                        let safe_sqrt_a = sqrt_a.max(1e-6);
                        let coeff = -sqrt_one_minus_a / safe_sqrt_a;
                        let grad_ce_eps = grad_hidden.mapv(|x| x * coeff);
                        // Epsilon MSE gradients: 2/N * (ε_pred − ε_true)
                        let grad_mse_eps = crate::loss::epsilon_mse_gradients(&eps_pred, &noise);
                        // Mix by λ
                        grad_ce_eps.mapv(|x| x * lambda_ce) + grad_mse_eps.mapv(|x| x * lambda_eps)
                    };

                    // Gradient clipping by global norm
                    let grad_norm_eps: f32 = grad_eps.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let clip_norm: f32 = 2.0;
                    if grad_norm_eps > clip_norm && grad_norm_eps.is_finite() {
                        let scale = clip_norm / grad_norm_eps;
                        grad_eps.mapv_inplace(|g| g * scale);
                    }

                    // Backprop through diffusion stack (reverse order)
                    for &idx in diffusion_blocks_idx.iter().rev() {
                        let (in_grad, param_grads) = match &self.network[idx] {
                            LayerEnum::DiffusionBlock(b) => b.compute_gradients(&x_t, &grad_eps),
                            _ => (grad_eps.clone(), Vec::new()),
                        };
                        if !param_grads.is_empty() {
                            if let Some(slot) = &mut grads_per_layer[idx] {
                                for (i, g) in param_grads.iter().enumerate() {
                                    if i < slot.len() {
                                        slot[i] = &slot[i] + g;
                                    } else {
                                        slot.push(g.clone());
                                    }
                                }
                            } else {
                                grads_per_layer[idx] = Some(param_grads.clone());
                            }
                        }
                        grad_eps = in_grad;
                    }

                    // Map gradients from x_t back to x_0 and update embeddings
                    let grad_x0 = if discrete_used {
                        // Discrete masked: x_t derived from embeddings(ids_masked) directly
                        grad_eps.clone()
                    } else {
                        // Continuous: x_t = sqrt(a) * x0 + sqrt(1-a) * noise
                        // dL/dx0 = sqrt(a) * dL/dx_t
                        let sa = sqrt_a.max(1e-6);
                        grad_eps.mapv(|g| g * sa)
                    };

                    if let Some(eidx) = embeddings_idx {
                        if let LayerEnum::TokenEmbeddings(layer) = &mut self.network[eidx] {
                            let (emb_in_grad, emb_param_grads) =
                                layer.compute_gradients(&ids_arr, &grad_x0);
                            let _ = emb_in_grad;
                            if !emb_param_grads.is_empty() {
                                if let Some(slot) = &mut grads_per_layer[eidx] {
                                    for (i, g) in emb_param_grads.iter().enumerate() {
                                        if i < slot.len() {
                                            slot[i] = &slot[i] + g;
                                        } else {
                                            slot.push(g.clone());
                                        }
                                    }
                                } else {
                                    grads_per_layer[eidx] = Some(emb_param_grads.clone());
                                }
                            }
                        }
                    }

                    // Losses and grad norm
                    // Track epsilon MSE separately for monitoring when using continuous noise
                    let mse = if discrete_used {
                        0.0f32
                    } else {
                        let value = crate::loss::epsilon_mse(&eps_pred, &noise);
                        total_mse += value;
                        mse_examples += 1;
                        value
                    };
                    let loss = if discrete_used {
                        sce
                    } else {
                        lambda_ce * sce + lambda_eps * mse
                    };
                    total_loss += loss;
                    total_ce += sce;
                    count += 1;
                    total_grad_norm_sq += grad_eps.iter().map(|&x| x * x).sum::<f32>();
                }
                // Apply averaged grads per layer after batch
                for (idx, maybe_grads) in grads_per_layer.into_iter().enumerate() {
                    if let Some(mut grads) = maybe_grads {
                        if examples_in_batch > 0 {
                            for g in &mut grads {
                                *g = g.mapv(|x| x / examples_in_batch as f32);
                            }
                        }
                        let clip_layer = 1000.0f32;
                        for g in &mut grads {
                            let nrm: f32 = g.iter().map(|&x| x * x).sum::<f32>().sqrt();
                            if nrm.is_finite() && nrm > clip_layer {
                                let scale = clip_layer / nrm;
                                g.mapv_inplace(|x| x * scale);
                            }
                        }
                        // Detect anomalies before applying
                        self.detect_gradient_anomalies(&grads)?;
                        match &mut self.network[idx] {
                            LayerEnum::DiffusionBlock(b) => {
                                b.apply_gradients(&grads, effective_lr)?
                            }
                            LayerEnum::OutputProjection(op) => {
                                op.apply_gradients(&grads, effective_lr)?
                            }
                            LayerEnum::TokenEmbeddings(layer) => {
                                layer.apply_gradients(&grads, effective_lr)?
                            }
                            _ => {}
                        }
                    }
                }
                batch_start = batch_end;
            }

            let avg_loss = if count > 0 {
                total_loss / count as f32
            } else {
                0.0
            };
            let avg_sce = if count > 0 {
                total_ce / count as f32
            } else {
                0.0
            };
            let avg_mse = if mse_examples > 0 {
                total_mse / mse_examples as f32
            } else {
                0.0
            };
            let avg_lambda_ce = if count > 0 {
                total_lambda_ce / count as f32
            } else {
                0.0
            };
            let grad_norm = total_grad_norm_sq.sqrt();
            info!(
                epoch = epoch,
                loss = avg_loss,
                sce = avg_sce,
                mse = avg_mse,
                lambda_ce = avg_lambda_ce,
                lr = effective_lr,
                grad_norm = grad_norm,
                "Diffusion mixed (CE+MSE) epoch"
            );
            if let Some(f) = &mut log_file {
                use std::io::Write;
                let _ = writeln!(
                    f,
                    "{},{},{},{},{},{},{}",
                    epoch, avg_loss, avg_sce, avg_mse, avg_lambda_ce, effective_lr, grad_norm
                );
            }
            if avg_loss + 1e-5 < best_loss {
                best_loss = avg_loss;
                plateau_epochs = 0;
            } else {
                plateau_epochs += 1;
            }
            if plateau_epochs >= plateau_patience {
                if lr_scale > min_lr_scale {
                    lr_scale = (lr_scale * plateau_reduce).max(min_lr_scale);
                    warn!(
                        epoch = epoch,
                        lr_scale = lr_scale,
                        "Reduce-on-plateau triggered: scaling LR"
                    );
                }
                plateau_epochs = 0;
            }
        }

        Ok(())
    }

    /// Sample from reverse diffusion process for generative decoding
    ///
    /// Starts from pure noise and progressively denoises to generate sequences.
    pub fn sample_diffusion(&mut self, max_length: usize, steps: Option<usize>) -> String {
        self.sample_diffusion_with_prompt("", max_length, steps)
    }

    pub fn sample_diffusion_with_prompt(
        &mut self,
        prompt: &str,
        max_length: usize,
        steps: Option<usize>,
    ) -> String {
        let steps = steps.unwrap_or(100);
        let mut rng = rand::rng();

        // Tokenize the prompt if provided
        let prompt_tokens = if !prompt.is_empty() {
            self.tokenize(prompt)
        } else {
            Vec::new()
        };

        // Get embedding dimension from the first layer (TokenEmbeddings)
        let embedding_dim =
            if let Some(LayerEnum::TokenEmbeddings(embeddings)) = self.network.first() {
                embeddings.token_embeddings.ncols()
            } else {
                return "Error: Cannot determine embedding dimension".to_string();
            };

        // Snapshot token embeddings (for prompt conditioning) before borrowing network mutably
        let token_embs_cloned = match self.network.get(0) {
            Some(LayerEnum::TokenEmbeddings(embeddings)) => {
                Some(embeddings.token_embeddings.clone())
            }
            _ => None,
        };

        // Get diffusion block indices
        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        for (i, layer) in self.network.iter().enumerate() {
            if let LayerEnum::DiffusionBlock(_) = layer {
                diffusion_blocks_idx.push(i);
            }
        }

        if diffusion_blocks_idx.is_empty() {
            return "Error: No diffusion blocks found".to_string();
        }

        // Calculate available length for generation (accounting for prompt)
        let _available_length = max_length.saturating_sub(prompt_tokens.len());

        // Start with pure noise: x_T ~ N(0, I), but condition first positions on prompt embeddings
        let mut current_sample = Array2::<f32>::zeros((max_length, embedding_dim));
        for i in 0..max_length {
            for j in 0..embedding_dim {
                current_sample[[i, j]] = rng.random::<f32>() * 2.0 - 1.0;
            }
        }
        if !prompt_tokens.is_empty() {
            // Replace the first K rows with prompt token embeddings
            let k = prompt_tokens.len().min(max_length);
            if let Some(token_embs) = token_embs_cloned {
                for i in 0..k {
                    let tid = prompt_tokens[i].min(token_embs.nrows().saturating_sub(1));
                    current_sample.row_mut(i).assign(&token_embs.row(tid));
                }
            }
        }

        // Reverse diffusion process: x_{t-1} = 1/√ᾱ_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t *
        // z
        let is_discrete = diffusion_blocks_idx.iter().any(|&idx| {
            if let LayerEnum::DiffusionBlock(b) = &self.network[idx] {
                b.is_discrete_masked()
            } else {
                false
            }
        });
        if is_discrete {
            let mask_token_id =
                if let LayerEnum::DiffusionBlock(b0) = &self.network[diffusion_blocks_idx[0]] {
                    b0.mask_token_id()
                } else {
                    None
                }
                .or_else(|| self.vocab.encode("<mask>"))
                .unwrap_or(self.vocab.encode_or_unknown("<unk>").unwrap_or(0));
            let mut ids_arr = Array2::<f32>::zeros((1, max_length));
            for i in 0..max_length {
                ids_arr[[0, i]] = mask_token_id as f32;
            }
            for i in 0..prompt_tokens.len().min(max_length) {
                ids_arr[[0, i]] = prompt_tokens[i] as f32;
            }

            for t in (1..=steps).rev() {
                let t_idx = t - 1;
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.set_timestep(t_idx);
                    }
                }
                let x_t = match &mut self.network[0] {
                    LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                    _ => current_sample.clone(),
                };
                let mut hidden = x_t.clone();
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        hidden = b.forward_with_timestep(&hidden, t_idx);
                    }
                }
                for layer in &mut self.network {
                    if let LayerEnum::DynamicTanhNorm(norm) = layer {
                        hidden = norm.forward(&hidden);
                    }
                }
                let mut logits: Option<Array2<f32>> = None;
                for layer in &mut self.network {
                    if let LayerEnum::OutputProjection(op) = layer {
                        logits = Some(op.forward(&hidden));
                        break;
                    }
                }
                let logits = match logits {
                    Some(l) => l,
                    None => break,
                };
                let softmax = crate::softmax::Softmax::new();
                let probs = softmax.forward_immutable(&logits.view());
                if let LayerEnum::DiffusionBlock(b0) = &self.network[diffusion_blocks_idx[0]] {
                    if let Some(ds) = &b0.discrete_scheduler {
                        ids_arr =
                            ds.reverse_unmask_step(&ids_arr, &probs, mask_token_id, t_idx, 0.9);
                    }
                }
                let mut cur_unmasked = 0usize;
                for i in 0..max_length {
                    if ids_arr[[0, i]] != mask_token_id as f32 {
                        cur_unmasked += 1;
                    }
                }
                if cur_unmasked >= max_length {
                    break;
                }
            }
            current_sample = match &mut self.network[0] {
                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                _ => current_sample,
            };
        } else {
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_use_ema_for_sampling(true);
                }
            }
            for t in (1..=steps).rev() {
                let t_idx = t - 1;
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        b.set_timestep(t_idx);
                    }
                }
                let mut predicted_noise = current_sample.clone();
                for &idx in &diffusion_blocks_idx {
                    if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                        predicted_noise = b.forward_with_timestep(&predicted_noise, t_idx);
                    }
                }
                let noise_scheduler =
                    if let LayerEnum::DiffusionBlock(b0) = &self.network[diffusion_blocks_idx[0]] {
                        &b0.noise_scheduler
                    } else {
                        unreachable!()
                    };
                current_sample =
                    noise_scheduler.ddim_step(&current_sample, t_idx, &predicted_noise);
            }
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_use_ema_for_sampling(false);
                }
            }
        }

        // Decode using OutputProjection on the denoised embeddings
        // Pass through final DynamicTanhNorm if present
        let mut hidden = current_sample.clone();
        for layer in &mut self.network {
            match layer {
                LayerEnum::DynamicTanhNorm(norm) => {
                    hidden = norm.forward(&hidden);
                }
                _ => {}
            }
        }

        // Find OutputProjection layer and compute logits
        let mut logits: Option<Array2<f32>> = None;
        for layer in &mut self.network {
            if let LayerEnum::OutputProjection(op) = layer {
                logits = Some(op.forward(&hidden));
                break;
            }
        }
        let logits = match logits {
            Some(l) => l,
            None => return "Error: No OutputProjection found".to_string(),
        };

        let mut tokens = prompt_tokens.clone();
        let temperature: f32 = 1.0;
        let top_p: f32 = 0.9;
        let softmax = crate::softmax::Softmax::new();
        for i in prompt_tokens.len()..max_length {
            let mut row_scaled = logits.row(i).to_owned();
            if temperature > 0.0 {
                row_scaled.mapv_inplace(|x| x / temperature);
            }
            let row2d = row_scaled.insert_axis(Axis(0));
            let probs_row2d = softmax.forward_immutable(&row2d.view());
            let probs_row = probs_row2d.row(0).to_owned();
            // Nucleus (top-p) sampling
            let mut indexed: Vec<(usize, f32)> = probs_row
                .iter()
                .enumerate()
                .map(|(tid, &p)| (tid, p.max(0.0)))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            let mut cutoff = 0usize;
            for (k, &(_, p)) in indexed.iter().enumerate() {
                cum += p;
                cutoff = k;
                if cum >= top_p {
                    break;
                }
            }
            let nucleus = &indexed[..=cutoff];
            let sum_p: f32 = nucleus.iter().map(|&(_, p)| p).sum();
            let r: f32 = rng.random::<f32>();
            let mut acc = 0.0f32;
            let mut chosen = nucleus[0].0;
            for &(tid, p) in nucleus {
                acc += p / (sum_p.max(1e-8));
                if r <= acc {
                    chosen = tid;
                    break;
                }
            }
            tokens.push(chosen);
            if chosen == 0 {
                break;
            }
        }

        let decoded_text = tokens
            .iter()
            .filter_map(|&token_id| self.vocab.decode(token_id))
            .collect::<Vec<&str>>()
            .join(" ");

        format!("Generated text: {}", decoded_text)
    }

    pub fn evaluate_perplexity_diffusion(&mut self, data: Vec<&str>) -> Result<f32> {
        let tokenized = data
            .par_iter()
            .map(|s| self.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let mut total_ce = 0.0f32;
        let mut count = 0usize;
        // Use t=0 path to approximate language modeling
        // Build layer indices once
        let mut diffusion_blocks_idx: Vec<usize> = Vec::new();
        let mut embeddings_idx: Option<usize> = None;
        let mut norm_idx: Option<usize> = None;
        let mut out_proj_idx: Option<usize> = None;
        for (i, layer) in self.network.iter().enumerate() {
            match layer {
                LayerEnum::TokenEmbeddings(_) => {
                    if embeddings_idx.is_none() {
                        embeddings_idx = Some(i)
                    }
                }
                LayerEnum::DiffusionBlock(_) => diffusion_blocks_idx.push(i),
                LayerEnum::DynamicTanhNorm(_) => norm_idx = Some(i),
                LayerEnum::OutputProjection(_) => out_proj_idx = Some(i),
                _ => {}
            }
        }
        if embeddings_idx.is_none() || diffusion_blocks_idx.is_empty() || out_proj_idx.is_none() {
            return Err(ModelError::Training {
                message: String::from("Missing layers for diffusion perplexity eval"),
            });
        }
        for seq in tokenized.iter() {
            if seq.len() < 2 {
                continue;
            }
            let input_ids = &seq[..seq.len() - 1];
            let target_ids = &seq[1..];
            let mut ids_arr = ndarray::Array2::<f32>::zeros((1, input_ids.len()));
            for (i, &tid) in input_ids.iter().enumerate() {
                ids_arr[[0, i]] = tid as f32;
            }
            let x0 = match &mut self.network[embeddings_idx.unwrap()] {
                LayerEnum::TokenEmbeddings(layer) => layer.forward(&ids_arr),
                _ => continue,
            };
            let mut hidden = x0.clone();
            for &idx in &diffusion_blocks_idx {
                if let LayerEnum::DiffusionBlock(b) = &mut self.network[idx] {
                    b.set_timestep(0);
                    hidden = b.forward_with_timestep(&hidden, 0);
                }
            }
            if let Some(nidx) = norm_idx {
                if let LayerEnum::DynamicTanhNorm(norm) = &mut self.network[nidx] {
                    hidden = norm.forward(&hidden);
                }
            }
            let logits = if let Some(opidx) = out_proj_idx {
                if let LayerEnum::OutputProjection(op) = &mut self.network[opidx] {
                    op.forward(&hidden)
                } else {
                    continue;
                }
            } else {
                continue;
            };
            let probs = crate::softmax::Softmax::new().forward_immutable(&logits.view());
            let target_len = target_ids.len();
            let probs_slice = probs.slice(ndarray::s![0..target_len, ..]).to_owned();
            let ce = crate::loss::cross_entropy(&probs_slice, target_ids);
            total_ce += ce;
            count += 1;
        }
        if count == 0 {
            return Ok(f32::INFINITY);
        }
        let avg_ce = total_ce / (count as f32);
        let ppl = (avg_ce).exp();
        Ok(ppl)
    }

    pub fn evaluate_bleu(&self, inputs: Vec<&str>, outputs: Vec<&str>) -> Result<(f32, f32)> {
        let refs = inputs
            .iter()
            .map(|s| self.vocab.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let cands = outputs
            .iter()
            .map(|s| self.vocab.tokenize(s))
            .collect::<Vec<Vec<usize>>>();
        let (b1, b2) = corpus_bleu_1_2(&refs, &cands);
        Ok((b1, b2))
    }

    /// Get token embedding vector (helper method)
    fn get_token_embedding(&self, token_id: usize, embedding_dim: usize) -> Array1<f32> {
        // Access TokenEmbeddings layer directly
        if let Some(LayerEnum::TokenEmbeddings(embeddings)) = self.network.first() {
            embeddings.token_embeddings.row(token_id).to_owned()
        } else {
            // Fallback: return random embedding if no embeddings layer found
            let mut rng = rand::rng();
            Array1::from_vec(
                (0..embedding_dim)
                    .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                    .collect(),
            )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss::symmetric_cross_entropy;

    #[test]
    fn test_network_description_includes_decoder() {
        let llm = LLM::default();
        let description = llm.network_description();

        // Should include network layers and decoder type
        assert!(description.contains("OutputProjection"));
        assert!(description.contains("GreedyDecoder"));
        println!("Network description: {}", description);
    }

    #[test]
    fn test_greedy_decoder_creation() {
        let vocab = Vocab::default();
        let network = Vec::new(); // Empty network for testing
        let llm = LLM::with_greedy_decoder(vocab, network);

        match llm.decoder {
            DecoderType::Greedy(_) => {}
        }

        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");
    }

    #[test]
    fn test_decoder_switching() {
        let mut llm = LLM::default();

        // Should start with GreedyDecoder
        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");

        // Switch to Greedy (should remain Greedy)
        llm.enable_greedy();
        assert_eq!(llm.decoder.layer_type(), "GreedyDecoder");
    }

    #[test]
    fn test_response_span_detection() {
        let vocab = Vocab::new(vec![
            "User", "Assistant", ":", "Hello", "World", "</s>", "<unk>", "<mask>",
        ]);
        let tokens = vec![
            vocab.encode("User").unwrap(),
            vocab.encode(":").unwrap(),
            vocab.encode("Hello").unwrap(),
            vocab.encode("Assistant").unwrap(),
            vocab.encode(":").unwrap(),
            vocab.encode("World").unwrap(),
            vocab.encode("</s>").unwrap(),
        ];
        let span = response_span_from_tokens(&vocab, &tokens).expect("span");
        assert_eq!(span, (5, 6));
    }
}
#[test]
fn test_ce_loss_normalized() {
    let probs = ndarray::Array2::<f32>::from_elem((4, 8), 1.0 / 8.0);
    let targets = vec![1usize, 2usize, 3usize, 4usize];
    let sce = crate::loss::symmetric_cross_entropy(&probs, &targets, 1.0, 1.0, 1e-4);
    let norm = sce / targets.len() as f32;
    assert!(norm.is_finite());
    assert!(norm > 0.0);
}
