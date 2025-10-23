use std::cmp::Ordering;
use std::fs;

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::{
    errors::{ModelError, Result},
    embeddings::Embeddings,
    output_projection::OutputProjection,
    MAX_SEQ_LEN,
    Vocab,
};

#[derive(Serialize, Deserialize, Debug)]
pub enum LayerEnum {
    Embeddings(Embeddings),
    // Removed SelfAttention variant
    // Removed FeedForward variant; SwiGLU is the only FFN
    SwiGLU(Box<crate::swiglu::SwiGLU>),

    DynamicTanhNorm(crate::dynamic_tanh_norm::DynamicTanhNorm),
    OutputProjection(OutputProjection),

    // Removed TRMBlock variant
    PolyAttention(Box<crate::poly_attention::PolyAttention>),
}

impl LayerEnum {
    // Removed downcast helpers for SelfAttention/TRM to simplify to PolyAttention-only
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            LayerEnum::Embeddings(layer) => layer.layer_type(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.layer_type(),

            LayerEnum::DynamicTanhNorm(layer) => layer.layer_type(),
            LayerEnum::OutputProjection(layer) => layer.layer_type(),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.layer_type(),
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.forward(input),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.forward(input),

            LayerEnum::DynamicTanhNorm(layer) => layer.forward(input),
            LayerEnum::OutputProjection(layer) => layer.forward(input),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.forward(input),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            LayerEnum::Embeddings(layer) => layer.compute_gradients(input, output_grads),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.compute_gradients(input, output_grads),

            LayerEnum::DynamicTanhNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::OutputProjection(layer) => layer.compute_gradients(input, output_grads),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match self {
            LayerEnum::Embeddings(layer) => layer.apply_gradients(param_grads, lr),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.apply_gradients(param_grads, lr),

            LayerEnum::DynamicTanhNorm(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::OutputProjection(layer) => layer.apply_gradients(param_grads, lr),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.backward(grads, lr),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.backward(grads, lr),

            LayerEnum::DynamicTanhNorm(layer) => layer.backward(grads, lr),
            LayerEnum::OutputProjection(layer) => layer.backward(grads, lr),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.backward(grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            LayerEnum::Embeddings(layer) => layer.parameters(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::SwiGLU(layer) => layer.parameters(),

            LayerEnum::DynamicTanhNorm(layer) => layer.parameters(),
            LayerEnum::OutputProjection(layer) => layer.parameters(),
            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.parameters(),
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
        use crate::model_builder::build_network;
        use crate::model_config::ModelConfig;

        let config = ModelConfig::default();
        let vocab = Vocab::default();
        let network = build_network(&config, &vocab);

        Self {
            vocab,
            network,
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<LayerEnum>) -> Self {
        Self {
            vocab,
            network,
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
            .map(|&t| self.vocab.decode(t).unwrap())
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
            let _training_progress = if epoch < warmup_epochs {
                0.0
            } else {
                (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32
            };


            // Process data in batches
            for batch in tokenized_data.chunks(batch_size) {
                let (batch_loss, grad_norm) = self.train_batch(batch, effective_lr)?;
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
                "Training epoch completed{}",
                warmup_status
            );

        }

        Ok(())
    }

    /// Train on a single batch of sequences
    /// Returns (batch_loss, gradient_norm)
    fn train_batch(&mut self, batch: &[Vec<usize>], lr: f32) -> Result<(f32, f32)> {
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


            // Backward pass: compute parameter gradients for each layer
            // Note: AttentionMoE layers use backward() directly and are handled separately
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let layer_idx = self.network.len() - 1 - rev_idx;


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

        // PolyAttention-only: no auxiliary routing losses
        
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
                layer.apply_gradients(&averaged_grads, adaptive_lr)?;
            }
        }

        // PolyAttention-only: no learned threshold predictors to update

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
            let sum_exp: f32 = exp_values.iter().copied().sum();

            // Normalize by sum without aliasing borrows
            for (r, &exp_val) in row.iter_mut().zip(exp_values.iter()) {
                *r = exp_val / sum_exp;
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
