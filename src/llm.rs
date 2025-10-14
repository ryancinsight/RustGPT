use std::cmp::Ordering;
use std::fs;

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tracing::{info, instrument};

use crate::{
    EMBEDDING_DIM, Embeddings, HIDDEN_DIM, MAX_SEQ_LEN, Vocab, errors::{ModelError, Result}, gradient_clipping::{AdaptiveClippingConfig, AdaptiveGradientClipping, GradientClipping}, output_projection::OutputProjection,
    transformer::TransformerBlock,
};

#[derive(Serialize, Deserialize)]
pub enum LayerEnum {
    Embeddings(Embeddings),
    SelfAttention(Box<crate::self_attention::SelfAttention>),
    FeedForward(Box<crate::feed_forward::FeedForward>),
    LayerNorm(crate::layer_norm::LayerNorm),
    OutputProjection(OutputProjection),
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            LayerEnum::Embeddings(layer) => layer.layer_type(),
            LayerEnum::SelfAttention(layer) => layer.layer_type(),
            LayerEnum::FeedForward(layer) => layer.layer_type(),
            LayerEnum::LayerNorm(layer) => layer.layer_type(),
            LayerEnum::OutputProjection(layer) => layer.layer_type(),
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.forward(input),
            LayerEnum::SelfAttention(layer) => layer.forward(input),
            LayerEnum::FeedForward(layer) => layer.forward(input),
            LayerEnum::LayerNorm(layer) => layer.forward(input),
            LayerEnum::OutputProjection(layer) => layer.forward(input),
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
            LayerEnum::FeedForward(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::LayerNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::OutputProjection(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        match self {
            LayerEnum::Embeddings(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::SelfAttention(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::FeedForward(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::LayerNorm(layer) => layer.apply_gradients(param_grads, lr),
            LayerEnum::OutputProjection(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerEnum::Embeddings(layer) => layer.backward(grads, lr),
            LayerEnum::SelfAttention(layer) => layer.backward(grads, lr),
            LayerEnum::FeedForward(layer) => layer.backward(grads, lr),
            LayerEnum::LayerNorm(layer) => layer.backward(grads, lr),
            LayerEnum::OutputProjection(layer) => layer.backward(grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            LayerEnum::Embeddings(layer) => layer.parameters(),
            LayerEnum::SelfAttention(layer) => layer.parameters(),
            LayerEnum::FeedForward(layer) => layer.parameters(),
            LayerEnum::LayerNorm(layer) => layer.parameters(),
            LayerEnum::OutputProjection(layer) => layer.parameters(),
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

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32);
}

#[derive(Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<LayerEnum>,
    #[serde(skip)]
    pub gradient_clipper: Option<Box<dyn GradientClipping>>,
}

impl Default for LLM {
    fn default() -> Self {
        let transformer_block = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        Self {
            vocab: Vocab::default(),
            network: vec![
                LayerEnum::Embeddings(Embeddings::default()),
                LayerEnum::SelfAttention(Box::new(transformer_block.attention)),
                LayerEnum::LayerNorm(transformer_block.norm1),
                LayerEnum::FeedForward(Box::new(transformer_block.feed_forward)),
                LayerEnum::LayerNorm(transformer_block.norm2),
                LayerEnum::OutputProjection(output_projection),
            ],
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

    #[instrument(skip(self))]
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

    #[instrument(skip(self))]
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
    pub fn train(&mut self, data: Vec<&str>, epochs: usize, lr: f32) {
        self.train_with_batch_size(data, epochs, lr, 1);
    }

    /// Train with configurable batch size for improved performance
    pub fn train_with_batch_size(
        &mut self,
        data: Vec<&str>,
        epochs: usize,
        lr: f32,
        batch_size: usize,
    ) {
        let tokenized_data = data
            .par_iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            // Process data in batches
            for batch in tokenized_data.chunks(batch_size) {
                let batch_loss = self.train_batch(batch, lr);
                total_loss += batch_loss;
            }

            println!(
                "Epoch {}: Loss = {:.4}",
                epoch,
                total_loss / tokenized_data.len() as f32
            );
            info!(
                epoch = epoch,
                loss = total_loss / tokenized_data.len() as f32,
                "Training epoch completed"
            );
        }
    }

    /// Train on a single batch of sequences
    fn train_batch(&mut self, batch: &[Vec<usize>], lr: f32) -> f32 {
        let mut batch_loss = 0.0;
        let mut accumulated_param_grads: Vec<Vec<Array2<f32>>> = Vec::new();

        // Initialize accumulated gradients for each layer
        for _ in &self.network {
            accumulated_param_grads.push(Vec::new());
        }

        // Process each sequence in the batch
        for training_row in batch {
            if training_row.len() < 2 {
                continue;
            }

            // 1. Slice input and targets
            let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
            let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab. 

            // Forward pass
            let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
            input
                .row_mut(0)
                .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;
            let probs = Self::softmax(&logits);

            batch_loss += Self::cross_entropy_loss_step(&probs, target_ids);

            // Compute gradients w.r.t. logits
            let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

            // Apply gradient clipping
            if let Some(ref mut clipper) = self.gradient_clipper {
                clipper.clip_gradients(&mut grads_output);
            }

            // Backward pass: compute parameter gradients for each layer
            for (rev_idx, layer) in self.network.iter().rev().enumerate() {
                let (input_grads, param_grads) =
                    layer.compute_gradients(&Array2::zeros((0, 0)), &grads_output);
                grads_output = input_grads;

                // Accumulate parameter gradients (correct index for reversed iteration)
                let layer_idx = self.network.len() - 1 - rev_idx;
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

        // Apply accumulated and averaged gradients
        for (layer, param_grads) in self.network.iter_mut().zip(accumulated_param_grads) {
            if !param_grads.is_empty() {
                // Average gradients across batch
                let averaged_grads: Vec<Array2<f32>> = param_grads
                    .into_iter()
                    .map(|grad| grad / batch.len() as f32)
                    .collect();
                layer.apply_gradients(&averaged_grads, lr);
            }
        }

        batch_loss
    }
    /// Configure gradient clipping strategy
    pub fn set_gradient_clipping(&mut self, clipper: Box<dyn GradientClipping>) {
        self.gradient_clipper = Some(clipper);
    }

    /// Disable gradient clipping
    pub fn disable_gradient_clipping(&mut self) {
        self.gradient_clipper = None;
    }

    #[instrument]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Split by whitespace first
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
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

        if probs.shape()[0] != target.len() {
            panic!("Probs and target must have the same number of rows");
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
        let encoded = bincode::serde::encode_to_vec(self, config).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        fs::write(path, encoded).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load model from binary format
    pub fn load_binary(path: &str) -> Result<Self> {
        let data = fs::read(path).map_err(ModelError::from)?;
        let config = bincode::config::standard();
        let (llm, _): (LLM, usize) = bincode::serde::decode_from_slice(&data, config).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
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
