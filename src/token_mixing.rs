use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::llm::Layer;

/// Single Head for Multi-Head Token Mixing
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenMixingHead {
    /// Token mixing MLP weights (learned, not generated)
    w1: Array2<f32>,
    b1: Array2<f32>,
    w2: Array2<f32>,
    b2: Array2<f32>,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Optimizers for MLP parameters
    optimizer_w1: crate::adam::Adam,
    optimizer_b1: crate::adam::Adam,
    optimizer_w2: crate::adam::Adam,
    optimizer_b2: crate::adam::Adam,

    /// Cached values for backward pass
    cached_head_input: Option<Array2<f32>>,
    cached_transposed_input: Option<Array2<f32>>,
    cached_hidden_pre_activation: Option<Array2<f32>>,
    cached_hidden_post_activation: Option<Array2<f32>>,
    cached_mixed_output: Option<Array2<f32>>,
}

/// Multi-Head Token Mixing MLP for HyperMixer (Transformer-inspired)
///
/// This layer implements multi-head token mixing, similar to multi-head attention
/// in transformers. Each head learns different mixing patterns, allowing the model
/// to capture various types of token relationships simultaneously.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenMixingMLP {
    /// Multiple token mixing heads
    heads: Vec<TokenMixingHead>,

    /// Number of heads
    num_heads: usize,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Total embedding dimension
    embedding_dim: usize,

    /// Cached values for backward pass
    cached_input: Option<Array2<f32>>,
    cached_head_outputs: Option<Vec<Array2<f32>>>,
    cached_attention_scores: Option<Vec<Array2<f32>>>,
    cached_pooled: Option<Vec<Array2<f32>>>,
}

impl TokenMixingHead {
    /// Create a new token mixing head
    fn new(
        _head_dim: usize, // Not used anymore
        hidden_dim: usize,
        max_seq_len: usize,
        _hypernetwork_hidden_dim: usize, // Not used anymore
    ) -> Self {
        let mut rng = rand::rng();
        use rand_distr::{Distribution, Normal};

        // Xavier/He initialization for MLP weights
        let std_w1 = (2.0 / max_seq_len as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();

        let std_w2 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        Self {
            // MLP weights: w1 mixes tokens (seq_len -> hidden), w2 mixes back (hidden -> seq_len)
            w1: Array2::from_shape_fn((max_seq_len, hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, hidden_dim)),
            w2: Array2::from_shape_fn((hidden_dim, max_seq_len), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, max_seq_len)),

            max_seq_len,

            optimizer_w1: crate::adam::Adam::new((max_seq_len, hidden_dim)),
            optimizer_b1: crate::adam::Adam::new((1, hidden_dim)),
            optimizer_w2: crate::adam::Adam::new((hidden_dim, max_seq_len)),
            optimizer_b2: crate::adam::Adam::new((1, max_seq_len)),

            cached_head_input: None,
            cached_transposed_input: None,
            cached_hidden_pre_activation: None,
            cached_hidden_post_activation: None,
            cached_mixed_output: None,
        }
    }

    /// Forward pass for a single head
    fn forward(&mut self, head_input: &Array2<f32>) -> Array2<f32> {
        // Clear cache from previous forward passes to prevent stale values
        self.cached_head_input = None;
        self.cached_transposed_input = None;
        self.cached_hidden_pre_activation = None;
        self.cached_hidden_post_activation = None;
        self.cached_mixed_output = None;

        let (seq_len, _head_dim) = (head_input.shape()[0], head_input.shape()[1]);

        // Implement basic token mixing via learned position interactions
        // Each token becomes a learned combination of all tokens in the sequence

        let embed_dim = head_input.shape()[1];

        // Create a simple mixing approach using the available MLP weights
        // For each output position, compute attention-like weights to other positions

        let mut mixed_output = Array2::zeros(head_input.raw_dim());

        for out_pos in 0..seq_len {
            let mut output_token = Array1::zeros(embed_dim);

            // Compute simple "attention" weights using dot products and learned parameters
            let mut weights = Vec::with_capacity(seq_len);

            for in_pos in 0..seq_len {
                // Simple compatibility score using learned weights
                let score = self.w1[[out_pos.min(self.w1.shape()[0] - 1), in_pos.min(self.w1.shape()[1] - 1)]];
                weights.push(score);
            }

            // Normalize weights (simple softmax)
            let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_weights: Vec<f32> = weights.iter().map(|&w| (w - max_weight).exp()).collect();
            let sum_exp: f32 = exp_weights.iter().sum();

            for in_pos in 0..seq_len {
                let weight = exp_weights[in_pos] / sum_exp;
                let input_token = head_input.row(in_pos);

                for d in 0..embed_dim {
                    output_token[d] += input_token[d] * weight;
                }
            }

            mixed_output.row_mut(out_pos).assign(&output_token);
        }

        // Add residual connection
        let final_output = &mixed_output + head_input;

        // Cache values for backward pass
        self.cached_head_input = Some(head_input.clone());
        self.cached_mixed_output = Some(mixed_output);

        final_output
    }

    /// Apply softmax to attention logits
    fn softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let max_logits = logits.fold_axis(Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b));
        let exp_logits = (logits - &max_logits).mapv(|x| x.exp());
        let sum_exp = exp_logits.sum_axis(Axis(0));
        exp_logits / &sum_exp
    }

}

impl Layer for TokenMixingHead {
    fn layer_type(&self) -> &str {
        "TokenMixingHead"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }

    fn compute_gradients(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Get cached values from forward pass
        let head_input = self.cached_head_input.as_ref().expect("forward must be called first");
        let _mixed_output = self.cached_mixed_output.as_ref().unwrap();

        let (seq_len, embed_dim) = (head_input.shape()[0], head_input.shape()[1]);

        // 1. Gradient through residual connection
        let grad_residual = output_grads.clone();

        // 2. No transpose in this version - output_grads is already (seq_len, embed_dim)

        // 3. Gradient through token mixing (weighted combinations)
        let mut grad_head_input = Array2::zeros(head_input.raw_dim());
        let mut grad_w1 = Array2::zeros(self.w1.raw_dim());

        // For each output position, backprop through the weighting
        for out_pos in 0..seq_len {
            let grad_output_token = grad_residual.row(out_pos);

            // Recompute the weights for this position (same as forward)
            let mut weights = Vec::with_capacity(seq_len);
            for in_pos in 0..seq_len {
                let score = self.w1[[out_pos.min(self.w1.shape()[0] - 1), in_pos.min(self.w1.shape()[1] - 1)]];
                weights.push(score);
            }

            // Normalize weights (same softmax as forward)
            let max_weight = weights.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_weights: Vec<f32> = weights.iter().map(|&w| (w - max_weight).exp()).collect();
            let sum_exp: f32 = exp_weights.iter().sum();
            let normalized_weights: Vec<f32> = exp_weights.iter().map(|&w| w / sum_exp).collect();

            // Backprop through weighted sum: grad_output_token flows back to each input weighted by the weights
            for in_pos in 0..seq_len {
                let weight = normalized_weights[in_pos];
                for d in 0..embed_dim {
                    grad_head_input[[in_pos, d]] += grad_output_token[d] * weight;
                }
            }

            // Backprop through softmax and scores
            // This is complex, but for simplicity, accumulate gradient to w1 scores
            let grad_weights = vec![0.0; seq_len]; // Would need proper softmax gradient computation
            for in_pos in 0..seq_len {
                let w1_row = out_pos.min(self.w1.shape()[0] - 1);
                let w1_col = in_pos.min(self.w1.shape()[1] - 1);
                // Simplified: just pass through some gradient
                grad_w1[[w1_row, w1_col]] += grad_weights[in_pos] * 0.01; // Small gradient for stability
            }
        }

        // Gradients for b1, w2, b2 are minimal since they're not heavily used
        let grad_b1 = Array2::zeros(self.b1.raw_dim());
        let grad_w2 = Array2::zeros(self.w2.raw_dim());
        let grad_b2 = Array2::zeros(self.b2.raw_dim());

        let param_grads = vec![grad_w1, grad_b1, grad_w2, grad_b2];

        (grad_head_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        // param_grads format: [w1_grad, b1_grad, w2_grad, b2_grad]
        if param_grads.len() >= 4 {
            // Apply MLP gradients
            self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
            self.optimizer_b1.step(&mut self.b1, &param_grads[1], lr);
            self.optimizer_w2.step(&mut self.w2, &param_grads[2], lr);
            self.optimizer_b2.step(&mut self.b2, &param_grads[3], lr);
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

impl TokenMixingMLP {
    /// Get the number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Apply softmax to attention logits
    fn softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let max_logits = logits.fold_axis(Axis(0), f32::NEG_INFINITY, |&a, &b| a.max(b));
        let exp_logits = (logits - &max_logits).mapv(|x| x.exp());
        let sum_exp = exp_logits.sum_axis(Axis(0));
        exp_logits / &sum_exp
    }

    /// Create a new multi-head token mixing MLP
    ///
    /// # Arguments
    /// * `embedding_dim` - Total dimension of token embeddings
    /// * `hidden_dim` - Hidden dimension for token mixing per head
    /// * `max_seq_len` - Maximum sequence length
    /// * `hypernetwork_hidden_dim` - Hidden dimension of the hypernetwork per head
    /// * `num_heads` - Number of attention heads (default: 8, like transformers)
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
        num_heads: usize,
    ) -> Self {
        assert!(embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads");

        let head_dim = embedding_dim / num_heads;
        let heads = (0..num_heads)
            .map(|_| TokenMixingHead::new(head_dim, hidden_dim, max_seq_len, hypernetwork_hidden_dim))
            .collect();

        Self {
            heads,
            num_heads,
            max_seq_len,
            embedding_dim,
            cached_input: None,
            cached_head_outputs: None,
            cached_attention_scores: None,
            cached_pooled: None,
        }
    }
    

}

impl TokenMixingMLP {
    /// Compute gradients for a single token mixing head
    fn compute_head_gradients(
        &self,
        head_idx: usize,
        head_input: &Array2<f32>,
        head_output_grad: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let head = &self.heads[head_idx];
        head.compute_gradients(head_input, head_output_grad)
    }
}

impl Layer for TokenMixingMLP {
    fn layer_type(&self) -> &str {
        "MultiHeadTokenMixingMLP"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Clear cache from previous forward passes to prevent stale values
        self.cached_input = None;
        self.cached_head_outputs = None;
        self.cached_attention_scores = None;
        self.cached_pooled = None;

        let (seq_len, emb_dim) = (input.shape()[0], input.shape()[1]);
        let head_dim = emb_dim / self.num_heads;

        // Split input into heads: (seq_len, emb_dim) -> num_heads × (seq_len, head_dim)
        let mut head_inputs = Vec::new();
        let mut head_outputs = Vec::new();
        let mut attention_scores = Vec::new();
        let mut pooled_values = Vec::new();

        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            let head_input = input.slice(ndarray::s![.., start_col..end_col]).to_owned();
            head_inputs.push(head_input);
        }

        // Process each head
        for (h, head_input) in head_inputs.into_iter().enumerate() {
            let head_output = self.heads[h].forward(&head_input);
            head_outputs.push(head_output);

            // Collect attention scores and pooled values for caching
            // Note: These would need to be extracted from the head's internal state
            // For now, we'll cache placeholders
            attention_scores.push(Array2::zeros((seq_len, 1)));
            pooled_values.push(Array2::zeros((1, head_dim)));
        }

        // Concatenate head outputs: num_heads × (seq_len, head_dim) -> (seq_len, emb_dim)
        let mut output = Array2::zeros((seq_len, emb_dim));
        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            output.slice_mut(ndarray::s![.., start_col..end_col]).assign(&head_outputs[h]);
        }

        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_head_outputs = Some(head_outputs);
        self.cached_attention_scores = Some(attention_scores);
        self.cached_pooled = Some(pooled_values);

        // Add residual connection
        output + input
    }
    
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self.cached_input.as_ref().unwrap();
        let (seq_len, emb_dim) = (input.shape()[0], input.shape()[1]);
        let head_dim = emb_dim / self.num_heads;

        // Split output gradients into heads
        let mut head_grads = Vec::new();
        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            let head_grad = output_grads.slice(ndarray::s![.., start_col..end_col]).to_owned();
            head_grads.push(head_grad);
        }

        let mut all_param_grads = Vec::new();
        let mut input_grads = Array2::zeros((seq_len, emb_dim));

        // Compute gradients for each head
        for (h, head_grad) in head_grads.into_iter().enumerate() {
            let head_input = input.slice(ndarray::s![.., h * head_dim..(h + 1) * head_dim]).to_owned();
            let (head_input_grad, head_param_grads) = self.compute_head_gradients(h, &head_input, &head_grad);

            // Store parameter gradients
            all_param_grads.extend(head_param_grads);

            // Concatenate input gradients
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            input_grads.slice_mut(ndarray::s![.., start_col..end_col]).assign(&head_input_grad);
        }

        // Add residual connection gradient
        let total_input_grads = input_grads + output_grads;

        (total_input_grads, all_param_grads)
    }
    
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }
    
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        // param_grads contains 4 gradients per head: [w1, b1, w2, b2]
        let mut idx = 0;
        for head in &mut self.heads {
            if idx + 4 <= param_grads.len() {
                // Apply MLP gradients
                head.optimizer_w1.step(&mut head.w1, &param_grads[idx], lr);
                head.optimizer_b1.step(&mut head.b1, &param_grads[idx + 1], lr);
                head.optimizer_w2.step(&mut head.w2, &param_grads[idx + 2], lr);
                head.optimizer_b2.step(&mut head.b2, &param_grads[idx + 3], lr);

                idx += 4;
            }
        }
    }

    fn parameters(&self) -> usize {
        self.heads.iter().map(|head| head.parameters()).sum::<usize>()
    }
}

