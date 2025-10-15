use ndarray::{Array2, Axis};
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

    /// Attention-like pooling weights (transformer-inspired)
    /// Shape: (head_dim, 1) - learns which dimensions to weight for pooling
    pooling_weights: Array2<f32>,

    /// Pooling bias
    pooling_bias: Array2<f32>,

    /// Head dimension (embedding_dim / num_heads)
    head_dim: usize,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Optimizers for MLP parameters
    optimizer_w1: crate::adam::Adam,
    optimizer_b1: crate::adam::Adam,
    optimizer_w2: crate::adam::Adam,
    optimizer_b2: crate::adam::Adam,

    /// Optimizers for pooling parameters
    optimizer_pooling_weights: crate::adam::Adam,
    optimizer_pooling_bias: crate::adam::Adam,

    /// Cached values for backward pass
    cached_head_input: Option<Array2<f32>>,
    cached_attention_logits: Option<Array2<f32>>,
    cached_attention_weights: Option<Array2<f32>>,
    cached_pooled: Option<Array2<f32>>,
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
        head_dim: usize,
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

        // Initialize pooling weights (transformer-inspired attention-like pooling)
        let std_pool = (2.0 / head_dim as f32).sqrt();
        let normal_pool = Normal::new(0.0, std_pool).unwrap();

        Self {
            // MLP weights: w1 mixes tokens (seq_len -> hidden), w2 mixes back (hidden -> seq_len)
            w1: Array2::from_shape_fn((max_seq_len, hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, hidden_dim)),
            w2: Array2::from_shape_fn((hidden_dim, max_seq_len), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, max_seq_len)),

            pooling_weights: Array2::from_shape_fn((head_dim, 1), |_| normal_pool.sample(&mut rng)),
            pooling_bias: Array2::zeros((1, 1)),

            head_dim,
            max_seq_len,

            optimizer_w1: crate::adam::Adam::new((max_seq_len, hidden_dim)),
            optimizer_b1: crate::adam::Adam::new((1, hidden_dim)),
            optimizer_w2: crate::adam::Adam::new((hidden_dim, max_seq_len)),
            optimizer_b2: crate::adam::Adam::new((1, max_seq_len)),

            optimizer_pooling_weights: crate::adam::Adam::new((head_dim, 1)),
            optimizer_pooling_bias: crate::adam::Adam::new((1, 1)),

            cached_head_input: None,
            cached_attention_logits: None,
            cached_attention_weights: None,
            cached_pooled: None,
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
        self.cached_attention_logits = None;
        self.cached_attention_weights = None;
        self.cached_pooled = None;
        self.cached_transposed_input = None;
        self.cached_hidden_pre_activation = None;
        self.cached_hidden_post_activation = None;
        self.cached_mixed_output = None;

        let (seq_len, _head_dim) = (head_input.shape()[0], head_input.shape()[1]);

        // 1. Attention-like pooling (transformer-inspired)
        let attention_logits = head_input.dot(&self.pooling_weights) + &self.pooling_bias;
        let attention_weights = self.softmax(&attention_logits);
        let pooled = (head_input.t().to_owned() * &attention_weights.t()).sum_axis(Axis(1)).insert_axis(Axis(1)).t().to_owned();

        // 2. Use fixed MLP weights (slice to current sequence length)
        let w1 = self.w1.slice(ndarray::s![0..seq_len, ..]).to_owned();
        let w2 = self.w2.slice(ndarray::s![.., 0..seq_len]).to_owned();
        let b2 = self.b2.slice(ndarray::s![.., 0..seq_len]).to_owned();

        // 3. Transpose input to mix across tokens
        let transposed_input = head_input.t().to_owned();

        // 4. Apply token mixing MLP across sequence dimension (batched)
        // transposed_input: (head_dim, seq_len)
        // w1: (seq_len, hidden_dim)
        // Result: (head_dim, hidden_dim)
        let hidden_pre_activation = transposed_input.dot(&w1) + &self.b1;
        let hidden_post_activation = hidden_pre_activation.mapv(|x| x.max(0.0));

        // hidden_post_activation: (head_dim, hidden_dim)
        // w2: (hidden_dim, seq_len)
        // Result: (head_dim, seq_len)
        let mixed_output = hidden_post_activation.dot(&w2) + &b2;

        // 5. Transpose back
        let output = mixed_output.t().to_owned();

        // 6. Add residual connection
        let final_output = &output + head_input;

        // Cache values for backward pass
        self.cached_head_input = Some(head_input.clone());
        self.cached_attention_logits = Some(attention_logits);
        self.cached_attention_weights = Some(attention_weights);
        self.cached_pooled = Some(pooled);
        self.cached_transposed_input = Some(transposed_input);
        self.cached_hidden_pre_activation = Some(hidden_pre_activation);
        self.cached_hidden_post_activation = Some(hidden_post_activation);
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
        let _attention_logits = self.cached_attention_logits.as_ref().unwrap();
        let attention_weights = self.cached_attention_weights.as_ref().unwrap();
        let _pooled = self.cached_pooled.as_ref().unwrap();
        let transposed_input = self.cached_transposed_input.as_ref().unwrap();
        let hidden_pre_activation = self.cached_hidden_pre_activation.as_ref().unwrap();
        let hidden_post_activation = self.cached_hidden_post_activation.as_ref().unwrap();
        let _mixed_output = self.cached_mixed_output.as_ref().unwrap();

        let (seq_len, head_dim) = (head_input.shape()[0], head_input.shape()[1]);

        // 1. Gradient through residual connection
        let grad_residual = output_grads.clone();

        // 2. Gradient through transpose back (mixed_output.t() -> mixed_output)
        let grad_mixed_output = grad_residual.t().to_owned();

        // 3. Gradient through token mixing MLP (batched)
        // Get the sliced weights used in forward pass
        let w1_sliced = self.w1.slice(ndarray::s![0..seq_len, ..]);
        let w2_sliced = self.w2.slice(ndarray::s![.., 0..seq_len]);
        let b2_sliced = self.b2.slice(ndarray::s![.., 0..seq_len]);

        // grad_mixed_output: (head_dim, seq_len)
        // w2_sliced: (hidden_dim, seq_len), so w2_sliced.t(): (seq_len, hidden_dim)
        // grad_hidden_post = grad_mixed_output.dot(w2_sliced.t()) : (head_dim, seq_len) x (seq_len, hidden_dim) = (head_dim, hidden_dim)
        let grad_hidden_post = grad_mixed_output.dot(&w2_sliced.t().to_owned());

        // Gradient through b2 addition: grad_mixed_output is already the gradient after b2
        // So grad_b2 = sum over head_dim dimension: (head_dim, seq_len) -> (1, seq_len)
        let grad_b2 = grad_mixed_output.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient through w2: hidden_post_activation.t().dot(grad_mixed_output)
        // hidden_post_activation: (head_dim, hidden_dim), grad_mixed_output: (head_dim, seq_len)
        // Result: (hidden_dim, seq_len)
        let grad_w2_full = hidden_post_activation.t().to_owned().dot(&grad_mixed_output);

        // Gradient through ReLU: element-wise multiplication with mask
        let relu_mask = hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad_hidden_pre = &grad_hidden_post * &relu_mask;

        // Gradient through b1: sum over head_dim dimension: (head_dim, hidden_dim) -> (1, hidden_dim)
        let grad_b1 = grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient through w1: transposed_input.t().dot(grad_hidden_pre)
        // transposed_input: (head_dim, seq_len), grad_hidden_pre: (head_dim, hidden_dim)
        // Result: (seq_len, hidden_dim)
        let grad_w1_full = transposed_input.t().to_owned().dot(&grad_hidden_pre);

        // Accumulate gradients into full-sized weight matrices
        let mut grad_w1_full_matrix = Array2::zeros(self.w1.raw_dim());
        let mut grad_w2_full_matrix = Array2::zeros(self.w2.raw_dim());
        let mut grad_b2_full_matrix = Array2::zeros(self.b2.raw_dim());

        // Fill in the gradients for the parts that were actually used
        grad_w1_full_matrix.slice_mut(ndarray::s![0..seq_len, ..]).assign(&grad_w1_full);
        grad_w2_full_matrix.slice_mut(ndarray::s![.., 0..seq_len]).assign(&grad_w2_full);
        grad_b2_full_matrix.slice_mut(ndarray::s![.., 0..seq_len]).assign(&grad_b2);

        let grad_w1 = grad_w1_full_matrix;
        let grad_w2 = grad_w2_full_matrix;
        let grad_b2_full = grad_b2_full_matrix;

        // 4. Gradient through transpose (transposed_input = head_input.t())
        // grad_hidden_pre: (head_dim, hidden_dim)
        // w1_sliced: (seq_len, hidden_dim), so w1_sliced.t(): (hidden_dim, seq_len)
        // grad_transposed_input = grad_hidden_pre.dot(w1_sliced.t()) : (head_dim, hidden_dim) x (hidden_dim, seq_len) = (head_dim, seq_len)
        let grad_transposed_input = grad_hidden_pre.dot(&w1_sliced.t().to_owned());
        let grad_head_input_from_mlp = grad_transposed_input.t().to_owned();

        // 5. Accumulate gradients for fixed MLP weights
        // grad_w1 and grad_w2 are already computed above
        // We need to accumulate them into the full-sized weight matrices

        // 6. Gradient through attention pooling
        // Since we're not using pooled anymore (no hypernetwork), pooling is just for show
        // We can simplify this - pooling doesn't affect the output anymore
        let grad_attention_weights = Array2::zeros((seq_len, 1)); // No gradient flows back through pooling

        // Gradient through softmax (not used, but keeping for consistency)
        let softmax_derivative = attention_weights.clone() * (attention_weights.clone() * -1.0 + 1.0);
        let grad_attention_logits = grad_attention_weights.clone() * softmax_derivative;

        // Gradient through pooling weights and bias (minimal impact since not used)
        let grad_pooling_weights = head_input.t().dot(&grad_attention_logits);
        let grad_pooling_bias = grad_attention_logits.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient from pooling back to input (zero since pooling doesn't affect output)
        let grad_head_input_from_pooling = grad_attention_logits.dot(&self.pooling_weights.t());

        // 7. Combine gradients from MLP and pooling
        let grad_head_input = grad_head_input_from_mlp + grad_head_input_from_pooling;

        // Return input gradients and parameter gradients (MLP weights + pooling)
        let param_grads = vec![
            grad_w1, grad_b1, grad_w2, grad_b2_full, // MLP parameters
            grad_pooling_weights, grad_pooling_bias // Pooling parameters (kept for compatibility)
        ];

        (grad_head_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        // param_grads format: [w1_grad, b1_grad, w2_grad, b2_grad, pooling_weights_grad, pooling_bias_grad]
        if param_grads.len() >= 6 {
            // Apply MLP gradients (first 4)
            self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
            self.optimizer_b1.step(&mut self.b1, &param_grads[1], lr);
            self.optimizer_w2.step(&mut self.w2, &param_grads[2], lr);
            self.optimizer_b2.step(&mut self.b2, &param_grads[3], lr);

            // Apply pooling gradients (last 2)
            self.optimizer_pooling_weights.step(&mut self.pooling_weights, &param_grads[4], lr);
            self.optimizer_pooling_bias.step(&mut self.pooling_bias, &param_grads[5], lr);
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
            + self.pooling_weights.len() + self.pooling_bias.len()
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
        // param_grads contains 6 gradients per head: [w1, b1, w2, b2, pooling_weights, pooling_bias]
        let mut idx = 0;
        for head in &mut self.heads {
            if idx + 6 <= param_grads.len() {
                // Apply MLP gradients (first 4)
                head.optimizer_w1.step(&mut head.w1, &param_grads[idx], lr);
                head.optimizer_b1.step(&mut head.b1, &param_grads[idx + 1], lr);
                head.optimizer_w2.step(&mut head.w2, &param_grads[idx + 2], lr);
                head.optimizer_b2.step(&mut head.b2, &param_grads[idx + 3], lr);

                // Apply pooling gradients (last 2)
                head.optimizer_pooling_weights.step(&mut head.pooling_weights, &param_grads[idx + 4], lr);
                head.optimizer_pooling_bias.step(&mut head.pooling_bias, &param_grads[idx + 5], lr);

                idx += 6;
            }
        }
    }

    fn parameters(&self) -> usize {
        self.heads.iter().map(|head| head.parameters()).sum::<usize>()
    }
}

