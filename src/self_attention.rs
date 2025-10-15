use std::f32;

use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::EMBEDDING_DIM;
use crate::adam::Adam;
use crate::llm::Layer;

/// Single head for multi-head attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionHead {
    pub head_dim: usize,
    w_q: Array2<f32>, // Weight matrices for this head's Q, K, V
    w_k: Array2<f32>,
    w_v: Array2<f32>,

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
}

/// Multi-head self-attention mechanism (standard transformer attention)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SelfAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    heads: Vec<AttentionHead>,

    cached_input: Option<Array2<f32>>,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM)
    }
}

impl AttentionHead {
    /// Create a new attention head
    fn new(head_dim: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / head_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        AttentionHead {
            head_dim,
            w_q: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            optimizer_w_q: Adam::new((head_dim, head_dim)),
            optimizer_w_k: Adam::new((head_dim, head_dim)),
            optimizer_w_v: Adam::new((head_dim, head_dim)),
        }
    }

    /// Compute Q, K, V for this head
    fn compute_qkv(&self, head_input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = head_input.dot(&self.w_q); // Q = X * W_Q
        let k = head_input.dot(&self.w_k); // K = X * W_K
        let v = head_input.dot(&self.w_v); // V = X * W_V
        (q, k, v)
    }

    /// Compute attention for this head
    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let dk = (self.head_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Apply causal masking - prevent attention to future tokens
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let weights = self.softmax(&scores);
        weights.dot(v)
    }

    /// Apply softmax to attention scores
    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }
}

impl SelfAttention {
    /// Initializes multi-head self-attention
    /// num_heads defaults to 8 (standard transformer)
    pub fn new(embedding_dim: usize) -> Self {
        Self::new_with_heads(embedding_dim, 8)
    }

    /// Initialize with specific number of heads
    pub fn new_with_heads(embedding_dim: usize, num_heads: usize) -> Self {
        assert!(embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads");

        let head_dim = embedding_dim / num_heads;
        let heads = (0..num_heads)
            .map(|_| AttentionHead::new(head_dim))
            .collect();

        SelfAttention {
            embedding_dim,
            num_heads,
            heads,
            cached_input: None,
        }
    }

}

impl SelfAttention {
    /// Compute gradients for a single attention head
    fn compute_head_gradients(
        &self,
        head_idx: usize,
        head_input: &Array2<f32>,
        head_output_grad: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let head = &self.heads[head_idx];
        let head_dim = head.head_dim;

        // Recompute forward pass
        let q = head_input.dot(&head.w_q);
        let k = head_input.dot(&head.w_k);
        let v = head_input.dot(&head.w_v);

        let dk = head_dim as f32;
        let mut scores = q.dot(&k.t()) / dk.sqrt();

        // Apply causal masking
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let attn_weights = head.softmax(&scores);

        // Backward pass
        let grad_attn_weights = head_output_grad.dot(&v.t());
        let grad_v = attn_weights.t().dot(head_output_grad);

        // Softmax backward
        let grad_scores = Self::softmax_backward(&attn_weights, &grad_attn_weights);

        // Q, K gradients
        let grad_q = grad_scores.dot(&k) / dk.sqrt();
        let grad_k = grad_scores.t().dot(&q) / dk.sqrt();

        // Weight gradients
        let grad_w_q = head_input.t().dot(&grad_q);
        let grad_w_k = head_input.t().dot(&grad_k);
        let grad_w_v = head_input.t().dot(&grad_v);

        // Input gradients
        let grad_input = grad_q.dot(&head.w_q.t()) + grad_k.dot(&head.w_k.t()) + grad_v.dot(&head.w_v.t());

        (grad_input, vec![grad_w_q, grad_w_k, grad_w_v])
    }

    /// Softmax backward pass for attention
    fn softmax_backward(
        softmax_output: &Array2<f32>,
        grad_output: &Array2<f32>,
    ) -> Array2<f32> {
        let mut grad_input = Array2::zeros(softmax_output.dim());

        for ((mut grad_row, softmax_row), grad_out_row) in grad_input
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            // dot product: y ⊙ dL/dy
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();

            for ((g, &y_i), &dy_i) in grad_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }

        grad_input
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadSelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());

        let (seq_len, emb_dim) = (input.shape()[0], input.shape()[1]);
        let head_dim = emb_dim / self.num_heads;

        // Split input into heads: (seq_len, emb_dim) -> num_heads × (seq_len, head_dim)
        let mut head_inputs = Vec::new();
        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            let head_input = input.slice(ndarray::s![.., start_col..end_col]).to_owned();
            head_inputs.push(head_input);
        }

        // Process each head
        let mut head_outputs = Vec::new();
        for (h, head_input) in head_inputs.into_iter().enumerate() {
            let qkv = self.heads[h].compute_qkv(&head_input);
            let head_attention = self.heads[h].attention(&qkv.0, &qkv.1, &qkv.2);
            head_outputs.push(head_attention);
        }

        // Concatenate head outputs: num_heads × (seq_len, head_dim) -> (seq_len, emb_dim)
        let mut output = Array2::zeros((seq_len, emb_dim));
        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;
            output.slice_mut(ndarray::s![.., start_col..end_col]).assign(&head_outputs[h]);
        }

        output + input // residual connection
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.heads.iter().map(|head| head.w_q.len() + head.w_k.len() + head.w_v.len()).sum::<usize>()
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

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        // Apply gradients to each head's parameters
        // param_grads should contain 3 gradients per head: [grad_w_q, grad_w_k, grad_w_v]
        let mut idx = 0;
        for head in &mut self.heads {
            if idx + 3 <= param_grads.len() {
                head.optimizer_w_q.step(&mut head.w_q, &param_grads[idx], lr);
                head.optimizer_w_k.step(&mut head.w_k, &param_grads[idx + 1], lr);
                head.optimizer_w_v.step(&mut head.w_v, &param_grads[idx + 2], lr);
                idx += 3;
            }
        }
    }
}
