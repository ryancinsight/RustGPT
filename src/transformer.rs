use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    feed_forward::FeedForward, layer_norm::LayerNorm, llm::Layer, self_attention::SelfAttention,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub feed_forward: FeedForward,
    pub norm1: LayerNorm, // After attention
    pub norm2: LayerNorm, // After feed forward
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Standard Transformer architecture: attention + norm -> feedforward + norm
        let attention_out = self.attention.forward(input); // includes residual
        let norm1_out = self.norm1.normalize(&attention_out);

        let feed_forward_out = self.feed_forward.forward(&norm1_out); // includes residual

        self.norm2.normalize(&feed_forward_out)
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

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        let mut idx = 0;

        // Apply attention gradients (3 params: w_q, w_k, w_v)
        let attention_params = &param_grads[idx..idx + 3];
        self.attention.apply_gradients(attention_params, lr);
        idx += 3;

        // Apply norm1 gradients (2 params: gamma, beta)
        let norm1_params = &param_grads[idx..idx + 2];
        self.norm1.apply_gradients(norm1_params, lr);
        idx += 2;

        // Apply feed_forward gradients (4 params: w1, b1, w2, b2)
        let ffn_params = &param_grads[idx..idx + 4];
        self.feed_forward.apply_gradients(ffn_params, lr);
        idx += 4;

        // Apply norm2 gradients (2 params: gamma, beta)
        let norm2_params = &param_grads[idx..idx + 2];
        self.norm2.apply_gradients(norm2_params, lr);
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
