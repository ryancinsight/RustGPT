use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    channel_mixing::ChannelMixingMLP,
    layer_norm::LayerNorm,
    llm::Layer,
    token_mixing::TokenMixingMLP,
};

/// HyperMixer Block
///
/// A complete HyperMixer block that combines token mixing and channel mixing
/// with layer normalization and residual connections. This is analogous to
/// a Transformer block but uses dynamic token mixing instead of attention.
///
/// Architecture (Pre-Norm):
/// ```text
/// Input
///   ↓
/// LayerNorm
///   ↓
/// TokenMixing (with residual connection inside)
///   ↓
/// LayerNorm
///   ↓
/// ChannelMixing (with residual connection inside)
///   ↓
/// Output
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HyperMixerBlock {
    /// Token mixing layer (replaces attention)
    pub token_mixing: TokenMixingMLP,
    /// Channel mixing layer (similar to feedforward)
    pub channel_mixing: ChannelMixingMLP,
    /// Layer norm after token mixing
    pub norm1: LayerNorm,
    /// Layer norm after channel mixing
    pub norm2: LayerNorm,
}

impl HyperMixerBlock {
    /// Create a new HyperMixer block
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of token embeddings
    /// * `hidden_dim` - Hidden dimension for channel mixing
    /// * `max_seq_len` - Maximum sequence length
    /// * `hypernetwork_hidden_dim` - Hidden dimension for the hypernetwork
    /// * `num_heads` - Number of attention heads for token mixing (default: 8)
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
        num_heads: usize,
    ) -> Self {
        // Token mixing hidden dim can be different from channel mixing
        // Using embedding_dim / 2 as a reasonable default for token mixing
        let token_mixing_hidden_dim = embedding_dim / 2;
        
        Self {
            token_mixing: TokenMixingMLP::new(
                embedding_dim,
                token_mixing_hidden_dim,
                max_seq_len,
                hypernetwork_hidden_dim,
                num_heads,
            ),
            channel_mixing: ChannelMixingMLP::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl HyperMixerBlock {
    /// Check for gradient instability (NaN/inf values)
    pub fn check_gradient_stability(&self, grads: &Array2<f32>) -> bool {
        !grads.iter().any(|&x| !x.is_finite())
    }

}

impl Layer for HyperMixerBlock {
    fn layer_type(&self) -> &str {
        "HyperMixerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Token mixing path: norm → token_mixing (includes residual inside)
        let norm1_out = self.norm1.normalize(input);
        let token_mixed = self.token_mixing.forward(&norm1_out);

        // Channel mixing path: norm → channel_mixing (includes residual inside)
        let norm2_out = self.norm2.normalize(&token_mixed);
        self.channel_mixing.forward(&norm2_out)
    }
    
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Backward pass matching the forward order: norm1 → token_mixing → norm2 → channel_mixing
        // Reverse: channel_mixing → norm2 → token_mixing → norm1

        // Backward through channel mixing (includes residual connection)
        let (grad_channel, channel_param_grads) = self
            .channel_mixing
            .compute_gradients(&Array2::zeros((0, 0)), output_grads);

        // Backward through second LayerNorm
        let (grad_norm2, norm2_param_grads) = self
            .norm2
            .compute_gradients(&Array2::zeros((0, 0)), &grad_channel);

        // Backward through token mixing (includes residual connection)
        let (grad_token, token_param_grads) = self
            .token_mixing
            .compute_gradients(&Array2::zeros((0, 0)), &grad_norm2);

        // Backward through first LayerNorm
        let (grad_input, norm1_param_grads) = self
            .norm1
            .compute_gradients(&Array2::zeros((0, 0)), &grad_token);

        // Collect all parameter gradients in order
        let mut all_param_grads = Vec::new();
        all_param_grads.extend(norm1_param_grads);
        all_param_grads.extend(token_param_grads);
        all_param_grads.extend(norm2_param_grads);
        all_param_grads.extend(channel_param_grads);

        (grad_input, all_param_grads)
    }
    
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        let mut idx = 0;

        // Apply gradients in the order they were collected:
        // norm1 → token_mixing → norm2 → channel_mixing

        // Apply norm1 gradients (2 params: gamma, beta)
        if idx + 2 <= param_grads.len() {
            let norm1_params = &param_grads[idx..idx + 2];
            self.norm1.apply_gradients(norm1_params, lr);
            idx += 2;
        }

        // Apply token mixing gradients (4 params per head: w1, b1, w2, b2)
        // Token mixing returns 4 * num_heads gradients
        let token_mixing_grads = 4 * self.token_mixing.num_heads();
        if idx + token_mixing_grads <= param_grads.len() {
            let token_params = &param_grads[idx..idx + token_mixing_grads];
            self.token_mixing.apply_gradients(token_params, lr);
            idx += token_mixing_grads;
        }

        // Apply norm2 gradients (2 params: gamma, beta)
        if idx + 2 <= param_grads.len() {
            let norm2_params = &param_grads[idx..idx + 2];
            self.norm2.apply_gradients(norm2_params, lr);
            idx += 2;
        }

        // Apply channel mixing gradients (4 params: w1, b1, w2, b2)
        if idx + 4 <= param_grads.len() {
            let channel_params = &param_grads[idx..idx + 4];
            self.channel_mixing.apply_gradients(channel_params, lr);
        }
    }
    
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // Backward pass in reverse order of forward pass
        // Forward: norm1 → token_mixing → norm2 → channel_mixing
        // Backward: channel_mixing → norm2 → token_mixing → norm1

        // Backward through channel mixing (includes residual)
        let grad_channel = self.channel_mixing.backward(grads, lr);

        // Backward through norm2
        let grad_norm2 = self.norm2.backward(&grad_channel, lr);

        // Backward through token mixing (includes residual)
        let grad_token = self.token_mixing.backward(&grad_norm2, lr);

        // Backward through norm1
        self.norm1.backward(&grad_token, lr)
    }
    
    fn parameters(&self) -> usize {
        self.token_mixing.parameters()
            + self.channel_mixing.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}

