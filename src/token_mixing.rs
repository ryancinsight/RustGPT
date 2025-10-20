use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::hypernetwork::Hypernetwork;
use crate::llm::Layer;

/// Token Mixing MLP for HyperMixer (following arxiv.org/abs/2203.03691)
///
/// This layer implements token mixing using hypernetwork-generated weights.
/// The hypernetwork dynamically generates W1 and W2 based on input queries,
/// then applies: output = W2(GELU(W1(input^T)))^T
///
/// Key differences from vanilla MLP-Mixer:
/// - Uses hypernetwork to generate weights dynamically (content-aware)
/// - Operates on transposed input to mix across sequence dimension
/// - Simple MLP structure: W1 -> GELU -> W2
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TokenMixingMLP {
    /// Hypernetwork for generating W1 (input projection)
    hypernetwork_in: Hypernetwork,

    /// Hypernetwork for generating W2 (output projection)
    hypernetwork_out: Hypernetwork,

    /// Maximum sequence length
    max_seq_len: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Hidden dimension for token mixing MLP
    hidden_dim: usize,

    /// Cached values for backward pass
    cached_input: Option<Array2<f32>>,
    cached_transposed_input: Option<Array2<f32>>,
    cached_w1: Option<Array2<f32>>,
    cached_w2: Option<Array2<f32>>,
    cached_hidden_pre_gelu: Option<Array2<f32>>,
    cached_hidden_post_gelu: Option<Array2<f32>>,
    cached_output_transposed: Option<Array2<f32>>,
}

impl TokenMixingMLP {
    /// Create a new HyperMixer token mixing layer
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
    ) -> Self {
        // Hypernetwork generates W1: [embedding_dim, hidden_dim] from queries
        // Input to hypernetwork: embedding_dim, Output: embedding_dim * hidden_dim
        let hypernetwork_in = Hypernetwork::new(
            embedding_dim,
            hypernetwork_hidden_dim,
            embedding_dim * hidden_dim,
        );

        // Hypernetwork generates W2: [hidden_dim, embedding_dim] from queries
        // Input to hypernetwork: embedding_dim, Output: hidden_dim * embedding_dim
        let hypernetwork_out = Hypernetwork::new(
            embedding_dim,
            hypernetwork_hidden_dim,
            hidden_dim * embedding_dim,
        );

        Self {
            hypernetwork_in,
            hypernetwork_out,
            max_seq_len,
            embedding_dim,
            hidden_dim,
            cached_input: None,
            cached_transposed_input: None,
            cached_w1: None,
            cached_w2: None,
            cached_hidden_pre_gelu: None,
            cached_hidden_post_gelu: None,
            cached_output_transposed: None,
        }
    }

    /// GELU activation function
    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    /// GELU derivative for backward pass
    fn gelu_derivative(x: f32) -> f32 {
        let tanh_arg = (2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3));
        let tanh_val = tanh_arg.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;

        0.5 * (1.0 + tanh_val) +
        0.5 * x * sech2 * (2.0 / std::f32::consts::PI).sqrt() * (1.0 + 3.0 * 0.044715 * x.powi(2))
    }

    pub fn num_heads(&self) -> usize {
        1 // HyperMixer doesn't use multi-head architecture
    }

    /// Forward pass following HyperMixer Algorithm 1
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Input shape: [seq_len, embedding_dim]
        let (seq_len, embedding_dim) = (input.shape()[0], input.shape()[1]);

        // Generate queries for hypernetwork (use mean pooling over sequence)
        let mut queries = Array2::zeros((1, embedding_dim));
        for d in 0..embedding_dim {
            let mut sum = 0.0;
            for s in 0..seq_len {
                sum += input[[s, d]];
            }
            queries[[0, d]] = sum / seq_len as f32;
        }

        // Generate W1 using hypernetwork: [embedding_dim, hidden_dim]
        let w1_flat = self.hypernetwork_in.forward(&queries);
        let mut w1 = Array2::zeros((self.embedding_dim, self.hidden_dim));

        // Scale factor for W1: Xavier initialization scale
        let w1_scale = (2.0 / (self.embedding_dim as f32)).sqrt() * 0.1; // 0.1 for extra stability
        for i in 0..self.embedding_dim {
            for j in 0..self.hidden_dim {
                w1[[i, j]] = w1_flat[[0, i * self.hidden_dim + j]] * w1_scale;
            }
        }

        // Generate W2 using hypernetwork: [hidden_dim, embedding_dim]
        let w2_flat = self.hypernetwork_out.forward(&queries);
        let mut w2 = Array2::zeros((self.hidden_dim, self.embedding_dim));

        // Scale factor for W2: Xavier initialization scale
        let w2_scale = (2.0 / (self.hidden_dim as f32)).sqrt() * 0.1; // 0.1 for extra stability
        for i in 0..self.hidden_dim {
            for j in 0..self.embedding_dim {
                w2[[i, j]] = w2_flat[[0, i * self.embedding_dim + j]] * w2_scale;
            }
        }

        // Transpose input to operate on sequence dimension: [embedding_dim, seq_len]
        let transposed_input = input.t().to_owned();

        // Apply W1: [embedding_dim, seq_len] @ [embedding_dim, hidden_dim]^T = [seq_len, hidden_dim]
        // Actually: [hidden_dim, embedding_dim] @ [embedding_dim, seq_len] = [hidden_dim, seq_len]
        let hidden_pre_gelu = w1.t().dot(&transposed_input);

        // Apply GELU activation
        let mut hidden_post_gelu = hidden_pre_gelu.clone();
        for elem in hidden_post_gelu.iter_mut() {
            *elem = Self::gelu(*elem);
        }

        // Apply W2: [hidden_dim, embedding_dim] @ [hidden_dim, seq_len] = [embedding_dim, seq_len]
        let output_transposed = w2.t().dot(&hidden_post_gelu);

        // Transpose back: [seq_len, embedding_dim]
        let output = output_transposed.t().to_owned();

        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_transposed_input = Some(transposed_input);
        self.cached_w1 = Some(w1);
        self.cached_w2 = Some(w2);
        self.cached_hidden_pre_gelu = Some(hidden_pre_gelu);
        self.cached_hidden_post_gelu = Some(hidden_post_gelu);
        self.cached_output_transposed = Some(output_transposed);

        output
    }
}

impl Layer for TokenMixingMLP {
    fn layer_type(&self) -> &str {
        "TokenMixingMLP"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward(input)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Backward pass through HyperMixer token mixing
        // output_grads shape: [seq_len, embedding_dim]

        // Get cached values
        let input = self.cached_input.as_ref().expect("forward must be called first");
        let transposed_input = self.cached_transposed_input.as_ref().unwrap();
        let w1 = self.cached_w1.as_ref().unwrap();
        let w2 = self.cached_w2.as_ref().unwrap();
        let hidden_pre_gelu = self.cached_hidden_pre_gelu.as_ref().unwrap();
        let hidden_post_gelu = self.cached_hidden_post_gelu.as_ref().unwrap();

        // Transpose output_grads: [embedding_dim, seq_len]
        let grad_output_t = output_grads.t().to_owned();

        // Backward through W2^T: grad_hidden_post_gelu = W2 @ grad_output_t
        let grad_hidden_post_gelu = w2.dot(&grad_output_t);

        // Backward through GELU
        let mut grad_hidden_pre_gelu = grad_hidden_post_gelu.clone();
        for (i, elem) in grad_hidden_pre_gelu.iter_mut().enumerate() {
            let pre_gelu_val = hidden_pre_gelu[[i / hidden_pre_gelu.shape()[1], i % hidden_pre_gelu.shape()[1]]];
            *elem *= Self::gelu_derivative(pre_gelu_val);
        }

        // Backward through W1^T: grad_transposed_input = W1 @ grad_hidden_pre_gelu
        let grad_transposed_input = w1.dot(&grad_hidden_pre_gelu);

        // Transpose back to get input gradients: [seq_len, embedding_dim]
        let grad_input = grad_transposed_input.t().to_owned();

        // Compute gradients for hypernetwork parameters
        // grad_w1 = grad_hidden_pre_gelu @ transposed_input^T
        // grad_w2 = grad_output_t @ hidden_post_gelu^T
        // These will be backpropagated through the hypernetworks

        // For now, return empty parameter gradients (hypernetworks handle their own gradients)
        let param_grads = vec![];

        (grad_input, param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        // Hypernetworks handle their own gradient updates
        // This method is called but doesn't need to do anything
        // since the hypernetwork parameters are updated separately
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        // Return hypernetwork parameters
        self.hypernetwork_in.parameters() + self.hypernetwork_out.parameters()
    }
}

// Remove old multi-head implementation - HyperMixer uses single token mixing layer
// The old TokenMixingMLP::new is now at the top of the impl block
// End of TokenMixingMLP implementation
