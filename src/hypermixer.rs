use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    adam::Adam,
    channel_mixing::ChannelMixingMLP,
    llm::Layer,
    rms_norm::RMSNorm,
    token_mixing::TokenMixingMLP,
};

/// HyperMixer Block (Refined with TRM learnings)
///
/// A complete HyperMixer block that combines token mixing and channel mixing
/// with RMS normalization and adaptive residual connections. This is analogous to
/// a Transformer block but uses MLP-based token mixing instead of attention.
///
/// Architecture (Pre-LN with ReZero-style adaptive scaling):
/// ```text
/// Input
///   ↓
/// x1 = x + token_scale * TokenMixing(RMSNorm(x))
///   ↓
/// x2 = x1 + channel_scale * ChannelMixing(RMSNorm(x1))
///   ↓
/// Output
/// ```
///
/// Key improvements from TRM:
/// - RMSNorm instead of LayerNorm (50% faster, better gradient flow)
/// - Pre-LN architecture (residuals outside sublayers for stability)
/// - ReZero-style adaptive residual scaling (learned per-block scales)
/// - SwiGLU activation in ChannelMixing (better than ReLU)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HyperMixerBlock {
    /// Token mixing layer (replaces attention)
    pub token_mixing: TokenMixingMLP,
    /// Channel mixing layer (similar to feedforward)
    pub channel_mixing: ChannelMixingMLP,
    /// RMS norm before token mixing
    pub norm1: RMSNorm,
    /// RMS norm before channel mixing
    pub norm2: RMSNorm,

    /// Adaptive residual scale for token mixing (learned, ReZero-inspired)
    token_mixing_scale: f32,
    /// Adaptive residual scale for channel mixing (learned, ReZero-inspired)
    channel_mixing_scale: f32,

    /// Optimizers for adaptive scales
    token_scale_optimizer: Adam,
    channel_scale_optimizer: Adam,

    /// Cached values for backward pass
    cached_token_output: Option<Array2<f32>>,
    cached_channel_output: Option<Array2<f32>>,
}

impl HyperMixerBlock {
    /// Create a new HyperMixer block with ReZero-style initialization
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of token embeddings
    /// * `hidden_dim` - Hidden dimension for channel mixing
    /// * `max_seq_len` - Maximum sequence length
    /// * `hypernetwork_hidden_dim` - Hidden dimension for the hypernetwork
    /// * `use_swiglu` - Whether to use SwiGLU in channel mixing (recommended: true)
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
        use_swiglu: bool,
    ) -> Self {
        // Token mixing hidden dim: use hidden_dim for consistency with channel mixing
        let token_mixing_hidden_dim = hidden_dim;

        // ReZero-style initialization: start with moderate scales for faster learning
        // Too small (0.01) causes slow learning, too large causes instability
        let initial_scale = 0.1;

        Self {
            token_mixing: TokenMixingMLP::new(
                embedding_dim,
                token_mixing_hidden_dim,
                max_seq_len,
                hypernetwork_hidden_dim,
            ),
            channel_mixing: ChannelMixingMLP::new(embedding_dim, hidden_dim, use_swiglu),
            norm1: RMSNorm::new(embedding_dim),
            norm2: RMSNorm::new(embedding_dim),
            token_mixing_scale: initial_scale,
            channel_mixing_scale: initial_scale,
            token_scale_optimizer: Adam::new((1, 1)),
            channel_scale_optimizer: Adam::new((1, 1)),
            cached_token_output: None,
            cached_channel_output: None,
        }
    }

    /// Get current adaptive scales for logging
    pub fn get_scales(&self) -> String {
        format!("TM:{:.2} CM:{:.2}", self.token_mixing_scale, self.channel_mixing_scale)
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
        // Pre-LN architecture with adaptive residual scaling (ReZero-inspired)

        // Token mixing sublayer: x1 = x + scale * TokenMixing(RMSNorm(x))
        let norm1_out = self.norm1.forward(input);
        let token_out = self.token_mixing.forward(&norm1_out);
        self.cached_token_output = Some(token_out.clone()); // Cache for scale gradient
        let x1 = input + &(&token_out * self.token_mixing_scale);

        // Channel mixing sublayer: x2 = x1 + scale * ChannelMixing(RMSNorm(x1))
        let norm2_out = self.norm2.forward(&x1);
        let channel_out = self.channel_mixing.forward(&norm2_out);
        self.cached_channel_output = Some(channel_out.clone()); // Cache for scale gradient
        let x2 = &x1 + &(&channel_out * self.channel_mixing_scale);

        x2
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Pre-LN backward pass with adaptive scaling
        // Forward was:
        //   x1 = x + token_scale * TokenMixing(RMSNorm(x))
        //   x2 = x1 + channel_scale * ChannelMixing(RMSNorm(x1))
        // Backward:
        //   grad_x2 = output_grads
        //   grad_x1 = grad_x2 + channel_scale * grad_channel_input
        //   grad_x = grad_x1 + token_scale * grad_token_input

        let mut grad_x2 = output_grads.clone();

        // Backward through channel mixing sublayer
        let (grad_channel_input, channel_param_grads) = self
            .channel_mixing
            .compute_gradients(&Array2::zeros((0, 0)), &grad_x2);

        // Backward through norm2
        let (grad_norm2_input, norm2_param_grads) = self
            .norm2
            .compute_gradients(&Array2::zeros((0, 0)), &(&grad_channel_input * self.channel_mixing_scale));

        // Gradient for channel_mixing_scale: d_loss/d_scale = sum(grad_output * sublayer_output)
        let cached_channel = self.cached_channel_output.as_ref().unwrap();
        let channel_scale_grad = (&grad_x2 * cached_channel).sum();

        // Accumulate gradient to x1 (residual path + scaled channel path)
        let grad_x1 = &grad_x2 + &grad_norm2_input;

        // Backward through token mixing sublayer
        let (grad_token_input, token_param_grads) = self
            .token_mixing
            .compute_gradients(&Array2::zeros((0, 0)), &grad_x1);

        // Backward through norm1
        let (grad_norm1_input, norm1_param_grads) = self
            .norm1
            .compute_gradients(&Array2::zeros((0, 0)), &(&grad_token_input * self.token_mixing_scale));

        // Gradient for token_mixing_scale: d_loss/d_scale = sum(grad_output * sublayer_output)
        let cached_token = self.cached_token_output.as_ref().unwrap();
        let token_scale_grad = (&grad_x1 * cached_token).sum();

        // Accumulate gradient to x (residual path + scaled token path)
        let grad_input = &grad_x1 + &grad_norm1_input;

        // Collect all parameter gradients in order
        let mut all_param_grads = Vec::new();
        all_param_grads.extend(norm1_param_grads);
        all_param_grads.extend(token_param_grads);
        all_param_grads.extend(norm2_param_grads);
        all_param_grads.extend(channel_param_grads);

        // Add scale gradients as 1x1 arrays
        all_param_grads.push(Array2::from_elem((1, 1), token_scale_grad));
        all_param_grads.push(Array2::from_elem((1, 1), channel_scale_grad));

        (grad_input, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        // Calculate expected number of gradient arrays (not scalar parameters)
        // norm1: 1, token_mixing: 0 (hypernetworks handle their own), norm2: 1, channel_mixing: 3 or 4, scales: 2
        let channel_grad_arrays = if self.channel_mixing.use_swiglu { 3 } else { 4 };
        let expected_grad_arrays = 1 + 0 + 1 + channel_grad_arrays + 2;

        if param_grads.len() != expected_grad_arrays {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "HyperMixerBlock expected {} gradient arrays, got {}",
                    expected_grad_arrays,
                    param_grads.len()
                ),
            });
        }

        let mut idx = 0;

        // Apply gradients in the order they were collected:
        // norm1 → (token_mixing handled by hypernetworks) → norm2 → channel_mixing → scales

        // Apply norm1 gradients (1 array for RMSNorm: gamma)
        let norm1_params = &param_grads[idx..idx + 1];
        self.norm1.apply_gradients(norm1_params, lr)?;
        idx += 1;

        // Token mixing gradients are handled by hypernetworks (no gradients here)
        // Skip token mixing gradient application

        // Apply norm2 gradients (1 array for RMSNorm: gamma)
        let norm2_params = &param_grads[idx..idx + 1];
        self.norm2.apply_gradients(norm2_params, lr)?;
        idx += 1;

        // Apply channel mixing gradients (3 arrays for SwiGLU, 4 for standard)
        let channel_params = &param_grads[idx..idx + channel_grad_arrays];
        self.channel_mixing.apply_gradients(channel_params, lr)?;
        idx += channel_grad_arrays;

        // Apply adaptive scale gradients with same LR as main parameters
        // Scales need to learn at similar rate to parameters for effective adaptation
        let scale_lr = lr;
        let token_scale_grad = &param_grads[idx];
        let channel_scale_grad = &param_grads[idx + 1];

        let mut token_scale_2d = Array2::from_elem((1, 1), self.token_mixing_scale);
        let mut channel_scale_2d = Array2::from_elem((1, 1), self.channel_mixing_scale);

        self.token_scale_optimizer.step(&mut token_scale_2d, token_scale_grad, scale_lr);
        self.channel_scale_optimizer.step(&mut channel_scale_2d, channel_scale_grad, scale_lr);

        // Clamp scales to [0.05, 1.0] to allow learning while preventing explosion
        self.token_mixing_scale = token_scale_2d[[0, 0]].clamp(0.05, 1.0);
        self.channel_mixing_scale = channel_scale_2d[[0, 0]].clamp(0.05, 1.0);

        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.token_mixing.parameters()
            + self.channel_mixing.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
            + 2 // token_mixing_scale + channel_mixing_scale
    }
}
