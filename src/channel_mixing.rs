use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;
use crate::swiglu::SwiGLU;

/// Channel Mixing MLP for HyperMixer (Refined with TRM learnings)
///
/// This layer mixes information across the embedding dimension (channels)
/// for each token independently. It's similar to the feedforward layer in
/// transformers but is conceptually part of the HyperMixer architecture.
///
/// Supports both ReLU and SwiGLU activations (SwiGLU recommended for better performance).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChannelMixingMLP {
    /// SwiGLU feedforward (if use_swiglu = true)
    #[serde(skip)]
    swiglu: Option<SwiGLU>,

    /// Standard MLP weights (if use_swiglu = false)
    /// First layer weights (embedding_dim → hidden_dim)
    w1: Option<Array2<f32>>,
    /// First layer bias
    b1: Option<Array2<f32>>,
    /// Second layer weights (hidden_dim → embedding_dim)
    w2: Option<Array2<f32>>,
    /// Second layer bias
    b2: Option<Array2<f32>>,

    /// Cached values for backward pass
    input: Option<Array2<f32>>,
    hidden_pre_activation: Option<Array2<f32>>,
    hidden_post_activation: Option<Array2<f32>>,

    /// Optimizers for each parameter (only used if use_swiglu = false)
    optimizer_w1: Option<Adam>,
    optimizer_b1: Option<Adam>,
    optimizer_w2: Option<Adam>,
    optimizer_b2: Option<Adam>,

    /// Whether to use SwiGLU (true) or standard ReLU MLP (false)
    pub use_swiglu: bool,

    /// Dimensions
    embedding_dim: usize,
    hidden_dim: usize,
}

impl ChannelMixingMLP {
    /// Create a new channel mixing MLP
    ///
    /// # Arguments
    /// * `embedding_dim` - Dimension of token embeddings
    /// * `hidden_dim` - Hidden dimension of the MLP
    /// * `use_swiglu` - Whether to use SwiGLU (true, recommended) or ReLU (false)
    pub fn new(embedding_dim: usize, hidden_dim: usize, use_swiglu: bool) -> Self {
        if use_swiglu {
            // Use SwiGLU for better performance (matches Transformer/TRM)
            Self {
                swiglu: Some(SwiGLU::new(embedding_dim, hidden_dim)),
                w1: None,
                b1: None,
                w2: None,
                b2: None,
                input: None,
                hidden_pre_activation: None,
                hidden_post_activation: None,
                optimizer_w1: None,
                optimizer_b1: None,
                optimizer_w2: None,
                optimizer_b2: None,
                use_swiglu: true,
                embedding_dim,
                hidden_dim,
            }
        } else {
            // Use standard ReLU MLP (legacy)
            let mut rng = rand::rng();

            // Xavier/He initialization
            let std_w1 = (2.0 / embedding_dim as f32).sqrt();
            let normal_w1 = Normal::new(0.0, std_w1).unwrap();

            let std_w2 = (2.0 / hidden_dim as f32).sqrt();
            let normal_w2 = Normal::new(0.0, std_w2).unwrap();

            Self {
                swiglu: None,
                w1: Some(Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng))),
                b1: Some(Array2::zeros((1, hidden_dim))),
                w2: Some(Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_w2.sample(&mut rng))),
                b2: Some(Array2::zeros((1, embedding_dim))),
                input: None,
                hidden_pre_activation: None,
                hidden_post_activation: None,
                optimizer_w1: Some(Adam::new((embedding_dim, hidden_dim))),
                optimizer_b1: Some(Adam::new((1, hidden_dim))),
                optimizer_w2: Some(Adam::new((hidden_dim, embedding_dim))),
                optimizer_b2: Some(Adam::new((1, embedding_dim))),
                use_swiglu: false,
                embedding_dim,
                hidden_dim,
            }
        }
    }
}

impl Layer for ChannelMixingMLP {
    fn layer_type(&self) -> &str {
        "ChannelMixingMLP"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if self.use_swiglu {
            // Use SwiGLU (no residual connection - handled by HyperMixerBlock)
            self.swiglu.as_mut().unwrap().forward(input)
        } else {
            // Use standard ReLU MLP (no residual connection - handled by HyperMixerBlock)
            let w1 = self.w1.as_ref().unwrap();
            let b1 = self.b1.as_ref().unwrap();
            let w2 = self.w2.as_ref().unwrap();
            let b2 = self.b2.as_ref().unwrap();

            let hidden_pre_activation = input.dot(w1) + b1;
            let hidden_post_activation = hidden_pre_activation.mapv(|x| x.max(0.0)); // ReLU

            let output = hidden_post_activation.dot(w2) + b2;

            // Cache values for backward pass
            self.input = Some(input.clone());
            self.hidden_pre_activation = Some(hidden_pre_activation);
            self.hidden_post_activation = Some(hidden_post_activation);

            output
        }
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.use_swiglu {
            // Delegate to SwiGLU (no residual connection)
            self.swiglu.as_ref().unwrap().compute_gradients(_input, output_grads)
        } else {
            // Standard ReLU MLP gradients (no residual connection)
            let input = self.input.as_ref().expect("forward must be called first");
            let hidden_pre_activation = self.hidden_pre_activation.as_ref().unwrap();
            let hidden_post_activation = self.hidden_post_activation.as_ref().unwrap();
            let w1 = self.w1.as_ref().unwrap();
            let w2 = self.w2.as_ref().unwrap();

            // Gradient w.r.t. w2 and b2
            let grad_w2 = hidden_post_activation.t().dot(output_grads);
            let grad_b2 = output_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Gradient w.r.t. hidden layer output
            let grad_hidden = output_grads.dot(&w2.t());

            // Gradient through ReLU
            let grad_hidden_pre =
                &grad_hidden * &hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

            // Gradient w.r.t. w1 and b1
            let grad_w1 = input.t().dot(&grad_hidden_pre);
            let grad_b1 = grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Gradient w.r.t. input (no residual connection)
            let grad_input = grad_hidden_pre.dot(&w1.t());

            (grad_input, vec![grad_w1, grad_b1, grad_w2, grad_b2])
        }
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if self.use_swiglu {
            // Delegate to SwiGLU
            self.swiglu.as_mut().unwrap().apply_gradients(param_grads, lr)
        } else {
            // Standard ReLU MLP
            if param_grads.len() != 4 {
                return Err(crate::errors::ModelError::GradientError {
                    message: format!(
                        "ChannelMixingMLP expected 4 parameter gradients, got {}",
                        param_grads.len()
                    ),
                });
            }

            let w1 = self.w1.as_mut().unwrap();
            let b1 = self.b1.as_mut().unwrap();
            let w2 = self.w2.as_mut().unwrap();
            let b2 = self.b2.as_mut().unwrap();

            self.optimizer_w1.as_mut().unwrap().step(w1, &param_grads[0], lr);
            self.optimizer_b1.as_mut().unwrap().step(b1, &param_grads[1], lr);
            self.optimizer_w2.as_mut().unwrap().step(w2, &param_grads[2], lr);
            self.optimizer_b2.as_mut().unwrap().step(b2, &param_grads[3], lr);
            Ok(())
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        if self.use_swiglu {
            self.swiglu.as_mut().unwrap().backward(grads, lr)
        } else {
            let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
            // Unwrap is safe: backward is only called from training loop which validates inputs
            self.apply_gradients(&param_grads, lr).unwrap();
            input_grads
        }
    }

    fn parameters(&self) -> usize {
        if self.use_swiglu {
            self.swiglu.as_ref().unwrap().parameters()
        } else {
            self.w1.as_ref().unwrap().len() + self.b1.as_ref().unwrap().len() +
            self.w2.as_ref().unwrap().len() + self.b2.as_ref().unwrap().len()
        }
    }
}
