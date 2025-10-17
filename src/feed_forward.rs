use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FeedForward {
    w1: Array2<f32>,
    w2: Array2<f32>,

    // Cached values for backward pass
    input: Option<Array2<f32>>,
    hidden_pre_activation: Option<Array2<f32>>,
    hidden_post_activation: Option<Array2<f32>>,

    optimizer_w1: Adam,
    optimizer_w2: Adam,
}

impl FeedForward {
    /// Initialize a feedforward layer with random weights (no bias - modern LLM practice)
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/He initialization for w1: std = sqrt(2 / fan_in)
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();

        // Xavier/He initialization for w2: std = sqrt(2 / fan_in)
        let std_w2 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        FeedForward {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            w2: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_w2.sample(&mut rng)),
            input: None,
            hidden_pre_activation: None,
            hidden_post_activation: None,
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim, embedding_dim)),
        }
    }
}

impl Layer for FeedForward {
    fn layer_type(&self) -> &str {
        "FeedForward"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let hidden_pre_activation = input.dot(&self.w1);
        let hidden_post_activation = hidden_pre_activation.mapv(|x| x.max(0.0)); // ReLU

        let output = hidden_post_activation.dot(&self.w2);

        // Cache values
        self.input = Some(input.clone());
        self.hidden_pre_activation = Some(hidden_pre_activation);
        self.hidden_post_activation = Some(hidden_post_activation);

        output + input // residual connection (no LayerNorm here)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Unwrap cached values
        let input = self.input.as_ref().expect("forward must be run first");
        let hidden_pre_activation = self.hidden_pre_activation.as_ref().unwrap();
        let hidden_post_activation = self.hidden_post_activation.as_ref().unwrap();

        // Compute gradients for W2 (no bias)
        let grad_w2 = hidden_post_activation.t().dot(output_grads);

        // Gradient w.r.t. hidden_post_activation
        let grad_hidden_post_activation = output_grads.dot(&self.w2.t());

        // Gradient through ReLU
        let relu_grad = hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad_hidden_pre_activation = grad_hidden_post_activation * relu_grad;

        // Gradient w.r.t. W1 (no bias)
        let grad_w1 = input.t().dot(&grad_hidden_pre_activation);

        // Gradient w.r.t. input (through feed-forward computation)
        let grad_input_feedforward = grad_hidden_pre_activation.dot(&self.w1.t());

        // Add gradient from residual connection
        // Forward: output = W2(ReLU(W1*input)) + input
        // Backward: grad_input = grad_feedforward + grad_residual
        let grad_input = grad_input_feedforward + output_grads;

        (grad_input, vec![grad_w1, grad_w2])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.len() != 2 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "FeedForward expected 2 parameter gradients (W1, W2), got {}",
                    param_grads.len()
                ),
            });
        }

        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w1.len() + self.w2.len()
    }
}
