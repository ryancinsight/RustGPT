use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, network::Layer, rng::get_rng};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OutputProjection {
    pub w_out: Array2<f32>, // Weight matrix (no bias - modern LLM practice)
    pub optimizer: Adam,
    pub cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    /// Initialize output layer with random weights (no bias - modern LLM practice)
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = get_rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    /// Forward pass: project embeddings to vocab logits (no bias)
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [sequence_length, embedding_dim]
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) // shape is [sequence_length, vocab_size]
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // grads shape is [sequence_length, vocab_size]
        let input = self.cached_input.as_ref().unwrap();
        let grad_w_out = input.t().dot(output_grads);
        let grad_input = output_grads.dot(&self.w_out.t());

        (grad_input, vec![grad_w_out])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.is_empty() {
            return Err(crate::errors::ModelError::GradientError {
                message: "OutputProjection expected 1 parameter gradient (weights), got 0"
                    .to_string(),
            });
        }
        let mut grad = param_grads[0].clone();
        grad.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
        let gnorm: f32 = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let wnorm = self.weight_norm().max(1e-6);
        let clip = 5.0f32;
        let mut scale = (wnorm / gnorm.max(1e-6)).clamp(0.5, 2.0);
        if gnorm.is_finite() && gnorm > clip && gnorm > 0.0 {
            scale *= clip / gnorm;
        }
        grad.mapv_inplace(|x| x * scale);
        self.optimizer.step(&mut self.w_out, &grad, lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w_out.len()
    }

    fn weight_norm(&self) -> f32 {
        let sumsq = self.w_out.iter().map(|&w| w * w).sum::<f32>();
        sumsq.sqrt()
    }

    fn zero_gradients(&mut self) {
        // OutputProjection doesn't maintain internal gradient state
        // Gradients are computed on-demand
    }
}
