use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

#[derive(Serialize, Deserialize, Clone)]
pub struct OutputProjection {
    pub w_out: Array2<f32>, // Weight matrix
    pub b_out: Array2<f32>, // Bias vector
    pub optimizer: Adam,
    pub cached_input: Option<Array2<f32>>,
}

impl OutputProjection {
    /// Initialize output layer with random weights and zero bias
    pub fn new(embedding_dim: usize, vocab_size: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        OutputProjection {
            w_out: Array2::from_shape_fn((embedding_dim, vocab_size), |_| normal.sample(&mut rng)),
            b_out: Array2::zeros((1, vocab_size)),
            optimizer: Adam::new((embedding_dim, vocab_size)),
            cached_input: None,
        }
    }
}

impl Layer for OutputProjection {
    fn layer_type(&self) -> &str {
        "OutputProjection"
    }

    /// Forward pass: project embeddings to vocab logits
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [sequence_length, embedding_dim]
        self.cached_input = Some(input.clone());
        input.dot(&self.w_out) + &self.b_out // shape is [sequence_length, vocab_size]
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // grads shape is [sequence_length, vocab_size]
        let input = self.cached_input.as_ref().unwrap();
        let grad_w_out = input.t().dot(output_grads);
        let grad_b_out = output_grads
            .mean_axis(Axis(0))
            .unwrap()
            .insert_axis(Axis(0));

        let grad_input = output_grads.dot(&self.w_out.t());

        (grad_input, vec![grad_w_out, grad_b_out])
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        self.optimizer.step(&mut self.w_out, &param_grads[0], lr);
        self.b_out -= &(lr * &param_grads[1].row(0));
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w_out.len() + self.b_out.len()
    }
}
