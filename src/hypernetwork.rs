use ndarray::{Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;

/// Hypernetwork that generates weights for token mixing MLP dynamically
///
/// The hypernetwork takes a mean-pooled representation of the input sequence
/// and generates the weights for the token-mixing MLP. This allows the model
/// to adapt its token mixing behavior based on the input content.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hypernetwork {
    /// First layer weights (embedding_dim → hypernetwork_hidden_dim)
    w1: Array2<f32>,
    /// First layer bias
    b1: Array2<f32>,
    /// Second layer weights (hypernetwork_hidden_dim → output_size)
    w2: Array2<f32>,
    /// Second layer bias
    b2: Array2<f32>,

    /// Cached values for backward pass
    cached_input: Option<Array2<f32>>,
    cached_hidden_pre_activation: Option<Array2<f32>>,
    cached_hidden_post_activation: Option<Array2<f32>>,

    /// Optimizers for each parameter
    optimizer_w1: Adam,
    optimizer_b1: Adam,
    optimizer_w2: Adam,
    optimizer_b2: Adam,
}

impl Hypernetwork {
    /// Create a new hypernetwork
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension (typically embedding_dim)
    /// * `hidden_dim` - Hidden dimension of the hypernetwork
    /// * `output_size` - Size of the output weight vector to generate
    pub fn new(input_dim: usize, hidden_dim: usize, output_size: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/He initialization
        let std_w1 = (2.0 / input_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();

        let std_w2 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();

        Self {
            w1: Array2::from_shape_fn((input_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            b1: Array2::zeros((1, hidden_dim)),
            w2: Array2::from_shape_fn((hidden_dim, output_size), |_| normal_w2.sample(&mut rng)),
            b2: Array2::zeros((1, output_size)),
            cached_input: None,
            cached_hidden_pre_activation: None,
            cached_hidden_post_activation: None,
            optimizer_w1: Adam::new((input_dim, hidden_dim)),
            optimizer_b1: Adam::new((1, hidden_dim)),
            optimizer_w2: Adam::new((hidden_dim, output_size)),
            optimizer_b2: Adam::new((1, output_size)),
        }
    }

    /// Forward pass: generate weights from input
    ///
    /// # Arguments
    /// * `input` - Mean-pooled input (1 × input_dim)
    ///
    /// # Returns
    /// Generated weights (1 × output_size)
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // First layer: input → hidden
        let hidden_pre_activation = input.dot(&self.w1) + &self.b1;
        let hidden_post_activation = hidden_pre_activation.mapv(|x| x.max(0.0)); // ReLU

        // Second layer: hidden → output
        let output = hidden_post_activation.dot(&self.w2) + &self.b2;

        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_hidden_pre_activation = Some(hidden_pre_activation);
        self.cached_hidden_post_activation = Some(hidden_post_activation);

        output
    }

    /// Compute gradients for backward pass
    ///
    /// # Arguments
    /// * `output_grads` - Gradients from the next layer (1 × output_size)
    ///
    /// # Returns
    /// * Input gradients (1 × input_dim)
    /// * Parameter gradients [grad_w1, grad_b1, grad_w2, grad_b2]
    pub fn compute_gradients(&self, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called first");
        let hidden_pre_activation = self.cached_hidden_pre_activation.as_ref().unwrap();
        let hidden_post_activation = self.cached_hidden_post_activation.as_ref().unwrap();

        // Gradient w.r.t. w2 and b2
        let grad_w2 = hidden_post_activation.t().dot(output_grads);
        let grad_b2 = output_grads.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient w.r.t. hidden layer output
        let grad_hidden = output_grads.dot(&self.w2.t());

        // Gradient through ReLU
        let grad_hidden_pre =
            &grad_hidden * &hidden_pre_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Gradient w.r.t. w1 and b1
        let grad_w1 = input.t().dot(&grad_hidden_pre);
        let grad_b1 = grad_hidden_pre.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Gradient w.r.t. input
        let grad_input = grad_hidden_pre.dot(&self.w1.t());

        (grad_input, vec![grad_w1, grad_b1, grad_w2, grad_b2])
    }

    /// Apply gradients using optimizers
    pub fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.len() != 4 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "Hypernetwork expected 4 parameter gradients (W1, b1, W2, b2), got {}",
                    param_grads.len()
                ),
            });
        }

        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_b1.step(&mut self.b1, &param_grads[1], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[2], lr);
        self.optimizer_b2.step(&mut self.b2, &param_grads[3], lr);
        Ok(())
    }

    /// Get total number of parameters
    pub fn parameters(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}
