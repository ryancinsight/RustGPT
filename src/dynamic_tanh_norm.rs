use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

/// Dynamic Tanh Normalization (DyT)
///
/// Element-wise normalization using a learnable scaling factor `alpha` applied
/// before `tanh`, followed by per-channel scale `gamma` and bias `beta`:
///
///   y = tanh(alpha * x) ⊙ gamma + beta
///
/// - `alpha`: learnable scalar controlling nonlinearity strength
/// - `gamma`: per-feature scale (shape: [1, d])
/// - `beta`: per-feature bias (shape: [1, d])
///
/// This provides a lightweight alternative to LayerNorm/RMSNorm without computing
/// batch statistics, while retaining adaptive scaling via `alpha`.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DynamicTanhNorm {
    /// Learnable nonlinearity scale (stored as 1x1 array for optimizer compatibility)
    alpha: Array2<f32>,
    /// Per-feature scale parameter
    gamma: Array2<f32>,
    /// Per-feature bias parameter
    beta: Array2<f32>,

    /// Cached input for backward
    cached_input: Option<Array2<f32>>, 

    /// Optimizers for parameters
    optimizer_alpha: Adam,
    optimizer_gamma: Adam,
    optimizer_beta: Adam,
}

impl DynamicTanhNorm {
    /// Create a new DynamicTanhNorm layer
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            alpha: Array2::from_shape_vec((1, 1), vec![1.0]).unwrap(), // Start at 1.0
            gamma: Array2::ones((1, embedding_dim)),
            beta: Array2::zeros((1, embedding_dim)),
            cached_input: None,
            optimizer_alpha: Adam::new((1, 1)),
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta: Adam::new((1, embedding_dim)),
        }
    }

    /// Forward normalization: y = tanh(alpha * x) ⊙ gamma + beta
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let a = self.alpha[[0, 0]];

        // Cache input for backward (needed for gradient computation)
        self.cached_input = Some(input.clone());

        // Single-pass compute without temporaries using Zip + broadcast
        let mut out = Array2::<f32>::zeros(input.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(input)
            .and_broadcast(&self.gamma)
            .and_broadcast(&self.beta)
            .for_each(|o, &x, &g, &b| {
                *o = (a * x).tanh() * g + b;
            });
        out
    }
}

impl Layer for DynamicTanhNorm {
    fn layer_type(&self) -> &str { "DynamicTanhNorm" }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> { self.normalize(input) }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let a = self.alpha[[0, 0]];
        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
        let d = input.ncols();
        let mut grad_gamma = Array2::<f32>::zeros((1, d));
        let mut grad_beta = Array2::<f32>::zeros((1, d));
        let mut grad_alpha_scalar = 0.0f32;

        // Single-pass accumulation without intermediate arrays
        ndarray::Zip::indexed(&mut grad_input)
            .and(input)
            .and(output_grads)
            .and_broadcast(&self.gamma)
            .for_each(|(_, j), o, &x, &dy, &g| {
                let t = (a * x).tanh();
                let s2 = 1.0 - t * t; // sech^2(alpha * x)
                *o = (g * dy) * s2 * a; // grad w.r.t. input
                grad_gamma[[0, j]] += t * dy; // grad w.r.t. gamma
                grad_beta[[0, j]] += dy;      // grad w.r.t. beta
                grad_alpha_scalar += (g * dy) * s2 * x; // grad w.r.t. alpha (scalar)
            });

        let grad_alpha = Array2::from_shape_vec((1, 1), vec![grad_alpha_scalar]).unwrap();
        (grad_input, vec![grad_alpha, grad_gamma, grad_beta])
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        if param_grads.len() != 3 {
            return Err(crate::errors::ModelError::GradientError { 
                message: format!(
                    "DynamicTanhNorm expected 3 parameter gradients (alpha, gamma, beta), got {}",
                    param_grads.len()
                ),
            });
        }
        self.optimizer_alpha.step(&mut self.alpha, &param_grads[0], lr);
        self.optimizer_gamma.step(&mut self.gamma, &param_grads[1], lr);
        self.optimizer_beta.step(&mut self.beta, &param_grads[2], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.alpha.len() + self.gamma.len() + self.beta.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_tanh_norm_creation() {
        let dim = 16;
        let layer = DynamicTanhNorm::new(dim);
        assert_eq!(layer.gamma.shape(), &[1, dim]);
        assert_eq!(layer.beta.shape(), &[1, dim]);
        assert_eq!(layer.alpha.shape(), &[1, 1]);
    }

    #[test]
    fn test_dynamic_tanh_norm_forward_shape() {
        let dim = 8;
        let mut layer = DynamicTanhNorm::new(dim);
        let input = Array2::ones((4, dim));
        let output = layer.forward(&input);
        assert_eq!(output.shape(), &[4, dim]);
    }
}