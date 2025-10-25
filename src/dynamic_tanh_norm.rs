use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, llm::Layer, richards::RichardsCurve};

/// Dynamic Tanh Normalization (DyT)
///
/// Element-wise normalization using Richards curve with learnable input scaling,
/// followed by per-channel scale `gamma` and bias `beta`:
///
///   y = Richards(scale * x) ⊙ gamma + beta
///
/// - `scale`: learnable scalar input scaling within Richards curve
/// - `gamma`: per-feature scale (shape: [1, d])
/// - `beta`: per-feature bias (shape: [1, d])
///
/// This provides a lightweight normalization alternative without computing
/// batch statistics, with adaptive scaling via Richards' learnable scale parameter.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DynamicTanhNorm {
    /// Per-feature scale parameter
    gamma: Array2<f32>,
    /// Per-feature bias parameter
    beta: Array2<f32>,

    /// Cached input for backward
    cached_input: Option<Array2<f32>>,

    /// Richards curve for tanh computation with learnable scale
    richards: RichardsCurve,

    /// Optimizers for parameters
    optimizer_gamma: Adam,
    optimizer_beta: Adam,
}

impl DynamicTanhNorm {
    /// Create a new DynamicTanhNorm layer
    pub fn new(embedding_dim: usize) -> Self {
        // Start with learnable Richards for Tanh variant
        let mut richards = RichardsCurve::new_learnable(crate::richards::Variant::Tanh);

        // Set fixed parameter values for tanh approximation (keep zero-mean properties)
        richards.nu = None; // Learnable
        richards.k = None; // Learnable
        richards.m = Some(0.0); // Fixed to maintain odd function (zero-mean)
        richards.beta = None; // Learnable
        richards.a = Some(1.0); // Fixed to maintain scale
        richards.b = Some(0.0); // Fixed to maintain zero-mean
        richards.scale = None; // Learnable
        richards.shift = Some(0.0); // Fixed to maintain odd function

        // Initialize learned parameters
        richards.learned_nu = Some(1.0);
        richards.learned_k = Some(1.0);
        richards.learned_beta = Some(1.0);
        richards.learned_scale = Some(1.0);

        // Set learnability: nu, k, beta, scale learnable; others fixed
        richards.nu_learnable = true;
        richards.k_learnable = true;
        richards.m_learnable = false;
        richards.beta_learnable = true;
        richards.a_learnable = false;
        richards.b_learnable = false;
        richards.scale_learnable = true;
        richards.shift_learnable = false;

        Self {
            gamma: Array2::ones((1, embedding_dim)),
            beta: Array2::zeros((1, embedding_dim)),
            cached_input: None,
            richards,
            optimizer_gamma: Adam::new((1, embedding_dim)),
            optimizer_beta: Adam::new((1, embedding_dim)),
        }
    }

    /// Forward normalization: y = Richards(scale * x) ⊙ gamma + beta
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward (needed for gradient computation)
        self.cached_input = Some(input.clone());

        // Single-pass compute without temporaries using Zip + broadcast
        let mut out = Array2::<f32>::zeros(input.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(input)
            .and_broadcast(&self.gamma)
            .and_broadcast(&self.beta)
            .for_each(|o, &x, &g, &b| {
                *o = self.richards.forward_scalar(x as f64) as f32 * g + b;
            });
        out
    }
}

impl Layer for DynamicTanhNorm {
    fn layer_type(&self) -> &str {
        "DynamicTanhNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
        let d = input.ncols();
        let mut grad_gamma = Array2::<f32>::zeros((1, d));
        let mut grad_beta = Array2::<f32>::zeros((1, d));
        let mut grad_nu_scalar = 0.0f64;
        let mut grad_k_scalar = 0.0f64;
        let mut grad_beta_richards_scalar = 0.0f64;
        let mut grad_scale_scalar = 0.0f64;

        // Single-pass accumulation without intermediate arrays
        ndarray::Zip::indexed(&mut grad_input)
            .and(input)
            .and(output_grads)
            .and_broadcast(&self.gamma)
            .for_each(|(_, j), o, &x, &dy, &g| {
                let x_f64 = x as f64;
                let grad_richards = (g * dy) as f64; // grad w.r.t. Richards output
                let t = self.richards.forward_scalar(x_f64) as f32;
                let dt_dx = self.richards.backward_scalar(x_f64) as f32; // derivative w.r.t. x
                *o = (g * dy) * dt_dx; // grad w.r.t. input
                grad_gamma[[0, j]] += t * dy; // grad w.r.t. gamma
                grad_beta[[0, j]] += dy; // grad w.r.t. beta

                // Accumulate gradients for Richards learnable parameters
                let richards_grads = self.richards.grad_weights_scalar(x_f64, grad_richards);
                grad_nu_scalar += richards_grads[0];
                grad_k_scalar += richards_grads[1];
                grad_beta_richards_scalar += richards_grads[2];
                grad_scale_scalar += richards_grads[3];
            });



        let grad_nu = Array2::from_shape_vec((1, 1), vec![grad_nu_scalar as f32]).unwrap();
        let grad_k = Array2::from_shape_vec((1, 1), vec![grad_k_scalar as f32]).unwrap();
        let grad_beta_richards = Array2::from_shape_vec((1, 1), vec![grad_beta_richards_scalar as f32]).unwrap();
        let grad_scale = Array2::from_shape_vec((1, 1), vec![grad_scale_scalar as f32]).unwrap();
        (grad_input, vec![grad_nu, grad_k, grad_beta_richards, grad_scale, grad_gamma, grad_beta])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.len() != 6 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "DynamicTanhNorm expected 6 parameter gradients (nu, k, beta_richards, scale, gamma, beta), got {}",
                    param_grads.len()
                ),
            });
        }
        self.richards.step(&[
            param_grads[0][[0, 0]] as f64, // nu
            param_grads[1][[0, 0]] as f64, // k
            param_grads[2][[0, 0]] as f64, // beta
            param_grads[3][[0, 0]] as f64, // scale
        ], lr as f64);
        self.optimizer_gamma
            .step(&mut self.gamma, &param_grads[4], lr);
        self.optimizer_beta
            .step(&mut self.beta, &param_grads[5], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.richards.weights().len() + self.gamma.len() + self.beta.len()
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
        // Richards has 4 learnable params: nu, k, beta, scale
        assert_eq!(layer.richards.weights().len(), 4);
        assert_eq!(layer.richards.weights()[0], 1.0); // nu
        assert_eq!(layer.richards.weights()[1], 1.0); // k
        assert_eq!(layer.richards.weights()[2], 1.0); // beta
        assert_eq!(layer.richards.weights()[3], 1.0); // scale
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
