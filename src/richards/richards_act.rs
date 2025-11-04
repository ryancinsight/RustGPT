use ndarray::Array1;
use serde::{Deserialize, Serialize};

use super::{RichardsCurve, Variant};

/// RichardsActivation: Multiplies input by Richards curve output (x * Richards(x))
/// This creates swish-like activations and other gated activations
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsActivation {
    pub richards_curve: RichardsCurve,
}

/// Backward compatibility alias: RichardsAttention is the same as RichardsActivation
pub type RichardsAttention = RichardsActivation;

impl RichardsActivation {
    /// Create learnable Richards activation with specified variant
    pub fn new_learnable(variant: Variant) -> Self {
        Self {
            richards_curve: RichardsCurve::new_learnable(variant),
        }
    }

    /// Create fully learnable Richards activation without variant constraints
    pub fn new_fully_learnable() -> Self {
        let mut richards = RichardsCurve::new_learnable(Variant::None);
        // Disable temperature learning for compatibility with existing training code
        richards.temperature_learnable = false;
        Self {
            richards_curve: richards,
        }
    }

    /// Create a new RichardsActivation with default Richards curve (sigmoid-like)
    pub fn new_default() -> Self {
        Self {
            richards_curve: RichardsCurve::new_default(),
        }
    }

    /// Create a sigmoid-based activation (similar to swish activation)
    pub fn sigmoid(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::sigmoid(learnable),
        }
    }

    /// Create a tanh-based activation
    pub fn tanh(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::tanh(learnable),
        }
    }

    /// Create a Gompertz-based activation
    pub fn gompertz(learnable: bool) -> Self {
        Self {
            richards_curve: RichardsCurve::gompertz(learnable),
        }
    }

    /// Forward pass: x * Richards(x) (elementwise multiplication)
    pub fn forward(&self, x: &Array1<f64>) -> Array1<f64> {
        let richards_output = self.richards_curve.forward(x);
        x * &richards_output
    }

    /// Forward pass for a single scalar
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let richards_output = self.richards_curve.forward_scalar(x);
        x * richards_output
    }

    /// Backward pass: derivative of x * Richards(x)
    /// d/dx[x * Richards(x)] = Richards(x) + x * Richards'(x)
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        let richards_output = self.richards_curve.forward(x);
        let richards_derivative = self.richards_curve.derivative(x);
        &richards_output + x * &richards_derivative
    }

    /// Backward pass for a single scalar
    pub fn backward_scalar(&self, x: f64) -> f64 {
        let richards_output = self.richards_curve.forward_scalar(x);
        let richards_derivative = self.richards_curve.backward_scalar(x);
        richards_output + x * richards_derivative
    }

    /// Get the weights from the underlying Richards curve
    pub fn weights(&self) -> Vec<f64> {
        self.richards_curve.weights()
    }

    /// Compute gradients with respect to the Richards curve parameters
    pub fn grad_weights_scalar(&self, x: f64, grad_output: f64) -> Vec<f64> {
        // For f(x) = x * Richards(x), we need:
        // df/dθ = x * dRichards/dθ where θ are the Richards parameters
        let richards_grads = self.richards_curve.grad_weights_scalar(x, x * grad_output);
        richards_grads
    }

    /// Update parameters using gradients
    pub fn step(&mut self, gradients: &[f64], learning_rate: f64) {
        self.richards_curve.step(gradients, learning_rate);
    }

    /// Reset the optimizer state
    pub fn reset_optimizer(&mut self) {
        self.richards_curve.reset_optimizer();
    }

    /// Update scaling based on input statistics
    pub fn update_scaling_from_max_abs(&mut self, max_abs_x: f64) {
        self.richards_curve.update_scaling_from_max_abs(max_abs_x);
    }

    /// Get scaling parameters
    pub fn get_scaling(&self) -> (f64, f64) {
        self.richards_curve.get_scaling()
    }

    /// Set parameters directly
    pub fn set_param(&mut self, nu: Option<f64>, k: Option<f64>, m: Option<f64>, beta: Option<f64>, output_gain: Option<f64>, output_bias: Option<f64>) {
        self.richards_curve.set_param(nu, k, m, beta, output_gain, output_bias);
    }
}
