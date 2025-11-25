//! # Shared Threshold Predictor for Mixture Models
//!
//! This module provides a shared threshold predictor for dynamic gating in mixture models.
//! Implements AutoDeco-inspired neural architecture with Richards normalization.
//!
//! ## Overview
//!
//! The threshold predictor learns to predict gating thresholds for component selection.
//! Uses a two-layer neural network with Xavier initialization, Richards normalization,
//! and learned Richards activations replacing traditional ReLU.
//!
//! ## Architecture
//!
//! Based on AutoDeco's design principles with the following components:
//! - Two-layer neural network (embed_dim → hidden_dim → 1)
//! - Xavier weight initialization
//! - Richards normalization for adaptive behavior
//! - Learned Richards activation replacing ReLU
//! - Richards sigmoid for stable [0,1] output range

use serde::{Deserialize, Serialize};

use crate::{network::Layer, rng::get_rng};

/// Enhanced threshold predictor inspired by AutoDeco
///
/// This implements a two-layer neural network for threshold prediction with proper
/// forward and backward computations. The architecture follows AutoDeco's
/// design principles with Xavier initialization and Richards normalization.
///
/// Used for predicting gating thresholds in both MoH and MoE systems.
/// Supports multiple output dimensions for different use cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdPredictor {
    /// First layer weights (embed_dim x hidden_dim)
    pub weights1: ndarray::Array2<f32>,
    /// First layer biases (hidden_dim)
    pub bias1: ndarray::Array1<f32>,
    /// Second layer weights (hidden_dim x num_outputs)
    pub weights2: ndarray::Array2<f32>,
    /// Second layer bias (num_outputs)
    pub bias2: ndarray::Array1<f32>,
    /// Richards normalization for adaptive behavior
    pub norm: crate::richards::RichardsNorm,
    /// Richards sigmoid for stable activation
    pub sigmoid: crate::richards::RichardsCurve,
    /// Learned Richards activation replacing ReLU
    pub activation: crate::richards::RichardsCurve,

    /// Cached activations for gradient computation
    #[serde(skip)]
    cached_input: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_hidden: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_normalized: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_activation: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_activated: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_output: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_cond_input: Option<ndarray::Array2<f32>>, 
    pub cond_w: ndarray::Array2<f32>,
}

impl ThresholdPredictor {
    /// Create a new threshold predictor with AutoDeco-inspired architecture
    pub fn new_with_cond(embed_dim: usize, hidden_dim: usize, num_outputs: usize, cond_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = get_rng();

        // Xavier initialization: weights ~ N(0, 1/sqrt(fan_in))
        let scale1 = 1.0 / (embed_dim as f32).sqrt();
        let scale2 = 1.0 / (hidden_dim as f32).sqrt();

        let weights1 = ndarray::Array2::from_shape_fn((embed_dim, hidden_dim), |_| {
            rng.random_range(-scale1..scale1)
        });

        let bias1 = ndarray::Array1::zeros(hidden_dim);

        let weights2 = ndarray::Array2::from_shape_fn((hidden_dim, num_outputs), |_| {
            rng.random_range(-scale2..scale2)
        });

        let bias2 = ndarray::Array1::zeros(num_outputs);

        let norm = crate::richards::RichardsNorm::new(hidden_dim);
        let sigmoid = crate::richards::RichardsCurve::sigmoid(false); // Non-learnable sigmoid
        let activation =
            crate::richards::RichardsCurve::new_learnable(crate::richards::Variant::None); // Learnable activation replacing ReLU
        let cond_w = ndarray::Array2::from_shape_fn((cond_dim, hidden_dim), |_| {
            rng.random_range(-(1.0 / (cond_dim as f32).sqrt())..(1.0 / (cond_dim as f32).sqrt()))
        });

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            norm,
            sigmoid,
            activation,
            cached_input: None,
            cached_hidden: None,
            cached_normalized: None,
            cached_activation: None,
            cached_activated: None,
            cached_output: None,
            cached_cond_input: None,
            cond_w,
        }
    }

    pub fn new(embed_dim: usize, hidden_dim: usize, num_outputs: usize) -> Self {
        Self::new_with_cond(embed_dim, hidden_dim, num_outputs, embed_dim)
    }

    /// Predict threshold values using AutoDeco-style architecture
    ///
    /// Returns sigmoid-activated values in [0, 1] range suitable for threshold prediction
    /// Caches intermediate activations for gradient computation
    pub fn predict_with_condition(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        cond: Option<ndarray::ArrayView2<f32>>,
    ) -> ndarray::Array2<f32> {
        self.cached_input = Some(input.to_owned());
        let hidden_base = input.dot(&self.weights1);
        let hidden = if let Some(c) = cond {
            let c_owned = c.to_owned();
            self.cached_cond_input = Some(c_owned.clone());
            hidden_base + c_owned.dot(&self.cond_w) + &self.bias1
        } else {
            self.cached_cond_input = None;
            hidden_base + &self.bias1
        };
        self.cached_hidden = Some(hidden.clone());

        // Apply Richards normalization for adaptive behavior
        let normalized = self.norm.forward(&hidden);
        self.cached_normalized = Some(normalized.clone());

        // Learned Richards activation replacing ReLU
        let activation_output = self
            .activation
            .forward_matrix(&normalized.mapv(|x| x as f64))
            .mapv(|x| x as f32);
        self.cached_activation = Some(activation_output.clone());

        // Second layer input (previously activated)
        let activated = activation_output;
        self.cached_activated = Some(activated.clone());

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;
        self.cached_output = Some(output.clone());

        // Richards sigmoid activation to get values in [0, 1] range
        self.sigmoid
            .forward_matrix(&output.mapv(|x| x as f64))
            .mapv(|x| x as f32)
    }

    /// Forward pass for auxiliary computation (immutable)
    ///
    /// Returns sigmoid-activated values in [0, 1] range suitable for threshold prediction
    /// Uses consistent Richards normalization and learned Richards activation
    pub fn forward(&self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;

        // Apply Richards normalization for consistent behavior (immutable version)
        let normalized = self.norm.normalize_immutable(&hidden);

        // Learned Richards activation replacing ReLU
        let activated = self
            .activation
            .forward_matrix(&normalized.mapv(|x| x as f64))
            .mapv(|x| x as f32);

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Richards sigmoid activation to get values in [0, 1] range
        let sigmoid = crate::richards::RichardsCurve::sigmoid(false);
        sigmoid
            .forward_matrix(&output.mapv(|x| x as f64))
            .mapv(|x| x as f32)
    }

    pub fn predict(&mut self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        self.predict_with_condition(input, None)
    }

    /// Compute gradients for the two-layer threshold network
    ///
    /// Returns gradients for (weights1, bias1, weights2, bias2, activation_params)
    pub fn compute_gradients(
        &self,
        output_grads: &ndarray::Array2<f32>,
    ) -> (
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
        Option<ndarray::Array2<f32>>, 
        Vec<f64>,
    ) {
        // Retrieve cached activations
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_output = self
            .cached_output
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_activated = self
            .cached_activated
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let _cached_activation = self
            .cached_activation
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_normalized = self
            .cached_normalized
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_hidden = self
            .cached_hidden
            .as_ref()
            .expect("predict must be called before compute_gradients");

        // Gradient through Richards sigmoid
        let output_f64 = cached_output.mapv(|x| x as f64);
        let output_grads_f64 = output_grads.mapv(|x| x as f64);
        let sigmoid_grad_f64 = self.sigmoid.backward_matrix(&output_f64, &output_grads_f64);
        let d_output = sigmoid_grad_f64.mapv(|x| x as f32);

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_output);
        let grad_bias2 = d_output.sum_axis(ndarray::Axis(0));

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_output.dot(&self.weights2.t());

        // Gradient through Richards activation (replacing ReLU)
        let normalized_f64 = cached_normalized.mapv(|x| x as f64);
        let d_activated_f64 = d_activated.mapv(|x| x as f64);
        let activation_grad_f64 = self
            .activation
            .backward_matrix(&normalized_f64, &d_activated_f64);
        let d_normalized = activation_grad_f64.mapv(|x| x as f32);

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1: ndarray::Array2<f32> = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));
        let grad_cond_w = if let Some(cond_in) = &self.cached_cond_input {
            Some(cond_in.t().dot(&d_hidden))
        } else { None };

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self
            .activation
            .grad_weights_matrix(&normalized_f64, &d_activated_f64);

        (
            grad_weights1,
            grad_bias1,
            grad_weights2,
            grad_bias2,
            grad_cond_w,
            activation_grads,
        )
    }

    /// Get parameters for gradient computation
    pub fn parameters(&self) -> Vec<&ndarray::Array2<f32>> {
        vec![&self.weights1, &self.weights2]
    }

    /// Get mutable parameters for gradient updates
    pub fn parameters_mut(&mut self) -> Vec<&mut ndarray::Array2<f32>> {
        vec![&mut self.weights1, &mut self.weights2]
    }

    /// Get bias parameters
    pub fn biases(&self) -> Vec<&ndarray::Array1<f32>> {
        vec![&self.bias1, &self.bias2]
    }

    /// Get mutable bias parameters
    pub fn biases_mut(&mut self) -> Vec<&mut ndarray::Array1<f32>> {
        vec![&mut self.bias1, &mut self.bias2]
    }

    /// Get activation parameters for gradient updates
    pub fn activation_parameters(&self) -> &crate::richards::RichardsCurve {
        &self.activation
    }

    /// Get mutable activation parameters for gradient updates
    pub fn activation_parameters_mut(&mut self) -> &mut crate::richards::RichardsCurve {
        &mut self.activation
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_threshold_predictor() {
        let mut predictor = ThresholdPredictor::new(64, 32, 1); // embed_dim, hidden_dim, num_outputs
        let input = Array2::<f32>::from_shape_vec((4, 64), vec![0.1; 256]).unwrap();

        let thresholds = predictor.predict(&input.view());
        assert_eq!(thresholds.shape(), &[4, 1]);

        // Check values are in [0, 1] range (sigmoid output)
        for &val in thresholds.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_threshold_predictor_forward() {
        let predictor = ThresholdPredictor::new(64, 32, 1);
        let input = Array2::<f32>::from_shape_vec((4, 64), vec![0.1; 256]).unwrap();

        let thresholds = predictor.forward(&input.view());
        assert_eq!(thresholds.shape(), &[4, 1]);

        // Check values are in [0, 1] range (sigmoid output)
        for &val in thresholds.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_threshold_predictor_multiple_outputs() {
        let predictor = ThresholdPredictor::new(64, 32, 4); // 4 outputs
        let input = Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();

        let thresholds = predictor.forward(&input.view());
        assert_eq!(thresholds.shape(), &[2, 4]); // batch_size x num_outputs

        // Check values are in [0, 1] range (sigmoid output)
        for &val in thresholds.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_threshold_predictor_gradient_computation() {
        let mut predictor = ThresholdPredictor::new(32, 16, 1);
        let input = Array2::<f32>::from_shape_vec((2, 32), vec![0.1; 64]).unwrap();

        // Forward pass to cache activations
        let _output = predictor.predict(&input.view());

        // Compute gradients
        let output_grads = Array2::<f32>::from_elem((2, 1), 1.0);
        let (grad_w1, grad_b1, grad_w2, grad_b2, _grad_cond_w, activation_grads) =
            predictor.compute_gradients(&output_grads);

        // Check gradient shapes
        assert_eq!(grad_w1.shape(), &[32, 16]); // embed_dim x hidden_dim
        assert_eq!(grad_b1.shape(), &[16]); // hidden_dim
        assert_eq!(grad_w2.shape(), &[16, 1]); // hidden_dim x num_outputs
        assert_eq!(grad_b2.shape(), &[1]); // num_outputs

        // Check activation gradients exist
        assert!(!activation_grads.is_empty());
    }
}
