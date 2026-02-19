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

use crate::{common::rng::get_rng, domain::network::Layer};

type ThresholdParamGrads = (
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    Option<ndarray::Array2<f32>>,
    Vec<f64>,
);

type ThresholdParamAndInputGrads = (
    ndarray::Array2<f32>,
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    Option<ndarray::Array2<f32>>,
    Vec<f64>,
);

#[derive(Debug, Clone)]
pub struct ThresholdPredictorCache {
    pub input: ndarray::Array2<f32>,
    pub hidden: ndarray::Array2<f32>,
    pub normalized: ndarray::Array2<f32>,
    pub activation: ndarray::Array2<f32>,
    pub activated: ndarray::Array2<f32>,
    pub output: ndarray::Array2<f32>,
    pub cond_input: Option<ndarray::Array2<f32>>,
    pub norm_state: Option<crate::domain::richards::RichardsNorm>,
}

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
    pub norm: crate::domain::richards::RichardsNorm,
    /// Richards sigmoid for stable activation
    pub sigmoid: crate::domain::richards::RichardsCurve,
    /// Learned Richards activation replacing ReLU
    pub activation: crate::domain::richards::RichardsCurve,

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
    pub fn new_with_cond(
        embed_dim: usize,
        hidden_dim: usize,
        num_outputs: usize,
        cond_dim: usize,
    ) -> Self {
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

        let norm = crate::domain::richards::RichardsNorm::new(hidden_dim);
        let sigmoid = crate::domain::richards::RichardsCurve::sigmoid(false); // Non-learnable sigmoid
        let activation = crate::domain::richards::RichardsCurve::new_learnable(
            crate::domain::richards::Variant::None,
        ); // Learnable activation replacing ReLU
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
        let mut activation_output = ndarray::Array2::<f32>::zeros(normalized.raw_dim());
        self.activation
            .forward_matrix_f32_into(&normalized, &mut activation_output);
        self.cached_activation = Some(activation_output.clone());

        // Second layer input (previously activated)
        let activated = activation_output;
        self.cached_activated = Some(activated.clone());

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;
        self.cached_output = Some(output.clone());

        // Richards sigmoid activation to get values in [0, 1] range
        let mut out_sigmoid = ndarray::Array2::<f32>::zeros(output.raw_dim());
        self.sigmoid
            .forward_matrix_f32_into(&output, &mut out_sigmoid);
        out_sigmoid
    }

    /// Detached prediction that returns cache explicitly without mutating self.
    /// Used for reproducing forward passes (e.g. in TitansMAC gradient computation) without side effects.
    pub fn predict_with_condition_detached(
        &self,
        input: &ndarray::ArrayView2<f32>,
        cond: Option<ndarray::ArrayView2<f32>>,
    ) -> (ndarray::Array2<f32>, ThresholdPredictorCache) {
        let input_owned = input.to_owned();
        let hidden_base = input.dot(&self.weights1);
        let (hidden, cond_input) = if let Some(c) = cond {
            let c_owned = c.to_owned();
            (
                hidden_base + c_owned.dot(&self.cond_w) + &self.bias1,
                Some(c_owned),
            )
        } else {
            (hidden_base + &self.bias1, None)
        };

        // Clone norm to capture its state (adjusted params) during forward
        let mut local_norm = self.norm.clone();
        // Use forward (mutable on local clone) to update local cache/state
        let normalized = local_norm.forward(&hidden);

        // Learned Richards activation replacing ReLU
        let mut activation_output = ndarray::Array2::<f32>::zeros(normalized.raw_dim());
        self.activation
            .forward_matrix_f32_into(&normalized, &mut activation_output);

        // Second layer input (previously activated)
        let activated = activation_output.clone();

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Richards sigmoid activation to get values in [0, 1] range
        let mut out_sigmoid = ndarray::Array2::<f32>::zeros(output.raw_dim());
        self.sigmoid
            .forward_matrix_f32_into(&output, &mut out_sigmoid);

        (
            out_sigmoid,
            ThresholdPredictorCache {
                input: input_owned,
                hidden,
                normalized,
                activation: activation_output,
                activated,
                output,
                cond_input,
                norm_state: Some(local_norm),
            },
        )
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
        let mut activated = ndarray::Array2::<f32>::zeros(normalized.raw_dim());
        self.activation
            .forward_matrix_f32_into(&normalized, &mut activated);

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Richards sigmoid activation to get values in [0, 1] range
        let mut out_sigmoid = ndarray::Array2::<f32>::zeros(output.raw_dim());
        self.sigmoid
            .forward_matrix_f32_into(&output, &mut out_sigmoid);
        out_sigmoid
    }

    pub fn predict(&mut self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        self.predict_with_condition(input, None)
    }

    /// Extract current internal cache into a detached cache object
    pub fn take_cache(&mut self) -> Option<ThresholdPredictorCache> {
        // We need to capture the state of norm as well.
        // However, self.norm is not an Option, it stays.
        // But its state (caches) is inside.
        // self.norm.cached_adjusted_richards etc are private.
        // But we can clone `self.norm` which clones its state.
        let norm_state = Some(self.norm.clone());

        Some(ThresholdPredictorCache {
            input: self.cached_input.take()?,
            hidden: self.cached_hidden.take()?,
            normalized: self.cached_normalized.take()?,
            activation: self.cached_activation.take()?,
            activated: self.cached_activated.take()?,
            output: self.cached_output.take()?,
            cond_input: self.cached_cond_input.take(),
            norm_state,
        })
    }

    /// Compute gradients using an external cache
    pub fn compute_gradients_from_cache(
        &self,
        cache: &ThresholdPredictorCache,
        output_grads: &ndarray::Array2<f32>,
    ) -> ThresholdParamGrads {
        // Gradient through Richards sigmoid
        let mut d_output = ndarray::Array2::<f32>::zeros(output_grads.raw_dim());
        self.sigmoid
            .backward_matrix_f32_into(&cache.output, output_grads, &mut d_output);

        // Second layer gradients
        let grad_weights2 = cache.activated.t().dot(&d_output);
        let grad_bias2 = d_output.sum_axis(ndarray::Axis(0));

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_output.dot(&self.weights2.t());

        // Gradient through Richards activation (replacing ReLU)
        let mut d_normalized = ndarray::Array2::<f32>::zeros(cache.normalized.raw_dim());
        self.activation.backward_matrix_f32_into(
            &cache.normalized,
            &d_activated,
            &mut d_normalized,
        );

        // Gradient through Richards normalization
        // Use the captured norm state if available, otherwise fallback to self.norm
        // (though if cache came from detached forward, norm_state should be present)
        let norm_ref = cache.norm_state.as_ref().unwrap_or(&self.norm);
        let (d_hidden, _) = norm_ref.compute_gradients(&cache.hidden, &d_normalized);

        // First layer gradients
        let grad_weights1: ndarray::Array2<f32> = cache.input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));
        let grad_cond_w = if let Some(cond_in) = &cache.cond_input {
            Some(cond_in.t().dot(&d_hidden))
        } else {
            None
        };

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self
            .activation
            .grad_weights_matrix_f32(&cache.normalized, &d_activated);

        (
            grad_weights1,
            grad_bias1,
            grad_weights2,
            grad_bias2,
            grad_cond_w,
            activation_grads,
        )
    }

    /// Compute gradients for the two-layer threshold network
    ///
    /// Returns gradients for (weights1, bias1, weights2, bias2, activation_params)
    pub fn compute_gradients(&self, output_grads: &ndarray::Array2<f32>) -> ThresholdParamGrads {
        // Re-use compute_gradients_from_cache by constructing a temporary cache reference
        // Note: This requires cloning the cached data into the struct, which is slightly inefficient
        // compared to direct access, but avoids code duplication.
        // Given that compute_gradients is usually called once per step, this is acceptable.
        // OR we can implement a private helper that takes references.
        // But ThresholdPredictorCache owns the data.

        // Let's rely on the direct implementation for standard path to avoid cloning.
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
        let mut d_output = ndarray::Array2::<f32>::zeros(output_grads.raw_dim());
        self.sigmoid
            .backward_matrix_f32_into(cached_output, output_grads, &mut d_output);

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_output);
        let grad_bias2 = d_output.sum_axis(ndarray::Axis(0));

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_output.dot(&self.weights2.t());

        // Gradient through Richards activation (replacing ReLU)
        let mut d_normalized = ndarray::Array2::<f32>::zeros(cached_normalized.raw_dim());
        self.activation.backward_matrix_f32_into(
            cached_normalized,
            &d_activated,
            &mut d_normalized,
        );

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1: ndarray::Array2<f32> = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));
        let grad_cond_w = if let Some(cond_in) = &self.cached_cond_input {
            Some(cond_in.t().dot(&d_hidden))
        } else {
            None
        };

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self
            .activation
            .grad_weights_matrix_f32(cached_normalized, &d_activated);

        (
            grad_weights1,
            grad_bias1,
            grad_weights2,
            grad_bias2,
            grad_cond_w,
            activation_grads,
        )
    }

    /// Compute gradients for parameters **and** return gradient w.r.t. the predictor input.
    ///
    /// This is useful when the gating predictor is part of a larger differentiable routing
    /// mechanism (e.g., MoH/MoE) and upstream layers need gradients through the router.
    pub fn compute_gradients_with_input(
        &self,
        output_grads: &ndarray::Array2<f32>,
    ) -> ThresholdParamAndInputGrads {
        // Retrieve cached activations
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("predict must be called before compute_gradients_with_input");
        let cached_output = self
            .cached_output
            .as_ref()
            .expect("predict must be called before compute_gradients_with_input");
        let cached_activated = self
            .cached_activated
            .as_ref()
            .expect("predict must be called before compute_gradients_with_input");
        let cached_normalized = self
            .cached_normalized
            .as_ref()
            .expect("predict must be called before compute_gradients_with_input");
        let cached_hidden = self
            .cached_hidden
            .as_ref()
            .expect("predict must be called before compute_gradients_with_input");

        // Gradient through Richards sigmoid
        let mut d_output = ndarray::Array2::<f32>::zeros(output_grads.raw_dim());
        self.sigmoid
            .backward_matrix_f32_into(cached_output, output_grads, &mut d_output);

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_output);
        let grad_bias2 = d_output.sum_axis(ndarray::Axis(0));

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_output.dot(&self.weights2.t());

        // Gradient through Richards activation
        let mut d_normalized = ndarray::Array2::<f32>::zeros(cached_normalized.raw_dim());
        self.activation.backward_matrix_f32_into(
            cached_normalized,
            &d_activated,
            &mut d_normalized,
        );

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1: ndarray::Array2<f32> = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));
        let grad_cond_w = if let Some(cond_in) = &self.cached_cond_input {
            Some(cond_in.t().dot(&d_hidden))
        } else {
            None
        };

        // Gradient w.r.t. predictor input
        let grad_input = d_hidden.dot(&self.weights1.t());

        // Activation parameter gradients
        let activation_grads = self
            .activation
            .grad_weights_matrix_f32(cached_normalized, &d_activated);

        (
            grad_input,
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
    pub fn activation_parameters(&self) -> &crate::domain::richards::RichardsCurve {
        &self.activation
    }

    /// Get mutable activation parameters for gradient updates
    pub fn activation_parameters_mut(&mut self) -> &mut crate::domain::richards::RichardsCurve {
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
            assert!((0.0..=1.0).contains(&val));
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
            assert!((0.0..=1.0).contains(&val));
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
            assert!((0.0..=1.0).contains(&val));
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

    #[test]
    fn test_threshold_predictor_detached_and_cache() {
        let predictor = ThresholdPredictor::new(32, 16, 1);
        let input = Array2::<f32>::from_shape_vec((2, 32), vec![0.1; 64]).unwrap();

        // Detached prediction
        let (output, cache) = predictor.predict_with_condition_detached(&input.view(), None);

        // Check output range
        for &val in output.iter() {
            assert!((0.0..=1.0).contains(&val));
        }

        // Compute gradients from cache
        let output_grads = Array2::<f32>::from_elem((2, 1), 1.0);
        let (grad_w1, grad_b1, grad_w2, grad_b2, _grad_cond_w, activation_grads) =
            predictor.compute_gradients_from_cache(&cache, &output_grads);

        assert_eq!(grad_w1.shape(), &[32, 16]);
        assert_eq!(grad_b1.shape(), &[16]);
        assert_eq!(grad_w2.shape(), &[16, 1]);
        assert_eq!(grad_b2.shape(), &[1]);
        assert!(!activation_grads.is_empty());

        // Check take_cache (requires mut predictor and normal predict)
        let mut predictor_mut = predictor.clone();
        let _ = predictor_mut.predict(&input.view());
        let cache_taken = predictor_mut.take_cache().unwrap();
        // norm_state is stored in take_cache now
        assert!(cache_taken.norm_state.is_some());
    }
}
