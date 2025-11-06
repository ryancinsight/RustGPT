//! # Mixture of Heads (MoH)
//!
//! This module implements Mixture-of-Heads (MoH), a dynamic head selection mechanism
//! for attention layers that reduces computational cost while maintaining quality.
//!
//! ## Overview
//!
//! Mixture-of-Heads dynamically selects which attention heads to activate per token
//! using learned AutoDeco-inspired predictors. This provides better computational efficiency
//! than traditional multi-head attention.
//!
//! ## Architecture
//!
//! Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024)
//! and inspired by AutoDeco's neural architecture for learned decoding. The implementation
//! uses a two-layer neural network with Richards normalization for adaptive head selection.
//!
//! ## Key Components
//!
//! - **HeadSelectionStrategy**: Configuration for fully adaptive head selection
//! - **HeadSelectionPredictor**: AutoDeco-inspired two-layer network for threshold prediction
//! - **Complexity-aware routing**: Learns optimal head usage patterns

use serde::{Deserialize, Serialize};
use crate::llm::Layer;
use crate::richards::RichardsCurve;

/// Strategy for selecting which attention heads to activate
///
/// Implements Mixture-of-Heads (MoH) for dynamic head selection per token.
/// Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    /// Fully Adaptive Mixture-of-Heads: AutoDeco-inspired dynamic head selection
    ///
    /// Uses a two-layer neural network with Richards normalization to learn optimal
    /// head activation patterns. All heads are candidates for selection.
    FullyAdaptiveMoH {
        /// Minimum number of heads to activate (safety constraint)
        min_heads: usize,
        /// Maximum number of heads to activate (efficiency constraint)
        max_heads: usize,
        /// Weight for load balance loss (prevents routing collapse)
        load_balance_weight: f32,
        /// Weight for complexity alignment loss (aligns head usage with predicted complexity)
        complexity_loss_weight: f32,
        /// Weight for sparsity loss (encourages minimal head usage)
        sparsity_weight: f32,
    },
}

/// Configuration for head selection metrics and learned parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadSelectionConfig {
    /// Use learned threshold predictor for dynamic head selection
    pub use_learned_threshold: bool,
    /// Minimum number of heads to activate (safety constraint)
    pub min_heads: usize,
    /// Maximum number of heads to activate (efficiency constraint)
    pub max_heads: usize,
    /// Weight for load balance loss (prevents routing collapse)
    pub load_balance_weight: f32,
    /// Weight for sparsity loss (encourages minimal head usage)
    pub sparsity_weight: f32,
    /// Weight for complexity alignment loss (aligns head usage with predicted complexity)
    pub complexity_loss_weight: f32,
    /// Metrics: sum of activation values per head (for load balancing)
    pub metrics_active_sum_per_head: Vec<f32>,
    /// Metrics: token count per head (for load balancing)
    pub metrics_token_count_per_head: Vec<usize>,
    /// Metrics: min threshold value seen
    pub metrics_tau_min: f32,
    /// Metrics: max threshold value seen
    pub metrics_tau_max: f32,
    /// Metrics: sum of threshold values
    pub metrics_tau_sum: f32,
    /// Metrics: count of threshold computations
    pub metrics_tau_count: usize,
    /// Metrics: sum of squared gate values
    pub metrics_g_sq_sum: f32,
    /// Metrics: count of gate computations
    pub metrics_g_count: usize,
}

impl Default for HeadSelectionConfig {
    fn default() -> Self {
        Self {
            use_learned_threshold: false,
            min_heads: 1,
            max_heads: 8,
            load_balance_weight: 0.0,
            sparsity_weight: 0.0,
            complexity_loss_weight: 0.0,
            metrics_active_sum_per_head: vec![0.0; 8], // Default to 8 heads
            metrics_token_count_per_head: vec![0; 8],
            metrics_tau_min: f32::INFINITY,
            metrics_tau_max: f32::NEG_INFINITY,
            metrics_tau_sum: 0.0,
            metrics_tau_count: 0,
            metrics_g_sq_sum: 0.0,
            metrics_g_count: 0,
        }
    }
}

impl HeadSelectionConfig {
    /// Create head selection config from strategy
    pub fn from_strategy(strategy: &HeadSelectionStrategy, num_heads: usize) -> Self {
        match strategy {
            HeadSelectionStrategy::FullyAdaptiveMoH {
                min_heads,
                max_heads,
                complexity_loss_weight,
                load_balance_weight,
                sparsity_weight,
            } => Self {
                use_learned_threshold: true,
                min_heads: *min_heads,
                max_heads: *max_heads,
                complexity_loss_weight: *complexity_loss_weight,
                load_balance_weight: *load_balance_weight,
                sparsity_weight: *sparsity_weight,
                metrics_active_sum_per_head: vec![0.0; num_heads],
                metrics_token_count_per_head: vec![0; num_heads],
                metrics_tau_min: f32::INFINITY,
                metrics_tau_max: f32::NEG_INFINITY,
                metrics_tau_sum: 0.0,
                metrics_tau_count: 0,
                metrics_g_sq_sum: 0.0,
                metrics_g_count: 0,
            },
        }
    }

    /// Reset metrics when strategy changes
    pub fn reset_metrics(&mut self) {
        for h in 0..self.metrics_active_sum_per_head.len() {
            self.metrics_active_sum_per_head[h] = 0.0;
            self.metrics_token_count_per_head[h] = 0;
        }
        self.metrics_tau_min = f32::INFINITY;
        self.metrics_tau_max = f32::NEG_INFINITY;
        self.metrics_tau_sum = 0.0;
        self.metrics_tau_count = 0;
        self.metrics_g_sq_sum = 0.0;
        self.metrics_g_count = 0;
    }

    /// Update metrics with new values
    pub fn update_metrics(&mut self, active_sums: &[f32], token_counts: &[usize], tau_min: f32, tau_max: f32, tau_sum: f32, tau_count: usize, g_sq_sum: f32, g_count: usize) {
        for h in 0..self.metrics_active_sum_per_head.len() {
            self.metrics_active_sum_per_head[h] += active_sums[h];
            self.metrics_token_count_per_head[h] += token_counts[h];
        }
        self.metrics_tau_min = self.metrics_tau_min.min(tau_min);
        self.metrics_tau_max = self.metrics_tau_max.max(tau_max);
        self.metrics_tau_sum += tau_sum;
        self.metrics_tau_count += tau_count;
        self.metrics_g_sq_sum += g_sq_sum;
        self.metrics_g_count += g_count;
    }

    /// Get load balancing loss for training
    pub fn compute_load_balance_loss(&self) -> f32 {
        if self.metrics_token_count_per_head.is_empty() {
            return 0.0;
        }

        let total_tokens: usize = self.metrics_token_count_per_head.iter().sum();
        if total_tokens == 0 {
            return 0.0;
        }

        let mean_active = self.metrics_active_sum_per_head.iter().sum::<f32>() / total_tokens as f32;
        let variance = self.metrics_active_sum_per_head.iter()
            .map(|&active| {
                let expected = mean_active;
                (active - expected).powi(2)
            })
            .sum::<f32>() / self.metrics_active_sum_per_head.len() as f32;

        variance
    }

    /// Get sparsity loss for training
    pub fn compute_sparsity_loss(&self) -> f32 {
        if self.metrics_token_count_per_head.is_empty() {
            return 0.0;
        }

        let total_tokens: usize = self.metrics_token_count_per_head.iter().sum();
        if total_tokens == 0 {
            return 0.0;
        }

        let avg_heads = self.metrics_active_sum_per_head.iter().sum::<f32>() / total_tokens as f32;
        let target_sparsity = 1.0; // Target: 1 head per token on average
        (avg_heads - target_sparsity).powi(2)
    }

    /// Get complexity alignment loss for training
    pub fn compute_complexity_loss(&self, target_avg_heads: f32) -> f32 {
        if self.metrics_token_count_per_head.is_empty() {
            return 0.0;
        }

        let total_tokens: usize = self.metrics_token_count_per_head.iter().sum();
        if total_tokens == 0 {
            return 0.0;
        }

        let current_avg_heads = self.metrics_active_sum_per_head.iter().sum::<f32>() / total_tokens as f32;
        (current_avg_heads - target_avg_heads).powi(2)
    }
}

/// Enhanced head selection predictor inspired by AutoDeco
///
/// This implements a two-layer neural network for head selection with proper
/// forward and backward computations. The architecture follows AutoDeco's
/// design principles with Xavier initialization and Richards normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadSelectionPredictor {
    /// First layer weights (embed_dim x head_hidden_dim)
    pub weights1: ndarray::Array2<f32>,
    /// First layer biases (head_hidden_dim)
    pub bias1: ndarray::Array1<f32>,
    /// Second layer weights (head_hidden_dim x 1)
    pub weights2: ndarray::Array2<f32>,
    /// Second layer bias (1)
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
}

impl HeadSelectionPredictor {
    /// Create a new head selection predictor with AutoDeco-inspired architecture
    pub fn new(embed_dim: usize, head_hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Xavier initialization: weights ~ N(0, 1/sqrt(fan_in))
        let scale1 = 1.0 / (embed_dim as f32).sqrt();
        let scale2 = 1.0 / (head_hidden_dim as f32).sqrt();

        let weights1 = ndarray::Array2::from_shape_fn((embed_dim, head_hidden_dim), |_| {
            rng.random_range(-scale1..scale1)
        });

        let bias1 = ndarray::Array1::zeros(head_hidden_dim);

        let weights2 = ndarray::Array2::from_shape_fn((head_hidden_dim, 1), |_| {
            rng.random_range(-scale2..scale2)
        });

        let bias2 = ndarray::Array1::zeros(1);

        let norm = crate::richards::RichardsNorm::new(head_hidden_dim);
        let sigmoid = crate::richards::RichardsCurve::sigmoid(false); // Non-learnable sigmoid
        let activation = crate::richards::RichardsCurve::new_learnable(crate::richards::Variant::None); // Learnable activation replacing ReLU

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
        }
    }


    /// Predict head selection parameters using AutoDeco-style architecture
    ///
    /// Returns sigmoid-activated values in [0, 1] range suitable for head selection
    /// Caches intermediate activations for gradient computation
    pub fn predict(&mut self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        // Cache input for gradient computation
        self.cached_input = Some(input.to_owned());

        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;
        self.cached_hidden = Some(hidden.clone());

        // Apply Richards normalization for adaptive behavior
        let normalized = self.norm.forward(&hidden);
        self.cached_normalized = Some(normalized.clone());

        // Learned Richards activation replacing ReLU
        let activation_output = self.activation.forward_matrix(&normalized.mapv(|x| x as f64)).mapv(|x| x as f32);
        self.cached_activation = Some(activation_output.clone());

        // Second layer input (previously activated)
        let activated = activation_output;
        self.cached_activated = Some(activated.clone());

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;
        self.cached_output = Some(output.clone());

        // Richards sigmoid activation to get values in [0, 1] range
        self.sigmoid.forward_matrix(&output.mapv(|x| x as f64)).mapv(|x| x as f32)
    }

    /// Forward pass for auxiliary computation (immutable)
    ///
    /// Returns sigmoid-activated values in [0, 1] range suitable for head selection
    /// Uses consistent Richards normalization and learned Richards activation
    pub fn forward(&self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;

        // Apply Richards normalization for consistent behavior (immutable version)
        let normalized = self.norm.normalize_immutable(&hidden);

        // Learned Richards activation replacing ReLU
        let activated = self.activation.forward_matrix(&normalized.mapv(|x| x as f64)).mapv(|x| x as f32);

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Richards sigmoid activation to get values in [0, 1] range
        let sigmoid = RichardsCurve::sigmoid(false);
        sigmoid.forward_matrix(&output.mapv(|x| x as f64)).mapv(|x| x as f32)
    }

    /// Compute gradients for the two-layer network
    ///
    /// Returns gradients for (weights1, bias1, weights2, bias2, activation_params)
    pub fn compute_gradients(&mut self, output_grads: &ndarray::Array2<f32>) -> (ndarray::Array2<f32>, ndarray::Array1<f32>, ndarray::Array2<f32>, ndarray::Array1<f32>, Vec<f64>) {
        // Retrieve cached activations
        let cached_input = self.cached_input.as_ref().expect("predict must be called before compute_gradients");
        let cached_output = self.cached_output.as_ref().expect("predict must be called before compute_gradients");
        let cached_activated = self.cached_activated.as_ref().expect("predict must be called before compute_gradients");
        let cached_activation = self.cached_activation.as_ref().expect("predict must be called before compute_gradients");
        let cached_normalized = self.cached_normalized.as_ref().expect("predict must be called before compute_gradients");
        let cached_hidden = self.cached_hidden.as_ref().expect("predict must be called before compute_gradients");

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
        let activation_grad_f64 = self.activation.backward_matrix(&normalized_f64, &d_activated_f64);
        let d_normalized = activation_grad_f64.mapv(|x| x as f32);

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1 = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self.activation.grad_weights_matrix(&normalized_f64, &d_activated_f64);

        (grad_weights1, grad_bias1, grad_weights2, grad_bias2, activation_grads)
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
}

/// Backward compatibility alias
pub type ThresholdPredictor = HeadSelectionPredictor;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_head_selection_config_default() {
        let config = HeadSelectionConfig::default();
        assert!(!config.use_learned_threshold);
        assert_eq!(config.min_heads, 1);
        assert_eq!(config.max_heads, 8);
        assert_eq!(config.load_balance_weight, 0.0);
    }

    #[test]
    fn test_head_selection_config_from_strategy() {
        let strategy = HeadSelectionStrategy::FullyAdaptiveMoH {
            min_heads: 2,
            max_heads: 6,
            load_balance_weight: 0.1,
            complexity_loss_weight: 0.05,
            sparsity_weight: 0.01,
        };

        let config = HeadSelectionConfig::from_strategy(&strategy, 8);
        assert!(config.use_learned_threshold);
        assert_eq!(config.min_heads, 2);
        assert_eq!(config.max_heads, 6);
        assert_eq!(config.load_balance_weight, 0.1);
        assert_eq!(config.complexity_loss_weight, 0.05);
        assert_eq!(config.sparsity_weight, 0.01);
    }

    #[test]
    fn test_threshold_predictor() {
        let mut predictor = ThresholdPredictor::new(64, 32); // embed_dim, head_hidden_dim
        let input = Array2::<f32>::from_shape_vec((4, 64), vec![0.1; 256]).unwrap();

        let thresholds = predictor.predict(&input.view());
        assert_eq!(thresholds.shape(), &[4, 1]);

        // Check values are in [0, 1] range (sigmoid output)
        for &val in thresholds.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_load_balance_loss() {
        let mut config = HeadSelectionConfig::default();
        config.metrics_active_sum_per_head = vec![10.0, 5.0, 15.0, 8.0];
        config.metrics_token_count_per_head = vec![4, 4, 4, 4]; // 16 total tokens

        let loss = config.compute_load_balance_loss();
        assert!(loss >= 0.0); // Loss should be non-negative
    }
}