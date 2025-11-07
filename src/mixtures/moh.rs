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
use crate::mixtures::gating::{GatingStrategy, GatingConfig};
use crate::mixtures::threshold::ThresholdPredictor;
use crate::mixtures::routing::{Router, RoutingConfig, RoutingResult, SelectionAlgorithm};

/// Strategy for selecting which attention heads to activate
///
/// Implements Mixture-of-Heads (MoH) for dynamic head selection per token.
/// Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
/// Uses the shared GatingStrategy with MoH-specific configuration.
pub type HeadSelectionStrategy = GatingStrategy;

/// Configuration for head selection metrics and learned parameters
///
/// Extends the shared GatingConfig with MoH-specific parameters and threshold metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadSelectionConfig {
    /// Shared gating configuration
    pub gating: GatingConfig,
    /// Minimum number of heads to activate (safety constraint)
    pub min_heads: usize,
    /// Maximum number of heads to activate (efficiency constraint)
    pub max_heads: usize,
    /// Threshold predictor metrics: min threshold value seen
    pub metrics_tau_min: f32,
    /// Threshold predictor metrics: max threshold value seen
    pub metrics_tau_max: f32,
    /// Threshold predictor metrics: sum of threshold values
    pub metrics_tau_sum: f32,
    /// Threshold predictor metrics: count of threshold computations
    pub metrics_tau_count: usize,
    /// Threshold predictor metrics: sum of squared gate values
    pub metrics_g_sq_sum: f32,
    /// Threshold predictor metrics: count of gate computations
    pub metrics_g_count: usize,
}

impl Default for HeadSelectionConfig {
    fn default() -> Self {
        Self {
            gating: GatingConfig::default(),
            min_heads: 1,
            max_heads: 8,
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
            GatingStrategy::Learned {
                num_active,
                complexity_loss_weight,
                load_balance_weight,
                sparsity_weight,
            } => Self {
                gating: GatingConfig::from_strategy(strategy, num_heads),
                min_heads: 1, // Default min, could be parameterized
                max_heads: *num_active,
                metrics_tau_min: f32::INFINITY,
                metrics_tau_max: f32::NEG_INFINITY,
                metrics_tau_sum: 0.0,
                metrics_tau_count: 0,
                metrics_g_sq_sum: 0.0,
                metrics_g_count: 0,
            },
            GatingStrategy::Fixed { num_active } => Self {
                gating: GatingConfig::from_strategy(strategy, num_heads),
                min_heads: *num_active,
                max_heads: *num_active,
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
        self.gating.reset_metrics();
        self.metrics_tau_min = f32::INFINITY;
        self.metrics_tau_max = f32::NEG_INFINITY;
        self.metrics_tau_sum = 0.0;
        self.metrics_tau_count = 0;
        self.metrics_g_sq_sum = 0.0;
        self.metrics_g_count = 0;
    }

    /// Update metrics with new values
    pub fn update_metrics(&mut self, gate_values: &ndarray::ArrayView2<f32>) {
        self.gating.update_metrics(gate_values);
    }

    /// Get load balancing loss for training
    pub fn compute_load_balance_loss(&self) -> f32 {
        self.gating.compute_load_balance_loss()
    }

    /// Get sparsity loss for training
    pub fn compute_sparsity_loss(&self) -> f32 {
        self.gating.compute_sparsity_loss()
    }

    /// Get complexity alignment loss for training
    pub fn compute_complexity_loss(&self, target_avg_components: f32) -> f32 {
        self.gating.compute_complexity_loss(target_avg_components)
    }
}

/// Router implementation for head selection in Mixture-of-Heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRouter {
    /// Routing configuration
    pub config: RoutingConfig,
    /// Number of heads available for selection
    pub num_heads: usize,
}

impl HeadRouter {
    /// Create a new head router
    pub fn new(num_heads: usize, config: RoutingConfig) -> Self {
        Self { config, num_heads }
    }

    /// Create router from gating strategy
    pub fn from_strategy(strategy: &GatingStrategy, num_heads: usize) -> Self {
        let config = match strategy {
            GatingStrategy::Learned { num_active, .. } => RoutingConfig {
                algorithm: SelectionAlgorithm::TopK { k: *num_active },
                use_learned_predictor: true,
                num_active: *num_active,
                temperature: 1.0,
            },
            GatingStrategy::Fixed { num_active } => RoutingConfig {
                algorithm: SelectionAlgorithm::TopK { k: *num_active },
                use_learned_predictor: false,
                num_active: *num_active,
                temperature: 1.0,
            },
        };
        Self::new(num_heads, config)
    }
}

impl Router for HeadRouter {
    fn route(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        predictor: Option<&mut ThresholdPredictor>,
    ) -> RoutingResult {
        // Generate raw gating values
        let raw_gates = if self.config.use_learned_predictor {
            if let Some(predictor) = predictor {
                // Use predictor to generate gating values for each head
                predictor.predict(input)
            } else {
                // Fallback: uniform gating
                ndarray::Array2::ones((input.nrows(), self.num_heads)) / self.num_heads as f32
            }
        } else {
            // Fixed selection: uniform gating for top-k heads using iterator chains
            let n_tokens = input.nrows();
            let active_heads = self.config.num_active.min(self.num_heads);

            let gate_data: Vec<f32> = (0..n_tokens)
                .flat_map(|token_idx| {
                    (0..self.num_heads).map(move |head_idx| {
                        if head_idx < active_heads { 1.0 } else { 0.0 }
                    })
                })
                .collect();

            ndarray::Array2::from_shape_vec((n_tokens, self.num_heads), gate_data)
                .unwrap_or_else(|_| ndarray::Array2::<f32>::zeros((n_tokens, self.num_heads)))
        };

        // Apply selection algorithm
        let routing_weights = crate::mixtures::routing::apply_selection_algorithm(&raw_gates.view(), &self.config);

        RoutingResult {
            routing_weights,
            raw_gates,
        }
    }
}

/// Backward compatibility alias for the shared threshold predictor
pub type HeadSelectionPredictor = ThresholdPredictor;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_head_selection_config_default() {
        let config = HeadSelectionConfig::default();
        assert!(!config.gating.use_learned_predictor);
        assert_eq!(config.min_heads, 1);
        assert_eq!(config.max_heads, 8);
        assert_eq!(config.gating.load_balance_weight, 0.0);
    }

    #[test]
    fn test_head_selection_config_from_strategy() {
        let strategy = HeadSelectionStrategy::Learned {
            num_active: 6,
            load_balance_weight: 0.1,
            complexity_loss_weight: 0.05,
            sparsity_weight: 0.01,
        };

        let config = HeadSelectionConfig::from_strategy(&strategy, 8);
        assert!(config.gating.use_learned_predictor);
        assert_eq!(config.min_heads, 1);
        assert_eq!(config.max_heads, 6);
        assert_eq!(config.gating.load_balance_weight, 0.1);
        assert_eq!(config.gating.complexity_loss_weight, 0.05);
        assert_eq!(config.gating.sparsity_weight, 0.01);
    }

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
    fn test_load_balance_loss() {
        let mut config = HeadSelectionConfig::default();
        // Simulate gating values for load balancing test
        let gate_values = ndarray::Array2::from_shape_vec((4, 8), vec![
            0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // token 1
            0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, // token 2
            0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, // token 3
            0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, // token 4
        ]).unwrap();

        config.update_metrics(&gate_values.view());

        let loss = config.compute_load_balance_loss();
        assert!(loss >= 0.0); // Loss should be non-negative
    }
}