//! # Shared Gating Logic for Mixture Models
//!
//! This module provides shared gating mechanisms for dynamic selection in mixture models.
//! Supports both Mixture-of-Heads (MoH) and Mixture-of-Experts (MoE) routing.
//!
//! ## Overview
//!
//! The gating system provides configurable strategies for selecting which components
//! (attention heads or experts) to activate per token. Uses learned predictors with
//! Richards normalization and AutoDeco-inspired architectures.
//!
//! ## Key Components
//!
//! - **GatingStrategy**: Unified configuration for different gating approaches
//! - **GatingConfig**: Configuration with metrics and learned parameters
//! - **Loss computation**: Delegates to shared metrics module

use serde::{Deserialize, Serialize};
use crate::mixtures::metrics::MixtureMetrics;

/// Strategy for gating component activation (heads or experts)
///
/// Provides unified configuration for both MoH and MoE gating approaches.
/// Based on AutoDeco-inspired learned selection with Richards normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GatingStrategy {
    /// Learned gating: AutoDeco-inspired dynamic selection using neural predictors
    ///
    /// Uses a two-layer neural network with Richards normalization to learn optimal
    /// component activation patterns. All components are candidates for selection.
    Learned {
        /// Number of components to activate per token (top-k selection)
        num_active: usize,
        /// Weight for load balance loss (prevents routing collapse)
        load_balance_weight: f32,
        /// Weight for sparsity loss (encourages minimal activation)
        sparsity_weight: f32,
        /// Weight for complexity alignment loss (aligns usage with predicted complexity)
        complexity_loss_weight: f32,
    },
    /// Fixed gating: Select fixed number of components per token
    ///
    /// Simple deterministic selection without learning.
    Fixed {
        /// Number of components to activate per token
        num_active: usize,
    },
}

/// Configuration for gating metrics and learned parameters
///
/// Tracks activation patterns, load balancing, and training metrics.
/// Supports both head selection (MoH) and expert routing (MoE).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatingConfig {
    /// Use learned predictor for dynamic gating
    pub use_learned_predictor: bool,
    /// Number of components to activate per token
    pub num_active: usize,
    /// Weight for load balance loss
    pub load_balance_weight: f32,
    /// Weight for sparsity loss
    pub sparsity_weight: f32,
    /// Weight for complexity alignment loss
    pub complexity_loss_weight: f32,
    /// Shared metrics for tracking activation patterns
    pub metrics: MixtureMetrics,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            use_learned_predictor: false,
            num_active: 2,
            load_balance_weight: 0.0,
            sparsity_weight: 0.0,
            complexity_loss_weight: 0.0,
            metrics: MixtureMetrics::default(),
        }
    }
}

impl GatingConfig {
    /// Create gating config from strategy
    pub fn from_strategy(strategy: &GatingStrategy, num_components: usize) -> Self {
        match strategy {
            GatingStrategy::Learned {
                num_active,
                load_balance_weight,
                sparsity_weight,
                complexity_loss_weight,
            } => Self {
                use_learned_predictor: true,
                num_active: *num_active,
                load_balance_weight: *load_balance_weight,
                sparsity_weight: *sparsity_weight,
                complexity_loss_weight: *complexity_loss_weight,
                metrics: MixtureMetrics::new(num_components),
            },
            GatingStrategy::Fixed { num_active } => Self {
                use_learned_predictor: false,
                num_active: *num_active,
                load_balance_weight: 0.0,
                sparsity_weight: 0.0,
                complexity_loss_weight: 0.0,
                metrics: MixtureMetrics::new(num_components),
            },
        }
    }

    /// Reset metrics when strategy changes
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Update metrics with new gating decisions
    /// gate_values: shape (num_tokens, num_components) - gating values for each token-component pair
    pub fn update_metrics(&mut self, gate_values: &ndarray::ArrayView2<f32>) {
        // Validate that the number of components matches our configuration
        let num_components = gate_values.ncols();
        if self.metrics.active_sum_per_component.len() != num_components {
            eprintln!("Warning: GatingConfig component count mismatch. Expected {}, got {}. This may indicate improper initialization.",
                     self.metrics.active_sum_per_component.len(), num_components);
        }
        self.metrics.update(gate_values);
    }

    /// Get load balancing loss for training (prevents single component dominance)
    pub fn compute_load_balance_loss(&self) -> f32 {
        self.metrics.compute_load_balance_loss()
    }

    /// Get sparsity loss for training (encourages minimal component usage)
    pub fn compute_sparsity_loss(&self) -> f32 {
        self.metrics.compute_sparsity_loss(self.num_active)
    }

    /// Get complexity alignment loss for training (aligns component usage with predicted complexity)
    pub fn compute_complexity_loss(&self, target_avg_components: f32) -> f32 {
        self.metrics.compute_complexity_loss(target_avg_components)
    }

    /// Get average number of active components per token (soft gating equivalent)
    pub fn get_avg_active_components(&self) -> f32 {
        self.metrics.get_avg_active_components()
    }

    /// Get average number of components with significant gate value (> 0.1)
    pub fn get_avg_significant_components(&self) -> f32 {
        self.metrics.get_avg_significant_components()
    }

    /// Get gating entropy (higher = more uniform distribution across components)
    pub fn get_gating_entropy(&self) -> f32 {
        self.metrics.get_gating_entropy()
    }
}

/// Select top-k components based on gating values
pub fn select_top_k_components(gate_values: &ndarray::Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let mut selections = Vec::new();

    for row in gate_values.outer_iter() {
        let mut component_gates: Vec<(usize, f32)> = row.iter()
            .enumerate()
            .map(|(idx, &gate)| (idx, gate))
            .collect();

        // Sort by gate value (descending)
        component_gates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top-k components
        let selected: Vec<usize> = component_gates.into_iter()
            .take(k)
            .map(|(idx, _)| idx)
            .collect();

        selections.push(selected);
    }

    selections
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_config_default() {
        let config = GatingConfig::default();
        assert!(!config.use_learned_predictor);
        assert_eq!(config.num_active, 2);
        assert_eq!(config.load_balance_weight, 0.0);
    }

    #[test]
    fn test_gating_config_from_strategy() {
        let strategy = GatingStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.1,
            sparsity_weight: 0.01,
            complexity_loss_weight: 0.05,
        };

        let config = GatingConfig::from_strategy(&strategy, 8);
        assert!(config.use_learned_predictor);
        assert_eq!(config.num_active, 4);
        assert_eq!(config.load_balance_weight, 0.1);
        assert_eq!(config.sparsity_weight, 0.01);
        assert_eq!(config.complexity_loss_weight, 0.05);
        assert_eq!(config.metrics.active_sum_per_component.len(), 8);
    }

    #[test]
    fn test_load_balance_loss() {
        let mut config = GatingConfig::default();
        // Simulate unbalanced gating: component 0 gets all tokens, others get none
        config.metrics.resize(8);
        config.metrics.token_count_per_component = vec![100, 0, 0, 0, 0, 0, 0, 0];
        config.metrics.total_decisions = 100;

        let loss = config.compute_load_balance_loss();
        assert!(loss > 0.0); // Should have high loss due to imbalance
    }

    #[test]
    fn test_select_top_k_components() {
        let gate_values = ndarray::Array2::from_shape_vec((2, 4), vec![
            0.1, 0.7, 0.1, 0.1,  // Token 1: component 1 has highest gate
            0.2, 0.2, 0.5, 0.1,  // Token 2: component 2 has highest gate
        ]).unwrap();

        let selections = select_top_k_components(&gate_values, 2);

        assert_eq!(selections.len(), 2);
        assert_eq!(selections[0].len(), 2); // Top 2 for token 1
        assert_eq!(selections[1].len(), 2); // Top 2 for token 2

        // Component 1 should be in top 2 for token 1
        assert!(selections[0].contains(&1));
        // Component 2 should be in top 2 for token 2
        assert!(selections[1].contains(&2));
    }
}
