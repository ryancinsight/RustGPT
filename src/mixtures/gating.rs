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

        /// Weight for importance loss (balances soft routing mass across components).
        ///
        /// Uses MixtureMetrics.active_sum_per_component rather than token counts.
        #[serde(default)]
        importance_loss_weight: f32,

        /// Weight for Switch/GShard-style combined load+importance loss.
        ///
        /// This is often a robust default for MoE routers.
        #[serde(default)]
        switch_balance_weight: f32,
    },
    /// Soft top-p gating: Differentiable top-p selection using AutoDeco-inspired soft sampling
    ///
    /// Uses soft top-p sampling for learned hard selection. Provides differentiable
    /// training while maintaining discrete selection behavior during inference.
    SoftTopP {
        /// Top-p threshold for component selection (0.0 to 1.0)
        top_p: f32,
        /// Steepness parameter for soft top-p decay (higher = sharper transitions)
        soft_top_p_alpha: f32,
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
    /// Use soft top-p sampling for differentiable selection
    pub use_soft_top_p: bool,
    /// Number of components to activate per token
    pub num_active: usize,
    /// Top-p threshold for soft top-p selection
    pub top_p: f32,
    /// Steepness parameter for soft top-p decay
    pub soft_top_p_alpha: f32,
    /// Weight for load balance loss
    pub load_balance_weight: f32,
    /// Weight for sparsity loss
    pub sparsity_weight: f32,
    /// Weight for complexity alignment loss
    pub complexity_loss_weight: f32,

    /// Weight for importance loss (balances routing probability mass)
    #[serde(default)]
    pub importance_loss_weight: f32,

    /// Weight for Switch/GShard-style combined balance loss
    #[serde(default)]
    pub switch_balance_weight: f32,
    /// Shared metrics for tracking activation patterns
    pub metrics: MixtureMetrics,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            use_learned_predictor: false,
            use_soft_top_p: false,
            num_active: 2,
            top_p: 0.9,
            soft_top_p_alpha: 50.0,
            load_balance_weight: 0.0,
            sparsity_weight: 0.0,
            complexity_loss_weight: 0.0,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
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
                importance_loss_weight,
                switch_balance_weight,
            } => Self {
                use_learned_predictor: true,
                use_soft_top_p: false,
                num_active: *num_active,
                top_p: 0.9,
                soft_top_p_alpha: 50.0,
                load_balance_weight: *load_balance_weight,
                sparsity_weight: *sparsity_weight,
                complexity_loss_weight: *complexity_loss_weight,
                importance_loss_weight: *importance_loss_weight,
                switch_balance_weight: *switch_balance_weight,
                metrics: MixtureMetrics::new(num_components),
            },
            GatingStrategy::SoftTopP {
                top_p,
                soft_top_p_alpha,
            } => Self {
                use_learned_predictor: false,
                use_soft_top_p: true,
                num_active: num_components, // All components available for selection
                top_p: *top_p,
                soft_top_p_alpha: *soft_top_p_alpha,
                load_balance_weight: 0.0,
                sparsity_weight: 0.0,
                complexity_loss_weight: 0.0,
                importance_loss_weight: 0.0,
                switch_balance_weight: 0.0,
                metrics: MixtureMetrics::new(num_components),
            },
            GatingStrategy::Fixed { num_active } => Self {
                use_learned_predictor: false,
                use_soft_top_p: false,
                num_active: *num_active,
                top_p: 0.9,
                soft_top_p_alpha: 50.0,
                load_balance_weight: 0.0,
                sparsity_weight: 0.0,
                complexity_loss_weight: 0.0,
                importance_loss_weight: 0.0,
                switch_balance_weight: 0.0,
                metrics: MixtureMetrics::new(num_components),
            },
        }
    }

    /// Importance loss for training (balances soft routing mass)
    pub fn compute_importance_loss(&self) -> f32 {
        self.metrics.compute_importance_loss()
    }

    /// Switch/GShard-style combined balance loss
    pub fn compute_switch_balance_loss(&self) -> f32 {
        self.metrics.compute_switch_balance_loss()
    }

    /// Reset metrics when strategy changes
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }

    /// Update metrics with new gating decisions
    /// gate_values: shape (num_tokens, num_components) - gating values for each token-component
    /// pair
    pub fn update_metrics(&mut self, gate_values: &ndarray::ArrayView2<f32>) {
        // Ensure metrics are properly sized. A default-constructed config starts with 0
        // components and is expected to be resized on first use.
        let num_components = gate_values.ncols();
        if self.metrics.active_sum_per_component.len() != num_components {
            self.metrics.resize(num_components);
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

    /// Get complexity alignment loss for training (aligns component usage with predicted
    /// complexity)
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

    if gate_values.nrows() == 0 || gate_values.ncols() == 0 {
        return selections;
    }

    let k = k.clamp(1, gate_values.ncols());

    for row in gate_values.outer_iter() {
        // Maintain a small set of best (score, idx) pairs (O(E*k)).
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for (idx, &gate) in row.iter().enumerate() {
            let score = if gate.is_finite() {
                gate
            } else {
                f32::NEG_INFINITY
            };
            if best.len() < k {
                best.push((score, idx));
                continue;
            }

            let mut min_pos = 0usize;
            let mut min_score = best[0].0;
            for (p, (s, _)) in best.iter().enumerate().skip(1) {
                if *s < min_score {
                    min_score = *s;
                    min_pos = p;
                }
            }

            if score > min_score {
                best[min_pos] = (score, idx);
            }
        }

        let selected: Vec<usize> = best.into_iter().map(|(_s, idx)| idx).collect();

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
        assert_eq!(config.importance_loss_weight, 0.0);
        assert_eq!(config.switch_balance_weight, 0.0);
    }

    #[test]
    fn test_gating_config_from_strategy() {
        let strategy = GatingStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.1,
            sparsity_weight: 0.01,
            complexity_loss_weight: 0.05,
            importance_loss_weight: 0.0,
            switch_balance_weight: 0.0,
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
        config.metrics.active_sum_per_component = vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        config.metrics.token_count_per_component = vec![100, 0, 0, 0, 0, 0, 0, 0];
        config.metrics.total_decisions = 100;

        let loss = config.compute_load_balance_loss();
        assert!(loss > 0.0); // Should have high loss due to imbalance
    }

    #[test]
    fn test_select_top_k_components() {
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 4),
            vec![
                0.1, 0.7, 0.1, 0.1, // Token 1: component 1 has highest gate
                0.2, 0.2, 0.5, 0.1, // Token 2: component 2 has highest gate
            ],
        )
        .unwrap();

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
