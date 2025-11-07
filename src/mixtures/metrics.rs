//! # Shared Metrics for Mixture Models
//!
//! This module provides shared metrics tracking and loss computation for mixture models.
//! Supports both Mixture-of-Heads (MoH) and Mixture-of-Experts (MoE) training.
//!
//! ## Overview
//!
//! The metrics system tracks component activation patterns, load balancing, and training
//! statistics. Provides loss functions for regularization during training.
//!
//! ## Key Components
//!
//! - **MixtureMetrics**: Core metrics storage and computation
//! - **Loss computation**: Load balancing, sparsity, and complexity alignment
//! - **Statistics**: Activation entropy, averages, and distribution analysis

use serde::{Deserialize, Serialize};

/// Shared metrics for mixture model training and monitoring
///
/// Tracks activation patterns, load balancing, and training statistics
/// for both MoH and MoE components (heads or experts).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureMetrics {
    /// Sum of activation values per component (for load balancing)
    pub active_sum_per_component: Vec<f32>,
    /// Token count per component (for load balancing)
    pub token_count_per_component: Vec<usize>,
    /// Min gate/threshold value seen
    pub gate_min: f32,
    /// Max gate/threshold value seen
    pub gate_max: f32,
    /// Sum of gate/threshold values
    pub gate_sum: f32,
    /// Count of gate computations
    pub gate_count: usize,
    /// Sum of squared gate values
    pub gate_sq_sum: f32,
    /// Total routing/gating decisions made
    pub total_decisions: usize,
}

impl Default for MixtureMetrics {
    fn default() -> Self {
        Self::new(0) // Start with no components, will be resized as needed
    }
}

impl MixtureMetrics {
    /// Create new metrics with specified number of components
    pub fn new(num_components: usize) -> Self {
        Self {
            active_sum_per_component: vec![0.0; num_components],
            token_count_per_component: vec![0; num_components],
            gate_min: f32::INFINITY,
            gate_max: f32::NEG_INFINITY,
            gate_sum: 0.0,
            gate_count: 0,
            gate_sq_sum: 0.0,
            total_decisions: 0,
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        for c in 0..self.active_sum_per_component.len() {
            self.active_sum_per_component[c] = 0.0;
            self.token_count_per_component[c] = 0;
        }
        self.gate_min = f32::INFINITY;
        self.gate_max = f32::NEG_INFINITY;
        self.gate_sum = 0.0;
        self.gate_count = 0;
        self.gate_sq_sum = 0.0;
        self.total_decisions = 0;
    }

    /// Resize metrics for different number of components
    pub fn resize(&mut self, num_components: usize) {
        self.active_sum_per_component.resize(num_components, 0.0);
        self.token_count_per_component.resize(num_components, 0);
    }

    /// Update metrics with new gate values
    /// gate_values: shape (num_tokens, num_components) - gating values for each token-component pair
    pub fn update(&mut self, gate_values: &ndarray::ArrayView2<f32>) {
        // Defensive programming: ensure metrics are properly sized
        let num_components = gate_values.ncols();
        if self.active_sum_per_component.len() != num_components {
            eprintln!("Warning: MixtureMetrics component count mismatch. Expected {}, got {}. Resizing metrics.",
                     self.active_sum_per_component.len(), num_components);
            self.resize(num_components);
        }

        // Update per-component activation sums across all tokens
        for component_idx in 0..self.active_sum_per_component.len() {
            let component_sum: f32 = gate_values.column(component_idx).sum();
            self.active_sum_per_component[component_idx] += component_sum;
        }

        // For token counts, count components with gate value > threshold as "active"
        for token_idx in 0..gate_values.nrows() {
            let token_gates = gate_values.row(token_idx);
            for component_idx in 0..self.active_sum_per_component.len() {
                if token_gates[component_idx] > 0.1 {  // Threshold for "active"
                    self.token_count_per_component[component_idx] += 1;
                }
            }
        }

        // Update gate value statistics
        for &gate_val in gate_values.iter() {
            self.gate_min = self.gate_min.min(gate_val);
            self.gate_max = self.gate_max.max(gate_val);
            self.gate_sum += gate_val;
            self.gate_sq_sum += gate_val * gate_val;
        }
        self.gate_count += gate_values.len();
        self.total_decisions += gate_values.nrows();
    }

    /// Get load balancing loss for training (prevents single component dominance)
    pub fn compute_load_balance_loss(&self) -> f32 {
        if self.token_count_per_component.is_empty() || self.total_decisions == 0 {
            return 0.0;
        }

        let total_tokens = self.total_decisions as f32;
        let expected_per_component = total_tokens / self.active_sum_per_component.len() as f32;

        // Coefficient of variation across components using iterator chains
        let component_count = self.active_sum_per_component.len() as f32;
        let counts_f32: Vec<f32> = self.token_count_per_component.iter().map(|&x| x as f32).collect();

        let mean_count = counts_f32.iter().sum::<f32>() / component_count;

        if mean_count == 0.0 {
            return 0.0;
        }

        let variance = counts_f32.iter()
            .map(|&count| (count - expected_per_component).powi(2))
            .sum::<f32>() / component_count;

        let std_dev = variance.sqrt();
        std_dev / mean_count // Coefficient of variation
    }

    /// Get sparsity loss for training (encourages minimal component usage)
    pub fn compute_sparsity_loss(&self, num_active: usize) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }

        let avg_components_per_token = num_active as f32;
        let target_sparsity = 1.0; // Target: 1 component per token on average
        (avg_components_per_token - target_sparsity).powi(2)
    }

    /// Get complexity alignment loss for training (aligns component usage with predicted complexity)
    pub fn compute_complexity_loss(&self, target_avg_components: f32) -> f32 {
        if self.active_sum_per_component.is_empty() || self.total_decisions == 0 {
            return 0.0;
        }

        let total_tokens = self.total_decisions as f32;
        let current_avg_components = self.active_sum_per_component.iter().sum::<f32>() / total_tokens;
        (current_avg_components - target_avg_components).powi(2)
    }

    /// Get average number of active components per token (soft gating equivalent)
    pub fn get_avg_active_components(&self) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }

        // Average active components = sum of all gate values / total tokens
        let total_active_sum: f32 = self.active_sum_per_component.iter().sum();
        total_active_sum / self.total_decisions as f32
    }

    /// Get average number of components with significant gate value (> 0.1)
    pub fn get_avg_significant_components(&self) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }

        // Count components with gate value > 0.1 as "significant"
        let significant_count: usize = self.gate_count
            .saturating_sub(self.active_sum_per_component.iter()
                .map(|&sum| if sum <= 0.1 { 1 } else { 0 })
                .sum::<usize>());

        significant_count as f32 / self.total_decisions as f32
    }

    /// Get gating entropy (higher = more uniform distribution across components)
    pub fn get_gating_entropy(&self) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }

        // Calculate entropy of the average gate values using iterator chains
        let total_sum: f32 = self.active_sum_per_component.iter().sum();
        if total_sum == 0.0 {
            return 0.0;
        }

        self.active_sum_per_component.iter()
            .map(|&sum| sum / total_sum)
            .filter(|&prob| prob > 0.0)
            .map(|prob| prob * prob.ln())
            .sum::<f32>()
            .abs() // Entropy is always positive
    }

    /// Get RMS of gate values (useful for monitoring training stability)
    pub fn get_gate_rms(&self) -> f32 {
        if self.gate_count == 0 {
            return 0.0;
        }

        (self.gate_sq_sum / self.gate_count as f32).sqrt()
    }

    /// Get gate value statistics (min, max, mean)
    pub fn get_gate_stats(&self) -> (f32, f32, f32) {
        if self.gate_count == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mean = self.gate_sum / self.gate_count as f32;
        (self.gate_min, self.gate_max, mean)
    }

    /// Get load distribution statistics (variance, std_dev, coefficient of variation)
    pub fn get_load_distribution_stats(&self) -> (f32, f32, f32) {
        if self.token_count_per_component.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let counts: Vec<f32> = self.token_count_per_component.iter()
            .map(|&c| c as f32)
            .collect();

        let mean = counts.iter().sum::<f32>() / counts.len() as f32;

        if mean == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let variance = counts.iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f32>() / counts.len() as f32;

        let std_dev = variance.sqrt();
        let coeff_var = std_dev / mean;

        (variance, std_dev, coeff_var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = MixtureMetrics::new(4);
        assert_eq!(metrics.active_sum_per_component.len(), 4);
        assert_eq!(metrics.token_count_per_component.len(), 4);
        assert_eq!(metrics.total_decisions, 0);
    }

    #[test]
    fn test_metrics_reset() {
        let mut metrics = MixtureMetrics::new(4);
        metrics.active_sum_per_component[0] = 10.0;
        metrics.total_decisions = 5;

        metrics.reset();

        assert_eq!(metrics.active_sum_per_component[0], 0.0);
        assert_eq!(metrics.total_decisions, 0);
    }

    #[test]
    fn test_metrics_resize() {
        let mut metrics = MixtureMetrics::new(4);
        metrics.resize(6);

        assert_eq!(metrics.active_sum_per_component.len(), 6);
        assert_eq!(metrics.token_count_per_component.len(), 6);
    }

    #[test]
    fn test_metrics_update() {
        let mut metrics = MixtureMetrics::new(3);
        let gate_values = ndarray::Array2::from_shape_vec((2, 3), vec![
            0.8, 0.1, 0.1,  // Token 1: component 0 active
            0.1, 0.9, 0.1,  // Token 2: component 1 active
        ]).unwrap();

        metrics.update(&gate_values.view());

        assert_eq!(metrics.total_decisions, 2);
        assert_eq!(metrics.active_sum_per_component[0], 0.8 + 0.1); // 0.9
        assert_eq!(metrics.active_sum_per_component[1], 0.1 + 0.9); // 1.0
        assert_eq!(metrics.token_count_per_component[0], 1); // Only token 1 has component 0 > 0.1
        assert_eq!(metrics.token_count_per_component[1], 1); // Only token 2 has component 1 > 0.1
    }

    #[test]
    fn test_load_balance_loss() {
        let mut metrics = MixtureMetrics::new(4);
        // Simulate unbalanced: component 0 gets all tokens, others get none
        metrics.token_count_per_component = vec![100, 0, 0, 0];
        metrics.total_decisions = 100;

        let loss = metrics.compute_load_balance_loss();
        assert!(loss > 0.0); // Should have high loss due to imbalance
    }

    #[test]
    fn test_sparsity_loss() {
        let metrics = MixtureMetrics::new(4);
        let loss = metrics.compute_sparsity_loss(2); // 2 active components
        assert_eq!(loss, 1.0); // (2.0 - 1.0)^2 = 1.0
    }

    #[test]
    fn test_gate_stats() {
        let mut metrics = MixtureMetrics::new(2);
        let gate_values = ndarray::Array2::from_shape_vec((2, 2), vec![
            0.2, 0.8,  // Token 1
            0.5, 0.3,  // Token 2
        ]).unwrap();

        metrics.update(&gate_values.view());

        let (min, max, mean) = metrics.get_gate_stats();
        assert_eq!(min, 0.2);
        assert_eq!(max, 0.8);
        assert!((mean - 0.45).abs() < 1e-6); // (0.2+0.8+0.5+0.3)/4 = 0.45
    }

    #[test]
    fn test_get_avg_active_components() {
        let mut metrics = MixtureMetrics::new(2);
        let gate_values = ndarray::Array2::from_shape_vec((2, 2), vec![
            0.3, 0.7,  // Token 1: total 1.0
            0.4, 0.6,  // Token 2: total 1.0
        ]).unwrap();

        metrics.update(&gate_values.view());

        let avg = metrics.get_avg_active_components();
        assert!((avg - 1.0).abs() < 1e-6); // Should be 1.0 (normalized)
    }
}
