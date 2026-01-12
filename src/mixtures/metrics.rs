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
    /// gate_values: shape (num_tokens, num_components) - gating values for each token-component
    /// pair
    pub fn update(&mut self, gate_values: &ndarray::ArrayView2<f32>) {
        // Defensive programming: ensure metrics are properly sized
        let num_components = gate_values.ncols();
        if self.active_sum_per_component.len() != num_components {
            // If we were default-constructed (0 components), this resize is expected on first use.
            // Only warn when the metrics were already tracking some other component count.
            if !self.active_sum_per_component.is_empty() {
                eprintln!(
                    "Warning: MixtureMetrics component count mismatch. Expected {}, got {}. Resizing metrics.",
                    self.active_sum_per_component.len(),
                    num_components
                );
            }
            self.resize(num_components);
        }

        // Update per-component activation sums across all tokens.
        // Be robust to any non-finite gate values (treat them as 0.0 for metrics).
        for component_idx in 0..self.active_sum_per_component.len() {
            let component_sum: f32 = gate_values
                .column(component_idx)
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .sum();
            self.active_sum_per_component[component_idx] += component_sum;
        }

        // For token counts, count components with gate value > threshold as "active"
        for token_idx in 0..gate_values.nrows() {
            let token_gates = gate_values.row(token_idx);
            for component_idx in 0..self.active_sum_per_component.len() {
                let gate_val = token_gates[component_idx];
                let gate_val = if gate_val.is_finite() { gate_val } else { 0.0 };
                if gate_val > 0.1 {
                    // Threshold for "active"
                    self.token_count_per_component[component_idx] += 1;
                }
            }
        }

        // Update gate value statistics
        for &gate_val in gate_values.iter() {
            if !gate_val.is_finite() {
                continue;
            }
            self.gate_min = self.gate_min.min(gate_val);
            self.gate_max = self.gate_max.max(gate_val);
            self.gate_sum += gate_val;
            self.gate_sq_sum += gate_val * gate_val;
            self.gate_count += 1;
        }
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
        let counts_f32: Vec<f32> = self
            .token_count_per_component
            .iter()
            .map(|&x| x as f32)
            .collect();

        let mean_count = counts_f32.iter().sum::<f32>() / component_count;

        if mean_count == 0.0 {
            return 0.0;
        }

        let variance = counts_f32
            .iter()
            .map(|&count| (count - expected_per_component).powi(2))
            .sum::<f32>()
            / component_count;

        let std_dev = variance.sqrt();
        std_dev / mean_count // Coefficient of variation
    }

    /// Importance loss based on the *soft* routing mass per component.
    ///
    /// This complements token-count load balancing by penalizing collapse where a few
    /// components receive most probability mass even if token counts look balanced.
    ///
    /// Returns coefficient-of-variation (std/mean) of per-component importance.
    pub fn compute_importance_loss(&self) -> f32 {
        if self.active_sum_per_component.is_empty() || self.total_decisions == 0 {
            return 0.0;
        }

        let total: f32 = self
            .active_sum_per_component
            .iter()
            .map(|&v| if v.is_finite() { v.max(0.0) } else { 0.0 })
            .sum();
        if !total.is_finite() || total <= 0.0 {
            return 0.0;
        }

        let k = self.active_sum_per_component.len() as f32;
        let importances: Vec<f32> = self
            .active_sum_per_component
            .iter()
            .map(|&v| {
                let v = if v.is_finite() { v.max(0.0) } else { 0.0 };
                v / total
            })
            .collect();

        let mean = 1.0 / k;
        if !mean.is_finite() || mean <= 0.0 {
            return 0.0;
        }
        let variance = importances.iter().map(|&p| (p - mean).powi(2)).sum::<f32>() / k;
        let std = variance.sqrt();
        let cv = std / mean;
        if cv.is_finite() { cv.max(0.0) } else { 0.0 }
    }

    /// Switch/GShard-style balancing loss combining load and importance.
    ///
    /// Following the common formulation: L = N * sum_i (load_i * importance_i), where
    /// load_i is the fraction of tokens routed to i (based on token_count_per_component)
    /// and importance_i is the fraction of routing probability mass assigned to i.
    pub fn compute_switch_balance_loss(&self) -> f32 {
        if self.active_sum_per_component.is_empty() || self.total_decisions == 0 {
            return 0.0;
        }

        let n = self.active_sum_per_component.len();
        if n == 0 {
            return 0.0;
        }

        let total_importance: f32 = self
            .active_sum_per_component
            .iter()
            .map(|&v| if v.is_finite() { v.max(0.0) } else { 0.0 })
            .sum();
        let total_load: f32 = self
            .token_count_per_component
            .iter()
            .map(|&c| c as f32)
            .sum();

        if !total_importance.is_finite() || total_importance <= 0.0 {
            return 0.0;
        }
        if !total_load.is_finite() || total_load <= 0.0 {
            return 0.0;
        }

        let mut sum = 0.0f32;
        for i in 0..n {
            let imp = self.active_sum_per_component[i];
            let imp = if imp.is_finite() { imp.max(0.0) } else { 0.0 };
            let load = self.token_count_per_component[i] as f32;
            let pi = imp / total_importance;
            let li = load / total_load;
            sum += pi * li;
        }

        let loss = (n as f32) * sum;
        if loss.is_finite() { loss.max(0.0) } else { 0.0 }
    }

    /// Get sparsity loss for training (encourages minimal component usage)
    pub fn compute_sparsity_loss(&self, num_active: usize) -> f32 {
        let avg_components_per_token = num_active as f32;
        let target_sparsity = 1.0; // Target: 1 component per token on average
        (avg_components_per_token - target_sparsity).powi(2)
    }

    /// Get complexity alignment loss for training (aligns component usage with predicted
    /// complexity)
    pub fn compute_complexity_loss(&self, target_avg_components: f32) -> f32 {
        if self.active_sum_per_component.is_empty() || self.total_decisions == 0 {
            return 0.0;
        }

        let total_tokens = self.total_decisions as f32;
        let current_avg_components =
            self.active_sum_per_component.iter().sum::<f32>() / total_tokens;
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

        // token_count_per_component tracks, for each component, how many tokens had gate > 0.1.
        // Summing across components yields the total number of "significant" component activations
        // across all tokens.
        let total_significant: usize = self.token_count_per_component.iter().copied().sum();
        total_significant as f32 / self.total_decisions as f32
    }

    /// Get gating entropy (higher = more uniform distribution across components)
    pub fn get_gating_entropy(&self) -> f32 {
        if self.total_decisions == 0 {
            return 0.0;
        }

        // Calculate entropy of the average gate values using iterator chains
        let total_sum: f32 = self.active_sum_per_component.iter().sum();
        if !total_sum.is_finite() || total_sum <= 0.0 {
            return 0.0;
        }

        let neg_sum = self
            .active_sum_per_component
            .iter()
            .map(|&sum| {
                let s = if sum.is_finite() { sum.max(0.0) } else { 0.0 };
                s / total_sum
            })
            .filter(|&prob| prob.is_finite() && prob > 0.0)
            .map(|prob| prob * prob.ln())
            .sum::<f32>();

        // H = -∑ p ln(p)
        let h = -neg_sum;
        if h.is_finite() { h.max(0.0) } else { 0.0 }
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

        let counts: Vec<f32> = self
            .token_count_per_component
            .iter()
            .map(|&c| c as f32)
            .collect();

        let mean = counts.iter().sum::<f32>() / counts.len() as f32;

        if mean == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let variance =
            counts.iter().map(|&c| (c - mean).powi(2)).sum::<f32>() / counts.len() as f32;

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
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![
                0.8, 0.1, 0.1, // Token 1: component 0 active
                0.1, 0.9, 0.1, // Token 2: component 1 active
            ],
        )
        .unwrap();

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
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 2),
            vec![
                0.2, 0.8, // Token 1
                0.5, 0.3, // Token 2
            ],
        )
        .unwrap();

        metrics.update(&gate_values.view());

        let (min, max, mean) = metrics.get_gate_stats();
        assert_eq!(min, 0.2);
        assert_eq!(max, 0.8);
        assert!((mean - 0.45).abs() < 1e-6); // (0.2+0.8+0.5+0.3)/4 = 0.45
    }

    #[test]
    fn test_get_avg_active_components() {
        let mut metrics = MixtureMetrics::new(2);
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 2),
            vec![
                0.3, 0.7, // Token 1: total 1.0
                0.4, 0.6, // Token 2: total 1.0
            ],
        )
        .unwrap();

        metrics.update(&gate_values.view());

        let avg = metrics.get_avg_active_components();
        assert!((avg - 1.0).abs() < 1e-6); // Should be 1.0 (normalized)
    }

    #[test]
    fn test_metrics_nan_inf_robustness() {
        let mut metrics = MixtureMetrics::new(3);
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 0.0, 0.0, f32::NAN, f32::INFINITY, 0.0],
        )
        .unwrap();

        metrics.update(&gate_values.view());

        assert!(metrics.get_avg_active_components().is_finite());
        assert!(metrics.get_avg_significant_components().is_finite());
        assert!(metrics.get_gating_entropy().is_finite());
    }

    #[test]
    fn test_get_avg_significant_components() {
        let mut metrics = MixtureMetrics::new(3);
        let gate_values = ndarray::Array2::from_shape_vec(
            (2, 3),
            vec![
                0.2, 0.2, 0.6, // 3 significant (>0.1)
                0.0, 0.15, 0.85, // 2 significant (>0.1)
            ],
        )
        .unwrap();

        metrics.update(&gate_values.view());

        // (3 + 2) / 2 tokens = 2.5
        let avg = metrics.get_avg_significant_components();
        assert!((avg - 2.5).abs() < 1e-6);
    }
}
