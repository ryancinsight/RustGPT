//! # Shared Routing Logic for Mixture Models
//!
//! This module provides shared routing and selection logic for dynamic mixture models,
//! including Mixture-of-Heads (MoH) and Mixture-of-Experts (MoE).
//!
//! ## Overview
//!
//! Centralizes common routing patterns and selection algorithms used across different
//! mixture model implementations. This promotes reusability and consistency.
//!
//! ## Key Components
//!
//! - **Router**: Trait for routing implementations
//! - **SelectionAlgorithm**: Common selection algorithms (TopK, Softmax, etc.)
//! - **RoutingConfig**: Shared configuration for routing behavior

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use crate::{mixtures::threshold::ThresholdPredictor, softmax::Softmax};

/// Common selection algorithms for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionAlgorithm {
    /// Select top-k components with highest gating values (hard selection)
    TopK { k: usize },
    /// Apply differentiable soft top-p sampling (AutoDeco-inspired)
    SoftTopP { top_p: f32 },
    /// Apply softmax to gating values for soft routing probabilities
    Softmax,
    /// Use raw gating values without modification
    Raw,
}

/// Configuration for routing behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Selection algorithm to use
    pub algorithm: SelectionAlgorithm,
    /// Whether to use learned predictor for gating values
    pub use_learned_predictor: bool,
    /// Number of components to route to/select
    pub num_active: usize,
    /// Temperature for softmax (only used with Softmax algorithm)
    pub temperature: f32,
    /// Steepness parameter for soft top-p decay (only used with SoftTopP algorithm)
    pub soft_top_p_alpha: f32,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            algorithm: SelectionAlgorithm::TopK { k: 1 },
            use_learned_predictor: false,
            num_active: 1,
            temperature: 1.0,
            soft_top_p_alpha: 50.0,
        }
    }
}

/// Result of a routing decision
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Routing decisions: shape (num_tokens, num_components)
    /// For TopK: binary mask, for Softmax: probabilities
    pub routing_weights: ndarray::Array2<f32>,
    /// Raw gating values before selection: shape (num_tokens, num_components)
    pub raw_gates: ndarray::Array2<f32>,
}

/// Trait for routing implementations
pub trait Router {
    /// Route input tokens to components
    ///
    /// # Arguments
    /// * `input` - Token embeddings: shape (num_tokens, embed_dim)
    /// * `predictor` - Optional threshold predictor for learned routing
    ///
    /// # Returns
    /// RoutingResult containing routing weights and raw gates
    fn route(
        &mut self,
        input: &ArrayView2<f32>,
        predictor: Option<&mut ThresholdPredictor>,
    ) -> RoutingResult;
}

/// Apply selection algorithm to raw gating values
pub fn apply_selection_algorithm(
    raw_gates: &ndarray::ArrayView2<f32>,
    config: &RoutingConfig,
) -> ndarray::Array2<f32> {
    match &config.algorithm {
        SelectionAlgorithm::TopK { k } => apply_top_k_selection(raw_gates, *k),
        SelectionAlgorithm::SoftTopP { top_p } => {
            apply_soft_top_p_selection(raw_gates, *top_p, config.soft_top_p_alpha)
        }
        SelectionAlgorithm::Softmax => apply_softmax_selection(raw_gates, config.temperature),
        SelectionAlgorithm::Raw => raw_gates.to_owned(),
    }
}

/// Apply top-k selection to gating values
/// Returns binary mask where 1 indicates selected component
fn apply_top_k_selection(gates: &ndarray::ArrayView2<f32>, k: usize) -> ndarray::Array2<f32> {
    let mut result = ndarray::Array2::<f32>::zeros(gates.raw_dim());

    // Process each token using iterator chains
    gates
        .outer_iter()
        .enumerate()
        .for_each(|(token_idx, token_gates)| {
            // Create sorted indices by value (descending) using iterator chains
            let mut indices: Vec<usize> = (0..token_gates.len()).collect();
            indices.sort_by(|&a, &b| {
                let va = token_gates[a];
                let vb = token_gates[b];
                let va = if va.is_finite() { va } else { f32::NEG_INFINITY };
                let vb = if vb.is_finite() { vb } else { f32::NEG_INFINITY };
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Set top-k indices to 1.0
            indices.into_iter().take(k).for_each(|idx| {
                result[[token_idx, idx]] = 1.0;
            });
        });

    result
}

/// Apply soft top-p selection to gating values (AutoDeco-inspired)
/// Returns differentiable probability distribution using soft top-p sampling
fn apply_soft_top_p_selection(
    gates: &ndarray::ArrayView2<f32>,
    top_p: f32,
    alpha: f32,
) -> ndarray::Array2<f32> {
    let mut result = ndarray::Array2::<f32>::zeros(gates.raw_dim());

    // Process each token
    for (token_idx, token_gates) in gates.outer_iter().enumerate() {
        // Convert to 1D array for processing
        let token_gates_1d = token_gates.as_slice().unwrap();

        // Sort probabilities and compute cumulative sum (following AutoDeco approach)
        let mut prob_indices: Vec<usize> = (0..token_gates_1d.len()).collect();
        prob_indices.sort_by(|&i, &j| token_gates_1d[j].partial_cmp(&token_gates_1d[i]).unwrap());

        let mut sorted_probs = Vec::with_capacity(token_gates_1d.len());
        for &idx in &prob_indices {
            sorted_probs.push(token_gates_1d[idx]);
        }

        // Compute cumulative sum
        let mut cumulative = Vec::with_capacity(sorted_probs.len());
        let mut sum = 0.0;
        for &val in &sorted_probs {
            sum += val;
            cumulative.push(sum);
        }

        // Apply soft mask: exp(-α * ReLU(cumulative - top_p)) using PadeExp
        let mut soft_mask = Vec::with_capacity(cumulative.len());
        for &c in &cumulative {
            let relu_val = (c - top_p).max(0.0);
            soft_mask.push(crate::richards::PadeExp::exp((-alpha * relu_val) as f64) as f32);
        }

        // Unsort the mask
        let mut unsorted_mask = vec![0.0; token_gates_1d.len()];
        for (i, &idx) in prob_indices.iter().enumerate() {
            unsorted_mask[idx] = soft_mask[i];
        }

        // Apply mask and renormalize
        let mut masked_probs = Vec::with_capacity(token_gates_1d.len());
        for (i, &prob) in token_gates_1d.iter().enumerate() {
            masked_probs.push(prob * unsorted_mask[i]);
        }

        let sum_masked: f32 = masked_probs.iter().sum();
        if sum_masked > 0.0 {
            for (i, prob) in masked_probs.into_iter().enumerate() {
                result[[token_idx, i]] = prob / sum_masked;
            }
        } else {
            // Fallback to original if all masked
            for (i, &prob) in token_gates_1d.iter().enumerate() {
                result[[token_idx, i]] = prob;
            }
        }
    }

    result
}

/// Apply softmax selection to gating values
/// Returns probability distribution over components
fn apply_softmax_selection(
    gates: &ndarray::ArrayView2<f32>,
    temperature: f32,
) -> ndarray::Array2<f32> {
    let mut softmax = Softmax::new();

    let temperature = if temperature.is_finite() && temperature > 1e-6 {
        temperature
    } else {
        1.0
    };

    // Scale gates by temperature
    let scaled_gates = gates.mapv(|x| {
        let x = if x.is_finite() { x } else { 0.0 };
        x / temperature
    });

    // Apply softmax using the existing implementation
    softmax.forward(&scaled_gates.view())
}

/// Compute routing entropy for a batch of routing decisions
pub fn compute_routing_entropy(routing_weights: &ndarray::ArrayView2<f32>) -> f32 {
    if routing_weights.nrows() == 0 {
        return 0.0;
    }
    let num_tokens = routing_weights.nrows() as f32;

    // Use iterator chains for zero-copy entropy computation
    routing_weights
        .outer_iter()
        .map(|token_weights| {
            token_weights
                .iter()
                .filter(|&&weight| weight > 0.0)
                .map(|&weight| weight * weight.ln())
                .sum::<f32>()
        })
        .sum::<f32>()
        .abs()
        / num_tokens
}

/// Get average number of active components per token
pub fn compute_avg_active_components(routing_weights: &ndarray::ArrayView2<f32>) -> f32 {
    if routing_weights.nrows() == 0 {
        return 0.0;
    }
    // Use iterator chains for zero-copy computation
    routing_weights
        .outer_iter()
        .map(|token_weights| token_weights.iter().filter(|&&w| w > 0.1).count() as f32)
        .sum::<f32>()
        / routing_weights.nrows() as f32
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_top_k_selection() {
        let gates = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.1, 0.5, 0.3, // token 0: should select idx 1
                0.8, 0.2, 0.4, // token 1: should select idx 0
            ],
        )
        .unwrap();

        let result = apply_top_k_selection(&gates.view(), 1);

        assert_eq!(result[[0, 0]], 0.0); // not selected
        assert_eq!(result[[0, 1]], 1.0); // selected
        assert_eq!(result[[0, 2]], 0.0); // not selected

        assert_eq!(result[[1, 0]], 1.0); // selected
        assert_eq!(result[[1, 1]], 0.0); // not selected
        assert_eq!(result[[1, 2]], 0.0); // not selected
    }

    #[test]
    fn test_soft_top_p_selection() {
        let gates = Array2::from_shape_vec(
            (1, 4),
            vec![
                0.4, 0.3, 0.2, 0.1, // Single token with decreasing probabilities
            ],
        )
        .unwrap();

        // Test with top_p = 0.7 (should keep top 2 components: 0.4 + 0.3 = 0.7)
        let result = apply_soft_top_p_selection(&gates.view(), 0.7, 50.0);

        // Check that result is properly normalized
        let total: f32 = result.row(0).iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "Soft top-p result should be normalized, got {}",
            total
        );

        // With high alpha (50.0), the third component should be almost zero
        assert!(
            result[[0, 2]] < 0.01,
            "Third component should be heavily penalized"
        );

        // First two components should have non-zero probability
        assert!(
            result[[0, 0]] > 0.0,
            "First component should have positive probability"
        );
        assert!(
            result[[0, 1]] > 0.0,
            "Second component should have positive probability"
        );

        // Test with top_p = 1.0 (should keep all components)
        let result_all = apply_soft_top_p_selection(&gates.view(), 1.0, 50.0);
        let total_all: f32 = result_all.row(0).iter().sum();
        assert!(
            (total_all - 1.0).abs() < 1e-6,
            "Soft top-p with top_p=1.0 should be normalized"
        );
    }

    #[test]
    fn test_softmax_selection() {
        let gates = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap();
        let result = apply_softmax_selection(&gates.view(), 1.0);

        // Should be approximately [0.269, 0.731]
        assert!(result[[0, 0]] > 0.2 && result[[0, 0]] < 0.3);
        assert!(result[[0, 1]] > 0.7 && result[[0, 1]] < 0.8);

        // Should sum to 1
        let sum: f32 = result.row(0).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_routing_entropy() {
        // Uniform distribution should have higher entropy
        let uniform = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let uniform_entropy = compute_routing_entropy(&uniform.view());

        // Single component should have zero entropy
        let single = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let single_entropy = compute_routing_entropy(&single.view());

        assert!(uniform_entropy > single_entropy);
        assert!(single_entropy < 1e-6); // approximately 0
    }
}
