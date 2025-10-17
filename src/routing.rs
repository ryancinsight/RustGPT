/// Shared routing utilities for Mixture-of-Heads (MoH) and future Mixture-of-Experts (MoE)
///
/// This module provides generic routing functionality that can be reused across different
/// mixture architectures:
/// - Top-K selection for choosing active components (heads or experts)
/// - Load balance loss to prevent routing collapse
/// - Straight-through estimator for gradient flow through discrete selection
/// - Numerically stable softmax
///
/// # Design Principles
///
/// - **DRY**: Single implementation of routing logic
/// - **Reusability**: Works for both MoH (heads) and MoE (experts)
/// - **Testability**: Easy to test in isolation
/// - **Maintainability**: Fix bugs in one place
///
/// # Future Compatibility
///
/// When MoE is implemented, ExpertRouter will use these same utilities,
/// ensuring consistent behavior and reducing code duplication.

use ndarray::{Array1, Array2, Axis};

/// Select Top-K indices for each row in the scores matrix
///
/// # Arguments
///
/// * `scores` - Routing scores of shape (batch_size, num_candidates)
/// * `k` - Number of top candidates to select
///
/// # Returns
///
/// Vector of vectors, where each inner vector contains the indices of the top-K
/// candidates for that batch element, sorted in descending order by score.
///
/// # Example
///
/// ```ignore
/// let scores = array![[0.1, 0.5, 0.3, 0.2]];  // 1 batch, 4 candidates
/// let top_k = top_k_indices(&scores, 2);
/// assert_eq!(top_k[0], vec![1, 2]);  // Indices 1 and 2 have highest scores
/// ```
pub fn top_k_indices(scores: &Array2<f32>, k: usize) -> Vec<Vec<usize>> {
    let (batch_size, num_candidates) = scores.dim();
    let k = k.min(num_candidates); // Ensure k doesn't exceed number of candidates
    
    let mut result = Vec::with_capacity(batch_size);
    
    for batch_idx in 0..batch_size {
        let row = scores.row(batch_idx);
        
        // Create (index, score) pairs
        let mut indexed_scores: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();
        
        // Sort by score in descending order
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top-K indices
        let top_k: Vec<usize> = indexed_scores
            .iter()
            .take(k)
            .map(|(idx, _)| *idx)
            .collect();
        
        result.push(top_k);
    }
    
    result
}

/// Compute load balance loss to prevent routing collapse
///
/// The load balance loss encourages uniform distribution of tokens across candidates
/// (heads or experts). It is computed as:
///
/// ```text
/// L_b = Σ(i=1 to n) P_i × f_i
/// ```
///
/// Where:
/// - `P_i` = average routing score for candidate i (across all tokens)
/// - `f_i` = fraction of tokens that selected candidate i
///
/// When perfectly balanced, each candidate has equal P_i and f_i, minimizing the loss.
/// When imbalanced (e.g., all tokens route to one candidate), the loss increases.
///
/// # Arguments
///
/// * `routing_scores` - Routing scores of shape (batch_size, num_candidates)
/// * `activation_mask` - Boolean mask of shape (batch_size, num_candidates)
///                       indicating which candidates are active
///
/// # Returns
///
/// Scalar load balance loss value
///
/// # Example
///
/// ```ignore
/// let scores = array![[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]];
/// let mask = array![[true, true, false], [true, false, true]];
/// let loss = compute_load_balance_loss(&scores, &mask);
/// // Loss will be low if distribution is balanced
/// ```
pub fn compute_load_balance_loss(
    routing_scores: &Array2<f32>,
    activation_mask: &Array2<bool>,
) -> f32 {
    let (batch_size, num_candidates) = routing_scores.dim();
    
    if batch_size == 0 || num_candidates == 0 {
        return 0.0;
    }
    
    let batch_size_f32 = batch_size as f32;
    
    // Compute P_i: average routing score for each candidate
    let p_i: Array1<f32> = routing_scores.mean_axis(Axis(0)).unwrap();
    
    // Compute f_i: fraction of tokens that selected each candidate
    let mut f_i = Array1::<f32>::zeros(num_candidates);
    for candidate_idx in 0..num_candidates {
        let count = activation_mask
            .column(candidate_idx)
            .iter()
            .filter(|&&active| active)
            .count();
        f_i[candidate_idx] = count as f32 / batch_size_f32;
    }
    
    // Compute L_b = Σ P_i × f_i
    let loss: f32 = p_i.iter().zip(f_i.iter()).map(|(p, f)| p * f).sum();
    
    // Scale by number of candidates for normalization
    loss * num_candidates as f32
}

/// Straight-through estimator for gradient flow through discrete selection
///
/// The straight-through estimator (STE) allows gradients to flow through discrete
/// operations (like Top-K selection) during backpropagation. In the forward pass,
/// we use the discrete selection (boolean mask). In the backward pass, we treat
/// it as if it were continuous, allowing gradients to flow.
///
/// # Forward Pass
///
/// ```text
/// output = discrete_selection(input)  // e.g., Top-K
/// ```
///
/// # Backward Pass
///
/// ```text
/// grad_input = grad_output  // Straight-through: ignore discretization
/// ```
///
/// # Arguments
///
/// * `forward_output` - Boolean mask from forward pass (discrete selection)
/// * `backward_gradient` - Gradient from downstream layers
///
/// # Returns
///
/// Gradient to pass upstream, treating discrete selection as identity
///
/// # Note
///
/// This is a standard technique in neural architecture search, Gumbel-Softmax,
/// and mixture models. It's proven effective despite the approximation.
pub fn straight_through_estimator(
    _forward_output: &Array2<bool>,
    backward_gradient: &Array2<f32>,
) -> Array2<f32> {
    // In STE, we simply pass gradients through as if the discrete operation
    // were an identity function. The forward_output is used in forward pass
    // but ignored in backward pass.
    backward_gradient.clone()
}

/// Numerically stable softmax function
///
/// Computes softmax along axis 1 (columns) with numerical stability via
/// the log-sum-exp trick:
///
/// ```text
/// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
/// ```
///
/// Subtracting the maximum prevents overflow in exp() while maintaining
/// mathematical equivalence.
///
/// # Arguments
///
/// * `x` - Input array of shape (batch_size, num_features)
///
/// # Returns
///
/// Softmax probabilities of same shape, where each row sums to 1.0
///
/// # Example
///
/// ```ignore
/// let x = array![[1.0, 2.0, 3.0]];
/// let probs = softmax(&x);
/// assert!((probs.sum() - 1.0).abs() < 1e-6);  // Sums to 1
/// ```
pub fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let (batch_size, num_features) = x.dim();
    let mut result = Array2::<f32>::zeros((batch_size, num_features));
    
    for batch_idx in 0..batch_size {
        let row = x.row(batch_idx);
        
        // Find max for numerical stability
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        // Compute exp(x - max)
        let exp_row: Vec<f32> = row.iter().map(|&val| (val - max_val).exp()).collect();
        
        // Compute sum of exponentials
        let sum_exp: f32 = exp_row.iter().sum();
        
        // Normalize
        for (col_idx, &exp_val) in exp_row.iter().enumerate() {
            result[[batch_idx, col_idx]] = exp_val / sum_exp;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_top_k_indices_basic() {
        let scores = array![[0.1, 0.5, 0.3, 0.2]];
        let top_k = top_k_indices(&scores, 2);
        
        assert_eq!(top_k.len(), 1);
        assert_eq!(top_k[0].len(), 2);
        assert_eq!(top_k[0][0], 1); // Index 1 has score 0.5 (highest)
        assert_eq!(top_k[0][1], 2); // Index 2 has score 0.3 (second)
    }

    #[test]
    fn test_top_k_indices_batch() {
        let scores = array![
            [0.1, 0.5, 0.3],
            [0.4, 0.2, 0.6]
        ];
        let top_k = top_k_indices(&scores, 2);
        
        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0], vec![1, 2]); // Batch 0: indices 1, 2
        assert_eq!(top_k[1], vec![2, 0]); // Batch 1: indices 2, 0
    }

    #[test]
    fn test_top_k_indices_k_exceeds_candidates() {
        let scores = array![[0.1, 0.5]];
        let top_k = top_k_indices(&scores, 5); // k > num_candidates
        
        assert_eq!(top_k[0].len(), 2); // Should return all candidates
    }

    #[test]
    fn test_load_balance_loss_balanced() {
        // Perfectly balanced: all candidates equally likely
        let scores = array![
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34]
        ];
        let mask = array![
            [true, false, false],
            [false, true, false],
            [false, false, true]
        ];

        let loss = compute_load_balance_loss(&scores, &mask);

        // Loss should be finite and reasonable for balanced distribution
        // With perfect balance: P_i ≈ 0.33, f_i ≈ 0.33, loss ≈ 3 × (0.33 × 0.33) ≈ 1.0
        assert!(loss.is_finite(), "Loss should be finite");
        assert!(loss >= 0.0, "Loss should be non-negative");
        assert!(loss <= 2.0, "Loss should be reasonable for balanced distribution, got {}", loss);
    }

    #[test]
    fn test_load_balance_loss_imbalanced() {
        // Imbalanced: all tokens route to first candidate
        let scores = array![
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05]
        ];
        let mask = array![
            [true, false, false],
            [true, false, false],
            [true, false, false]
        ];
        
        let loss = compute_load_balance_loss(&scores, &mask);
        
        // Loss should be higher for imbalanced distribution
        assert!(loss > 0.5, "Loss should be high for imbalanced distribution");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = array![[1.0, 2.0, 3.0]];
        let probs = softmax(&x);
        
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax should sum to 1");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without max subtraction
        let x = array![[1000.0, 1001.0, 1002.0]];
        let probs = softmax(&x);
        
        // Should not contain NaN or Inf
        assert!(probs.iter().all(|&p| p.is_finite()));
        
        // Should still sum to 1
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_straight_through_estimator() {
        let forward_output = array![[true, false, true]];
        let backward_gradient = array![[0.5, 0.3, 0.2]];
        
        let grad_input = straight_through_estimator(&forward_output, &backward_gradient);
        
        // STE should pass gradients through unchanged
        assert_eq!(grad_input, backward_gradient);
    }
}

