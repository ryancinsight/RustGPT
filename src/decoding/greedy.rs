//! # Greedy Decoder
//!
//! This module implements greedy decoding, the simplest decoding strategy that
//! always selects the most likely token at each step.
//!
//! ## Algorithm
//!
//! For each position in the sequence:
//! 1. Take the probability distribution over vocabulary
//! 2. Select the token with highest probability
//! 3. Return the selected token indices
//!
//! ## Characteristics
//!
//! - **Deterministic**: Always produces the same output for same input
//! - **Fast**: Minimal computational overhead
//! - **Simple**: Easy to understand and implement
//! - **Limited Diversity**: No exploration of alternative sequences

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Greedy decoder that always selects the most probable token
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GreedyDecoder;

impl GreedyDecoder {
    /// Create a new greedy decoder
    pub fn new() -> Self {
        Self
    }

    /// Decode a batch of probability distributions using greedy selection
    ///
    /// # Arguments
    /// * `probs` - Probability distributions of shape (batch_size, vocab_size)
    ///
    /// # Returns
    /// Vector of selected token indices, one per batch element
    #[inline]
    pub fn decode(&self, probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(ndarray::Axis(1), |row| {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;
                for (i, &val) in row.iter().enumerate() {
                    if val > max_val || (val == max_val && i < max_idx) {
                        max_val = val;
                        max_idx = i;
                    }
                }
                max_idx
            })
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_greedy_decode_single() {
        let decoder = GreedyDecoder::new();
        let probs = Array2::from_shape_vec((1, 4), vec![0.1, 0.8, 0.05, 0.05]).unwrap();

        let result = decoder.decode(&probs);
        assert_eq!(result, vec![1]); // Should select index 1 (highest probability)
    }

    #[test]
    fn test_greedy_decode_batch() {
        let decoder = GreedyDecoder::new();
        let probs = Array2::from_shape_vec(
            (2, 3),
            vec![0.2, 0.7, 0.1,  // First sequence: index 1 should be selected
                 0.9, 0.05, 0.05] // Second sequence: index 0 should be selected
        ).unwrap();

        let result = decoder.decode(&probs);
        assert_eq!(result, vec![1, 0]);
    }

    #[test]
    fn test_greedy_decode_ties() {
        let decoder = GreedyDecoder::new();
        let probs = Array2::from_shape_vec((1, 3), vec![0.5, 0.5, 0.0]).unwrap();

        let result = decoder.decode(&probs);
        assert_eq!(result, vec![0]); // Should select first occurrence of max (index 0)
    }

    #[test]
    fn test_empty_probs() {
        let decoder = GreedyDecoder::new();
        let probs = Array2::from_shape_vec((0, 5), vec![]).unwrap();

        let result = decoder.decode(&probs);
        assert_eq!(result, Vec::<usize>::new());
    }
}
