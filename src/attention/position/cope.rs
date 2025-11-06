use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;

/// Contextual Position Embeddings (CoPE) for attention mechanisms.
/// CoPE provides position-aware attention by adding learnable positional
/// embeddings to attention logits based on relative positions.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CoPE {
    /// Maximum position to handle
    pub max_pos: usize,
    /// Learnable positional embeddings (max_pos+1, embed_dim)
    pub pos_embeddings: Array2<f32>,
    /// Optimizer for positional embeddings
    pub optimizer: Adam,
}

impl CoPE {
    /// Create a new CoPE instance
    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        let mut rng = rand::rng();
        let normal_pe = Normal::new(0.0, 0.02).unwrap();
        let pe = Array2::<f32>::from_shape_fn((max_pos + 1, embed_dim), |_| normal_pe.sample(&mut rng));
        let optimizer = Adam::new((max_pos + 1, embed_dim));

        Self {
            max_pos,
            pos_embeddings: pe,
            optimizer,
        }
    }


    /// Get the positional embedding for a specific position
    pub fn get_pos_embedding(&self, pos: usize) -> Option<ndarray::ArrayView1<'_, f32>> {
        if pos <= self.max_pos {
            Some(self.pos_embeddings.row(pos))
        } else {
            None
        }
    }

    /// Apply gradients to the positional embeddings
    pub fn apply_gradients(&mut self, grads: &Array2<f32>, lr: f32) {
        self.optimizer.step(&mut self.pos_embeddings, grads, lr);
    }

    /// Get the number of parameters in this CoPE instance
    pub fn parameters(&self) -> usize {
        self.pos_embeddings.len()
    }

    /// Get the weight norm (L2 norm) of the positional embeddings
    pub fn weight_norm(&self) -> f32 {
        self.pos_embeddings.iter().map(|&w| w * w).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cope_creation() {
        let cope = CoPE::new(10, 8);
        assert_eq!(cope.max_pos, 10);
        assert_eq!(cope.pos_embeddings.shape(), &[11, 8]); // max_pos + 1
    }

}
