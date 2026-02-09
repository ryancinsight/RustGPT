use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};

/// Gradients container for CoPE
#[derive(Clone, Debug)]
pub struct CoPEGradients {
    pub pos_embeddings: Option<Array2<f32>>,
}

impl CoPEGradients {
    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        Self {
            pos_embeddings: Some(Array2::zeros((max_pos + 1, embed_dim))),
        }
    }

    pub fn accumulate(&mut self, other: &CoPEGradients) {
        if let (Some(s), Some(o)) = (&mut self.pos_embeddings, &other.pos_embeddings) {
            *s += o;
        }
    }
}

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
        let mut rng = get_rng();
        let normal_pe = Normal::new(0.0, 0.02).unwrap();
        let pe =
            Array2::<f32>::from_shape_fn((max_pos + 1, embed_dim), |_| normal_pe.sample(&mut rng));
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

    /// Apply gradients from a slice of arrays
    pub fn apply_gradients_from_slice(&mut self, grads: &[Array2<f32>], lr: f32) {
        if !grads.is_empty() {
            self.optimizer
                .step(&mut self.pos_embeddings, &grads[0], lr);
        }
    }

    /// Initialize gradients structure
    pub fn init_gradients(&self) -> CoPEGradients {
        CoPEGradients::new(self.max_pos, self.pos_embeddings.ncols())
    }

    /// Get the contribution (gradient magnitude) for a position
    pub fn get_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        _k: &ArrayView1<'_, f32>,
        i: usize,
        j: usize,
        _inputs: Option<&ArrayView2<'_, f32>>,
    ) -> f32 {
        let pos = i.saturating_sub(j);
        if pos <= self.max_pos {
            q.dot(&self.pos_embeddings.row(pos))
        } else {
            0.0
        }
    }

    /// Backward pass for CoPE
    pub fn backward(
        &self,
        q: &ArrayView1<'_, f32>,
        _k: &ArrayView1<'_, f32>,
        i: usize,
        j: usize,
        _inputs: Option<&ArrayView2<'_, f32>>,
        d_s_ij: f32,
        grads: &mut CoPEGradients,
    ) -> (Array1<f32>, Array1<f32>) {
        let pos = i.saturating_sub(j);
        let mut dq = Array1::zeros(q.dim());
        let dk = Array1::zeros(q.dim());

        if pos <= self.max_pos {
            // Legacy CoPE gradient: s += q dot P[pos]
            // dL/dq = d_s * P[pos]
            let p_emb = self.pos_embeddings.row(pos);
            for (d, &p) in dq.iter_mut().zip(p_emb.iter()) {
                *d += p * d_s_ij;
            }

            // dL/dP[pos] = d_s * q
            if let Some(grad_pe) = &mut grads.pos_embeddings {
                let mut row = grad_pe.row_mut(pos);
                for (r, &q_val) in row.iter_mut().zip(q.iter()) {
                    *r += q_val * d_s_ij;
                }
            }
        }

        (dq, dk)
    }

    /// Get the number of parameters in this CoPE instance
    pub fn parameters(&self) -> usize {
        self.pos_embeddings.len()
    }

    /// Get the weight norm (L2 norm) of the positional embeddings
    pub fn weight_norm(&self) -> f32 {
        self.pos_embeddings
            .iter()
            .map(|&w| w * w)
            .sum::<f32>()
            .sqrt()
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
