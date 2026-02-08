use ndarray::{Array1, Array2, ArrayView1};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{infrastructure::optimizer::adam::Adam, common::rng::get_rng};

/// Factorized Contextual Position Embeddings (FactorizedCoPE)
///
/// Reduces memory footprint by factorizing position embeddings using
/// low-rank decomposition: PE = U @ V where U ∈ ℝ^(max_pos × r), V ∈ ℝ^(r × embed_dim)
///
/// Mathematical Formulation:
/// ```
/// CoPE(q, pos) = q · PE[pos] = q · (U[pos, :] @ V)
///
/// where PE[pos] = U[pos, :] (1×r) @ V (r×embed_dim) = (1×embed_dim)
///
/// = (U[pos, :] @ V^T) · q
/// = (U[pos, :] @ V^T @ q^T)
/// = (U[pos, :] @ (V @ q)^T)
/// ```
///
/// Benefits:
/// - O(max_pos × r) parameters instead of O(max_pos × embed_dim)
/// - r << embed_dim for significant memory savings
/// - Still learns position-dependent embeddings
/// - Log1p-style formulation for numerical stability
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FactorizedCoPE {
    /// Up projection: (max_pos, rank) - position factors
    up_proj: Array2<f32>,
    opt_up_proj: Adam,
    /// Down projection: (rank, embed_dim) - embedding factors  
    down_proj: Array2<f32>,
    opt_down_proj: Adam,
    /// Rank for factorization (r << embed_dim)
    rank: usize,
    /// Maximum position
    max_pos: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Temperature for normalized scaling
    temperature: f32,
}

impl FactorizedCoPE {
    /// Create a new FactorizedCoPE instance
    ///
    /// # Arguments
    /// * `max_pos` - Maximum position to handle
    /// * `embed_dim` - Embedding dimension
    /// * `rank` - Factorization rank (recommended: embed_dim / 4)
    pub fn new(max_pos: usize, embed_dim: usize, rank: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02 / (rank as f32).sqrt()).unwrap();
        
        let up_proj = Array2::from_shape_fn((max_pos + 1, rank), |_| normal.sample(&mut rng));
        let opt_up_proj = Adam::new((max_pos + 1, rank));
        
        let down_proj = Array2::from_shape_fn((rank, embed_dim), |_| normal.sample(&mut rng));
        let opt_down_proj = Adam::new((rank, embed_dim));

        Self {
            up_proj,
            opt_up_proj,
            down_proj,
            opt_down_proj,
            rank,
            max_pos,
            embed_dim,
            temperature: 1.0,
        }
    }

    /// Create with custom temperature
    pub fn with_temperature(max_pos: usize, embed_dim: usize, rank: usize, temp: f32) -> Self {
        let mut instance = Self::new(max_pos, embed_dim, rank);
        instance.temperature = temp;
        instance
    }

    /// Compute factorized CoPE contribution using log1p formulation
    ///
    /// Computes: `log(1 + exp(U[pos] @ V @ q))` for numerical stability
    /// This provides smooth gradients and prevents vanishing gradients.
    ///
    /// # Arguments
    /// * `q` - Query vector (embed_dim,)
    /// * `pos` - Relative position
    ///
    /// # Returns
    /// Factorized CoPE contribution with log1p stabilization
    #[inline]
    pub fn factorized_cope_contribution(&self, q: &ArrayView1<'_, f32>, pos: usize) -> f32 {
        if pos > self.max_pos {
            return 0.0;
        }

        // Compute intermediate: V @ q^T (rank,)
        let mut vq = Array1::zeros(self.rank);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj, q, 0.0, &mut vq);

        // Compute: U[pos] @ (V @ q) (scalar)
        let up_row = self.up_proj.row(pos);
        let raw = up_row.dot(&vq);

        // Log1p formulation: log(1 + exp(x / T)) * T
        // This provides smooth, well-behaved gradients
        let scaled = raw / self.temperature;
        let stable = self.log1p_exp(scaled);
        stable * self.temperature
    }

    /// Numerically stable log(1 + exp(x))
    ///
    /// Uses the identity: log(1 + exp(x)) = softplus(x)
    /// With numerical safeguards for large |x|
    #[inline]
    fn log1p_exp(&self, x: f32) -> f32 {
        // For large positive x: log(1 + exp(x)) ≈ x
        // For large negative x: log(1 + exp(x)) ≈ exp(x) (very small)
        if x > 20.0 {
            x  // exp(x) dominates, log(1 + exp(x)) ≈ x
        } else if x < -20.0 {
            x.exp()  // exp(x) is very small, log(1 + exp(x)) ≈ exp(x)
        } else {
            (1.0 + x.exp()).ln()  // Standard computation
        }
    }

    /// Compute embedding for a specific position
    ///
    /// Returns the reconstructed position embedding: U[pos] @ V
    #[inline]
    pub fn get_embedding(&self, pos: usize) -> Option<Array1<f32>> {
        if pos > self.max_pos {
            return None;
        }
        let up_row = self.up_proj.row(pos);
        let mut embedding = Array1::zeros(self.embed_dim);
        // U[pos] (1,r) @ V.T (embed_dim, r).T = U[pos] @ V.T = (1, embed_dim)
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj.t(), &up_row.to_owned(), 0.0, &mut embedding);
        Some(embedding)
    }

    /// Get rank used for factorization
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let full_params = (self.max_pos + 1) * self.embed_dim;
        let factored_params = self.up_proj.len() + self.down_proj.len();
        full_params as f32 / factored_params as f32
    }

    /// Apply gradients
    pub fn apply_gradients(
        &mut self,
        grads: &(Array2<f32>, Array2<f32>),  // (up_grad, down_grad)
        lr: f32,
    ) {
        self.opt_up_proj.step(&mut self.up_proj, &grads.0, lr);
        self.opt_down_proj.step(&mut self.down_proj, &grads.1, lr);
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        self.up_proj.len() + self.down_proj.len()
    }

    /// Get weight norm
    pub fn weight_norm(&self) -> f32 {
        let up_norm: f32 = self.up_proj.iter().map(|&w| w * w).sum::<f32>().sqrt();
        let down_norm: f32 = self.down_proj.iter().map(|&w| w * w).sum::<f32>().sqrt();
        (up_norm.powi(2) + down_norm.powi(2)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorized_cope_creation() {
        let fc = FactorizedCoPE::new(128, 64, 16);
        assert_eq!(fc.max_pos, 128);
        assert_eq!(fc.embed_dim, 64);
        assert_eq!(fc.rank, 16);
        assert_eq!(fc.up_proj.shape(), &[129, 16]);
        assert_eq!(fc.down_proj.shape(), &[16, 64]);
    }

    #[test]
    fn test_factorized_cope_contribution() {
        let fc = FactorizedCoPE::new(128, 64, 16);
        let q = Array1::from_elem(64, 0.5);
        
        let contrib0 = fc.factorized_cope_contribution(&q.view(), 0);
        let contrib100 = fc.factorized_cope_contribution(&q.view(), 100);
        
        assert!(contrib0.is_finite());
        assert!(contrib100.is_finite());
    }

    #[test]
    fn test_compression_ratio() {
        let fc = FactorizedCoPE::new(512, 128, 32);
        let ratio = fc.compression_ratio();
        
        // Full: 513 * 128 = 65664
        // Factored: 513*32 + 32*128 = 16416 + 4096 = 20512
        // Ratio: 65664 / 20512 ≈ 3.2
        assert!(ratio > 2.0);
    }

    #[test]
    fn test_log1p_stability() {
        let fc = FactorizedCoPE::with_temperature(64, 32, 8, 1.0);
        
        // Test numerical stability for extreme values
        let large_pos = fc.log1p_exp(100.0);
        let large_neg = fc.log1p_exp(-100.0);
        
        // log(1 + exp(100)) ≈ 100
        assert!(large_pos.is_finite());
        assert!((large_pos - 100.0).abs() < 1e-10, "Expected ~100, got {}", large_pos);
        
        // log(1 + exp(-100)) ≈ exp(-100) (very small but finite)
        assert!(large_neg.is_finite());
        assert!(large_neg > 0.0);
        assert!(large_neg < 1e-10, "Expected very small, got {}", large_neg);
    }

    #[test]
    fn test_embedding_reconstruction() {
        let fc = FactorizedCoPE::new(64, 32, 8);
        let emb = fc.get_embedding(0);
        
        assert!(emb.is_some());
        let emb = emb.unwrap();
        assert_eq!(emb.len(), 32);
        assert!(emb.iter().all(|&x| x.is_finite()));
    }
}
