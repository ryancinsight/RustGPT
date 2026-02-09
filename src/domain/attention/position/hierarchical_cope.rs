use ndarray::{Array2, ArrayView1};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};

/// Hierarchical Contextual Position Embeddings (HierarchicalCoPE)
///
/// Extends CoPE with multiple levels of positional granularity for better
/// generalization to longer sequences.
///
/// Structure:
/// - Local CoPE: Fine-grained positions (within-chunk), positions 0 to chunk_size-1
/// - Global CoPE: Chunk-level positions for document/sentence structure
/// - Chunk boundaries learned via content-aware clustering
///
/// Mathematical Formulation:
/// ```
/// pos_ij = chunk_idx * chunk_size + local_pos
/// CoPE_total = α_local * CoPE_local(local_pos) + α_global * CoPE_global(chunk_idx)
/// ```
///
/// Benefits:
/// - Better generalization to sequences longer than max_pos
/// - Learns natural chunking boundaries from data
/// - Reduced parameters for equivalent range
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HierarchicalCoPE {
    /// Chunk size for local positions
    chunk_size: usize,
    /// Maximum number of chunks
    max_chunks: usize,
    /// Local CoPE: positions within a chunk (chunk_size, embed_dim)
    local_cope: Array2<f32>,
    opt_local_cope: Adam,
    /// Global CoPE: chunk-level embeddings (max_chunks, embed_dim)
    global_cope: Array2<f32>,
    opt_global_cope: Adam,
    /// Learnable mixing weights for local vs global
    alpha_local: f32,
    alpha_global: f32,
    /// Chunk boundary predictor weights (embed_dim, 2)
    chunk_predictor_w: Array2<f32>,
    opt_chunk_predictor_w: Adam,
    /// Chunk predictor bias (1, 2) for Adam compatibility
    chunk_predictor_b: Array2<f32>,
    opt_chunk_predictor_b: Adam,
    /// Embedding dimension
    embed_dim: usize,
}

impl HierarchicalCoPE {
    /// Create a new HierarchicalCoPE instance
    pub fn new(chunk_size: usize, max_chunks: usize, embed_dim: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Local CoPE: learnable within-chunk positions
        let local_cope =
            Array2::from_shape_fn((chunk_size, embed_dim), |_| normal.sample(&mut rng));
        let opt_local_cope = Adam::new((chunk_size, embed_dim));

        // Global CoPE: chunk-level positions
        let global_cope =
            Array2::from_shape_fn((max_chunks, embed_dim), |_| normal.sample(&mut rng));
        let opt_global_cope = Adam::new((max_chunks, embed_dim));

        // Chunk boundary predictor
        let chunk_predictor_w = Array2::from_shape_fn((embed_dim, 2), |_| normal.sample(&mut rng));
        let opt_chunk_predictor_w = Adam::new((embed_dim, 2));

        let chunk_predictor_b = Array2::zeros((1, 2));
        let opt_chunk_predictor_b = Adam::new((1, 2));

        Self {
            chunk_size,
            max_chunks,
            local_cope,
            opt_local_cope,
            global_cope,
            opt_global_cope,
            alpha_local: 0.7, // Default: emphasize local positions more
            alpha_global: 0.3,
            chunk_predictor_w,
            opt_chunk_predictor_w,
            chunk_predictor_b,
            opt_chunk_predictor_b,
            embed_dim,
        }
    }

    /// Compute hierarchical CoPE contribution
    ///
    /// # Arguments
    /// * `q` - Query vector (embed_dim,)
    /// * `pos` - Relative position (0-indexed)
    /// * `content_sim` - Content similarity for chunk boundary detection
    ///
    /// # Returns
    /// Hierarchical CoPE contribution with learned chunking
    #[inline]
    pub fn hierarchical_cope_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        pos: usize,
        content_sim: f32,
    ) -> f32 {
        // Determine chunk index and local position
        let chunk_idx = (pos / self.chunk_size).min(self.max_chunks - 1);
        let local_pos = pos % self.chunk_size;

        // Predict chunk boundary adjustment
        let boundary_logit = q.dot(&self.chunk_predictor_w.column(0))
            + content_sim * self.chunk_predictor_w[[0, 1]]
            + self.chunk_predictor_b[[0, 0]];
        let boundary_gate = self.sigmoid(boundary_logit);

        // Local contribution: q · PE_local
        let local_contrib = q.dot(&self.local_cope.row(local_pos));

        // Global contribution: q · PE_global
        let global_contrib = q.dot(&self.global_cope.row(chunk_idx));

        // Blend with boundary-aware mixing
        let mixed = self.alpha_local * local_contrib + self.alpha_global * global_contrib;

        // Apply boundary adjustment (smooth transition near chunk boundaries)
        mixed * (1.0 + boundary_gate * 0.1)
    }

    /// Compute local and global components separately for gradient flow
    #[inline]
    pub fn hierarchical_components(&self, q: &ArrayView1<'_, f32>, pos: usize) -> (f32, f32) {
        let chunk_idx = (pos / self.chunk_size).min(self.max_chunks - 1);
        let local_pos = pos % self.chunk_size;

        let local_contrib = q.dot(&self.local_cope.row(local_pos));
        let global_contrib = q.dot(&self.global_cope.row(chunk_idx));

        (local_contrib, global_contrib)
    }

    /// Sigmoid with numerical stability
    #[inline]
    fn sigmoid(&self, x: f32) -> f32 {
        let x = x.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply gradients
    pub fn apply_gradients(
        &mut self,
        grads: &(
            Array2<f32>, // local_cope gradient
            Array2<f32>, // global_cope gradient
            Array2<f32>, // chunk_predictor_w gradient
            Array2<f32>, // chunk_predictor_b gradient
        ),
        lr: f32,
    ) {
        self.opt_local_cope.step(&mut self.local_cope, &grads.0, lr);
        self.opt_global_cope
            .step(&mut self.global_cope, &grads.1, lr);
        self.opt_chunk_predictor_w
            .step(&mut self.chunk_predictor_w, &grads.2, lr);
        self.opt_chunk_predictor_b
            .step(&mut self.chunk_predictor_b, &grads.3, lr);
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        self.local_cope.len()
            + self.global_cope.len()
            + self.chunk_predictor_w.len()
            + self.chunk_predictor_b.len()
    }

    /// Get weight norm
    pub fn weight_norm(&self) -> f32 {
        let local_norm: f32 = self.local_cope.iter().map(|&w| w * w).sum::<f32>().sqrt();
        let global_norm: f32 = self.global_cope.iter().map(|&w| w * w).sum::<f32>().sqrt();
        let pred_norm: f32 = self
            .chunk_predictor_w
            .iter()
            .map(|&w| w * w)
            .sum::<f32>()
            .sqrt();
        let bias_norm: f32 = self
            .chunk_predictor_b
            .iter()
            .map(|&w| w * w)
            .sum::<f32>()
            .sqrt();
        (local_norm.powi(2) + global_norm.powi(2) + pred_norm.powi(2) + bias_norm.powi(2)).sqrt()
    }

    /// Get maximum effective position
    pub fn effective_max_pos(&self) -> usize {
        self.chunk_size * self.max_chunks
    }

    /// Update mixing weights
    pub fn update_mixing(&mut self, alpha_local: f32, alpha_global: f32) {
        let sum = alpha_local + alpha_global;
        self.alpha_local = alpha_local / sum;
        self.alpha_global = alpha_global / sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_hierarchical_cope_creation() {
        let hcope = HierarchicalCoPE::new(64, 16, 32);
        assert_eq!(hcope.chunk_size, 64);
        assert_eq!(hcope.max_chunks, 16);
        assert_eq!(hcope.local_cope.shape(), &[64, 32]);
        assert_eq!(hcope.global_cope.shape(), &[16, 32]);
        assert_eq!(hcope.chunk_predictor_w.shape(), &[32, 2]);
        assert_eq!(hcope.chunk_predictor_b.shape(), &[1, 2]);
    }

    #[test]
    fn test_hierarchical_cope_contribution() {
        let hcope = HierarchicalCoPE::new(64, 16, 32);
        let q = Array1::from_elem(32, 0.5);

        // Test various positions
        let contrib0 = hcope.hierarchical_cope_contribution(&q.view(), 0, 0.0);
        let contrib63 = hcope.hierarchical_cope_contribution(&q.view(), 63, 0.0);
        let contrib64 = hcope.hierarchical_cope_contribution(&q.view(), 64, 0.0);
        let contrib1000 = hcope.hierarchical_cope_contribution(&q.view(), 1000, 0.5);

        assert!(contrib0.is_finite());
        assert!(contrib63.is_finite());
        assert!(contrib64.is_finite());
        assert!(contrib1000.is_finite());
    }

    #[test]
    fn test_hierarchical_parameters() {
        let hcope = HierarchicalCoPE::new(64, 16, 32);
        let params = hcope.parameters();

        // local_cope: 64 * 32 = 2048
        // global_cope: 16 * 32 = 512
        // chunk_predictor_w: 32 * 2 = 64
        // chunk_predictor_b: 1 * 2 = 2
        // Total: 2048 + 512 + 64 + 2 = 2626
        assert_eq!(params, 2626);
    }
}
