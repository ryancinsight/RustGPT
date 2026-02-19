use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use super::gradient_ops::{accumulate_optional_arrays, append_optional_array_to_vec};
use super::traits::PositionEmbedding;
use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};

/// Gradients container for HierarchicalCoPE
#[derive(Clone, Debug)]
pub struct HierarchicalCoPEGradients {
    pub local_cope_grads: Option<Array2<f32>>,
    pub global_cope_grads: Option<Array2<f32>>,
    pub chunk_predictor_w_grads: Option<Array2<f32>>,
    pub chunk_predictor_b_grads: Option<Array2<f32>>,
}

impl HierarchicalCoPEGradients {
    pub fn new(chunk_size: usize, max_chunks: usize, embed_dim: usize) -> Self {
        Self {
            local_cope_grads: Some(Array2::zeros((chunk_size, embed_dim))),
            global_cope_grads: Some(Array2::zeros((max_chunks, embed_dim))),
            chunk_predictor_w_grads: Some(Array2::zeros((embed_dim, 2))),
            chunk_predictor_b_grads: Some(Array2::zeros((1, 2))),
        }
    }

    /// Accumulate gradients from another HierarchicalCoPEGradients instance.
    /// Uses zero-cost generic abstractions for optional array handling.
    pub fn accumulate(&mut self, other: &Self) {
        accumulate_optional_arrays(&mut self.local_cope_grads, &other.local_cope_grads);
        accumulate_optional_arrays(&mut self.global_cope_grads, &other.global_cope_grads);
        accumulate_optional_arrays(
            &mut self.chunk_predictor_w_grads,
            &other.chunk_predictor_w_grads,
        );
        accumulate_optional_arrays(
            &mut self.chunk_predictor_b_grads,
            &other.chunk_predictor_b_grads,
        );
    }

    /// Serialize gradients to a flat vector.
    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = Vec::new();
        append_optional_array_to_vec(&mut v, &self.local_cope_grads);
        append_optional_array_to_vec(&mut v, &self.global_cope_grads);
        append_optional_array_to_vec(&mut v, &self.chunk_predictor_w_grads);
        append_optional_array_to_vec(&mut v, &self.chunk_predictor_b_grads);
        v
    }
}

/// Hierarchical Contextual Position Embeddings (HierarchicalCoPE)
///
/// Extends CoPE with multiple levels of positional granularity for better
/// generalization to longer sequences.
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

impl PositionEmbedding for HierarchicalCoPE {
    type Gradients = HierarchicalCoPEGradients;

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        _inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        let pos = query_pos.saturating_sub(key_pos);
        if pos >= self.effective_max_pos() {
            return 0.0;
        }

        // Determine chunk index and local position
        let chunk_idx = (pos / self.chunk_size).min(self.max_chunks - 1);
        let local_pos = pos % self.chunk_size;

        // Content similarity for boundary detection
        let content_sim = q.dot(k);

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

    fn backward(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        _inputs: Option<&ArrayView2<f32>>,
        d_s_ij: f32,
        grads: &mut Self::Gradients,
    ) -> (Array1<f32>, Array1<f32>) {
        let pos = query_pos.saturating_sub(key_pos);
        if pos >= self.effective_max_pos() {
            return (Array1::zeros(q.dim()), Array1::zeros(k.dim()));
        }

        let chunk_idx = (pos / self.chunk_size).min(self.max_chunks - 1);
        let local_pos = pos % self.chunk_size;
        let content_sim = q.dot(k);

        // Recompute forward values
        let boundary_logit = q.dot(&self.chunk_predictor_w.column(0))
            + content_sim * self.chunk_predictor_w[[0, 1]]
            + self.chunk_predictor_b[[0, 0]];
        let boundary_gate = self.sigmoid(boundary_logit);

        let local_row = self.local_cope.row(local_pos);
        let global_row = self.global_cope.row(chunk_idx);

        let local_contrib = q.dot(&local_row);
        let global_contrib = q.dot(&global_row);
        let mixed = self.alpha_local * local_contrib + self.alpha_global * global_contrib;

        // Gradients
        // Output = mixed * (1 + 0.1 * gate)
        // dL/dOutput = d_s_ij

        let d_output = d_s_ij;

        // dOutput/dMixed = 1 + 0.1 * gate
        let d_mixed = d_output * (1.0 + 0.1 * boundary_gate);

        // dOutput/dGate = mixed * 0.1
        let d_gate = d_output * mixed * 0.1;

        // dGate/dLogit = gate * (1 - gate)
        let d_logit = d_gate * boundary_gate * (1.0 - boundary_gate);

        // dMixed/dLocalContrib = alpha_local
        let d_local_contrib = d_mixed * self.alpha_local;

        // dMixed/dGlobalContrib = alpha_global
        let d_global_contrib = d_mixed * self.alpha_global;

        // Accumulate gradients for embeddings
        // dLocalContrib/dLocalRow = q
        if let Some(lg) = &mut grads.local_cope_grads {
            let mut row = lg.row_mut(local_pos);
            // row += q * d_local_contrib
            for (r, &q_val) in row.iter_mut().zip(q.iter()) {
                *r += q_val * d_local_contrib;
            }
        }

        if let Some(gg) = &mut grads.global_cope_grads {
            let mut row = gg.row_mut(chunk_idx);
            for (r, &q_val) in row.iter_mut().zip(q.iter()) {
                *r += q_val * d_global_contrib;
            }
        }

        // Gradients for predictor
        // Logit = q . w_col0 + sim * w_col1 + b

        // dLogit/dw_col0 = q
        if let Some(wg) = &mut grads.chunk_predictor_w_grads {
            let mut col0 = wg.column_mut(0);
            for (w, &q_val) in col0.iter_mut().zip(q.iter()) {
                *w += q_val * d_logit;
            }

            // dLogit/dw_col1 = sim
            wg[[0, 1]] += content_sim * d_logit;
        }

        // dLogit/db = 1
        if let Some(bg) = &mut grads.chunk_predictor_b_grads {
            bg[[0, 0]] += d_logit;
        }

        // Gradients w.r.t q and k
        // dMixed/dq = alpha_local * local_row + alpha_global * global_row
        let d_mixed_dq = &local_row * self.alpha_local + &global_row * self.alpha_global;

        // dLogit/dq = w_col0 + w_col1 * dSim/dq
        // dSim/dq = k
        // dLogit/dq = w_col0 + w_col1 * k
        let w_col0 = self.chunk_predictor_w.column(0);
        let w_col1 = self.chunk_predictor_w[[0, 1]];
        let d_logit_dq = &w_col0 + &(&k.to_owned() * w_col1);

        let dq = &d_mixed_dq * d_mixed + &d_logit_dq * d_logit;

        // dSim/dk = q
        // dLogit/dk = w_col1 * dSim/dk = w_col1 * q
        let dk = &q.to_owned() * (w_col1 * d_logit);

        (dq, dk)
    }

    fn init_gradients(&self) -> Self::Gradients {
        HierarchicalCoPEGradients::new(self.chunk_size, self.max_chunks, self.embed_dim)
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        if let Some(lg) = &grads.local_cope_grads {
            self.opt_local_cope.step(&mut self.local_cope, lg, lr);
        }
        if let Some(gg) = &grads.global_cope_grads {
            self.opt_global_cope.step(&mut self.global_cope, gg, lr);
        }
        if let Some(wg) = &grads.chunk_predictor_w_grads {
            self.opt_chunk_predictor_w
                .step(&mut self.chunk_predictor_w, wg, lr);
        }
        if let Some(bg) = &grads.chunk_predictor_b_grads {
            self.opt_chunk_predictor_b
                .step(&mut self.chunk_predictor_b, bg, lr);
        }
    }

    fn max_pos(&self) -> usize {
        self.effective_max_pos()
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn parameters(&self) -> usize {
        self.local_cope.len()
            + self.global_cope.len()
            + self.chunk_predictor_w.len()
            + self.chunk_predictor_b.len()
    }

    fn weight_norm(&self) -> f32 {
        let local_norm: f32 = self.local_cope.iter().map(|x| x * x).sum();
        let global_norm: f32 = self.global_cope.iter().map(|x| x * x).sum();
        let w_norm: f32 = self.chunk_predictor_w.iter().map(|x| x * x).sum();
        let b_norm: f32 = self.chunk_predictor_b.iter().map(|x| x * x).sum();
        (local_norm + global_norm + w_norm + b_norm).sqrt()
    }
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

    /// Sigmoid with numerical stability
    #[inline]
    fn sigmoid(&self, x: f32) -> f32 {
        let x = x.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-x).exp())
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
        let k = Array1::from_elem(32, 0.3); // Need k for content_sim

        // Test various positions
        // contribution(q, k, query_pos, key_pos, inputs)
        // pos = query_pos - key_pos

        // pos = 0
        let contrib0 = hcope.contribution(&q.view(), &k.view(), 0, 0, None);

        // pos = 63 (local boundary)
        let contrib63 = hcope.contribution(&q.view(), &k.view(), 63, 0, None);

        // pos = 64 (new chunk)
        let contrib64 = hcope.contribution(&q.view(), &k.view(), 64, 0, None);

        // pos = 1000
        let contrib1000 = hcope.contribution(&q.view(), &k.view(), 1000, 0, None);

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
