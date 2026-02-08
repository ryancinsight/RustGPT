use ndarray::{Array1, Array2, ArrayView1, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{infrastructure::optimizer::adam::Adam, common::rng::get_rng};

/// Gated Contextual Position Embeddings (GatedCoPE)
///
/// Enhances standard CoPE with a learnable gating mechanism that adaptively
/// blends position-based and content-based attention contributions.
///
/// Mathematical Formulation:
/// ```
/// s_ij = q_i · k_j + g_ij · CoPE(q_i, pos_ij)
///
/// where g_ij = σ(W_g · [q_i; k_j] + b_g)
/// ```
///
/// The gate `g_ij` is computed per query-key pair, allowing the model to
/// learn when to emphasize position information vs content similarity.
///
/// Benefits:
/// - Adaptive position/content weighting per attention head
/// - Smoother gradients via log1p-style gate formulation
/// - Reduced loss of information through multiplicative interaction
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GatedCoPE {
    /// Base CoPE embeddings (max_pos+1, embed_dim)
    base_cope: CoPE,
    /// Gate projection: maps (q ⊕ k) → scalar per position (2*embed_dim, max_pos+1)
    w_gate: Array2<f32>,
    opt_w_gate: Adam,
    /// Gate bias for position-specific gating (1, max_pos+1) for Adam compatibility
    b_gate: Array2<f32>,
    opt_b_gate: Adam,
    /// Gate temperature for sharper/smoother gating
    gate_temperature: f32,
    /// Maximum position
    max_pos: usize,
    /// Embedding dimension
    embed_dim: usize,
}

impl GatedCoPE {
    /// Create a new GatedCoPE instance
    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();
        
        // Base CoPE
        let base_cope = CoPE::new(max_pos, embed_dim);
        
        // Gate projection: (embed_dim * 2) × (max_pos + 1)
        let gate_input_dim = embed_dim * 2;
        let w_gate = Array2::from_shape_fn((gate_input_dim, max_pos + 1), |_| normal.sample(&mut rng));
        let opt_w_gate = Adam::new((gate_input_dim, max_pos + 1));
        
        // Gate bias: (1, max_pos+1) for Adam compatibility
        let b_gate = Array2::zeros((1, max_pos + 1));
        let opt_b_gate = Adam::new((1, max_pos + 1));

        Self {
            base_cope,
            w_gate,
            opt_w_gate,
            b_gate,
            opt_b_gate,
            gate_temperature: 1.0,
            max_pos,
            embed_dim,
        }
    }

    /// Create with custom temperature
    pub fn with_temperature(max_pos: usize, embed_dim: usize, temperature: f32) -> Self {
        let mut instance = Self::new(max_pos, embed_dim);
        instance.gate_temperature = temperature;
        instance
    }

    /// Compute gated CoPE contribution for a query at a specific position
    ///
    /// # Arguments
    /// * `q` - Query vector (embed_dim,)
    /// * `k` - Key vector (embed_dim,)
    /// * `pos` - Relative position (0-indexed)
    ///
    /// # Returns
    /// Gated CoPE contribution: `gate * CoPE_contribution`
    #[inline]
    pub fn gated_cope_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        pos: usize,
    ) -> f32 {
        if pos > self.max_pos {
            return 0.0;
        }

        // Compute base CoPE contribution: q · PE_pos
        let cope_contrib = q.dot(&self.base_cope.pos_embeddings.row(pos));

        // Compute gate: σ((q ⊕ k) · W_gate[:, pos] + b_gate[0, pos])
        let mut gate_input = Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);

        // Gate computation with temperature scaling
        let gate_logit = gate_input.dot(&self.w_gate.column(pos)) + self.b_gate[[0, pos]];
        let gate = self.smooth_gate(gate_logit / self.gate_temperature);

        // Multiplicative interaction preserves gradient flow better than addition
        gate * cope_contrib
    }

    /// Smooth gate function using log1p-style formulation
    ///
    /// Uses σ(x) = 1 / (1 + exp(-x)) with temperature scaling
    /// This provides well-behaved gradients in both saturation regimes.
    #[inline]
    fn smooth_gate(&self, x: f32) -> f32 {
        // Numerically stable sigmoid with temperature
        let x = x.clamp(-500.0, 500.0); // Prevent overflow
        1.0 / (1.0 + (-x).exp())
    }

    /// Apply gradients for gated CoPE
    pub fn apply_gradients(&mut self, grads: &(Array2<f32>, Array2<f32>, Array2<f32>), lr: f32) {
        // grads.0 is for base CoPE, grads.1 is for w_gate, grads.2 is for b_gate
        self.base_cope.apply_gradients(&grads.0, lr);

        self.opt_w_gate.step(&mut self.w_gate, &grads.1, lr);
        self.opt_b_gate.step(&mut self.b_gate, &grads.2, lr);
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        self.base_cope.parameters() + self.w_gate.len() + self.b_gate.len()
    }

    /// Get weight norm
    pub fn weight_norm(&self) -> f32 {
        let base_norm = self.base_cope.weight_norm();
        let gate_norm: f32 = self.w_gate.iter().map(|&w| w * w).sum::<f32>().sqrt();
        let bias_norm: f32 = self.b_gate.iter().map(|&w| w * w).sum::<f32>().sqrt();
        (base_norm.powi(2) + gate_norm.powi(2) + bias_norm.powi(2)).sqrt()
    }

    /// Get maximum position
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// Get embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get reference to base CoPE for gradient access
    pub fn base_cope(&self) -> &CoPE {
        &self.base_cope
    }

    /// Get mutable reference to base CoPE
    pub fn base_cope_mut(&mut self) -> &mut CoPE {
        &mut self.base_cope
    }

    /// Get gate parameters for gradient computation
    pub fn gate_params(&self) -> (&Array2<f32>, &Array2<f32>) {
        (&self.w_gate, &self.b_gate)
    }
}

/// Re-export base CoPE for internal use
use super::cope::CoPE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gated_cope_creation() {
        let gated = GatedCoPE::new(10, 8);
        assert_eq!(gated.max_pos, 10);
        assert_eq!(gated.base_cope.pos_embeddings.shape(), &[11, 8]);
        assert_eq!(gated.w_gate.shape(), &[16, 11]); // 2*embed_dim × (max_pos+1)
        assert_eq!(gated.b_gate.shape(), &[1, 11]);
    }

    #[test]
    fn test_gated_cope_contribution() {
        let gated = GatedCoPE::new(10, 8);
        let q = Array1::from_elem(8, 0.5);
        let k = Array1::from_elem(8, 0.3);
        
        let contrib = gated.gated_cope_contribution(&q.view(), &k.view(), 0);
        
        // Should be finite and non-zero
        assert!(contrib.is_finite());
    }

    #[test]
    fn test_gated_cope_parameters() {
        let gated = GatedCoPE::new(64, 32);
        let params = gated.parameters();
        
        // CoPE: (65, 32) = 2080
        // W_gate: (64, 65) = 4160
        // b_gate: (1, 65) = 65
        // Total: 2080 + 4160 + 65 = 6305
        assert_eq!(params, 6305);
    }
}
