use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};
use super::traits::PositionEmbedding;
use super::cope::{CoPE, CoPEGradients};

/// Gradients container for GatedCoPE
#[derive(Clone, Debug)]
pub struct GatedCoPEGradients {
    pub base_grads: CoPEGradients,
    pub w_gate_grads: Option<Array2<f32>>,
    pub b_gate_grads: Option<Array2<f32>>,
}

impl GatedCoPEGradients {
    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        Self {
            base_grads: CoPEGradients::new(max_pos, embed_dim),
            w_gate_grads: Some(Array2::zeros((embed_dim * 2, max_pos + 1))),
            b_gate_grads: Some(Array2::zeros((1, max_pos + 1))),
        }
    }
}

/// Gated Contextual Position Embeddings (GatedCoPE)
///
/// Enhances standard CoPE with a learnable gating mechanism that adaptively
/// blends position-based and content-based attention contributions.
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

impl PositionEmbedding for GatedCoPE {
    type Gradients = GatedCoPEGradients;

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        _inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        let pos = query_pos.saturating_sub(key_pos);
        if pos > self.max_pos {
            return 0.0;
        }

        // Compute base CoPE contribution: q · PE_pos
        // We can access base_cope directly since we are inside the module or it has public fields.
        // The PositionEmbedding trait on CoPE uses (q, k, q_pos, k_pos, inputs)
        // but here we just need the internal dot product which contribution() provides.
        let cope_contrib = self.base_cope.contribution(q, k, query_pos, key_pos, None);

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

    fn backward(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
        d_s_ij: f32,
        grads: &mut Self::Gradients,
    ) -> (Array1<f32>, Array1<f32>) {
        let pos = query_pos.saturating_sub(key_pos);
        if pos > self.max_pos {
            return (Array1::zeros(q.dim()), Array1::zeros(k.dim()));
        }

        // Recompute forward pass values for gradients
        let cope_contrib = self.base_cope.contribution(q, k, query_pos, key_pos, None);
        
        let mut gate_input = Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);

        let gate_logit = gate_input.dot(&self.w_gate.column(pos)) + self.b_gate[[0, pos]];
        let scaled_logit = gate_logit / self.gate_temperature;
        let gate = self.smooth_gate(scaled_logit);

        // dL/d(gate) = dL/ds * cope_contrib
        let d_gate = d_s_ij * cope_contrib;
        
        // dL/d(cope_contrib) = dL/ds * gate
        let d_cope = d_s_ij * gate;

        // 1. Backprop through base CoPE
        // We use d_cope as the gradient for the base contribution
        let (dq_base, dk_base) = self.base_cope.backward(q, k, query_pos, key_pos, inputs, d_cope, &mut grads.base_grads);

        // 2. Backprop through gate
        // gate = sigmoid(scaled_logit)
        // d(gate)/d(scaled_logit) = gate * (1 - gate)
        let d_scaled_logit = d_gate * gate * (1.0 - gate);
        
        // scaled_logit = logit / temp
        // d(logit) = d(scaled_logit) / temp
        let d_logit = d_scaled_logit / self.gate_temperature;

        // logit = gate_input . W[:, pos] + b[pos]
        // d(b[pos]) = d_logit
        if let Some(bg) = &mut grads.b_gate_grads {
            bg[[0, pos]] += d_logit;
        }

        // d(W[:, pos]) = gate_input * d_logit
        if let Some(wg) = &mut grads.w_gate_grads {
            let mut col = wg.column_mut(pos);
            // col += gate_input * d_logit
            for (w, &g) in col.iter_mut().zip(gate_input.iter()) {
                *w += g * d_logit;
            }
        }

        // d(gate_input) = W[:, pos] * d_logit
        let w_col = self.w_gate.column(pos);
        let d_gate_input = &w_col * d_logit; // Array1

        // Split d_gate_input into dq_gate and dk_gate
        let dq_gate = d_gate_input.slice(s![0..self.embed_dim]);
        let dk_gate = d_gate_input.slice(s![self.embed_dim..]);

        // Combine gradients
        let dq = dq_base + dq_gate;
        let dk = dk_base + dk_gate;

        (dq, dk)
    }

    fn init_gradients(&self) -> Self::Gradients {
        GatedCoPEGradients::new(self.max_pos, self.embed_dim)
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        self.base_cope.apply_gradients(&grads.base_grads, lr);
        
        if let Some(wg) = &grads.w_gate_grads {
            self.opt_w_gate.step(&mut self.w_gate, wg, lr);
        }
        
        if let Some(bg) = &grads.b_gate_grads {
            self.opt_b_gate.step(&mut self.b_gate, bg, lr);
        }
    }

    fn max_pos(&self) -> usize {
        self.max_pos
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn parameters(&self) -> usize {
        self.base_cope.parameters() + self.w_gate.len() + self.b_gate.len()
    }

    fn weight_norm(&self) -> f32 {
        let base_norm = self.base_cope.weight_norm();
        let gate_w_norm: f32 = self.w_gate.iter().map(|x| x * x).sum();
        let gate_b_norm: f32 = self.b_gate.iter().map(|x| x * x).sum();
        (base_norm.powi(2) + gate_w_norm + gate_b_norm).sqrt()
    }
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
        let w_gate =
            Array2::from_shape_fn((gate_input_dim, max_pos + 1), |_| normal.sample(&mut rng));
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

        let contrib = gated.contribution(&q.view(), &k.view(), 0, 0, None);

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
