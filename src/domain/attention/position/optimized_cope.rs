use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{common::rng::get_rng, infrastructure::optimizer::adam::Adam};
use super::traits::PositionEmbedding;

/// Gradients for OptimizedCoPE
#[derive(Clone, Debug)]
pub struct OptimizedCoPEGradients {
    pub up_proj_grads: Option<Array2<f32>>,
    pub down_proj_grads: Option<Array2<f32>>,
    pub w_gate_grads: Option<Array2<f32>>,
    pub b_gate_grads: Option<Array2<f32>>,
}

impl OptimizedCoPEGradients {
    pub fn new(max_pos: usize, embed_dim: usize, rank: usize) -> Self {
        Self {
            up_proj_grads: Some(Array2::zeros((max_pos + 1, rank))),
            down_proj_grads: Some(Array2::zeros((rank, embed_dim))),
            w_gate_grads: Some(Array2::zeros((2 * embed_dim, 1))),
            b_gate_grads: Some(Array2::zeros((1, 1))),
        }
    }
}

/// Optimized Contextual Position Embeddings (OptimizedCoPE)
///
/// A unified, production-ready CoPE that combines:
/// - **Gated CoPE**: Adaptive position/content weighting per attention head
/// - **Factorized CoPE**: Memory-efficient low-rank position embeddings
/// - **Log1p stabilization**: Smooth gradients, no vanishing/exploding
/// - **Temperature scaling**: Controllable gradient magnitude
///
/// Mathematical Formulation:
/// ```
/// CoPE(q, pos) = log(1 + exp(gate · (U[pos] @ V @ q)))
/// gate = σ(W_gate · [q; k] + b_gate) * temperature
/// ```
///
/// Benefits:
/// - Adaptive: Learns when to emphasize position vs content
/// - Efficient: O(max_pos × r) parameters instead of O(max_pos × embed_dim)
/// - Stable: Log1p formulation prevents gradient issues
/// - Drop-in replacement for existing CoPE
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OptimizedCoPE {
    /// Factorized position embeddings: PE = U @ V
    /// Up projection: (max_pos + 1, rank)
    up_proj: Array2<f32>,
    opt_up_proj: Adam,
    /// Down projection: (rank, embed_dim)
    down_proj: Array2<f32>,
    opt_down_proj: Adam,

    /// Gate projection for adaptive position weighting: (2*embed_dim, 1)
    w_gate: Array2<f32>,
    opt_w_gate: Adam,
    /// Gate bias: (1, 1)
    b_gate: Array2<f32>,
    opt_b_gate: Adam,

    /// Factorization rank
    rank: usize,
    /// Maximum position
    max_pos: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Temperature for gate scaling
    temperature: f32,
}

impl PositionEmbedding for OptimizedCoPE {
    type Gradients = OptimizedCoPEGradients;

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        _inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        let pos = query_pos.saturating_sub(key_pos);
        self.optimized_cope_contribution(q, k, pos)
    }

    fn init_gradients(&self) -> Self::Gradients {
        OptimizedCoPEGradients::new(self.max_pos, self.embed_dim, self.rank)
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        if let Some(grad) = &grads.up_proj_grads {
            self.opt_up_proj.step(&mut self.up_proj, grad, lr);
        }
        if let Some(grad) = &grads.down_proj_grads {
            self.opt_down_proj.step(&mut self.down_proj, grad, lr);
        }
        if let Some(grad) = &grads.w_gate_grads {
            self.opt_w_gate.step(&mut self.w_gate, grad, lr);
        }
        if let Some(grad) = &grads.b_gate_grads {
            self.opt_b_gate.step(&mut self.b_gate, grad, lr);
        }
    }

    fn max_pos(&self) -> usize {
        self.max_pos
    }

    fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    fn parameters(&self) -> usize {
        self.up_proj.len() + self.down_proj.len() + self.w_gate.len() + self.b_gate.len()
    }

    fn weight_norm(&self) -> f32 {
        let up_norm: f32 = self.up_proj.iter().map(|x| x * x).sum();
        let down_norm: f32 = self.down_proj.iter().map(|x| x * x).sum();
        let w_gate_norm: f32 = self.w_gate.iter().map(|x| x * x).sum();
        let b_gate_norm: f32 = self.b_gate.iter().map(|x| x * x).sum();
        (up_norm + down_norm + w_gate_norm + b_gate_norm).sqrt()
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
        if pos > self.max_pos {
            return (Array1::zeros(q.dim()), Array1::zeros(k.dim()));
        }

        // --- Recompute Forward Pass ---
        
        // 1. vq = Down @ q
        let mut vq = Array1::zeros(self.rank);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj, q, 0.0, &mut vq);

        // 2. cope_raw = Up[pos] . vq
        let up_row = self.up_proj.row(pos);
        let cope_raw = up_row.dot(&vq);

        // 3. Gate
        let mut gate_input = Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);
        
        let gate_logit = gate_input.dot(&self.w_gate.column(0)) + self.b_gate[[0, 0]];
        let gate_scaled = gate_logit / self.temperature;
        let gate = self.smooth_gate(gate_scaled);

        // 4. Interaction
        let interaction = gate * cope_raw;
        let interaction_scaled = interaction / self.temperature;

        // --- Backward Pass ---

        // dOutput/dInteraction
        // y = T * log(1 + exp(x/T))
        // dy/dx = sigmoid(x/T)
        // d_interaction = d_s_ij * sigmoid(interaction/T)
        let sigmoid_int = self.smooth_gate(interaction_scaled);
        let d_interaction = d_s_ij * sigmoid_int;

        // dInteraction/dGate = cope_raw
        let d_gate = d_interaction * cope_raw;

        // dInteraction/dCopeRaw = gate
        let d_cope_raw = d_interaction * gate;

        // dGate/dLogit
        // gate = sigmoid(logit/T)
        // dGate/dLogit = gate * (1-gate) / T
        let d_logit = d_gate * gate * (1.0 - gate) / self.temperature;

        // Gradients for Gate Parameters
        if let Some(wg) = &mut grads.w_gate_grads {
            let mut col = wg.column_mut(0);
            // dLogit/dW = gate_input
            for (w, &val) in col.iter_mut().zip(gate_input.iter()) {
                *w += val * d_logit;
            }
        }
        if let Some(bg) = &mut grads.b_gate_grads {
            bg[[0, 0]] += d_logit;
        }

        // dLogit/dGateInput = W_gate
        let w_gate_col = self.w_gate.column(0);
        let d_gate_input = &w_gate_col * d_logit;
        
        let d_q_gate = d_gate_input.slice(s![0..self.embed_dim]);
        let d_k_gate = d_gate_input.slice(s![self.embed_dim..]);

        // Gradients for Factorized Embeddings
        
        // dCopeRaw/dUp[pos] = vq
        if let Some(ug) = &mut grads.up_proj_grads {
            let mut row = ug.row_mut(pos);
            for (u, &v) in row.iter_mut().zip(vq.iter()) {
                *u += v * d_cope_raw;
            }
        }

        // dCopeRaw/dVq = Up[pos]
        let d_vq = &up_row * d_cope_raw;

        // dVq/dDown = q
        // dVq_i = sum_j Down_ij * q_j
        // dVq_i / dDown_ij = q_j
        if let Some(dg) = &mut grads.down_proj_grads {
            // outer product d_vq * q
            for (i, &dv) in d_vq.iter().enumerate() {
                let mut row = dg.row_mut(i);
                for (d, &q_val) in row.iter_mut().zip(q.iter()) {
                    *d += dv * q_val;
                }
            }
        }

        // dVq/dq = Down^T * d_vq
        let mut d_q_content = Array1::zeros(self.embed_dim);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj.t(), &d_vq, 0.0, &mut d_q_content);

        // Total Gradients
        let d_q = &d_q_content + &d_q_gate;
        let d_k = d_k_gate.to_owned();

        (d_q, d_k)
    }
}

impl OptimizedCoPE {
    /// Create a new OptimizedCoPE instance
    ///
    /// # Arguments
    /// * `max_pos` - Maximum position to handle
    /// * `embed_dim` - Embedding dimension
    /// * `rank` - Factorization rank (recommended: embed_dim / 4 for 4× compression)
    pub fn new(max_pos: usize, embed_dim: usize, rank: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02 / (rank as f32).sqrt()).unwrap();

        // Factorized embeddings
        let up_proj = Array2::from_shape_fn((max_pos + 1, rank), |_| normal.sample(&mut rng));
        let opt_up_proj = Adam::new((max_pos + 1, rank));

        let down_proj = Array2::from_shape_fn((rank, embed_dim), |_| normal.sample(&mut rng));
        let opt_down_proj = Adam::new((rank, embed_dim));

        // Gate parameters
        let w_gate = Array2::from_shape_fn((2 * embed_dim, 1), |_| normal.sample(&mut rng) * 0.01);
        let opt_w_gate = Adam::new((2 * embed_dim, 1));

        let b_gate = Array2::zeros((1, 1));
        let opt_b_gate = Adam::new((1, 1));

        Self {
            up_proj,
            opt_up_proj,
            down_proj,
            opt_down_proj,
            w_gate,
            opt_w_gate,
            b_gate,
            opt_b_gate,
            rank,
            max_pos,
            embed_dim,
            temperature: 1.0,
        }
    }

    /// Create with custom temperature
    pub fn with_temperature(
        max_pos: usize,
        embed_dim: usize,
        rank: usize,
        temperature: f32,
    ) -> Self {
        let mut instance = Self::new(max_pos, embed_dim, rank);
        instance.temperature = temperature;
        instance
    }

    /// Compute optimized CoPE contribution
    ///
    /// Combines content-based gating with factorized position embeddings
    ///
    /// # Arguments
    /// * `q` - Query vector (embed_dim,)
    /// * `k` - Key vector (embed_dim,) for gate computation
    /// * `pos` - Relative position
    ///
    /// # Returns
    /// Optimized CoPE contribution with adaptive gating
    #[inline]
    pub fn optimized_cope_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        pos: usize,
    ) -> f32 {
        if pos > self.max_pos {
            return 0.0;
        }

        // Compute factorized embedding: V @ q^T (rank,)
        let mut vq = Array1::zeros(self.rank);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj, q, 0.0, &mut vq);

        // Compute: U[pos] @ (V @ q) = (1, rank) @ (rank,) = scalar
        let up_row = self.up_proj.row(pos);
        let cope_raw = up_row.dot(&vq);

        // Compute gate: σ((q ⊕ k) · W_gate + b)
        let mut gate_input = Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);

        let gate_logit = gate_input.dot(&self.w_gate.column(0)) + self.b_gate[[0, 0]];
        let gate = self.smooth_gate(gate_logit / self.temperature);

        // Log1p formulation for stability: log(1 + exp(gate * cope))
        let interaction = gate * cope_raw;
        self.log1p_exp(interaction / self.temperature) * self.temperature
    }

    /// Compute CoPE without gate (for analysis/comparison)
    #[inline]
    pub fn factorized_cope_contribution(&self, q: &ArrayView1<'_, f32>, pos: usize) -> f32 {
        if pos > self.max_pos {
            return 0.0;
        }

        let mut vq = Array1::zeros(self.rank);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.down_proj, q, 0.0, &mut vq);

        let up_row = self.up_proj.row(pos);
        let raw = up_row.dot(&vq);

        self.log1p_exp(raw / self.temperature) * self.temperature
    }

    /// Numerically stable log(1 + exp(x)) with gradient preservation
    #[inline]
    fn log1p_exp(&self, x: f32) -> f32 {
        if x > 20.0 {
            x // Approximation for large positive
        } else if x < -20.0 {
            x.exp() // Approximation for large negative (very small)
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// Smooth sigmoid with temperature
    #[inline]
    fn smooth_gate(&self, x: f32) -> f32 {
        let x = x.clamp(-500.0, 500.0);
        1.0 / (1.0 + (-x).exp())
    }

    /// Get the gate value for a query-key pair (for analysis)
    #[inline]
    pub fn gate_value(&self, q: &ArrayView1<'_, f32>, k: &ArrayView1<'_, f32>) -> f32 {
        let mut gate_input = Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);

        let gate_logit = gate_input.dot(&self.w_gate.column(0)) + self.b_gate[[0, 0]];
        self.smooth_gate(gate_logit / self.temperature)
    }

    /// Get reconstructed position embedding for a specific position
    #[inline]
    pub fn get_embedding(&self, pos: usize) -> Option<Array1<f32>> {
        if pos > self.max_pos {
            return None;
        }
        let up_row = self.up_proj.row(pos);
        let mut embedding = Array1::zeros(self.embed_dim);
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            &self.down_proj.t(),
            &up_row.to_owned(),
            0.0,
            &mut embedding,
        );
        Some(embedding)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let full_params = (self.max_pos + 1) * self.embed_dim;
        let factored_params = self.up_proj.len() + self.down_proj.len();
        full_params as f32 / factored_params as f32
    }

    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        self.up_proj.len() + self.down_proj.len() + self.w_gate.len() + self.b_gate.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_cope_creation() {
        let cope = OptimizedCoPE::new(128, 64, 16);
        assert_eq!(cope.max_pos, 128);
        assert_eq!(cope.embed_dim, 64);
        assert_eq!(cope.rank, 16);
        assert_eq!(cope.up_proj.shape(), &[129, 16]);
        assert_eq!(cope.down_proj.shape(), &[16, 64]);
        assert_eq!(cope.w_gate.shape(), &[128, 1]);
    }

    #[test]
    fn test_optimized_cope_contribution() {
        let cope = OptimizedCoPE::new(128, 64, 16);
        let q = Array1::from_elem(64, 0.5);
        let k = Array1::from_elem(64, 0.3);

        let contrib = cope.optimized_cope_contribution(&q.view(), &k.view(), 0);
        assert!(contrib.is_finite());

        let contrib_100 = cope.optimized_cope_contribution(&q.view(), &k.view(), 100);
        assert!(contrib_100.is_finite());
    }

    #[test]
    fn test_factorized_only() {
        let cope = OptimizedCoPE::new(128, 64, 16);
        let q = Array1::from_elem(64, 0.5);

        let contrib = cope.factorized_cope_contribution(&q.view(), 0);
        assert!(contrib.is_finite());
    }

    #[test]
    fn test_gate_value() {
        let cope = OptimizedCoPE::new(128, 64, 16);
        let q = Array1::from_elem(64, 0.5);
        let k = Array1::from_elem(64, 0.3);

        let gate = cope.gate_value(&q.view(), &k.view());
        assert!(gate > 0.0);
        assert!(gate <= 1.0);
    }

    #[test]
    fn test_compression_ratio() {
        let cope = OptimizedCoPE::new(512, 128, 32);
        let ratio = cope.compression_ratio();

        // Full: 513 * 128 = 65664
        // Factored: 513*32 + 32*128 = 16416 + 4096 = 20512
        // Plus gates: 128 + 1 = 20541
        // Ratio: 65664 / 20541 ≈ 3.2
        assert!(ratio > 2.5);
    }

    #[test]
    fn test_embedding_reconstruction() {
        let cope = OptimizedCoPE::new(64, 32, 8);
        let emb = cope.get_embedding(0);

        assert!(emb.is_some());
        let emb = emb.unwrap();
        assert_eq!(emb.len(), 32);
        assert!(emb.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_parameters() {
        let cope = OptimizedCoPE::new(512, 128, 32);
        let params = cope.parameters();

        // up_proj: 513 * 32 = 16416
        // down_proj: 32 * 128 = 4096
        // w_gate: 256 * 1 = 256
        // b_gate: 1
        // Total: 16416 + 4096 + 256 + 1 = 20769
        assert_eq!(params, 20769);
    }

    #[test]
    fn test_temperature_scaling() {
        let cope_hot = OptimizedCoPE::with_temperature(64, 32, 8, 0.5);
        let cope_cold = OptimizedCoPE::with_temperature(64, 32, 8, 2.0);

        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.3);

        let contrib_hot = cope_hot.optimized_cope_contribution(&q.view(), &k.view(), 0);
        let contrib_cold = cope_cold.optimized_cope_contribution(&q.view(), &k.view(), 0);

        assert!(contrib_hot.is_finite());
        assert!(contrib_cold.is_finite());
    }
}
