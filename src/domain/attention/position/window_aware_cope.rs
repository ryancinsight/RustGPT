use ndarray::{Array2, ArrayView1, s};
use serde::{Deserialize, Serialize};

/// Unified CoPE variant selection for window-aware positioning
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WindowAwareCoPEType {
    /// Standard CoPE (baseline)
    #[default]
    Standard,
    /// OptimizedCoPE: Unified variant with gating, factorization, and log1p
    Optimized,
    /// GatedCoPE: Adaptive position/content weighting
    Gated,
    /// FactorizedCoPE: Memory-efficient low-rank embeddings
    Factorized,
}

impl WindowAwareCoPEType {
    /// Check if using optimized variant
    pub fn is_optimized(&self) -> bool {
        matches!(self, WindowAwareCoPEType::Optimized)
    }
}

/// Window-aware wrapper for CoPE variants that enforces sliding window boundaries.
///
/// This wrapper ensures that position embeddings respect the sliding window constraint:
/// - Positions within [0, window_size) get their normal CoPE values
/// - Positions >= window_size are clamped to window_size-1 (or return 0)
///
/// Mathematical Invariant:
/// ```
/// PE_windowed(q, pos) = PE_raw(q, min(pos, window_size - 1))
/// ```
///
/// Benefits:
/// - Consistent window handling across all CoPE variants
/// - Zero overhead for non-windowed use cases (window_size = None)
/// - Graceful degradation when sequence exceeds window
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WindowAwareCoPE {
    /// CoPE variant type
    cope_type: WindowAwareCoPEType,
    /// Maximum position from inner CoPE
    max_pos: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Factorization rank (if applicable)
    rank: Option<usize>,
    /// Position embeddings for standard CoPE (max_pos + 1, embed_dim)
    pos_embeddings: Option<Array2<f32>>,
    /// OptimizedCoPE: Up projection (max_pos + 1, rank)
    up_proj: Option<Array2<f32>>,
    /// OptimizedCoPE: Down projection (rank, embed_dim)
    down_proj: Option<Array2<f32>>,
    /// Gate projection weights (2*embed_dim, 1)
    w_gate: Option<Array2<f32>>,
    /// Gate bias (1, 1)
    b_gate: Option<Array2<f32>>,
    /// Maximum window size (None = unlimited, use full CoPE)
    window_size: Option<usize>,
}

impl WindowAwareCoPE {
    /// Create a new window-aware wrapper around standard CoPE
    pub fn new_standard(max_pos: usize, embed_dim: usize, window_size: Option<usize>) -> Self {
        use crate::common::rng::get_rng;
        use rand_distr::Distribution;

        let mut rng = get_rng();
        let normal_pe = rand_distr::Normal::new(0.0, 0.02).unwrap();
        let pos_embeddings =
            Array2::<f32>::from_shape_fn((max_pos + 1, embed_dim), |_| normal_pe.sample(&mut rng));

        Self {
            cope_type: WindowAwareCoPEType::Standard,
            max_pos,
            embed_dim,
            rank: None,
            pos_embeddings: Some(pos_embeddings),
            up_proj: None,
            down_proj: None,
            w_gate: None,
            b_gate: None,
            window_size,
        }
    }

    /// Create a new window-aware wrapper around OptimizedCoPE
    pub fn new_optimized(
        max_pos: usize,
        embed_dim: usize,
        rank: usize,
        window_size: Option<usize>,
    ) -> Self {
        use crate::common::rng::get_rng;
        use rand_distr::Distribution;

        let mut rng = get_rng();
        let normal = rand_distr::Normal::new(0.0, 0.02 / (rank as f32).sqrt()).unwrap();

        let up_proj = Array2::from_shape_fn((max_pos + 1, rank), |_| normal.sample(&mut rng));
        let down_proj = Array2::from_shape_fn((rank, embed_dim), |_| normal.sample(&mut rng));
        let w_gate = Array2::from_shape_fn((2 * embed_dim, 1), |_| normal.sample(&mut rng) * 0.01);
        let b_gate = Array2::zeros((1, 1));

        Self {
            cope_type: WindowAwareCoPEType::Optimized,
            max_pos,
            embed_dim,
            rank: Some(rank),
            pos_embeddings: None,
            up_proj: Some(up_proj),
            down_proj: Some(down_proj),
            w_gate: Some(w_gate),
            b_gate: Some(b_gate),
            window_size,
        }
    }

    /// Create a new window-aware wrapper around GatedCoPE
    pub fn new_gated(max_pos: usize, embed_dim: usize, window_size: Option<usize>) -> Self {
        use crate::common::rng::get_rng;
        use rand_distr::Distribution;

        let mut rng = get_rng();
        let normal = rand_distr::Normal::new(0.0, 0.02).unwrap();

        // Base CoPE
        let pos_embeddings =
            Array2::<f32>::from_shape_fn((max_pos + 1, embed_dim), |_| normal.sample(&mut rng));

        // Gate projection: (2*embed_dim, max_pos + 1)
        let w_gate = Array2::from_shape_fn((2 * embed_dim, max_pos + 1), |_| {
            normal.sample(&mut rng) * 0.01
        });
        let b_gate = Array2::zeros((1, max_pos + 1));

        Self {
            cope_type: WindowAwareCoPEType::Gated,
            max_pos,
            embed_dim,
            rank: None,
            pos_embeddings: Some(pos_embeddings),
            up_proj: None,
            down_proj: None,
            w_gate: Some(w_gate),
            b_gate: Some(b_gate),
            window_size,
        }
    }

    /// Create a new window-aware wrapper around FactorizedCoPE
    pub fn new_factorized(
        max_pos: usize,
        embed_dim: usize,
        rank: usize,
        window_size: Option<usize>,
    ) -> Self {
        use crate::common::rng::get_rng;
        use rand_distr::Distribution;

        let mut rng = get_rng();
        let normal = rand_distr::Normal::new(0.0, 0.02 / (rank as f32).sqrt()).unwrap();

        let up_proj = Array2::from_shape_fn((max_pos + 1, rank), |_| normal.sample(&mut rng));
        let down_proj = Array2::from_shape_fn((rank, embed_dim), |_| normal.sample(&mut rng));

        Self {
            cope_type: WindowAwareCoPEType::Factorized,
            max_pos,
            embed_dim,
            rank: Some(rank),
            pos_embeddings: None,
            up_proj: Some(up_proj),
            down_proj: Some(down_proj),
            w_gate: None,
            b_gate: None,
            window_size,
        }
    }

    /// Get the effective position (clamped to window if needed)
    #[inline]
    fn effective_pos(&self, pos: usize) -> usize {
        match self.window_size {
            Some(ws) => pos.min(ws - 1),
            None => pos,
        }
    }

    /// Check if a position is within the window
    #[inline]
    fn is_in_window(&self, pos: usize) -> bool {
        match self.window_size {
            Some(ws) => pos < ws,
            None => true,
        }
    }

    /// Update window size at runtime
    pub fn set_window_size(&mut self, window_size: Option<usize>) {
        self.window_size = window_size;
    }

    /// Get current window size
    pub fn window_size(&self) -> Option<usize> {
        self.window_size
    }

    /// Get CoPE type
    pub fn cope_type(&self) -> WindowAwareCoPEType {
        self.cope_type
    }
}

/// Helper for sigmoid with clamping
#[inline]
fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x).exp())
}

/// Numerically stable log(1 + exp(x))
#[inline]
fn log1p_exp(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

impl WindowAwareCoPE {
    /// Compute contribution for standard CoPE (window-aware)
    #[inline]
    pub fn cope_contribution(&self, q: &ArrayView1<'_, f32>, pos: usize) -> f32 {
        if !self.is_in_window(pos) {
            return 0.0;
        }
        let eff_pos = self.effective_pos(pos);

        match &self.cope_type {
            WindowAwareCoPEType::Standard => {
                if let Some(pe) = &self.pos_embeddings {
                    if eff_pos <= self.max_pos {
                        q.dot(&pe.row(eff_pos))
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            WindowAwareCoPEType::Optimized => self.optimized_cope_contribution(q, q, pos),
            WindowAwareCoPEType::Gated => {
                if let (Some(pe), Some(w), Some(b)) =
                    (&self.pos_embeddings, &self.w_gate, &self.b_gate)
                {
                    if eff_pos > self.max_pos {
                        return 0.0;
                    }

                    // Compute base CoPE contribution: q · PE_pos
                    let cope_contrib = q.dot(&pe.row(eff_pos));

                    // Compute gate: σ((q ⊕ k) · W_gate[:, pos] + b_gate[:, pos])
                    // For GatedCoPE, gate depends on position
                    let gate_input_len = 2 * self.embed_dim;
                    let mut gate_input = ndarray::Array1::zeros(gate_input_len);
                    gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
                    gate_input.slice_mut(s![self.embed_dim..]).assign(q); // k = q for self-attention

                    let gate_logit = gate_input.dot(&w.column(eff_pos)) + b[[0, eff_pos]];
                    let gate = sigmoid(gate_logit);

                    gate * cope_contrib
                } else {
                    0.0
                }
            }
            WindowAwareCoPEType::Factorized => self.factorized_cope_contribution(q, pos),
        }
    }

    /// Compute contribution for OptimizedCoPE (window-aware)
    #[inline]
    pub fn optimized_cope_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        pos: usize,
    ) -> f32 {
        if !self.is_in_window(pos) {
            return 0.0;
        }
        let eff_pos = self.effective_pos(pos);

        let (up, down, w_gate, b_gate) =
            match (&self.up_proj, &self.down_proj, &self.w_gate, &self.b_gate) {
                (Some(u), Some(d), Some(w), Some(b)) => (u, d, w, b),
                _ => return 0.0,
            };

        let rank = self.rank.unwrap_or(16);

        // Compute factorized embedding: V @ q^T (rank,)
        let mut vq = ndarray::Array1::zeros(rank);
        ndarray::linalg::general_mat_vec_mul(1.0, down, q, 0.0, &mut vq);

        // Compute: U[pos] @ (V @ q) = (1, rank) @ (rank,) = scalar
        let up_row = up.row(eff_pos);
        let cope_raw = up_row.dot(&vq);

        // Compute gate: σ((q ⊕ k) · W_gate + b)
        let mut gate_input = ndarray::Array1::zeros(self.embed_dim * 2);
        gate_input.slice_mut(s![0..self.embed_dim]).assign(q);
        gate_input.slice_mut(s![self.embed_dim..]).assign(k);

        let gate_logit = gate_input.dot(&w_gate.column(0)) + b_gate[[0, 0]];
        let gate = sigmoid(gate_logit);

        // Log1p formulation for stability
        let interaction = gate * cope_raw;
        log1p_exp(interaction)
    }

    /// Compute factorized contribution (window-aware)
    #[inline]
    pub fn factorized_cope_contribution(&self, q: &ArrayView1<'_, f32>, pos: usize) -> f32 {
        if !self.is_in_window(pos) {
            return 0.0;
        }
        let eff_pos = self.effective_pos(pos);

        let (up, down) = match (&self.up_proj, &self.down_proj) {
            (Some(u), Some(d)) => (u, d),
            _ => return 0.0,
        };

        let rank = self.rank.unwrap_or(16);

        let mut vq = ndarray::Array1::zeros(rank);
        ndarray::linalg::general_mat_vec_mul(1.0, down, q, 0.0, &mut vq);

        let up_row = up.row(eff_pos);
        let raw = up_row.dot(&vq);

        log1p_exp(raw)
    }

    /// Get total parameter count
    pub fn parameters(&self) -> usize {
        match self.cope_type {
            WindowAwareCoPEType::Standard => self.pos_embeddings.as_ref().map_or(0, |pe| pe.len()),
            WindowAwareCoPEType::Optimized => {
                let up = self.up_proj.as_ref().map_or(0, |m| m.len());
                let down = self.down_proj.as_ref().map_or(0, |m| m.len());
                let wg = self.w_gate.as_ref().map_or(0, |m| m.len());
                let bg = self.b_gate.as_ref().map_or(0, |m| m.len());
                up + down + wg + bg
            }
            WindowAwareCoPEType::Gated => {
                let pe = self.pos_embeddings.as_ref().map_or(0, |m| m.len());
                let wg = self.w_gate.as_ref().map_or(0, |m| m.len());
                let bg = self.b_gate.as_ref().map_or(0, |m| m.len());
                pe + wg + bg
            }
            WindowAwareCoPEType::Factorized => {
                let up = self.up_proj.as_ref().map_or(0, |m| m.len());
                let down = self.down_proj.as_ref().map_or(0, |m| m.len());
                up + down
            }
        }
    }

    /// Get weight norm
    pub fn weight_norm(&self) -> f32 {
        let mut sum = 0.0f32;
        if let Some(pe) = &self.pos_embeddings {
            sum += pe.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(up) = &self.up_proj {
            sum += up.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(down) = &self.down_proj {
            sum += down.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(wg) = &self.w_gate {
            sum += wg.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(bg) = &self.b_gate {
            sum += bg.iter().map(|&w| w * w).sum::<f32>();
        }
        sum.sqrt()
    }

    /// Apply gradients placeholder - for full implementation
    pub fn apply_gradients(&mut self, _grads: &[Array2<f32>], _lr: f32) {
        // Full gradient implementation would go here
        // For now, this is a placeholder for the window-aware interface
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_window_aware_standard() {
        let cope = WindowAwareCoPE::new_standard(128, 32, Some(16));
        let q = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work normally
        for pos in 0..16 {
            let contrib = cope.cope_contribution(&q.view(), pos);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.cope_contribution(&q.view(), pos);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }

    #[test]
    fn test_window_aware_optimized() {
        let cope = WindowAwareCoPE::new_optimized(128, 32, 8, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.3);

        // Positions 0-15 should work normally
        for pos in 0..16 {
            let contrib = cope.optimized_cope_contribution(&q.view(), &k.view(), pos);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.optimized_cope_contribution(&q.view(), &k.view(), pos);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }

    #[test]
    fn test_dynamic_window_resize() {
        let mut cope = WindowAwareCoPE::new_standard(128, 32, Some(16));
        let q = Array1::from_elem(32, 0.5);

        // Initially window is 16: positions 0-15 are valid, 16+ return 0
        assert!(
            cope.cope_contribution(&q.view(), 15).is_finite(),
            "pos 15 should be in window"
        );
        assert_eq!(
            cope.cope_contribution(&q.view(), 16),
            0.0,
            "pos 16 should be out of window"
        );

        // Expand window to 32: positions 0-31 valid (pos < 32)
        cope.set_window_size(Some(32));
        assert!(cope.cope_contribution(&q.view(), 15).is_finite());
        assert!(
            cope.cope_contribution(&q.view(), 31).is_finite(),
            "pos 31 should be in window"
        );
        assert_eq!(
            cope.cope_contribution(&q.view(), 32),
            0.0,
            "pos 32 should be out of window"
        );

        // Disable window: all positions valid
        cope.set_window_size(None);
        assert!(cope.cope_contribution(&q.view(), 64).is_finite());
    }

    #[test]
    fn test_parameters() {
        let cope = WindowAwareCoPE::new_standard(128, 32, Some(64));
        // Parameters should be the same regardless of window
        assert_eq!(cope.parameters(), (128 + 1) * 32);
    }

    #[test]
    fn test_window_aware_gated() {
        let cope = WindowAwareCoPE::new_gated(128, 32, Some(16));
        let q = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work
        for pos in 0..16 {
            let contrib = cope.cope_contribution(&q.view(), pos);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.cope_contribution(&q.view(), pos);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }

    #[test]
    fn test_window_aware_factorized() {
        let cope = WindowAwareCoPE::new_factorized(128, 32, 8, Some(16));
        let q = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work
        for pos in 0..16 {
            let contrib = cope.factorized_cope_contribution(&q.view(), pos);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.factorized_cope_contribution(&q.view(), pos);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }
}
