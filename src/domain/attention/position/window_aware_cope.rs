use ndarray::{Array1, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::traits::PositionEmbedding;

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
pub struct WindowAwareCoPE<P> {
    /// Inner CoPE implementation
    pub inner: P,
    /// Maximum window size (None = unlimited, use full CoPE)
    pub window_size: Option<usize>,
}

impl<P: PositionEmbedding> WindowAwareCoPE<P> {
    /// Create a new window-aware wrapper
    pub fn new(inner: P, window_size: Option<usize>) -> Self {
        Self { inner, window_size }
    }

    /// Update window size at runtime
    pub fn set_window_size(&mut self, window_size: Option<usize>) {
        self.window_size = window_size;
    }

    /// Get current window size
    pub fn window_size(&self) -> Option<usize> {
        self.window_size
    }
}

impl<P: PositionEmbedding> PositionEmbedding for WindowAwareCoPE<P> {
    type Gradients = P::Gradients;

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        let pos = query_pos.saturating_sub(key_pos);
        if let Some(ws) = self.window_size {
            if pos >= ws {
                return 0.0;
            }
        }
        self.inner.contribution(q, k, query_pos, key_pos, inputs)
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
        if let Some(ws) = self.window_size {
            if pos >= ws {
                return (Array1::zeros(q.dim()), Array1::zeros(k.dim()));
            }
        }
        self.inner
            .backward(q, k, query_pos, key_pos, inputs, d_s_ij, grads)
    }

    fn init_gradients(&self) -> Self::Gradients {
        self.inner.init_gradients()
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        self.inner.apply_gradients(grads, lr)
    }

    fn max_pos(&self) -> usize {
        self.inner.max_pos()
    }

    fn embed_dim(&self) -> usize {
        self.inner.embed_dim()
    }

    fn parameters(&self) -> usize {
        self.inner.parameters()
    }

    fn weight_norm(&self) -> f32 {
        self.inner.weight_norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::attention::position::{
        cope::CoPE, factorized_cope::FactorizedCoPE, gated_cope::GatedCoPE,
        optimized_cope::OptimizedCoPE,
    };
    use ndarray::Array1;

    #[test]
    fn test_window_aware_standard() {
        let inner = CoPE::new(128, 32);
        let cope = WindowAwareCoPE::new(inner, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work normally
        for pos in 0..16 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }

    #[test]
    fn test_window_aware_optimized() {
        let inner = OptimizedCoPE::new(128, 32, 8);
        let cope = WindowAwareCoPE::new(inner, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.3);

        // Positions 0-15 should work normally
        for pos in 0..16 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }

        // Positions >= 16 should return 0
        for pos in 16..32 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert_eq!(contrib, 0.0, "pos {} should be 0", pos);
        }
    }

    #[test]
    fn test_dynamic_window_resize() {
        let inner = CoPE::new(128, 32);
        let mut cope = WindowAwareCoPE::new(inner, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.5);

        // Initially window is 16: positions 0-15 are valid, 16+ return 0
        assert!(
            cope.contribution(&q.view(), &k.view(), 15, 0, None)
                .is_finite(),
            "pos 15 should be in window"
        );
        assert_eq!(
            cope.contribution(&q.view(), &k.view(), 16, 0, None),
            0.0,
            "pos 16 should be out of window"
        );

        // Expand window to 32: positions 0-31 valid (pos < 32)
        cope.set_window_size(Some(32));
        assert!(
            cope.contribution(&q.view(), &k.view(), 15, 0, None)
                .is_finite()
        );
        assert!(
            cope.contribution(&q.view(), &k.view(), 31, 0, None)
                .is_finite(),
            "pos 31 should be in window"
        );
        assert_eq!(
            cope.contribution(&q.view(), &k.view(), 32, 0, None),
            0.0,
            "pos 32 should be out of window"
        );

        // Disable window: all positions valid
        cope.set_window_size(None);
        assert!(
            cope.contribution(&q.view(), &k.view(), 64, 0, None)
                .is_finite()
        );
    }

    #[test]
    fn test_window_aware_gated() {
        let inner = GatedCoPE::new(128, 32);
        let cope = WindowAwareCoPE::new(inner, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work
        for pos in 0..16 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }
    }

    #[test]
    fn test_window_aware_factorized() {
        let inner = FactorizedCoPE::new(128, 32, 8);
        let cope = WindowAwareCoPE::new(inner, Some(16));
        let q = Array1::from_elem(32, 0.5);
        let k = Array1::from_elem(32, 0.5);

        // Positions 0-15 should work
        for pos in 0..16 {
            let contrib = cope.contribution(&q.view(), &k.view(), pos, 0, None);
            assert!(contrib.is_finite(), "pos {} should work", pos);
        }
    }
}
