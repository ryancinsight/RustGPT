use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Rotary Positional Embedding (RoPE)
///
/// RoPE encodes absolute positional information with rotation matrices and naturally
/// incorporates relative position dependency in self-attention formulation.
///
/// # Mathematical Formulation
///
/// For a d-dimensional embedding, RoPE applies rotation to pairs of dimensions:
///
/// ```text
/// f(x, m) = [
///   x₁ cos(mθ₁) - x₂ sin(mθ₁),
///   x₁ sin(mθ₁) + x₂ cos(mθ₁),
///   x₃ cos(mθ₂) - x₄ sin(mθ₂),
///   x₃ sin(mθ₂) + x₄ cos(mθ₂),
///   ...
/// ]
/// ```
///
/// Where:
/// - `m` is the position index
/// - `θᵢ = base^(-2i/d)` are the frequency bands (default base=10000)
/// - `d` is the embedding dimension
///
/// # Key Properties
///
/// 1. **Relative Position Encoding**: The dot product between rotated queries and keys
///    depends only on their relative position: `<f(q,m), f(k,n)> = g(q, k, m-n)`
///
/// 2. **Zero Parameters**: No learned weights, only geometric transformations
///
/// 3. **Length Extrapolation**: Can handle sequences longer than training length
///
/// # References
///
/// - Su et al. (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding",
///   arXiv:2104.09864
/// - EleutherAI Blog: https://blog.eleuther.ai/rotary-embeddings/
///
/// # Example
///
/// ```rust
/// use llm::rope::RotaryEmbedding;
/// use ndarray::Array2;
///
/// let rope = RotaryEmbedding::new(128, 512); // dim=128, max_seq_len=512
/// let query = Array2::zeros((10, 128)); // (seq_len=10, dim=128)
/// let rotated_query = rope.apply(&query);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RotaryEmbedding {
    /// Embedding dimension (must be even)
    dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Base for frequency computation (default: 10000)
    base: f32,
    /// Precomputed cosine values: (max_seq_len, dim/2)
    cos_cached: Array2<f32>,
    /// Precomputed sine values: (max_seq_len, dim/2)
    sin_cached: Array2<f32>,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `max_seq_len` - Maximum sequence length to support
    ///
    /// # Panics
    ///
    /// Panics if `dim` is odd
    pub fn new(dim: usize, max_seq_len: usize) -> Self {
        Self::with_base(dim, max_seq_len, 10000.0)
    }

    /// Create a new RotaryEmbedding with custom base
    ///
    /// # Arguments
    ///
    /// * `dim` - Embedding dimension (must be even)
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `base` - Base for frequency computation (default: 10000)
    ///
    /// # Panics
    ///
    /// Panics if `dim` is odd
    pub fn with_base(dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(dim % 2 == 0, "Embedding dimension must be even for RoPE");

        // Compute inverse frequencies: θᵢ = base^(-2i/d)
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in 0..(dim / 2) {
            let freq = base.powf(-2.0 * (i as f32) / (dim as f32));
            inv_freq.push(freq);
        }

        // Precompute cos and sin for all positions
        let mut cos_cached = Array2::zeros((max_seq_len, dim / 2));
        let mut sin_cached = Array2::zeros((max_seq_len, dim / 2));

        for pos in 0..max_seq_len {
            for (i, &freq) in inv_freq.iter().enumerate() {
                let angle = (pos as f32) * freq;
                cos_cached[[pos, i]] = angle.cos();
                sin_cached[[pos, i]] = angle.sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            base,
            cos_cached,
            sin_cached,
        }
    }

    /// Apply rotary positional embedding to input tensor
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape (seq_len, dim)
    ///
    /// # Returns
    ///
    /// Rotated tensor of shape (seq_len, dim)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `x.shape()[1]` != `self.dim`
    /// - `x.shape()[0]` > `self.max_seq_len`
    pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, dim) = (x.shape()[0], x.shape()[1]);
        assert_eq!(dim, self.dim, "Input dimension must match RoPE dimension");
        assert!(
            seq_len <= self.max_seq_len,
            "Sequence length exceeds maximum"
        );

        let mut output = Array2::zeros((seq_len, dim));

        // Apply rotation to pairs of dimensions
        for pos in 0..seq_len {
            for i in 0..(dim / 2) {
                let x1 = x[[pos, 2 * i]];
                let x2 = x[[pos, 2 * i + 1]];
                let cos = self.cos_cached[[pos, i]];
                let sin = self.sin_cached[[pos, i]];

                // Rotation matrix application:
                // [cos -sin] [x1]   [x1*cos - x2*sin]
                // [sin  cos] [x2] = [x1*sin + x2*cos]
                output[[pos, 2 * i]] = x1 * cos - x2 * sin;
                output[[pos, 2 * i + 1]] = x1 * sin + x2 * cos;
            }
        }

        output
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Apply rotary positional embedding to query and key tensors
///
/// This is a convenience function for applying RoPE to both Q and K in attention.
///
/// # Arguments
///
/// * `q` - Query tensor of shape (seq_len, dim)
/// * `k` - Key tensor of shape (seq_len, dim)
/// * `rope` - RotaryEmbedding instance
///
/// # Returns
///
/// Tuple of (rotated_q, rotated_k)
pub fn apply_rotary_pos_emb(
    q: &Array2<f32>,
    k: &Array2<f32>,
    rope: &RotaryEmbedding,
) -> (Array2<f32>, Array2<f32>) {
    (rope.apply(q), rope.apply(k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_rope_creation() {
        let rope = RotaryEmbedding::new(128, 512);
        assert_eq!(rope.dim(), 128);
        assert_eq!(rope.max_seq_len(), 512);
    }

    #[test]
    fn test_rope_apply_shape() {
        let rope = RotaryEmbedding::new(64, 100);
        let input = Array2::ones((10, 64));
        let output = rope.apply(&input);
        assert_eq!(output.shape(), &[10, 64]);
    }

    #[test]
    fn test_rope_rotation_properties() {
        let rope = RotaryEmbedding::new(4, 10);
        let input = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
        let output = rope.apply(&input);

        // At position 0, rotation should be identity (cos(0)=1, sin(0)=0)
        assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(output[[0, 1]].abs() < 1e-6);
    }

    #[test]
    fn test_rope_relative_position() {
        // Test that relative position is preserved in dot product
        let rope = RotaryEmbedding::new(64, 100);

        let q1 = Array2::ones((1, 64));
        let k1 = Array2::ones((1, 64));

        let q1_rot = rope.apply(&q1);
        let k1_rot = rope.apply(&k1);

        // Dot product at same position
        let dot1: f32 = q1_rot.iter().zip(k1_rot.iter()).map(|(a, b)| a * b).sum();

        // Create inputs at different positions
        let mut q2 = Array2::ones((2, 64));
        let mut k2 = Array2::ones((2, 64));

        // Set second position
        for i in 0..64 {
            q2[[1, i]] = q1[[0, i]];
            k2[[1, i]] = k1[[0, i]];
        }

        let q2_rot = rope.apply(&q2);
        let k2_rot = rope.apply(&k2);

        // Dot product at same relative position (both at pos 0 vs both at pos 1)
        let dot2: f32 = (0..64).map(|i| q2_rot[[1, i]] * k2_rot[[1, i]]).sum();

        // Should be approximately equal (relative position is the same)
        assert!(
            (dot1 - dot2).abs() < 1e-3,
            "Relative position not preserved: {} vs {}",
            dot1,
            dot2
        );
    }

    #[test]
    #[should_panic(expected = "Embedding dimension must be even")]
    fn test_rope_odd_dimension() {
        RotaryEmbedding::new(63, 100);
    }

    #[test]
    #[should_panic(expected = "Sequence length exceeds maximum")]
    fn test_rope_exceeds_max_len() {
        let rope = RotaryEmbedding::new(64, 10);
        let input = Array2::ones((20, 64));
        rope.apply(&input);
    }
}
