use ndarray::Array2;
use serde::{Deserialize, Serialize};
use crate::adam::Adam;

/// Contextual Position Encoding (CoPE)
///
/// CoPE is a context-aware positional encoding method that allows positions to be
/// conditioned on context by incrementing position only on certain tokens determined
/// by the model. Unlike token-based position encoding (RoPE, learned embeddings),
/// CoPE can count abstract units like words, sentences, or specific token types.
///
/// # Mathematical Formulation
///
/// For a sequence of tokens with query `q_i` and key `k_j` vectors:
///
/// 1. **Gate Computation**: Determines which tokens to count
///    ```text
///    g_ij = σ(q_i^T k_j)
///    ```
///    where σ is the sigmoid function. Gate value of 1 means the token is counted,
///    0 means it's ignored.
///
/// 2. **Position Computation**: Cumulative sum of gates
///    ```text
///    p_ij = Σ(k=j to i) g_ik
///    ```
///    This gives fractional position values based on which tokens are counted.
///
/// 3. **Position Embedding Interpolation**: Since positions are fractional
///    ```text
///    e[p_ij] = (⌈p_ij⌉ - p_ij) * e[⌈p_ij⌉] + (1 - ⌈p_ij⌉ + p_ij) * e[⌊p_ij⌋]
///    ```
///
/// 4. **Attention**: Add position embeddings to keys
///    ```text
///    a_ij = Softmax(q_i^T (k_j + e[p_ij]))
///    ```
///
/// # Key Properties
///
/// 1. **Context-Dependent**: Positions adapt based on input content, not just token count
/// 2. **Fractional Positions**: Smooth interpolation between integer positions
/// 3. **Multi-Head Flexibility**: Each attention head can count different units
/// 4. **Limited Positions**: Can use p_max << T for efficiency
/// 5. **Zero Extra Parameters**: Same parameter count as relative PE
///
/// # Benefits over RoPE
///
/// - Can attend to abstract units (i-th sentence, i-th noun, etc.)
/// - Better out-of-distribution generalization on counting tasks
/// - Improved perplexity on language modeling (22.55 vs 22.90 for relative PE)
/// - Better length extrapolation to longer contexts
///
/// # References
///
/// - Golovneva et al. (2024), "Contextual Position Encoding: Learning to Count What's Important",
///   arXiv:2405.18719
/// - Meta FAIR: https://arxiv.org/abs/2405.18719
///
/// # Example
///
/// ```rust
/// use llm::cop::ContextualPositionEncoding;
/// use ndarray::Array2;
///
/// let cope = ContextualPositionEncoding::new(128, 64); // head_dim=128, max_pos=64
/// let q = Array2::ones((10, 128)); // 10 tokens, 128 dims
/// let k = Array2::ones((10, 128));
///
/// // Compute position-aware attention logits
/// let pos_logits = cope.apply(&q, &k);
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ContextualPositionEncoding {
    /// Dimension of each attention head
    head_dim: usize,

    /// Maximum position value (p_max)
    /// Can be much smaller than sequence length for efficiency
    max_pos: usize,

    /// Position embeddings for integer positions [0, 1, ..., max_pos]
    /// Shape: (max_pos + 1, head_dim)
    pos_embeddings: Array2<f32>,

    /// Optimizer for learned position embeddings
    optimizer: Adam,
}

impl ContextualPositionEncoding {
    /// Create a new CoPE instance
    ///
    /// # Arguments
    ///
    /// * `head_dim` - Dimension of each attention head
    /// * `max_pos` - Maximum position value (p_max). Can be much smaller than sequence length.
    ///   For example, max_pos=64 works well for context length 1024.
    ///
    /// # Returns
    ///
    /// A new `ContextualPositionEncoding` instance with randomly initialized position embeddings
    ///
    /// # Example
    ///
    /// ```rust
    /// use llm::cop::ContextualPositionEncoding;
    ///
    /// let cope = ContextualPositionEncoding::new(128, 64);
    /// assert_eq!(cope.head_dim(), 128);
    /// assert_eq!(cope.max_pos(), 64);
    /// ```
    pub fn new(head_dim: usize, max_pos: usize) -> Self {
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        // Initialize position embeddings: (max_pos + 1) x head_dim
        let pos_embeddings =
            Array2::from_shape_fn((max_pos + 1, head_dim), |_| normal.sample(&mut rng));

        Self {
            head_dim,
            max_pos,
            pos_embeddings,
            optimizer: Adam::new((max_pos + 1, head_dim)),
        }
    }

    /// Apply contextual position encoding to compute position-aware attention logits
    ///
    /// This is the core CoPE operation that:
    /// 1. Computes gates from query-key dot products
    /// 2. Computes fractional positions via cumulative sum
    /// 3. Interpolates position embeddings
    /// 4. Returns position contribution to attention logits
    ///
    /// # Arguments
    ///
    /// * `q` - Query tensor of shape (seq_len, head_dim)
    /// * `k` - Key tensor of shape (seq_len, head_dim)
    ///
    /// # Returns
    ///
    /// Position logits of shape (seq_len, seq_len) to be added to attention scores
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Query or key dimension doesn't match head_dim
    /// - Query and key have different sequence lengths
    pub fn apply(&self, q: &Array2<f32>, k: &Array2<f32>) -> Array2<f32> {
        // Delegate to window-aware implementation with no window limit
        self.apply_with_window(q, k, None)
    }
    
    /// Apply contextual position encoding with optional sliding window limit
    pub fn apply_with_window(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        window_size: Option<usize>,
    ) -> Array2<f32> {
        let (seq_len_q, dim_q) = (q.shape()[0], q.shape()[1]);
        let (seq_len_k, dim_k) = (k.shape()[0], k.shape()[1]);
    
        assert_eq!(dim_q, self.head_dim, "Query dimension must match head_dim");
        assert_eq!(dim_k, self.head_dim, "Key dimension must match head_dim");
        assert_eq!(
            seq_len_q, seq_len_k,
            "Query and key must have same sequence length",
        );
    
        let seq_len = seq_len_q;
    
        // Precompute q dot E^T for all integer positions (max_pos+1)
        // Shape: (seq_len, max_pos+1)
        let q_pos = q.dot(&self.pos_embeddings.t());
    
        // Allocate position logits output
        let mut pos_logits = Array2::<f32>::zeros((seq_len, seq_len));

        // Pre-allocate reusable temporaries to reduce allocations
        let mut g_i = Array2::<f32>::zeros((seq_len, 1));
        let mut prefix: Vec<f32> = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Compute gates for row i: g_i = sigmoid(K @ q_i)
            let q_i = q.row(i);
            g_i.fill(0.0);
            {
                // Fill g_i column directly without intermediate vec
                for j in 0..seq_len {
                    let dot = k.row(j).dot(&q_i);
                    g_i[[j, 0]] = 1.0 / (1.0 + (-dot).exp());
                }
            }

            // Build prefix sums for gates row i
            prefix.clear();
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                acc += g_i[[j, 0]];
                // Clamp to max_pos to stay within embedding bounds
                prefix.push(acc.min(self.max_pos as f32));
            }
    
            // Compute logits only for causal region j <= i, optionally window-limited
            let j_start = match window_size {
                Some(w) => i.saturating_sub(w) + 1, // inclusive start index within window
                None => 0,
            };
            for j in j_start..=i {
                let p = if j == 0 { prefix[i] } else { (prefix[i] - prefix[j - 1]).max(0.0) };
                let p_floor = p.floor() as usize;
                let p_ceil = p.ceil() as usize;
                let p_floor = p_floor.min(self.max_pos);
                let p_ceil = p_ceil.min(self.max_pos);
    
                if p_floor == p_ceil {
                    // Integer position
                    pos_logits[[i, j]] = q_pos[[i, p_floor]];
                } else {
                    let weight_ceil = p - p_floor as f32;
                    let weight_floor = 1.0 - weight_ceil;
                    pos_logits[[i, j]] =
                        weight_floor * q_pos[[i, p_floor]] + weight_ceil * q_pos[[i, p_ceil]];
                }
            }
        }
        
        pos_logits
    }

    /// Compute gradients for position embeddings and the additional contribution to dL/dQ
    /// from the CoPE position logits path. This ignores gradients through the gating function
    /// (p_ij dependence on Q/K), focusing on direct interpolation path for stability.
    pub fn compute_embedding_grads_and_q_contrib(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        grad_pos_logits: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let (seq_len_q, dim_q) = (q.shape()[0], q.shape()[1]);
        let (seq_len_k, dim_k) = (k.shape()[0], k.shape()[1]);
        assert_eq!(dim_q, self.head_dim);
        assert_eq!(dim_k, self.head_dim);
        assert_eq!(seq_len_q, seq_len_k);
        assert_eq!(grad_pos_logits.shape(), &[seq_len_q, seq_len_q]);

        let seq_len = seq_len_q;

        let mut grad_pos = Array2::<f32>::zeros((self.max_pos + 1, self.head_dim));
        let mut grad_q_cope = Array2::<f32>::zeros((seq_len, self.head_dim));

        // Pre-allocate reusable prefix vector
        let mut prefix = Vec::with_capacity(seq_len);

        for i in 0..seq_len {
            // Compute prefix sums for row i without storing full gates matrix
            prefix.clear();
            let q_i = q.row(i);
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                let dot = q_i.dot(&k.row(j));
                let gate = 1.0 / (1.0 + (-dot).exp());
                acc += gate;
                prefix.push(acc.min(self.max_pos as f32));
            }

            // Views for current q_i and grad_q_cope_i
            let q_i = q.row(i);
            let mut grad_q_row = grad_q_cope.row_mut(i);

            for j in 0..=i {
                let grad_scalar = grad_pos_logits[[i, j]];
                if grad_scalar == 0.0 { continue; }

                let p = if j == 0 { prefix[i] } else { (prefix[i] - prefix[j - 1]).max(0.0) };
                let p_floor = p.floor() as usize;
                let p_ceil = p.ceil() as usize;
                let p_floor = p_floor.min(self.max_pos);
                let p_ceil = p_ceil.min(self.max_pos);

                let weight_ceil = p - p_floor as f32;
                let weight_floor = 1.0 - weight_ceil;

                // dL/dE[p_floor] += grad_scalar * weight_floor * q_i
                // dL/dE[p_ceil]  += grad_scalar * weight_ceil  * q_i
                {
                    let mut grad_floor_row = grad_pos.row_mut(p_floor);
                    ndarray::Zip::from(&mut grad_floor_row)
                        .and(q_i)
                        .for_each(|gf, &qv| { *gf += grad_scalar * weight_floor * qv; });
                }
                if p_ceil != p_floor {
                    let mut grad_ceil_row = grad_pos.row_mut(p_ceil);
                    ndarray::Zip::from(&mut grad_ceil_row)
                        .and(q_i)
                        .for_each(|gc, &qv| { *gc += grad_scalar * weight_ceil * qv; });
                }

                // Additional gradient on q_i from CoPE path: grad_scalar * (weight_floor * E[p_floor] + weight_ceil * E[p_ceil])
                let e_floor = self.pos_embeddings.row(p_floor);
                if p_ceil == p_floor {
                    ndarray::Zip::from(&mut grad_q_row)
                        .and(e_floor)
                        .for_each(|gq, &ev| { *gq += grad_scalar * ev; });
                } else {
                    let e_ceil = self.pos_embeddings.row(p_ceil);
                    ndarray::Zip::from(&mut grad_q_row)
                        .and(e_floor)
                        .and(e_ceil)
                        .for_each(|gq, &ef, &ec| { *gq += grad_scalar * (weight_floor * ef + weight_ceil * ec); });
                }
            }
        }

        (grad_pos, grad_q_cope)
    }

    /// Apply gradients to learned position embeddings
    pub fn apply_gradients(&mut self, grad_pos_embeddings: &Array2<f32>, lr: f32) {
        self.optimizer.step(&mut self.pos_embeddings, grad_pos_embeddings, lr);
    }

    /// Get mutable reference to position embeddings for gradient updates
    pub fn pos_embeddings_mut(&mut self) -> &mut Array2<f32> {
        &mut self.pos_embeddings
    }

    /// Get reference to position embeddings
    pub fn pos_embeddings(&self) -> &Array2<f32> {
        &self.pos_embeddings
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the maximum position value
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cope_creation() {
        let cope = ContextualPositionEncoding::new(128, 64);
        assert_eq!(cope.head_dim(), 128);
        assert_eq!(cope.max_pos(), 64);
        assert_eq!(cope.pos_embeddings().shape(), &[65, 128]); // max_pos + 1
    }

    #[test]
    fn test_cope_apply_shape() {
        let cope = ContextualPositionEncoding::new(64, 32);
        let q = Array2::ones((10, 64));
        let k = Array2::ones((10, 64));

        let pos_logits = cope.apply(&q, &k);
        assert_eq!(pos_logits.shape(), &[10, 10]);
    }

    #[test]
    fn test_cope_causal_structure() {
        let cope = ContextualPositionEncoding::new(64, 32);
        let q = Array2::ones((5, 64));
        let k = Array2::ones((5, 64));

        let pos_logits = cope.apply(&q, &k);

        // Check that future positions (j > i) are zero (not computed)
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert_eq!(pos_logits[[i, j]], 0.0);
            }
        }
    }
}
