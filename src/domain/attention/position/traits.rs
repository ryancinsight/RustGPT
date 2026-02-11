use ndarray::{Array1, ArrayView1, ArrayView2};

/// Trait for Position Embeddings (CoPE)
///
/// Defines the interface for calculating position-aware attention contributions
/// and handling backpropagation.
pub trait PositionEmbedding: Send + Sync {
    /// Associated type for gradients container
    type Gradients: Send + Sync;

    /// Compute the position embedding contribution to the attention score.
    ///
    /// # Arguments
    /// * `q` - Query vector
    /// * `k` - Key vector
    /// * `query_pos` - Query position (i)
    /// * `key_pos` - Key position (j)
    /// * `inputs` - Optional full input sequence (required for PathCoPE)
    ///
    /// # Returns
    /// The scalar contribution to add to the attention logit.
    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
    ) -> f32;

    /// Compute gradients for the backward pass.
    ///
    /// # Arguments
    /// * `q` - Query vector
    /// * `k` - Key vector
    /// * `query_pos` - Query position (i)
    /// * `key_pos` - Key position (j)
    /// * `inputs` - Optional full input sequence
    /// * `d_s_ij` - Gradient of the loss with respect to the attention score
    /// * `grads` - Mutable reference to the gradients container
    ///
    /// # Returns
    /// Tuple of (dL/dq, dL/dk)
    fn backward(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
        d_s_ij: f32,
        grads: &mut Self::Gradients,
    ) -> (Array1<f32>, Array1<f32>);

    /// Initialize a new gradients container.
    fn init_gradients(&self) -> Self::Gradients;

    /// Apply gradients to the model parameters.
    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32);

    /// Get the maximum position supported.
    fn max_pos(&self) -> usize;

    /// Get the embedding dimension.
    fn embed_dim(&self) -> usize;

    /// Get the number of parameters.
    fn parameters(&self) -> usize;

    /// Get the weight norm (L2 norm) of the parameters.
    fn weight_norm(&self) -> f32;
}
