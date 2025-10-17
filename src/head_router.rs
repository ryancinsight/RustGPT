/// Head Router for Mixture-of-Heads (MoH) attention
///
/// Implements dynamic head selection per token using a learned routing mechanism.
/// Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
///
/// # Architecture
///
/// The router uses a two-stage routing mechanism:
///
/// 1. **Shared Heads** (always active): Capture common knowledge across all tokens
/// 2. **Routed Heads** (Top-K selection): Specialize for specific patterns
/// 3. **Head Type Balancing**: Learns to balance shared vs. routed contributions
///
/// # Mathematical Formulation
///
/// ```text
/// Shared head scores:  g_s = α₁ × Softmax(W_s @ x^T)
/// Routed head scores:  g_r = α₂ × Softmax(W_r @ x^T)  [Top-K selected]
/// Head type weights:   [α₁, α₂] = Softmax(W_h @ x^T)
/// ```
///
/// # Load Balance Loss
///
/// ```text
/// L_b = Σ(i=1 to num_routed) P_i × f_i
/// ```
///
/// Where:
/// - P_i = average routing score for head i
/// - f_i = fraction of tokens that selected head i
///
/// This prevents routing collapse (all tokens routing to same heads).
///
/// # Example
///
/// ```ignore
/// let router = HeadRouter::new(
///     128,  // embedding_dim
///     2,    // num_shared_heads
///     6,    // num_routed_heads
///     4,    // num_active_routed_heads (Top-4)
///     0.01, // load_balance_weight
/// );
///
/// let input = Array2::ones((10, 128));  // batch_size=10, embedding_dim=128
/// let mask = router.route(&input);      // (10, 8) boolean mask
/// // mask[i][j] = true if head j is active for token i
/// ```

use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::routing;

/// Router network for Mixture-of-Heads attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeadRouter {
    /// Number of shared heads (always active)
    num_shared_heads: usize,
    
    /// Number of routed heads (Top-K selection)
    num_routed_heads: usize,
    
    /// Number of routed heads to activate (K in Top-K)
    num_active_routed_heads: usize,
    
    /// Embedding dimension
    embedding_dim: usize,
    
    /// Router for shared heads: W_s ∈ R^(num_shared × embedding_dim)
    w_shared: Array2<f32>,
    
    /// Router for routed heads: W_r ∈ R^(num_routed × embedding_dim)
    w_routed: Array2<f32>,
    
    /// Head type balancing: W_h ∈ R^(2 × embedding_dim)
    w_head_type: Array2<f32>,
    
    /// Optimizers for router weights
    optimizer_shared: Adam,
    optimizer_routed: Adam,
    optimizer_head_type: Adam,
    
    /// Cached routing scores for backward pass
    #[serde(skip)]
    cached_routing_scores_shared: Option<Array2<f32>>,
    
    #[serde(skip)]
    cached_routing_scores_routed: Option<Array2<f32>>,
    
    #[serde(skip)]
    cached_head_type_weights: Option<Array2<f32>>,
    
    #[serde(skip)]
    cached_activation_mask: Option<Array2<bool>>,
    
    /// Load balance loss weight (β in paper)
    load_balance_weight: f32,
}

impl HeadRouter {
    /// Create a new head router
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input embedding dimension
    /// * `num_shared_heads` - Number of shared heads (always active)
    /// * `num_routed_heads` - Number of routed heads (Top-K selection)
    /// * `num_active_routed_heads` - Number of routed heads to activate (K)
    /// * `load_balance_weight` - Weight for load balance loss (typically 0.01)
    ///
    /// # Returns
    ///
    /// A new HeadRouter with Xavier-initialized weights
    ///
    /// # Panics
    ///
    /// Panics if `num_active_routed_heads > num_routed_heads`
    pub fn new(
        embedding_dim: usize,
        num_shared_heads: usize,
        num_routed_heads: usize,
        num_active_routed_heads: usize,
        load_balance_weight: f32,
    ) -> Self {
        assert!(
            num_active_routed_heads <= num_routed_heads,
            "num_active_routed_heads ({}) must be <= num_routed_heads ({})",
            num_active_routed_heads,
            num_routed_heads
        );
        
        let mut rng = rand::rng();
        
        // Xavier initialization: std = sqrt(2 / fan_in)
        let std_shared = (2.0 / embedding_dim as f32).sqrt();
        let std_routed = (2.0 / embedding_dim as f32).sqrt();
        let std_head_type = (2.0 / embedding_dim as f32).sqrt();
        
        let normal_shared = Normal::new(0.0, std_shared).unwrap();
        let normal_routed = Normal::new(0.0, std_routed).unwrap();
        let normal_head_type = Normal::new(0.0, std_head_type).unwrap();
        
        HeadRouter {
            num_shared_heads,
            num_routed_heads,
            num_active_routed_heads,
            embedding_dim,
            w_shared: Array2::from_shape_fn(
                (num_shared_heads, embedding_dim),
                |_| normal_shared.sample(&mut rng),
            ),
            w_routed: Array2::from_shape_fn(
                (num_routed_heads, embedding_dim),
                |_| normal_routed.sample(&mut rng),
            ),
            w_head_type: Array2::from_shape_fn(
                (2, embedding_dim),
                |_| normal_head_type.sample(&mut rng),
            ),
            optimizer_shared: Adam::new((num_shared_heads, embedding_dim)),
            optimizer_routed: Adam::new((num_routed_heads, embedding_dim)),
            optimizer_head_type: Adam::new((2, embedding_dim)),
            cached_routing_scores_shared: None,
            cached_routing_scores_routed: None,
            cached_head_type_weights: None,
            cached_activation_mask: None,
            load_balance_weight,
        }
    }
    
    /// Route input to select which heads to activate
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Boolean mask of shape (seq_len, total_heads) where:
    /// - First `num_shared_heads` columns are always true (shared heads)
    /// - Remaining columns have exactly `num_active_routed_heads` true values per row
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mask = router.route(&input);  // (seq_len, 8)
    /// // mask[i][0..2] = [true, true]  (shared heads always active)
    /// // mask[i][2..8] has exactly 4 true values (Top-4 routed heads)
    /// ```
    pub fn route(&mut self, input: &Array2<f32>) -> Array2<bool> {
        let (seq_len, _) = input.dim();
        let total_heads = self.num_shared_heads + self.num_routed_heads;
        
        // 1. Compute shared head scores: Softmax(W_s @ input^T)
        let shared_logits = input.dot(&self.w_shared.t());
        let shared_scores = routing::softmax(&shared_logits);
        
        // 2. Compute routed head scores: Softmax(W_r @ input^T)
        let routed_logits = input.dot(&self.w_routed.t());
        let routed_scores = routing::softmax(&routed_logits);
        
        // 3. Compute head type balancing: Softmax(W_h @ input^T)
        let head_type_logits = input.dot(&self.w_head_type.t());
        let head_type_weights = routing::softmax(&head_type_logits);
        
        // 4. Select Top-K routed heads
        let top_k_indices = routing::top_k_indices(&routed_scores, self.num_active_routed_heads);
        
        // 5. Create activation mask
        let mut mask = Array2::<bool>::from_elem((seq_len, total_heads), false);
        
        for token_idx in 0..seq_len {
            // Shared heads are always active
            for head_idx in 0..self.num_shared_heads {
                mask[[token_idx, head_idx]] = true;
            }
            
            // Routed heads: activate Top-K
            for &routed_idx in &top_k_indices[token_idx] {
                let head_idx = self.num_shared_heads + routed_idx;
                mask[[token_idx, head_idx]] = true;
            }
        }
        
        // Cache for backward pass
        self.cached_routing_scores_shared = Some(shared_scores);
        self.cached_routing_scores_routed = Some(routed_scores);
        self.cached_head_type_weights = Some(head_type_weights);
        self.cached_activation_mask = Some(mask.clone());
        
        mask
    }
    
    /// Compute load balance loss for the last routing operation
    ///
    /// # Returns
    ///
    /// Scalar load balance loss value, or 0.0 if no routing has been performed
    ///
    /// # Note
    ///
    /// This should be called after `route()` and added to the total loss during training.
    pub fn compute_load_balance_loss(&self) -> f32 {
        if let (Some(routed_scores), Some(mask)) = (
            &self.cached_routing_scores_routed,
            &self.cached_activation_mask,
        ) {
            // Extract routed head portion of mask
            let routed_mask = mask.slice(ndarray::s![.., self.num_shared_heads..]).to_owned();
            
            routing::compute_load_balance_loss(routed_scores, &routed_mask)
        } else {
            0.0
        }
    }
    
    /// Get the total number of heads (shared + routed)
    pub fn total_heads(&self) -> usize {
        self.num_shared_heads + self.num_routed_heads
    }
    
    /// Get the number of parameters in the router
    pub fn parameters(&self) -> usize {
        self.w_shared.len() + self.w_routed.len() + self.w_head_type.len()
    }
    
    /// Get the load balance weight
    pub fn load_balance_weight(&self) -> f32 {
        self.load_balance_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_head_router_creation() {
        let router = HeadRouter::new(128, 2, 6, 4, 0.01);
        
        assert_eq!(router.total_heads(), 8);
        assert_eq!(router.num_shared_heads, 2);
        assert_eq!(router.num_routed_heads, 6);
        assert_eq!(router.num_active_routed_heads, 4);
        
        // Check parameter count: 2×128 + 6×128 + 2×128 = 1280
        assert_eq!(router.parameters(), 1280);
    }

    #[test]
    fn test_head_router_route_shape() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01);
        let input = Array2::ones((10, 128));
        
        let mask = router.route(&input);
        
        assert_eq!(mask.shape(), &[10, 8]);
    }

    #[test]
    fn test_head_router_shared_heads_always_active() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01);
        let input = Array2::ones((10, 128));
        
        let mask = router.route(&input);
        
        // Check that first 2 heads (shared) are always active
        for token_idx in 0..10 {
            assert!(mask[[token_idx, 0]], "Shared head 0 should be active");
            assert!(mask[[token_idx, 1]], "Shared head 1 should be active");
        }
    }

    #[test]
    fn test_head_router_correct_number_active() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01);
        let input = Array2::ones((10, 128));
        
        let mask = router.route(&input);
        
        // Check that exactly 6 heads are active per token (2 shared + 4 routed)
        for token_idx in 0..10 {
            let active_count = mask.row(token_idx).iter().filter(|&&x| x).count();
            assert_eq!(active_count, 6, "Should have exactly 6 active heads");
        }
    }

    #[test]
    fn test_head_router_load_balance_loss() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01);
        let input = Array2::ones((10, 128));
        
        router.route(&input);
        let loss = router.compute_load_balance_loss();
        
        // Loss should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    #[should_panic(expected = "must be <=")]
    fn test_head_router_invalid_k() {
        // Should panic: num_active_routed_heads > num_routed_heads
        HeadRouter::new(128, 2, 6, 10, 0.01);
    }
}

