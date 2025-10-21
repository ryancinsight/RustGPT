use std::f32;

use ndarray::{Array2, ArrayView2, ArrayBase, Ix2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::EMBEDDING_DIM;
use crate::adam::Adam;
use crate::cop::ContextualPositionEncoding;
use crate::head_router::{RouterType, HeadRouterStandard, FullyAdaptiveHeadRouter};
use crate::llm::Layer;
use crate::model_config::{HeadSelectionStrategy, WindowAdaptationStrategy};
use crate::rope::RotaryEmbedding;

fn default_entropy_ema_alpha() -> f32 { 0.2 }

/// Positional encoding variant used in attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PositionalEncodingVariant {
    /// No positional encoding in attention (handled by Embeddings layer)
    Learned,
    /// Rotary Positional Encoding
    RoPE(RotaryEmbedding),
    /// Contextual Position Encoding (one per head for multi-head flexibility)
    CoPE(Vec<ContextualPositionEncoding>),
}

/// Single head for multi-head attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AttentionHead {
    pub head_dim: usize,
    w_q: Array2<f32>, // Weight matrices for this head's Q, K, V
    w_k: Array2<f32>,
    w_v: Array2<f32>,

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
}

/// Multi-head self-attention mechanism (standard transformer attention)
/// Supports both Multi-Head Attention (MHA) and Group-Query Attention (GQA)
/// Supports Sliding Window Attention for efficient long-context processing
/// Supports Adaptive Window Sizing (Phase 4 enhancement)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SelfAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    /// If equal to num_heads, this is standard MHA
    /// If less than num_heads, this is GQA with grouped queries
    pub num_kv_heads: usize,
    heads: Vec<AttentionHead>,

    cached_input: Option<Array2<f32>>,

    /// Positional encoding type (Learned, RoPE, or CoPE)
    /// Learned embeddings are handled in the Embeddings layer
    /// RoPE and CoPE are applied within attention
    positional_encoding: PositionalEncodingVariant,

    /// Optional sliding window size for attention
    /// If None, uses full attention (all tokens attend to all previous tokens)
    /// If Some(w), each token only attends to the last w tokens
    window_size: Option<usize>,

    /// Enable adaptive window sizing (Phase 4)
    use_adaptive_window: bool,

    /// Minimum window size for adaptive sizing
    min_window_size: usize,

    /// Maximum window size for adaptive sizing
    max_window_size: usize,

    /// Strategy for adapting window size
    window_adaptation_strategy: WindowAdaptationStrategy,

    /// Current adaptive window size (computed per forward pass)
    #[serde(skip)]
    current_window_size: Option<usize>,

    /// Attention entropy from last forward pass (for AttentionEntropy strategy)
    #[serde(skip)]
    last_attention_entropy: Option<f32>,

    /// EMA alpha for attention entropy smoothing
    #[serde(default = "default_entropy_ema_alpha")]
    entropy_ema_alpha: f32,

    /// Head selection strategy (AllHeads, MixtureOfHeads, StaticPruning, FullyAdaptiveMoH)
    head_selection: HeadSelectionStrategy,

    /// Router for Mixture-of-Heads (if enabled)
    router: Option<RouterType>,

    /// Cached head activation mask from last forward pass (for backward)
    #[serde(skip)]
    cached_head_mask: Option<Array2<bool>>,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM)
    }
}

impl AttentionHead {
    /// Create a new attention head
    fn new(head_dim: usize) -> Self {
        let mut rng = rand::rng();
        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / head_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        AttentionHead {
            head_dim,
            w_q: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((head_dim, head_dim), |_| normal.sample(&mut rng)),
            optimizer_w_q: Adam::new((head_dim, head_dim)),
            optimizer_w_k: Adam::new((head_dim, head_dim)),
            optimizer_w_v: Adam::new((head_dim, head_dim)),
        }
    }

    /// Compute Q, K, V for this head with optional RoPE
    /// Note: This method is kept for backward compatibility but is not used in GQA forward pass
    #[allow(dead_code)]
    fn compute_qkv_with_rope(
        &self,
        head_input: &Array2<f32>,
        rope: Option<&RotaryEmbedding>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let mut q = head_input.dot(&self.w_q); // Q = X * W_Q
        let mut k = head_input.dot(&self.w_k); // K = X * W_K
        let v = head_input.dot(&self.w_v); // V = X * W_V

        // Apply RoPE to Q and K if enabled
        if let Some(rope_emb) = rope {
            rope_emb.apply_inplace(&mut q);
            rope_emb.apply_inplace(&mut k);
        }

        (q, k, v)
    }

    /// Compute attention for this head with optional sliding window
    fn attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        window_size: Option<usize>,
    ) -> Array2<f32> {
        let dk = (self.head_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Apply causal masking and optional sliding window masking
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal masking: prevent attention to future tokens
                if j > i {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
                // Sliding window masking: prevent attention to tokens outside window
                else if let Some(window) = window_size {
                    if j < i.saturating_sub(window) {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // In-place softmax to avoid allocating weights
        SelfAttention::softmax_inplace(&mut scores);
        // Use scores (now weights) directly
        scores.dot(v)
    }

    /// Compute attention with additional position bias (for CoPE)
    fn attention_with_position_bias(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        position_logits: &Array2<f32>,
        window_size: Option<usize>,
    ) -> Array2<f32> {
        let dk = (self.head_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Add position logits from CoPE
        scores += position_logits;

        // Apply causal masking and optional sliding window masking
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal masking: prevent attention to future tokens
                if j > i {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
                // Sliding window masking: prevent attention to tokens outside window
                else if let Some(window) = window_size {
                    if j < i.saturating_sub(window) {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // In-place softmax with position bias
        SelfAttention::softmax_inplace(&mut scores);
        scores.dot(v)
    }

    /// Apply softmax to attention scores
    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
         let mut result = scores.clone();

         // Apply softmax row-wise
         for mut row in result.rows_mut() {
             let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
             // Calculate exp for each element
             let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
             let sum_exp: f32 = exp_values.iter().sum();

             // Normalize by sum
             for (i, &exp_val) in exp_values.iter().enumerate() {
                 row[i] = exp_val / sum_exp;
             }
         }

         result
     }
}

impl SelfAttention {
    /// Apply softmax to attention scores (in-place, row-wise)
    fn softmax_inplace(scores: &mut Array2<f32>) {
        for mut row in scores.outer_iter_mut() {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for &x in row.iter() {
                if x > max_val { max_val = x; }
            }
            // Subtract max, exponentiate, and accumulate sum
            let mut sum_exp = 0.0f32;
            for x in row.iter_mut() {
                let e = (*x - max_val).exp();
                *x = e;
                sum_exp += e;
            }
            // Normalize
            let norm = sum_exp.max(1e-12);
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
    /// Initializes multi-head self-attention
    /// num_heads defaults to 8 (standard transformer)
    pub fn new(embedding_dim: usize) -> Self {
        Self::new_with_heads(embedding_dim, 8)
    }

    /// Initialize with specific number of heads
    pub fn new_with_heads(embedding_dim: usize, num_heads: usize) -> Self {
        Self::new_with_config(embedding_dim, num_heads, false, 512)
    }

    /// Initialize with full configuration including RoPE
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `num_heads` - Number of attention heads
    /// * `use_rope` - Whether to use Rotary Positional Encoding
    /// * `max_seq_len` - Maximum sequence length (for RoPE)
    pub fn new_with_config(
        embedding_dim: usize,
        num_heads: usize,
        use_rope: bool,
        max_seq_len: usize,
    ) -> Self {
        // Default to MHA (num_kv_heads = num_heads) and full attention (window_size = None)
        Self::new_with_gqa(
            embedding_dim,
            num_heads,
            num_heads,
            use_rope,
            max_seq_len,
            None,
        )
    }

    /// Initialize with Group-Query Attention (GQA) configuration
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `use_rope` - Whether to use Rotary Positional Encoding
    /// * `max_seq_len` - Maximum sequence length (for RoPE)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `embedding_dim` is not divisible by `num_heads`
    /// - `num_heads` is not divisible by `num_kv_heads`
    /// - `num_kv_heads` is greater than `num_heads`
    pub fn new_with_gqa(
        embedding_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        use_rope: bool,
        max_seq_len: usize,
        window_size: Option<usize>,
    ) -> Self {
        assert!(
            embedding_dim % num_heads == 0,
            "embedding_dim must be divisible by num_heads"
        );
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads (for GQA grouping)"
        );
        assert!(
            num_kv_heads <= num_heads,
            "num_kv_heads cannot be greater than num_heads"
        );

        let head_dim = embedding_dim / num_heads;

        // For GQA: Create num_heads query heads, but only num_kv_heads key/value heads
        // Query heads will be grouped to share KV heads
        let heads = (0..num_heads)
            .map(|_| AttentionHead::new(head_dim))
            .collect();

        // Convert boolean use_rope to PositionalEncodingType for backward compatibility
        let positional_encoding = if use_rope {
            PositionalEncodingVariant::RoPE(RotaryEmbedding::new(head_dim, max_seq_len))
        } else {
            PositionalEncodingVariant::Learned
        };

        SelfAttention {
            embedding_dim,
            num_heads,
            num_kv_heads,
            heads,
            cached_input: None,
            positional_encoding,
            window_size,
            use_adaptive_window: false, // Default to fixed window
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            current_window_size: None,
            last_attention_entropy: None,
            entropy_ema_alpha: 0.2,
            head_selection: HeadSelectionStrategy::AllHeads, // Default for backward compatibility
            router: None,
            cached_head_mask: None,
        }
    }

    /// Create a new SelfAttention with explicit positional encoding type
    ///
    /// This is the modern constructor that accepts `PositionalEncodingType` directly.
    /// Use this for new code instead of the boolean `use_rope` parameter.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Embedding dimension
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `positional_encoding` - Type of positional encoding to use
    /// * `max_seq_len` - Maximum sequence length
    /// * `window_size` - Optional sliding window size
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `embedding_dim` is not divisible by `num_heads`
    /// - `num_heads` is not divisible by `num_kv_heads`
    /// - `num_kv_heads` is greater than `num_heads`
    pub fn new_with_positional_encoding(
        embedding_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        positional_encoding: &crate::model_config::PositionalEncodingType,
        max_seq_len: usize,
        window_size: Option<usize>,
    ) -> Self {
        assert!(
            embedding_dim % num_heads == 0,
            "embedding_dim must be divisible by num_heads"
        );
        assert!(
            num_heads % num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads (for GQA grouping)"
        );
        assert!(
            num_kv_heads <= num_heads,
            "num_kv_heads cannot be greater than num_heads"
        );

        let head_dim = embedding_dim / num_heads;

        // Create attention heads
        let heads = (0..num_heads)
            .map(|_| AttentionHead::new(head_dim))
            .collect();

        // Convert PositionalEncodingType to PositionalEncodingVariant
        use crate::model_config::PositionalEncodingType;
        let positional_encoding_variant = match positional_encoding {
            PositionalEncodingType::Learned => PositionalEncodingVariant::Learned,
            PositionalEncodingType::RoPE => {
                PositionalEncodingVariant::RoPE(RotaryEmbedding::new(head_dim, max_seq_len))
            }
            PositionalEncodingType::CoPE { max_pos } => {
                // Create one CoPE instance per head for multi-head flexibility
                let cope_heads = (0..num_heads)
                    .map(|_| ContextualPositionEncoding::new(head_dim, *max_pos))
                    .collect();
                PositionalEncodingVariant::CoPE(cope_heads)
            }
        };

        SelfAttention {
            embedding_dim,
            num_heads,
            num_kv_heads,
            heads,
            cached_input: None,
            positional_encoding: positional_encoding_variant,
            window_size,
            use_adaptive_window: false,
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            current_window_size: None,
            last_attention_entropy: None,
            entropy_ema_alpha: 0.2,
            head_selection: HeadSelectionStrategy::AllHeads, // Default for backward compatibility
            router: None,
            cached_head_mask: None,
        }
    }

    /// Create a new SelfAttention with full adaptive window configuration
    /// Uses a builder pattern to avoid too many parameters
    pub fn new_with_adaptive_window(
        embedding_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        use_rope: bool,
        max_seq_len: usize,
        window_size: Option<usize>,
    ) -> AdaptiveWindowBuilder {
        AdaptiveWindowBuilder {
            embedding_dim,
            num_heads,
            num_kv_heads,
            use_rope,
            max_seq_len,
            window_size,
            use_adaptive_window: true,
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            entropy_ema_alpha: default_entropy_ema_alpha(),
        }
    }

    /// Set the head selection strategy for this attention layer
    ///
    /// This method should be called after construction to enable MoH or other head selection strategies.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The head selection strategy to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut attention = SelfAttention::new_with_positional_encoding(...);
    /// attention.set_head_selection(HeadSelectionStrategy::MixtureOfHeads {
    ///     num_shared_heads: 2,
    ///     num_active_routed_heads: 4,
    ///     load_balance_weight: 0.01,
    /// });
    /// ```
    pub fn set_head_selection(&mut self, strategy: HeadSelectionStrategy, layer_idx: usize) {
        self.head_selection = strategy.clone();

        // Initialize router if MoH is enabled
        match &strategy {
            HeadSelectionStrategy::MixtureOfHeads {
                num_shared_heads,
                num_active_routed_heads,
                load_balance_weight,
                threshold_p_base,
                dynamic_loss_weight_base,
                use_learned_threshold,
                target_avg_routed_heads,
                confidence_threshold,
                use_confidence_fallback,
            } => {
                let num_routed_heads = self.num_heads - num_shared_heads;
                self.router = Some(RouterType::Standard(HeadRouterStandard::new(
                    self.embedding_dim,
                    *num_shared_heads,
                    num_routed_heads,
                    *num_active_routed_heads,
                    *load_balance_weight,
                    *threshold_p_base,
                    *dynamic_loss_weight_base,
                    layer_idx,
                    *use_learned_threshold,
                    *target_avg_routed_heads,
                    *confidence_threshold,
                    *use_confidence_fallback,
                )));
            }
            HeadSelectionStrategy::FullyAdaptiveMoH {
                min_heads,
                max_heads,
                load_balance_weight,
                complexity_loss_weight,
                sparsity_weight,
            } => {
                self.router = Some(RouterType::FullyAdaptive(FullyAdaptiveHeadRouter::new(
                    self.embedding_dim,
                    self.num_heads,
                    *min_heads,
                    *max_heads,
                    *load_balance_weight,
                    *complexity_loss_weight,
                    *sparsity_weight,
                )));
            }
            _ => {
                self.router = None;
            }
        }
    }

    /// Get the total auxiliary loss from the router (if MoH is enabled)
    ///
    /// For standard MoH: load balance + dynamic loss
    /// For fully adaptive MoH: load balance + complexity + sparsity loss
    ///
    /// Returns 0.0 if MoH is not enabled or no routing has been performed yet.
    pub fn get_auxiliary_loss(&self) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| r.compute_auxiliary_loss())
    }

    /// Get the load balance loss from the router (if MoH is enabled)
    ///
    /// Returns 0.0 if MoH is not enabled or no routing has been performed yet.
    pub fn get_load_balance_loss(&self) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| match r {
                RouterType::Standard(router) => router.compute_load_balance_loss(),
                RouterType::FullyAdaptive(router) => router.compute_load_balance_loss(),
            })
    }

    /// Get the dynamic loss (entropy minimization) from the router (if MoH is enabled)
    ///
    /// For Fully Adaptive MoH, this returns complexity + sparsity losses instead of entropy.
    ///
    /// Returns 0.0 if MoH is not enabled or no routing has been performed yet.
    pub fn get_dynamic_loss(&self) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| match r {
                RouterType::Standard(router) => router.compute_dynamic_loss(),
                RouterType::FullyAdaptive(router) => {
                    // CRITICAL FIX: Include complexity and sparsity losses
                    router.compute_complexity_loss() + router.compute_sparsity_loss()
                }
            })
    }

    /// Get the adaptive dynamic loss weight from the router (if MoH is enabled)
    ///
    /// # Arguments
    ///
    /// * `training_progress` - Training progress in range [0.0, 1.0]
    ///
    /// Returns 0.0 if MoH is not enabled.
    pub fn get_dynamic_loss_weight(&self, training_progress: f32) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| match r {
                RouterType::Standard(router) => router.dynamic_loss_weight(training_progress),
                RouterType::FullyAdaptive(_) => 0.0, // Not applicable
            })
    }

    /// Get average number of active routed heads per token (if MoH is enabled)
    ///
    /// Returns 0.0 if MoH is not enabled or no routing has been performed yet.
    pub fn get_avg_active_routed_heads(&self) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| match r {
                RouterType::Standard(router) => router.avg_active_routed_heads(),
                RouterType::FullyAdaptive(router) => {
                    let (avg_heads, _, _, _, _) = router.routing_stats();
                    avg_heads
                }
            })
    }

    /// Get threshold statistics (min, max, mean, std) from the router (if MoH is enabled)
    ///
    /// Returns (0.0, 0.0, 0.0, 0.0) if MoH is not enabled or no routing has been performed yet.
    pub fn get_threshold_stats(&self) -> (f32, f32, f32, f32) {
        self.router
            .as_ref()
            .map_or((0.0, 0.0, 0.0, 0.0), |r| match r {
                RouterType::Standard(router) => router.threshold_stats(),
                RouterType::FullyAdaptive(router) => {
                    // Get threshold stats from routing_stats method
                    let (_, _, _, _, avg_threshold) = router.routing_stats();
                    // For fully adaptive, we return avg as all values since we don't track min/max separately
                    (avg_threshold, avg_threshold, avg_threshold, 0.0)
                }
            })
    }

    /// Update router parameters (threshold predictor for standard MoH, all params for fully adaptive)
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for parameter updates
    pub fn update_router(&mut self, lr: f32) {
        if let Some(router) = &mut self.router {
            match router {
                RouterType::Standard(r) => r.update_threshold_predictor(lr),
                RouterType::FullyAdaptive(r) => r.backward(lr),
            }
        }
    }

    /// Update threshold predictor if MoH with learned predictor is enabled (deprecated, use update_router)
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for predictor updates
    pub fn update_threshold_predictor(&mut self, lr: f32) {
        self.update_router(lr);
    }

    /// Check if this layer has a learned threshold predictor
    pub fn has_learned_predictor(&self) -> bool {
        self.router
            .as_ref()
            .map_or(false, |r| r.has_learned_predictor())
    }

    /// Get confidence statistics (avg, min, fallback_pct) from the router (if MoH is enabled)
    ///
    /// Returns (0.0, 0.0, 0.0) if MoH is not enabled or no routing has been performed yet.
    pub fn get_confidence_stats(&self) -> (f32, f32, f32) {
        self.router
            .as_ref()
            .and_then(|r| r.confidence_stats())
            .unwrap_or((0.0, 0.0, 0.0))
    }

    /// Get complexity statistics (avg, min, max) from the router (if MoH is enabled)
    ///
    /// Returns (0.0, 0.0, 0.0) if MoH is not enabled or no routing has been performed yet.
    pub fn get_complexity_stats(&self) -> (f32, f32, f32) {
        self.router
            .as_ref()
            .and_then(|r| r.complexity_stats())
            .unwrap_or((0.0, 0.0, 0.0))
    }

    /// Get predictor weight norm (for tracking convergence)
    ///
    /// Returns 0.0 if MoH is not enabled or no learned predictor is enabled.
    pub fn get_predictor_weight_norm(&self) -> f32 {
        self.router
            .as_ref()
            .map_or(0.0, |r| r.predictor_weight_norm())
    }

    /// Get temperature statistics (avg, min, max) from the router (if fully adaptive MoH is enabled)
    ///
    /// Returns None if MoH is not enabled or not using learned temperature.
    pub fn get_temperature_stats(&self) -> Option<(f32, f32, f32)> {
        self.router
            .as_ref()
            .and_then(|r| r.temperature_stats())
    }

    /// Get all MoH statistics in one call
    ///
    /// Returns (avg_routed_heads, mean_threshold, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm)
    /// Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if MoH is not enabled.
    pub fn get_moh_stats(&self) -> (f32, f32, f32, f32, f32, f32, f32) {
        if self.router.is_none() {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let avg_routed = self.get_avg_active_routed_heads();
        let (_, _, mean_thresh, _) = self.get_threshold_stats();
        let (conf_avg, conf_min, fallback_pct) = self.get_confidence_stats();
        let (complexity_avg, _, _) = self.get_complexity_stats();
        let pred_norm = self.get_predictor_weight_norm();

        (avg_routed, mean_thresh, conf_avg, conf_min, fallback_pct, complexity_avg, pred_norm)
    }

    /// Set epoch information for warm-up and annealing (if MoH is enabled)
    ///
    /// # Arguments
    ///
    /// * `current_epoch` - Current training epoch (0-indexed)
    /// * `max_epochs` - Maximum number of training epochs
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        if let Some(router) = &mut self.router {
            router.set_epoch_info(current_epoch, max_epochs);
        }
    }

    /// Enable adaptive window sizing
    pub fn enable_adaptive_window(
        &mut self,
        min_size: usize,
        max_size: usize,
        strategy: WindowAdaptationStrategy,
    ) {
        self.use_adaptive_window = true;
        self.min_window_size = min_size;
        self.max_window_size = max_size;
        self.window_adaptation_strategy = strategy;
    }

    pub fn set_entropy_ema_alpha(&mut self, alpha: f32) {
        self.entropy_ema_alpha = alpha;
    }

    /// Expose current adaptive window size (for monitoring and tests)
    pub fn get_current_window_size(&self) -> Option<usize> {
        self.current_window_size
    }

    #[cfg(any(test, debug_assertions))]
    /// Test-only setter to inject entropy for adaptation tests
    pub fn set_last_attention_entropy_for_test(&mut self, entropy: f32) {
        self.last_attention_entropy = Some(entropy);
    }

    #[cfg(any(test, debug_assertions))]
    /// Test-only method to recompute window using current entropy without running forward
    pub fn recompute_window_for_test(&mut self, seq_len: usize) -> usize {
        let ws = self.compute_adaptive_window_size(seq_len);
        self.current_window_size = Some(ws);
        ws
    }
}

/// Builder for SelfAttention with adaptive window configuration
pub struct AdaptiveWindowBuilder {
    embedding_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    use_rope: bool,
    max_seq_len: usize,
    window_size: Option<usize>,
    use_adaptive_window: bool,
    min_window_size: usize,
    max_window_size: usize,
    window_adaptation_strategy: WindowAdaptationStrategy,
    // EMA alpha for smoothing attention entropy in adaptive strategy
    entropy_ema_alpha: f32,
}

impl AdaptiveWindowBuilder {
    pub fn min_window_size(mut self, size: usize) -> Self {
        self.min_window_size = size;
        self
    }

    pub fn max_window_size(mut self, size: usize) -> Self {
        self.max_window_size = size;
        self
    }

    pub fn strategy(mut self, strategy: WindowAdaptationStrategy) -> Self {
        self.window_adaptation_strategy = strategy;
        self
    }

    // Configure EMA alpha for entropy smoothing
    pub fn entropy_ema_alpha(mut self, alpha: f32) -> Self {
        self.entropy_ema_alpha = alpha;
        self
    }

    pub fn build(self) -> SelfAttention {
        assert!(
            self.embedding_dim % self.num_heads == 0,
            "embedding_dim must be divisible by num_heads"
        );
        assert!(
            self.num_heads % self.num_kv_heads == 0,
            "num_heads must be divisible by num_kv_heads (for GQA grouping)"
        );
        assert!(
            self.num_kv_heads <= self.num_heads,
            "num_kv_heads cannot be greater than num_heads"
        );
        assert!(
            self.min_window_size <= self.max_window_size,
            "min_window_size must be <= max_window_size"
        );

        let head_dim = self.embedding_dim / self.num_heads;

        let heads = (0..self.num_heads)
            .map(|_| AttentionHead::new(head_dim))
            .collect();

        // Convert boolean use_rope to PositionalEncodingType for backward compatibility
        let positional_encoding = if self.use_rope {
            PositionalEncodingVariant::RoPE(RotaryEmbedding::new(head_dim, self.max_seq_len))
        } else {
            PositionalEncodingVariant::Learned
        };

        SelfAttention {
            embedding_dim: self.embedding_dim,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            heads,
            cached_input: None,
            positional_encoding,
            window_size: self.window_size,
            use_adaptive_window: self.use_adaptive_window,
            min_window_size: self.min_window_size,
            max_window_size: self.max_window_size,
            window_adaptation_strategy: self.window_adaptation_strategy,
            current_window_size: None,
            last_attention_entropy: None,
            entropy_ema_alpha: self.entropy_ema_alpha,
            head_selection: HeadSelectionStrategy::AllHeads,
            router: None,
            cached_head_mask: None,
        }
    }
}

impl SelfAttention {
    /// Compute adaptive window size based on the configured strategy
    fn compute_adaptive_window_size(&self, seq_len: usize) -> usize {
        if !self.use_adaptive_window {
            // Use fixed window size if adaptive is disabled
            return self.window_size.unwrap_or(seq_len);
        }

        match self.window_adaptation_strategy {
            WindowAdaptationStrategy::Fixed => {
                // Fixed strategy: use configured window_size
                self.window_size.unwrap_or(seq_len)
            }
            WindowAdaptationStrategy::SequenceLengthBased => {
                // Scale window with sequence length: window = min(max, max(min, seq_len / 2))
                let proposed_window = seq_len / 2;
                proposed_window.clamp(self.min_window_size, self.max_window_size)
            }
            WindowAdaptationStrategy::AttentionEntropy => {
                // Use attention entropy from last forward pass
                // Higher entropy (diffuse attention) → larger window
                // Lower entropy (focused attention) → smaller window
                if let Some(entropy) = self.last_attention_entropy {
                    // Normalize entropy using theoretical max entropy for current effective window
                    // Max entropy for a uniform distribution over W tokens is ln(W)
                    let ref_window = self.current_window_size.unwrap_or(seq_len).max(2);
                    let max_entropy = (ref_window as f32).ln();
                    let normalized_entropy = if max_entropy > 1e-6 {
                        (entropy / max_entropy).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    // Map entropy to window size: high entropy → max window, low entropy → min window
                    let window_range = self.max_window_size - self.min_window_size;
                    let adaptive_window =
                        self.min_window_size + (window_range as f32 * normalized_entropy) as usize;

                    adaptive_window.clamp(self.min_window_size, self.max_window_size)
                } else {
                    // Fallback to sequence-length-based if no entropy available
                    let proposed_window = seq_len / 2;
                    proposed_window.clamp(self.min_window_size, self.max_window_size)
                }
            }
            WindowAdaptationStrategy::PerplexityBased => {
                // Perplexity-based adaptation: adapts window size based on model uncertainty
                // Implementation requires perplexity computation from model output logits,
                // which would need to be passed from the LLM training loop to this layer.
                // This is a future enhancement tracked in NFR-8.6.
                //
                // Current behavior: Uses sequence-length-based adaptation as a reasonable
                // approximation, since longer sequences often correlate with higher perplexity.
                let proposed_window = seq_len / 2;
                proposed_window.clamp(self.min_window_size, self.max_window_size)
            }
        }
    }

    /// Compute attention entropy for adaptive window sizing
    /// Entropy measures how diffuse the attention distribution is
    #[allow(dead_code)]
    fn compute_attention_entropy(&self, attention_weights: &Array2<f32>) -> f32 {
        let seq_len = attention_weights.shape()[0];
        let mut total_entropy = 0.0;

        for i in 0..seq_len {
            let mut row_entropy = 0.0;
            for j in 0..seq_len {
                let p = attention_weights[[i, j]];
                if p > 1e-10 {
                    // Avoid log(0)
                    row_entropy -= p * p.ln();
                }
            }
            total_entropy += row_entropy;
        }

        // Average entropy across all positions
        total_entropy / seq_len as f32
    }

    /// Compute gradients for a single attention head
    fn compute_head_gradients<S>(
        &self,
        head_idx: usize,
        head_input: ArrayView2<f32>,
        head_output_grad: &ArrayBase<S, Ix2>,
        softmax_scratch: &mut Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>)
    where
        S: ndarray::Data<Elem = f32>,
    {
        let head = &self.heads[head_idx];
        let head_dim = head.head_dim;

        // Recompute forward pass
        let mut q = head_input.dot(&head.w_q);
        let mut k = head_input.dot(&head.w_k);
        let v = head_input.dot(&head.w_v);

        // Apply positional encoding if needed
        match &self.positional_encoding {
            PositionalEncodingVariant::Learned => {
                // No-op: positional encoding handled in Embeddings layer
            }
            PositionalEncodingVariant::RoPE(rope) => {
                rope.apply_inplace(&mut q);
                rope.apply_inplace(&mut k);
            }
            PositionalEncodingVariant::CoPE(_) => {
                // CoPE is applied differently in the forward pass
                // For backward pass, we skip it here
            }
        }

        let dk = head_dim as f32;
        let mut scores = q.dot(&k.t()) / dk.sqrt();

        // Apply causal masking and optional sliding window masking
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Causal masking: prevent attention to future tokens
                if j > i {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
                // Sliding window masking: prevent attention to tokens outside window
                else if let Some(window) = self.window_size {
                    if j < i.saturating_sub(window) {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // In-place softmax and use a view for weights
        Self::softmax_inplace(&mut scores);
        let attn_weights = scores.view();

        // Backward pass
        let grad_attn_weights = head_output_grad.dot(&v.t());
        let grad_v = attn_weights.t().dot(head_output_grad);

        // Softmax backward into reusable buffer
        Self::softmax_backward_into(&attn_weights, &grad_attn_weights, softmax_scratch);
        let grad_scores_view = softmax_scratch.view();

        // Q, K gradients
        let grad_q = grad_scores_view.dot(&k) / dk.sqrt();
        let grad_k = grad_scores_view.t().dot(&q) / dk.sqrt();

        // Weight gradients
        let grad_w_q = head_input.t().dot(&grad_q);
        let grad_w_k = head_input.t().dot(&grad_k);
        let grad_w_v = head_input.t().dot(&grad_v);

        // Input gradients
        let grad_input =
            grad_q.dot(&head.w_q.t()) + grad_k.dot(&head.w_k.t()) + grad_v.dot(&head.w_v.t());

        (grad_input, vec![grad_w_q, grad_w_k, grad_w_v])
    }

    /// Softmax backward: write into provided output buffer
    fn softmax_backward_into<S1, S2>(
        softmax_output: &ArrayBase<S1, Ix2>,
        grad_output: &ArrayBase<S2, Ix2>,
        out: &mut Array2<f32>,
    )
    where
        S1: ndarray::Data<Elem = f32>,
        S2: ndarray::Data<Elem = f32>,
    {
        assert_eq!(out.dim(), softmax_output.dim());

        for ((mut out_row, softmax_row), grad_out_row) in out
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            // dot product: y ⊙ dL/dy
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();

            for ((g, &y_i), &dy_i) in out_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }
    }

    /// Softmax backward pass for attention (allocating)
    fn softmax_backward<S1, S2>(
        softmax_output: &ArrayBase<S1, Ix2>,
        grad_output: &ArrayBase<S2, Ix2>,
    ) -> Array2<f32>
    where
        S1: ndarray::Data<Elem = f32>,
        S2: ndarray::Data<Elem = f32>,
    {
        let mut out = Array2::zeros(softmax_output.dim());
        Self::softmax_backward_into(softmax_output, grad_output, &mut out);
        out
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "MultiHeadSelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());

        let (seq_len, emb_dim) = (input.shape()[0], input.shape()[1]);
        let head_dim = emb_dim / self.num_heads;

        // Compute adaptive window size if enabled
        let effective_window_size = if self.use_adaptive_window {
            let adaptive_size = self.compute_adaptive_window_size(seq_len);
            self.current_window_size = Some(adaptive_size);
            Some(adaptive_size)
        } else {
            self.window_size
        };

        // Compute head activation weights if MoH is enabled (soft weights for differentiable routing)
        let head_weights = if let Some(router) = &mut self.router {
            Some(router.route(input))
        } else {
            // AllHeads or StaticPruning: create weights based on strategy
            match &self.head_selection {
                HeadSelectionStrategy::AllHeads => None, // All heads active (weight = 1.0)
                HeadSelectionStrategy::StaticPruning { num_active_heads } => {
                    // Only first K heads active (weight = 1.0), rest inactive (weight = 0.0)
                    let mut weights = Array2::<f32>::zeros((seq_len, self.num_heads));
                    for token_idx in 0..seq_len {
                        for head_idx in 0..*num_active_heads {
                            weights[[token_idx, head_idx]] = 1.0;
                        }
                    }
                    Some(weights)
                }
                HeadSelectionStrategy::MixtureOfHeads { .. } | HeadSelectionStrategy::FullyAdaptiveMoH { .. } => {
                    // Should not reach here (router should be initialized)
                    None
                }
            }
        };

        // Split input into heads using zero-copy views: (seq_len, emb_dim) -> num_heads × (seq_len, head_dim)
        let head_inputs: Vec<_> = (0..self.num_heads)
            .map(|h| {
                let start_col = h * head_dim;
                let end_col = (h + 1) * head_dim;
                input.slice(ndarray::s![.., start_col..end_col])
            })
            .collect();

        // For GQA: compute K, V only for num_kv_heads, then share across query groups
        let queries_per_kv = self.num_heads / self.num_kv_heads;

        // Compute K, V for each KV head (only num_kv_heads, not num_heads)
        let mut keys: Vec<Array2<f32>> = Vec::with_capacity(self.num_kv_heads);
        let mut values: Vec<Array2<f32>> = Vec::with_capacity(self.num_kv_heads);
        for kv_idx in 0..self.num_kv_heads {
            // Use the first query head in each group to compute K, V
            let head_idx = kv_idx * queries_per_kv;
            let head_input_view = head_inputs[head_idx].view();

            let mut k = head_input_view.dot(&self.heads[head_idx].w_k);
            let v = head_input_view.dot(&self.heads[head_idx].w_v);

            // Apply RoPE to keys if enabled
            match &self.positional_encoding {
                PositionalEncodingVariant::Learned => {}
                PositionalEncodingVariant::RoPE(rope) => {
                    rope.apply_inplace(&mut k);
                }
                PositionalEncodingVariant::CoPE(_) => {}
            }

            keys.push(k);
            values.push(v);
        }

        // Stream per-head computation and write directly into output to avoid intermediate allocations
        let mut output = Array2::zeros((seq_len, emb_dim));
        let should_collect_entropy = self.use_adaptive_window
            && matches!(self.window_adaptation_strategy, WindowAdaptationStrategy::AttentionEntropy);
        let mut entropy_acc = 0.0f32;
        let mut entropy_heads = 0usize;

        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;

            // Skip completely inactive heads when weights are provided
            let is_head_active = if let Some(weights) = &head_weights {
                weights.column(h).iter().any(|&w| w > 0.0)
            } else {
                true
            };
            if !is_head_active {
                continue;
            }

            // Compute Q for this head (on the fly)
            let mut q = head_inputs[h].dot(&self.heads[h].w_q);
            match &self.positional_encoding {
                PositionalEncodingVariant::Learned => {}
                PositionalEncodingVariant::RoPE(rope) => {
                    rope.apply_inplace(&mut q);
                }
                PositionalEncodingVariant::CoPE(_) => {}
            }

            // Select corresponding KV head
            let kv_idx = h / queries_per_kv;
            let k = &keys[kv_idx];
            let v = &values[kv_idx];

            // Compute attention output for this head, optionally collecting entropy
            let head_out = match &self.positional_encoding {
                PositionalEncodingVariant::CoPE(cope_heads) => {
                    let position_logits = cope_heads[h].apply(&q, k);

                    if should_collect_entropy {
                        // Reproduce attention with position bias to access weights
                        let dk = (head_dim as f32).sqrt();
                        let k_t = k.t();
                        let mut scores = q.dot(&k_t) / dk;
                        scores += &position_logits;

                        // Apply causal + sliding window masking
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                if j > i {
                                    scores[[i, j]] = f32::NEG_INFINITY;
                                } else if let Some(window) = effective_window_size {
                                    if j < i.saturating_sub(window) {
                                        scores[[i, j]] = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }

                        // Softmax to get weights
                        SelfAttention::softmax_inplace(&mut scores);
                        // Collect entropy for this head
                        entropy_acc += self.compute_attention_entropy(&scores);
                        entropy_heads += 1;
                        // Use weights directly
                        scores.dot(v)
                    } else {
                        self.heads[h].attention_with_position_bias(&q, k, v, &position_logits, effective_window_size)
                    }
                }
                _ => {
                    if should_collect_entropy {
                        // Reproduce attention to access weights
                        let dk = (head_dim as f32).sqrt();
                        let k_t = k.t();
                        let mut scores = q.dot(&k_t) / dk;

                        // Apply causal + sliding window masking
                        for i in 0..seq_len {
                            for j in 0..seq_len {
                                if j > i {
                                    scores[[i, j]] = f32::NEG_INFINITY;
                                } else if let Some(window) = effective_window_size {
                                    if j < i.saturating_sub(window) {
                                        scores[[i, j]] = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }

                        // Softmax to get weights
                        SelfAttention::softmax_inplace(&mut scores);
                        // Collect entropy for this head
                        entropy_acc += self.compute_attention_entropy(&scores);
                        entropy_heads += 1;
                        // Use weights directly
                        scores.dot(v)
                    } else {
                        self.heads[h].attention(&q, k, v, effective_window_size)
                    }
                }
            };

            // Write into output with optional per-token weighting, avoiding temporary arrays
            if let Some(weights) = &head_weights {
                for token_idx in 0..seq_len {
                    let weight = weights[[token_idx, h]];
                    if weight > 0.0 {
                        let mut out_row = output.slice_mut(ndarray::s![token_idx, start_col..end_col]);
                        let head_row = head_out.row(token_idx);
                        ndarray::Zip::from(&mut out_row).and(head_row).for_each(|o, &hval| {
                            *o = hval * weight;
                        });
                    }
                }
            } else {
                output
                    .slice_mut(ndarray::s![.., start_col..end_col])
                    .assign(&head_out);
            }
        }

        // Update entropy EMA after processing heads
        if should_collect_entropy && entropy_heads > 0 {
            let measured_entropy = entropy_acc / entropy_heads as f32;
            let alpha = self.entropy_ema_alpha;
            self.last_attention_entropy = Some(match self.last_attention_entropy {
                Some(prev) => prev * (1.0 - alpha) + measured_entropy * alpha,
                None => measured_entropy,
            });
        }

        output += input; // residual connection in-place
        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        // For GQA: num_heads query projections + num_kv_heads key/value projections
        let head_dim = self.embedding_dim / self.num_heads;

        // All query heads have their own W_q
        let q_params = self.num_heads * head_dim * head_dim;

        // Only num_kv_heads have W_k and W_v
        let kv_params = self.num_kv_heads * 2 * head_dim * head_dim;

        // Add router parameters if MoH is enabled
        let router_params = self.router.as_ref().map_or(0, |r| r.parameters());

        q_params + kv_params + router_params
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self.cached_input.as_ref().unwrap();
        let (seq_len, emb_dim) = (input.shape()[0], input.shape()[1]);
        let head_dim = emb_dim / self.num_heads;

        // Stream per-head gradients using zero-copy views
        let mut all_param_grads = Vec::new();
        let mut input_grads = Array2::zeros((seq_len, emb_dim));
        // Reusable scratch buffer for softmax_backward: [seq_len, seq_len]
        let mut softmax_scratch = Array2::<f32>::zeros((seq_len, seq_len));

        for h in 0..self.num_heads {
            let start_col = h * head_dim;
            let end_col = (h + 1) * head_dim;

            let head_input_view: ndarray::ArrayView2<f32> = input.slice(ndarray::s![.., start_col..end_col]);
            let head_grad_view: ndarray::ArrayView2<f32> = output_grads.slice(ndarray::s![.., start_col..end_col]);

            let (head_input_grad, head_param_grads) =
                self.compute_head_gradients(h, head_input_view, &head_grad_view, &mut softmax_scratch);

            // Store parameter gradients
            all_param_grads.extend(head_param_grads);

            // Concatenate input gradients
            input_grads
                .slice_mut(ndarray::s![.., start_col..end_col])
                .assign(&head_input_grad);
        }

        // In-place add residual connection gradient (avoid allocation)
        input_grads += output_grads;

        (input_grads, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        let expected_grad_arrays = self.heads.len() * 3;
        if param_grads.len() != expected_grad_arrays {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "SelfAttention expected {} gradient arrays (3 per head), got {}",
                    expected_grad_arrays,
                    param_grads.len()
                ),
            });
        }

        // Apply gradients to each head's parameters
        // param_grads contains 3 gradients per head: [grad_w_q, grad_w_k, grad_w_v]
        let mut idx = 0;
        for head in &mut self.heads {
            head.optimizer_w_q
                .step(&mut head.w_q, &param_grads[idx], lr);
            head.optimizer_w_k
                .step(&mut head.w_k, &param_grads[idx + 1], lr);
            head.optimizer_w_v
                .step(&mut head.w_v, &param_grads[idx + 2], lr);
            idx += 3;
        }

        // CRITICAL FIX: Update router parameters using auxiliary loss gradients
        if let Some(router) = &mut self.router {
            router.backward(lr);
        }

        Ok(())
    }
}
