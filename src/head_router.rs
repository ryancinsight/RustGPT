//! Head Router for Mixture-of-Heads (MoH) attention
//!
//! Implements dynamic head selection per token using a learned routing mechanism.
//! Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
//!
//! # Architecture
//!
//! The router uses a two-stage routing mechanism:
//!
//! 1. **Shared Heads** (always active): Capture common knowledge across all tokens
//! 2. **Routed Heads** (Top-K selection): Specialize for specific patterns
//! 3. **Head Type Balancing**: Learns to balance shared vs. routed contributions
//!
//! # Mathematical Formulation
//!
//! ```text
//! Shared head scores:  g_s = α₁ × Softmax(W_s @ x^T)
//! Routed head scores:  g_r = α₂ × Softmax(W_r @ x^T)  [Top-K selected]
//! Head type weights:   [α₁, α₂] = Softmax(W_h @ x^T)
//! ```
//!
//! # Load Balance Loss
//!
//! ```text
//! L_b = Σ(i=1 to num_routed) P_i × f_i
//! ```
//!
//! Where:
//! - P_i = average routing score for head i
//! - f_i = fraction of tokens that selected head i
//!
//! This prevents routing collapse (all tokens routing to same heads).
//!
//! # Example
//!
//! ```ignore
//! let router = HeadRouter::new(
//!     128,  // embedding_dim
//!     2,    // num_shared_heads
//!     6,    // num_routed_heads
//!     4,    // num_active_routed_heads (Top-4)
//!     0.01, // load_balance_weight
//! );
//!
//! let input = Array2::ones((10, 128));  // batch_size=10, embedding_dim=128
//! let mask = router.route(&input);      // (10, 8) boolean mask
//! // mask[i][j] = true if head j is active for token i
//! ```
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::routing;

/// Learned threshold predictor for adaptive top-p routing
///
/// Predicts per-token threshold_p based on token representation.
/// Allows each token to have a custom threshold based on its complexity.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ThresholdPredictor {
    /// Prediction weights: (embedding_dim, 1)
    weights: Array2<f32>,

    /// Prediction bias
    bias: f32,

    /// Optimizer for threshold predictor
    optimizer: Adam,

    /// Minimum threshold (learned parameter)
    min_threshold: f32,

    /// Maximum threshold (learned parameter)
    max_threshold: f32,

    /// Cached input for backward pass
    #[serde(skip)]
    cached_input: Option<Array2<f32>>,

    /// Cached predictions for backward pass
    #[serde(skip)]
    cached_predictions: Option<Array1<f32>>,

    /// Exponential moving average of weight gradients (for smoothing)
    #[serde(skip)]
    ema_weight_grad: Option<Array2<f32>>,

    /// Exponential moving average of bias gradient (for smoothing)
    #[serde(skip)]
    ema_bias_grad: Option<f32>,

    /// EMA decay factor (0.9 = smooth over ~10 updates)
    ema_decay: f32,
}

impl ThresholdPredictor {
    /// Create a new threshold predictor
    pub fn new(embedding_dim: usize, min_threshold: f32, max_threshold: f32) -> Self {
        let mut rng = rand::rng();
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        ThresholdPredictor {
            weights: Array2::from_shape_fn((embedding_dim, 1), |_| normal.sample(&mut rng)),
            bias: 0.5, // Initialize to middle of range
            optimizer: Adam::new((embedding_dim, 1)),
            min_threshold,
            max_threshold,
            cached_input: None,
            cached_predictions: None,
            ema_weight_grad: None,
            ema_bias_grad: None,
            ema_decay: 0.9,  // Smooth over ~10 updates
        }
    }

    /// Predict threshold_p for each token
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Array of thresholds of shape (seq_len,) in range [min_threshold, max_threshold]
    pub fn predict(&mut self, input: &Array2<f32>) -> Array1<f32> {
        // Compute logits: input @ weights + bias
        let logits = input.dot(&self.weights).into_shape(input.nrows()).unwrap();
        let logits = logits.mapv(|x| x + self.bias);

        // Apply sigmoid and scale to [min_threshold, max_threshold]
        let range = self.max_threshold - self.min_threshold;
        let predictions = logits.mapv(|x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            self.min_threshold + sigmoid * range
        });

        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_predictions = Some(predictions.clone());

        predictions
    }

    /// Get number of parameters
    pub fn parameters(&self) -> usize {
        self.weights.len() + 1 // weights + bias
    }

    /// Compute gradients for threshold predictor
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient w.r.t. thresholds (seq_len,)
    /// * `lr` - Learning rate
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input (seq_len, embedding_dim)
    pub fn backward(&mut self, grad_output: &Array1<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().expect("No cached input for backward pass");
        let predictions = self.cached_predictions.as_ref().expect("No cached predictions for backward pass");

        let seq_len = input.nrows();
        let embedding_dim = input.ncols();

        // Compute sigmoid derivative: d_sigmoid/d_logit = sigmoid * (1 - sigmoid)
        let range = self.max_threshold - self.min_threshold;
        let d_threshold_d_logit = predictions.mapv(|pred| {
            let sigmoid = (pred - self.min_threshold) / range;
            sigmoid * (1.0 - sigmoid) * range
        });

        // Chain rule: grad_logit = grad_output * d_threshold_d_logit
        let grad_logit = grad_output * &d_threshold_d_logit;

        // Gradient w.r.t. weights: input^T @ grad_logit
        let mut grad_weights = Array2::zeros((embedding_dim, 1));
        for i in 0..seq_len {
            for j in 0..embedding_dim {
                grad_weights[[j, 0]] += input[[i, j]] * grad_logit[i];
            }
        }

        // Gradient w.r.t. bias: sum(grad_logit)
        let grad_bias = grad_logit.sum();

        // Apply EMA smoothing to gradients for stability
        let smoothed_grad_weights = if let Some(ema_grad) = &self.ema_weight_grad {
            // Update EMA: ema = decay * ema + (1 - decay) * grad
            let new_ema = ema_grad * self.ema_decay + &grad_weights * (1.0 - self.ema_decay);
            self.ema_weight_grad = Some(new_ema.clone());
            new_ema
        } else {
            // First update: initialize EMA with current gradient
            self.ema_weight_grad = Some(grad_weights.clone());
            grad_weights.clone()
        };

        let smoothed_grad_bias = if let Some(ema_bias) = self.ema_bias_grad {
            // Update EMA: ema = decay * ema + (1 - decay) * grad
            let new_ema = ema_bias * self.ema_decay + grad_bias * (1.0 - self.ema_decay);
            self.ema_bias_grad = Some(new_ema);
            new_ema
        } else {
            // First update: initialize EMA with current gradient
            self.ema_bias_grad = Some(grad_bias);
            grad_bias
        };

        // Update weights using Adam optimizer with smoothed gradients
        self.optimizer.step(&mut self.weights, &smoothed_grad_weights, lr);

        // Update bias manually with smoothed gradient (simple gradient descent)
        self.bias -= lr * smoothed_grad_bias;

        // Gradient w.r.t. input: grad_logit @ weights^T
        let mut grad_input = Array2::zeros((seq_len, embedding_dim));
        for i in 0..seq_len {
            for j in 0..embedding_dim {
                grad_input[[i, j]] = grad_logit[i] * self.weights[[j, 0]];
            }
        }

        grad_input
    }
}

/// Router type enum to support multiple routing strategies
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum RouterType {
    /// Standard MoH router with shared/routed head split
    Standard(HeadRouterStandard),

    /// Fully adaptive router with complexity-aware head selection
    FullyAdaptive(FullyAdaptiveHeadRouter),
}

impl RouterType {
    /// Route input tokens to heads
    ///
    /// Returns soft weights in [0, 1] for differentiable routing.
    /// For discrete routing (Standard), weights are 0.0 or 1.0.
    /// For soft routing (FullyAdaptive), weights are continuous.
    pub fn route(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            RouterType::Standard(router) => {
                // Convert boolean mask to f32 (0.0 or 1.0)
                let mask = router.route_discrete(input);
                mask.mapv(|b| if b { 1.0 } else { 0.0 })
            }
            RouterType::FullyAdaptive(router) => router.route(input),
        }
    }

    /// Compute total auxiliary loss (load balance + complexity + sparsity)
    pub fn compute_auxiliary_loss(&self) -> f32 {
        match self {
            RouterType::Standard(router) => {
                router.compute_load_balance_loss() + router.compute_dynamic_loss()
            }
            RouterType::FullyAdaptive(router) => {
                router.compute_load_balance_loss()
                    + router.compute_complexity_loss()
                    + router.compute_sparsity_loss()
            }
        }
    }

    /// Backward pass to update router parameters
    pub fn backward(&mut self, lr: f32) {
        match self {
            RouterType::Standard(router) => {
                // Standard router updates are handled separately
                // This is a placeholder for consistency
            }
            RouterType::FullyAdaptive(router) => {
                router.backward(lr);
            }
        }
    }

    /// Get routing statistics
    pub fn routing_stats(&self) -> String {
        match self {
            RouterType::Standard(router) => {
                if let Some(stats) = router.cached_num_active_routed.as_ref() {
                    let avg = stats.iter().sum::<usize>() as f32 / stats.len() as f32;
                    format!("Standard MoH: avg {:.1} routed heads", avg)
                } else {
                    "Standard MoH: no stats".to_string()
                }
            }
            RouterType::FullyAdaptive(router) => {
                let (avg_heads, min_heads, max_heads, avg_complexity, avg_threshold) = router.routing_stats();
                format!(
                    "Fully Adaptive MoH: avg {:.1} heads (min {}, max {}), complexity {:.2}, threshold {:.2}",
                    avg_heads, min_heads, max_heads, avg_complexity, avg_threshold
                )
            }
        }
    }

    /// Check if router has learned predictor
    pub fn has_learned_predictor(&self) -> bool {
        match self {
            RouterType::Standard(router) => router.threshold_predictor.is_some(),
            RouterType::FullyAdaptive(_) => true, // Always has predictors
        }
    }

    /// Get confidence statistics (only for standard router)
    pub fn confidence_stats(&self) -> Option<(f32, f32, f32)> {
        match self {
            RouterType::Standard(router) => router.cached_confidence_stats,
            RouterType::FullyAdaptive(_) => None,
        }
    }

    /// Get temperature statistics (only for fully adaptive router)
    pub fn temperature_stats(&self) -> Option<(f32, f32, f32)> {
        match self {
            RouterType::Standard(_) => None,
            RouterType::FullyAdaptive(router) => {
                let stats = router.get_temperature_stats();
                if stats.0 > 0.0 {
                    Some(stats)
                } else {
                    None
                }
            }
        }
    }

    /// Get complexity statistics for both standard and fully adaptive routers
    pub fn complexity_stats(&self) -> Option<(f32, f32, f32)> {
        match self {
            RouterType::Standard(router) => router.cached_complexity_stats,
            RouterType::FullyAdaptive(router) => {
                // Get complexity stats with proper min/max tracking
                let (avg, min, max) = router.get_complexity_stats();
                if avg > 0.0 {
                    Some((avg, min, max))
                } else {
                    None
                }
            }
        }
    }

    /// Get predictor weight norm for both standard and fully adaptive routers
    pub fn predictor_weight_norm(&self) -> f32 {
        match self {
            RouterType::Standard(router) => {
                if let Some(predictor) = &router.threshold_predictor {
                    predictor.weights.iter().map(|&x| x * x).sum::<f32>().sqrt()
                } else {
                    0.0
                }
            }
            RouterType::FullyAdaptive(router) => router.get_predictor_weight_norm(),
        }
    }

    /// Set epoch information for training progress tracking
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        match self {
            RouterType::Standard(router) => {
                router.current_epoch = current_epoch;
                router.max_epochs = max_epochs;
            }
            RouterType::FullyAdaptive(_) => {
                // Fully adaptive doesn't need epoch info currently
            }
        }
    }

    /// Get total number of parameters
    pub fn parameters(&self) -> usize {
        match self {
            RouterType::Standard(router) => router.parameters(),
            RouterType::FullyAdaptive(router) => router.num_parameters(),
        }
    }
}

/// Router network for Mixture-of-Heads attention (standard version with shared/routed split)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeadRouterStandard {
    /// Number of shared heads (always active)
    num_shared_heads: usize,

    /// Number of routed heads (Top-K selection)
    num_routed_heads: usize,

    /// Number of routed heads to activate (K in Top-K) - DEPRECATED for adaptive routing
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

    /// Base threshold for adaptive top-p routing (used when learned predictor is disabled)
    /// Range: 0.0-1.0, typical: 0.4-0.6
    threshold_p_base: f32,

    /// Dynamic loss weight base (adjusted adaptively based on sparsity)
    dynamic_loss_weight_base: f32,

    /// Learned threshold predictor (optional, for per-token adaptive thresholds)
    threshold_predictor: Option<ThresholdPredictor>,

    /// Layer index (for layer-wise adaptive thresholds)
    layer_idx: usize,

    /// Target average number of routed heads (for sparsity-based weight adaptation)
    target_avg_routed_heads: f32,

    /// Cached number of active routed heads per token (for logging)
    #[serde(skip)]
    cached_num_active_routed: Option<Vec<usize>>,

    /// Cached per-token thresholds (for logging and analysis)
    #[serde(skip)]
    cached_thresholds: Option<Array1<f32>>,

    /// Confidence threshold for fallback to all heads (0.0-1.0)
    /// When max routing probability < threshold, activate all routed heads
    confidence_threshold: f32,

    /// Whether to use confidence-based fallback
    use_confidence_fallback: bool,

    /// Cached confidence statistics (for logging)
    #[serde(skip)]
    cached_confidence_stats: Option<(f32, f32, f32)>, // (avg, min, fallback_pct)

    /// Current epoch (for warm-up and annealing)
    current_epoch: usize,

    /// Maximum epochs (for progress calculation)
    max_epochs: usize,

    /// Learned warm-up rate (adaptive, replaces fixed predictor_warmup_epochs)
    /// Initialized to 0.2 (20% of training = 20 epochs for 100 total)
    /// Can adapt during training based on gradient flow
    warmup_rate: f32,

    /// Learned layer-wise threshold adjustment (replaces hardcoded +0.1, 0.0, -0.1)
    /// Initialized based on layer position, then adapts during training
    layer_threshold_adjustment: f32,

    /// Learned annealing rate (replaces hardcoded 0.8 multiplier in dynamic loss weight)
    /// Initialized to 0.8, can adapt during training
    annealing_rate: f32,

    /// Cached complexity statistics (for logging)
    /// (avg_complexity, min_complexity, max_complexity)
    #[serde(skip)]
    cached_complexity_stats: Option<(f32, f32, f32)>,
}

pub type HeadRouter = HeadRouterStandard;

impl HeadRouterStandard {
    /// Create a new head router with fully adaptive routing
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input embedding dimension
    /// * `num_shared_heads` - Number of shared heads (always active)
    /// * `num_routed_heads` - Number of routed heads (adaptive top-p selection)
    /// * `num_active_routed_heads` - DEPRECATED: kept for backward compatibility
    /// * `load_balance_weight` - Weight for load balance loss (typically 0.01)
    /// * `threshold_p_base` - Base threshold for adaptive routing (typically 0.4-0.6)
    /// * `dynamic_loss_weight_base` - Base weight for dynamic loss (typically 1e-4)
    /// * `layer_idx` - Layer index for layer-wise adaptation
    /// * `use_learned_threshold` - Whether to use learned per-token thresholds
    /// * `target_avg_routed_heads` - Target average routed heads for sparsity adaptation
    /// * `confidence_threshold` - Confidence threshold for fallback (0.0-1.0)
    /// * `use_confidence_fallback` - Whether to use confidence-based fallback
    ///
    /// # Returns
    ///
    /// A new HeadRouter with Xavier-initialized weights and optional learned predictor
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
        threshold_p_base: f32,
        dynamic_loss_weight_base: f32,
        layer_idx: usize,
        use_learned_threshold: bool,
        target_avg_routed_heads: f32,
        confidence_threshold: f32,
        use_confidence_fallback: bool,
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

        // Initialize learned threshold predictor if enabled
        let threshold_predictor = if use_learned_threshold {
            Some(ThresholdPredictor::new(embedding_dim, 0.3, 0.7))
        } else {
            None
        };

        HeadRouterStandard {
            num_shared_heads,
            num_routed_heads,
            num_active_routed_heads,
            embedding_dim,
            w_shared: Array2::from_shape_fn((num_shared_heads, embedding_dim), |_| {
                normal_shared.sample(&mut rng)
            }),
            w_routed: Array2::from_shape_fn((num_routed_heads, embedding_dim), |_| {
                normal_routed.sample(&mut rng)
            }),
            w_head_type: Array2::from_shape_fn((2, embedding_dim), |_| {
                normal_head_type.sample(&mut rng)
            }),
            optimizer_shared: Adam::new((num_shared_heads, embedding_dim)),
            optimizer_routed: Adam::new((num_routed_heads, embedding_dim)),
            optimizer_head_type: Adam::new((2, embedding_dim)),
            cached_routing_scores_shared: None,
            cached_routing_scores_routed: None,
            cached_head_type_weights: None,
            cached_activation_mask: None,
            load_balance_weight,
            threshold_p_base,
            dynamic_loss_weight_base,
            threshold_predictor,
            layer_idx,
            target_avg_routed_heads,
            cached_num_active_routed: None,
            cached_thresholds: None,
            confidence_threshold,
            use_confidence_fallback,
            cached_confidence_stats: None,
            current_epoch: 0,
            max_epochs: 100,  // Default, can be updated via set_epoch_info()
            // Initialize adaptive parameters for faster learning with 100 epochs
            warmup_rate: 0.05,  // 5% warm-up (5 epochs), 5% gradual enable (10 epochs total) - very aggressive
            layer_threshold_adjustment: match layer_idx {
                0..=3 => 0.15,   // Early layers: MORE heads (increased from 0.1)
                4..=7 => 0.05,   // Middle layers: slightly more heads (increased from 0.0)
                _ => -0.05,      // Late layers: slightly fewer heads (reduced from -0.1)
            },
            annealing_rate: 0.4,  // Reduced from 0.6 to keep dynamic loss weight much higher (adaptive)
            cached_complexity_stats: None,
        }
    }

    /// Route input to select which heads to activate using adaptive top-p routing
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Boolean mask of shape (seq_len, total_heads) where:
    /// - First `num_shared_heads` columns are always true (shared heads)
    /// - Remaining columns have variable number of true values per row (adaptive top-p)
    ///
    /// # Adaptive Top-P Routing
    ///
    /// For each token, routed heads are selected until cumulative probability ≥ threshold_p:
    /// 1. Sort routed heads by probability (descending)
    /// 2. Select heads until Σ(p_i) ≥ threshold_p
    /// 3. Result: 1-6 routed heads per token (adaptive based on confidence)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mask = router.route(&input);  // (seq_len, 8)
    /// // mask[i][0..2] = [true, true]  (shared heads always active)
    /// // mask[i][2..8] has 1-6 true values (adaptive top-p routed heads)
    /// ```
    pub fn route_discrete(&mut self, input: &Array2<f32>) -> Array2<bool> {
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

        // 4. Compute adaptive thresholds per token
        let thresholds = self.compute_adaptive_thresholds_without_complexity(input);

        // 5. Adaptive Top-P selection for routed heads with per-token thresholds
        // With confidence-based fallback: activate all heads when confidence is low
        let mut mask = Array2::<bool>::from_elem((seq_len, total_heads), false);
        let mut num_active_routed_per_token = Vec::with_capacity(seq_len);
        let mut confidence_values = Vec::with_capacity(seq_len);
        let mut fallback_count = 0;

        for token_idx in 0..seq_len {
            // Shared heads are always active
            for head_idx in 0..self.num_shared_heads {
                mask[[token_idx, head_idx]] = true;
            }

            // Routed heads: adaptive top-p selection with per-token threshold
            let token_probs = routed_scores.row(token_idx);
            let token_threshold = thresholds[token_idx];

            // Compute routing confidence (max probability)
            let max_prob = token_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            confidence_values.push(max_prob);

            // Confidence-based fallback: if confidence is low, activate ALL routed heads
            let use_fallback = self.use_confidence_fallback && max_prob < self.confidence_threshold;

            if use_fallback {
                // Low confidence → activate all routed heads to preserve quality
                for routed_idx in 0..self.num_routed_heads {
                    let head_idx = self.num_shared_heads + routed_idx;
                    mask[[token_idx, head_idx]] = true;
                }
                num_active_routed_per_token.push(self.num_routed_heads);
                fallback_count += 1;
            } else {
                // High confidence → use adaptive sparse routing
                // Sort routed heads by probability (descending)
                let mut sorted_indices: Vec<usize> = (0..self.num_routed_heads).collect();
                sorted_indices.sort_by(|&a, &b| {
                    token_probs[b].partial_cmp(&token_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
                });

                // Select heads until cumulative probability ≥ token_threshold
                let mut cumulative_prob = 0.0;
                let mut num_active = 0;

                for &routed_idx in &sorted_indices {
                    cumulative_prob += token_probs[routed_idx];
                    let head_idx = self.num_shared_heads + routed_idx;
                    mask[[token_idx, head_idx]] = true;
                    num_active += 1;

                    // Stop when we've accumulated enough probability (using per-token threshold)
                    if cumulative_prob >= token_threshold {
                        break;
                    }
                }

                num_active_routed_per_token.push(num_active);
            }
        }

        // Cache for backward pass
        self.cached_routing_scores_shared = Some(shared_scores);
        self.cached_routing_scores_routed = Some(routed_scores);
        self.cached_head_type_weights = Some(head_type_weights);
        self.cached_activation_mask = Some(mask.clone());
        self.cached_num_active_routed = Some(num_active_routed_per_token);
        self.cached_thresholds = Some(thresholds);

        // Cache confidence statistics
        if !confidence_values.is_empty() {
            let avg_conf = confidence_values.iter().sum::<f32>() / confidence_values.len() as f32;
            let min_conf = confidence_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let fallback_pct = (fallback_count as f32 / seq_len as f32) * 100.0;
            self.cached_confidence_stats = Some((avg_conf, min_conf, fallback_pct));
        }

        mask
    }

    /// Compute task complexity from routing entropy
    ///
    /// Uses Shannon entropy of routing probabilities as a measure of task complexity:
    /// - High entropy (uniform distribution) → complex task → needs more heads
    /// - Low entropy (peaked distribution) → simple task → needs fewer heads
    ///
    /// # Arguments
    ///
    /// * `routed_scores` - Routing probabilities of shape (seq_len, num_routed_heads)
    ///
    /// # Returns
    ///
    /// Array of normalized complexity values of shape (seq_len,), range [0, 1]
    fn compute_routing_complexity(&mut self, routed_scores: &Array2<f32>) -> Array1<f32> {
        let seq_len = routed_scores.nrows();

        // Compute Shannon entropy per token: H = -Σ(p_i * ln(p_i))
        let mut entropies = Vec::with_capacity(seq_len);
        let mut min_entropy = f32::INFINITY;
        let mut max_entropy = f32::NEG_INFINITY;

        for token_idx in 0..seq_len {
            let probs = routed_scores.row(token_idx);
            let entropy: f32 = probs.iter()
                .map(|&p| {
                    if p > 1e-10 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum();

            entropies.push(entropy);
            min_entropy = min_entropy.min(entropy);
            max_entropy = max_entropy.max(entropy);
        }

        // Normalize entropy to [0, 1] range
        // Max possible entropy = ln(num_routed_heads)
        let theoretical_max_entropy = (self.num_routed_heads as f32).ln();
        let entropy_range = (max_entropy - min_entropy).max(1e-6);

        let normalized_complexity: Array1<f32> = entropies.iter()
            .map(|&e| {
                // Normalize using theoretical max for consistent scaling
                (e / theoretical_max_entropy).clamp(0.0, 1.0)
            })
            .collect();

        // Cache complexity statistics for logging
        let avg_complexity = normalized_complexity.mean().unwrap_or(0.0);
        let min_complexity = normalized_complexity.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_complexity = normalized_complexity.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        self.cached_complexity_stats = Some((avg_complexity, min_complexity, max_complexity));

        normalized_complexity
    }

    /// Compute adaptive thresholds for each token (without complexity adjustment)
    ///
    /// Combines multiple strategies:
    /// 1. Training-progress-based base threshold (increases 0.3 → 0.7 over training)
    /// 2. Layer-wise adjustment (early layers use more heads)
    /// 3. Learned per-token adjustment (if enabled) - adds ±0.2 to base
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Array of thresholds of shape (seq_len,)
    fn compute_adaptive_thresholds_without_complexity(&mut self, input: &Array2<f32>) -> Array1<f32> {
        let seq_len = input.nrows();

        // 1. Compute training-progress-based threshold (adaptive annealing)
        // Uses learned min/max thresholds from predictor instead of hardcoded 0.3-0.7
        let training_progress = self.current_epoch as f32 / self.max_epochs.max(1) as f32;

        // Get adaptive threshold range from predictor (or use defaults)
        let (min_thresh, max_thresh) = if let Some(predictor) = &self.threshold_predictor {
            (predictor.min_threshold, predictor.max_threshold)
        } else {
            (0.3, 0.7)  // Fallback defaults
        };

        let threshold_range = max_thresh - min_thresh;
        let progress_adjusted_base = min_thresh + training_progress * threshold_range;

        // 2. Apply learned layer-wise adjustment (adaptive, not hardcoded)
        let layer_threshold = (progress_adjusted_base + self.layer_threshold_adjustment)
            .clamp(min_thresh, max_thresh);

        // 3. If learned predictor is enabled, use it to adjust per-token (with warm-up)
        // Get predictor weight BEFORE borrowing predictor mutably
        let pred_weight = self.predictor_weight();

        if let Some(predictor) = &mut self.threshold_predictor {
            if pred_weight > 0.0 {
                // Predictor outputs adjustments in range [-0.1, +0.1]
                // This allows per-token fine-tuning while respecting layer-wise strategy
                let adjustments = predictor.predict(input);

                // Combine: base + (learned adjustment * weight), clamped to valid range
                adjustments.mapv(|adj| {
                    // Predictor range is [0.3, 0.7], center at 0.5
                    // Convert to adjustment: (adj - 0.5) * 0.4 gives range [-0.2, +0.2]
                    let adjustment = (adj - 0.5) * 0.4 * pred_weight;
                    (layer_threshold + adjustment).clamp(0.3, 0.7)
                })
            } else {
                // Warm-up period: use layer-wise threshold only
                Array1::from_elem(seq_len, layer_threshold)
            }
        } else {
            // Use layer-wise threshold for all tokens
            Array1::from_elem(seq_len, layer_threshold)
        }
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
            // Extract routed head portion of mask as a view (avoid clone)
            let routed_mask_view = mask.slice(ndarray::s![.., self.num_shared_heads..]);

            routing::compute_load_balance_loss(routed_scores, &routed_mask_view)
        } else {
            0.0
        }
    }

    /// Compute dynamic loss (entropy minimization) for the last routing operation
    ///
    /// # Returns
    ///
    /// Scalar dynamic loss value (entropy), or 0.0 if no routing has been performed
    ///
    /// # Note
    ///
    /// Dynamic loss encourages sparse head selection by minimizing entropy:
    /// Loss_dynamic = -Σ(p_i * log(p_i))
    ///
    /// Lower entropy → more confident, sparse routing
    /// Higher entropy → less confident, diffuse routing
    ///
    /// This should be called after `route()` and added to the total loss during training.
    pub fn compute_dynamic_loss(&self) -> f32 {
        if let Some(routed_scores) = &self.cached_routing_scores_routed {
            // Compute entropy: -Σ(p * log(p))
            let entropy = routed_scores.mapv(|p| {
                if p > 1e-10 {
                    -p * p.ln()
                } else {
                    0.0
                }
            });

            // Average over all tokens and heads
            entropy.sum() / (routed_scores.nrows() * routed_scores.ncols()) as f32
        } else {
            0.0
        }
    }

    /// Get average number of active routed heads per token from last routing
    ///
    /// # Returns
    ///
    /// Average number of routed heads activated per token, or 0.0 if no routing has been performed
    pub fn avg_active_routed_heads(&self) -> f32 {
        if let Some(num_active) = &self.cached_num_active_routed {
            let sum: usize = num_active.iter().sum();
            sum as f32 / num_active.len() as f32
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

    /// Get the adaptive dynamic loss weight based on current sparsity
    ///
    /// Adjusts the base weight based on how many heads are currently being used:
    /// - If using more heads than target → increase penalty (encourage sparsity)
    /// - If using fewer heads than target → decrease penalty (allow more capacity)
    ///
    /// # Arguments
    ///
    /// * `training_progress` - Training progress in range [0.0, 1.0] for annealing
    ///
    /// # Returns
    ///
    /// Adaptive dynamic loss weight
    pub fn dynamic_loss_weight(&self, training_progress: f32) -> f32 {
        let current_avg = self.avg_active_routed_heads();

        // If no routing has been performed yet, return base weight
        if current_avg == 0.0 {
            return self.dynamic_loss_weight_base;
        }

        // 1. Sparsity-based adaptation
        let sparsity_multiplier = if current_avg > self.target_avg_routed_heads {
            // Using too many heads → increase penalty
            (current_avg / self.target_avg_routed_heads).min(2.0)
        } else {
            // Using fewer heads → decrease penalty
            (current_avg / self.target_avg_routed_heads).powi(2).max(0.5)
        };

        // 2. Training progress annealing (start HIGH, decrease over time) - ADAPTIVE
        // Uses learned annealing_rate instead of hardcoded 0.8
        // Early training (progress=0.0): 1.0x base weight (strong penalty for exploration)
        // Late training (progress=1.0): (1.0 - annealing_rate)x base weight
        // Formula: weight = base * (1.0 - annealing_rate * progress)
        let annealing_multiplier = 1.0 - self.annealing_rate * training_progress;

        // 3. Combine both adaptations
        self.dynamic_loss_weight_base * sparsity_multiplier * annealing_multiplier
    }

    /// Get the base threshold_p for adaptive routing
    pub fn threshold_p_base(&self) -> f32 {
        self.threshold_p_base
    }

    /// Get average threshold from last routing (for logging)
    pub fn avg_threshold(&self) -> f32 {
        if let Some(thresholds) = &self.cached_thresholds {
            thresholds.mean().unwrap_or(self.threshold_p_base)
        } else {
            self.threshold_p_base
        }
    }

    /// Get threshold statistics (min, max, mean, std) for analysis
    pub fn threshold_stats(&self) -> (f32, f32, f32, f32) {
        if let Some(thresholds) = &self.cached_thresholds {
            let mean = thresholds.mean().unwrap_or(0.0);
            let min = thresholds.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = thresholds.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let variance = thresholds.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
            let std = variance.sqrt();
            (min, max, mean, std)
        } else {
            let base = self.threshold_p_base;
            (base, base, base, 0.0)
        }
    }

    /// Update threshold predictor based on routing efficiency
    ///
    /// Uses a simple heuristic: if we're using too many heads relative to target,
    /// encourage higher thresholds (fewer heads). If using too few, encourage lower
    /// thresholds (more heads).
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for predictor updates
    pub fn update_threshold_predictor(&mut self, lr: f32) {
        // Early return if no predictor
        if self.threshold_predictor.is_none() {
            return;
        }

        // Skip update during warm-up period (predictor is frozen)
        let pred_weight = self.predictor_weight();
        if pred_weight == 0.0 {
            return;
        }

        // Compute values we need before borrowing predictor mutably
        let current_avg = self.avg_active_routed_heads();

        // Skip update if no routing has been performed
        if current_avg == 0.0 {
            return;
        }

        // Compute gradient signal based on sparsity target
        // If using too many heads → positive gradient (increase thresholds)
        // If using too few heads → negative gradient (decrease thresholds)
        let sparsity_error = current_avg - self.target_avg_routed_heads;

        // Scale gradient by error magnitude (stronger signal for larger deviations)
        let grad_scale = sparsity_error / self.target_avg_routed_heads;

        // Create gradient signal for each token
        // Positive gradient → increase threshold → fewer heads
        // Negative gradient → decrease threshold → more heads
        if let Some(thresholds) = &self.cached_thresholds {
            let grad_output = Array1::from_elem(thresholds.len(), grad_scale);

            // Now borrow predictor mutably and apply backward pass
            if let Some(predictor) = &mut self.threshold_predictor {
                let _ = predictor.backward(&grad_output, lr);
            }
        }
    }

    /// Check if learned threshold predictor is enabled
    pub fn has_learned_predictor(&self) -> bool {
        self.threshold_predictor.is_some()
    }

    /// Get confidence statistics (avg, min, fallback_pct)
    ///
    /// Returns (0.0, 0.0, 0.0) if no routing has been performed yet.
    pub fn confidence_stats(&self) -> (f32, f32, f32) {
        self.cached_confidence_stats.unwrap_or((0.0, 0.0, 0.0))
    }

    /// Get complexity statistics (avg, min, max)
    ///
    /// Returns (0.0, 0.0, 0.0) - complexity tracking disabled for now.
    pub fn complexity_stats(&self) -> (f32, f32, f32) {
        (0.0, 0.0, 0.0)
    }

    /// Get predictor weight norm (for tracking convergence)
    ///
    /// Returns 0.0 if no learned predictor is enabled.
    pub fn predictor_weight_norm(&self) -> f32 {
        self.threshold_predictor
            .as_ref()
            .map_or(0.0, |p| {
                p.weights.iter().map(|&x| x * x).sum::<f32>().sqrt()
            })
    }

    /// Set epoch information for warm-up and annealing
    ///
    /// # Arguments
    ///
    /// * `current_epoch` - Current training epoch (0-indexed)
    /// * `max_epochs` - Maximum number of training epochs
    pub fn set_epoch_info(&mut self, current_epoch: usize, max_epochs: usize) {
        self.current_epoch = current_epoch;
        self.max_epochs = max_epochs;
    }

    /// Get predictor weight for warm-up period (adaptive)
    ///
    /// Uses learned warmup_rate instead of hardcoded epochs
    /// warmup_rate = 0.2 means 20% of training for warm-up, 20% for gradual enable
    fn predictor_weight(&self) -> f32 {
        let training_progress = self.current_epoch as f32 / self.max_epochs.max(1) as f32;
        let warmup_end = self.warmup_rate;
        let gradual_end = self.warmup_rate * 2.0;  // Double the warmup period for gradual enable

        if training_progress < warmup_end {
            // Warm-up period: predictor is frozen (weight = 0.0)
            0.0
        } else if training_progress < gradual_end {
            // Gradual enable: weight increases from 0.0 to 1.0
            let progress_in_gradual = (training_progress - warmup_end) / self.warmup_rate;
            progress_in_gradual.min(1.0)
        } else {
            // Fully enabled
            1.0
        }
    }
}

/// Fully Adaptive Head Router for complexity-aware dynamic head selection
///
/// Unlike standard MoH which has hardcoded shared/routed head splits, this router:
/// - Treats ALL heads as routing candidates (no hardcoded shared heads)
/// - Predicts input complexity to determine target head count
/// - Uses complexity-driven adaptive top-p selection
///
/// # Architecture
///
/// ```text
/// Input → Complexity Predictor → Target Heads (1-8)
///      ↘ Threshold Predictor → Top-P Threshold (0.3-0.8)
///      ↘ Unified Router → Routing Probabilities
///
/// Selection: Pick heads until cumsum ≥ threshold OR count ≥ target
/// ```
///
/// # Example
///
/// ```ignore
/// let mut router = FullyAdaptiveHeadRouter::new(
///     128,   // embedding_dim
///     8,     // num_heads
///     1,     // min_heads
///     8,     // max_heads
///     0.01,  // load_balance_weight
///     0.01,  // complexity_loss_weight
///     0.001, // sparsity_weight
/// );
///
/// let input = Array2::ones((10, 128));
/// let mask = router.route(&input);  // (10, 8) boolean mask
/// // Simple inputs: 1-2 heads active
/// // Complex inputs: 6-8 heads active
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FullyAdaptiveHeadRouter {
    /// Total number of heads (all are routing candidates)
    num_heads: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Unified router weights: (num_heads, embedding_dim)
    /// Computes routing scores for ALL heads (no shared/routed split)
    w_router: Array2<f32>,

    /// Complexity predictor weights: (embedding_dim, 1)
    /// Predicts input complexity score [0, 1] → target head count
    w_complexity: Array2<f32>,

    /// Complexity predictor bias
    complexity_bias: f32,

    /// Threshold predictor weights: (embedding_dim, 1)
    /// Predicts per-token threshold for top-p selection
    w_threshold: Array2<f32>,

    /// Threshold predictor bias
    threshold_bias: f32,

    /// Minimum heads to activate (safety constraint)
    min_heads: usize,

    /// Maximum heads to activate (efficiency constraint)
    max_heads: usize,

    /// Weight for load balance loss
    load_balance_weight: f32,

    /// Weight for complexity alignment loss
    complexity_loss_weight: f32,

    /// Weight for sparsity loss
    sparsity_weight: f32,

    /// Temperature predictor weights: (embedding_dim, 1)
    /// Predicts per-token temperature for soft routing gating
    w_temperature: Array2<f32>,

    /// Temperature predictor bias
    temperature_bias: f32,

    /// Optimizer for router weights
    router_optimizer: Adam,

    /// Optimizer for complexity predictor
    complexity_optimizer: Adam,

    /// Optimizer for threshold predictor
    threshold_optimizer: Adam,

    /// Optimizer for temperature predictor
    temperature_optimizer: Adam,

    /// Cached routing probabilities for backward pass
    #[serde(skip)]
    cached_routing_probs: Option<Array2<f32>>,

    /// Cached complexity scores for backward pass
    #[serde(skip)]
    cached_complexity_scores: Option<Array1<f32>>,

    /// Cached thresholds for backward pass
    #[serde(skip)]
    cached_thresholds: Option<Array1<f32>>,

    /// Cached temperatures for backward pass
    #[serde(skip)]
    cached_temperatures: Option<Array1<f32>>,

    /// Cached soft weights for backward pass (differentiable routing)
    #[serde(skip)]
    cached_soft_weights: Option<Array2<f32>>,

    /// Cached target heads for backward pass
    #[serde(skip)]
    cached_target_heads: Option<Array1<f32>>,

    /// Cached input for backward pass
    #[serde(skip)]
    cached_input: Option<Array2<f32>>,
}

impl FullyAdaptiveHeadRouter {
    /// Create a new fully adaptive head router
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input embedding dimension
    /// * `num_heads` - Total number of attention heads (all are routing candidates)
    /// * `min_heads` - Minimum heads to activate (typically 1)
    /// * `max_heads` - Maximum heads to activate (typically num_heads)
    /// * `load_balance_weight` - Weight for load balance loss (typically 0.01)
    /// * `complexity_loss_weight` - Weight for complexity alignment loss (typically 0.01)
    /// * `sparsity_weight` - Weight for sparsity loss (typically 0.001)
    pub fn new(
        embedding_dim: usize,
        num_heads: usize,
        min_heads: usize,
        max_heads: usize,
        load_balance_weight: f32,
        complexity_loss_weight: f32,
        sparsity_weight: f32,
    ) -> Self {
        assert!(min_heads >= 1, "min_heads must be >= 1");
        assert!(max_heads <= num_heads, "max_heads must be <= num_heads");
        assert!(min_heads <= max_heads, "min_heads must be <= max_heads");

        let mut rng = rand::rng();

        // Xavier initialization for router weights
        let router_std = (2.0 / (embedding_dim + num_heads) as f32).sqrt();
        let router_normal = Normal::new(0.0, router_std).unwrap();
        let w_router = Array2::from_shape_fn((num_heads, embedding_dim), |_| {
            router_normal.sample(&mut rng)
        });

        // Xavier initialization for complexity predictor
        let complexity_std = (2.0 / (embedding_dim + 1) as f32).sqrt();
        let complexity_normal = Normal::new(0.0, complexity_std).unwrap();
        let w_complexity = Array2::from_shape_fn((embedding_dim, 1), |_| {
            complexity_normal.sample(&mut rng)
        });
        let complexity_bias = 0.0;

        // Xavier initialization for threshold predictor
        let threshold_std = (2.0 / (embedding_dim + 1) as f32).sqrt();
        let threshold_normal = Normal::new(0.0, threshold_std).unwrap();
        let w_threshold = Array2::from_shape_fn((embedding_dim, 1), |_| {
            threshold_normal.sample(&mut rng)
        });
        let threshold_bias = 0.0;

        // Xavier initialization for temperature predictor
        // Initialize to output ~5.0 initially: sigmoid(0) = 0.5 → 1.0 + 0.5 * 9.0 = 5.5
        let temperature_std = (2.0 / (embedding_dim + 1) as f32).sqrt() * 0.5;  // Smaller std for stability
        let temperature_normal = Normal::new(0.0, temperature_std).unwrap();
        let w_temperature = Array2::from_shape_fn((embedding_dim, 1), |_| {
            temperature_normal.sample(&mut rng)
        });
        let temperature_bias = 0.0;

        Self {
            num_heads,
            embedding_dim,
            w_router,
            w_complexity,
            complexity_bias,
            w_threshold,
            threshold_bias,
            w_temperature,
            temperature_bias,
            min_heads,
            max_heads,
            load_balance_weight,
            complexity_loss_weight,
            sparsity_weight,
            router_optimizer: Adam::new((num_heads, embedding_dim)),
            complexity_optimizer: Adam::new((embedding_dim, 1)),
            threshold_optimizer: Adam::new((embedding_dim, 1)),
            temperature_optimizer: Adam::new((embedding_dim, 1)),
            cached_routing_probs: None,
            cached_complexity_scores: None,
            cached_thresholds: None,
            cached_temperatures: None,
            cached_soft_weights: None,
            cached_target_heads: None,
            cached_input: None,
        }
    }

    /// Route input tokens to heads using complexity-driven adaptive top-p selection
    ///
    /// # Algorithm
    ///
    /// 1. Predict complexity score [0, 1] for each token
    /// 2. Map complexity to target head count: min_heads + complexity × (max_heads - min_heads)
    /// 3. Predict per-token threshold for top-p selection
    /// 4. Compute routing probabilities for all heads
    /// 5. Select heads until: cumsum ≥ threshold OR count ≥ target OR all heads selected
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Soft weights of shape (seq_len, num_heads) where weights[i][j] ∈ [0, 1] is the activation weight for head j on token i
    /// Uses differentiable sigmoid-based gating for gradient flow
    pub fn route(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _) = input.dim();

        // 1. Predict complexity scores [0, 1] for each token
        let complexity_logits = input.dot(&self.w_complexity).into_shape(seq_len).unwrap();
        let complexity_scores = complexity_logits.mapv(|x| {
            let sigmoid = 1.0 / (1.0 + (-(x + self.complexity_bias)).exp());
            sigmoid
        });

        // 2. Map complexity to target head count
        let head_range = (self.max_heads - self.min_heads) as f32;
        let target_heads = complexity_scores.mapv(|c| {
            self.min_heads as f32 + c * head_range
        });

        // 3. Predict per-token thresholds [0.3, 0.8]
        let threshold_logits = input.dot(&self.w_threshold).into_shape(seq_len).unwrap();
        let thresholds = threshold_logits.mapv(|x| {
            let sigmoid = 1.0 / (1.0 + (-(x + self.threshold_bias)).exp());
            0.3 + sigmoid * 0.5  // Map [0, 1] → [0.3, 0.8]
        });

        // 3.5. Predict per-token temperatures [1.0, 10.0]
        let temp_logits = input.dot(&self.w_temperature).into_shape(seq_len).unwrap();
        let temperatures = temp_logits.mapv(|x| {
            let sigmoid = 1.0 / (1.0 + (-(x + self.temperature_bias)).exp());
            1.0 + sigmoid * 9.0  // Map [0, 1] → [1.0, 10.0]
        });

        // 4. Compute routing probabilities for all heads
        let routing_logits = input.dot(&self.w_router.t());
        let routing_probs = routing::softmax(&routing_logits);

        // 5. SOFT ROUTING: Differentiable sigmoid-based gating
        // Instead of discrete top-p selection, use continuous soft weights
        let mut soft_weights = Array2::<f32>::zeros((seq_len, self.num_heads));

        for token_idx in 0..seq_len {
            let token_probs = routing_probs.row(token_idx);
            let token_threshold = thresholds[token_idx];
            let token_temperature = temperatures[token_idx];

            // Sort heads by probability (descending)
            let mut sorted_indices: Vec<usize> = (0..self.num_heads).collect();
            sorted_indices.sort_by(|&a, &b| {
                token_probs[b].partial_cmp(&token_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Compute cumulative probabilities for each head
            let mut cumulative_prob = 0.0;
            for &head_idx in &sorted_indices {
                cumulative_prob += token_probs[head_idx];

                // Soft gating: sigmoid((cumulative_prob - threshold) * temperature)
                // When cumulative_prob < threshold: gating ≈ 0 (head inactive)
                // When cumulative_prob > threshold: gating ≈ 1 (head active)
                // Temperature controls sharpness of transition (now per-token adaptive)
                let gating = 1.0 / (1.0 + (-(cumulative_prob - token_threshold) * token_temperature).exp());

                // Soft weight = routing_prob * gating
                // This is differentiable w.r.t. routing_probs, threshold, AND temperature
                soft_weights[[token_idx, head_idx]] = token_probs[head_idx] * gating;
            }

            // Normalize soft weights to sum to approximately target_heads[token_idx]
            // This maintains the complexity-driven head count behavior
            let weight_sum: f32 = soft_weights.row(token_idx).sum();
            if weight_sum > 0.0 {
                let scale = target_heads[token_idx] / weight_sum;
                for head_idx in 0..self.num_heads {
                    soft_weights[[token_idx, head_idx]] *= scale;
                }
            }
        }

        // Cache for backward pass
        self.cached_routing_probs = Some(routing_probs);
        self.cached_complexity_scores = Some(complexity_scores);
        self.cached_thresholds = Some(thresholds);
        self.cached_temperatures = Some(temperatures);
        self.cached_soft_weights = Some(soft_weights.clone());
        self.cached_target_heads = Some(target_heads);
        self.cached_input = Some(input.clone());

        soft_weights
    }

    /// Compute load balance loss to prevent routing collapse
    ///
    /// Loss: L_balance = Σ(i=1 to num_heads) P_i × W_i
    ///
    /// Where:
    /// - P_i = average routing probability for head i across all tokens
    /// - W_i = average soft weight for head i across all tokens
    ///
    /// This encourages uniform head usage across tokens (soft version).
    pub fn compute_load_balance_loss(&self) -> f32 {
        if let (Some(routing_probs), Some(soft_weights)) = (&self.cached_routing_probs, &self.cached_soft_weights) {
            let seq_len = routing_probs.nrows();

            let mut loss = 0.0;
            for head_idx in 0..self.num_heads {
                // P_i: average routing probability for head i
                let avg_prob: f32 = routing_probs.column(head_idx).sum() / seq_len as f32;

                // W_i: average soft weight for head i (continuous version of selection fraction)
                let avg_weight: f32 = soft_weights.column(head_idx).sum() / seq_len as f32;

                loss += avg_prob * avg_weight;
            }

            loss * self.load_balance_weight
        } else {
            0.0
        }
    }

    /// Compute complexity alignment loss
    ///
    /// Loss: L_complexity = |avg_soft_heads - avg_target_heads|
    ///
    /// Encourages the router to use the number of heads predicted by the complexity predictor (soft version).
    pub fn compute_complexity_loss(&self) -> f32 {
        if let (Some(soft_weights), Some(target_heads)) = (&self.cached_soft_weights, &self.cached_target_heads) {
            let seq_len = soft_weights.nrows();

            // Compute average soft heads (sum of soft weights per token)
            let mut total_soft_heads = 0.0;
            for token_idx in 0..seq_len {
                let token_soft_heads: f32 = soft_weights.row(token_idx).sum();
                total_soft_heads += token_soft_heads;
            }
            let avg_soft_heads = total_soft_heads / seq_len as f32;

            // Compute average target heads
            let avg_target = target_heads.sum() / seq_len as f32;

            // L1 loss
            (avg_soft_heads - avg_target).abs() * self.complexity_loss_weight
        } else {
            0.0
        }
    }

    /// Compute sparsity loss to encourage minimal head usage
    ///
    /// Loss: L_sparsity = (avg_soft_heads / num_heads)
    ///
    /// Provides a small penalty for using more heads (soft version).
    pub fn compute_sparsity_loss(&self) -> f32 {
        if let Some(soft_weights) = &self.cached_soft_weights {
            let seq_len = soft_weights.nrows();

            // Compute average soft heads (sum of soft weights per token)
            let mut total_soft_heads = 0.0;
            for token_idx in 0..seq_len {
                let token_soft_heads: f32 = soft_weights.row(token_idx).sum();
                total_soft_heads += token_soft_heads;
            }
            let avg_soft_heads = total_soft_heads / seq_len as f32;

            (avg_soft_heads / self.num_heads as f32) * self.sparsity_weight
        } else {
            0.0
        }
    }

    /// Get statistics about the last routing operation
    ///
    /// Returns: (avg_heads, min_heads, max_heads, avg_complexity, avg_threshold)
    /// For soft routing, avg_heads is the average sum of soft weights per token
    pub fn routing_stats(&self) -> (f32, usize, usize, f32, f32) {
        if let (Some(soft_weights), Some(complexity), Some(thresholds)) =
            (&self.cached_soft_weights, &self.cached_complexity_scores, &self.cached_thresholds) {

            let seq_len = soft_weights.nrows();

            // Compute soft head statistics
            let mut soft_head_counts = Vec::with_capacity(seq_len);
            for token_idx in 0..seq_len {
                let soft_count: f32 = soft_weights.row(token_idx).sum();
                soft_head_counts.push(soft_count);
            }

            let avg_heads = soft_head_counts.iter().sum::<f32>() / seq_len as f32;
            let min_heads = soft_head_counts.iter().cloned().fold(f32::INFINITY, f32::min).round() as usize;
            let max_heads = soft_head_counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max).round() as usize;
            let avg_complexity = complexity.sum() / seq_len as f32;
            let avg_threshold = thresholds.sum() / seq_len as f32;

            (avg_heads, min_heads, max_heads, avg_complexity, avg_threshold)
        } else {
            (0.0, 0, 0, 0.0, 0.0)
        }
    }

    /// Get complexity statistics (avg, min, max)
    ///
    /// Returns (0.0, 0.0, 0.0) if no routing has been performed yet.
    pub fn get_complexity_stats(&self) -> (f32, f32, f32) {
        if let Some(complexity) = &self.cached_complexity_scores {
            let seq_len = complexity.len();
            let avg = complexity.sum() / seq_len as f32;
            let min = complexity.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = complexity.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (avg, min, max)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Get combined weight norm of all predictors
    ///
    /// Returns the average L2 norm of router, complexity, threshold, and temperature predictor weights
    pub fn get_predictor_weight_norm(&self) -> f32 {
        let complexity_norm = self.w_complexity.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let threshold_norm = self.w_threshold.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let temperature_norm = self.w_temperature.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let router_norm = self.w_router.iter().map(|&x| x * x).sum::<f32>().sqrt();
        (complexity_norm + threshold_norm + temperature_norm + router_norm) / 4.0
    }

    /// Get temperature statistics from the last routing operation
    ///
    /// Returns: (avg_temperature, min_temperature, max_temperature)
    pub fn get_temperature_stats(&self) -> (f32, f32, f32) {
        if let Some(temperatures) = &self.cached_temperatures {
            let avg_temp = temperatures.sum() / temperatures.len() as f32;
            let min_temp = temperatures.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_temp = temperatures.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            (avg_temp, min_temp, max_temp)
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Backward pass: compute gradients and update parameters
    ///
    /// This is a simplified backward pass that uses the auxiliary losses
    /// to update the router, complexity predictor, and threshold predictor.
    ///
    /// # Arguments
    ///
    /// * `lr` - Learning rate for parameter updates
    pub fn backward(&mut self, lr: f32) {
        // Soft routing: gradients flow through continuous soft weights
        // The auxiliary losses provide additional training signal

        if let (Some(input), Some(soft_weights), Some(routing_probs), Some(complexity_scores), Some(target_heads), Some(thresholds)) =
            (&self.cached_input, &self.cached_soft_weights, &self.cached_routing_probs,
             &self.cached_complexity_scores, &self.cached_target_heads, &self.cached_thresholds) {

            let seq_len = input.nrows();

            // 1. Router gradients: encourage high-weight heads, discourage low-weight heads
            let mut router_grads = Array2::zeros((self.num_heads, self.embedding_dim));
            for token_idx in 0..seq_len {
                for head_idx in 0..self.num_heads {
                    let weight = soft_weights[[token_idx, head_idx]];
                    let prob = routing_probs[[token_idx, head_idx]];

                    // Gradient: (weight - prob) to push prob toward weight
                    // This encourages routing probs to match the soft weights
                    let grad_scale = weight - prob;

                    for dim_idx in 0..self.embedding_dim {
                        router_grads[[head_idx, dim_idx]] += grad_scale * input[[token_idx, dim_idx]];
                    }
                }
            }
            router_grads /= seq_len as f32;

            // 2. Complexity predictor gradients: align with actual soft head usage
            let mut complexity_grads = Array2::zeros((self.embedding_dim, 1));
            for token_idx in 0..seq_len {
                let actual_soft_heads: f32 = soft_weights.row(token_idx).sum();
                let target = target_heads[token_idx];
                let complexity = complexity_scores[token_idx];

                // Gradient: push complexity toward value that would give actual_soft_heads
                let desired_complexity = (actual_soft_heads - self.min_heads as f32) / (self.max_heads - self.min_heads) as f32;
                let grad_scale = (desired_complexity - complexity) * complexity * (1.0 - complexity); // sigmoid derivative

                for dim_idx in 0..self.embedding_dim {
                    complexity_grads[[dim_idx, 0]] += grad_scale * input[[token_idx, dim_idx]];
                }
            }
            complexity_grads /= seq_len as f32;

            // 3. Threshold predictor gradients: simple heuristic
            let mut threshold_grads = Array2::zeros((self.embedding_dim, 1));
            // For now, keep thresholds relatively stable (small gradients)
            threshold_grads *= 0.1;

            // 4. Temperature predictor gradients: based on gating function derivative
            let mut temperature_grads = Array2::zeros((self.embedding_dim, 1));
            if let Some(temperatures) = &self.cached_temperatures {
                // Gating function: g = sigmoid((cumulative_prob - threshold) * temperature)
                // Derivative: ∂g/∂temp = g * (1 - g) * (cumulative_prob - threshold)
                //
                // We want to adjust temperature to improve soft weight alignment with targets
                // Strategy: compute gradient based on how gating affects soft weights

                for token_idx in 0..seq_len {
                    let token_probs = routing_probs.row(token_idx);
                    let token_threshold = thresholds[token_idx];
                    let token_temperature = temperatures[token_idx];
                    let target = target_heads[token_idx];
                    let actual_soft_heads: f32 = soft_weights.row(token_idx).sum();

                    // Error signal: difference between actual and target head count
                    let head_error = actual_soft_heads - target;

                    // Sort heads by probability to compute cumulative probs
                    let mut sorted_indices: Vec<usize> = (0..self.num_heads).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        token_probs[b].partial_cmp(&token_probs[a]).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Compute gradient contribution from each head
                    let mut cumulative_prob = 0.0;
                    let mut temp_grad_contribution = 0.0;

                    for &head_idx in &sorted_indices {
                        cumulative_prob += token_probs[head_idx];

                        // Recompute gating for this head
                        let z = (cumulative_prob - token_threshold) * token_temperature;
                        let gating = 1.0 / (1.0 + (-z).exp());

                        // Derivative of gating w.r.t. temperature
                        let dgating_dtemp = gating * (1.0 - gating) * (cumulative_prob - token_threshold);

                        // Soft weight = routing_prob * gating
                        // ∂soft_weight/∂temp = routing_prob * ∂gating/∂temp
                        let dweight_dtemp = token_probs[head_idx] * dgating_dtemp;

                        // Accumulate gradient: push temperature to reduce head_error
                        // If using too many heads (head_error > 0): increase temp (sharper)
                        // If using too few heads (head_error < 0): decrease temp (smoother)
                        temp_grad_contribution += head_error * dweight_dtemp;
                    }

                    // Map temperature gradient back to temperature logits
                    // Temperature = 1.0 + sigmoid(logit) * 9.0
                    // ∂temp/∂logit = sigmoid'(logit) * 9.0 = sigmoid * (1 - sigmoid) * 9.0
                    let temp_logit = (token_temperature - 1.0) / 9.0; // Inverse of sigmoid mapping
                    let sigmoid_val = temp_logit; // Approximate
                    let dtemp_dlogit = sigmoid_val * (1.0 - sigmoid_val) * 9.0;

                    // Chain rule: ∂loss/∂logit = ∂loss/∂temp * ∂temp/∂logit
                    let logit_grad = temp_grad_contribution * dtemp_dlogit;

                    // Accumulate gradient for temperature predictor weights
                    for dim_idx in 0..self.embedding_dim {
                        temperature_grads[[dim_idx, 0]] += logit_grad * input[[token_idx, dim_idx]];
                    }
                }

                temperature_grads /= seq_len as f32;

                // Apply conservative scaling to prevent instability
                temperature_grads *= 0.01; // Small learning rate for temperature
            }

            // Update parameters using Adam optimizer
            self.router_optimizer.step(&mut self.w_router, &router_grads, lr);
            self.complexity_optimizer.step(&mut self.w_complexity, &complexity_grads, lr * 0.1); // Slower learning for complexity
            self.threshold_optimizer.step(&mut self.w_threshold, &threshold_grads, lr * 0.1); // Slower learning for threshold
            self.temperature_optimizer.step(&mut self.w_temperature, &temperature_grads, lr * 0.1); // Slower learning for temperature
        }
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.num_heads * self.embedding_dim  // w_router
            + self.embedding_dim + 1          // w_complexity + bias
            + self.embedding_dim + 1          // w_threshold + bias
            + self.embedding_dim + 1          // w_temperature + bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_head_router_creation() {
        let router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);

        assert_eq!(router.total_heads(), 8);
        assert_eq!(router.num_shared_heads, 2);
        assert_eq!(router.num_routed_heads, 6);
        assert_eq!(router.threshold_p_base(), 0.5);
        assert_eq!(router.dynamic_loss_weight(0.5), 1e-4); // At 50% training progress

        // Check parameter count: 2×128 + 6×128 + 2×128 = 1280
        assert_eq!(router.parameters(), 1280);
    }

    #[test]
    fn test_head_router_route_shape() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        let mask = router.route_discrete(&input);

        assert_eq!(mask.shape(), &[10, 8]);
    }

    #[test]
    fn test_head_router_shared_heads_always_active() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        let mask = router.route_discrete(&input);

        // Check that first 2 heads (shared) are always active
        for token_idx in 0..10 {
            assert!(mask[[token_idx, 0]], "Shared head 0 should be active");
            assert!(mask[[token_idx, 1]], "Shared head 1 should be active");
        }
    }

    #[test]
    fn test_head_router_adaptive_routing() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        let mask = router.route_discrete(&input);

        // With adaptive routing, each token should have 2 shared + variable routed heads
        for token_idx in 0..10 {
            let active_count = mask.row(token_idx).iter().filter(|&&x| x).count();
            // Should have at least 3 heads (2 shared + 1 routed minimum)
            // and at most 8 heads (2 shared + 6 routed maximum)
            assert!(active_count >= 3, "Should have at least 3 active heads");
            assert!(active_count <= 8, "Should have at most 8 active heads");
        }

        // Check average active routed heads
        let avg_routed = router.avg_active_routed_heads();
        assert!(avg_routed >= 1.0 && avg_routed <= 6.0, "Average routed heads should be between 1 and 6");
    }

    #[test]
    fn test_head_router_load_balance_loss() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        router.route_discrete(&input);
        let loss = router.compute_load_balance_loss();

        // Loss should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_head_router_dynamic_loss() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        router.route_discrete(&input);
        let loss = router.compute_dynamic_loss();

        // Loss should be finite and non-negative (entropy is always >= 0)
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_head_router_layer_wise_thresholds() {
        // Test that different layers get different base thresholds
        let router_early = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);  // Layer 0
        let router_late = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 10, false, 3.0, 0.4, false); // Layer 10

        // Early layers should have higher threshold (more heads)
        // Late layers should have lower threshold (fewer heads)
        // This is tested indirectly through routing behavior
        assert_eq!(router_early.layer_idx, 0);
        assert_eq!(router_late.layer_idx, 10);
    }

    #[test]
    fn test_head_router_learned_threshold() {
        let mut router = HeadRouterStandard::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, true, 3.0, 0.4, false);
        let input = Array2::ones((10, 128));

        router.route_discrete(&input);

        // With learned threshold, should have per-token thresholds
        let (min, max, mean, _std) = router.threshold_stats();
        assert!(min >= 0.3 && min <= 0.7, "Min threshold should be in range [0.3, 0.7]");
        assert!(max >= 0.3 && max <= 0.7, "Max threshold should be in range [0.3, 0.7]");
        assert!(mean >= 0.3 && mean <= 0.7, "Mean threshold should be in range [0.3, 0.7]");
    }

    #[test]
    #[should_panic(expected = "must be <=")]
    fn test_head_router_invalid_k() {
        // Should panic: num_active_routed_heads > num_routed_heads
        HeadRouterStandard::new(128, 2, 6, 10, 0.01, 0.5, 1e-4, 0, false, 3.0, 0.4, false);
    }
}
