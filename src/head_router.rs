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

/// Router network for Mixture-of-Heads attention
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HeadRouter {
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

impl HeadRouter {
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

        HeadRouter {
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
            // Extract routed head portion of mask
            let routed_mask = mask
                .slice(ndarray::s![.., self.num_shared_heads..])
                .to_owned();

            routing::compute_load_balance_loss(routed_scores, &routed_mask)
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

        // 1. Sparsity-based adaptation
        let sparsity_multiplier = if current_avg > self.target_avg_routed_heads {
            // Using too many heads → increase penalty
            (current_avg / self.target_avg_routed_heads).min(2.0)
        } else if current_avg > 0.0 {
            // Using fewer heads → decrease penalty
            (current_avg / self.target_avg_routed_heads).powi(2).max(0.5)
        } else {
            1.0
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_head_router_creation() {
        let router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);

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
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);
        let input = Array2::ones((10, 128));

        let mask = router.route(&input);

        assert_eq!(mask.shape(), &[10, 8]);
    }

    #[test]
    fn test_head_router_shared_heads_always_active() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);
        let input = Array2::ones((10, 128));

        let mask = router.route(&input);

        // Check that first 2 heads (shared) are always active
        for token_idx in 0..10 {
            assert!(mask[[token_idx, 0]], "Shared head 0 should be active");
            assert!(mask[[token_idx, 1]], "Shared head 1 should be active");
        }
    }

    #[test]
    fn test_head_router_adaptive_routing() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);
        let input = Array2::ones((10, 128));

        let mask = router.route(&input);

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
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);
        let input = Array2::ones((10, 128));

        router.route(&input);
        let loss = router.compute_load_balance_loss();

        // Loss should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_head_router_dynamic_loss() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);
        let input = Array2::ones((10, 128));

        router.route(&input);
        let loss = router.compute_dynamic_loss();

        // Loss should be finite and non-negative (entropy is always >= 0)
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_head_router_layer_wise_thresholds() {
        // Test that different layers get different base thresholds
        let router_early = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, false, 3.0);  // Layer 0
        let router_late = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 10, false, 3.0); // Layer 10

        // Early layers should have higher threshold (more heads)
        // Late layers should have lower threshold (fewer heads)
        // This is tested indirectly through routing behavior
        assert_eq!(router_early.layer_idx, 0);
        assert_eq!(router_late.layer_idx, 10);
    }

    #[test]
    fn test_head_router_learned_threshold() {
        let mut router = HeadRouter::new(128, 2, 6, 4, 0.01, 0.5, 1e-4, 0, true, 3.0);
        let input = Array2::ones((10, 128));

        router.route(&input);

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
        HeadRouter::new(128, 2, 6, 10, 0.01, 0.5, 1e-4, 0, false, 3.0);
    }
}
