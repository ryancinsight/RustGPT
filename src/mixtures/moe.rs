//! # Mixture of Experts (MoE)
//!
//! This module implements Mixture-of-Experts (MoE), a sparse routing mechanism
//! for feedforward layers that increases model capacity while maintaining efficiency.
//!
//! ## Overview
//!
//! Mixture-of-Experts dynamically selects which expert networks to activate per token
//! using learned AutoDeco-inspired predictors. This provides better parameter efficiency
//! than dense feedforward layers.
//!
//! ## Architecture
//!
//! Based on "Switch Transformers: Scaling to Trillion Parameter Models with Simple and
//! Efficient Sparsity" (Fedus et al., 2021) and inspired by AutoDeco's neural
//! architecture for learned decoding. The implementation uses a two-layer neural
//! network with Richards normalization for adaptive expert routing.
//!
//! ## Key Components
//!
//! - **ExpertRouter**: Configuration for learned expert selection
//! - **ExpertSelector**: AutoDeco-inspired two-layer network for routing prediction
//! - **RichardsExpert**: Individual expert using Richards GLU components
//! - **Complexity-aware routing**: Learns optimal expert usage patterns
//! - **Load balancing**: Prevents routing collapse to single expert

use serde::{Deserialize, Serialize};

use crate::{
    mixtures::{
        gating::{GatingConfig, GatingStrategy},
        routing::{Router, RoutingConfig, RoutingResult, SelectionAlgorithm},
        threshold::ThresholdPredictor,
    },
    network::Layer,
    rng::get_rng,
};

#[inline]
fn default_true() -> bool {
    true
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    use std::sync::OnceLock;
    static CURVE: OnceLock<crate::richards::RichardsCurve> = OnceLock::new();
    let curve = CURVE.get_or_init(|| crate::richards::RichardsCurve::sigmoid(false));
    curve.forward_scalar(x as f64) as f32
}

/// Strategy for selecting which experts to activate
///
/// Implements Mixture-of-Experts (MoE) for dynamic expert selection per token.
/// Based on "Switch Transformers" (Fedus et al., 2021) with learned routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertRouter {
    /// Learned Mixture-of-Experts: Uses shared gating strategy with expert-specific config
    LearnedMoE {
        /// Number of experts in the mixture
        num_experts: usize,
        /// Number of experts to activate per token (top-k routing)
        num_active_experts: usize,
        /// Hidden dimension for each expert (smaller than main feedforward)
        expert_hidden_dim: usize,
        /// Weight for load balance loss (prevents routing collapse)
        load_balance_weight: f32,
        /// Weight for sparsity loss (encourages minimal expert usage)
        sparsity_weight: f32,
        /// Weight for diversity loss (encourages expert specialization)
        diversity_weight: f32,

        /// If true, route experts using an extra conditioning feature derived from
        /// Mixture-of-Heads activity (e.g. avg active heads / num_heads).
        ///
        /// This makes MoE routing explicitly depend on MoH behavior while keeping routing fully
        /// learned.
        #[serde(default)]
        use_head_conditioning: bool,

        /// If true, use a small learned adapter to make expert sparsity adaptive.
        ///
        /// This predicts a smooth blend between top-1 and configured top-k routing based on
        /// routing uncertainty (entropy) and MoH head activity.
        #[serde(default = "default_true")]
        use_learned_k_adaptation: bool,
    },
}

/// Configuration for expert routing metrics and learned parameters
///
/// Extends the shared GatingConfig with MoE-specific parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRouterConfig {
    /// Shared gating configuration
    pub gating: GatingConfig,
    /// Number of experts in the mixture
    pub num_experts: usize,
    /// Hidden dimension for each expert
    pub expert_hidden_dim: usize,

    /// Whether to append a head-activity conditioning scalar to the router input.
    #[serde(default)]
    pub use_head_conditioning: bool,

    /// If true, learn a smooth adaptive expert-count signal (blend top-1 and top-k).
    #[serde(default = "default_true")]
    pub use_learned_k_adaptation: bool,

    /// Weight for diversity loss (encourages expert specialization)
    pub diversity_weight: f32,
    /// Metrics: average routing probability per expert
    pub metrics_avg_routing_prob: Vec<f32>,
    /// Metrics: diversity score (average pairwise expert correlation)
    pub metrics_diversity_score: f32,
}

impl Default for ExpertRouterConfig {
    fn default() -> Self {
        Self {
            gating: GatingConfig::default(),
            num_experts: 4,
            expert_hidden_dim: 64,
            use_head_conditioning: false,
            use_learned_k_adaptation: true,
            diversity_weight: 0.005,
            metrics_avg_routing_prob: vec![0.0; 4],
            metrics_diversity_score: 0.0,
        }
    }
}

impl ExpertRouterConfig {
    /// Create expert router config from strategy
    pub fn from_router(router: &ExpertRouter) -> Self {
        match router {
            ExpertRouter::LearnedMoE {
                num_experts,
                num_active_experts,
                expert_hidden_dim,
                load_balance_weight,
                sparsity_weight,
                diversity_weight,
                use_head_conditioning,
                use_learned_k_adaptation,
            } => Self {
                gating: GatingConfig::from_strategy(
                    &GatingStrategy::Learned {
                        num_active: *num_active_experts,
                        load_balance_weight: *load_balance_weight,
                        sparsity_weight: *sparsity_weight,
                        complexity_loss_weight: 0.005, // Default
                    },
                    *num_experts,
                ),
                num_experts: *num_experts,
                expert_hidden_dim: *expert_hidden_dim,
                use_head_conditioning: *use_head_conditioning,
                use_learned_k_adaptation: *use_learned_k_adaptation,
                diversity_weight: *diversity_weight,
                metrics_avg_routing_prob: vec![0.0; *num_experts],
                metrics_diversity_score: 0.0,
            },
        }
    }
}

/// Small learned adapter that predicts how much to "open up" expert routing.
///
/// Produces $\alpha \in [0,1]$ used to blend between:
/// - top-1 masked routing probabilities
/// - top-k masked routing probabilities (k = configured `gating.num_active`)
///
/// Features: (normalized routing entropy, MoH head activity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedKAdapter {
    /// Linear weights for [entropy_norm, head_activity] -> logit(alpha). Shape: (2, 1)
    pub w: ndarray::Array2<f32>,
    /// Bias. Shape: (1, 1)
    pub b: ndarray::Array2<f32>,
}

impl LearnedKAdapter {
    pub fn new() -> Self {
        // Initialize close to the old behavior (alpha ≈ head_activity) around 0.5.
        // sigmoid(4*h - 2) has midpoint at h=0.5.
        let mut w = ndarray::Array2::<f32>::zeros((2, 1));
        w[[0, 0]] = 0.0; // entropy weight starts neutral
        w[[1, 0]] = 4.0; // head activity drives alpha initially
        let mut b = ndarray::Array2::<f32>::zeros((1, 1));
        b[[0, 0]] = -2.0;
        Self { w, b }
    }

    #[inline]
    pub fn alpha(&self, entropy_norm: f32, head_activity: f32) -> f32 {
        let e = if entropy_norm.is_finite() {
            entropy_norm.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let h = if head_activity.is_finite() {
            head_activity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let z = self.w[[0, 0]] * e + self.w[[1, 0]] * h + self.b[[0, 0]];
        sigmoid(z)
    }
}

impl ExpertRouterConfig {
    /// Reset metrics when router changes
    pub fn reset_metrics(&mut self) {
        self.gating.reset_metrics();
        for e in 0..self.metrics_avg_routing_prob.len() {
            self.metrics_avg_routing_prob[e] = 0.0;
        }
        self.metrics_diversity_score = 0.0;
    }

    /// Update routing metrics for training optimization
    /// routing_probs: shape (num_tokens, num_experts) - routing probabilities for each token-expert
    /// pair
    pub fn update_metrics(&mut self, routing_probs: &ndarray::ArrayView2<f32>) {
        // Update shared gating metrics
        self.gating.update_metrics(routing_probs);

        // Update MoE-specific routing probability averages
        let num_tokens = routing_probs.nrows() as f32;
        let total_decisions = self.gating.metrics.total_decisions as f32 + num_tokens;

        // Use zip to iterate over metrics and routing columns simultaneously (zero-copy)
        self.metrics_avg_routing_prob
            .iter_mut()
            .zip(routing_probs.columns())
            .for_each(|(metric, routing_col)| {
                let expert_avg_prob = routing_col.mean().unwrap_or(0.0);
                let current_avg = *metric;
                *metric =
                    current_avg + (expert_avg_prob - current_avg) * num_tokens / total_decisions;
            });
    }

    /// Get load balancing loss for training (prevents single expert dominance)
    pub fn compute_load_balance_loss(&self) -> f32 {
        self.gating.compute_load_balance_loss()
    }

    /// Get sparsity loss for training (encourages minimal expert usage)
    pub fn compute_sparsity_loss(&self) -> f32 {
        self.gating.compute_sparsity_loss()
    }

    /// Get complexity alignment loss for training (aligns expert usage with predicted complexity)
    pub fn compute_complexity_loss(&self, target_avg_experts: f32) -> f32 {
        self.gating.compute_complexity_loss(target_avg_experts)
    }

    /// Get diversity loss for training (encourages expert specialization)
    pub fn compute_diversity_loss(&self) -> f32 {
        if self.gating.metrics.total_decisions == 0 {
            return 0.0;
        }

        // Compute average pairwise correlation between expert routing probabilities
        // using iterator chains for zero-copy and functional composition
        let probs_slice = &self.metrics_avg_routing_prob;

        let (total_correlation, pair_count) = (0..self.num_experts)
            .flat_map(|i| ((i + 1)..self.num_experts).map(move |j| (i, j)))
            .filter_map(|(i, j)| {
                let prob_i = probs_slice[i];
                let prob_j = probs_slice[j];
                let norm_i = prob_i.abs();
                let norm_j = prob_j.abs();

                if norm_i > 0.0 && norm_j > 0.0 {
                    let correlation = (prob_i * prob_j) / (norm_i * norm_j);
                    Some(correlation.abs())
                } else {
                    None
                }
            })
            .fold((0.0, 0), |(total, count), correlation| {
                (total + correlation, count + 1)
            });

        if pair_count == 0 {
            0.0
        } else {
            total_correlation / pair_count as f32
        }
    }

    /// Compute MoE auxiliary losses: (load-balance, complexity, sparsity, diversity).
    pub fn compute_moe_aux_losses(&self, target_avg_experts: f32) -> (f32, f32, f32, f32) {
        let lb = self.compute_load_balance_loss();
        let cx = self.compute_complexity_loss(target_avg_experts);
        let sp = self.compute_sparsity_loss();
        let dv = self.compute_diversity_loss();
        (lb, cx, sp, dv)
    }

    /// Weighted MoE auxiliary penalty used during training.
    pub fn compute_moe_aux_weighted_total(&self, target_avg_experts: f32) -> f32 {
        let (lb, cx, sp, dv) = self.compute_moe_aux_losses(target_avg_experts);
        let g = &self.gating;
        (lb * g.load_balance_weight)
            + (cx * g.complexity_loss_weight)
            + (sp * g.sparsity_weight)
            + (dv * self.diversity_weight)
    }

    /// Get average number of active experts per token (soft routing equivalent)
    pub fn get_avg_active_experts(&self) -> f32 {
        self.gating.get_avg_active_components()
    }

    /// Get average number of experts with significant routing probability (> 0.1)
    pub fn get_avg_significant_experts(&self) -> f32 {
        self.gating.get_avg_significant_components()
    }

    /// Get routing entropy (higher = more uniform distribution across experts)
    pub fn get_routing_entropy(&self) -> f32 {
        self.gating.get_gating_entropy()
    }
}

/// Router implementation for expert selection in Mixture-of-Experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertRouterImpl {
    /// Routing configuration
    pub config: RoutingConfig,
    /// Number of experts available for selection
    pub num_experts: usize,
}

impl ExpertRouterImpl {
    /// Create a new expert router
    pub fn new(num_experts: usize, config: RoutingConfig) -> Self {
        Self {
            config,
            num_experts,
        }
    }

    /// Create router from gating strategy
    pub fn from_strategy(strategy: &GatingStrategy, num_experts: usize) -> Self {
        let config = match strategy {
            GatingStrategy::Learned { num_active, .. } => RoutingConfig {
                algorithm: SelectionAlgorithm::Softmax,
                use_learned_predictor: true,
                num_active: *num_active,
                temperature: 1.0,
                soft_top_p_alpha: 50.0,
            },
            GatingStrategy::SoftTopP {
                top_p,
                soft_top_p_alpha,
            } => RoutingConfig {
                algorithm: SelectionAlgorithm::SoftTopP { top_p: *top_p },
                use_learned_predictor: false,
                num_active: num_experts, // All experts available for soft selection
                temperature: 1.0,
                soft_top_p_alpha: *soft_top_p_alpha,
            },
            GatingStrategy::Fixed { num_active } => RoutingConfig {
                algorithm: SelectionAlgorithm::TopK { k: *num_active },
                use_learned_predictor: false,
                num_active: *num_active,
                temperature: 1.0,
                soft_top_p_alpha: 50.0,
            },
        };
        Self::new(num_experts, config)
    }
}

impl Router for ExpertRouterImpl {
    fn route(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        predictor: Option<&mut ThresholdPredictor>,
    ) -> RoutingResult {
        // Generate raw gating values (routing logits)
        let raw_gates = if self.config.use_learned_predictor {
            if let Some(predictor) = predictor {
                // Use predictor to generate routing logits for each expert
                predictor.predict(input)
            } else {
                // Fallback: uniform routing
                ndarray::Array2::zeros((input.nrows(), self.num_experts))
            }
        } else {
            // Fixed selection: route to first k experts equally using iterator chains
            let n_tokens = input.nrows();
            let active_experts = self.config.num_active.min(self.num_experts);
            let uniform_weight = 1.0 / self.config.num_active as f32;

            // Use iterator chains to construct gate values (zero-copy array construction)
            let gate_data: Vec<f32> = (0..n_tokens)
                .flat_map(|_| {
                    (0..self.num_experts).map(move |expert_idx| {
                        if expert_idx < active_experts {
                            uniform_weight
                        } else {
                            0.0
                        }
                    })
                })
                .collect();

            ndarray::Array2::from_shape_vec((n_tokens, self.num_experts), gate_data)
                .unwrap_or_else(|_| ndarray::Array2::<f32>::zeros((n_tokens, self.num_experts)))
        };

        // Apply selection algorithm (for MoE, typically softmax for soft routing)
        let routing_weights =
            crate::mixtures::routing::apply_selection_algorithm(&raw_gates.view(), &self.config);

        RoutingResult {
            routing_weights,
            raw_gates,
        }
    }
}

/// Parameter information for the expert selector (router)
#[derive(Debug, Clone)]
struct RouterParamInfo {
    /// Shapes of weight matrices: [w1_shape, w2_shape]
    weight_shapes: Vec<(usize, usize)>,
    /// Shapes of bias vectors: [b1_shape, b2_shape]
    bias_shapes: Vec<usize>,
    /// Number of Richards normalization parameters
    norm_params: usize,
    /// Number of Richards activation parameters
    activation_params: usize,
    /// Number of Richards sigmoid parameters
    sigmoid_params: usize,
    /// Total parameter count
    total_params: usize,
}

/// Enhanced expert selector inspired by AutoDeco
///
/// This implements a two-layer neural network for expert routing with proper
/// forward and backward computations. Follows the same architecture as the shared
/// ThresholdPredictor (AutoDeco-inspired with Richards normalization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertSelector {
    /// First layer weights (embed_dim x router_hidden_dim)
    pub weights1: ndarray::Array2<f32>,
    /// First layer biases (router_hidden_dim)
    pub bias1: ndarray::Array1<f32>,
    /// Second layer weights (router_hidden_dim x num_experts)
    pub weights2: ndarray::Array2<f32>,
    /// Second layer bias (num_experts)
    pub bias2: ndarray::Array1<f32>,
    /// Richards normalization for adaptive behavior
    pub norm: crate::richards::RichardsNorm,
    /// Richards sigmoid for stable activation
    pub sigmoid: crate::richards::RichardsCurve,
    /// Learned Richards activation replacing ReLU
    pub activation: crate::richards::RichardsGate,
    /// Softmax layer for probability normalization
    pub softmax: crate::soft::Softmax,

    /// Optional learned mapping from per-head activity (MoH) into per-expert logit biases.
    /// Shape: (num_heads, num_experts).
    ///
    /// This is a learned/adaptive representation of how different heads influence each expert.
    #[serde(default)]
    pub head_to_expert: Option<ndarray::Array2<f32>>,

    /// Cached parameter information for gradient computation
    #[serde(skip)]
    param_info: Option<RouterParamInfo>,

    /// Cached activations for gradient computation
    #[serde(skip)]
    cached_input: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_hidden: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_normalized: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_activated: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_logits: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_output: Option<ndarray::Array2<f32>>,

    /// Cached per-head activity vector used for conditioning during the last predict.
    #[serde(skip)]
    cached_head_activity_vec: Option<ndarray::Array1<f32>>,
}

impl ExpertSelector {
    /// Create a new expert selector with AutoDeco-inspired architecture
    pub fn new(embed_dim: usize, router_hidden_dim: usize, num_experts: usize) -> Self {
        use rand::Rng;
        let mut rng = get_rng();

        // Xavier initialization: weights ~ N(0, 1/sqrt(fan_in))
        let scale1 = 1.0 / (embed_dim as f32).sqrt();
        let scale2 = 1.0 / (router_hidden_dim as f32).sqrt();

        let weights1 = ndarray::Array2::from_shape_fn((embed_dim, router_hidden_dim), |_| {
            rng.random_range(-scale1..scale1)
        });

        let bias1 = ndarray::Array1::zeros(router_hidden_dim);

        let weights2 = ndarray::Array2::from_shape_fn((router_hidden_dim, num_experts), |_| {
            rng.random_range(-scale2..scale2)
        });

        let bias2 = ndarray::Array1::zeros(num_experts);

        let norm = crate::richards::RichardsNorm::new(router_hidden_dim);
        let sigmoid = crate::richards::RichardsCurve::sigmoid(false); // Non-learnable sigmoid
        let activation = crate::richards::RichardsGate::new(); // Learned Richards gating replacing ReLU

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            norm,
            sigmoid,
            activation,
            softmax: crate::soft::Softmax::new(),
            head_to_expert: None,
            param_info: None,
            cached_input: None,
            cached_hidden: None,
            cached_normalized: None,
            cached_activated: None,
            cached_logits: None,
            cached_output: None,
            cached_head_activity_vec: None,
        }
    }

    fn ensure_head_to_expert(&mut self, num_heads: usize, num_experts: usize) {
        let needs_init = match self.head_to_expert.as_ref() {
            Some(w) => w.nrows() != num_heads || w.ncols() != num_experts,
            None => true,
        };
        if !needs_init {
            return;
        }

        use rand::Rng;
        let mut rng = get_rng();
        let scale = 0.01_f32;
        let w = ndarray::Array2::from_shape_fn((num_heads, num_experts), |_| {
            rng.random_range(-scale..scale)
        });
        self.head_to_expert = Some(w);
        // Param accounting depends on whether this optional matrix exists.
        self.param_info = None;
    }

    fn compute_head_bias(
        &mut self,
        head_activity: &[f32],
        num_experts: usize,
    ) -> ndarray::Array1<f32> {
        let num_heads = head_activity.len();
        self.ensure_head_to_expert(num_heads, num_experts);
        let w = self
            .head_to_expert
            .as_ref()
            .expect("head_to_expert must be initialized");

        let mut bias = ndarray::Array1::<f32>::zeros(num_experts);
        for h in 0..num_heads {
            let a = head_activity[h];
            let a = if a.is_finite() { a.max(0.0) } else { 0.0 };
            if a == 0.0 {
                continue;
            }
            for e in 0..num_experts {
                bias[e] += a * w[[h, e]];
            }
        }
        bias
    }

    /// Predict expert routing probabilities using AutoDeco-style architecture
    ///
    /// Returns softmax-normalized probabilities in [0, 1] range suitable for expert selection
    /// Caches intermediate activations for gradient computation
    pub fn predict(&mut self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        self.predict_with_head_activity(input, None)
    }

    /// Predict expert routing probabilities, optionally conditioned by per-head activity.
    ///
    /// Conditioning is applied as a learned additive bias to logits:
    /// logits = f(x) + head_activity · W_head_to_expert
    pub fn predict_with_head_activity(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        head_activity: Option<&[f32]>,
    ) -> ndarray::Array2<f32> {
        // Cache input for gradient computation (zero-copy where possible)
        self.cached_input = Some(input.to_owned());
        self.cached_head_activity_vec = head_activity.map(|v| {
            let mut a = ndarray::Array1::<f32>::zeros(v.len());
            for (i, &x) in v.iter().enumerate() {
                a[i] = if x.is_finite() { x.max(0.0) } else { 0.0 };
            }
            a
        });

        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;
        self.cached_hidden = Some(hidden);

        // Apply Richards normalization for adaptive behavior
        let hidden_ref = self
            .cached_hidden
            .as_ref()
            .expect("predict must cache hidden activations");
        let normalized = self.norm.forward(hidden_ref);
        self.cached_normalized = Some(normalized);

        // Learned Richards gating replacing ReLU
        let normalized_ref = self
            .cached_normalized
            .as_ref()
            .expect("predict must cache normalized activations");
        let activation_output = self.activation.forward(normalized_ref);
        self.cached_activated = Some(activation_output);

        // Second layer: W2 * activated + b2
        let activated_ref = self
            .cached_activated
            .as_ref()
            .expect("predict must cache activated values");
        let mut logits = activated_ref.dot(&self.weights2) + &self.bias2;
        if let Some(h) = head_activity {
            if !h.is_empty() {
                let bias = self.compute_head_bias(h, self.bias2.len());
                logits = logits + &bias;
            }
        }
        self.cached_logits = Some(logits);

        // Softmax normalization for routing probabilities
        let logits_ref = self
            .cached_logits
            .as_ref()
            .expect("predict must cache logits");
        let output = self.softmax.forward(&logits_ref.view());
        self.cached_output = Some(output.clone());

        output
    }

    /// Forward pass for auxiliary computation (immutable)
    ///
    /// Returns softmax probabilities for expert routing
    pub fn forward(&self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;

        // Apply Richards normalization
        let normalized = self.norm.normalize_immutable(&hidden);

        // Learned Richards activation
        let activated = self
            .activation
            .forward_matrix(&normalized.mapv(|x| x as f64))
            .mapv(|x| x as f32);

        // Second layer: W2 * activated + b2
        let logits = activated.dot(&self.weights2) + &self.bias2;

        // Softmax normalization
        self.softmax.forward_immutable(&logits.view())
    }

    /// Select top-k experts based on routing probabilities
    pub fn select_experts(
        &self,
        routing_probs: &ndarray::Array2<f32>,
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut selections = Vec::new();

        for row in routing_probs.outer_iter() {
            let mut expert_probs: Vec<(usize, f32)> = row
                .iter()
                .enumerate()
                .map(|(idx, &prob)| {
                    let p = if prob.is_finite() { prob } else { 0.0 };
                    (idx, p)
                })
                .collect();

            // Sort by probability (descending)
            expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k experts
            let selected: Vec<usize> = expert_probs
                .into_iter()
                .take(k)
                .map(|(idx, _)| idx)
                .collect();

            selections.push(selected);
        }

        selections
    }

    /// Compute gradients for the two-layer routing network
    pub fn compute_gradients(
        &mut self,
        output_grads: &ndarray::Array2<f32>,
    ) -> (
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
        ndarray::Array2<f32>,
        ndarray::Array1<f32>,
        Vec<f64>,
    ) {
        // Retrieve cached activations
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_activated = self
            .cached_activated
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_normalized = self
            .cached_normalized
            .as_ref()
            .expect("predict must be called before compute_gradients");
        let cached_hidden = self
            .cached_hidden
            .as_ref()
            .expect("predict must be called before compute_gradients");

        // Gradient through softmax
        let d_output = self.softmax.backward(output_grads);

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_output);
        let grad_bias2 = d_output.sum_axis(ndarray::Axis(0));

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_output.dot(&self.weights2.t());

        // Gradient through Richards activation (replacing ReLU)
        let normalized_f64 = cached_normalized.mapv(|x| x as f64);
        let d_activated_f64 = d_activated.mapv(|x| x as f64);
        let activation_grad_f64 = self
            .activation
            .backward_matrix(&normalized_f64, &d_activated_f64);
        let d_normalized = activation_grad_f64.mapv(|x| x as f32);

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1 = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self
            .activation
            .grad_weights_matrix(&normalized_f64, &d_activated_f64);

        (
            grad_weights1,
            grad_bias1,
            grad_weights2,
            grad_bias2,
            activation_grads,
        )
    }

    /// Get parameters for gradient computation (iterator-based, zero-copy)
    pub fn parameters(&self) -> impl Iterator<Item = &ndarray::Array2<f32>> {
        [&self.weights1, &self.weights2].into_iter()
    }

    /// Get mutable parameters for gradient updates (iterator-based, zero-copy)
    pub fn parameters_mut(&mut self) -> impl Iterator<Item = &mut ndarray::Array2<f32>> {
        [&mut self.weights1, &mut self.weights2].into_iter()
    }

    /// Get bias parameters (iterator-based, zero-copy)
    pub fn biases(&self) -> impl Iterator<Item = &ndarray::Array1<f32>> {
        [&self.bias1, &self.bias2].into_iter()
    }

    /// Get mutable bias parameters (iterator-based, zero-copy)
    pub fn biases_mut(&mut self) -> impl Iterator<Item = &mut ndarray::Array1<f32>> {
        [&mut self.bias1, &mut self.bias2].into_iter()
    }

    /// Get parameter information for this router
    fn get_param_info(&mut self) -> &RouterParamInfo {
        if self.param_info.is_none() {
            // Extract parameter information from the router components
            let mut weight_shapes = vec![
                (self.weights1.nrows(), self.weights1.ncols()),
                (self.weights2.nrows(), self.weights2.ncols()),
            ];

            if let Some(w) = self.head_to_expert.as_ref() {
                weight_shapes.push((w.nrows(), w.ncols()));
            }

            let bias_shapes = vec![self.bias1.len(), self.bias2.len()];

            let norm_params = self.norm.parameters();
            let activation_params = self.activation.parameters();
            let sigmoid_params = self.sigmoid.weights().len();

            let head_params = self.head_to_expert.as_ref().map(|w| w.len()).unwrap_or(0);

            let total_params = self.parameters().map(|p| p.len()).sum::<usize>()
                + head_params
                + self.biases().map(|b| b.len()).sum::<usize>()
                + norm_params
                + activation_params
                + sigmoid_params;

            self.param_info = Some(RouterParamInfo {
                weight_shapes,
                bias_shapes,
                norm_params,
                activation_params,
                sigmoid_params,
                total_params,
            });
        }

        self.param_info.as_ref().unwrap()
    }

    /// Get the number of parameters for this router
    pub fn param_count(&mut self) -> usize {
        self.get_param_info().total_params
    }

    /// Get parameter shapes for gradient computation
    pub fn param_shapes(&mut self) -> (&[(usize, usize)], &[usize], usize, usize, usize) {
        let info = self.get_param_info();
        (
            &info.weight_shapes,
            &info.bias_shapes,
            info.norm_params,
            info.activation_params,
            info.sigmoid_params,
        )
    }
}

/// Individual expert using Richards GLU components
///
/// Each expert is a smaller RichardsGlu network specialized for different input patterns.
/// Experts share the same architecture but learn different parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichardsExpert {
    /// The underlying Richards GLU network for this expert
    pub glu: crate::richards::RichardsGlu,
    /// Cached parameter information for gradient computation
    #[serde(skip)]
    param_info: Option<ExpertParamInfo>,
}

/// Parameter information for an expert
#[derive(Debug, Clone)]
struct ExpertParamInfo {
    /// Shapes of weight matrices: [w1_shape, w2_shape, w_out_shape]
    weight_shapes: Vec<(usize, usize)>,
    /// Number of Richards activation parameters
    richards_activation_params: usize,
    /// Number of Richards gate parameters
    richards_gate_params: usize,
    /// Total parameter count
    total_params: usize,
}

impl RichardsExpert {
    /// Create a new expert with specified dimensions
    pub fn new(embedding_dim: usize, expert_hidden_dim: usize) -> Self {
        Self {
            glu: crate::richards::RichardsGlu::new(embedding_dim, expert_hidden_dim),
            param_info: None,
        }
    }

    /// Get parameter information for this expert
    fn get_param_info(&mut self) -> &ExpertParamInfo {
        if self.param_info.is_none() {
            // Extract parameter information from the underlying GLU
            let weight_shapes = vec![
                (self.glu.w1.nrows(), self.glu.w1.ncols()),
                (self.glu.w2.nrows(), self.glu.w2.ncols()),
                (self.glu.w_out.nrows(), self.glu.w_out.ncols()),
            ];

            let richards_activation_params = self.glu.richards_activation.weights().len();
            let richards_gate_params = self.glu.gate.parameters();

            let total_params = self.glu.parameters();

            self.param_info = Some(ExpertParamInfo {
                weight_shapes,
                richards_activation_params,
                richards_gate_params,
                total_params,
            });
        }

        self.param_info.as_ref().unwrap()
    }

    /// Get the number of parameters for this expert
    pub fn param_count(&mut self) -> usize {
        self.get_param_info().total_params
    }

    /// Get parameter shapes for gradient computation
    pub fn param_shapes(&mut self) -> (&[(usize, usize)], usize, usize) {
        let info = self.get_param_info();
        (
            &info.weight_shapes,
            info.richards_activation_params,
            info.richards_gate_params,
        )
    }
}

impl Layer for RichardsExpert {
    fn layer_type(&self) -> &str {
        "RichardsExpert"
    }

    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.glu.forward(input)
    }

    fn backward(&mut self, grads: &ndarray::Array2<f32>, lr: f32) -> ndarray::Array2<f32> {
        self.glu.backward(grads, lr)
    }

    fn parameters(&self) -> usize {
        self.glu.parameters()
    }

    fn compute_gradients(
        &self,
        _input: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> (ndarray::Array2<f32>, Vec<ndarray::Array2<f32>>) {
        self.glu.compute_gradients(_input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[ndarray::Array2<f32>],
        lr: f32,
    ) -> Result<(), crate::errors::ModelError> {
        self.glu.apply_gradients(param_grads, lr)
    }

    fn weight_norm(&self) -> f32 {
        self.glu.weight_norm()
    }

    fn zero_gradients(&mut self) {
        // RichardsExpert delegates to underlying GLU layer
        // GLU layer handles its own gradient state
    }
}

/// Parameter information for the MoE layer
#[derive(Debug, Clone)]
struct MoeParamInfo {
    /// Total parameter count across all components
    total_params: usize,
}

/// Mixture of Experts layer combining routing and expert execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureOfExperts {
    /// Router for predicting expert routing probabilities
    pub router: ExpertSelector,
    /// Individual expert networks
    pub experts: Vec<RichardsExpert>,
    /// Router configuration
    pub config: ExpertRouterConfig,
    /// Router hidden dimension
    pub router_hidden_dim: usize,
    /// Cached parameter information
    #[serde(skip)]
    param_info: Option<MoeParamInfo>,
    /// Cached routing probabilities for gradient computation
    #[serde(skip)]
    cached_routing_probs: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_input: Option<ndarray::Array2<f32>>,
    /// Cached router input for gradient computation (may include head-conditioning feature)
    #[serde(skip)]
    cached_router_input: Option<ndarray::Array2<f32>>,
    /// Cached active-expert mask for gradient computation (set by forward)
    #[serde(skip)]
    cached_active_expert_mask: Option<Vec<bool>>,
    /// Cached expert outputs for gradient computation
    #[serde(skip)]
    cached_expert_outputs: Option<Vec<ndarray::Array2<f32>>>,

    /// Optional learned adapter controlling expert sparsity (blend top-1 vs top-k).
    #[serde(default)]
    pub k_adapter: Option<LearnedKAdapter>,

    /// Cached alpha used for k-adaptation in the last forward pass.
    #[serde(skip)]
    cached_k_alpha: Option<f32>,
    /// Cached features (entropy_norm, head_activity) for k-adaptation gradients.
    #[serde(skip)]
    cached_k_features: Option<(f32, f32)>,
    /// Cached delta probabilities (p_topk - p_top1) used for d(alpha).
    #[serde(skip)]
    cached_k_delta_probs: Option<ndarray::Array2<f32>>,

    /// Cached per-expert weighted grad buffers for backward() to reduce allocations
    #[serde(skip)]
    cached_weighted_grads: Option<Vec<ndarray::Array2<f32>>>,
}

impl MixtureOfExperts {
    /// Create a new MoE layer
    pub fn new(embedding_dim: usize, router_hidden_dim: usize, config: ExpertRouterConfig) -> Self {
        let mut config = config;

        // Ensure metrics vectors are correctly sized up-front to avoid runtime warnings.
        // (Some call sites construct configs via struct-literals + `..Default::default()`
        // and forget to size `gating.metrics` / `metrics_avg_routing_prob`.)
        config.gating.metrics.resize(config.num_experts);
        config
            .metrics_avg_routing_prob
            .resize(config.num_experts, 0.0);

        let use_learned_k_adaptation = config.use_learned_k_adaptation;
        let router_input_dim = embedding_dim + if config.use_head_conditioning { 1 } else { 0 };
        let router = ExpertSelector::new(router_input_dim, router_hidden_dim, config.num_experts);

        let experts = (0..config.num_experts)
            .map(|_| RichardsExpert::new(embedding_dim, config.expert_hidden_dim))
            .collect();

        Self {
            router,
            experts,
            config,
            router_hidden_dim,
            param_info: None,
            cached_routing_probs: None,
            cached_input: None,
            cached_router_input: None,
            cached_active_expert_mask: None,
            cached_expert_outputs: None,
            k_adapter: if use_learned_k_adaptation {
                Some(LearnedKAdapter::new())
            } else {
                None
            },
            cached_k_alpha: None,
            cached_k_features: None,
            cached_k_delta_probs: None,
            cached_weighted_grads: None,
        }
    }

    /// Forward pass: predict routing → all experts process → weighted sum
    pub fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.forward_with_head_activity(input, None)
    }

    /// Forward pass with optional Mixture-of-Heads activity signal.
    ///
    /// If head conditioning is enabled in the router config, a single scalar feature
    /// (head_activity in [0,1]) is appended to the router input per token.
    ///
    /// Additionally, the number of *active experts* can be coupled to head activity by
    /// smoothly scaling the configured top-k (gating.num_active) into an *effective* k.
    ///
    /// Important: we avoid a hard `round()` threshold (which causes a cliff around
    /// h≈0.5 for base_k=2) by interpolating between top-k masks for k=floor(kf) and
    /// k=ceil(kf). This keeps behavior adaptive while preventing brittle collapse.
    pub fn forward_with_head_activity(
        &mut self,
        input: &ndarray::Array2<f32>,
        head_activity: Option<f32>,
    ) -> ndarray::Array2<f32> {
        self.forward_with_head_features(input, head_activity, None)
    }

    /// Forward pass with optional Mixture-of-Heads activity signal (scalar + per-head vector).
    ///
    /// - If `use_head_conditioning` is enabled: appends scalar `head_activity` to the router input.
    /// - If `head_activity_vec` is provided: applies a learned per-head → per-expert logit bias
    ///   inside the router (see `ExpertSelector::predict_with_head_activity`).
    pub fn forward_with_head_features(
        &mut self,
        input: &ndarray::Array2<f32>,
        head_activity: Option<f32>,
        head_activity_vec: Option<&[f32]>,
    ) -> ndarray::Array2<f32> {
        // Cache input for gradient computation
        self.cached_input = Some(input.to_owned());

        // Build (and reuse) cached router input buffer for gradient computation.
        // This avoids allocating a new router-input matrix every forward.
        let n = input.nrows();
        let d = input.ncols();
        let cond = if self.config.use_head_conditioning {
            1
        } else {
            0
        };
        let desired_rows = n;
        let desired_cols = d + cond;

        let mut router_in = self
            .cached_router_input
            .take()
            .unwrap_or_else(|| ndarray::Array2::<f32>::zeros((desired_rows, desired_cols)));
        if router_in.nrows() != desired_rows || router_in.ncols() != desired_cols {
            router_in = ndarray::Array2::<f32>::zeros((desired_rows, desired_cols));
        }
        if cond == 1 {
            let h = head_activity.unwrap_or(0.0);
            let h = if h.is_finite() { h } else { 0.0 };
            let h = h.clamp(0.0, 1.0);
            router_in.slice_mut(ndarray::s![.., 0..d]).assign(input);
            for i in 0..n {
                router_in[[i, d]] = h;
            }
        } else {
            router_in.assign(input);
        }
        self.cached_router_input = Some(router_in);

        // Router predicts routing probabilities for all tokens (optionally head-conditioned).
        let router_in = self
            .cached_router_input
            .as_ref()
            .expect("router input must be cached");

        let routing_probs_full = if head_activity_vec.is_some() {
            self.router
                .predict_with_head_activity(&router_in.view(), head_activity_vec)
        } else {
            self.router.predict(&router_in.view())
        };

        // Base top-k for sparse masking of routing probabilities.
        let base_k = self
            .config
            .gating
            .num_active
            .max(1)
            .min(self.config.num_experts);

        // Sparse top-k masking + renormalization computed directly from router logits.
        // For head-activity coupling, interpolate between k=floor(kf) and k=ceil(kf)
        // to avoid a hard regime change.
        let cached_logits = self
            .router
            .cached_logits
            .as_ref()
            .expect("router logits must be cached by predict()");

        // Clear learned-k caches by default; we'll fill them only when learned adaptation is used.
        self.cached_k_alpha = None;
        self.cached_k_features = None;
        self.cached_k_delta_probs = None;

        let (masked_probs, active_mask) = if self.config.use_learned_k_adaptation
            && head_activity.is_some()
        {
            if self.k_adapter.is_none() {
                self.k_adapter = Some(LearnedKAdapter::new());
            }
            let h = head_activity.unwrap_or(0.0);
            let h = if h.is_finite() { h } else { 0.0 };
            let h = h.clamp(0.0, 1.0);

            // Mean routing entropy across tokens (from full softmax probs), normalized by log(E).
            let n_tok = routing_probs_full.nrows().max(1) as f32;
            let denom = (self.config.num_experts.max(2) as f32).ln();
            let mut entropy_sum = 0.0f32;
            for t in 0..routing_probs_full.nrows() {
                let mut ent = 0.0f32;
                for e in 0..self.config.num_experts {
                    let mut p = routing_probs_full[[t, e]];
                    p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                    if p > 0.0 {
                        ent -= p * p.ln();
                    }
                }
                entropy_sum += ent;
            }
            let entropy = entropy_sum / n_tok;
            let entropy_norm = if denom.is_finite() && denom > 0.0 {
                (entropy / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let alpha = self
                .k_adapter
                .as_ref()
                .expect("k_adapter must exist")
                .alpha(entropy_norm, h);

            // Blend between top-1 and configured top-k.
            let (p_top1, m_top1) = masked_top_k_from_logits_and_active(cached_logits, 1);
            let (p_topk, m_topk) = masked_top_k_from_logits_and_active(cached_logits, base_k);

            let mut p = p_top1.clone();
            p.zip_mut_with(&p_topk, |a, &b| {
                *a = (1.0 - alpha) * (*a) + alpha * b;
            });

            let mut delta = p_topk;
            delta.zip_mut_with(&p_top1, |a, &b| {
                *a = *a - b;
            });

            let mut m = m_top1;
            for i in 0..m.len().min(m_topk.len()) {
                m[i] = m[i] || m_topk[i];
            }

            self.cached_k_alpha = Some(alpha);
            self.cached_k_features = Some((entropy_norm, h));
            self.cached_k_delta_probs = Some(delta);

            (p, m)
        } else if let Some(h) = head_activity {
            // Heuristic smooth coupling (no cliff): interpolate between k=floor(kf) and k=ceil(kf).
            let h = if h.is_finite() { h } else { 0.0 };
            let h = h.clamp(0.0, 1.0);
            let kf = 1.0 + (base_k.saturating_sub(1) as f32) * h;

            let k_low = (kf.floor() as usize).clamp(1, base_k);
            let k_high = (kf.ceil() as usize).clamp(1, base_k);
            let alpha = (kf - k_low as f32).clamp(0.0, 1.0);

            if k_low == k_high || alpha == 0.0 {
                masked_top_k_from_logits_and_active(cached_logits, k_low)
            } else {
                let (p_low, m_low) = masked_top_k_from_logits_and_active(cached_logits, k_low);
                let (p_high, m_high) = masked_top_k_from_logits_and_active(cached_logits, k_high);

                // Blend probabilities; both are already per-row renormalized.
                let mut p = p_low;
                p.zip_mut_with(&p_high, |a, &b| {
                    *a = (1.0 - alpha) * (*a) + alpha * b;
                });

                // Union the expert-activity masks so we compute any expert needed by either path.
                let mut m = m_low;
                for i in 0..m.len().min(m_high.len()) {
                    m[i] = m[i] || m_high[i];
                }
                (p, m)
            }
        } else {
            masked_top_k_from_logits_and_active(cached_logits, base_k)
        };

        self.cached_active_expert_mask = Some(active_mask);
        self.cached_routing_probs = Some(masked_probs);

        let masked_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("masked routing probabilities must be cached");

        // Update routing metrics for training based on *active* routing.
        self.config.update_metrics(&masked_probs.view());

        let active_experts: Vec<usize> = self
            .cached_active_expert_mask
            .as_ref()
            .expect("active expert mask must be cached")
            .iter()
            .enumerate()
            .filter_map(|(i, &a)| if a { Some(i) } else { None })
            .collect();

        // Compute only active experts; keep cache length = num_experts.
        // Reuse cached buffers when possible and avoid cloning large expert outputs.
        let mut expert_outputs = self.cached_expert_outputs.take().unwrap_or_default();
        let desired_len = self.config.num_experts;
        if expert_outputs.len() != desired_len {
            expert_outputs = vec![ndarray::Array2::<f32>::zeros(input.raw_dim()); desired_len];
        } else if !expert_outputs.is_empty() && expert_outputs[0].raw_dim() != input.raw_dim() {
            for out in &mut expert_outputs {
                *out = ndarray::Array2::<f32>::zeros(input.raw_dim());
            }
        }

        for &e in &active_experts {
            expert_outputs[e] = self.experts[e].forward(input);
        }
        self.cached_expert_outputs = Some(expert_outputs);
        let expert_outputs = self
            .cached_expert_outputs
            .as_ref()
            .expect("expert outputs must be cached");

        // Weighted sum of expert outputs using masked routing probabilities.
        let mut output = ndarray::Array2::zeros(input.raw_dim());
        for e in 0..self.config.num_experts {
            let routing_col = masked_probs.column(e);
            let expert_out = &expert_outputs[e];
            output
                .outer_iter_mut()
                .zip(expert_out.outer_iter())
                .zip(routing_col.iter())
                .for_each(|((mut out_row, expert_row), &w)| {
                    let ws = if w.is_finite() { w } else { 0.0 };
                    if ws != 0.0 {
                        out_row.scaled_add(ws, &expert_row);
                    }
                });
        }

        output
    }

    /// Get parameter information for the MoE layer
    fn get_param_info(&mut self) -> &MoeParamInfo {
        if self.param_info.is_none() {
            // Get router parameter info (avoid clone by taking ownership)
            let router_info = (*self.router.get_param_info()).clone();

            // Get expert parameter info using iterator chains
            let expert_infos = self
                .experts
                .iter_mut()
                .map(|expert| (*expert.get_param_info()).clone())
                .collect::<Vec<_>>();

            // Calculate total parameters using iterator chains
            let total_params = router_info.total_params
                + expert_infos
                    .iter()
                    .map(|info| info.total_params)
                    .sum::<usize>();

            let total_params = total_params
                + self
                    .k_adapter
                    .as_ref()
                    .map(|a| a.w.len() + a.b.len())
                    .unwrap_or(0);

            self.param_info = Some(MoeParamInfo { total_params });
        }

        self.param_info.as_ref().unwrap()
    }

    /// Get total parameters in the MoE layer
    pub fn total_parameters(&mut self) -> usize {
        self.get_param_info().total_params
    }
}

impl Layer for MixtureOfExperts {
    fn layer_type(&self) -> &str {
        "MixtureOfExperts"
    }

    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.forward(input)
    }

    fn backward(&mut self, grads: &ndarray::Array2<f32>, lr: f32) -> ndarray::Array2<f32> {
        // Backward: route gradients to experts weighted by routing probabilities
        let routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("forward must be called before backward");

        let mut total_grad_input = ndarray::Array2::zeros(grads.raw_dim());

        let active_mask = self
            .cached_active_expert_mask
            .as_ref()
            .map(|m| m.as_slice());

        // Reuse weighted gradient buffers per expert.
        let mut weighted_buffers = self.cached_weighted_grads.take().unwrap_or_default();
        if weighted_buffers.len() != self.experts.len() {
            weighted_buffers =
                vec![ndarray::Array2::<f32>::zeros(grads.raw_dim()); self.experts.len()];
        } else if !weighted_buffers.is_empty() && weighted_buffers[0].raw_dim() != grads.raw_dim() {
            for b in &mut weighted_buffers {
                *b = ndarray::Array2::<f32>::zeros(grads.raw_dim());
            }
        }

        for (expert_idx, expert) in self.experts.iter_mut().enumerate() {
            if let Some(m) = active_mask {
                if expert_idx < m.len() && !m[expert_idx] {
                    continue;
                }
            }

            let routing_col = routing_probs.column(expert_idx);
            let weighted_grads_2d = &mut weighted_buffers[expert_idx];
            weighted_grads_2d.fill(0.0);

            for (token_idx, (grad_row, &weight)) in
                grads.outer_iter().zip(routing_col.iter()).enumerate()
            {
                let w = if weight.is_finite() { weight } else { 0.0 };
                if w == 0.0 {
                    continue;
                }

                let mut dst = weighted_grads_2d.row_mut(token_idx);
                for (d, &g) in dst.iter_mut().zip(grad_row.iter()) {
                    let g = if g.is_finite() { g } else { 0.0 };
                    *d = g * w;
                }
            }

            // Get expert input gradients
            let expert_grad_input = expert.backward(weighted_grads_2d, lr);

            // Weight input gradients back by routing probabilities
            for ((grad_row, &weight), mut total_row) in expert_grad_input
                .outer_iter()
                .zip(routing_col.iter())
                .zip(total_grad_input.outer_iter_mut())
            {
                let w = if weight.is_finite() { weight } else { 0.0 };
                if w != 0.0 {
                    total_row.scaled_add(w, &grad_row);
                }
            }
        }

        self.cached_weighted_grads = Some(weighted_buffers);

        total_grad_input
    }

    fn parameters(&self) -> usize {
        // We need to use a mutable reference, so we can't implement this directly
        // This is a limitation - we'll need to compute parameters differently
        // For now, return a cached value or compute without mutation
        if let Some(ref info) = self.param_info {
            info.total_params
        } else {
            // Fallback: compute without caching
            let mut total = 0;
            total += self.router.weights1.len() + self.router.weights2.len();
            total += self.router.bias1.len() + self.router.bias2.len();
            total += self.router.norm.parameters();
            total += self.router.activation.parameters();

            total += self
                .experts
                .iter()
                .map(|expert| expert.glu.parameters())
                .sum::<usize>();

            total += self
                .k_adapter
                .as_ref()
                .map(|a| a.w.len() + a.b.len())
                .unwrap_or(0);

            total
        }
    }

    fn compute_gradients(
        &self,
        _input: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> (ndarray::Array2<f32>, Vec<ndarray::Array2<f32>>) {
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let cached_router_input = self
            .cached_router_input
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let cached_routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let cached_expert_outputs = self
            .cached_expert_outputs
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let active_mask = self
            .cached_active_expert_mask
            .as_ref()
            .map(|m| m.as_slice());

        // 1. Route gradients to experts weighted by (post-mask) routing probabilities.
        // Only build grads for experts that were active for at least one token.
        let mut expert_output_grads =
            vec![ndarray::Array2::zeros(output_grads.raw_dim()); self.config.num_experts];
        for expert_idx in 0..self.config.num_experts {
            if let Some(m) = active_mask {
                if expert_idx < m.len() && !m[expert_idx] {
                    continue;
                }
            }
            for token_idx in 0..output_grads.nrows() {
                let mut w = cached_routing_probs[[token_idx, expert_idx]];
                w = if w.is_finite() { w } else { 0.0 };
                if w == 0.0 {
                    continue;
                }

                let src_row = output_grads.row(token_idx);
                let mut dst_row = expert_output_grads[expert_idx].row_mut(token_idx);
                for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                    let src = if src.is_finite() { src } else { 0.0 };
                    *dst = src * w;
                }
            }
        }

        // 2. Compute gradients for each expert
        let mut all_param_grads = Vec::new();
        let mut grad_input = ndarray::Array2::zeros(cached_input.raw_dim());

        let zero_expert_grads = |expert: &RichardsExpert| -> Vec<ndarray::Array2<f32>> {
            let act_len = expert.glu.richards_activation.weights().len();
            vec![
                ndarray::Array2::<f32>::zeros(expert.glu.w1.raw_dim()),
                ndarray::Array2::<f32>::zeros(expert.glu.w2.raw_dim()),
                ndarray::Array2::<f32>::zeros(expert.glu.w_out.raw_dim()),
                ndarray::Array2::<f32>::zeros((1, act_len)),
                ndarray::Array2::<f32>::zeros((1, 1)),
                ndarray::Array2::<f32>::zeros((1, 1)),
                ndarray::Array2::<f32>::zeros((1, 1)),
                ndarray::Array2::<f32>::zeros((1, 1)),
            ]
        };

        for (expert_idx, expert) in self.experts.iter().enumerate() {
            if let Some(m) = active_mask {
                if expert_idx < m.len() && !m[expert_idx] {
                    all_param_grads.extend(zero_expert_grads(expert));
                    continue;
                }
            }
            let expert_grads = &expert_output_grads[expert_idx];
            let (expert_input_grad, expert_param_grads) =
                expert.compute_gradients(cached_input, expert_grads);

            // Weight input gradients by routing probabilities
            for token_idx in 0..expert_input_grad.nrows() {
                let routing_weight = cached_routing_probs[[token_idx, expert_idx]];
                grad_input
                    .row_mut(token_idx)
                    .scaled_add(routing_weight, &expert_input_grad.row(token_idx));
            }

            all_param_grads.extend(expert_param_grads);
        }

        // 3. Compute router gradients from the main loss (only for experts with non-zero
        // routing weight after sparse masking).

        // Optional learned-k adapter gradients.
        // If y = (1-a)*p_top1 + a*p_topk, then dL/da = sum_{t,e} dL/dy[t,e] * (p_topk -
        // p_top1)[t,e]. Here dL/dy[t,e] = <output_grads[t], expert_output[t,e]>.
        let adapter_grads = if self.config.use_learned_k_adaptation {
            match (
                self.k_adapter.as_ref(),
                self.cached_k_alpha,
                self.cached_k_features,
                self.cached_k_delta_probs.as_ref(),
            ) {
                (Some(_), Some(alpha), Some((entropy_norm, head_activity)), Some(delta)) => {
                    let mut d_alpha = 0.0f32;
                    for t in 0..output_grads.nrows() {
                        let token_output_grad = output_grads.row(t);
                        for e in 0..self.config.num_experts {
                            let dp = delta[[t, e]];
                            let dp = if dp.is_finite() { dp } else { 0.0 };
                            if dp == 0.0 {
                                continue;
                            }

                            let expert_output = cached_expert_outputs[e].row(t);
                            let g = token_output_grad
                                .iter()
                                .zip(expert_output.iter())
                                .map(|(&g, &o)| {
                                    let g = if g.is_finite() { g } else { 0.0 };
                                    let o = if o.is_finite() { o } else { 0.0 };
                                    g * o
                                })
                                .sum::<f32>();
                            d_alpha += g * dp;
                        }
                    }

                    let dz = d_alpha * alpha * (1.0 - alpha);
                    let mut g_w = ndarray::Array2::<f32>::zeros((2, 1));
                    g_w[[0, 0]] = dz * entropy_norm;
                    g_w[[1, 0]] = dz * head_activity;
                    let mut g_b = ndarray::Array2::<f32>::zeros((1, 1));
                    g_b[[0, 0]] = dz;
                    Some((g_w, g_b))
                }
                _ => None,
            }
        } else {
            None
        };

        // Compute router gradients manually using cached activations
        // Use cached router activations from the predict() call
        let cached_activated = self
            .router
            .cached_activated
            .as_ref()
            .expect("Router predict must be called before MoE gradient computation");
        let cached_hidden = self
            .router
            .cached_hidden
            .as_ref()
            .expect("Router predict must be called before MoE gradient computation");
        let cached_normalized = self
            .router
            .cached_normalized
            .as_ref()
            .expect("Router predict must be called before MoE gradient computation");
        let _cached_logits = self
            .router
            .cached_logits
            .as_ref()
            .expect("Router predict must be called before MoE gradient computation");

        // Compute softmax gradients efficiently (vector-Jacobian product).
        // If y = softmax(z) and g = dL/dy, then dL/dz = y * (g - <g, y>).
        let routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("routing probs must be cached");
        let mut d_logits = ndarray::Array2::zeros(routing_probs.raw_dim());

        for token_idx in 0..routing_probs.nrows() {
            let token_output_grad = output_grads.row(token_idx);
            let mut dot_gy = 0.0f32;
            let mut active_pairs: Vec<(usize, f32, f32)> = Vec::new();

            for expert_idx in 0..self.config.num_experts {
                let y = routing_probs[[token_idx, expert_idx]];
                let y = if y.is_finite() { y } else { 0.0 };
                if y == 0.0 {
                    continue;
                }
                let expert_output = cached_expert_outputs[expert_idx].row(token_idx);
                let g = token_output_grad
                    .iter()
                    .zip(expert_output.iter())
                    .map(|(&g, &o)| {
                        let g = if g.is_finite() { g } else { 0.0 };
                        let o = if o.is_finite() { o } else { 0.0 };
                        g * o
                    })
                    .sum::<f32>();
                active_pairs.push((expert_idx, g, y));
                dot_gy += g * y;
            }

            for (expert_idx, g, y) in active_pairs {
                d_logits[[token_idx, expert_idx]] = y * (g - dot_gy);
            }
        }

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_logits);
        let grad_bias2 = d_logits.sum_axis(ndarray::Axis(0));

        // Optional learned per-head -> per-expert conditioning gradients.
        // If logits were biased by head_activity · W_head_to_expert, then:
        // dL/dW[h,e] = head_activity[h] * sum_t d_logits[t,e]
        let grad_head_to_expert = match (
            self.router.cached_head_activity_vec.as_ref(),
            self.router.head_to_expert.as_ref(),
        ) {
            (Some(head_activity_vec), Some(w))
                if head_activity_vec.len() == w.nrows() && w.ncols() == self.config.num_experts =>
            {
                let mut g = ndarray::Array2::<f32>::zeros(w.raw_dim());
                for h in 0..head_activity_vec.len() {
                    let a = head_activity_vec[h];
                    let a = if a.is_finite() { a.max(0.0) } else { 0.0 };
                    if a == 0.0 {
                        continue;
                    }
                    for e in 0..self.config.num_experts {
                        g[[h, e]] = a * grad_bias2[e];
                    }
                }
                Some(g)
            }
            _ => None,
        };

        // Gradient w.r.t. activated (before second layer)
        let d_activated = d_logits.dot(&self.router.weights2.t());

        // Gradient through Richards activation (replacing ReLU)
        let normalized_f64 = cached_normalized.mapv(|x| x as f64);
        let d_activated_f64 = d_activated.mapv(|x| x as f64);
        let activation_grad_f64 = self
            .router
            .activation
            .backward_matrix(&normalized_f64, &d_activated_f64);
        let d_normalized = activation_grad_f64.mapv(|x| x as f32);

        // Gradient through Richards normalization
        let (d_hidden, _) = self
            .router
            .norm
            .compute_gradients(cached_hidden, &d_normalized);

        // Propagate router gradients back into the MoE input.
        // router_input = [input, head_activity?]; only the first `input_dim` columns map to
        // `input`.
        let d_router_in = d_hidden.dot(&self.router.weights1.t());
        let input_dim = cached_input.ncols();
        let router_in_dim = cached_router_input.ncols();
        let take_cols = input_dim.min(router_in_dim);
        if take_cols > 0 {
            for t in 0..grad_input.nrows() {
                for j in 0..take_cols {
                    grad_input[[t, j]] += d_router_in[[t, j]];
                }
            }
        }

        // First layer gradients
        let grad_weights1 = cached_router_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));

        // Activation parameter gradients
        let normalized_f32 = normalized_f64.mapv(|x| x as f32);
        let d_activated_f32 = d_activated_f64.mapv(|x| x as f32);
        let (_, activation_param_grads) = self
            .router
            .activation
            .compute_gradients(&normalized_f32, &d_activated_f32);

        let mut router_grads = vec![
            grad_weights1,
            grad_bias1.insert_axis(ndarray::Axis(0)),
            grad_weights2,
            grad_bias2.insert_axis(ndarray::Axis(0)),
        ];

        if let Some(g) = grad_head_to_expert {
            router_grads.push(g);
        }
        router_grads.extend(activation_param_grads);
        all_param_grads.extend(router_grads);

        if let Some((g_w, g_b)) = adapter_grads {
            all_param_grads.push(g_w);
            all_param_grads.push(g_b);
        }

        (grad_input, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[ndarray::Array2<f32>],
        lr: f32,
    ) -> Result<(), crate::errors::ModelError> {
        let mut grad_idx = 0;

        // Apply gradients to each expert
        for expert in &mut self.experts {
            // RichardsGlu always has 8 parameters: w1, w2, w_out, richards_activation, gate (4
            // params)
            let num_expert_params = 8;

            if grad_idx + num_expert_params > param_grads.len() {
                return Err(crate::errors::ModelError::GradientError {
                    message: format!(
                        "Not enough gradients for experts: expected at least {}, got {}",
                        grad_idx + num_expert_params,
                        param_grads.len()
                    ),
                });
            }

            let expert_grads = &param_grads[grad_idx..grad_idx + num_expert_params];
            expert.apply_gradients(expert_grads, lr)?;
            grad_idx += num_expert_params;
        }

        // Apply router gradients (weights1, bias1, weights2, bias2, [head_to_expert],
        // activation_params) Base: 4 grads (w1,b1,w2,b2) + 4 activation grads.
        let mut router_grad_idx = grad_idx;
        if router_grad_idx + 8 > param_grads.len() {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "Not enough gradients for router: expected at least {}, got {}",
                    router_grad_idx + 8,
                    param_grads.len()
                ),
            });
        }

        let g_w1 = &param_grads[router_grad_idx];
        if g_w1.raw_dim() == self.router.weights1.raw_dim() {
            self.router.weights1.scaled_add(-lr, g_w1);
        } else if self.config.use_head_conditioning
            && g_w1.ncols() == self.router.weights1.ncols()
            && g_w1.nrows() + 1 == self.router.weights1.nrows()
        {
            // If the conditioning feature was appended, but the gradient was computed
            // without it (older caches/paths), pad the extra row with zeros.
            let mut padded = ndarray::Array2::<f32>::zeros(self.router.weights1.raw_dim());
            padded
                .slice_mut(ndarray::s![0..g_w1.nrows(), ..])
                .assign(g_w1);
            self.router.weights1.scaled_add(-lr, &padded);
        } else {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "Router weights1 gradient shape mismatch: expected {:?}, got {:?}",
                    self.router.weights1.raw_dim(),
                    g_w1.raw_dim()
                ),
            });
        }
        self.router
            .bias1
            .scaled_add(-lr, &param_grads[router_grad_idx + 1].row(0));
        self.router
            .weights2
            .scaled_add(-lr, &param_grads[router_grad_idx + 2]);
        self.router
            .bias2
            .scaled_add(-lr, &param_grads[router_grad_idx + 3].row(0));

        router_grad_idx += 4;

        // Optional head_to_expert gradient (if present in param_grads and router has the param)
        if let Some(w) = self.router.head_to_expert.as_mut() {
            if router_grad_idx < param_grads.len()
                && param_grads[router_grad_idx].raw_dim() == w.raw_dim()
            {
                w.scaled_add(-lr, &param_grads[router_grad_idx]);
                router_grad_idx += 1;
            }
        }

        // Apply activation parameter gradients (4 separate arrays: nu, k, m, temperature)
        if router_grad_idx + 4 > param_grads.len() {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "Not enough gradients for router activation params: expected at least {}, got {}",
                    router_grad_idx + 4,
                    param_grads.len()
                ),
            });
        }
        let activation_grads = &param_grads[router_grad_idx..router_grad_idx + 4];
        let _ = self.router.activation.apply_gradients(activation_grads, lr);
        router_grad_idx += 4;

        grad_idx = router_grad_idx;

        // Optional learned-k adapter (2 grads: w,b).
        if let Some(adapter) = self.k_adapter.as_mut() {
            if grad_idx + 2 <= param_grads.len()
                && param_grads[grad_idx].raw_dim() == adapter.w.raw_dim()
                && param_grads[grad_idx + 1].raw_dim() == adapter.b.raw_dim()
            {
                adapter.w.scaled_add(-lr, &param_grads[grad_idx]);
                adapter.b.scaled_add(-lr, &param_grads[grad_idx + 1]);
            }
        }

        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let router_norm = self
            .router
            .weights1
            .iter()
            .map(|&w| w * w)
            .sum::<f32>()
            .sqrt()
            + self
                .router
                .weights2
                .iter()
                .map(|&w| w * w)
                .sum::<f32>()
                .sqrt()
            + self
                .router
                .head_to_expert
                .as_ref()
                .map(|w| w.iter().map(|&x| x * x).sum::<f32>().sqrt())
                .unwrap_or(0.0)
            + self.router.activation.weight_norm();

        let expert_norm = self.experts.iter().map(|e| e.weight_norm()).sum::<f32>();

        router_norm + expert_norm
    }

    fn zero_gradients(&mut self) {
        // MixtureOfExperts doesn't maintain internal gradient state beyond cached routing
        // Reset cached routing decisions and expert outputs
        self.cached_routing_probs = None;
        self.cached_expert_outputs = None;
        self.cached_weighted_grads = None;
        self.cached_k_alpha = None;
        self.cached_k_features = None;
        self.cached_k_delta_probs = None;
    }
}

fn masked_top_k_from_logits_and_active(
    logits: &ndarray::Array2<f32>,
    k: usize,
) -> (ndarray::Array2<f32>, Vec<bool>) {
    let n_tokens = logits.nrows();
    let n_experts = logits.ncols();

    if n_tokens == 0 || n_experts == 0 {
        return (
            ndarray::Array2::<f32>::zeros((n_tokens, n_experts)),
            vec![false; n_experts],
        );
    }

    let k = k.clamp(1, n_experts);
    let mut masked = ndarray::Array2::<f32>::zeros(logits.raw_dim());
    let mut active = vec![false; n_experts];

    for (token_idx, row) in logits.outer_iter().enumerate() {
        // Track top-k by logit value (non-finite treated as -inf).
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for (idx, &v) in row.iter().enumerate() {
            let score = if v.is_finite() { v } else { f32::NEG_INFINITY };
            if best.len() < k {
                best.push((score, idx));
                continue;
            }

            // Find current minimum in best.
            let mut min_pos = 0usize;
            let mut min_score = best[0].0;
            for (p, (s, _)) in best.iter().enumerate().skip(1) {
                if *s < min_score {
                    min_score = *s;
                    min_pos = p;
                }
            }

            if score > min_score {
                best[min_pos] = (score, idx);
            }
        }

        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Stable softmax over selected logits (log-sum-exp).
        let mut max_val = f32::NEG_INFINITY;
        let mut any_finite = false;
        for &(s, _) in &best {
            if s.is_finite() {
                any_finite = true;
                max_val = max_val.max(s);
            }
        }
        if !any_finite {
            // Fallback: uniform over selected indices.
            let w = 1.0 / best.len() as f32;
            for &(_s, idx) in &best {
                active[idx] = true;
                masked[[token_idx, idx]] = w;
            }
            continue;
        }

        let mut exp_sum: f64 = 0.0;
        for &(s, _) in &best {
            if s.is_finite() {
                exp_sum += crate::pade::PadeExp::exp((s - max_val) as f64);
            }
        }

        if exp_sum <= 0.0 || !exp_sum.is_finite() {
            // Degenerate: put all mass on the best element.
            let idx = best[0].1;
            active[idx] = true;
            masked[[token_idx, idx]] = 1.0;
            continue;
        }

        let inv_sum = 1.0 / exp_sum;
        for &(s, idx) in &best {
            active[idx] = true;
            if s.is_finite() {
                masked[[token_idx, idx]] =
                    (crate::pade::PadeExp::exp((s - max_val) as f64) * inv_sum) as f32;
            }
        }
    }

    (masked, active)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_router_config_default() {
        let config = ExpertRouterConfig::default();
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.gating.num_active, 2);
        assert_eq!(config.expert_hidden_dim, 64);
        assert_eq!(config.gating.load_balance_weight, 0.0);
    }

    #[test]
    fn test_expert_router_config_from_strategy() {
        let router = ExpertRouter::LearnedMoE {
            num_experts: 8,
            num_active_experts: 3,
            expert_hidden_dim: 32,
            load_balance_weight: 0.1,
            sparsity_weight: 0.01,
            diversity_weight: 0.005,
            use_head_conditioning: true,
            use_learned_k_adaptation: false,
        };

        let config = ExpertRouterConfig::from_router(&router);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.gating.num_active, 3);
        assert_eq!(config.expert_hidden_dim, 32);
        assert_eq!(config.gating.load_balance_weight, 0.1);
        assert!(config.use_head_conditioning);
    }

    #[test]
    fn test_moe_forward_with_head_conditioning() {
        let mut config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 16,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 3,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };
        config.use_head_conditioning = true;

        let mut moe = MixtureOfExperts::new(32, 8, config);
        let input = ndarray::Array2::<f32>::from_shape_vec((5, 32), vec![0.1; 160]).unwrap();

        let out_low = moe.forward_with_head_activity(&input, Some(0.1));
        let out_high = moe.forward_with_head_activity(&input, Some(0.9));
        assert_eq!(out_low.shape(), input.shape());
        assert_eq!(out_high.shape(), input.shape());
    }

    #[test]
    fn test_expert_selector() {
        let mut selector = ExpertSelector::new(64, 32, 4); // embed_dim, router_hidden, num_experts
        let input = ndarray::Array2::<f32>::from_shape_vec((4, 64), vec![0.1; 256]).unwrap();

        let routing_probs = selector.predict(&input.view());
        assert_eq!(routing_probs.shape(), &[4, 4]);

        // Check probabilities sum to 1 per token
        for row in routing_probs.outer_iter() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check all probabilities are non-negative
        for &prob in routing_probs.iter() {
            assert!(prob >= 0.0);
        }
    }

    #[test]
    fn test_expert_selection() {
        let selector = ExpertSelector::new(64, 32, 4);
        let routing_probs = ndarray::Array2::from_shape_vec(
            (2, 4),
            vec![
                0.1, 0.7, 0.1, 0.1, // Token 1: expert 1 has highest prob
                0.2, 0.2, 0.5, 0.1, // Token 2: expert 2 has highest prob
            ],
        )
        .unwrap();

        let selections = selector.select_experts(&routing_probs, 2);

        assert_eq!(selections.len(), 2);
        assert_eq!(selections[0].len(), 2); // Top 2 for token 1
        assert_eq!(selections[1].len(), 2); // Top 2 for token 2

        // Expert 1 should be in top 2 for token 1
        assert!(selections[0].contains(&1));
        // Expert 2 should be in top 2 for token 2
        assert!(selections[1].contains(&2));
    }

    #[test]
    fn test_load_balance_loss() {
        let mut config = ExpertRouterConfig::default();
        // Simulate unbalanced routing: expert 0 gets all tokens, others get none
        config.gating.metrics.resize(4);
        config.gating.metrics.token_count_per_component = vec![100, 0, 0, 0];
        config.gating.metrics.total_decisions = 100;

        let loss = config.compute_load_balance_loss();
        assert!(loss > 0.0); // Should have high loss due to imbalance
    }

    #[test]
    fn test_richards_expert() {
        let mut expert = RichardsExpert::new(64, 32);
        let input = ndarray::Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();

        let output = expert.forward(&input);
        assert_eq!(output.shape(), input.shape()); // Residual connection preserves shape
    }

    #[test]
    fn test_moe_forward() {
        let config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 32,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 2,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(64, 16, config);
        let input = ndarray::Array2::<f32>::from_shape_vec((3, 64), vec![0.1; 192]).unwrap();

        let output = moe.forward(&input);

        // Output should have same shape as input
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_moe_gradient_computation() {
        let config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 32,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 2,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(64, 16, config);
        let input = ndarray::Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();

        // First do forward pass to cache routing decisions
        let _output = moe.forward(&input);

        // Now compute gradients
        let output_grads = ndarray::Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();
        let (grad_input, param_grads) = moe.compute_gradients(&input, &output_grads);

        // Check that gradients are computed (not empty)
        assert!(
            !param_grads.is_empty(),
            "Parameter gradients should not be empty"
        );

        // Check that input gradients have correct shape
        assert_eq!(grad_input.shape(), input.shape());

        // Verify that router gradients are included (8 matrices: weights1, bias1, weights2, bias2,
        // activation_nu, activation_k, activation_m, activation_temperature)
        // Expert gradients come first, then router gradients
        let expected_router_grad_start = moe.experts.len() * 5; // 5 parameter groups per expert (w1, w2, w_out, richards_activation, gate_parameters)
        assert!(
            param_grads.len() >= expected_router_grad_start + 8,
            "Should have gradients for all experts plus 8 router matrices, got {}",
            param_grads.len()
        );
    }

    #[test]
    fn test_moe_apply_gradients() {
        let config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 32,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 2,
                load_balance_weight: 0.01,
                sparsity_weight: 0.001,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut moe = MixtureOfExperts::new(64, 16, config);
        let input = ndarray::Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();

        // Do forward and backward passes
        let _output = moe.forward(&input);
        let output_grads = ndarray::Array2::<f32>::from_shape_vec((2, 64), vec![0.1; 128]).unwrap();
        let (_grad_input, param_grads) = moe.compute_gradients(&input, &output_grads);

        // Store original weights for comparison
        let original_router_w1 = moe.router.weights1.clone();
        let original_expert_w1s = moe
            .experts
            .iter()
            .map(|e| e.glu.w1.clone())
            .collect::<Vec<_>>();

        // Apply gradients
        moe.apply_gradients(&param_grads, 0.01)
            .expect("Apply gradients should succeed");

        // Check that weights were updated
        assert_ne!(
            moe.router.weights1, original_router_w1,
            "Router weights should be updated"
        );
        let any_expert_updated = moe
            .experts
            .iter()
            .zip(original_expert_w1s.iter())
            .any(|(e, w1)| e.glu.w1 != *w1);
        assert!(any_expert_updated, "At least one expert should be updated");
    }
}
