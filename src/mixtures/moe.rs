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
    richards::RichardsCurve,
    rng::get_rng,
};

#[inline]
fn default_true() -> bool {
    true
}

type RouterParamGrads = (
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    ndarray::Array2<f32>,
    ndarray::Array1<f32>,
    Vec<f64>,
);

type RouterParamShapes<'a> = (&'a [(usize, usize)], &'a [usize], usize, usize, usize);

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

        /// Routing mode (token-choice vs expert-choice).
        #[serde(default)]
        routing_mode: ExpertRoutingMode,

        /// Capacity factor used when routing mode applies capacity (Switch-style).
        ///
        /// Typical values: 1.0–2.0. 0.0 disables capacity limiting.
        #[serde(default)]
        capacity_factor: f32,

        /// Minimum capacity per expert (guards tiny batches).
        #[serde(default)]
        min_expert_capacity: usize,

        /// Renormalize per-token routing probabilities after capacity drops.
        #[serde(default = "default_true")]
        renormalize_after_capacity: bool,

        /// Router z-loss weight (stabilizes router logits).
        #[serde(default)]
        z_loss_weight: f32,

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

        /// Indices of "shared" experts that are always executed and added to the routed output.
        ///
        /// This implements the common "routed + shared" pattern: the router selects sparse
        /// experts per token, while a small set of experts are always-on to provide a stable
        /// baseline path.
        #[serde(default)]
        shared_experts: Vec<usize>,

        /// Scale applied to the mean output of shared experts.
        ///
        /// If 0.0 (default), shared experts are disabled.
        #[serde(default)]
        shared_expert_scale: f32,
    },
}

/// Routing mode for sparse Mixture-of-Experts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum ExpertRoutingMode {
    /// Token chooses its top-k experts.
    #[default]
    TokenChoiceTopK,
    /// Token chooses its top-k experts and a per-expert capacity is enforced.
    TokenChoiceTopKWithCapacity,
    /// Each expert chooses its top tokens (then tokens may be top-k filtered).
    ExpertChoice,
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

    /// Routing mode.
    #[serde(default)]
    pub routing_mode: ExpertRoutingMode,

    /// Capacity factor (Switch-style). 0.0 disables capacity limiting.
    #[serde(default)]
    pub capacity_factor: f32,

    /// Minimum capacity per expert.
    #[serde(default)]
    pub min_expert_capacity: usize,

    /// Renormalize per-token routing probabilities after capacity drops.
    #[serde(default = "default_true")]
    pub renormalize_after_capacity: bool,

    /// Router z-loss weight.
    #[serde(default)]
    pub z_loss_weight: f32,

    /// Indices of "shared" experts that are always executed and added to the routed output.
    #[serde(default)]
    pub shared_experts: Vec<usize>,

    /// Scale applied to the mean output of shared experts. If 0.0, shared experts are disabled.
    #[serde(default)]
    pub shared_expert_scale: f32,

    /// Metrics: accumulated router z-loss (sum of squared logsumexp).
    #[serde(default)]
    pub metrics_z_loss_sum_sq: f32,

    /// Metrics: number of router z-loss samples accumulated.
    #[serde(default)]
    pub metrics_z_loss_count: usize,

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
            use_head_conditioning: true,
            use_learned_k_adaptation: true,
            routing_mode: ExpertRoutingMode::default(),
            capacity_factor: 0.0,
            min_expert_capacity: 0,
            renormalize_after_capacity: true,
            z_loss_weight: 0.0,
            shared_experts: Vec::new(),
            shared_expert_scale: 0.0,
            metrics_z_loss_sum_sq: 0.0,
            metrics_z_loss_count: 0,
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
                routing_mode,
                capacity_factor,
                min_expert_capacity,
                renormalize_after_capacity,
                z_loss_weight,
                use_head_conditioning,
                use_learned_k_adaptation,
                shared_experts,
                shared_expert_scale,
            } => Self {
                gating: GatingConfig::from_strategy(
                    &GatingStrategy::Learned {
                        num_active: *num_active_experts,
                        load_balance_weight: *load_balance_weight,
                        sparsity_weight: *sparsity_weight,
                        complexity_loss_weight: 0.005, // Default
                        importance_loss_weight: 0.0,
                        switch_balance_weight: 0.0,
                    },
                    *num_experts,
                ),
                num_experts: *num_experts,
                expert_hidden_dim: *expert_hidden_dim,
                use_head_conditioning: *use_head_conditioning,
                use_learned_k_adaptation: *use_learned_k_adaptation,
                routing_mode: *routing_mode,
                capacity_factor: *capacity_factor,
                min_expert_capacity: *min_expert_capacity,
                renormalize_after_capacity: *renormalize_after_capacity,
                z_loss_weight: *z_loss_weight,
                shared_experts: shared_experts.clone(),
                shared_expert_scale: *shared_expert_scale,
                metrics_z_loss_sum_sq: 0.0,
                metrics_z_loss_count: 0,
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
        let mut w = ndarray::Array2::<f32>::zeros((2, 1));
        w[[0, 0]] = 0.0;
        w[[1, 0]] = 4.0;
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
        RichardsCurve::sigmoid(false).forward_scalar_f32(z)
    }
}

impl Default for LearnedKAdapter {
    fn default() -> Self {
        Self::new()
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
        self.metrics_z_loss_sum_sq = 0.0;
        self.metrics_z_loss_count = 0;
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

    /// Importance loss for training (balances soft routing probability mass)
    pub fn compute_importance_loss(&self) -> f32 {
        self.gating.compute_importance_loss()
    }

    /// Switch/GShard-style balance loss combining load and importance.
    pub fn compute_switch_balance_loss(&self) -> f32 {
        self.gating.compute_switch_balance_loss()
    }

    /// Router z-loss (mean of squared logsumexp(router_logits)).
    pub fn compute_z_loss(&self) -> f32 {
        if self.metrics_z_loss_count == 0 {
            return 0.0;
        }
        let v = self.metrics_z_loss_sum_sq / self.metrics_z_loss_count as f32;
        if v.is_finite() { v.max(0.0) } else { 0.0 }
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
        let imp = self.compute_importance_loss();
        let sw = self.compute_switch_balance_loss();
        let z = self.compute_z_loss();
        (lb * g.load_balance_weight)
            + (cx * g.complexity_loss_weight)
            + (sp * g.sparsity_weight)
            + (imp * g.importance_loss_weight)
            + (sw * g.switch_balance_weight)
            + (z * self.z_loss_weight)
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
        if let Some(h) = head_activity
            && !h.is_empty()
        {
            let bias = self.compute_head_bias(h, self.bias2.len());
            logits += &bias;
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
        let activated = self.activation.forward_const(&normalized);

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

        let n_experts = routing_probs.ncols();
        if routing_probs.nrows() == 0 || n_experts == 0 {
            return selections;
        }
        let k = k.clamp(1, n_experts);

        for row in routing_probs.outer_iter() {
            // Maintain a small set of best (score, idx) pairs (O(E*k), avoids full sort).
            let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
            for (idx, &prob) in row.iter().enumerate() {
                let score = if prob.is_finite() { prob } else { 0.0 };
                if best.len() < k {
                    best.push((score, idx));
                    continue;
                }

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

            let selected: Vec<usize> = best.into_iter().map(|(_s, idx)| idx).collect();
            selections.push(selected);
        }

        selections
    }

    /// Compute gradients for the two-layer routing network
    pub fn compute_gradients(&mut self, output_grads: &ndarray::Array2<f32>) -> RouterParamGrads {
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
        let mut d_normalized = ndarray::Array2::<f32>::zeros(cached_normalized.raw_dim());
        self.activation.curve.backward_matrix_f32_into(
            cached_normalized,
            &d_activated,
            &mut d_normalized,
        );

        // Gradient through Richards normalization
        let (d_hidden, _) = self.norm.compute_gradients(cached_hidden, &d_normalized);

        // First layer gradients
        let grad_weights1 = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));

        // Activation parameter gradients (Richards curve parameters)
        let activation_grads = self
            .activation
            .curve
            .grad_weights_matrix_f32(cached_normalized, &d_activated);

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
    pub fn param_shapes(&mut self) -> RouterParamShapes<'_> {
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
    cached_k_alpha: Option<Vec<f32>>,
    /// Cached features (entropy_norm, head_activity) for k-adaptation gradients.
    #[serde(skip)]
    cached_k_features: Option<Vec<(f32, f32)>>,
    /// Cached delta probabilities (p_topk - p_top1) used for d(alpha).
    #[serde(skip)]
    cached_k_delta_probs: Option<ndarray::Array2<f32>>,

    /// Cached per-expert weighted grad buffers for backward() to reduce allocations
    #[serde(skip)]
    cached_weighted_grads: Option<Vec<ndarray::Array2<f32>>>,

    #[serde(skip)]
    cached_aux_loss: f32,
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
            cached_aux_loss: 0.0,
        }
    }

    /// Set training mode for the MoE layer.
    ///
    /// Some layers toggle dropout/regularization behavior between train/eval.
    /// MoE currently has no explicit train/eval-only behavior, so this is a no-op
    /// kept for API compatibility with other modules.
    pub fn set_training_mode(&mut self, _training: bool) {
        // Intentionally no-op.
    }

    #[cfg(test)]
    pub(crate) fn test_cached_router_input(&self) -> Option<&ndarray::Array2<f32>> {
        self.cached_router_input.as_ref()
    }

    #[cfg(test)]
    pub(crate) fn test_cached_k_alpha(&self) -> Option<&[f32]> {
        self.cached_k_alpha.as_deref()
    }

    pub fn last_aux_loss(&self) -> f32 {
        if self.cached_aux_loss.is_finite() {
            self.cached_aux_loss.max(0.0)
        } else {
            0.0
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
        self.forward_with_head_features_and_token_activity(
            input,
            head_activity,
            head_activity_vec,
            None,
        )
    }

    pub fn forward_with_head_features_and_token_activity(
        &mut self,
        input: &ndarray::Array2<f32>,
        head_activity: Option<f32>,
        head_activity_vec: Option<&[f32]>,
        token_head_activity: Option<&[f32]>,
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
            router_in.slice_mut(ndarray::s![.., 0..d]).assign(input);
            if let Some(hv) = token_head_activity {
                debug_assert_eq!(hv.len(), n);
            }
            if let Some(hv) = token_head_activity
                && hv.len() == n
            {
                for i in 0..n {
                    let h = hv[i];
                    let h = if h.is_finite() { h } else { 0.0 };
                    router_in[[i, d]] = h.clamp(0.0, 1.0);
                }
            } else {
                let h = head_activity.unwrap_or(0.0);
                let h = if h.is_finite() { h } else { 0.0 };
                let h = h.clamp(0.0, 1.0);
                for i in 0..n {
                    router_in[[i, d]] = h;
                }
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

        // Track router z-loss statistics (mean of squared logsumexp(router_logits)).
        // This can be weighted into the training loss via config.z_loss_weight.
        update_router_z_loss_metrics(&mut self.config, cached_logits);

        let mut k_alpha_scratch = self.cached_k_alpha.take().unwrap_or_default();
        k_alpha_scratch.clear();
        self.cached_k_alpha = Some(k_alpha_scratch);
        let mut k_feat_scratch = self.cached_k_features.take().unwrap_or_default();
        k_feat_scratch.clear();
        self.cached_k_features = Some(k_feat_scratch);
        self.cached_k_delta_probs = None;

        let (mut masked_probs, mut active_mask) = match self.config.routing_mode {
            ExpertRoutingMode::ExpertChoice => {
                // Expert-choice routing: each expert selects its top tokens.
                expert_choice_routing(
                    &routing_probs_full,
                    base_k,
                    self.config.capacity_factor,
                    self.config.min_expert_capacity,
                )
            }
            ExpertRoutingMode::TokenChoiceTopK | ExpertRoutingMode::TokenChoiceTopKWithCapacity => {
                // Token-choice routing with optional MoH coupling (existing behavior).
                if self.config.use_learned_k_adaptation
                    && (head_activity.is_some()
                        || token_head_activity
                            .is_some_and(|hv| hv.len() == routing_probs_full.nrows()))
                {
                    if self.k_adapter.is_none() {
                        self.k_adapter = Some(LearnedKAdapter::new());
                    }
                    let denom = (self.config.num_experts.max(2) as f32).ln();
                    let denom = if denom.is_finite() && denom > 0.0 {
                        denom
                    } else {
                        1.0
                    };

                    // Blend between top-1 and configured top-k.
                    let (p_top1, m_top1) = masked_top_k_from_logits_and_active(cached_logits, 1);
                    let (p_topk, m_topk) =
                        masked_top_k_from_logits_and_active(cached_logits, base_k);

                    let n_tok = routing_probs_full.nrows();
                    let mut alpha_vec = self.cached_k_alpha.take().unwrap_or_default();
                    alpha_vec.resize(n_tok, 0.0);
                    let mut features = self.cached_k_features.take().unwrap_or_default();
                    features.clear();
                    features.reserve(n_tok);
                    for t in 0..n_tok {
                        let mut ent = 0.0f32;
                        for e in 0..self.config.num_experts {
                            let mut p = routing_probs_full[[t, e]];
                            p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                            if p > 0.0 {
                                ent -= p * p.ln();
                            }
                        }
                        let entropy_norm = (ent / denom).clamp(0.0, 1.0);
                        let h = if let Some(hv) = token_head_activity
                            && hv.len() == n_tok
                        {
                            let h = hv[t];
                            if h.is_finite() { h } else { 0.0 }
                        } else {
                            head_activity.unwrap_or(0.0)
                        };
                        let h = if h.is_finite() { h } else { 0.0 };
                        let h = h.clamp(0.0, 1.0);
                        let alpha = self
                            .k_adapter
                            .as_ref()
                            .expect("k_adapter must exist")
                            .alpha(entropy_norm, h);
                        alpha_vec[t] = if alpha.is_finite() {
                            alpha.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        features.push((entropy_norm, h));
                    }

                    let mut delta = p_topk.clone();
                    delta.zip_mut_with(&p_top1, |a, &b| {
                        *a -= b;
                    });

                    let mut p = p_top1;
                    for t in 0..n_tok {
                        let a = alpha_vec[t];
                        for e in 0..self.config.num_experts {
                            let v1 = p[[t, e]];
                            let vk = p_topk[[t, e]];
                            p[[t, e]] = (1.0 - a) * v1 + a * vk;
                        }
                    }

                    let mut m = m_top1;
                    for i in 0..m.len().min(m_topk.len()) {
                        m[i] = m[i] || m_topk[i];
                    }

                    self.cached_k_alpha = Some(alpha_vec);
                    self.cached_k_features = Some(features);
                    self.cached_k_delta_probs = Some(delta);

                    (p, m)
                } else if let Some(h) = head_activity {
                    // Heuristic smooth coupling (no cliff): interpolate between k=floor(kf) and
                    // k=ceil(kf).
                    let h = if h.is_finite() { h } else { 0.0 };
                    let h = h.clamp(0.0, 1.0);
                    let kf = 1.0 + (base_k.saturating_sub(1) as f32) * h;

                    let k_low = (kf.floor() as usize).clamp(1, base_k);
                    let k_high = (kf.ceil() as usize).clamp(1, base_k);
                    let alpha = (kf - k_low as f32).clamp(0.0, 1.0);

                    if k_low == k_high || alpha == 0.0 {
                        masked_top_k_from_logits_and_active(cached_logits, k_low)
                    } else {
                        let (p_low, m_low) =
                            masked_top_k_from_logits_and_active(cached_logits, k_low);
                        let (p_high, m_high) =
                            masked_top_k_from_logits_and_active(cached_logits, k_high);

                        // Blend probabilities; both are already per-row renormalized.
                        let mut p = p_low;
                        p.zip_mut_with(&p_high, |a, &b| {
                            *a = (1.0 - alpha) * (*a) + alpha * b;
                        });

                        // Union the expert-activity masks so we compute any expert needed by either
                        // path.
                        let mut m = m_low;
                        for i in 0..m.len().min(m_high.len()) {
                            m[i] = m[i] || m_high[i];
                        }
                        (p, m)
                    }
                } else {
                    masked_top_k_from_logits_and_active(cached_logits, base_k)
                }
            }
        };

        // Routed + shared experts: mark shared experts as active (they must be executed even if
        // routing probability is zero after masking/capacity).
        let shared_scale = if self.config.shared_expert_scale.is_finite() {
            self.config.shared_expert_scale
        } else {
            0.0
        };
        let mut shared_experts: Vec<usize> = Vec::new();
        if shared_scale != 0.0 {
            let mut seen = vec![false; self.config.num_experts];
            for &idx in &self.config.shared_experts {
                if idx < self.config.num_experts && !seen[idx] {
                    seen[idx] = true;
                    shared_experts.push(idx);
                }
            }
            for &e in &shared_experts {
                if e < active_mask.len() {
                    active_mask[e] = true;
                }
            }
        }
        let shared_per_expert = if !shared_experts.is_empty() {
            shared_scale / (shared_experts.len() as f32)
        } else {
            0.0
        };

        // Optional Switch-style per-expert capacity limiting.
        if self.config.routing_mode == ExpertRoutingMode::TokenChoiceTopKWithCapacity
            && self.config.capacity_factor > 0.0
        {
            let cap = compute_expert_capacity(
                masked_probs.nrows(),
                base_k,
                self.config.num_experts,
                self.config.capacity_factor,
                self.config.min_expert_capacity,
            );
            active_mask = apply_capacity_limit_inplace(
                &mut masked_probs,
                cap,
                self.config.renormalize_after_capacity,
            );
        }

        self.cached_active_expert_mask = Some(active_mask);
        self.cached_routing_probs = Some(masked_probs);

        let masked_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("masked routing probabilities must be cached");

        // Update routing metrics for training based on *active* routing.
        self.config.update_metrics(&masked_probs.view());
        self.cached_aux_loss = compute_moe_aux_loss_from_probs_and_logits(
            masked_probs,
            cached_logits,
            self.config.num_experts,
            self.config.gating.num_active as f32,
            &self.config.gating,
            self.config.z_loss_weight,
            self.config.diversity_weight,
        );

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
        if let Some(active_mask) = self.cached_active_expert_mask.as_deref() {
            for e in 0..self.config.num_experts {
                if e >= active_mask.len() || !active_mask[e] {
                    continue;
                }
                let expert_out = &expert_outputs[e];
                let routing_col = masked_probs.column(e);
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
        } else {
            for (e, expert_out) in expert_outputs
                .iter()
                .enumerate()
                .take(self.config.num_experts)
            {
                let routing_col = masked_probs.column(e);
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
        }

        // Add shared experts as an always-on path.
        if shared_per_expert != 0.0 {
            for &e in &shared_experts {
                if e >= self.config.num_experts {
                    continue;
                }
                let expert_out = &expert_outputs[e];
                output
                    .outer_iter_mut()
                    .zip(expert_out.outer_iter())
                    .for_each(|(mut out_row, expert_row)| {
                        out_row.scaled_add(shared_per_expert, &expert_row);
                    });
            }
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

        let active_mask = self.cached_active_expert_mask.as_deref();

        let shared_scale = if self.config.shared_expert_scale.is_finite() {
            self.config.shared_expert_scale
        } else {
            0.0
        };
        let mut shared_flags = vec![false; self.config.num_experts];
        let mut shared_count = 0usize;
        if shared_scale != 0.0 {
            for &idx in &self.config.shared_experts {
                if idx < shared_flags.len() && !shared_flags[idx] {
                    shared_flags[idx] = true;
                    shared_count += 1;
                }
            }
        }
        let shared_per_expert = if shared_count > 0 {
            shared_scale / (shared_count as f32)
        } else {
            0.0
        };

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
            if let Some(m) = active_mask
                && expert_idx < m.len()
                && !m[expert_idx]
            {
                continue;
            }

            let routing_col = routing_probs.column(expert_idx);
            let weighted_grads_2d = &mut weighted_buffers[expert_idx];
            weighted_grads_2d.fill(0.0);

            let shared_bonus = if expert_idx < shared_flags.len() && shared_flags[expert_idx] {
                shared_per_expert
            } else {
                0.0
            };

            for (token_idx, (grad_row, &weight)) in
                grads.outer_iter().zip(routing_col.iter()).enumerate()
            {
                let mut w = if weight.is_finite() { weight } else { 0.0 };
                w += shared_bonus;
                if !w.is_finite() {
                    w = 0.0;
                }
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
                let mut w = if weight.is_finite() { weight } else { 0.0 };
                w += shared_bonus;
                if !w.is_finite() {
                    w = 0.0;
                }
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

        let active_mask = self.cached_active_expert_mask.as_deref();

        let shared_scale = if self.config.shared_expert_scale.is_finite() {
            self.config.shared_expert_scale
        } else {
            0.0
        };
        let mut shared_flags = vec![false; self.config.num_experts];
        let mut shared_count = 0usize;
        if shared_scale != 0.0 {
            for &idx in &self.config.shared_experts {
                if idx < shared_flags.len() && !shared_flags[idx] {
                    shared_flags[idx] = true;
                    shared_count += 1;
                }
            }
        }
        let shared_per_expert = if shared_count > 0 {
            shared_scale / (shared_count as f32)
        } else {
            0.0
        };

        // 1. Route gradients to experts weighted by (post-mask) routing probabilities.
        // Only build grads for experts that were active for at least one token.
        let mut expert_output_grads =
            vec![ndarray::Array2::zeros(output_grads.raw_dim()); self.config.num_experts];
        for expert_idx in 0..self.config.num_experts {
            if let Some(m) = active_mask
                && expert_idx < m.len()
                && !m[expert_idx]
            {
                continue;
            }
            let shared_bonus = if expert_idx < shared_flags.len() && shared_flags[expert_idx] {
                shared_per_expert
            } else {
                0.0
            };
            for token_idx in 0..output_grads.nrows() {
                let mut w = cached_routing_probs[[token_idx, expert_idx]];
                w = if w.is_finite() { w } else { 0.0 };
                w += shared_bonus;
                if !w.is_finite() {
                    w = 0.0;
                }
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
            if let Some(m) = active_mask
                && expert_idx < m.len()
                && !m[expert_idx]
            {
                all_param_grads.extend(zero_expert_grads(expert));
                continue;
            }
            let expert_grads = &expert_output_grads[expert_idx];
            let (expert_input_grad, expert_param_grads) =
                expert.compute_gradients(cached_input, expert_grads);

            let shared_bonus = if expert_idx < shared_flags.len() && shared_flags[expert_idx] {
                shared_per_expert
            } else {
                0.0
            };

            // Weight input gradients by routing probabilities
            for token_idx in 0..expert_input_grad.nrows() {
                let mut routing_weight = cached_routing_probs[[token_idx, expert_idx]];
                routing_weight = if routing_weight.is_finite() {
                    routing_weight
                } else {
                    0.0
                };
                routing_weight += shared_bonus;
                if !routing_weight.is_finite() {
                    routing_weight = 0.0;
                }
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
                self.cached_k_alpha.as_ref(),
                self.cached_k_features.as_ref(),
                self.cached_k_delta_probs.as_ref(),
            ) {
                (Some(_), Some(alpha_vec), Some(features), Some(delta))
                    if alpha_vec.len() == output_grads.nrows()
                        && features.len() == output_grads.nrows() =>
                {
                    let mut g_w = ndarray::Array2::<f32>::zeros((2, 1));
                    let mut g_b = ndarray::Array2::<f32>::zeros((1, 1));

                    for t in 0..output_grads.nrows() {
                        let alpha = alpha_vec[t];
                        let alpha = if alpha.is_finite() {
                            alpha.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let (entropy_norm, head_activity) = features[t];
                        let entropy_norm = if entropy_norm.is_finite() {
                            entropy_norm.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let head_activity = if head_activity.is_finite() {
                            head_activity.clamp(0.0, 1.0)
                        } else {
                            0.0
                        };

                        let token_output_grad = output_grads.row(t);
                        let mut d_alpha_t = 0.0f32;
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
                            d_alpha_t += g * dp;
                        }

                        let dz = d_alpha_t * alpha * (1.0 - alpha);
                        g_w[[0, 0]] += dz * entropy_norm;
                        g_w[[1, 0]] += dz * head_activity;
                        g_b[[0, 0]] += dz;
                    }

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

        let n_tok = routing_probs.nrows();
        let n_exp = self.config.num_experts;
        let ln_n_exp = if n_exp >= 2 { (n_exp as f32).ln() } else { 1.0 };
        let inv_n_tok = if n_tok > 0 { 1.0 / (n_tok as f32) } else { 0.0 };

        let lb_w = if self.config.gating.load_balance_weight.is_finite() {
            self.config.gating.load_balance_weight.max(0.0)
        } else {
            0.0
        };
        let sp_w = if self.config.gating.sparsity_weight.is_finite() {
            self.config.gating.sparsity_weight.max(0.0)
        } else {
            0.0
        };
        let cx_w = if self.config.gating.complexity_loss_weight.is_finite() {
            self.config.gating.complexity_loss_weight.max(0.0)
        } else {
            0.0
        };
        let imp_w = if self.config.gating.importance_loss_weight.is_finite() {
            self.config.gating.importance_loss_weight.max(0.0)
        } else {
            0.0
        };
        let sw_w = if self.config.gating.switch_balance_weight.is_finite() {
            self.config.gating.switch_balance_weight.max(0.0)
        } else {
            0.0
        };
        let dv_w = if self.config.diversity_weight.is_finite() {
            self.config.diversity_weight.max(0.0)
        } else {
            0.0
        };
        let z_w = if self.config.z_loss_weight.is_finite() {
            self.config.z_loss_weight.max(0.0)
        } else {
            0.0
        };

        let bal_w = lb_w + imp_w + sw_w;
        let target_avg_experts = self.config.gating.num_active as f32;

        let mut importance: Vec<f32> = vec![0.0; n_exp];
        if bal_w != 0.0 && n_tok > 0 && n_exp > 0 {
            for t in 0..n_tok {
                for e in 0..n_exp {
                    let p = routing_probs[[t, e]];
                    let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                    importance[e] += p;
                }
            }
            for v in importance.iter_mut().take(n_exp) {
                *v *= inv_n_tok;
            }
        }

        let mut k_eff_per_token: Vec<f32> = Vec::new();
        let mut mean_k_eff = 0.0f32;
        if cx_w != 0.0 && n_tok > 0 && n_exp > 0 {
            k_eff_per_token.resize(n_tok, 0.0);
            for t in 0..n_tok {
                let mut h = 0.0f32;
                for e in 0..n_exp {
                    let p = routing_probs[[t, e]];
                    let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                    if p > 0.0 {
                        h -= p * p.ln();
                    }
                }
                let k_eff = crate::pade::PadeExp::exp(h as f64) as f32;
                let k_eff = if k_eff.is_finite() {
                    k_eff.clamp(1.0, n_exp as f32)
                } else {
                    1.0
                };
                k_eff_per_token[t] = k_eff;
                mean_k_eff += k_eff;
            }
            mean_k_eff *= inv_n_tok;
        }

        let cx_coeff_base = if cx_w != 0.0 && n_tok > 0 {
            2.0 * (mean_k_eff - target_avg_experts) * inv_n_tok
        } else {
            0.0
        };

        let dv_norm = if dv_w != 0.0 && n_tok > 0 && n_exp > 1 {
            (n_exp as f32) * ((n_exp - 1) as f32)
        } else {
            1.0
        };

        let mut active_pairs: Vec<(usize, f32, f32)> = Vec::new();
        for token_idx in 0..n_tok {
            let token_output_grad = output_grads.row(token_idx);
            let mut dot_gy = 0.0f32;
            active_pairs.clear();

            for expert_idx in 0..n_exp {
                let y = routing_probs[[token_idx, expert_idx]];
                let y = if y.is_finite() { y } else { 0.0 };
                if y == 0.0 {
                    continue;
                }
                let expert_output = cached_expert_outputs[expert_idx].row(token_idx);
                let g_main = token_output_grad
                    .iter()
                    .zip(expert_output.iter())
                    .map(|(&g, &o)| {
                        let g = if g.is_finite() { g } else { 0.0 };
                        let o = if o.is_finite() { o } else { 0.0 };
                        g * o
                    })
                    .sum::<f32>();

                let mut g_aux = 0.0f32;
                if bal_w != 0.0 && n_tok > 0 {
                    let d_lb = (2.0 * (n_exp as f32) * inv_n_tok) * importance[expert_idx];
                    if d_lb.is_finite() {
                        g_aux += bal_w * d_lb;
                    }
                }

                if (sp_w != 0.0 || cx_w != 0.0) && n_tok > 0 {
                    let p = y;
                    let ln_p = p.ln();
                    let d_h = -(ln_p + 1.0);
                    if sp_w != 0.0 && ln_n_exp > 0.0 {
                        let d_sp = d_h * inv_n_tok / ln_n_exp;
                        if d_sp.is_finite() {
                            g_aux += sp_w * d_sp;
                        }
                    }
                    if cx_w != 0.0 {
                        let k_eff = if token_idx < k_eff_per_token.len() {
                            k_eff_per_token[token_idx]
                        } else {
                            1.0
                        };
                        let d_cx = cx_coeff_base * k_eff * d_h;
                        if d_cx.is_finite() {
                            g_aux += cx_w * d_cx;
                        }
                    }
                }

                if dv_w != 0.0 && n_tok > 0 && n_exp > 1 {
                    let d_dv = (-2.0 * y) * inv_n_tok / dv_norm;
                    if d_dv.is_finite() {
                        g_aux += dv_w * d_dv;
                    }
                }

                let g = g_main + g_aux;
                active_pairs.push((expert_idx, g, y));
                dot_gy += g * y;
            }

            for &(expert_idx, g, y) in &active_pairs {
                d_logits[[token_idx, expert_idx]] = y * (g - dot_gy);
            }
        }

        if z_w != 0.0 && n_tok > 0 && n_exp > 0 {
            let cached_logits = self
                .router
                .cached_logits
                .as_ref()
                .expect("Router predict must cache logits");
            let y_full = self
                .router
                .cached_output
                .as_ref()
                .expect("Router predict must cache full softmax output");

            for t in 0..n_tok {
                let row = cached_logits.row(t);
                let mut max_v = f32::NEG_INFINITY;
                let mut any = false;
                for &v in row.iter() {
                    if v.is_finite() {
                        any = true;
                        max_v = max_v.max(v);
                    }
                }
                if !any {
                    continue;
                }

                let mut sum_exp: f64 = 0.0;
                for &v in row.iter() {
                    if v.is_finite() {
                        sum_exp += crate::pade::PadeExp::exp((v - max_v) as f64);
                    }
                }
                if !sum_exp.is_finite() || sum_exp <= 0.0 {
                    continue;
                }
                let z = (sum_exp.ln() as f32) + max_v;
                if !z.is_finite() {
                    continue;
                }

                let coeff = (2.0 * z_w * z) * inv_n_tok;
                if !coeff.is_finite() {
                    continue;
                }

                for e in 0..n_exp {
                    let p = y_full[[t, e]];
                    let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                    d_logits[[t, e]] += coeff * p;
                }
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
        let (d_normalized, activation_param_grads) = self
            .router
            .activation
            .compute_gradients(cached_normalized, &d_activated);

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
        if let Some(w) = self.router.head_to_expert.as_mut()
            && router_grad_idx < param_grads.len()
            && param_grads[router_grad_idx].raw_dim() == w.raw_dim()
        {
            w.scaled_add(-lr, &param_grads[router_grad_idx]);
            router_grad_idx += 1;
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
        if let Some(adapter) = self.k_adapter.as_mut()
            && grad_idx + 2 <= param_grads.len()
            && param_grads[grad_idx].raw_dim() == adapter.w.raw_dim()
            && param_grads[grad_idx + 1].raw_dim() == adapter.b.raw_dim()
        {
            adapter.w.scaled_add(-lr, &param_grads[grad_idx]);
            adapter.b.scaled_add(-lr, &param_grads[grad_idx + 1]);
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

fn update_router_z_loss_metrics(config: &mut ExpertRouterConfig, logits: &ndarray::Array2<f32>) {
    if logits.nrows() == 0 || logits.ncols() == 0 {
        return;
    }

    for row in logits.outer_iter() {
        // Stable logsumexp.
        let mut max_v = f32::NEG_INFINITY;
        let mut any = false;
        for &v in row.iter() {
            if v.is_finite() {
                any = true;
                max_v = max_v.max(v);
            }
        }
        if !any {
            continue;
        }

        let mut sum_exp: f64 = 0.0;
        for &v in row.iter() {
            if v.is_finite() {
                sum_exp += crate::pade::PadeExp::exp((v - max_v) as f64);
            }
        }
        if sum_exp <= 0.0 || !sum_exp.is_finite() {
            continue;
        }

        let z = (sum_exp.ln() as f32) + max_v;
        if z.is_finite() {
            config.metrics_z_loss_sum_sq += z * z;
            config.metrics_z_loss_count += 1;
        }
    }
}

fn compute_moe_aux_loss_from_probs_and_logits(
    masked_probs: &ndarray::Array2<f32>,
    logits: &ndarray::Array2<f32>,
    num_experts: usize,
    target_avg_experts: f32,
    gating: &GatingConfig,
    z_loss_weight: f32,
    diversity_weight: f32,
) -> f32 {
    let n_tok = masked_probs.nrows();
    if n_tok == 0 || num_experts == 0 {
        return 0.0;
    }

    let n_exp_f = num_experts as f32;
    let inv_n = 1.0 / (n_tok as f32);
    let ln_n = if num_experts >= 2 {
        (num_experts as f32).ln()
    } else {
        1.0
    };

    let bal_w = (if gating.load_balance_weight.is_finite() {
        gating.load_balance_weight.max(0.0)
    } else {
        0.0
    }) + (if gating.importance_loss_weight.is_finite() {
        gating.importance_loss_weight.max(0.0)
    } else {
        0.0
    }) + (if gating.switch_balance_weight.is_finite() {
        gating.switch_balance_weight.max(0.0)
    } else {
        0.0
    });

    let sp_w = if gating.sparsity_weight.is_finite() {
        gating.sparsity_weight.max(0.0)
    } else {
        0.0
    };

    let cx_w = if gating.complexity_loss_weight.is_finite() {
        gating.complexity_loss_weight.max(0.0)
    } else {
        0.0
    };

    let dv_w = if diversity_weight.is_finite() {
        diversity_weight.max(0.0)
    } else {
        0.0
    };

    let z_w = if z_loss_weight.is_finite() {
        z_loss_weight.max(0.0)
    } else {
        0.0
    };

    let mut loss = 0.0f32;

    if bal_w != 0.0 {
        let mut imp = vec![0.0f32; num_experts];
        for t in 0..n_tok {
            for e in 0..num_experts {
                let p = masked_probs[[t, e]];
                let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                imp[e] += p;
            }
        }
        for v in imp.iter_mut().take(num_experts) {
            *v *= inv_n;
        }
        let sum_sq = imp.iter().map(|&x| x * x).sum::<f32>();
        let bal = (n_exp_f * sum_sq) - 1.0;
        if bal.is_finite() {
            loss += bal_w * bal.max(0.0);
        }
    }

    if sp_w != 0.0 || cx_w != 0.0 || dv_w != 0.0 {
        let mut entropy_sum = 0.0f32;
        let mut k_eff_sum = 0.0f32;
        let mut diversity_sum = 0.0f32;
        let dv_norm = if num_experts > 1 {
            n_exp_f * ((num_experts - 1) as f32)
        } else {
            1.0
        };

        for t in 0..n_tok {
            let mut h = 0.0f32;
            let mut sum_p2 = 0.0f32;
            for e in 0..num_experts {
                let p = masked_probs[[t, e]];
                let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
                if p > 0.0 {
                    h -= p * p.ln();
                }
                sum_p2 += p * p;
            }
            entropy_sum += h;
            if cx_w != 0.0 {
                let k_eff = crate::pade::PadeExp::exp(h as f64) as f32;
                let k_eff = if k_eff.is_finite() {
                    k_eff.clamp(1.0, n_exp_f)
                } else {
                    1.0
                };
                k_eff_sum += k_eff;
            }
            if dv_w != 0.0 && num_experts > 1 {
                let dv = (1.0 - sum_p2) / dv_norm;
                if dv.is_finite() {
                    diversity_sum += dv.max(0.0);
                }
            }
        }

        if sp_w != 0.0 && ln_n > 0.0 {
            let ent = (entropy_sum * inv_n) / ln_n;
            if ent.is_finite() {
                loss += sp_w * ent.max(0.0);
            }
        }
        if cx_w != 0.0 {
            let mean_k = k_eff_sum * inv_n;
            let cx = (mean_k - target_avg_experts).powi(2);
            if cx.is_finite() {
                loss += cx_w * cx.max(0.0);
            }
        }
        if dv_w != 0.0 && num_experts > 1 {
            let dv = diversity_sum * inv_n;
            if dv.is_finite() {
                loss += dv_w * dv.max(0.0);
            }
        }
    }

    if z_w != 0.0 && logits.nrows() == n_tok && logits.ncols() == num_experts {
        let mut z_sum = 0.0f32;
        let mut z_cnt = 0usize;
        for row in logits.outer_iter() {
            let mut max_v = f32::NEG_INFINITY;
            let mut any = false;
            for &v in row.iter() {
                if v.is_finite() {
                    any = true;
                    max_v = max_v.max(v);
                }
            }
            if !any {
                continue;
            }

            let mut sum_exp: f64 = 0.0;
            for &v in row.iter() {
                if v.is_finite() {
                    sum_exp += crate::pade::PadeExp::exp((v - max_v) as f64);
                }
            }
            if sum_exp <= 0.0 || !sum_exp.is_finite() {
                continue;
            }

            let z = (sum_exp.ln() as f32) + max_v;
            if z.is_finite() {
                z_sum += z * z;
                z_cnt += 1;
            }
        }
        if z_cnt > 0 {
            let z = z_sum / (z_cnt as f32);
            if z.is_finite() {
                loss += z_w * z.max(0.0);
            }
        }
    }

    if loss.is_finite() { loss.max(0.0) } else { 0.0 }
}

fn compute_expert_capacity(
    n_tokens: usize,
    k: usize,
    n_experts: usize,
    capacity_factor: f32,
    min_capacity: usize,
) -> usize {
    if n_tokens == 0 || n_experts == 0 {
        return 0;
    }
    if !(capacity_factor.is_finite()) || capacity_factor <= 0.0 {
        return usize::MAX;
    }

    let k = k.max(1);
    let n_experts = n_experts.max(1);
    let expected = (n_tokens as f32) * (k as f32) / (n_experts as f32);
    let cap = (capacity_factor * expected).ceil() as usize;
    cap.max(min_capacity).max(1)
}

fn apply_capacity_limit_inplace(
    masked_probs: &mut ndarray::Array2<f32>,
    capacity: usize,
    renormalize: bool,
) -> Vec<bool> {
    let n_tokens = masked_probs.nrows();
    let n_experts = masked_probs.ncols();

    if n_tokens == 0 || n_experts == 0 {
        return vec![false; n_experts];
    }
    if capacity == 0 {
        masked_probs.fill(0.0);
        return vec![false; n_experts];
    }
    if capacity == usize::MAX {
        // Just compute active mask.
        let mut active = vec![false; n_experts];
        for e in 0..n_experts {
            for t in 0..n_tokens {
                let w = masked_probs[[t, e]];
                let w = if w.is_finite() { w } else { 0.0 };
                if w > 0.0 {
                    active[e] = true;
                    break;
                }
            }
        }
        return active;
    }

    // Drop lowest-weight assignments per expert.
    // Use partial selection to avoid O(T log T) sorts when capacity is active.
    let mut candidates: Vec<(f32, usize)> = Vec::with_capacity(n_tokens);
    for e in 0..n_experts {
        candidates.clear();
        for t in 0..n_tokens {
            let w = masked_probs[[t, e]];
            let w = if w.is_finite() { w } else { 0.0 };
            if w > 0.0 {
                candidates.push((w, t));
            }
        }

        if candidates.len() <= capacity {
            continue;
        }

        let nth = capacity.saturating_sub(1);
        candidates.select_nth_unstable_by(nth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &(_w, t) in candidates.iter().skip(capacity) {
            masked_probs[[t, e]] = 0.0;
        }
    }

    if renormalize {
        let eps = 1e-6f32;
        for t in 0..n_tokens {
            let mut sum = 0.0f32;
            for e in 0..n_experts {
                let w = masked_probs[[t, e]];
                let w = if w.is_finite() { w } else { 0.0 };
                sum += w;
            }
            // Guard against division by a tiny sum which can create huge scales/gradients.
            if sum > eps && sum.is_finite() {
                let inv = 1.0 / sum;
                for e in 0..n_experts {
                    let w = masked_probs[[t, e]];
                    masked_probs[[t, e]] = if w.is_finite() { w * inv } else { 0.0 };
                }
            } else {
                // At minimum, keep the row finite.
                for e in 0..n_experts {
                    if !masked_probs[[t, e]].is_finite() {
                        masked_probs[[t, e]] = 0.0;
                    }
                }
            }
        }
    }

    // Active mask after drops.
    let mut active = vec![false; n_experts];
    for e in 0..n_experts {
        for t in 0..n_tokens {
            let w = masked_probs[[t, e]];
            let w = if w.is_finite() { w } else { 0.0 };
            if w > 0.0 {
                active[e] = true;
                break;
            }
        }
    }
    active
}

fn expert_choice_routing(
    routing_probs_full: &ndarray::Array2<f32>,
    token_top_k: usize,
    capacity_factor: f32,
    min_capacity: usize,
) -> (ndarray::Array2<f32>, Vec<bool>) {
    let n_tokens = routing_probs_full.nrows();
    let n_experts = routing_probs_full.ncols();

    if n_tokens == 0 || n_experts == 0 {
        return (
            ndarray::Array2::<f32>::zeros((n_tokens, n_experts)),
            vec![false; n_experts],
        );
    }

    let k = token_top_k.max(1).min(n_experts);
    let cap = compute_expert_capacity(n_tokens, k, n_experts, capacity_factor, min_capacity)
        .min(n_tokens)
        .max(1);

    // Step 1: experts select top-cap tokens by probability.
    let mut w = ndarray::Array2::<f32>::zeros((n_tokens, n_experts));
    let mut best: Vec<(f32, usize)> = Vec::with_capacity(n_tokens);
    for e in 0..n_experts {
        best.clear();
        for t in 0..n_tokens {
            let p = routing_probs_full[[t, e]];
            let p = if p.is_finite() { p.max(0.0) } else { 0.0 };
            best.push((p, t));
        }

        if cap < best.len() {
            let nth = cap.saturating_sub(1);
            best.select_nth_unstable_by(nth, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for &(p, t) in best.iter().take(cap) {
            if p > 0.0 {
                w[[t, e]] = p;
            }
        }
    }

    // Step 2: enforce per-token top-k (optional but keeps compute bounded and consistent).
    for t in 0..n_tokens {
        // Track top-k by weight.
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k);
        for e in 0..n_experts {
            let p = w[[t, e]];
            let p = if p.is_finite() { p } else { 0.0 };
            if p <= 0.0 {
                continue;
            }
            if best.len() < k {
                best.push((p, e));
                continue;
            }
            let mut min_pos = 0usize;
            let mut min_score = best[0].0;
            for (pos, (s, _)) in best.iter().enumerate().skip(1) {
                if *s < min_score {
                    min_score = *s;
                    min_pos = pos;
                }
            }
            if p > min_score {
                best[min_pos] = (p, e);
            }
        }

        // Zero out everything not in best (avoid allocating a full keep mask).
        if best.is_empty() {
            continue;
        }
        for e in 0..n_experts {
            let mut keep_e = false;
            for &(_p, be) in &best {
                if be == e {
                    keep_e = true;
                    break;
                }
            }
            if !keep_e {
                w[[t, e]] = 0.0;
            }
        }

        // Renormalize row.
        let mut sum = 0.0f32;
        for e in 0..n_experts {
            sum += w[[t, e]];
        }
        // Same epsilon guard as other normalization sites to prevent rare amplification
        // when the kept mass collapses.
        let eps = 1e-6f32;
        if sum > eps && sum.is_finite() {
            let inv = 1.0 / sum;
            for e in 0..n_experts {
                let v = w[[t, e]];
                w[[t, e]] = if v.is_finite() { v * inv } else { 0.0 };
            }
        } else {
            for e in 0..n_experts {
                if !w[[t, e]].is_finite() {
                    w[[t, e]] = 0.0;
                }
            }
        }
    }

    // Active mask.
    let mut active = vec![false; n_experts];
    for e in 0..n_experts {
        for t in 0..n_tokens {
            if w[[t, e]] > 0.0 {
                active[e] = true;
                break;
            }
        }
    }

    (w, active)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_expert_router_config_default() {
        let config = ExpertRouterConfig::default();
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.gating.num_active, 2);
        assert_eq!(config.expert_hidden_dim, 64);
        assert_eq!(config.gating.load_balance_weight, 0.0);
        assert!(config.use_head_conditioning);
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
            routing_mode: ExpertRoutingMode::TokenChoiceTopK,
            capacity_factor: 0.0,
            min_expert_capacity: 0,
            renormalize_after_capacity: true,
            z_loss_weight: 0.0,
            use_head_conditioning: true,
            use_learned_k_adaptation: false,
            shared_experts: vec![],
            shared_expert_scale: 0.0,
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
    fn test_moe_token_head_activity_affects_k_adaptation_and_router_input() {
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
        config.use_learned_k_adaptation = true;

        let mut moe = MixtureOfExperts::new(32, 8, config);
        moe.k_adapter = Some(LearnedKAdapter {
            w: ndarray::Array2::from_shape_vec((2, 1), vec![0.0, 20.0]).unwrap(),
            b: ndarray::Array2::from_shape_vec((1, 1), vec![-10.0]).unwrap(),
        });

        let input = ndarray::Array2::<f32>::from_shape_vec((2, 32), vec![0.1; 64]).unwrap();
        let token_h = vec![0.0f32, 1.0f32];
        let _out = moe.forward_with_head_features_and_token_activity(
            &input,
            Some(0.0),
            None,
            Some(token_h.as_slice()),
        );

        let router_in = moe.cached_router_input.as_ref().unwrap();
        assert!(approx_eq(router_in[[0, 32]], 0.0, 1e-6));
        assert!(approx_eq(router_in[[1, 32]], 1.0, 1e-6));

        let alpha = moe.cached_k_alpha.as_ref().unwrap();
        assert!(alpha[0] < 0.01);
        assert!(alpha[1] > 0.99);
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
        config.gating.metrics.active_sum_per_component = vec![100.0, 0.0, 0.0, 0.0];
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

    #[test]
    fn test_apply_capacity_limit_inplace_respects_capacity() {
        // 5 tokens, 2 experts.
        let mut probs = ndarray::Array2::from_shape_vec(
            (5, 2),
            vec![
                0.90, 0.10, // t0
                0.80, 0.20, // t1
                0.10, 0.30, // t2
                0.05, 0.40, // t3
                0.01, 0.50, // t4
            ],
        )
        .unwrap();

        let active = apply_capacity_limit_inplace(&mut probs, 2, false);
        assert_eq!(active.len(), 2);

        // Expert 0 should keep t0,t1 only.
        let mut kept0 = 0usize;
        for t in 0..5 {
            if probs[[t, 0]] > 0.0 {
                kept0 += 1;
            }
        }
        assert_eq!(kept0, 2);
        assert!(probs[[0, 0]] > 0.0);
        assert!(probs[[1, 0]] > 0.0);
        assert_eq!(probs[[2, 0]], 0.0);
        assert_eq!(probs[[3, 0]], 0.0);
        assert_eq!(probs[[4, 0]], 0.0);

        // Expert 1 should keep t4,t3 only (0.5 and 0.4).
        let mut kept1 = 0usize;
        for t in 0..5 {
            if probs[[t, 1]] > 0.0 {
                kept1 += 1;
            }
        }
        assert_eq!(kept1, 2);
        assert!(probs[[4, 1]] > 0.0);
        assert!(probs[[3, 1]] > 0.0);
        assert_eq!(probs[[0, 1]], 0.0);
        assert_eq!(probs[[1, 1]], 0.0);
        assert_eq!(probs[[2, 1]], 0.0);

        // Active mask should reflect both experts still active.
        assert!(active[0]);
        assert!(active[1]);
    }

    #[test]
    fn test_apply_capacity_limit_inplace_renormalizes_rows() {
        // 3 tokens, 2 experts.
        let mut probs = ndarray::Array2::from_shape_vec(
            (3, 2),
            vec![
                0.60, 0.40, // t0 sum=1
                0.90, 0.10, // t1 sum=1
                0.20, 0.80, // t2 sum=1
            ],
        )
        .unwrap();

        // Capacity=1 per expert will drop some assignments.
        let _active = apply_capacity_limit_inplace(&mut probs, 1, true);

        // For any row with any non-zero entries, the row should sum to ~1.
        for t in 0..3 {
            let mut sum = 0.0f32;
            for e in 0..2 {
                sum += probs[[t, e]];
            }
            if sum > 0.0 {
                assert!(approx_eq(sum, 1.0, 1e-6));
            }
        }
    }

    #[test]
    fn test_expert_choice_routing_invariants() {
        // 4 tokens, 3 experts.
        let probs = ndarray::Array2::from_shape_vec(
            (4, 3),
            vec![
                0.70, 0.20, 0.10, // t0
                0.10, 0.80, 0.10, // t1
                0.20, 0.20, 0.60, // t2
                0.34, 0.33, 0.33, // t3
            ],
        )
        .unwrap();

        let (w, active) = expert_choice_routing(&probs, 2, 1.0, 1);
        assert_eq!(w.dim(), (4, 3));
        assert_eq!(active.len(), 3);

        // Per-token nonzeros should be <= k.
        for t in 0..4 {
            let mut nz = 0usize;
            let mut sum = 0.0f32;
            for e in 0..3 {
                let v = w[[t, e]];
                if v > 0.0 {
                    nz += 1;
                }
                sum += v;
            }
            assert!(nz <= 2);
            if nz > 0 {
                assert!(approx_eq(sum, 1.0, 1e-5));
            }
        }

        // Active flags match presence of nonzero weights.
        for e in 0..3 {
            let mut any = false;
            for t in 0..4 {
                if w[[t, e]] > 0.0 {
                    any = true;
                    break;
                }
            }
            assert_eq!(active[e], any);
        }
    }

    #[test]
    fn test_router_z_loss_metrics_accumulate() {
        let mut cfg = ExpertRouterConfig::default();
        let logits = ndarray::Array2::<f32>::zeros((4, 3));
        update_router_z_loss_metrics(&mut cfg, &logits);

        // logsumexp(0,0,0) = ln(3)
        let z = (3.0f32).ln();
        let expected = 4.0 * z * z;
        assert_eq!(cfg.metrics_z_loss_count, 4);
        assert!(approx_eq(cfg.metrics_z_loss_sum_sq, expected, 1e-5));
    }

    #[test]
    fn test_non_finite_logits_do_not_produce_nan_routing() {
        // Construct logits with NaN and -inf; selection should still produce finite weights.
        let logits = ndarray::Array2::from_shape_vec(
            (3, 4),
            vec![
                f32::NAN,
                f32::NEG_INFINITY,
                -1.0,
                0.0,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                5.0,
                f32::NAN,
                1.0,
                f32::NEG_INFINITY,
            ],
        )
        .unwrap();

        let (mut masked, _active) = masked_top_k_from_logits_and_active(&logits, 2);

        // Apply a tight capacity to force drops and renormalization.
        let _active2 = apply_capacity_limit_inplace(&mut masked, 1, true);

        for v in masked.iter() {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }

        // Rows should sum to ~1 for any row that has any mass.
        for t in 0..masked.nrows() {
            let mut sum = 0.0f32;
            for e in 0..masked.ncols() {
                sum += masked[[t, e]];
            }
            if sum > 0.0 {
                assert!(approx_eq(sum, 1.0, 1e-5));
            }
        }

        // z-loss metrics should ignore non-finite logits and remain finite.
        let mut cfg = ExpertRouterConfig::default();
        update_router_z_loss_metrics(&mut cfg, &logits);
        assert!(cfg.metrics_z_loss_sum_sq.is_finite());
        // At least rows with any finite values should be counted.
        assert!(cfg.metrics_z_loss_count >= 2);
    }

    #[test]
    fn test_shared_experts_change_output() {
        let base_cfg = ExpertRouterConfig {
            num_experts: 2,
            expert_hidden_dim: 16,
            diversity_weight: 0.005,
            gating: GatingConfig {
                num_active: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut moe_base = MixtureOfExperts::new(32, 8, base_cfg);
        let mut moe_shared = moe_base.clone();

        moe_shared.config.shared_experts = vec![1];
        moe_shared.config.shared_expert_scale = 1.0;

        let input = ndarray::Array2::<f32>::from_shape_vec((4, 32), vec![0.1; 128]).unwrap();
        let out_base = moe_base.forward(&input);
        let out_shared = moe_shared.forward(&input);

        // Shared experts add an extra always-on path, so output should differ.
        let mut l1 = 0.0f32;
        for (a, b) in out_base.iter().zip(out_shared.iter()) {
            l1 += (a - b).abs();
        }
        assert!(l1 > 1e-6);
    }

    #[test]
    fn test_compute_moe_aux_loss_balance_zero_for_uniform() {
        let probs = ndarray::Array2::from_shape_vec(
            (4, 4),
            vec![
                0.25, 0.25, 0.25, 0.25, //
                0.25, 0.25, 0.25, 0.25, //
                0.25, 0.25, 0.25, 0.25, //
                0.25, 0.25, 0.25, 0.25, //
            ],
        )
        .unwrap();
        let logits = ndarray::Array2::<f32>::zeros((4, 4));
        let gating = GatingConfig {
            num_active: 2,
            load_balance_weight: 1.0,
            ..Default::default()
        };
        let loss =
            compute_moe_aux_loss_from_probs_and_logits(&probs, &logits, 4, 2.0, &gating, 0.0, 0.0);
        assert!(approx_eq(loss, 0.0, 1e-6));
    }

    #[test]
    fn test_compute_moe_aux_loss_balance_positive_for_collapsed() {
        let probs = ndarray::Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
            ],
        )
        .unwrap();
        let logits = ndarray::Array2::<f32>::zeros((4, 4));
        let gating = GatingConfig {
            num_active: 2,
            load_balance_weight: 1.0,
            ..Default::default()
        };
        let loss =
            compute_moe_aux_loss_from_probs_and_logits(&probs, &logits, 4, 2.0, &gating, 0.0, 0.0);
        assert!(approx_eq(loss, 3.0, 1e-6));
    }

    #[test]
    fn test_compute_moe_aux_loss_z_loss_matches_ln_e_sq() {
        let probs = ndarray::Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
            ],
        )
        .unwrap();
        let logits = ndarray::Array2::<f32>::zeros((4, 3));
        let gating = GatingConfig {
            num_active: 1,
            ..Default::default()
        };
        let loss =
            compute_moe_aux_loss_from_probs_and_logits(&probs, &logits, 3, 1.0, &gating, 1.0, 0.0);
        let expected = (3.0f32).ln().powi(2);
        assert!(approx_eq(loss, expected, 1e-5));
    }

    #[test]
    fn test_moe_router_receives_aux_grads_when_output_grads_zero() {
        let mut config = ExpertRouterConfig {
            num_experts: 4,
            expert_hidden_dim: 16,
            diversity_weight: 0.0,
            gating: GatingConfig {
                num_active: 2,
                load_balance_weight: 1.0,
                ..Default::default()
            },
            ..Default::default()
        };
        config.z_loss_weight = 0.0;

        let mut moe = MixtureOfExperts::new(8, 8, config);
        moe.router.bias2.fill(0.0);
        moe.router.bias2[0] = 10.0;

        let input = ndarray::Array2::<f32>::zeros((4, 8));
        let _out = moe.forward(&input);

        let output_grads = ndarray::Array2::<f32>::zeros((4, 8));
        let (_grad_in, grads) = moe.compute_gradients(&input, &output_grads);

        let mut found_bias2 = false;
        let mut sum_abs = 0.0f32;
        for g in &grads {
            if g.nrows() == 1 && g.ncols() == moe.config.num_experts {
                found_bias2 = true;
                for &v in g.iter() {
                    sum_abs += v.abs();
                }
            }
        }
        assert!(found_bias2);
        assert!(sum_abs > 0.0);
    }
}
