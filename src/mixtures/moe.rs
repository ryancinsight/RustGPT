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
    llm::Layer,
    mixtures::{
        gating::{GatingConfig, GatingStrategy},
        routing::{Router, RoutingConfig, RoutingResult, SelectionAlgorithm},
        threshold::ThresholdPredictor,
    },
};

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
                diversity_weight: *diversity_weight,
                metrics_avg_routing_prob: vec![0.0; *num_experts],
                metrics_diversity_score: 0.0,
            },
        }
    }

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
    pub activation: crate::richards::RichardsCurve,
    /// Softmax layer for probability normalization
    pub softmax: crate::softmax::Softmax,

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
    cached_activation: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_activated: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_logits: Option<ndarray::Array2<f32>>,
    #[serde(skip)]
    cached_output: Option<ndarray::Array2<f32>>,
}

impl ExpertSelector {
    /// Create a new expert selector with AutoDeco-inspired architecture
    pub fn new(embed_dim: usize, router_hidden_dim: usize, num_experts: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

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
        let activation =
            crate::richards::RichardsCurve::new_learnable(crate::richards::Variant::None); // Learnable activation replacing ReLU

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            norm,
            sigmoid,
            activation,
            softmax: crate::softmax::Softmax::new(),
            param_info: None,
            cached_input: None,
            cached_hidden: None,
            cached_normalized: None,
            cached_activation: None,
            cached_activated: None,
            cached_logits: None,
            cached_output: None,
        }
    }

    /// Predict expert routing probabilities using AutoDeco-style architecture
    ///
    /// Returns softmax-normalized probabilities in [0, 1] range suitable for expert selection
    /// Caches intermediate activations for gradient computation
    pub fn predict(&mut self, input: &ndarray::ArrayView2<f32>) -> ndarray::Array2<f32> {
        // Cache input for gradient computation (zero-copy where possible)
        self.cached_input = Some(input.to_owned());

        // First layer: W1 * x + b1
        let hidden = input.dot(&self.weights1) + &self.bias1;
        self.cached_hidden = Some(hidden.clone());

        // Apply Richards normalization for adaptive behavior
        let normalized = self.norm.forward(&hidden);
        self.cached_normalized = Some(normalized.clone());

        // Learned Richards activation replacing ReLU (avoid double conversion)
        let activation_output = self
            .activation
            .forward_matrix(&normalized.mapv(|x| x as f64))
            .mapv(|x| x as f32);
        self.cached_activation = Some(activation_output.clone());

        // Second layer: W2 * activated + b2
        let logits = activation_output.dot(&self.weights2) + &self.bias2;
        self.cached_logits = Some(logits.clone());
        self.cached_activated = Some(activation_output); // Cache after use

        // Softmax normalization for routing probabilities
        let output = self.softmax.forward(&logits.view());
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
                .map(|(idx, &prob)| (idx, prob))
                .collect();

            // Sort by probability (descending)
            expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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
        let _cached_activation = self
            .cached_activation
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
            let weight_shapes = vec![
                (self.weights1.nrows(), self.weights1.ncols()),
                (self.weights2.nrows(), self.weights2.ncols()),
            ];

            let bias_shapes = vec![self.bias1.len(), self.bias2.len()];

            let norm_params = self.norm.parameters();
            let activation_params = self.activation.weights().len();
            let sigmoid_params = self.sigmoid.weights().len();

            let total_params = self.parameters().map(|p| p.len()).sum::<usize>()
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
            let richards_gate_params = self.glu.gate_curve.weights().len();

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
    /// Cached expert outputs for gradient computation
    #[serde(skip)]
    cached_expert_outputs: Option<Vec<ndarray::Array2<f32>>>,
}

impl MixtureOfExperts {
    /// Create a new MoE layer
    pub fn new(embedding_dim: usize, router_hidden_dim: usize, config: ExpertRouterConfig) -> Self {
        let router = ExpertSelector::new(embedding_dim, router_hidden_dim, config.num_experts);

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
            cached_expert_outputs: None,
        }
    }

    /// Forward pass: predict routing → all experts process → weighted sum
    pub fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        // Cache input for gradient computation
        self.cached_input = Some(input.to_owned());

        // Router predicts routing probabilities for all tokens
        let routing_probs = self.router.predict(&input.view());
        self.cached_routing_probs = Some(routing_probs.clone());

        // Update routing metrics for training
        self.config.update_metrics(&routing_probs.view());

        // All experts process all tokens in parallel
        let mut expert_outputs = Vec::new();
        for expert in &mut self.experts {
            let output = expert.forward(input);
            expert_outputs.push(output);
        }
        self.cached_expert_outputs = Some(expert_outputs.clone());

        // Weighted sum of expert outputs using routing probabilities
        let mut output = ndarray::Array2::zeros(input.raw_dim());

        // Use zip to combine expert outputs with routing columns and accumulate results
        expert_outputs
            .into_iter()
            .zip(routing_probs.columns())
            .for_each(|(expert_output, routing_col)| {
                // Use zip to iterate over output rows and routing weights simultaneously
                output
                    .outer_iter_mut()
                    .zip(expert_output.outer_iter())
                    .zip(routing_col.iter())
                    .for_each(|((mut output_row, expert_row), &weight)| {
                        output_row.scaled_add(weight, &expert_row);
                    });
            });

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
        // Simplified backward: route gradients to all experts weighted by routing probabilities
        let routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("forward must be called before backward");

        let mut total_grad_input = ndarray::Array2::zeros(grads.raw_dim());

        // Use zip to iterate over experts and their corresponding routing columns simultaneously
        self.experts
            .iter_mut()
            .zip(routing_probs.columns())
            .for_each(|(expert, routing_col)| {
                // Weight gradients by routing probabilities for this expert using zip
                let weighted_grads: Vec<ndarray::Array1<f32>> = grads
                    .outer_iter()
                    .zip(routing_col.iter())
                    .map(|(grad_row, &weight)| grad_row.mapv(|x| x * weight))
                    .collect();

                // Convert Vec<Array1> to Array2 for expert backward pass
                let weighted_grads_2d = if !weighted_grads.is_empty() {
                    let nrows = weighted_grads.len();
                    let ncols = weighted_grads[0].len();
                    let flat_data = weighted_grads
                        .into_iter()
                        .flat_map(|row: ndarray::Array1<f32>| row.into_iter())
                        .collect::<Vec<f32>>();
                    ndarray::Array2::from_shape_vec((nrows, ncols), flat_data).unwrap()
                } else {
                    ndarray::Array2::zeros(grads.raw_dim())
                };

                // Get expert input gradients
                let expert_grad_input = expert.backward(&weighted_grads_2d, lr);

                // Weight input gradients back by routing probabilities using zip
                expert_grad_input
                    .outer_iter()
                    .zip(routing_col.iter())
                    .zip(total_grad_input.outer_iter_mut())
                    .for_each(|((grad_row, &weight), mut total_row)| {
                        total_row.scaled_add(weight, &grad_row);
                    });
            });

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
            total += self.router.activation.weights().len();

            total += self
                .experts
                .iter()
                .map(|expert| expert.glu.parameters())
                .sum::<usize>();

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
        let cached_routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let cached_expert_outputs = self
            .cached_expert_outputs
            .as_ref()
            .expect("forward must be called before compute_gradients");

        // 1. Route gradients to all experts weighted by routing probabilities
        let mut expert_output_grads =
            vec![ndarray::Array2::zeros(output_grads.raw_dim()); self.config.num_experts];
        for expert_idx in 0..self.config.num_experts {
            for token_idx in 0..output_grads.nrows() {
                let routing_weight = cached_routing_probs[[token_idx, expert_idx]];
                expert_output_grads[expert_idx]
                    .row_mut(token_idx)
                    .assign(&output_grads.row(token_idx).mapv(|x| x * routing_weight));
            }
        }

        // 2. Compute gradients for each expert
        let mut all_param_grads = Vec::new();
        let mut grad_input = ndarray::Array2::zeros(cached_input.raw_dim());

        for (expert_idx, expert) in self.experts.iter().enumerate() {
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

        // 3. Compute router gradients from the main loss
        // Router gradients based on how routing affects the weighted expert outputs
        let mut routing_grads = ndarray::Array2::zeros(cached_routing_probs.raw_dim());
        for token_idx in 0..cached_routing_probs.nrows() {
            let token_output_grad = output_grads.row(token_idx);
            for expert_idx in 0..self.config.num_experts {
                let expert_output = cached_expert_outputs[expert_idx].row(token_idx);
                let dot_product = token_output_grad
                    .iter()
                    .zip(expert_output.iter())
                    .map(|(&g, &o)| g * o)
                    .sum::<f32>();
                routing_grads[[token_idx, expert_idx]] = dot_product;
            }
        }

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

        // Compute softmax gradients manually: dL/d_logits = routing_grads * (routing_probs * (1 -
        // routing_probs)) But since routing_grads is already w.r.t. routing_probs, we need
        // to compute the Jacobian
        let routing_probs = self
            .cached_routing_probs
            .as_ref()
            .expect("routing probs must be cached");
        let mut d_logits = ndarray::Array2::zeros(routing_grads.raw_dim());

        for i in 0..routing_grads.nrows() {
            for j in 0..routing_grads.ncols() {
                let y_j = routing_probs[[i, j]];
                let mut grad_sum = 0.0;

                for k in 0..routing_grads.ncols() {
                    let y_k = routing_probs[[i, k]];
                    let dy_k_d_logits_j = if j == k {
                        y_j * (1.0 - y_j)
                    } else {
                        -y_j * y_k
                    };
                    grad_sum += routing_grads[[i, k]] * dy_k_d_logits_j;
                }

                d_logits[[i, j]] = grad_sum;
            }
        }

        // Second layer gradients
        let grad_weights2 = cached_activated.t().dot(&d_logits);
        let grad_bias2 = d_logits.sum_axis(ndarray::Axis(0));

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

        // First layer gradients
        let grad_weights1 = cached_input.t().dot(&d_hidden);
        let grad_bias1 = d_hidden.sum_axis(ndarray::Axis(0));

        // Activation parameter gradients
        let activation_grads = self
            .router
            .activation
            .grad_weights_matrix(&normalized_f64, &d_activated_f64);

        // Convert to the format expected by apply_gradients (Vec<Array2<f32>>)
        let router_grads = vec![
            grad_weights1,
            grad_bias1.insert_axis(ndarray::Axis(0)),
            grad_weights2,
            grad_bias2.insert_axis(ndarray::Axis(0)),
            ndarray::Array2::from_shape_vec(
                (1, activation_grads.len()),
                activation_grads.into_iter().map(|x| x as f32).collect(),
            )
            .unwrap(),
        ];
        all_param_grads.extend(router_grads);

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
            // RichardsGlu always has 5 parameters
            let num_expert_params = 5;

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

        // Apply router gradients (weights1, bias1, weights2, bias2, activation_params)
        let router_grad_count = 5; // weights1, bias1, weights2, bias2, activation
        if grad_idx + router_grad_count <= param_grads.len() {
            self.router.weights1.scaled_add(-lr, &param_grads[grad_idx]);
            self.router
                .bias1
                .scaled_add(-lr, &param_grads[grad_idx + 1].row(0));
            self.router
                .weights2
                .scaled_add(-lr, &param_grads[grad_idx + 2]);
            self.router
                .bias2
                .scaled_add(-lr, &param_grads[grad_idx + 3].row(0));

            // Apply activation parameter gradients
            let activation_grads_f64: Vec<f64> = param_grads[grad_idx + 4]
                .iter()
                .map(|&x| x as f64)
                .collect();
            self.router
                .activation
                .step(&activation_grads_f64, lr as f64);
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
                .sqrt();

        let expert_norm = self.experts.iter().map(|e| e.weight_norm()).sum::<f32>();

        router_norm + expert_norm
    }
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
        };

        let config = ExpertRouterConfig::from_router(&router);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.gating.num_active, 3);
        assert_eq!(config.expert_hidden_dim, 32);
        assert_eq!(config.gating.load_balance_weight, 0.1);
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

        // Verify that router gradients are included (5 matrices: weights1, bias1, weights2, bias2,
        // activation) Expert gradients come first, then router gradients
        let expected_router_grad_start = moe.experts.len() * 5; // 5 parameters per expert (w1, w2, w_out, richards_activation, gate_curve)
        assert!(
            param_grads.len() >= expected_router_grad_start + 5,
            "Should have gradients for all experts plus 5 router matrices"
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
        let original_expert_w1 = moe.experts[0].glu.w1.clone();

        // Apply gradients
        moe.apply_gradients(&param_grads, 0.01)
            .expect("Apply gradients should succeed");

        // Check that weights were updated
        assert_ne!(
            moe.router.weights1, original_router_w1,
            "Router weights should be updated"
        );
        assert_ne!(
            moe.experts[0].glu.w1, original_expert_w1,
            "Expert weights should be updated"
        );
    }
}
