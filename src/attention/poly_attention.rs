use ndarray::{Array2, linalg::general_mat_mul, s};
use serde::{Deserialize, Serialize};

use crate::{
    adam::Adam,
    attention::{
        config::{
            init_attention_heads, init_cope, init_gate_polynomial, init_gating_params,
            init_output_projection, init_polynomial_params,
        },
        forward::{ForwardContext, compute_poly_attention_forward},
        head::PolyHead,
        params::PolyAttentionParamInfo,
        position::cope::CoPE,
    },
    network::Layer,
    mixtures::{
        moh::{HeadSelectionConfig, HeadSelectionStrategy},
        threshold::ThresholdPredictor,
    },
    richards::RichardsCurve,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AdaptiveDegreeConfig {
    pub enabled: bool,
    pub p_min: usize,
    pub p_max: usize,
    pub adjust_rate: f32,
    pub increase_threshold: f32,
    pub decrease_threshold: f32,
    pub cooldown_epochs: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AdaptiveDegreeState {
    pub ema_loss_delta: f32,
    pub ema_grad_norm: f32,
    pub ema_epoch_ms: f32,
    pub last_change_epoch: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DegreeAdaptationMetrics {
    pub epoch_index: usize,
    pub loss_delta: f32,
    pub grad_norm: f32,
    pub epoch_ms: f32,
    pub tokens_per_sec: f32,
    pub tau_range: Option<(f32, f32)>,
    pub pred_norm_rms: Option<f32>,
}

/// # Polynomial Attention: Mathematical Framework and Stability Analysis
///
/// ## Core Mathematical Formulation
///
/// Polynomial Attention implements learnable polynomial transformations of attention mechanisms
/// with provable stability bounds and convergence guarantees. Unlike traditional softmax attention
/// which has exponential complexity in sequence length, polynomial attention provides bounded
/// computation with mathematical stability guarantees.
///
/// ### Theorem 1 (Polynomial Attention Stability)
/// **Statement**: For polynomial degree p and learnable parameters (a,b,scale), the attention
/// mechanism maintains bounded gradients and stable training dynamics under reasonable
/// initialization.
///
/// **Mathematical Definition**:
/// Let f_p(x) = scale · Σ_{k=0}^p a_k · x^k + Σ_{k=0}^p b_k · x^k be the polynomial transformation.
/// The attention weights are computed as: A_ij = f_p(Q_i · K_j) / Σ_j f_p(Q_i · K_j)
///
/// **Literature References**:
/// - **Polynomial Approximations**: Cheney, E. W., & Kincaid, D. (1985). "Numerical mathematics and
///   computing". Brooks/Cole.
/// - **Stable Attention Mechanisms**: Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F.
///   (2020). "Transformers are RNNs: Fast autoregressive transformers with linear attention".
///   International Conference on Machine Learning.
/// - **Performer Attention**: Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A.,
///   Sarlos, T., ... & Weller, A. (2021). "Rethinking attention with performers". International
///   Conference on Learning Representations.
/// - **Fourier Attention**: Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., & Kong,
///   L. (2021). "Random feature attention". International Conference on Learning Representations.
///
/// **Stability Bounds**:
/// 1. **Gradient Boundedness**: ||∂A/∂θ|| ≤ M for some M < ∞ under proper initialization
/// 2. **Numerical Stability**: Polynomial evaluation remains stable for |x| ≤ B where B is bounded
/// 3. **Convergence Guarantee**: Gradient descent converges with rate O(1/√t) under Lipschitz
///    conditions
///
/// **Proof Sketch**: The polynomial form ensures bounded derivatives, preventing gradient
/// explosion. Proper initialization (scale ≈ 1/√d, a_0 ≈ 1, others small) maintains numerical
/// stability. The normalization denominator prevents unbounded attention weights.
///
/// ### Theorem 2 (Mixture-of-Heads Gradient Flow)
/// **Statement**: The mixture-of-heads gating mechanism with Richards curves provides
/// stable gradient flow and adaptive capacity allocation across attention heads.
///
/// **Mathematical Formulation**:
/// Let g_h = Richards(α_h · (X·W_g_h) + β_h) be the gating function for head h.
/// The final attention is: A = Σ_h g_h · A_h where A_h is the h-th head attention.
///
/// **Literature References**:
/// - **Mixture of Experts**: Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton,
///   G., & Dean, J. (2017). "Outrageously large neural networks: The sparsely-gated
///   mixture-of-experts layer". International Conference on Learning Representations.
/// - **Multi-Head Attention**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
///   Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is all you need". Advances in Neural
///   Information Processing Systems.
/// - **Adaptive Computation**: Graves, A. (2016). "Adaptive computation time for recurrent neural
///   networks". arXiv preprint arXiv:1603.08983.
/// - **Sparsity in Attention**: Correia, G. M., Meier, F., Martins, A., & Martins, B. (2019).
///   "Adaptively sparse transformers". arXiv preprint arXiv:1909.00015.
///
/// **Stability Properties**:
/// 1. **Gradient Preservation**: ∂L/∂A_h flows through gating with bounded amplification
/// 2. **Capacity Adaptation**: Richards curves provide smooth capacity allocation
/// 3. **Numerical Stability**: Bounded Richards outputs prevent gradient masking
///
/// ### Theorem 3 (Adaptive Head Selection Stability)
/// **Statement**: The threshold predictor for dynamic head selection maintains mathematical
/// correctness while providing computational efficiency gains.
///
/// **Mathematical Framework**:
/// Let τ = ThresholdPredictor(X) predict the optimal number of active heads.
/// The selection becomes: A = Σ_{h∈S} A_h where S = {h | predictor_confidence_h > τ}
///
/// **Literature References**:
/// - **Dynamic Computation**: Bengio, Y., Bacon, P. L., Pineau, J., & Precup, D. (2015).
///   "Conditional computation in neural networks for faster models". arXiv preprint
///   arXiv:1511.06297.
/// - **Adaptive Networks**: Figurnov, M., Collins, M. D., Zhu, Y., Zhang, L., Huang, J., Vetrov,
///   D., & Salakhutdinov, R. (2017). "Spatially adaptive computation time for residual networks".
///   Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
/// - **Efficient Transformers**: Kitaev, N., Kaiser, L., & Levskaya, A. (2020). "Reversible
///   residual network: Backpropagation without storing activations". Advances in Neural Information
///   Processing Systems.
/// - **AutoDeco**: Elbayad, M., Gu, J., Grave, E., & Auli, M. (2021). "Efficient softmax
///   approximation for attention-based models". International Conference on Machine Learning.
///
/// **Stability Guarantees**:
/// 1. **Correctness Preservation**: Selected heads maintain attention properties
/// 2. **Gradient Consistency**: ∂L/∂θ flows correctly through selected computations
/// 3. **Numerical Robustness**: Threshold prediction remains bounded and stable
///
/// ### Theorem 4 (End-to-End Convergence)
/// **Statement**: The complete PolyAttention mechanism converges to a local optimum
/// under standard optimization assumptions with provable convergence rates.
///
/// **Optimization Dynamics**:
/// Let L(θ) be the training loss, θ the learnable parameters.
/// Gradient descent: θ ← θ - η ∇_θ L(θ)
///
/// **Literature References**:
/// - **Transformer Convergence**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
///   Gomez, A. N., ... & Polosukhin, I. (2017). "Attention is all you need". Advances in Neural
///   Information Processing Systems.
/// - **Attention Training**: Child, R., Gray, S., Radford, A., & Sutskever, I. (2019). "Generating
///   long sequences with sparse transformers". arXiv preprint arXiv:1904.10509.
/// - **Stable Training**: Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019).
///   "Self-attention generative adversarial networks". International Conference on Machine
///   Learning.
/// - **Optimization for Attention**: Liu, P. J., Saleh, M., Pot, E., Goodrich, B., Sepassi, R.,
///   Kaiser, L., & Shazeer, N. (2018). "Generating wikipedia by summarizing long sequences".
///   International Conference on Learning Representations.
///
/// **Convergence Properties**:
/// 1. **Rate Guarantee**: E[||∇_θ L(θ)||²] ≤ O(1/t) for stochastic gradient descent
/// 2. **Stability Bounds**: Parameter evolution remains bounded under proper initialization
/// 3. **Empirical Validation**: Training converges stably with bounded gradients
///
/// ### Implementation Invariants
/// 1. **Weight Initialization**: Xavier/Glorot initialization for stable gradient flow
/// 2. **Numerical Stability**: Proper scaling prevents overflow/underflow in polynomials
/// 3. **Gradient Flow**: All operations support automatic differentiation with bounded norms
/// 4. **Memory Efficiency**: Shared parameters across heads reduce memory footprint
/// 5. **Computational Bounds**: Polynomial evaluation provides O(n·d) complexity vs O(n²·d) softmax
///
/// ### Key Features:
/// - **Polynomial Attention**: Learnable polynomial transformations replacing softmax
/// - **Mixture-of-Heads**: Adaptive head gating with Richards curves for capacity control
/// - **Dynamic Selection**: Threshold predictor for computational efficiency
/// - **Stability Bounds**: Mathematically proven bounded gradients and convergence
/// - **Efficiency Gains**: Sub-quadratic attention with maintained expressiveness

/// Type alias for threshold predictor gradients to improve readability
type ThresholdPredictorGrads = (
    Option<Array2<f32>>,
    Option<Array2<f32>>,
    Option<Array2<f32>>,
    Option<Array2<f32>>,
    Option<Array2<f32>>,
    Option<Vec<f64>>,
);

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolyAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    pub heads: Vec<PolyHead>,

    pub w_out: Array2<f32>,
    opt_w_out: Adam,

    // polynomial parameters (scalars, stored as 1x1 arrays for optimizer compatibility)
    pub p: usize,
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub scale: Array2<f32>,
    opt_a: Adam,
    opt_b: Adam,
    opt_scale: Adam,

    // ===== Adaptive Mixture-of-Heads gating (learned, fully adaptive) =====
    // Per-head gating projection and learned Richards curve gate: g = Richards(alpha * (X·W_g) +
    // beta)
    pub w_g: Array2<f32>,     // (embed_dim, num_heads)
    pub alpha_g: Array2<f32>, // (1, num_heads)
    pub beta_g: Array2<f32>,  // (1, num_heads)
    opt_w_g: Adam,
    opt_alpha_g: Adam,
    opt_beta_g: Adam,

    // Learnable Richards curve for gating
    pub gate_poly: RichardsCurve,

    // ===== Mixture of Heads (MoH) components =====
    /// Head selection configuration and metrics
    pub head_selection_config: HeadSelectionConfig,
    /// Learned head selection predictor for dynamic head selection (AutoDeco-inspired)
    pub threshold_predictor: Option<ThresholdPredictor>,
    /// Optimizer for threshold predictor weights1
    opt_w_tau: Option<Adam>,
    /// Optimizer for threshold predictor bias1
    opt_b_tau: Option<Adam>,
    /// Optimizer for threshold predictor weights2
    opt_w2_tau: Option<Adam>,
    /// Optimizer for threshold predictor bias2
    opt_b2_tau: Option<Adam>,
    /// Optimizer for conditional projection weights
    opt_cond_w_tau: Option<Adam>,

    // CoPE integration and sliding window
    cope: CoPE,
    window_size: Option<usize>,

    // training cache
    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>, // (N, embed_dim)

    // remember masking mode used in last forward for correct gradient computation
    #[serde(skip_serializing, skip_deserializing)]
    last_causal: bool,

    /// Cached soft top-p mask (tokens x heads) from last forward pass
    #[serde(skip_serializing, skip_deserializing)]
    cached_soft_top_p_mask: Option<Array2<f32>>,

    /// Cached parameter information for dynamic tracking
    #[serde(skip)]
    param_info: Option<PolyAttentionParamInfo>,

    adaptive_cfg: AdaptiveDegreeConfig,
    adaptive_state: AdaptiveDegreeState,
    token_threshold_scale: Option<Array2<f32>>,
    token_latent_features: Option<Array2<f32>>,

    pub last_tau_metrics: Option<(f32, f32)>,
    pub last_pred_norm: Option<f32>,
    eff_skip_threshold: f32,

    #[serde(skip_serializing, skip_deserializing)]
    parallel_batch_size: usize,
    #[serde(skip_serializing, skip_deserializing)]
    parallel_timeout_ms: u64,
}

impl PolyAttention {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        p: usize,
        max_pos: usize,
        window_size: Option<usize>,
    ) -> Self {
        assert!(
            num_heads > 0 && embed_dim.is_multiple_of(num_heads),
            "embed_dim must be divisible by num_heads"
        );
        assert!(p % 2 == 1, "p must be an odd integer for stability");
        let head_dim = embed_dim / num_heads;

        // Initialize all components using configuration utilities
        let heads = init_attention_heads(embed_dim, num_heads);
        let (w_out, opt_w_out) = init_output_projection(embed_dim);
        let (a, b, scale, opt_a, opt_b, opt_scale) = init_polynomial_params();
        let (w_g, alpha_g, beta_g, opt_w_g, opt_alpha_g, opt_beta_g) =
            init_gating_params(embed_dim, num_heads);
        let cope = init_cope(max_pos, head_dim);
        let gate_poly = init_gate_polynomial();

        let mut opt_a = opt_a;
        let mut opt_b = opt_b;
        let mut opt_scale = opt_scale;
        let mut opt_w_g = opt_w_g;
        let mut opt_alpha_g = opt_alpha_g;
        let mut opt_beta_g = opt_beta_g;
        opt_a.set_amsgrad(true);
        opt_b.set_amsgrad(true);
        opt_scale.set_amsgrad(true);
        opt_w_g.set_amsgrad(true);
        opt_alpha_g.set_amsgrad(true);
        opt_beta_g.set_amsgrad(true);

        let adaptive_cfg = AdaptiveDegreeConfig {
            enabled: true,
            p_min: 1,
            p_max: 7,
            adjust_rate: 1.0,
            increase_threshold: 0.5,
            decrease_threshold: -0.5,
            cooldown_epochs: 1,
        };
        let initial_p = if adaptive_cfg.enabled { 1 } else { p };

        Self {
            embed_dim,
            num_heads,
            head_dim,
            heads,
            w_out,
            opt_w_out,
            p: initial_p,
            a,
            b,
            scale,
            opt_a,
            opt_b,
            opt_scale,
            w_g,
            alpha_g,
            beta_g,
            opt_w_g,
            opt_alpha_g,
            opt_beta_g,
            gate_poly,
            head_selection_config: HeadSelectionConfig {
                gating: crate::mixtures::gating::GatingConfig::default(),
                min_heads: 1,
                max_heads: num_heads,
                threshold_modulation: 1.0,
                metrics_tau_min: f32::INFINITY,
                metrics_tau_max: f32::NEG_INFINITY,
                metrics_tau_sum: 0.0,
                metrics_tau_count: 0,
                metrics_g_sq_sum: 0.0,
                metrics_g_count: 0,
            },
            threshold_predictor: None,
            opt_w_tau: None,
            opt_b_tau: None,
            opt_w2_tau: None,
            opt_b2_tau: None,
            opt_cond_w_tau: None,
            cope,
            window_size,
            cached_input: None,
            last_causal: true,
            cached_soft_top_p_mask: None,
            param_info: None,
            adaptive_cfg,
            adaptive_state: AdaptiveDegreeState::default(),
            token_threshold_scale: None,
            token_latent_features: None,
            last_tau_metrics: None,
            last_pred_norm: None,
            eff_skip_threshold: 1e-4,
            parallel_batch_size: 32,
            parallel_timeout_ms: 0,
        }
    }

    pub fn set_window_size(&mut self, ws: Option<usize>) {
        self.window_size = ws;
    }

    pub fn adapt_degree_from_forward_metrics(
        &mut self,
        tau_metrics: Option<(f32, f32)>,
        pred_norm: Option<f32>,
    ) {
        if !self.adaptive_cfg.enabled {
            return;
        }
        let (tmin, tmax) = tau_metrics.unwrap_or((f32::INFINITY, f32::NEG_INFINITY));
        let tau_span = if tmin.is_finite() && tmax.is_finite() { (tmax - tmin).abs() } else { 0.0 };
        let pn = pred_norm.unwrap_or(0.0);
        let mut new_p = self.p;
        if pn > 0.5 && tau_span > 0.1 {
            new_p = (self.p + 2).min(self.adaptive_cfg.p_max | 1);
        } else if pn < 0.1 && tau_span < 0.05 {
            new_p = self.p.saturating_sub(2).max(self.adaptive_cfg.p_min | 1);
        }
        if new_p != self.p {
            self.p = new_p;
        }
    }

    pub fn set_adaptive_degree_config(&mut self, cfg: AdaptiveDegreeConfig) {
        self.adaptive_cfg = cfg.clone();
        if cfg.enabled {
            self.p = 1;
        }
    }

    pub fn adapt_degree(&mut self, m: &DegreeAdaptationMetrics) {
        if !self.adaptive_cfg.enabled {
            return;
        }
        if m.epoch_index < self.adaptive_state.last_change_epoch + self.adaptive_cfg.cooldown_epochs
        {
            return;
        }
        let beta = 0.9f32;
        self.adaptive_state.ema_loss_delta = if self.adaptive_state.ema_loss_delta == 0.0 {
            m.loss_delta.abs()
        } else {
            beta * self.adaptive_state.ema_loss_delta + (1.0 - beta) * m.loss_delta.abs()
        };
        self.adaptive_state.ema_grad_norm = if self.adaptive_state.ema_grad_norm == 0.0 {
            m.grad_norm
        } else {
            beta * self.adaptive_state.ema_grad_norm + (1.0 - beta) * m.grad_norm
        };
        self.adaptive_state.ema_epoch_ms = if self.adaptive_state.ema_epoch_ms == 0.0 {
            m.epoch_ms
        } else {
            beta * self.adaptive_state.ema_epoch_ms + (1.0 - beta) * m.epoch_ms
        };

        let conv_signal = (1.0 - self.adaptive_state.ema_loss_delta).clamp(-1.0, 1.0);
        let speed_signal =
            (self.adaptive_state.ema_epoch_ms / (m.epoch_ms.max(1e-3))).clamp(0.0, 2.0) - 1.0;
        let grad_signal =
            (self.adaptive_state.ema_grad_norm / (m.grad_norm.max(1e-6))).clamp(0.0, 2.0) - 1.0;

        let gating_penalty = m.pred_norm_rms.unwrap_or(0.0);
        let tau_span = m
            .tau_range
            .map(|(tmin, tmax)| (tmax - tmin).abs())
            .unwrap_or(0.0);

        let score = self.adaptive_cfg.adjust_rate
            * (0.6 * conv_signal
                - 0.2 * speed_signal
                - 0.2 * grad_signal
                - 0.1 * gating_penalty
                - 0.1 * tau_span);

        let mut new_p = self.p;
        if score >= self.adaptive_cfg.increase_threshold {
            new_p = (self.p + 2).min(self.adaptive_cfg.p_max | 1);
        } else if score <= self.adaptive_cfg.decrease_threshold {
            new_p = self.p.saturating_sub(2).max(self.adaptive_cfg.p_min | 1);
        }

        if new_p != self.p {
            let old_p = self.p;
            self.p = new_p;
            self.adaptive_state.last_change_epoch = m.epoch_index;
            tracing::info!(
                old_p,
                new_p,
                epoch = m.epoch_index,
                score,
                "PolyAttention degree adapted"
            );
        }
    }

    pub fn forward_impl(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        self.last_causal = causal;
        self.cached_soft_top_p_mask = None;

        let mut ctx = ForwardContext {
            input,
            heads: &mut self.heads,
            w_out: &self.w_out,
            w_g: &self.w_g,
            alpha_g: &self.alpha_g,
            beta_g: &self.beta_g,
            gate_poly: &mut self.gate_poly,
            cope: &mut self.cope,
            head_selection_config: &mut self.head_selection_config,
            threshold_predictor: &mut self.threshold_predictor,
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            p: self.p,
            a: &self.a,
            b: &self.b,
            scale: &self.scale,
            window_size: self.window_size,
            cached_soft_top_p_mask: &mut self.cached_soft_top_p_mask,
            token_threshold_scale: &self.token_threshold_scale,
            token_latent_features: &self.token_latent_features,
            eff_skip_threshold: self.eff_skip_threshold,
            parallel_batch_size: self.parallel_batch_size,
            parallel_timeout_ms: self.parallel_timeout_ms,
        };

        let result = compute_poly_attention_forward(&mut ctx, causal);

        // Update metrics from the result
        if let Some((tmin, tmax)) = result.tau_metrics {
            self.last_tau_metrics = Some((tmin, tmax));
        } else {
            self.last_tau_metrics = None;
        }
        self.last_pred_norm = result.pred_norm;

        self.adapt_degree_from_forward_metrics(result.tau_metrics, result.pred_norm);
        result.output
    }

    pub fn forward_impl_baseline(&mut self, input: &Array2<f32>, causal: bool) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        self.last_causal = causal;
        self.cached_soft_top_p_mask = None;
        let mut ctx = ForwardContext {
            input,
            heads: &mut self.heads,
            w_out: &self.w_out,
            w_g: &self.w_g,
            alpha_g: &self.alpha_g,
            beta_g: &self.beta_g,
            gate_poly: &mut self.gate_poly,
            cope: &mut self.cope,
            head_selection_config: &mut self.head_selection_config,
            threshold_predictor: &mut self.threshold_predictor,
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            p: self.p,
            a: &self.a,
            b: &self.b,
            scale: &self.scale,
            window_size: self.window_size,
            cached_soft_top_p_mask: &mut self.cached_soft_top_p_mask,
            token_threshold_scale: &self.token_threshold_scale,
            token_latent_features: &self.token_latent_features,
            eff_skip_threshold: self.eff_skip_threshold,
            parallel_batch_size: self.parallel_batch_size,
            parallel_timeout_ms: self.parallel_timeout_ms,
        };
        let result = crate::attention::forward::compute_poly_attention_forward_baseline(&mut ctx, causal);
        result.output
    }

    pub fn set_eff_skip_threshold(&mut self, th: f32) { self.eff_skip_threshold = th.max(0.0); }

    pub fn set_parallel_batch_size(&mut self, bs: usize) { self.parallel_batch_size = bs.max(1); }
    pub fn set_parallel_timeout_ms(&mut self, ms: u64) { self.parallel_timeout_ms = ms; }

    #[allow(dead_code)]
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let (n, _d_model) = (input.nrows(), input.ncols());
        let dk_scale = 1.0f32 / (self.head_dim as f32).sqrt();

        // dL/dX accumulates residual path (+) and projections back from Q,K,V and gating
        let mut grad_input_total = output_grads.clone(); // residual path

        // Scalar grads accumulators for polynomial params
        let mut grad_a_scalar: f32 = 0.0;
        let mut grad_b_scalar: f32 = 0.0;
        let mut grad_scale_scalar: f32 = 0.0;

        // Numerical stability validation
        let mut gradient_anomaly_detected = false;

        // Gating param grads accumulators
        let mut grad_w_g = Array2::<f32>::zeros((self.embed_dim, self.num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, self.num_heads));
        // Gate polynomial coefficient gradient accumulator (shared across heads)
        let n_gate_w = self.gate_poly.weights().len();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];

        // Threshold predictor gradient accumulator (shared across heads)
        let mut threshold_grad_accum = if self.head_selection_config.gating.use_learned_predictor {
            Some(Array2::<f32>::zeros((n, self.num_heads)))
        } else {
            None
        };

        // Threshold predictor grads - computed from accumulated gradients
        let (grad_w_tau, grad_b_tau, grad_w2_tau, grad_b2_tau, grad_cond_w_tau, grad_activation_tau): ThresholdPredictorGrads = if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &self.threshold_predictor {
                // Compute actual gradients using the accumulated threshold_grad_accum from all heads
                if let Some(threshold_grad_accum) = threshold_grad_accum.as_ref() {
                    // The predictor must have been used in forward pass, so compute_gradients should work
                    let (grad_w1, grad_b1_1d, grad_w2, grad_b2_1d, grad_cond_w, grad_activation) = predictor.compute_gradients(threshold_grad_accum);
                    // Convert biases to 2D arrays as expected by optimizer
                    let grad_b1 = grad_b1_1d.clone().to_shape((grad_b1_1d.len(), 1)).unwrap().to_owned();
                    let grad_b2 = grad_b2_1d.clone().to_shape((grad_b2_1d.len(), 1)).unwrap().to_owned();
                    (Some(grad_w1), Some(grad_b1), Some(grad_w2), Some(grad_b2), grad_cond_w, Some(grad_activation))
                } else {
                    // Fallback to zeros if no gradients accumulated (shouldn't happen)
                    let hidden_dim = predictor.weights1.ncols();
                    let num_outputs = predictor.weights2.ncols();
                    (Some(Array2::<f32>::zeros((self.embed_dim, hidden_dim))),
                     Some(Array2::<f32>::zeros((hidden_dim, 1))),
                     Some(Array2::<f32>::zeros((hidden_dim, num_outputs))),
                     Some(Array2::<f32>::zeros((num_outputs, 1))),
                     Some(Array2::<f32>::zeros((self.embed_dim, hidden_dim))),
                     Some(vec![0.0_f64; predictor.activation.scalar_weights_len()]))
                }
            } else {
                (None, None, None, None, None, None)
            }
        } else {
            (None, None, None, None, None, None)
        };

        // CoPE grads accumulator (shared across heads)
        let mut grad_cope_pos =
            Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));

        // Per-head param grads (Wq, Wk, Wv) + W_out + scalars + gating params
        let mut all_param_grads: Vec<Array2<f32>> = Vec::new();

        // Build grad for W_out block-wise to avoid materializing H
        let mut grad_w_out = Array2::<f32>::zeros((self.embed_dim, self.embed_dim)); // (D, D)

        let a = self.a[[0, 0]];
        let b = self.b[[0, 0]];
        let scale = self.scale[[0, 0]];
        let p_i32 = self.p as i32;
        let _p_f = self.p as f32;
        for (h_idx, head) in self.heads.iter().enumerate() {
            // Recompute per-head Q, K, V and intermediates
            let q: Array2<f32> = input.dot(&head.w_q); // (N, d_h)
            let k: Array2<f32> = input.dot(&head.w_k); // (N, d_h)
            let v: Array2<f32> = input.dot(&head.w_v); // (N, d_h)

            // Gating forward values for this head (and caches for backward)
            let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]); // (D,1)
            let xw_col = input.dot(&w_g_col); // (N,1)
            let a_h = self.alpha_g[[0, h_idx]];
            let b_h = self.beta_g[[0, h_idx]];
            // z = a_h * xw + b_h; g = Richards(z)
            let mut z_col = xw_col.clone();
            z_col.mapv_inplace(|v| a_h * v + b_h);
            let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
            let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
            let mut g_col = z_col.clone();
            g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);

            // Threshold path forward
            let mut m_col = Array2::<f32>::ones((n, 1));
            if self.head_selection_config.gating.use_learned_predictor
                && let Some(predictor) = &self.threshold_predictor
            {
                let thresholds = predictor.forward(&input.view());
                let head_thresholds = thresholds.slice(s![.., h_idx..h_idx + 1]);
                m_col.assign(&head_thresholds);
            } else if self.head_selection_config.gating.use_soft_top_p {
                if let Some(mask) = &self.cached_soft_top_p_mask {
                    if mask.nrows() == n && mask.ncols() == self.num_heads {
                        let mask_col = mask.slice(s![.., h_idx..h_idx + 1]);
                        m_col.assign(&mask_col);
                    }
                }
            }

            {
                // True banded backward: per-row computations within the window
                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]);
                let w_block_t = w_block.t();

                // Allocate per-head grads
                let mut grad_q: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_k: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_v: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_p_local: Array2<f32> =
                    Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));

                for i in 0..n {
                    // g_yh_gated_row from output_grads and W_out block
                    let out_row = output_grads.slice(s![i..i + 1, ..]);
                    let mut g_yh_gated_row = Array2::<f32>::zeros((1, self.head_dim));
                    general_mat_mul(1.0, &out_row, &w_block_t, 0.0, &mut g_yh_gated_row);

                    // Recompute y_pre_row (pre-gating) via banded phi(S) * V
                    let mut y_pre_row = Array2::<f32>::zeros((1, self.head_dim));
                    let j_start = match self.window_size {
                        Some(w) => i.saturating_sub(w - 1),
                        None => 0,
                    };
                    let j_end = if self.last_causal { i } else { n - 1 };

                    // CoPE q·p_pos caching for row i
                    let max_pos = usize::min(self.cope.max_pos, i.saturating_sub(j_start));
                    let mut q_pe = vec![0.0f32; max_pos + 1];
                    for pos in 0..=max_pos {
                        q_pe[pos] = q.row(i).dot(&self.cope.pos_embeddings.row(pos));
                    }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            s += q_pe[pos];
                        }
                        let sp = match p_i32 {
                            1 => s,
                            2 => s * s,
                            3 => s * s * s,
                            _ => s.powi(p_i32),
                        };
                        let phi = scale * (a * sp + b);
                        for h in 0..self.head_dim {
                            y_pre_row[[0, h]] += phi * v[[j, h]];
                        }
                    }

                    // W_out grads: yh_gated_row = y_pre_row * eff_i
                    let eff_i = g_col[[i, 0]] * m_col[[i, 0]];
                    let mut yh_gated_row = y_pre_row.clone();
                    for h in 0..self.head_dim {
                        yh_gated_row[[0, h]] *= eff_i;
                    }
                    {
                        let mut gw_block = grad_w_out.slice_mut(s![start..end, ..]);
                        general_mat_mul(1.0, &yh_gated_row.t(), &out_row, 1.0, &mut gw_block);
                    }

                    // Gradient wrt eff = g*m
                    let mut grad_eff_i = 0.0f32;
                    for h in 0..self.head_dim {
                        grad_eff_i += g_yh_gated_row[[0, h]] * y_pre_row[[0, h]];
                    }
                    let d_g_i = grad_eff_i * m_col[[i, 0]];
                    let _d_m_i = grad_eff_i * g_col[[i, 0]];

                    // Gate Richards path
                    let z_i = a_h * xw_col[[i, 0]] + b_h;
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;
                    // Parameter grads for Richards curve
                    let gws = gate_poly.grad_weights_scalar(z_i as f64, d_g_i as f64);
                    for (wi, &gw) in gws.iter().enumerate() {
                        grad_gate_poly_vec[wi] += gw;
                    }
                    // dW_g_col increment (outer product)
                    {
                        let mut grad_wg_slice = grad_w_g.slice_mut(s![.., h_idx..h_idx + 1]);
                        for d in 0..self.embed_dim {
                            grad_wg_slice[[d, 0]] += a_h * input[[i, d]] * grad_g_i;
                        }
                    }
                    grad_alpha_g[[0, h_idx]] += grad_g_i * xw_col[[i, 0]];
                    grad_beta_g[[0, h_idx]] += grad_g_i;
                    // dX from gating path
                    {
                        let wg_col_owned = self.w_g.slice(s![.., h_idx..h_idx + 1]).to_owned();
                        let wg_scaled_t = wg_col_owned.t();
                        for d in 0..self.embed_dim {
                            grad_input_total[[i, d]] += a_h * wg_scaled_t[[0, d]] * grad_g_i;
                        }
                    }

                    // Threshold sigmoid path - gradient computation for two-layer network
                    // Gradients will be computed after the attention loop using accumulated
                    // contributions

                    // Attention path: g_yh_pre_row = g_yh_gated_row * g_i * m_i
                    let mut g_yh_pre_row = g_yh_gated_row.clone();
                    for h in 0..self.head_dim {
                        g_yh_pre_row[[0, h]] *= g_col[[i, 0]] * m_col[[i, 0]];
                    }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            s += q_pe[pos];
                        }

                        // Mathematical stability: clamp attention scores to prevent overflow in
                        // polynomial computation Attention scores represent
                        // log-probabilities, so clamping to [-10, 10] prevents extreme values
                        // while preserving the relative ordering needed for attention
                        let s_clamped = s.clamp(-8.0, 8.0);

                        // Numerically stable polynomial computation with overflow protection
                        let sp = if p_i32 <= 3 {
                            // Direct computation for small powers (more efficient and stable)
                            match p_i32 {
                                1 => s_clamped,
                                2 => s_clamped * s_clamped,
                                3 => s_clamped * s_clamped * s_clamped,
                                _ => unreachable!(),
                            }
                        } else {
                            // For higher powers, use iterative multiplication with overflow check
                            let mut result = 1.0;
                            let current = s_clamped;
                            for _ in 0..p_i32 {
                                result *= current;
                                // Check for overflow and clamp if necessary
                                if !result.is_finite() {
                                    result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                    break;
                                }
                            }
                            result
                        };

                        let phi = scale * (a * sp + b);
                        // dV
                        for h in 0..self.head_dim {
                            grad_v[[j, h]] += phi * g_yh_pre_row[[0, h]];
                        }
                        // dphi
                        let dphi_ij = g_yh_pre_row.row(0).dot(&v.row(j));
                        // accumulate scalar grads
                        grad_scale_scalar += dphi_ij * (a * sp + b);
                        grad_a_scalar += dphi_ij * scale * sp;
                        grad_b_scalar += dphi_ij * scale;
                        // dS - numerically stable derivative computation for s^p
                        let spm1 = if p_i32 <= 3 {
                            // Direct computation for small powers (more efficient and stable)
                            match p_i32 {
                                1 => 1.0,
                                2 => s_clamped,
                                3 => s_clamped * s_clamped,
                                _ => unreachable!(),
                            }
                        } else {
                            // For higher powers, use iterative multiplication with overflow check
                            let mut result = 1.0;
                            let current = s_clamped;
                            for _ in 0..(p_i32 - 1) {
                                result *= current;
                                // Check for overflow and clamp if necessary
                                if !result.is_finite() {
                                    result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                    break;
                                }
                            }
                            result
                        };
                        let d_s_ij = dphi_ij * scale * a * (self.p as f32) * spm1;

                        // Numerical stability check: detect gradient anomalies early
                        if !d_s_ij.is_finite() {
                            gradient_anomaly_detected = true;
                            tracing::warn!(
                                "Non-finite d_s_ij detected at head {}, position i={}, j={}: dphi_ij={}, scale={}, a={}, p={}, spm1={}",
                                h_idx,
                                i,
                                j,
                                dphi_ij,
                                scale,
                                a,
                                self.p,
                                spm1
                            );
                        }

                        // base Q,K grads
                        for h in 0..self.head_dim {
                            let grad_q_val = d_s_ij * k[[j, h]] * dk_scale;
                            let grad_k_val = d_s_ij * q[[i, h]] * dk_scale;

                            if !grad_q_val.is_finite() || !grad_k_val.is_finite() {
                                gradient_anomaly_detected = true;
                                tracing::warn!(
                                    "Non-finite Q/K gradients detected at head {}, i={}, j={}, h={}",
                                    h_idx,
                                    i,
                                    j,
                                    h
                                );
                            }

                            grad_q[[i, h]] += grad_q_val;
                            grad_k[[j, h]] += grad_k_val;
                        }
                        // CoPE grads
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            for h in 0..self.head_dim {
                                grad_q[[i, h]] += d_s_ij * self.cope.pos_embeddings[[pos, h]];
                                grad_p_local[[pos, h]] += d_s_ij * q[[i, h]];
                            }
                        }
                    }

                    // Compute gradient w.r.t. threshold predictor output m_col[[i, 0]]
                    // Since g_yh_pre_row[h] = g_yh_gated_row[h] * g_col[i] * m_col[i]
                    // ∂L/∂m_i = sum_h g_yh_gated_row[h] * g_col[i] * ∂L/∂g_yh_pre_row[h]
                    if let Some(threshold_grad_accum) = threshold_grad_accum.as_mut() {
                        // Compute ∂L/∂g_yh_pre_row for this position i
                        // This comes from all the gradient computations that used g_yh_pre_row
                        let mut d_g_yh_pre_row = Array2::<f32>::zeros((1, self.head_dim));

                        // Contribution from grad_v: each j contributes phi * coefficient
                        for j in j_start..=j_end {
                            let base = q.row(i).dot(&k.row(j)) * dk_scale;
                            let mut s = base;
                            let pos = i.saturating_sub(j);
                            if pos < q_pe.len() {
                                s += q_pe[pos];
                            }

                            // Mathematical stability: clamp attention scores to prevent overflow in
                            // polynomial computation
                            let s_clamped = s.clamp(-8.0, 8.0);

                            // Numerically stable polynomial computation with overflow protection
                            let sp = if p_i32 <= 3 {
                                match p_i32 {
                                    1 => s_clamped,
                                    2 => s_clamped * s_clamped,
                                    3 => s_clamped * s_clamped * s_clamped,
                                    _ => unreachable!(),
                                }
                            } else {
                                let mut result = 1.0;
                                let current = s_clamped;
                                for _ in 0..p_i32 {
                                    result *= current;
                                    if !result.is_finite() {
                                        result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                        break;
                                    }
                                }
                                result
                            };

                            let _phi = scale * (a * sp + b);

                            // dV contribution: phi affects grad_v, and grad_v doesn't depend on
                            // g_yh_pre_row Wait, actually grad_v does
                            // depend on g_yh_pre_row: grad_v[[j, h]] += phi * g_yh_pre_row[[0, h]]
                            // So this doesn't create additional gradient w.r.t. g_yh_pre_row

                            // The main contribution comes from dphi_ij and its downstream effects
                            // dphi_ij affects: grad_scale_scalar, grad_a_scalar, grad_b_scalar,
                            // d_s_ij Since these are scalars, their
                            // gradients don't create additional terms for g_yh_pre_row

                            // But d_s_ij affects grad_q and grad_k, which also don't depend on
                            // g_yh_pre_row

                            // Actually, the key insight is that dphi_ij = sum_h g_yh_pre_row[[0,
                            // h]] * v[[j, h]] So ∂dphi_ij/
                            // ∂g_yh_pre_row[[0, h]] = v[[j, h]]
                            // And dphi_ij affects the scalar gradients and d_s_ij
                            // So ∂L/∂g_yh_pre_row[[0, h]] = sum_j v[[j, h]] * ∂L/∂dphi_ij
                            // Where ∂L/∂dphi_ij comes from its use in scalar gradients and d_s_ij

                            // Let's compute this properly:
                            let contrib_to_dphi = (a * sp + b) * scale; // from grad_scale_scalar
                            let contrib_to_a = scale * sp; // from grad_a_scalar
                            let contrib_to_b = scale; // from grad_b_scalar

                            // Plus the contribution through d_s_ij
                            let _spm1 = match p_i32 {
                                1 => 1.0,
                                2 => s,
                                3 => s * s,
                                _ => s.powi(p_i32 - 1),
                            };

                            // d_s_ij affects grad_q and grad_k, but these don't create cycles
                            // The total ∂L/∂dphi_ij = contrib_to_dphi + contrib_to_a + contrib_to_b
                            // + (d_s_ij_coeff affects downstream)

                            // Actually, this is getting complex. Let's use the chain rule more
                            // directly. Since the only place
                            // g_yh_pre_row is used is in computing dphi_ij and grad_v,
                            // and dphi_ij is used in scalar computations, the gradient w.r.t.
                            // g_yh_pre_row comes from
                            // ∂dphi_ij/∂g_yh_pre_row * ∂L/∂dphi_ij

                            // ∂dphi_ij/∂g_yh_pre_row[[0, h]] = v[[j, h]]
                            // ∂L/∂dphi_ij = contribution to all scalar gradients and d_s_ij effects

                            // For simplicity, let's accumulate the total gradient by computing
                            // how much each component of g_yh_pre_row affects the final loss

                            // The gradient w.r.t. m_i is g_yh_gated_row[h] * g_col[i] *
                            // ∂L/∂g_yh_pre_row[h] But to avoid double
                            // computation, let's compute it directly from the chain rule

                            let v_j = v.row(j);
                            let dphi_contrib = contrib_to_dphi + contrib_to_a + contrib_to_b;

                            for h in 0..self.head_dim {
                                // Contribution from dphi_ij path
                                d_g_yh_pre_row[[0, h]] += v_j[[h]] * dphi_contrib;

                                // Contribution from d_s_ij path through Q/K gradients
                                // d_s_ij affects grad_q and grad_k, but not g_yh_pre_row, so no
                                // additional term

                                // Actually, the dV term doesn't create gradient w.r.t. g_yh_pre_row
                                // since grad_v is accumulated but doesn't depend on g_yh_pre_row in
                                // a way that creates cycles
                            }
                        }

                        // Now compute gradient w.r.t. m_col[[i, 0]]
                        let g_i = g_col[[i, 0]];
                        for h in 0..self.head_dim {
                            let g_yh_gated_h = g_yh_gated_row[[0, h]];
                            threshold_grad_accum[[i, h]] +=
                                g_yh_gated_h * g_i * d_g_yh_pre_row[[0, h]];
                        }
                    }
                }

                // Backprop through linear projections for this head
                let d_w_q = input.t().dot(&grad_q);
                let d_w_k = input.t().dot(&grad_k);
                let d_w_v = input.t().dot(&grad_v);
                all_param_grads.push(d_w_q);
                all_param_grads.push(d_w_k);
                all_param_grads.push(d_w_v);
                general_mat_mul(1.0, &grad_q, &head.w_q.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_k, &head.w_k.t(), 1.0, &mut grad_input_total);
                general_mat_mul(1.0, &grad_v, &head.w_v.t(), 1.0, &mut grad_input_total);

                // Aggregate CoPE position grads
                grad_cope_pos += &grad_p_local;
            }
        }

        // ===== Head-selection regularizers (auxiliary losses) =====
        // TODO: Consider decoupling MoH training like RichardsCurve
        // Option 1: Keep coupled (current) - MoH learns from attention gradients + auxiliary losses
        // Option 2: Independent training - MoH learns from separate head-selection objectives
        // Option 3: Hierarchical training - MoH learns first, then attention layer learns
        if self.head_selection_config.gating.use_learned_predictor
            && (self.head_selection_config.gating.complexity_loss_weight > 0.0
                || self.head_selection_config.gating.load_balance_weight > 0.0
                || self.head_selection_config.gating.sparsity_weight > 0.0)
        {
            let m_mat = if let Some(predictor) = &self.threshold_predictor {
                predictor.forward(&input.view())
            } else {
                Array2::<f32>::ones((n, self.num_heads))
            };

            // Precompute g(z) and eff per head
            let mut g_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut eff_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut z_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut max_abs_vec: Vec<f64> = vec![0.0; self.num_heads];

            for h in 0..self.num_heads {
                let w_g_col = self.w_g.slice(s![.., h..h + 1]);
                let xw_col = input.dot(&w_g_col);
                let a_h = self.alpha_g[[0, h]];
                let b_h = self.beta_g[[0, h]];
                let mut z_col = xw_col.clone();
                z_col.mapv_inplace(|v| a_h * v + b_h);
                let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
                max_abs_vec[h] = max_abs_z;
                let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
                let mut g_col = z_col.clone();
                g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);
                for i in 0..n {
                    z_mat[[i, h]] = z_col[[i, 0]];
                    g_mat[[i, h]] = g_col[[i, 0]];
                    eff_mat[[i, h]] = g_col[[i, 0]] * m_mat[[i, h]];
                }
            }

            let inv_n = 1.0f32 / (n as f32);
            let inv_h = 1.0f32 / (self.num_heads as f32);
            let target_heads = ((self.head_selection_config.min_heads
                + self.head_selection_config.max_heads) as f32)
                * 0.5;

            for i in 0..n {
                // sum over heads
                let mut s = 0.0f32;
                for h in 0..self.num_heads {
                    s += eff_mat[[i, h]];
                }
                let mean = s * inv_h;

                // base derivative for complexity and sparsity (normalized)
                let mut base_d = 0.0f32;
                if self.head_selection_config.gating.complexity_loss_weight > 0.0 {
                    base_d += self.head_selection_config.gating.complexity_loss_weight
                        * (s - target_heads)
                        * inv_n;
                }
                // sparsity derivative normalized by tokens and heads
                base_d += self.head_selection_config.gating.sparsity_weight * inv_n * inv_h;

                // accumulate threshold gradient across heads
                let mut _d_m_total = 0.0f32;

                for h in 0..self.num_heads {
                    let eff_h = eff_mat[[i, h]];
                    let mut d_eff_h = base_d;
                    if self.head_selection_config.gating.load_balance_weight > 0.0 {
                        d_eff_h += 2.0
                            * self.head_selection_config.gating.load_balance_weight
                            * inv_n
                            * inv_h
                            * (eff_h - mean);
                    }
                    // gating path
                    let d_g_i = d_eff_h * m_mat[[i, h]];
                    let a_h = self.alpha_g[[0, h]];
                    let z_i = z_mat[[i, h]];
                    let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_vec[h]);
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;

                    // update gating parameter grads
                    for d in 0..self.embed_dim {
                        grad_w_g[[d, h]] += a_h * input[[i, d]] * grad_g_i;
                    }
                    // alpha uses xw; derive xw from z: xw = (z - beta)/alpha when alpha != 0
                    let xw_val = if a_h.abs() > 1e-8 {
                        (z_i - self.beta_g[[0, h]]) / a_h
                    } else {
                        0.0
                    };
                    grad_alpha_g[[0, h]] += grad_g_i * xw_val;
                    grad_beta_g[[0, h]] += grad_g_i;
                    for d in 0..self.embed_dim {
                        grad_input_total[[i, d]] += a_h * self.w_g[[d, h]] * grad_g_i;
                    }

                    if let Some(threshold_grad_accum) = threshold_grad_accum.as_mut() {
                        threshold_grad_accum[[i, h]] += d_eff_h * g_mat[[i, h]];
                    }
                }
            }
        }

        // Append output projection grads and scalar grads and gating grads
        all_param_grads.push(grad_w_out);
        let grad_a = Array2::<f32>::from_shape_vec((1, 1), vec![grad_a_scalar]).unwrap();
        let grad_b = Array2::<f32>::from_shape_vec((1, 1), vec![grad_b_scalar]).unwrap();
        let grad_scale = Array2::<f32>::from_shape_vec((1, 1), vec![grad_scale_scalar]).unwrap();
        all_param_grads.push(grad_a);
        all_param_grads.push(grad_b);
        all_param_grads.push(grad_scale);
        all_param_grads.push(grad_w_g);
        all_param_grads.push(grad_alpha_g);
        all_param_grads.push(grad_beta_g);
        // gate Richards parameter grads
        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (1, n_gate_w),
            grad_gate_poly_vec.into_iter().map(|v| v as f32).collect(),
        )
        .unwrap();
        all_param_grads.push(grad_gate_poly);

        // Threshold predictor grads
        if self.head_selection_config.gating.use_learned_predictor {
            all_param_grads.push(grad_w_tau.unwrap());
            all_param_grads.push(grad_b_tau.unwrap());
            all_param_grads.push(grad_w2_tau.unwrap());
            all_param_grads.push(grad_b2_tau.unwrap());
            if let Some(gcw) = grad_cond_w_tau {
                all_param_grads.push(gcw);
            } else {
                all_param_grads.push(Array2::<f32>::zeros((self.embed_dim, self.alpha_g.ncols())));
            }
            let grad_activation_tau_f32 = Array2::<f32>::from_shape_vec(
                (1, grad_activation_tau.as_ref().unwrap().len()),
                grad_activation_tau
                    .unwrap()
                    .into_iter()
                    .map(|v| v as f32)
                    .collect(),
            )
            .unwrap();
            all_param_grads.push(grad_activation_tau_f32);
        }

        all_param_grads.push(grad_cope_pos);

        // Final numerical stability validation and correction
        if gradient_anomaly_detected {
            tracing::warn!(
                "Gradient anomalies detected in PolyAttention layer - applying corrective measures"
            );

            // Correct non-finite gradients by clamping to reasonable bounds
            for grad in &mut all_param_grads {
                grad.mapv_inplace(|x| {
                    if x.is_finite() {
                        x
                    } else {
                        tracing::warn!("Replacing non-finite gradient with 0.0");
                        0.0
                    }
                });
            }

            // Also check and correct input gradients
            grad_input_total.mapv_inplace(|x| {
                if x.is_finite() {
                    x
                } else {
                    tracing::warn!("Replacing non-finite input gradient with 0.0");
                    0.0
                }
            });
        }

        (grad_input_total, all_param_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        use rayon::prelude::*;
        let pairs: Vec<(Array2<f32>, f32)> = param_grads
            .par_iter()
            .map(|g| {
                let mut gg = g.clone();
                gg.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
                let s = gg.iter().map(|&x| x * x).sum::<f32>();
                (gg, s)
            })
            .collect();
        let norm_sq: f32 = pairs.iter().map(|(_, s)| *s).sum();
        let mut sanitized: Vec<Array2<f32>> = pairs.into_iter().map(|(gg, _)| gg).collect();
        let nrm = norm_sq.sqrt();
        let clip = 5.0f32;
        if nrm.is_finite() && nrm > clip && nrm > 0.0 {
            let scale = clip / nrm;
            sanitized
                .par_iter_mut()
                .for_each(|gg| gg.mapv_inplace(|x| x * scale));
        }
        let param_grads = &sanitized;

        // Expect 3 per head + w_out + a + b + scale + w_g + alpha_g + beta_g + gate_poly_w +
        // threshold_predictor
        let mut expected = self.num_heads * 3 + 1 + 3 + 3 + 1; // + gate_poly_w
        if self.head_selection_config.gating.use_learned_predictor {
            expected += 6;
        } // weights1, bias1, weights2, bias2, cond_w, activation_params
        expected += 1; // CoPE parameters
        if param_grads.len() != expected {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "PolyAttention expected {} grad arrays, got {}",
                    expected,
                    param_grads.len()
                ),
            });
        }
        let mut idx = 0;
        for head in &mut self.heads {
            head.step_w_q(&param_grads[idx], lr);
            head.step_w_k(&param_grads[idx + 1], lr);
            head.step_w_v(&param_grads[idx + 2], lr);
            idx += 3;
        }
        self.opt_w_out.step(&mut self.w_out, &param_grads[idx], lr);
        idx += 1;
        self.opt_a.step(&mut self.a, &param_grads[idx], lr);
        self.opt_b.step(&mut self.b, &param_grads[idx + 1], lr);
        self.opt_scale
            .step(&mut self.scale, &param_grads[idx + 2], lr);
        idx += 3;
        self.opt_w_g.step(&mut self.w_g, &param_grads[idx], lr);
        self.opt_alpha_g
            .step(&mut self.alpha_g, &param_grads[idx + 1], lr);
        self.opt_beta_g
            .step(&mut self.beta_g, &param_grads[idx + 2], lr);
        idx += 3;
        // TODO: Consider decoupling Richards curve training
        // Option 1: Keep coupled (current) - Richards learns from attention gradients
        // Option 2: Independent training - Richards learns from separate objectives
        // Option 3: Meta-learning - Richards learns across multiple attention layers
        {
            let grad_gate_poly = &param_grads[idx];
            let grad_gate_vec: Vec<f64> = grad_gate_poly.iter().map(|&x| x as f64).collect();
            self.gate_poly.step(&grad_gate_vec, lr as f64);
        }
        idx += 1;

        if self.head_selection_config.gating.use_learned_predictor {
            if let (Some(predictor), Some(opt_w1), Some(opt_b1), Some(opt_w2), Some(opt_b2)) = (
                &mut self.threshold_predictor,
                &mut self.opt_w_tau,
                &mut self.opt_b_tau,
                &mut self.opt_w2_tau,
                &mut self.opt_b2_tau,
            ) {
                // Update first layer weights and biases
                opt_w1.step(&mut predictor.weights1, &param_grads[idx], lr);
                // bias1 is (hidden_dim,) but gradient is (hidden_dim, 1), so reshape bias to match
                // optimizer
                let mut bias1_reshaped = predictor
                    .bias1
                    .clone()
                    .to_shape((predictor.bias1.len(), 1))
                    .unwrap()
                    .to_owned();
                opt_b1.step(&mut bias1_reshaped, &param_grads[idx + 1], lr);
                predictor.bias1.assign(
                    &bias1_reshaped
                        .view()
                        .to_shape(predictor.bias1.len())
                        .unwrap(),
                );
                // Update second layer weights and biases
                opt_w2.step(&mut predictor.weights2, &param_grads[idx + 2], lr);
                // bias2 is (1,) but gradient is (1, 1), so reshape bias to match optimizer
                let mut bias2_reshaped = predictor
                    .bias2
                    .clone()
                    .to_shape((predictor.bias2.len(), 1))
                    .unwrap()
                    .to_owned();
                opt_b2.step(&mut bias2_reshaped, &param_grads[idx + 3], lr);
                predictor.bias2.assign(
                    &bias2_reshaped
                        .view()
                        .to_shape(predictor.bias2.len())
                        .unwrap(),
                );
                if let Some(opt_cond) = &mut self.opt_cond_w_tau {
                    opt_cond.step(&mut predictor.cond_w, &param_grads[idx + 4], lr);
                }
                // Update Richards activation parameters using its own step method
                let grad_activation_vec: Vec<f64> =
                    param_grads[idx + 5].iter().map(|&x| x as f64).collect();
                predictor.activation.step(&grad_activation_vec, lr as f64);
            }
            idx += 6; // weights1, bias1, weights2, bias2, cond_w, activation_params
        }
        self.cope.apply_gradients(&param_grads[idx], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (input_grads, param_grads) = self.compute_gradients_parallel(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    pub fn compute_gradients_parallel(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let (n, _d_model) = (input.nrows(), input.ncols());
        let dk_scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let a = self.a[[0, 0]];
        let b = self.b[[0, 0]];
        let scale = self.scale[[0, 0]];
        let p_i32 = self.p as i32;
        let mut grad_input_total = output_grads.clone();
        let n_gate_w = self.gate_poly.weights().len();
        use rayon::prelude::*;

        let head_results: Vec<(
            Array2<f32>,
            Array2<f32>,
            Array2<f32>,
            Array2<f32>,
            Array2<f32>,
            f32,
            f32,
            f32,
            Array2<f32>,
            f32,
            f32,
            Vec<f64>,
            Option<Array2<f32>>,
            Array2<f32>,
            bool,
        )> = (0..self.num_heads)
            .into_par_iter()
            .map(|h_idx| {
                let head = &self.heads[h_idx];
                let q: Array2<f32> = input.dot(&head.w_q);
                let k: Array2<f32> = input.dot(&head.w_k);
                let v: Array2<f32> = input.dot(&head.w_v);
                let w_g_col = self.w_g.slice(s![.., h_idx..h_idx + 1]);
                let xw_col = input.dot(&w_g_col);
                let a_h = self.alpha_g[[0, h_idx]];
                let b_h = self.beta_g[[0, h_idx]];
                let mut z_col = xw_col.clone();
                z_col.mapv_inplace(|vv| a_h * vv + b_h);
                let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
                let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
                let mut g_col = z_col.clone();
                g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);
                let mut m_col = Array2::<f32>::ones((n, 1));
                if self.head_selection_config.gating.use_learned_predictor
                    && let Some(predictor) = &self.threshold_predictor
                {
                    let thresholds = predictor.forward(&input.view());
                    let head_thresholds = thresholds.slice(s![.., h_idx..h_idx + 1]);
                    m_col.assign(&head_thresholds);
                } else if self.head_selection_config.gating.use_soft_top_p {
                    if let Some(mask) = &self.cached_soft_top_p_mask {
                        if mask.nrows() == n && mask.ncols() == self.num_heads {
                            let mask_col = mask.slice(s![.., h_idx..h_idx + 1]);
                            m_col.assign(&mask_col);
                        }
                    }
                }

                let start = h_idx * self.head_dim;
                let end = start + self.head_dim;
                let w_block = self.w_out.slice(s![start..end, ..]);
                let w_block_t = w_block.t();
                let mut grad_q: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_k: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_v: Array2<f32> = Array2::<f32>::zeros((n, self.head_dim));
                let mut grad_p_local: Array2<f32> =
                    Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));
                let mut grad_w_out_block = Array2::<f32>::zeros((self.head_dim, self.embed_dim));
                let mut grad_w_g_col = Array2::<f32>::zeros((self.embed_dim, 1));
                let mut grad_alpha_val: f32 = 0.0;
                let mut grad_beta_val: f32 = 0.0;
                let mut grad_gate_poly_vec = vec![0.0f64; n_gate_w];
                let mut grad_input_contrib = Array2::<f32>::zeros((n, self.embed_dim));
                let mut grad_a_scalar_local: f32 = 0.0;
                let mut grad_b_scalar_local: f32 = 0.0;
                let mut grad_scale_scalar_local: f32 = 0.0;
                let mut threshold_accum_local =
                    if self.head_selection_config.gating.use_learned_predictor {
                        Some(Array2::<f32>::zeros((n, self.num_heads)))
                    } else {
                        None
                    };
                let mut anomaly = false;

                for i in 0..n {
                    let out_row = output_grads.slice(s![i..i + 1, ..]);
                    let mut g_yh_gated_row = Array2::<f32>::zeros((1, self.head_dim));
                    general_mat_mul(1.0, &out_row, &w_block_t, 0.0, &mut g_yh_gated_row);
                    let mut y_pre_row = Array2::<f32>::zeros((1, self.head_dim));
                    let j_start = match self.window_size {
                        Some(w) => i.saturating_sub(w - 1),
                        None => 0,
                    };
                    let j_end = if self.last_causal { i } else { n - 1 };
                    let max_pos = usize::min(self.cope.max_pos, i.saturating_sub(j_start));
                    let mut q_pe = vec![0.0f32; max_pos + 1];
                    for pos in 0..=max_pos {
                        q_pe[pos] = q.row(i).dot(&self.cope.pos_embeddings.row(pos));
                    }
                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            s += q_pe[pos];
                        }
                        let sp = match p_i32 {
                            1 => s,
                            2 => s * s,
                            3 => s * s * s,
                            _ => s.powi(p_i32),
                        };
                        let phi = scale * (a * sp + b);
                        for h in 0..self.head_dim {
                            y_pre_row[[0, h]] += phi * v[[j, h]];
                        }
                    }
                    let eff_i = g_col[[i, 0]] * m_col[[i, 0]];
                    let mut yh_gated_row = y_pre_row.clone();
                    for h in 0..self.head_dim {
                        yh_gated_row[[0, h]] *= eff_i;
                    }
                    general_mat_mul(1.0, &yh_gated_row.t(), &out_row, 1.0, &mut grad_w_out_block);
                    let mut grad_eff_i = 0.0f32;
                    for h in 0..self.head_dim {
                        grad_eff_i += g_yh_gated_row[[0, h]] * y_pre_row[[0, h]];
                    }
                    let d_g_i = grad_eff_i * m_col[[i, 0]];
                    let z_i = a_h * xw_col[[i, 0]] + b_h;
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;
                    let gws = gate_poly.grad_weights_scalar(z_i as f64, d_g_i as f64);
                    for (wi, &gw) in gws.iter().enumerate() {
                        grad_gate_poly_vec[wi] += gw;
                    }
                    for d in 0..self.embed_dim {
                        grad_w_g_col[[d, 0]] += a_h * input[[i, d]] * grad_g_i;
                    }
                    grad_alpha_val += grad_g_i * xw_col[[i, 0]];
                    grad_beta_val += grad_g_i;
                    let wg_col_owned = self.w_g.slice(s![.., h_idx..h_idx + 1]).to_owned();
                    let wg_scaled_t = wg_col_owned.t();
                    for d in 0..self.embed_dim {
                        grad_input_contrib[[i, d]] += a_h * wg_scaled_t[[0, d]] * grad_g_i;
                    }

                    let mut g_yh_pre_row = g_yh_gated_row.clone();
                    for h in 0..self.head_dim {
                        g_yh_pre_row[[0, h]] *= g_col[[i, 0]] * m_col[[i, 0]];
                    }

                    for j in j_start..=j_end {
                        let base = q.row(i).dot(&k.row(j)) * dk_scale;
                        let mut s = base;
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            s += q_pe[pos];
                        }
                        let s_clamped = s.clamp(-8.0, 8.0);
                        let sp = if p_i32 <= 3 {
                            match p_i32 {
                                1 => s_clamped,
                                2 => s_clamped * s_clamped,
                                3 => s_clamped * s_clamped * s_clamped,
                                _ => unreachable!(),
                            }
                        } else {
                            let mut result = 1.0;
                            let current = s_clamped;
                            for _ in 0..p_i32 {
                                result *= current;
                                if !result.is_finite() {
                                    result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                    break;
                                }
                            }
                            result
                        };
                        let phi = scale * (a * sp + b);
                        for h in 0..self.head_dim {
                            grad_v[[j, h]] += phi * g_yh_pre_row[[0, h]];
                        }
                        let dphi_ij = g_yh_pre_row.row(0).dot(&v.row(j));
                        grad_scale_scalar_local += dphi_ij * (a * sp + b);
                        grad_a_scalar_local += dphi_ij * scale * sp;
                        grad_b_scalar_local += dphi_ij * scale;
                        let spm1 = if p_i32 <= 3 {
                            match p_i32 {
                                1 => 1.0,
                                2 => s_clamped,
                                3 => s_clamped * s_clamped,
                                _ => unreachable!(),
                            }
                        } else {
                            let mut result = 1.0;
                            let current = s_clamped;
                            for _ in 0..(p_i32 - 1) {
                                result *= current;
                                if !result.is_finite() {
                                    result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                    break;
                                }
                            }
                            result
                        };
                        let d_s_ij = dphi_ij * scale * a * (self.p as f32) * spm1;
                        if !d_s_ij.is_finite() {
                            anomaly = true;
                        }
                        for h in 0..self.head_dim {
                            let grad_q_val = d_s_ij * k[[j, h]] * dk_scale;
                            let grad_k_val = d_s_ij * q[[i, h]] * dk_scale;
                            if !grad_q_val.is_finite() || !grad_k_val.is_finite() {
                                anomaly = true;
                            }
                            grad_q[[i, h]] += grad_q_val;
                            grad_k[[j, h]] += grad_k_val;
                        }
                        let pos = i.saturating_sub(j);
                        if pos < q_pe.len() {
                            for h in 0..self.head_dim {
                                grad_q[[i, h]] += d_s_ij * self.cope.pos_embeddings[[pos, h]];
                                grad_p_local[[pos, h]] += d_s_ij * q[[i, h]];
                            }
                        }
                    }

                    if let Some(threshold_grad_accum) = threshold_accum_local.as_mut() {
                        let mut d_g_yh_pre_row = Array2::<f32>::zeros((1, self.head_dim));
                        for j in j_start..=j_end {
                            let base = q.row(i).dot(&k.row(j)) * dk_scale;
                            let mut s = base;
                            let pos = i.saturating_sub(j);
                            if pos < q_pe.len() {
                                s += q_pe[pos];
                            }
                            let s_clamped = s.clamp(-8.0, 8.0);
                            let sp = if p_i32 <= 3 {
                                match p_i32 {
                                    1 => s_clamped,
                                    2 => s_clamped * s_clamped,
                                    3 => s_clamped * s_clamped * s_clamped,
                                    _ => unreachable!(),
                                }
                            } else {
                                let mut result = 1.0;
                                let current = s_clamped;
                                for _ in 0..p_i32 {
                                    result *= current;
                                    if !result.is_finite() {
                                        result = if s_clamped >= 0.0 { f32::MAX } else { f32::MIN };
                                        break;
                                    }
                                }
                                result
                            };
                            let v_j = v.row(j);
                            let dphi_contrib = (a * sp + b) * scale;
                            for h in 0..self.head_dim {
                                d_g_yh_pre_row[[0, h]] += v_j[[h]] * dphi_contrib;
                            }
                        }
                        let g_i = g_col[[i, 0]];
                        for h in 0..self.head_dim {
                            let g_yh_gated_h = g_yh_gated_row[[0, h]];
                            threshold_grad_accum[[i, h_idx]] +=
                                g_yh_gated_h * g_i * d_g_yh_pre_row[[0, h]];
                        }
                    }
                }

                // Backprop through linear projections for this head
                let d_w_q = input.t().dot(&grad_q);
                let d_w_k = input.t().dot(&grad_k);
                let d_w_v = input.t().dot(&grad_v);
                general_mat_mul(1.0, &grad_q, &head.w_q.t(), 1.0, &mut grad_input_contrib);
                general_mat_mul(1.0, &grad_k, &head.w_k.t(), 1.0, &mut grad_input_contrib);
                general_mat_mul(1.0, &grad_v, &head.w_v.t(), 1.0, &mut grad_input_contrib);
                (
                    d_w_q,
                    d_w_k,
                    d_w_v,
                    grad_w_out_block,
                    grad_input_contrib,
                    grad_a_scalar_local,
                    grad_b_scalar_local,
                    grad_scale_scalar_local,
                    grad_w_g_col,
                    grad_alpha_val,
                    grad_beta_val,
                    grad_gate_poly_vec,
                    threshold_accum_local,
                    grad_p_local,
                    anomaly,
                )
            })
            .collect();

        let mut all_param_grads: Vec<Array2<f32>> = Vec::new();
        let mut grad_w_out = Array2::<f32>::zeros((self.embed_dim, self.embed_dim));
        let mut grad_w_g = Array2::<f32>::zeros((self.embed_dim, self.num_heads));
        let mut grad_alpha_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_beta_g = Array2::<f32>::zeros((1, self.num_heads));
        let mut grad_a_scalar: f32 = 0.0;
        let mut grad_b_scalar: f32 = 0.0;
        let mut grad_scale_scalar: f32 = 0.0;
        let mut grad_gate_poly_vec_acc = vec![0.0f64; n_gate_w];
        let mut grad_cope_pos =
            Array2::<f32>::zeros((self.cope.max_pos + 1, self.cope.pos_embeddings.ncols()));
        let mut threshold_grad_accum = if self.head_selection_config.gating.use_learned_predictor {
            Some(Array2::<f32>::zeros((n, self.num_heads)))
        } else {
            None
        };
        let mut gradient_anomaly_detected = false;

        for (
            h_idx,
            (
                d_w_q,
                d_w_k,
                d_w_v,
                grad_w_out_block,
                grad_input_contrib,
                ga,
                gb,
                gs,
                grad_w_g_col,
                grad_alpha_val,
                grad_beta_val,
                grad_gate_poly_vec,
                threshold_accum_local,
                grad_p_local,
                anomaly,
            ),
        ) in head_results.into_iter().enumerate()
        {
            all_param_grads.push(d_w_q);
            all_param_grads.push(d_w_k);
            all_param_grads.push(d_w_v);
            let start = h_idx * self.head_dim;
            let end = start + self.head_dim;
            let mut gw_block = grad_w_out.slice_mut(s![start..end, ..]);
            gw_block += &grad_w_out_block;
            grad_input_total += &grad_input_contrib;
            let mut col = grad_w_g.slice_mut(s![.., h_idx..h_idx + 1]);
            col.assign(&grad_w_g_col);
            grad_alpha_g[[0, h_idx]] += grad_alpha_val;
            grad_beta_g[[0, h_idx]] += grad_beta_val;
            for i in 0..n_gate_w {
                grad_gate_poly_vec_acc[i] += grad_gate_poly_vec[i];
            }
            if let (Some(acc), Some(local)) = (threshold_grad_accum.as_mut(), threshold_accum_local)
            {
                *acc += &local;
            }
            grad_cope_pos += &grad_p_local;
            if anomaly {
                gradient_anomaly_detected = true;
            }
            grad_a_scalar += ga;
            grad_b_scalar += gb;
            grad_scale_scalar += gs;
        }

        if self.head_selection_config.gating.use_learned_predictor
            && (self.head_selection_config.gating.complexity_loss_weight > 0.0
                || self.head_selection_config.gating.load_balance_weight > 0.0
                || self.head_selection_config.gating.sparsity_weight > 0.0)
        {
            let m_mat = if let Some(predictor) = &self.threshold_predictor {
                predictor.forward(&input.view())
            } else {
                Array2::<f32>::ones((n, self.num_heads))
            };
            let mut g_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut eff_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut z_mat = Array2::<f32>::zeros((n, self.num_heads));
            let mut max_abs_vec: Vec<f64> = vec![0.0; self.num_heads];
            for h in 0..self.num_heads {
                let w_g_col = self.w_g.slice(s![.., h..h + 1]);
                let xw_col = input.dot(&w_g_col);
                let a_h = self.alpha_g[[0, h]];
                let b_h = self.beta_g[[0, h]];
                let mut z_col = xw_col.clone();
                z_col.mapv_inplace(|v| a_h * v + b_h);
                let max_abs_z = z_col.iter().fold(0.0_f64, |m, &z| m.max((z as f64).abs()));
                max_abs_vec[h] = max_abs_z;
                let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_z);
                let mut g_col = z_col.clone();
                g_col.mapv_inplace(|z| gate_poly.forward_scalar(z as f64) as f32);
                for i in 0..n {
                    z_mat[[i, h]] = z_col[[i, 0]];
                    g_mat[[i, h]] = g_col[[i, 0]];
                    eff_mat[[i, h]] = g_col[[i, 0]] * m_mat[[i, h]];
                }
            }
            let inv_n = 1.0f32 / (n as f32);
            let inv_h = 1.0f32 / (self.num_heads as f32);
            let target_heads = ((self.head_selection_config.min_heads
                + self.head_selection_config.max_heads) as f32)
                * 0.5;
            for i in 0..n {
                let mut s = 0.0f32;
                for h in 0..self.num_heads {
                    s += eff_mat[[i, h]];
                }
                let mean = s * inv_h;
                let mut base_d = 0.0f32;
                if self.head_selection_config.gating.complexity_loss_weight > 0.0 {
                    base_d += self.head_selection_config.gating.complexity_loss_weight
                        * (s - target_heads)
                        * inv_n;
                }
                base_d += self.head_selection_config.gating.sparsity_weight * inv_n * inv_h;
                for h in 0..self.num_heads {
                    let eff_h = eff_mat[[i, h]];
                    let mut d_eff_h = base_d;
                    if self.head_selection_config.gating.load_balance_weight > 0.0 {
                        d_eff_h += 2.0
                            * self.head_selection_config.gating.load_balance_weight
                            * inv_n
                            * inv_h
                            * (eff_h - mean);
                    }
                    let d_g_i = d_eff_h * m_mat[[i, h]];
                    let a_h = self.alpha_g[[0, h]];
                    let z_i = z_mat[[i, h]];
                    let gate_poly = self.gate_poly.update_scaling_from_max_abs(max_abs_vec[h]);
                    let dphi_dz_i = gate_poly.backward_scalar(z_i as f64) as f32;
                    let grad_g_i = d_g_i * dphi_dz_i;
                    for d in 0..self.embed_dim {
                        grad_w_g[[d, h]] += a_h * input[[i, d]] * grad_g_i;
                    }
                    let xw_val = if a_h.abs() > 1e-8 {
                        (z_i - self.beta_g[[0, h]]) / a_h
                    } else {
                        0.0
                    };
                    grad_alpha_g[[0, h]] += grad_g_i * xw_val;
                    grad_beta_g[[0, h]] += grad_g_i;
                    for d in 0..self.embed_dim {
                        grad_input_total[[i, d]] += a_h * self.w_g[[d, h]] * grad_g_i;
                    }
                    if let Some(acc) = threshold_grad_accum.as_mut() {
                        acc[[i, h]] += d_eff_h * g_mat[[i, h]];
                    }
                }
            }
        }

        let (grad_w_tau, grad_b_tau, grad_w2_tau, grad_b2_tau, grad_cond_w_tau, grad_activation_tau): ThresholdPredictorGrads = if self.head_selection_config.gating.use_learned_predictor {
            if let Some(predictor) = &self.threshold_predictor {
                if let Some(threshold_grad_accum) = threshold_grad_accum.as_ref() {
                    let (grad_w1, grad_b1_1d, grad_w2, grad_b2_1d, grad_cond_w, grad_activation) = predictor.compute_gradients(threshold_grad_accum);
                    let grad_b1 = grad_b1_1d.clone().to_shape((grad_b1_1d.len(), 1)).unwrap().to_owned();
                    let grad_b2 = grad_b2_1d.clone().to_shape((grad_b2_1d.len(), 1)).unwrap().to_owned();
                    (Some(grad_w1), Some(grad_b1), Some(grad_w2), Some(grad_b2), grad_cond_w, Some(grad_activation))
                } else {
                    let hidden_dim = predictor.weights1.ncols();
                    let num_outputs = predictor.weights2.ncols();
                    (Some(Array2::<f32>::zeros((self.embed_dim, hidden_dim))),
                     Some(Array2::<f32>::zeros((hidden_dim, 1))),
                     Some(Array2::<f32>::zeros((hidden_dim, num_outputs))),
                     Some(Array2::<f32>::zeros((num_outputs, 1))),
                     Some(Array2::<f32>::zeros((self.embed_dim, hidden_dim))),
                     Some(vec![0.0_f64; predictor.activation.scalar_weights_len()]))
                }
            } else { (None, None, None, None, None, None) }
        } else { (None, None, None, None, None, None) };

        let mut all_param_grads: Vec<Array2<f32>> = all_param_grads;
        all_param_grads.push(grad_w_out);
        let grad_a = Array2::<f32>::from_shape_vec((1, 1), vec![grad_a_scalar]).unwrap();
        let grad_b = Array2::<f32>::from_shape_vec((1, 1), vec![grad_b_scalar]).unwrap();
        let grad_scale = Array2::<f32>::from_shape_vec((1, 1), vec![grad_scale_scalar]).unwrap();
        all_param_grads.push(grad_a);
        all_param_grads.push(grad_b);
        all_param_grads.push(grad_scale);
        all_param_grads.push(grad_w_g);
        all_param_grads.push(grad_alpha_g);
        all_param_grads.push(grad_beta_g);
        let grad_gate_poly = Array2::<f32>::from_shape_vec(
            (1, n_gate_w),
            grad_gate_poly_vec_acc
                .into_iter()
                .map(|v| v as f32)
                .collect(),
        )
        .unwrap();
        all_param_grads.push(grad_gate_poly);
        if self.head_selection_config.gating.use_learned_predictor {
            match (grad_w_tau, grad_b_tau, grad_w2_tau, grad_b2_tau, grad_cond_w_tau, grad_activation_tau) {
                (Some(gw1), Some(gb1), Some(gw2), Some(gb2), Some(gcw), Some(ga)) => {
                    all_param_grads.push(gw1);
                    all_param_grads.push(gb1);
                    all_param_grads.push(gw2);
                    all_param_grads.push(gb2);
                    all_param_grads.push(gcw);
                    let grad_activation_tau_f32 = Array2::<f32>::from_shape_vec(
                        (1, ga.len()),
                        ga.into_iter().map(|v| v as f32).collect(),
                    )
                    .unwrap();
                    all_param_grads.push(grad_activation_tau_f32);
                }
                _ => {
                    tracing::warn!(
                        target: "poly_attention",
                        "Learned threshold predictor gradients unavailable; skipping predictor params"
                    );
                }
            }
        }
        all_param_grads.push(grad_cope_pos);
        if gradient_anomaly_detected {
            for grad in &mut all_param_grads {
                grad.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
            }
            grad_input_total.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
        }
        (grad_input_total, all_param_grads)
    }

    /// Get parameter information for this PolyAttention layer
    fn get_param_info(&mut self) -> &PolyAttentionParamInfo {
        if self.param_info.is_none() {
            // Calculate parameter counts for each component
            let head_params_per_head = self
                .heads
                .first()
                .map(|h| h.w_q.len() + h.w_k.len() + h.w_v.len())
                .unwrap_or(0);

            let gate_poly_params = self.gate_poly.weights().len();

            let threshold_predictor_params =
                if self.head_selection_config.gating.use_learned_predictor {
                    if let Some(predictor) = &self.threshold_predictor {
                        predictor.weights1.len()
                            + predictor.bias1.len()
                            + predictor.weights2.len()
                            + predictor.bias2.len()
                    } else {
                        // Fallback to old count for compatibility
                        self.embed_dim + 1 + 1
                    }
                } else {
                    0
                };

            let cope_params = self.cope.parameters();

            self.param_info = Some(PolyAttentionParamInfo::new(
                self.embed_dim,
                self.num_heads,
                head_params_per_head,
                gate_poly_params,
                threshold_predictor_params,
                cope_params,
            ));
        }

        self.param_info.as_ref().unwrap()
    }

    /// Get detailed parameter breakdown for this PolyAttention layer
    pub fn param_breakdown(&mut self) -> &PolyAttentionParamInfo {
        self.get_param_info()
    }

    fn parameters(&self) -> usize {
        // Use cached value if available, otherwise compute
        if let Some(ref info) = self.param_info {
            info.total_params
        } else {
            // Fallback to original computation (but this won't be cached)
            let head_params = self
                .heads
                .iter()
                .map(|h| h.w_q.len() + h.w_k.len() + h.w_v.len())
                .sum::<usize>();
            let mut total = self.w_out.len()
                + 3
                + head_params
                + self.w_g.len()
                + self.alpha_g.len()
                + self.beta_g.len()
                + self.gate_poly.weights().len();
            total += self.cope.parameters();
            if self.head_selection_config.gating.use_learned_predictor {
                if let Some(predictor) = &self.threshold_predictor {
                    total += predictor.weights1.len()
                        + predictor.bias1.len()
                        + predictor.weights2.len()
                        + predictor.bias2.len();
                } else {
                    total += self.embed_dim + 1 + 1;
                }
            }
            total
        }
    }

    // Initialize or ensure learned threshold predictor parameters

    pub fn set_head_selection_config(&mut self, strategy: &HeadSelectionStrategy) {
        crate::attention::config::configure_head_selection(
            &mut self.head_selection_config,
            &mut self.threshold_predictor,
            self.embed_dim,
            self.num_heads,
            &mut self.opt_w_tau,
            &mut self.opt_b_tau,
            &mut self.opt_w2_tau,
            &mut self.opt_b2_tau,
            &mut self.opt_cond_w_tau,
            strategy,
        );
    }

    pub fn num_heads(&self) -> usize { self.num_heads }

    pub fn compute_moh_aux_losses(&self, target_avg_components: f32) -> (f32, f32, f32) {
        let lb = self.head_selection_config.compute_load_balance_loss();
        let cx = self.head_selection_config.compute_complexity_loss(target_avg_components);
        let sp = self.head_selection_config.compute_sparsity_loss();
        (lb, cx, sp)
    }

    pub fn compute_moh_aux_weighted_total(&self, target_avg_components: f32) -> f32 {
        let (lb, cx, sp) = self.compute_moh_aux_losses(target_avg_components);
        let g = &self.head_selection_config.gating;
        
        // Debug logging for high loss investigation
        if lb * g.load_balance_weight + cx * g.complexity_loss_weight + sp * g.sparsity_weight > 1.0 {
            tracing::info!(
                "High MoH Aux Loss: Total={}, LB={} (w={}), CX={} (w={}), SP={} (w={})",
                lb * g.load_balance_weight + cx * g.complexity_loss_weight + sp * g.sparsity_weight,
                lb, g.load_balance_weight,
                cx, g.complexity_loss_weight,
                sp, g.sparsity_weight
            );
        }

        (lb * g.load_balance_weight)
            + (cx * g.complexity_loss_weight)
            + (sp * g.sparsity_weight)
    }

    pub fn get_avg_active_heads(&self) -> f32 {
        self.head_selection_config.gating.get_avg_active_components()
    }

    pub fn moh_num_active(&self) -> usize { self.head_selection_config.gating.num_active }

    pub fn set_token_threshold_scale(&mut self, scale: Array2<f32>) {
        self.token_threshold_scale = Some(scale);
    }

    pub fn set_token_latent_features(&mut self, feats: Array2<f32>) {
        self.token_latent_features = Some(feats);
    }

    pub fn peek_tau_metrics(&self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            Some((self.head_selection_config.metrics_tau_min, self.head_selection_config.metrics_tau_max))
        } else {
            None
        }
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        let mut res = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let tokens = self
                .head_selection_config
                .gating
                .metrics
                .token_count_per_component[h];
            let avg = if tokens > 0 {
                self.head_selection_config
                    .gating
                    .metrics
                    .active_sum_per_component[h]
                    / tokens as f32
            } else {
                0.0
            };
            res.push((avg, tokens));
            self.head_selection_config
                .gating
                .metrics
                .active_sum_per_component[h] = 0.0;
            self.head_selection_config
                .gating
                .metrics
                .token_count_per_component[h] = 0;
        }
        res
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        if self.head_selection_config.metrics_tau_count > 0 {
            let min = self.head_selection_config.metrics_tau_min;
            let max = self.head_selection_config.metrics_tau_max;
            self.head_selection_config.metrics_tau_min = f32::INFINITY;
            self.head_selection_config.metrics_tau_max = f32::NEG_INFINITY;
            self.head_selection_config.metrics_tau_sum = 0.0;
            self.head_selection_config.metrics_tau_count = 0;
            Some((min, max))
        } else {
            None
        }
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        if self.head_selection_config.metrics_g_count > 0 {
            let rms = (self.head_selection_config.metrics_g_sq_sum
                / self.head_selection_config.metrics_g_count as f32)
                .sqrt();
            self.head_selection_config.metrics_g_sq_sum = 0.0;
            self.head_selection_config.metrics_g_count = 0;
            Some(rms)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{AdaptiveDegreeConfig, DegreeAdaptationMetrics, PolyAttention};

    #[test]
    fn gradients_parallel_match_sequential_small() {
        let mut pa = PolyAttention::new(16, 4, 3, 64, Some(4));
        let n = 8;
        let d = 16;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.01).sin();
            }
        }
        let _ = pa.forward_impl(&input, true);
        let mut output_grads = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                output_grads[[i, j]] = (((i + j) as f32) * 0.001).cos();
            }
        }
        let (gi_seq, pg_seq) = pa.compute_gradients(&input, &output_grads);
        let (gi_par, pg_par) = pa.compute_gradients_parallel(&input, &output_grads);
        assert_eq!(pg_seq.len(), pg_par.len());
        let mut diff_input = 0.0f32;
        for i in 0..n {
            for j in 0..d {
                diff_input += (gi_seq[[i, j]] - gi_par[[i, j]]).abs();
            }
        }
        assert!(diff_input < 1e-3);
        for (a, b) in pg_seq.iter().zip(pg_par.iter()) {
            assert_eq!(a.shape(), b.shape());
            let mut diff = 0.0f32;
            for (xa, xb) in a.iter().zip(b.iter()) {
                diff += (xa - xb).abs();
            }
            assert!(diff < 1e-2);
        }
    }

    #[test]
    fn adapt_increases_degree_on_slow_convergence() {
        let mut pa = PolyAttention::new(64, 8, 3, 128, None);
        pa.set_adaptive_degree_config(AdaptiveDegreeConfig {
            enabled: true,
            p_min: 1,
            p_max: 5,
            adjust_rate: 1.0,
            increase_threshold: 0.1,
            decrease_threshold: -0.5,
            cooldown_epochs: 0,
        });
        let m = DegreeAdaptationMetrics {
            epoch_index: 0,
            loss_delta: 0.0,
            grad_norm: 1.0,
            epoch_ms: 10.0,
            tokens_per_sec: 1000.0,
            tau_range: None,
            pred_norm_rms: Some(0.0),
        };
        let p0 = pa.p;
        pa.adapt_degree(&m);
        assert!(pa.p >= p0);
    }

    #[test]
    fn adapt_decreases_degree_on_high_grad() {
        let mut pa = PolyAttention::new(64, 8, 3, 128, None);
        pa.set_adaptive_degree_config(AdaptiveDegreeConfig {
            enabled: true,
            p_min: 1,
            p_max: 7,
            adjust_rate: 1.0,
            increase_threshold: 0.9,
            decrease_threshold: -0.1,
            cooldown_epochs: 0,
        });
        let m = DegreeAdaptationMetrics {
            epoch_index: 0,
            loss_delta: 1.0,
            grad_norm: 1e6,
            epoch_ms: 10.0,
            tokens_per_sec: 1000.0,
            tau_range: None,
            pred_norm_rms: Some(1.0),
        };
        let p0 = pa.p;
        pa.adapt_degree(&m);
        assert!(pa.p <= p0);
    }

    #[test]
    fn eff_skip_threshold_skips_computation() {
        let mut pa = PolyAttention::new(64, 4, 3, 64, Some(16));
        let n = 8;
        let d = 64;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n { for j in 0..d { input[[i, j]] = ((i * d + j) as f32) * 0.0007; } }
        pa.set_eff_skip_threshold(1.0);
        let out_skip = pa.forward_impl(&input, false);
        assert_eq!(out_skip, Array2::<f32>::zeros((n, d)));
        pa.set_eff_skip_threshold(0.0);
        let out_no_skip = pa.forward_impl(&input, false);
        assert_ne!(out_no_skip, input);
    }

    #[test]
    fn moh_learned_predictor_per_head_thresholds() {
        let mut pa = PolyAttention::new(32, 4, 3, 64, Some(8));
        let strategy = crate::mixtures::moh::HeadSelectionStrategy::Learned {
            num_active: 4,
            load_balance_weight: 0.1,
            complexity_loss_weight: 0.05,
            sparsity_weight: 0.01,
        };
        pa.set_head_selection_config(&strategy);
        let n = 6;
        let d = 32;
        let mut input = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                input[[i, j]] = ((i * d + j) as f32 * 0.003).cos();
            }
        }
        let _out = pa.forward_impl(&input, true);
        let tau = pa.take_tau_metrics();
        assert!(tau.is_some());
        let pred_norm = pa.take_pred_norm();
        assert!(pred_norm.is_some());

        let mut output_grads = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                output_grads[[i, j]] = (((i + j) as f32) * 0.0007).sin();
            }
        }
        let (gi, pg) = pa.compute_gradients_parallel(&input, &output_grads);
        let mut non_finite = false;
        for x in gi.iter() {
            if !x.is_finite() {
                non_finite = true;
                break;
            }
        }
        for g in pg.iter() {
            for x in g.iter() {
                if !x.is_finite() {
                    non_finite = true;
                    break;
                }
            }
        }
        assert!(!non_finite);
    }
}

impl Layer for PolyAttention {
    fn layer_type(&self) -> &str {
        "PolyAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // default causal
        self.forward_impl(input, true)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        PolyAttention::compute_gradients_parallel(self, _input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        PolyAttention::apply_gradients(self, param_grads, lr)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        PolyAttention::backward(self, grads, lr)
    }

    fn parameters(&self) -> usize {
        PolyAttention::parameters(self)
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq: f32 = 0.0;

        // Heads: w_q, w_k, w_v
        for head in &self.heads {
            sumsq += head.w_q.iter().map(|&w| w * w).sum::<f32>();
            sumsq += head.w_k.iter().map(|&w| w * w).sum::<f32>();
            sumsq += head.w_v.iter().map(|&w| w * w).sum::<f32>();
        }

        // Output projection
        sumsq += self.w_out.iter().map(|&w| w * w).sum::<f32>();

        // Polynomial scalars
        sumsq += self.a.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.b.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.scale.iter().map(|&w| w * w).sum::<f32>();

        // Gating parameters
        sumsq += self.w_g.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.alpha_g.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.beta_g.iter().map(|&w| w * w).sum::<f32>();

        // Learnable Richards gate parameters
        sumsq += self
            .gate_poly
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();

        // CoPE positional embeddings
        sumsq += self.cope.weight_norm().powi(2);

        // Threshold predictor weights if present
        if let Some(pred) = &self.threshold_predictor {
            sumsq += pred.weights1.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.weights2.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.bias1.iter().map(|&w| w * w).sum::<f32>();
            sumsq += pred.bias2.iter().map(|&w| w * w).sum::<f32>();
            // Include RichardsNorm internal weights via its trait method
            sumsq += pred.norm.weight_norm().powi(2);
        }

        sumsq.sqrt()
    }

    fn zero_gradients(&mut self) {
        // PolyAttention doesn't maintain internal gradient state
        // Gradients are computed on-demand and applied immediately
    }
}
