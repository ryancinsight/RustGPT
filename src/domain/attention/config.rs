use ndarray::Array2;
use rand_distr::{Distribution, Normal};

use crate::{
    common::rng::get_rng,
    domain::{
        attention::position::{optimized_cope::OptimizedCoPE, unified::UnifiedCoPE},
        mixtures::{
            moh::{HeadSelectionConfig, HeadSelectionStrategy},
            threshold::ThresholdPredictor,
        },
        richards::{RichardsCurve, Variant},
    },
    infrastructure::optimizer::adam::Adam,
};

/// Configuration utilities for PolyAttention initialization and setup
/// Provides modular functions for initializing different components of attention layers
/// Initialize polynomial attention parameters (a, b, scale)
pub fn init_polynomial_params(
    max_seq_len: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Adam, Adam, Adam) {
    let a = Array2::<f32>::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let b = Array2::<f32>::from_shape_vec((1, 1), vec![0.0]).unwrap();
    let denom = max_seq_len.max(1) as f32;
    let scale = Array2::<f32>::from_shape_vec((1, 1), vec![1.0 / denom.sqrt()]).unwrap();

    let opt_a = Adam::new((1, 1));
    let opt_b = Adam::new((1, 1));
    let opt_scale = Adam::new((1, 1));

    (a, b, scale, opt_a, opt_b, opt_scale)
}

/// Initialize output projection parameters
pub fn init_output_projection(embed_dim: usize) -> (Array2<f32>, Adam) {
    let mut rng = get_rng();
    let std_out = (2.0f32 / (embed_dim as f32 + embed_dim as f32)).sqrt();
    let normal_out = Normal::new(0.0, std_out).unwrap();

    let w_out =
        Array2::<f32>::from_shape_fn((embed_dim, embed_dim), |_| normal_out.sample(&mut rng));
    let opt_w_out = Adam::new((embed_dim, embed_dim));

    (w_out, opt_w_out)
}

/// Initialize gating parameters for mixture-of-heads
pub fn init_gating_params(
    embed_dim: usize,
    num_heads: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Adam, Adam, Adam) {
    let mut rng = get_rng();
    let std_g = (2.0f32 / embed_dim as f32).sqrt();
    let normal_g = Normal::new(0.0, std_g).unwrap();

    let w_g = Array2::<f32>::from_shape_fn((embed_dim, num_heads), |_| normal_g.sample(&mut rng));
    let alpha_g = Array2::<f32>::ones((1, num_heads));
    let beta_g = Array2::<f32>::zeros((1, num_heads));

    let opt_w_g = Adam::new((embed_dim, num_heads));
    let opt_alpha_g = Adam::new((1, num_heads));
    let opt_beta_g = Adam::new((1, num_heads));

    (w_g, alpha_g, beta_g, opt_w_g, opt_alpha_g, opt_beta_g)
}

/// Initialize CoPE positional embeddings
pub fn init_cope(max_pos: usize, head_dim: usize) -> UnifiedCoPE {
    UnifiedCoPE::new(max_pos, head_dim)
}

/// Initialize OptimizedCoPE (unified default with gating + factorization + log1p)
pub fn init_optimized_cope(max_pos: usize, head_dim: usize) -> OptimizedCoPE {
    OptimizedCoPE::new(max_pos, head_dim, head_dim / 4) // rank = head_dim / 4
}

/// Initialize head selection configuration with default settings
pub fn init_head_selection_config(num_heads: usize) -> HeadSelectionConfig {
    HeadSelectionConfig {
        gating: crate::domain::mixtures::gating::GatingConfig::default(),
        min_heads: 1,
        max_heads: num_heads,
        always_on_heads: Vec::new(),
        threshold_modulation: crate::domain::richards::AdaptiveScalar::Fixed(1.0),
        metrics_tau_min: f32::INFINITY,
        metrics_tau_max: f32::NEG_INFINITY,
        metrics_tau_sum: 0.0,
        metrics_tau_count: 0,
        metrics_g_sq_sum: 0.0,
        metrics_g_count: 0,
    }
}

/// Initialize Richards curve gating function
pub fn init_gate_polynomial() -> RichardsCurve {
    RichardsCurve::new_learnable(Variant::Sigmoid)
}

pub struct ThresholdPredictorOptimizers<'a> {
    pub opt_w_tau: &'a mut Option<Adam>,
    pub opt_b_tau: &'a mut Option<Adam>,
    pub opt_w2_tau: &'a mut Option<Adam>,
    pub opt_b2_tau: &'a mut Option<Adam>,
    pub opt_cond_w_tau: &'a mut Option<Adam>,
}

/// Ensure threshold predictor is initialized with appropriate configuration
pub fn ensure_threshold_predictor_initialized(
    threshold_predictor: &mut Option<ThresholdPredictor>,
    embed_dim: usize,
    num_heads: usize,
    optimizers: ThresholdPredictorOptimizers<'_>,
) {
    if threshold_predictor.is_none() {
        let predictor_hidden_dim = 128.min(embed_dim / 2).max(32);
        *threshold_predictor = Some(ThresholdPredictor::new_with_cond(
            embed_dim,
            predictor_hidden_dim,
            num_heads,
            embed_dim,
        ));

        *optimizers.opt_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
        *optimizers.opt_b_tau = Some(Adam::new((predictor_hidden_dim, 1)));
        *optimizers.opt_w2_tau = Some(Adam::new((predictor_hidden_dim, num_heads)));
        *optimizers.opt_b2_tau = Some(Adam::new((num_heads, 1)));
        *optimizers.opt_cond_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
    }
}

/// Configure head selection strategy and initialize predictor if needed
pub fn configure_head_selection(
    head_selection_config: &mut HeadSelectionConfig,
    threshold_predictor: &mut Option<ThresholdPredictor>,
    embed_dim: usize,
    num_heads: usize,
    optimizers: ThresholdPredictorOptimizers<'_>,
    strategy: &HeadSelectionStrategy,
) {
    *head_selection_config = HeadSelectionConfig::from_strategy(strategy, num_heads);

    // Initialize threshold predictor if needed (AutoDeco-inspired architecture)
    if head_selection_config.gating.use_learned_predictor && threshold_predictor.is_none() {
        ensure_threshold_predictor_initialized(
            threshold_predictor,
            embed_dim,
            num_heads,
            optimizers,
        );
    }
}

/// Initialize attention weights (monolithic Q, K, V)
pub fn init_attention_weights(
    embed_dim: usize,
    num_heads: usize,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Adam, Adam, Adam) {
    let head_dim = embed_dim / num_heads;
    let total_head_dim = num_heads * head_dim;
    let mut rng = get_rng();

    // Initialize with Xavier/Glorot normal similar to PolyHead
    let std_qk = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();
    let std_v = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();

    let normal_qk = Normal::new(0.0, std_qk).unwrap();
    let normal_v = Normal::new(0.0, std_v).unwrap();

    let w_q =
        Array2::<f32>::from_shape_fn((embed_dim, total_head_dim), |_| normal_qk.sample(&mut rng));
    let w_k =
        Array2::<f32>::from_shape_fn((embed_dim, total_head_dim), |_| normal_qk.sample(&mut rng));
    let w_v =
        Array2::<f32>::from_shape_fn((embed_dim, total_head_dim), |_| normal_v.sample(&mut rng));

    let opt_w_q = Adam::new((embed_dim, total_head_dim));
    let opt_w_k = Adam::new((embed_dim, total_head_dim));
    let opt_w_v = Adam::new((embed_dim, total_head_dim));

    (w_q, w_k, w_v, opt_w_q, opt_w_k, opt_w_v)
}
