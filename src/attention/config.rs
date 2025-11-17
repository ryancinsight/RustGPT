use ndarray::Array2;
use rand_distr::{Distribution, Normal};

use crate::{
    MAX_SEQ_LEN,
    adam::Adam,
    attention::{head::PolyHead, position::cope::CoPE},
    mixtures::{
        moh::{HeadSelectionConfig, HeadSelectionStrategy},
        threshold::ThresholdPredictor,
    },
    richards::{RichardsCurve, Variant},
};

/// Configuration utilities for PolyAttention initialization and setup
/// Provides modular functions for initializing different components of attention layers
/// Initialize polynomial attention parameters (a, b, scale)
pub fn init_polynomial_params() -> (Array2<f32>, Array2<f32>, Array2<f32>, Adam, Adam, Adam) {
    let a = Array2::<f32>::from_shape_vec((1, 1), vec![1.0]).unwrap();
    let b = Array2::<f32>::from_shape_vec((1, 1), vec![0.0]).unwrap();
    let scale =
        Array2::<f32>::from_shape_vec((1, 1), vec![1.0 / (MAX_SEQ_LEN as f32).sqrt()]).unwrap();

    let opt_a = Adam::new((1, 1));
    let opt_b = Adam::new((1, 1));
    let opt_scale = Adam::new((1, 1));

    (a, b, scale, opt_a, opt_b, opt_scale)
}

/// Initialize output projection parameters
pub fn init_output_projection(embed_dim: usize) -> (Array2<f32>, Adam) {
    let mut rng = rand::rng();
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
    let mut rng = rand::rng();
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
pub fn init_cope(max_pos: usize, head_dim: usize) -> CoPE {
    CoPE::new(max_pos, head_dim)
}

/// Initialize head selection configuration with default settings
pub fn init_head_selection_config(num_heads: usize) -> HeadSelectionConfig {
    HeadSelectionConfig {
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
    }
}

/// Initialize Richards curve gating function
pub fn init_gate_polynomial() -> RichardsCurve {
    RichardsCurve::new_learnable(Variant::Sigmoid)
}

/// Ensure threshold predictor is initialized with appropriate configuration
pub fn ensure_threshold_predictor_initialized(
    threshold_predictor: &mut Option<ThresholdPredictor>,
    embed_dim: usize,
    num_heads: usize,
    opt_w_tau: &mut Option<Adam>,
    opt_b_tau: &mut Option<Adam>,
    opt_w2_tau: &mut Option<Adam>,
    opt_b2_tau: &mut Option<Adam>,
    opt_cond_w_tau: &mut Option<Adam>,
) {
    if threshold_predictor.is_none() {
        let predictor_hidden_dim = 128.min(embed_dim / 2).max(32);
        *threshold_predictor = Some(ThresholdPredictor::new_with_cond(
            embed_dim,
            predictor_hidden_dim,
            num_heads,
            embed_dim,
        ));

        *opt_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
        *opt_b_tau = Some(Adam::new((predictor_hidden_dim, 1)));
        *opt_w2_tau = Some(Adam::new((predictor_hidden_dim, num_heads)));
        *opt_b2_tau = Some(Adam::new((num_heads, 1)));
        *opt_cond_w_tau = Some(Adam::new((embed_dim, predictor_hidden_dim)));
    }
}

/// Configure head selection strategy and initialize predictor if needed
pub fn configure_head_selection(
    head_selection_config: &mut HeadSelectionConfig,
    threshold_predictor: &mut Option<ThresholdPredictor>,
    embed_dim: usize,
    num_heads: usize,
    opt_w_tau: &mut Option<Adam>,
    opt_b_tau: &mut Option<Adam>,
    opt_w2_tau: &mut Option<Adam>,
    opt_b2_tau: &mut Option<Adam>,
    opt_cond_w_tau: &mut Option<Adam>,
    strategy: &HeadSelectionStrategy,
) {
    *head_selection_config = HeadSelectionConfig::from_strategy(strategy, num_heads);

    // Initialize threshold predictor if needed (AutoDeco-inspired architecture)
    if head_selection_config.gating.use_learned_predictor && threshold_predictor.is_none() {
        ensure_threshold_predictor_initialized(
            threshold_predictor,
            embed_dim,
            num_heads,
            opt_w_tau,
            opt_b_tau,
            opt_w2_tau,
            opt_b2_tau,
            opt_cond_w_tau,
        );
    }
}

/// Initialize attention heads
pub fn init_attention_heads(embed_dim: usize, num_heads: usize) -> Vec<PolyHead> {
    let head_dim = embed_dim / num_heads;
    (0..num_heads)
        .map(|_| PolyHead::new(embed_dim, head_dim))
        .collect::<Vec<_>>()
}
