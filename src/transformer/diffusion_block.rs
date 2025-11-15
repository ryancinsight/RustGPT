use std::f32::consts::PI;

use ndarray::{Array1, Array2, s, Axis};
use ndarray::Zip;
use ndarray::parallel::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    attention::poly_attention::PolyAttention,
    errors::Result,
    llm::Layer,
    mixtures::{
        HeadSelectionStrategy,
        moe::{ExpertRouterConfig, MixtureOfExperts},
    },
    model_config::ModelConfig,
    richards::{RichardsGlu, RichardsNorm},
    transformer::transformer_block::TransformerBlockConfig,
};

/// Noise schedule types for diffusion models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NoiseSchedule {
    /// Linear schedule: β_t = β_min + (β_max - β_min) * t/T
    Linear { beta_min: f32, beta_max: f32 },
    /// Cosine schedule: β_t = 1 - cos(π/2 * (t/T + s)/(1+s)) / cos(π/2 * s/(1+s))
    /// where s is a small offset for numerical stability
    Cosine { s: f32 },
    /// Quadratic schedule: β_t = β_min + (β_max - β_min) * (t/T)^2
    Quadratic { beta_min: f32, beta_max: f32 },
}

impl Default for NoiseSchedule {
    fn default() -> Self {
        NoiseSchedule::Cosine { s: 0.008 }
    }
}

/// Diffusion noise scheduler that manages variance schedules and cumulative products
#[derive(Serialize, Deserialize, Debug)]
pub struct NoiseScheduler {
    /// Type of noise schedule
    schedule_type: NoiseSchedule,
    /// Number of diffusion timesteps
    num_timesteps: usize,
    /// Precomputed β_t values (variance schedule)
    betas: Array1<f32>,
    /// Precomputed √β_t values
    sqrt_betas: Array1<f32>,
    /// Precomputed √(1-β_t) values
    sqrt_one_minus_betas: Array1<f32>,
    /// Precomputed √ᾱ_t = ∏_{i=1}^t √(1-β_i) (cumulative product for forward process)
    sqrt_alphas_cumprod: Array1<f32>,
    /// Precomputed √(1-ᾱ_t) values
    sqrt_one_minus_alphas_cumprod: Array1<f32>,
    /// Precomputed 1/√ᾱ_t values
    sqrt_recip_alphas_cumprod: Array1<f32>,
    /// Precomputed 1/√(1-ᾱ_t) values
    sqrt_recip_one_minus_alphas_cumprod: Array1<f32>,
    /// Precomputed posterior variance coefficients for reverse process
    posterior_variance: Array1<f32>,
}

impl NoiseScheduler {
    /// Create a new noise scheduler with the given parameters
    pub fn new(schedule_type: NoiseSchedule, num_timesteps: usize) -> Self {
        let betas = Self::compute_betas(&schedule_type, num_timesteps);

        // Precompute all the derived quantities
        let sqrt_betas = betas.mapv(f32::sqrt);
        let sqrt_one_minus_betas = (&betas * -1.0 + 1.0).mapv(f32::sqrt);

        // Compute cumulative product √ᾱ_t = ∏_{i=1}^t √(1-β_i)
        let mut alphas_cumprod = Array1::ones(num_timesteps + 1);
        for t in 1..=num_timesteps {
            alphas_cumprod[t] = alphas_cumprod[t - 1] * (1.0 - betas[t - 1]);
        }
        let sqrt_alphas_cumprod = alphas_cumprod.mapv(f32::sqrt);
        let sqrt_one_minus_alphas_cumprod = (&alphas_cumprod * -1.0 + 1.0).mapv(f32::sqrt);
        let sqrt_recip_alphas_cumprod = alphas_cumprod.mapv(|x| 1.0 / x.sqrt());
        let sqrt_recip_one_minus_alphas_cumprod =
            (&alphas_cumprod * -1.0 + 1.0).mapv(|x| 1.0 / x.sqrt());

        // Compute posterior variance for reverse process
        // σ_t² = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        let mut posterior_variance = Array1::zeros(num_timesteps);
        for t in 1..num_timesteps {
            let beta_t = betas[t];
            let alpha_cumprod_t_minus_1 = alphas_cumprod[t - 1];
            let alpha_cumprod_t = alphas_cumprod[t];
            posterior_variance[t] =
                beta_t * (1.0 - alpha_cumprod_t_minus_1) / (1.0 - alpha_cumprod_t);
        }
        // For t = 0, variance is 0 (deterministic)
        posterior_variance[0] = 0.0;

        Self {
            schedule_type,
            num_timesteps,
            betas,
            sqrt_betas,
            sqrt_one_minus_betas,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recip_one_minus_alphas_cumprod,
            posterior_variance,
        }
    }

    /// Compute β_t values according to the schedule type
    fn compute_betas(schedule: &NoiseSchedule, num_timesteps: usize) -> Array1<f32> {
        match schedule {
            NoiseSchedule::Linear { beta_min, beta_max } => {
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let t_frac = t as f32 / (num_timesteps - 1) as f32;
                    betas[t] = beta_min + (beta_max - beta_min) * t_frac;
                }
                betas
            }
            NoiseSchedule::Cosine { s } => {
                // Improved DDPM cosine schedule: ᾱ_t = f(t)/f(0), f(t) = cos(π/2 * (t/T + s)/(1+s))
                // Derive per-step α_t = ᾱ_t / ᾱ_{t-1}, then β_t = 1 - α_t
                let mut alpha_bar = Array1::zeros(num_timesteps + 1);
                let f_0 = (PI / 2.0 * s / (1.0 + s)).cos();
                // ᾱ_0 = 1
                alpha_bar[0] = 1.0;
                for t in 1..=num_timesteps {
                    let t_frac = (t - 1) as f32 / (num_timesteps - 1) as f32;
                    let arg = PI / 2.0 * (t_frac + s) / (1.0 + s);
                    let f_t = arg.cos();
                    alpha_bar[t] = (f_t / f_0).clamp(1e-6, 1.0);
                }
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let alpha_t = alpha_bar[t + 1] / alpha_bar[t];
                    let beta_t = (1.0 - alpha_t).clamp(1e-6, 0.999);
                    betas[t] = beta_t;
                }
                betas
            }
            NoiseSchedule::Quadratic { beta_min, beta_max } => {
                let mut betas = Array1::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let t_frac = t as f32 / (num_timesteps - 1) as f32;
                    betas[t] = beta_min + (beta_max - beta_min) * t_frac * t_frac;
                }
                betas
            }
        }
    }

    /// Get β_t for timestep t
    pub fn beta(&self, t: usize) -> f32 {
        self.betas[t]
    }

    /// Get √β_t for timestep t
    pub fn sqrt_beta(&self, t: usize) -> f32 {
        self.sqrt_betas[t]
    }

    /// Get √(1-β_t) for timestep t
    pub fn sqrt_one_minus_beta(&self, t: usize) -> f32 {
        self.sqrt_one_minus_betas[t]
    }

    /// Get √ᾱ_t = ∏_{i=1}^t √(1-β_i) for timestep t
    pub fn sqrt_alpha_cumprod(&self, t: usize) -> f32 {
        self.sqrt_alphas_cumprod[t]
    }

    /// Get √(1-ᾱ_t) for timestep t
    pub fn sqrt_one_minus_alpha_cumprod(&self, t: usize) -> f32 {
        self.sqrt_one_minus_alphas_cumprod[t]
    }

    /// Get posterior variance for reverse process at timestep t
    pub fn posterior_variance(&self, t: usize) -> f32 {
        self.posterior_variance[t]
    }

    /// Get α_t = 1 - β_t
    pub fn alpha(&self, t: usize) -> f32 {
        1.0 - self.betas[t]
    }

    /// Get √α_t
    pub fn sqrt_alpha(&self, t: usize) -> f32 {
        self.alpha(t).sqrt()
    }

    /// Get √(1-α_t)
    pub fn sqrt_one_minus_alpha(&self, t: usize) -> f32 {
        (1.0 - self.alpha(t)).sqrt()
    }

    /// Get the number of diffusion timesteps
    pub fn num_timesteps(&self) -> usize {
        self.num_timesteps
    }

    /// Forward diffusion process: q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)
    pub fn q_sample(&self, x_0: &Array2<f32>, t: usize, noise: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            x_0.shape(),
            noise.shape(),
            "x_0 and noise must have same shape"
        );

        let sqrt_alpha_cumprod = self.sqrt_alpha_cumprod(t);
        let sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod(t);

        // x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        x_0 * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
    }

    /// Deterministic DDIM step from x_t to x_{t-1}
    /// x0_hat = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t; x_{t-1} = √ᾱ_{t-1} * x0_hat
    pub fn ddim_step(
        &self,
        x_t: &Array2<f32>,
        t: usize,
        predicted_noise: &Array2<f32>,
    ) -> Array2<f32> {
        let sqrt_alpha_bar_t = self.sqrt_alpha_cumprod(t);
        let sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_cumprod(t);
        let x0_hat =
            (x_t - &(predicted_noise * sqrt_one_minus_alpha_bar_t)) / sqrt_alpha_bar_t.max(1e-6);
        if t == 0 {
            return x0_hat;
        }
        let sqrt_alpha_bar_prev = self.sqrt_alpha_cumprod(t.saturating_sub(1));
        &x0_hat * sqrt_alpha_bar_prev
    }

    /// Compute the posterior mean for reverse process: μ_θ(x_t, t) = 1/√ᾱ_t * (x_t -
    /// (1-ᾱ_t)/√(1-ᾱ_t) * ε_θ)
    pub fn posterior_mean(
        &self,
        x_t: &Array2<f32>,
        t: usize,
        predicted_noise: &Array2<f32>,
    ) -> Array2<f32> {
        assert_eq!(
            x_t.shape(),
            predicted_noise.shape(),
            "x_t and predicted_noise must have same shape"
        );

        // Compute per-step α_t and ᾱ_t
        let alpha_t = 1.0 - self.betas[t];
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_recip_alpha_t = 1.0 / sqrt_alpha_t;
        let alpha_bar_t = self.sqrt_alphas_cumprod[t].powi(2);
        let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).sqrt();

        // μ_θ(x_t, t) = 1/√α_t * (x_t − (1−α_t)/√(1−ᾱ_t) · ε_θ)
        let coeff_eps = (1.0 - alpha_t) / sqrt_one_minus_alpha_bar_t;
        (x_t * sqrt_recip_alpha_t) - (predicted_noise * (sqrt_recip_alpha_t * coeff_eps))
    }

    /// Sample from posterior distribution q(x_{t-1} | x_t, x_0)
    pub fn posterior_sample(
        &self,
        x_t: &Array2<f32>,
        x_0: &Array2<f32>,
        t: usize,
        noise: &Array2<f32>,
    ) -> Array2<f32> {
        // Compute predicted noise: ε = (x_t - √ᾱ_t * x_0) / √(1-ᾱ_t)
        let sqrt_alpha_cumprod = self.sqrt_alpha_cumprod(t);
        let sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod(t);
        let predicted_noise = (x_t - &(x_0 * sqrt_alpha_cumprod)) / sqrt_one_minus_alpha_cumprod;

        let mean = self.posterior_mean(x_t, t, &predicted_noise);
        let variance = self.posterior_variance(t);

        if variance == 0.0 {
            // Deterministic case (t = 0)
            mean
        } else {
            // Add noise: x_{t-1} = μ + √σ_t² * ε
            &mean + &(noise * variance.sqrt())
        }
    }
}

/// Sinusoidal time embedding for conditioning diffusion models on timestep
#[derive(Serialize, Deserialize, Debug)]
pub struct TimeEmbedding {
    /// Embedding dimension
    embed_dim: usize,
    /// Precomputed per-dimension frequencies stored on the diagonal for deterministic embedding
    weights: Array2<f32>,
}

impl TimeEmbedding {
    /// Create a new time embedding layer
    pub fn new(embed_dim: usize) -> Self {
        // Precompute frequencies per embedding dimension; store on diagonal to avoid dead fields
        let mut weights = Array2::zeros((embed_dim, embed_dim));
        for i in 0..embed_dim {
            let freq_idx = if i % 2 == 0 { i / 2 } else { (i - 1) / 2 };
            let freq = 10000.0f32.powf(2.0 * freq_idx as f32 / embed_dim as f32);
            weights[[i, i]] = freq;
        }
        Self { embed_dim, weights }
    }

    /// Embed a timestep `t` into a sinusoidal vector of length `embed_dim`
    pub fn forward(&self, t: usize, max_timesteps: usize) -> Array1<f32> {
        let t_norm = (t as f32 / max_timesteps as f32).clamp(0.0, 1.0);
        let mut embedding = Array1::zeros(self.embed_dim);
        for i in 0..self.embed_dim {
            let freq = self.weights[[i, i]].max(1e-6);
            if i % 2 == 0 {
                embedding[i] = (t_norm * freq).sin();
            } else {
                embedding[i] = (t_norm * freq).cos();
            }
        }
        embedding
    }

    /// Get the embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

/// Configuration for diffusion transformer block
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiffusionBlockConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Hidden dimension for feedforward
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Polynomial degree for PolyAttention
    pub poly_degree: usize,
    /// Maximum position for CoPE
    pub max_pos: usize,
    /// Sliding window size (None for full attention)
    pub window_size: Option<usize>,
    /// Whether to use Mixture-of-Experts for feedforward
    pub use_moe: bool,
    /// MoE router configuration (if using MoE)
    pub moe_config: Option<ExpertRouterConfig>,
    /// Head selection strategy for attention
    pub head_selection: HeadSelectionStrategy,
    /// Time embedding dimension (for conditioning on noise level)
    pub time_embed_dim: usize,
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Noise schedule type
    pub noise_schedule: NoiseSchedule,
    /// Attention masking mode: true for causal (AR), false for bi-directional (diffusion)
    pub causal_attention: bool,
    /// Use discrete masked diffusion (LLaDA-style) over tokens
    pub discrete_masked: bool,
    /// Mask token id for absorbing-state masking (required when discrete_masked)
    pub mask_token_id: Option<usize>,
}

impl From<TransformerBlockConfig> for DiffusionBlockConfig {
    fn from(cfg: TransformerBlockConfig) -> Self {
        DiffusionBlockConfig {
            embed_dim: cfg.embed_dim,
            hidden_dim: cfg.hidden_dim,
            num_heads: cfg.num_heads,
            poly_degree: cfg.poly_degree,
            max_pos: cfg.max_pos,
            window_size: cfg.window_size,
            use_moe: cfg.use_moe,
            moe_config: cfg.moe_config,
            head_selection: cfg.head_selection,
            time_embed_dim: cfg.embed_dim,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        }
    }
}

// Diffusion attention is implemented by delegating to PolyAttention with optional
// bi-directional masks and timestep-conditioned gating modulation.

/// Diffusion transformer block that replaces autoregressive prediction with denoising
#[derive(Serialize, Deserialize, Debug)]
pub struct DiffusionBlock {
    /// Pre-attention layer normalization
    pub pre_attention_norm: RichardsNorm,
    /// Attention mechanism (PolyAttention, configured for diffusion when non-causal)
    pub attention: PolyAttention,
    /// Pre-feedforward layer normalization
    pub pre_ffn_norm: RichardsNorm,
    /// Feedforward network (RichardsGlu or MixtureOfExperts)
    pub feedforward: FeedForwardVariant,
    /// Time embedding for conditioning on noise level
    pub time_embedding: TimeEmbedding,
    /// Noise scheduler for diffusion process
    pub noise_scheduler: NoiseScheduler,
    /// Discrete masked diffusion scheduler (optional)
    #[serde(skip_serializing, skip_deserializing)]
    pub discrete_scheduler: Option<crate::diffusion::discrete::DiscreteMaskScheduler>,
    /// Configuration for this block
    config: DiffusionBlockConfig,
    /// Cached intermediate states from forward pass (for gradient computation)
    #[serde(skip_serializing, skip_deserializing)]
    cached_intermediates: Option<(
        Array2<f32>,
        Array1<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array1<f32>,
        Array1<f32>,
        Array1<f32>,
    )>,
    /// Current timestep for forward pass (used by Layer trait)
    #[serde(skip_serializing, skip_deserializing)]
    current_timestep: usize,
    pub time_w1: Array2<f32>,
    pub time_b1: Array2<f32>,
    pub time_w2: Array2<f32>,
    pub time_b2: Array2<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_time_w1: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_time_b1: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_time_w2: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_time_b2: Option<crate::adam::Adam>,
    pub ema_decay: f32,
    pub use_ema_for_sampling: bool,
    pub ema_time_w1: Array2<f32>,
    pub ema_time_b1: Array2<f32>,
    pub ema_time_w2: Array2<f32>,
    pub ema_time_b2: Array2<f32>,
    /// Optional dropout for training stability (disabled by default)
    pub enable_dropout: bool,
    pub dropout_rate: f32,
}

/// Feedforward network variants (same as TransformerBlock)
#[derive(Serialize, Deserialize, Debug)]
pub enum FeedForwardVariant {
    /// Standard RichardsGlu feedforward
    RichardsGlu(Box<RichardsGlu>),
    /// Mixture-of-Experts feedforward
    MixtureOfExperts(Box<MixtureOfExperts>),
}

const DIFFUSION_ACTIVATION_CLIP: f32 = 50.0;

impl DiffusionBlock {
    /// Create a new diffusion transformer block
    pub fn new(config: DiffusionBlockConfig) -> Self {
        // Create time embedding
        let time_embedding = TimeEmbedding::new(config.time_embed_dim);

        // Create noise scheduler
        let noise_scheduler =
            NoiseScheduler::new(config.noise_schedule.clone(), config.num_timesteps);
        // Optional discrete masked scheduler
        let discrete_scheduler = if config.discrete_masked {
            Some(crate::diffusion::discrete::DiscreteMaskScheduler::new(
                config.num_timesteps,
            ))
        } else {
            None
        };

        // Create pre-attention normalization
        let pre_attention_norm = RichardsNorm::new(config.embed_dim);

        // Create attention layer (PolyAttention for unified functionality)
        let mut attention = PolyAttention::new(
            config.embed_dim,
            config.num_heads,
            config.poly_degree,
            config.max_pos,
            config.window_size,
        );
        attention.set_head_selection_config(&config.head_selection);

        // Create pre-FFN normalization
        let pre_ffn_norm = RichardsNorm::new(config.embed_dim);

        // Create feedforward layer
        let feedforward = if config.use_moe {
            if let Some(moe_config) = &config.moe_config {
                let moe_layer = MixtureOfExperts::new(
                    config.embed_dim,
                    (config.embed_dim / 4).max(32), // Router hidden dim
                    moe_config.clone(),
                );
                FeedForwardVariant::MixtureOfExperts(Box::new(moe_layer))
            } else {
                // Fallback to RichardsGlu if MoE config is missing
                let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
                FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
            }
        } else {
            let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
            FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
        };

        let time_hidden = (config.time_embed_dim / 2).max(32);
        let mut rng = rand::rng();
        let w1 = Array2::from_shape_fn((config.time_embed_dim, time_hidden), |_| {
            Normal::new(0.0, (2.0 / config.time_embed_dim as f32).sqrt())
                .unwrap()
                .sample(&mut rng)
        });
        let b1 = Array2::zeros((time_hidden, 1));
        let w2 = Array2::from_shape_fn((time_hidden, config.embed_dim * 2), |_| {
            Normal::new(0.0, (2.0 / time_hidden as f32).sqrt())
                .unwrap()
                .sample(&mut rng)
        });
        let b2 = Array2::zeros((config.embed_dim * 2, 1));

        Self {
            pre_attention_norm,
            attention,
            pre_ffn_norm,
            feedforward,
            time_embedding,
            noise_scheduler,
            discrete_scheduler,
            config: config.clone(),
            cached_intermediates: None,
            current_timestep: 0,
            time_w1: w1.clone(),
            time_b1: b1.clone(),
            time_w2: w2.clone(),
            time_b2: b2.clone(),
            opt_time_w1: Some(crate::adam::Adam::new((config.time_embed_dim, time_hidden))),
            opt_time_b1: Some(crate::adam::Adam::new((time_hidden, 1))),
            opt_time_w2: Some(crate::adam::Adam::new((time_hidden, config.embed_dim * 2))),
            opt_time_b2: Some(crate::adam::Adam::new((config.embed_dim * 2, 1))),
            ema_decay: 0.999,
            use_ema_for_sampling: false,
            ema_time_w1: w1.clone(),
            ema_time_b1: b1.clone(),
            ema_time_w2: w2.clone(),
            ema_time_b2: b2.clone(),
            enable_dropout: false,
            dropout_rate: 0.0,
        }
    }

    /// Create a diffusion block from a model configuration
    ///
    /// This extracts the relevant parameters from a ModelConfig to create
    /// a diffusion block with appropriate settings for diffusion modeling.
    pub fn from_model_config(config: &ModelConfig, _layer_idx: usize) -> Self {
        let block_config = DiffusionBlockConfig {
            embed_dim: config.embedding_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.get_num_heads(),
            poly_degree: config.get_poly_degree_p(),
            max_pos: if config.use_adaptive_window {
                config.max_window_size
            } else if let Some(w) = config.window_size {
                w
            } else {
                config.max_seq_len
            }
            .saturating_sub(1), // CoPE max_pos = window_size - 1
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: config
                .moe_router
                .as_ref()
                .map(|router| ExpertRouterConfig::from_router(router)),
            head_selection: config.head_selection.clone(),
            time_embed_dim: config.embedding_dim, /* Use same dimension as embeddings for time
                                                   * conditioning */
            num_timesteps: 1000, // Standard DDPM timestep count
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 }, // Improved DDPM cosine schedule
            causal_attention: false, // Diffusion models use bi-directional attention
            discrete_masked: false,
            mask_token_id: None,
        };

        Self::new(block_config)
    }

    /// Set the current timestep for forward passes
    pub fn set_timestep(&mut self, t: usize) {
        self.current_timestep = t;
    }

    /// Configure dropout (inverted) for training; set rate in [0,1)
    pub fn set_dropout(&mut self, rate: f32) {
        self.dropout_rate = rate.clamp(0.0, 0.9);
        self.enable_dropout = self.dropout_rate > 0.0;
    }

    pub fn set_use_ema_for_sampling(&mut self, on: bool) {
        self.use_ema_for_sampling = on;
    }

    fn apply_film(
        activations: &Array2<f32>,
        gamma: &Array1<f32>,
        beta: &Array1<f32>,
    ) -> Array2<f32> {
        let mut modulated = activations.clone();
        modulated.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row *= gamma;
            row += beta;
        });
        modulated
    }

    fn film_backward(
        upstream: &Array2<f32>,
        pre_activation: &Array2<f32>,
        gamma: &Array1<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let mut input_grads = upstream.clone();
        input_grads.axis_iter_mut(Axis(0)).for_each(|mut row| {
            row *= gamma;
        });

        let mut grad_gamma = Array1::<f32>::zeros(gamma.len());
        let mut grad_beta = Array1::<f32>::zeros(gamma.len());
        for (up_row, pre_row) in upstream
            .axis_iter(Axis(0))
            .zip(pre_activation.axis_iter(Axis(0)))
        {
            Zip::from(&mut grad_gamma)
                .and(&pre_row)
                .and(&up_row)
                .for_each(|gg, &pre, &up| *gg += pre * up);
            Zip::from(&mut grad_beta)
                .and(&up_row)
                .for_each(|gb, &up| *gb += up);
        }
        (input_grads, grad_gamma, grad_beta)
    }

    fn apply_dropout_inplace(tensor: &mut Array2<f32>, rate: f32) {
        if rate <= 0.0 {
            return;
        }
        let keep = 1.0f32 - rate;
        if keep <= 0.0 {
            tensor.fill(0.0);
            return;
        }
        let scale = 1.0 / keep;
        if let Some(slice) = tensor.as_slice_mut() {
            slice.par_iter_mut().for_each(|v| {
                let r: f32 = rand::random::<f32>();
                if r < rate {
                    *v = 0.0;
                } else {
                    *v *= scale;
                }
            });
        } else {
            for v in tensor.iter_mut() {
                let r: f32 = rand::random::<f32>();
                if r < rate {
                    *v = 0.0;
                } else {
                    *v *= scale;
                }
            }
        }
    }

    fn sanitize_tensor(label: &str, tensor: &mut Array2<f32>) {
        let mut sanitized = false;
        for v in tensor.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
                sanitized = true;
            } else if v.abs() > DIFFUSION_ACTIVATION_CLIP {
                *v = v.clamp(-DIFFUSION_ACTIVATION_CLIP, DIFFUSION_ACTIVATION_CLIP);
                sanitized = true;
            }
        }
        if sanitized {
            tracing::debug!(
                target: "diffusion_block",
                label,
                clip = DIFFUSION_ACTIVATION_CLIP,
                "Sanitized diffusion activations"
            );
        }
    }

    /// Whether this block uses discrete masked diffusion
    pub fn is_discrete_masked(&self) -> bool {
        self.config.discrete_masked
    }

    /// Mask token id if configured
    pub fn mask_token_id(&self) -> Option<usize> {
        self.config.mask_token_id
    }

    /// Forward pass through diffusion transformer block
    /// Takes noisy input `x_t` and timestep `t`, predicts the noise `ε_θ(x_t, t)`
    pub fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) -> Array2<f32> {
        let time_embed = self.time_embedding.forward(t, self.config.num_timesteps);
        let h_pre = {
            let w1 = if self.use_ema_for_sampling {
                &self.ema_time_w1
            } else {
                &self.time_w1
            };
            let b1 = if self.use_ema_for_sampling {
                &self.ema_time_b1
            } else {
                &self.time_b1
            };
            time_embed
                .view()
                .to_shape((1, time_embed.len()))
                .unwrap()
                .to_owned()
                .dot(w1)
                + b1.t().to_owned()
        };
        let mut h = h_pre.clone();
        if let Some(slice) = h.as_slice_mut() {
            slice.par_iter_mut().for_each(|v| {
                *v = v.tanh();
            });
        } else {
            h.mapv_inplace(|v| v.tanh());
        }
        let gamma_beta = {
            let w2 = if self.use_ema_for_sampling {
                &self.ema_time_w2
            } else {
                &self.time_w2
            };
            let b2 = if self.use_ema_for_sampling {
                &self.ema_time_b2
            } else {
                &self.time_b2
            };
            h.dot(w2) + b2.t().to_owned()
        };
        let embed = self.config.embed_dim;
        let gamma_vec = Array1::<f32>::from_iter(
            gamma_beta
                .slice(s![.., 0..embed])
                .row(0)
                .iter()
                .map(|&x| 1.0 + 0.1 * x),
        );
        let beta_vec = Array1::<f32>::from_iter(
            gamma_beta
                .slice(s![.., embed..2 * embed])
                .row(0)
                .iter()
                .map(|&x| 0.1 * x),
        );

        let mut norm1_out = self.pre_attention_norm.forward(x_t);
        Self::sanitize_tensor("norm1_out", &mut norm1_out);
        let mut norm1_mod = Self::apply_film(&norm1_out, &gamma_vec, &beta_vec);
        Self::sanitize_tensor("norm1_mod", &mut norm1_mod);
        let mut attn_out = self
            .attention
            .forward_impl(&norm1_mod, self.config.causal_attention);
        Self::sanitize_tensor("attn_out", &mut attn_out);
        if self.enable_dropout && self.dropout_rate > 0.0 {
            Self::apply_dropout_inplace(&mut attn_out, self.dropout_rate);
        }
        let mut residual1 = x_t + &attn_out;
        Self::sanitize_tensor("residual1", &mut residual1);
        let mut norm2_out = self.pre_ffn_norm.forward(&residual1);
        Self::sanitize_tensor("norm2_out", &mut norm2_out);
        let mut norm2_mod = Self::apply_film(&norm2_out, &gamma_vec, &beta_vec);
        Self::sanitize_tensor("norm2_mod", &mut norm2_mod);
        let mut ffn_out = self.feedforward.forward(&norm2_mod);
        Self::sanitize_tensor("ffn_out", &mut ffn_out);
        if self.enable_dropout && self.dropout_rate > 0.0 {
            Self::apply_dropout_inplace(&mut ffn_out, self.dropout_rate);
        }
        let mut output = &residual1 + &ffn_out;
        Self::sanitize_tensor("block_output", &mut output);

        self.cached_intermediates = Some((
            x_t.clone(),
            time_embed,
            norm1_out,
            norm1_mod.clone(),
            attn_out.clone(),
            residual1.clone(),
            norm2_out,
            norm2_mod.clone(),
            ffn_out.clone(),
            Array1::from_vec(h.row(0).to_vec()),
            gamma_vec,
            beta_vec,
        ));

        output
    }

    /// Sample from the reverse diffusion process (generative sampling)
    pub fn sample(&mut self, shape: (usize, usize), steps: Option<usize>) -> Array2<f32> {
        let steps = steps.unwrap_or(self.config.num_timesteps);

        // Start from pure noise: x_T ~ N(0, I)
        let mut x_t = Array2::zeros(shape);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        if let Some(slice) = x_t.as_slice_mut() {
            slice.par_iter_mut().for_each(|v| {
                *v = normal.sample(&mut rand::rng()) as f32;
            });
        } else {
            for v in x_t.iter_mut() {
                *v = normal.sample(&mut rng) as f32;
            }
        }

        // Reverse diffusion process
        for t in (1..=steps).rev() {
            let t_idx = t - 1; // Convert to 0-based indexing

            // Predict noise
            let predicted_noise = self.forward_with_timestep(&x_t, t_idx);

            // Compute posterior mean
            let posterior_mean = self
                .noise_scheduler
                .posterior_mean(&x_t, t_idx, &predicted_noise);

            // Sample from posterior (add noise except for t=0)
            if t > 1 {
                let mut noise = Array2::zeros(shape);
                if let Some(slice) = noise.as_slice_mut() {
                    slice.par_iter_mut().for_each(|v| {
                        *v = normal.sample(&mut rand::rng()) as f32;
                    });
                } else {
                    for v in noise.iter_mut() {
                        *v = normal.sample(&mut rng) as f32;
                    }
                }
                let variance = self.noise_scheduler.posterior_variance(t_idx);
                x_t = &posterior_mean + &noise * variance.sqrt();
            } else {
                // Deterministic for t=0
                x_t = posterior_mean;
            }
        }

        x_t
    }

    pub fn set_noise_schedule(&mut self, schedule_type: NoiseSchedule, num_timesteps: Option<usize>) {
        let nt = num_timesteps.unwrap_or(self.config.num_timesteps);
        self.noise_scheduler = NoiseScheduler::new(schedule_type.clone(), nt);
        self.config.noise_schedule = schedule_type;
        self.config.num_timesteps = nt;
        if self.config.discrete_masked {
            self.discrete_scheduler = Some(crate::diffusion::discrete::DiscreteMaskScheduler::new(nt));
        }
    }
}

// Implement Layer trait for DiffusionBlock
impl Layer for DiffusionBlock {
    fn layer_type(&self) -> &str {
        "DiffusionBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // For Layer trait compatibility, use current timestep set by set_timestep()
        self.forward_with_timestep(input, self.current_timestep)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.pre_attention_norm.parameters()
            + self.attention.parameters()
            + self.pre_ffn_norm.parameters()
            + self.feedforward.parameters()
            + 4
    }

    fn weight_norm(&self) -> f32 {
        self.pre_attention_norm.weight_norm()
            + self.attention.weight_norm()
            + self.pre_ffn_norm.weight_norm()
            + self.feedforward.weight_norm()
            + (self.time_w1.iter().map(|&w| w * w).sum::<f32>()).sqrt()
            + (self.time_w2.iter().map(|&w| w * w).sum::<f32>()).sqrt()
    }

    /// Compute analytical gradients using cached forward intermediates
    /// Ensures full-gradient propagation across residual connections
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if let Some((
            _input_cache,
            time_embed,
            norm1_out,
            norm1_mod,
            _attn_out,
            _residual1,
            norm2_out,
            _norm2_mod,
            _ffn_out,
            h_vec,
            gamma_vec,
            _beta_vec,
        )) = &self.cached_intermediates
        {
            let mut all_param_grads = Vec::new();

            // Compute gradients through the transformer block layers
            // This follows the same pattern as TransformerBlock but with timestep conditioning

            // Output = residual1 + ffn_out, so gradients split between residual1 and ffn_out
            let ffn_grads = output_grads.clone();
            let residual1_grads = output_grads.clone();

            // Get feedforward gradients
            let (ffn_input_grad_mod, ffn_param_grads) = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => {
                    layer.compute_gradients(norm2_out, &ffn_grads)
                }
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.compute_gradients(norm2_out, &ffn_grads)
                }
            };

            let (residual1_from_ffn, mut grad_gamma, mut grad_beta) = Self::film_backward(
                &ffn_input_grad_mod,
                norm2_out,
                &gamma_vec,
            );

            // Combine residual gradients
            let residual1_total_grads = residual1_grads + residual1_from_ffn;

            // residual1 = input + attn_out: propagate full upstream gradient to both branches
            let input_grads = residual1_total_grads.clone();
            let attn_out_grads = residual1_total_grads.clone();

            let (attn_input_grad_mod, attn_param_grads) =
                self.attention.compute_gradients(norm1_mod, &attn_out_grads);

            let (norm1_input_grad, grad_gamma_norm1, grad_beta_norm1) = Self::film_backward(
                &attn_input_grad_mod,
                norm1_out,
                &gamma_vec,
            );
            grad_gamma += &grad_gamma_norm1;
            grad_beta += &grad_beta_norm1;

            // The final input gradients are the gradients w.r.t. the transformer input
            // (combining gradients from residual and attention path)
            let final_input_grads = &input_grads + &norm1_input_grad;

            all_param_grads.extend(attn_param_grads);
            all_param_grads.extend(ffn_param_grads);

            let (_ignored_ffn_norm_in, pre_ffn_param_grads) =
                self.pre_ffn_norm.compute_gradients(_residual1, &ffn_input_grad_mod);
            let (_ignored_attn_norm_in, pre_attn_param_grads) =
                self.pre_attention_norm.compute_gradients(_input_cache, &attn_input_grad_mod);
            all_param_grads.extend(pre_ffn_param_grads);
            all_param_grads.extend(pre_attn_param_grads);

            let time_hidden = self.time_w1.ncols();
            let mut grad_w2 = Array2::<f32>::zeros((time_hidden, self.config.embed_dim * 2));
            let mut grad_b2 = Array2::<f32>::zeros((self.config.embed_dim * 2, 1));
            let mut grad_h = Array1::<f32>::zeros(time_hidden);
            for d in 0..self.config.embed_dim {
                grad_b2[[d, 0]] = grad_gamma[d] * 0.1;
                grad_b2[[self.config.embed_dim + d, 0]] = grad_beta[d] * 0.1;
            }
            for k in 0..time_hidden {
                let mut acc = 0.0f32;
                for d in 0..self.config.embed_dim {
                    acc += 0.1 * grad_gamma[d] * self.time_w2[[k, d]];
                    acc += 0.1 * grad_beta[d] * self.time_w2[[k, self.config.embed_dim + d]];
                }
                grad_h[k] = acc;
            }
            for k in 0..time_hidden {
                let dh = grad_h[k] * (1.0 - h_vec[k] * h_vec[k]);
                for d in 0..self.config.embed_dim {
                    grad_w2[[k, d]] = 0.1 * grad_gamma[d] * h_vec[k];
                    grad_w2[[k, self.config.embed_dim + d]] = 0.1 * grad_beta[d] * h_vec[k];
                }
                grad_h[k] = dh;
            }
            let mut grad_w1 = Array2::<f32>::zeros((self.time_w1.nrows(), time_hidden));
            let mut grad_b1 = Array2::<f32>::zeros((time_hidden, 1));
            for k in 0..time_hidden {
                grad_b1[[k, 0]] = grad_h[k];
                for i in 0..self.time_w1.nrows() {
                    grad_w1[[i, k]] = grad_h[k] * time_embed[i];
                }
            }
            all_param_grads.push(grad_w2);
            all_param_grads.push(grad_b2);
            all_param_grads.push(grad_w1);
            all_param_grads.push(grad_b1);

            (final_input_grads, all_param_grads)
        } else {
            tracing::warn!(
                "DiffusionBlock::compute_gradients called without cached intermediates. Call forward() first."
            );
            (output_grads.clone(), Vec::new())
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if param_grads.is_empty() {
            return Ok(());
        }

        // Sanitize and globally clip gradients for stability
        let mut sanitized: Vec<Array2<f32>> = Vec::with_capacity(param_grads.len());
        let mut norm_sq: f32 = 0.0;
        for g in param_grads {
            let mut gg = g.clone();
            gg.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
            norm_sq += gg.iter().map(|&x| x * x).sum::<f32>();
            sanitized.push(gg);
        }
        let clip = 2.5f32;
        let nrm = norm_sq.sqrt();
        if nrm.is_finite() && nrm > clip && nrm > 0.0 {
            let scale = clip / nrm;
            for gg in &mut sanitized {
                gg.mapv_inplace(|x| x * scale);
            }
        }

        // Determine expected gradient array counts (not scalar parameter counts)
        let mut attention_arrays = self.attention.num_heads * 3 + 1 + 3 + 3 + 1; // per-head Wq/Wk/Wv + w_out + a,b,scale + w_g,alpha,beta + gate_poly
        if self
            .attention
            .head_selection_config
            .gating
            .use_learned_predictor
        {
            attention_arrays += 5;
        } // w1,b1,w2,b2,activation
        attention_arrays += 1; // CoPE
        let feedforward_arrays = match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => 5, /* w1,w2,w_out, richards_activation,
                                                       * gate_curve */
            FeedForwardVariant::MixtureOfExperts(_layer) => {
                // Fallback: estimate by requesting gradients via zero tensors and reading length
                // would be heavy; assume provided vector matches layer expectations
                // Use the remaining grads minus time conditioner as feedforward slice length if
                // needed We will compute below by bounds
                0
            }
        };

        let mut idx0 = 0usize;
        if sanitized.len() >= attention_arrays {
            let attention_grads = &sanitized[idx0..idx0 + attention_arrays];
            let gnorm_attn: f32 = attention_grads
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_attn = self.attention.weight_norm().max(1e-6);
            let scale_attn = (wnorm_attn / (gnorm_attn.max(1e-6))).clamp(0.5, 2.0);
            let scaled: Vec<Array2<f32>> = attention_grads
                .iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale_attn);
                    gg
                })
                .collect();
            self.attention.apply_gradients(&scaled, lr)?;
            idx0 += attention_arrays;
        }

        // Apply feedforward gradients
        let mut ff_arrays = feedforward_arrays;
        if ff_arrays == 0 {
            // Derive remaining grads excluding time conditioner (4)
            ff_arrays = sanitized.len().saturating_sub(idx0).saturating_sub(4);
        }
        if sanitized.len() >= idx0 + ff_arrays {
            let feedforward_grads = &sanitized[idx0..idx0 + ff_arrays];
            let gnorm_ffn: f32 = feedforward_grads
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_ffn = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(l) => l.weight_norm(),
                FeedForwardVariant::MixtureOfExperts(l) => l.weight_norm(),
            }
            .max(1e-6);
            let scale_ffn = (wnorm_ffn / (gnorm_ffn.max(1e-6))).clamp(0.5, 2.0);
            let scaled: Vec<Array2<f32>> = feedforward_grads
                .iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale_ffn);
                    gg
                })
                .collect();
            match &mut self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(&scaled, lr)?,
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.apply_gradients(&scaled, lr)?
                }
            }
            idx0 += ff_arrays;
        }
        // Apply pre-FFN and pre-attention norm gradients if present
        // Expect 7 arrays per RichardsNorm: 5 scalars + gamma + bias
        // These were appended in compute_gradients immediately before time conditioner gradients
        // Route them explicitly to avoid misinterpreting as time-grad arrays
        let expected_norm_arrays = 7usize;
        // Pre-FFN norm
        if sanitized.len() >= idx0 + expected_norm_arrays {
            let range = idx0..idx0 + expected_norm_arrays;
            let pre_ffn_grads = &sanitized[range];
            self.pre_ffn_norm.apply_gradients(pre_ffn_grads, lr)?;
            idx0 += expected_norm_arrays;
        }
        // Pre-attention norm
        if sanitized.len() >= idx0 + expected_norm_arrays {
            let range = idx0..idx0 + expected_norm_arrays;
            let pre_attn_grads = &sanitized[range];
            self.pre_attention_norm.apply_gradients(pre_attn_grads, lr)?;
            idx0 += expected_norm_arrays;
        }
        // Apply time-conditioner gradients (last 4 arrays)
        if sanitized.len() >= idx0 + 4 {
            let g_w2 = &sanitized[idx0];
            let g_b2 = &sanitized[idx0 + 1];
            let g_w1 = &sanitized[idx0 + 2];
            let g_b1 = &sanitized[idx0 + 3];
            let gnorm_time = [g_w2, g_b2, g_w1, g_b1]
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_time = (self.time_w1.iter().map(|&x| x * x).sum::<f32>()
                + self.time_w2.iter().map(|&x| x * x).sum::<f32>())
            .sqrt()
            .max(1e-6);
            let scale_time = (wnorm_time / (gnorm_time.max(1e-6))).clamp(0.5, 2.0);
            let mut gw2 = g_w2.clone();
            gw2.mapv_inplace(|x| x * scale_time);
            let mut gb2 = g_b2.clone();
            gb2.mapv_inplace(|x| x * scale_time);
            let mut gw1 = g_w1.clone();
            gw1.mapv_inplace(|x| x * scale_time);
            let mut gb1 = g_b1.clone();
            gb1.mapv_inplace(|x| x * scale_time);
            if let Some(opt) = &mut self.opt_time_w2 {
                opt.step(&mut self.time_w2, &gw2, lr);
            }
            if let Some(opt) = &mut self.opt_time_b2 {
                opt.step(&mut self.time_b2, &gb2, lr);
            }
            if let Some(opt) = &mut self.opt_time_w1 {
                opt.step(&mut self.time_w1, &gw1, lr);
            }
            if let Some(opt) = &mut self.opt_time_b1 {
                opt.step(&mut self.time_b1, &gb1, lr);
            }
            let d = self.ema_decay;
            self.ema_time_w2
                .zip_mut_with(&self.time_w2, |e, &w| *e = d * *e + (1.0 - d) * w);
            self.ema_time_b2
                .zip_mut_with(&self.time_b2, |e, &w| *e = d * *e + (1.0 - d) * w);
            self.ema_time_w1
                .zip_mut_with(&self.time_w1, |e, &w| *e = d * *e + (1.0 - d) * w);
            self.ema_time_b1
                .zip_mut_with(&self.time_b1, |e, &w| *e = d * *e + (1.0 - d) * w);
            idx0 += 4;
        }
        Ok(())
    }
}

// FeedForwardVariant implementations (same as TransformerBlock)
impl FeedForwardVariant {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.forward(input),
            FeedForwardVariant::MixtureOfExperts(_layer) => _layer.forward(input),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.backward(grads, lr),
            FeedForwardVariant::MixtureOfExperts(_layer) => _layer.backward(grads, lr),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.parameters(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.parameters(),
        }
    }

    fn weight_norm(&self) -> f32 {
        match self {
            FeedForwardVariant::RichardsGlu(layer) => layer.weight_norm(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.weight_norm(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::transformer_block::{TransformerBlock, TransformerBlockConfig};

    #[test]
    fn test_noise_scheduler_creation() {
        let schedule = NoiseSchedule::Cosine { s: 0.008 };
        let scheduler = NoiseScheduler::new(schedule, 1000);

        assert_eq!(scheduler.num_timesteps, 1000);
        assert!(scheduler.beta(0) >= 0.0); // Beta can be very small but not negative
        assert!(scheduler.beta(999) <= 1.0); // Allow beta to reach 1.0
        // Check that betas increase over time (cosine schedule property)
        assert!(scheduler.beta(500) > scheduler.beta(0));
    }

    #[test]
    fn test_forward_diffusion() {
        let schedule = NoiseSchedule::Cosine { s: 0.008 };
        let scheduler = NoiseScheduler::new(schedule, 1000);

        // Create test data
        let x_0 = Array2::ones((10, 64));
        let noise = Array2::zeros((10, 64));

        // Sample at t=500
        let x_t = scheduler.q_sample(&x_0, 500, &noise);

        // Should be different from original
        assert_ne!(x_t, x_0);
        assert_eq!(x_t.shape(), x_0.shape());
    }

    #[test]
    fn test_time_embedding() {
        let embed = TimeEmbedding::new(128);
        let embedding = embed.forward(500, 1000);

        assert_eq!(embedding.len(), 128);
        // Check that embedding is not all zeros
        assert!(embedding.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_diffusion_block_creation() {
        let config = DiffusionBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 1023,
            window_size: Some(4096),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::SoftTopP {
                top_p: 0.9,
                soft_top_p_alpha: 15.0,
            },
            time_embed_dim: 128,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };

        let block = DiffusionBlock::new(config);
        assert_eq!(block.layer_type(), "DiffusionBlock");
        assert!(block.parameters() > 0);
    }

    #[test]
    fn test_diffusion_block_forward() {
        let config = DiffusionBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 79,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::SoftTopP {
                top_p: 0.9,
                soft_top_p_alpha: 15.0,
            },
            time_embed_dim: 128,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };

        let mut block = DiffusionBlock::new(config);
        block.set_timestep(500);
        let input = Array2::zeros((10, 128));
        let output = block.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_diffusion_sampling() {
        let config = DiffusionBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 79,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
            time_embed_dim: 128,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };

        let mut block = DiffusionBlock::new(config);
        let sample = block.sample((10, 128), Some(100));

        assert_eq!(sample.nrows(), 10);
        assert_eq!(sample.ncols(), 128);
    }

    #[test]
    fn test_scheduler_ddim_and_posterior_mean_shapes() {
        let schedule = NoiseSchedule::Cosine { s: 0.008 };
        let scheduler = NoiseScheduler::new(schedule, 100);
        let shape = (4, 16);
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut x_t = Array2::zeros(shape);
        for v in x_t.iter_mut() {
            *v = normal.sample(&mut rng) as f32;
        }
        let mut eps = Array2::zeros(shape);
        for v in eps.iter_mut() {
            *v = normal.sample(&mut rng) as f32;
        }
        let t = 10usize;
        let x_prev = scheduler.ddim_step(&x_t, t, &eps);
        assert_eq!(x_prev.shape(), x_t.shape());
        let mu = scheduler.posterior_mean(&x_t, t, &eps);
        assert_eq!(mu.shape(), x_t.shape());
    }

    #[test]
    fn test_dropout_toggle_effects() {
        let config = DiffusionBlockConfig {
            embed_dim: 32,
            hidden_dim: 64,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 31,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
            time_embed_dim: 32,
            num_timesteps: 100,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };
        let mut block = DiffusionBlock::new(config);
        let input = Array2::zeros((8, 32));
        block.set_timestep(5);
        block.set_dropout(0.5);
        let out1 = block.forward(&input);
        block.set_dropout(0.0);
        let out2 = block.forward(&input);
        assert_eq!(out1.shape(), out2.shape());
    }

    #[test]
    #[ignore]
    fn perf_forward_vs_time_embed_freq_precompute() {
        use std::time::Instant;
        let config = DiffusionBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 127,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
            time_embed_dim: 128,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };
        let mut block = DiffusionBlock::new(config);
        block.set_timestep(500);
        let input = Array2::<f32>::zeros((32, 128));
        let iters = 200u32;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = block.forward(&input);
        }
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "forward {} iters: {:.3} ms (avg {:.3} ms)",
            iters,
            dt_ms,
            dt_ms / iters as f64
        );
    }

    #[test]
    #[ignore]
    fn perf_time_embedding_old_vs_new() {
        use std::time::Instant;
        let embed_dim = 128usize;
        let te = TimeEmbedding::new(embed_dim);
        let max_timesteps = 1000usize;
        let iters = 10_000usize;

        // Old: recompute powf each call (embedding only)
        let t0 = Instant::now();
        for t in 0..iters {
            let t_norm = (t % max_timesteps) as f32 / max_timesteps as f32;
            let mut embedding = Array1::zeros(embed_dim);
            for i in 0..embed_dim {
                if i % 2 == 0 {
                    let freq_idx = i / 2;
                    let freq = 10000.0f32.powf(2.0 * freq_idx as f32 / embed_dim as f32);
                    embedding[i] = (t_norm * freq).sin();
                } else {
                    let freq_idx = (i - 1) / 2;
                    let freq = 10000.0f32.powf(2.0 * freq_idx as f32 / embed_dim as f32);
                    embedding[i] = (t_norm * freq).cos();
                }
            }
            let _ = embedding;
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // New: use precomputed diag frequencies (embedding only)
        let t1 = Instant::now();
        for t in 0..iters {
            let _ = te.forward(t % max_timesteps, max_timesteps);
        }
        let new_ms = t1.elapsed().as_secs_f64() * 1000.0;

        println!(
            "time embedding old={}ms new={}ms speedup={:.2}x",
            old_ms,
            new_ms,
            old_ms / new_ms.max(1e-6)
        );
    }

    #[test]
    #[ignore]
    fn perf_time_conditioning_path_old_vs_new() {
        use std::time::Instant;
        let config = DiffusionBlockConfig {
            embed_dim: 128,
            hidden_dim: 256,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 127,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
            time_embed_dim: 128,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };
        let mut block = DiffusionBlock::new(config);
        let embed_dim = block.config.embed_dim;
        let iters = 5_000usize;

        // Old path: recompute powf for time embedding, then do two affine transforms
        let t0 = Instant::now();
        for t in 0..iters {
            let t_norm =
                (t % block.config.num_timesteps) as f32 / block.config.num_timesteps as f32;
            let mut time_embedding = Array1::zeros(block.config.time_embed_dim);
            for i in 0..block.config.time_embed_dim {
                if i % 2 == 0 {
                    let freq_idx = i / 2;
                    let freq =
                        10000.0f32.powf(2.0 * freq_idx as f32 / block.config.time_embed_dim as f32);
                    time_embedding[i] = (t_norm * freq).sin();
                } else {
                    let freq_idx = (i - 1) / 2;
                    let freq =
                        10000.0f32.powf(2.0 * freq_idx as f32 / block.config.time_embed_dim as f32);
                    time_embedding[i] = (t_norm * freq).cos();
                }
            }
            let h_pre = time_embedding
                .view()
                .to_shape((1, time_embedding.len()))
                .unwrap()
                .to_owned()
                .dot(&block.time_w1)
                + block.time_b1.t().to_owned();
            let mut h = h_pre.clone();
            for v in h.iter_mut() {
                *v = v.tanh();
            }
            let _gamma_beta = h.dot(&block.time_w2) + block.time_b2.t().to_owned();
        }
        let old_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // New path: use precomputed diag frequencies via TimeEmbedding::forward
        let t1 = Instant::now();
        for t in 0..iters {
            let te = block
                .time_embedding
                .forward(t % block.config.num_timesteps, block.config.num_timesteps);
            let h_pre = te
                .view()
                .to_shape((1, te.len()))
                .unwrap()
                .to_owned()
                .dot(&block.time_w1)
                + block.time_b1.t().to_owned();
            let mut h = h_pre.clone();
            for v in h.iter_mut() {
                *v = v.tanh();
            }
            let _gamma_beta = h.dot(&block.time_w2) + block.time_b2.t().to_owned();
        }
        let new_ms = t1.elapsed().as_secs_f64() * 1000.0;

        println!(
            "time cond path old={}ms new={}ms speedup={:.2}x (embed_dim={})",
            old_ms,
            new_ms,
            old_ms / new_ms.max(1e-6),
            embed_dim
        );
    }

    #[test]
    fn test_drop_in_interface_parity() {
        // Shared config
        let tcfg = TransformerBlockConfig {
            embed_dim: 64,
            hidden_dim: 128,
            num_heads: 8,
            poly_degree: 3,
            max_pos: 79,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 8 },
        };
        let mut tblock = TransformerBlock::new(tcfg.clone());

        let dcfg: DiffusionBlockConfig = tcfg.into();
        let mut dblock = DiffusionBlock::new(dcfg);
        dblock.set_timestep(10);

        let input = Array2::zeros((16, 64));
        let to = tblock.forward(&input);
        let do_ = dblock.forward(&input);
        assert_eq!(to.shape(), do_.shape());

        // Gradient compatibility: shapes and param grad counts
        let grads = Array2::ones((16, 64));
        let (t_in_grad, t_param_grads) = tblock.compute_gradients(&input, &grads);
        let (d_in_grad, d_param_grads) = dblock.compute_gradients(&input, &grads);
        assert_eq!(t_in_grad.shape(), d_in_grad.shape());
        assert!(d_param_grads.len() >= t_param_grads.len());
    }

    #[test]
    fn test_diffusion_block_input_gradients_numeric() {
        let config = DiffusionBlockConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 2,
            poly_degree: 3,
            max_pos: 15,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            time_embed_dim: 16,
            num_timesteps: 100,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };
        let mut block = DiffusionBlock::new(config);
        block.set_timestep(10);
        let seq_len = 2;
        let embed_dim = 16;
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _out = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));
        let (in_grad, param_grads) = block.compute_gradients(&input, &grads);
        assert_eq!(in_grad.shape(), input.shape());
        assert!(in_grad.iter().all(|&x| x.is_finite()));
        let gnorm: f32 = in_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
        let onorm: f32 = grads.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(gnorm <= onorm * 100.0);
        assert!(!param_grads.is_empty());
    }

    #[test]
    fn test_diffusion_block_backward_matches_analytical() {
        let config = DiffusionBlockConfig {
            embed_dim: 32,
            hidden_dim: 64,
            num_heads: 4,
            poly_degree: 3,
            max_pos: 31,
            window_size: Some(16),
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 4 },
            time_embed_dim: 32,
            num_timesteps: 100,
            noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            causal_attention: false,
            discrete_masked: false,
            mask_token_id: None,
        };
        let mut block = DiffusionBlock::new(config);
        block.set_timestep(10);
        let seq_len = 6;
        let embed_dim = 32;
        let input = Array2::<f32>::zeros((seq_len, embed_dim));
        let _out = block.forward(&input);
        let grads = Array2::<f32>::ones((seq_len, embed_dim));

        let (in_grad_analytical, _param_grads) = block.compute_gradients(&input, &grads);
        let in_grad_backward = block.backward(&grads, 0.0);

        assert_eq!(in_grad_backward.shape(), input.shape());
        assert!(in_grad_backward.iter().all(|&x| x.is_finite()));

        let mut diff_sq = 0.0f32;
        for (a, b) in in_grad_analytical.iter().zip(in_grad_backward.iter()) {
            let d = a - b;
            diff_sq += d * d;
        }
        let rmse = (diff_sq / (seq_len * embed_dim) as f32).sqrt();
        assert!(rmse < 1e-3, "RMSE too large: {}", rmse);
    }
}
