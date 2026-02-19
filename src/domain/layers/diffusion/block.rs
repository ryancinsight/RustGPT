#![allow(dead_code)]
use std::{
    f32::consts::PI,
    sync::{Arc, RwLock},
};

use ndarray::{Array1, Array2, parallel::prelude::*};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    common::{errors::Result, rng::get_rng},
    domain::{
        compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_gpu},
        layers::{
            components::{
                adaptive_residuals::AdaptiveResiduals,
                attention_context::SharedAttentionContext,
                block_core::build_shared_block_core,
                common::{CommonLayerConfig, FeedForwardVariant, TitanMemoryWorkspace},
                conditioning::{SharedFilmModulation, TimeConditioner, TimeEmbedding},
                feedforward::SharedFeedforward,
                gradient_router::GradientRouter,
                temporal_processing::SharedTemporalProcessing,
                unified_layer_workspace::UnifiedLayerWorkspace,
            },
            diffusion::edm,
            transformer::TransformerBlockConfig,
        },
        mixtures::{HeadSelectionStrategy, moe::ExpertRouterConfig},
        models::config::{DiffusionTimestepStrategy, TemporalMixingType, TitanMemoryConfig},
        network::Layer,
        richards::RichardsNorm,
    },
    infrastructure::optimizer::adam::Adam,
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

    /// Karras/EDM-inspired sigma schedule (mapped to VP-style ᾱ via σ^2 = (1-ᾱ)/ᾱ).
    ///
    /// The schedule is constructed with σ increasing from `sigma_min` → `sigma_max`.
    /// Typical image-model defaults are sigma_min≈0.002, sigma_max≈80, rho≈7.
    Karras {
        sigma_min: f32,
        sigma_max: f32,
        rho: f32,
    },
}

impl Default for NoiseSchedule {
    fn default() -> Self {
        NoiseSchedule::Cosine { s: 0.008 }
    }
}

/// Configuration for the Diffusion Block
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiffusionBlockConfig {
    pub embed_dim: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_timesteps: usize,
    pub noise_schedule: NoiseSchedule,
    pub prediction_target: DiffusionPredictionTarget,
    pub timestep_strategy: DiffusionTimestepStrategy,
    pub causal_attention: bool,
    pub window_size: Option<usize>,
    pub use_adaptive_window: bool,
    pub discrete_masked: bool,

    // Fields required by model_builder.rs
    pub poly_degree: usize,
    pub max_pos: usize,
    pub use_moe: bool,
    pub moe_config: Option<ExpertRouterConfig>,
    pub head_selection: HeadSelectionStrategy,
    #[serde(default)]
    pub moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar,
    #[serde(default)]
    pub titan_memory: TitanMemoryConfig,
    pub time_embed_dim: usize,
    pub mask_token_id: Option<usize>,

    /// Temporal mixing mechanism (Attention, RG-LRU, Mamba, or Mamba2)
    #[serde(default)]
    pub temporal_mixing: TemporalMixingType,

    /// Enable advanced weight similarity-based adaptive residuals (enabled by default)
    pub use_advanced_adaptive_residuals: bool,

    /// EDM sigma_data used for EDM-style preconditioning when `prediction_target=EdmX0`.
    ///
    /// Common default in EDM literature is `sigma_data=0.5` for images; for this
    /// embedding-space diffusion we default to `1.0`.
    #[serde(default = "edm::diffusion_edm_sigma_data_default")]
    pub edm_sigma_data: f32,

    /// Sampling method for diffusion process
    #[serde(default)]
    pub sampler: DiffusionSampler,

    /// Guidance configuration (optional)
    #[serde(default)]
    pub guidance: Option<GuidanceConfig>,

    /// Loss weighting strategy
    #[serde(default)]
    pub loss_weighting: LossWeighting,

    /// Enable P2 loss weighting (overrides loss_weighting when enabled)
    #[serde(default)]
    pub use_p2_weighting: bool,

    /// Enable SNR loss weighting (overrides loss_weighting when enabled)
    #[serde(default)]
    pub use_snr_weighting: bool,

    /// Enable adaptive guidance scale
    #[serde(default)]
    pub adaptive_guidance: bool,

    /// Minimum guidance scale for adaptive guidance
    #[serde(default = "default_min_guidance")]
    pub min_guidance_scale: f32,

    /// Maximum guidance scale for adaptive guidance
    #[serde(default = "default_max_guidance")]
    pub max_guidance_scale: f32,

    /// Policy for selecting DDIM sampling steps when the caller does not specify an explicit
    /// step count.
    #[serde(default)]
    pub ddim_steps_policy: crate::domain::layers::diffusion::DdimStepsPolicy,
}

impl From<&DiffusionBlockConfig> for CommonLayerConfig {
    fn from(config: &DiffusionBlockConfig) -> Self {
        Self {
            embed_dim: config.embed_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            poly_degree: config.poly_degree,
            max_pos: config.max_pos,
            window_size: config.window_size,
            use_moe: config.use_moe,
            moe_config: config.moe_config.clone(),
            head_selection: config.head_selection.clone(),
            moh_threshold_modulation: config.moh_threshold_modulation.clone(),
            temporal_mixing: config.temporal_mixing,
            titan_memory: config.titan_memory.clone(),
        }
    }
}

fn default_min_guidance() -> f32 {
    1.0
}

fn default_max_guidance() -> f32 {
    10.0
}

fn default_similarity_context_strength() -> Array2<f32> {
    Array2::zeros((1, 1))
}

/// Prediction target for the diffusion model
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub enum DiffusionPredictionTarget {
    /// Predict the noise (epsilon) added to the input
    #[default]
    Epsilon,
    /// Predict the velocity (v) - see "Progressive Distillation for Fast Sampling of Diffusion
    /// Models"
    VPrediction,
    /// Predict the original sample (x_0)
    Sample,

    /// EDM-style preconditioned denoised sample (x_0) computed as:
    /// x0_hat = c_skip(σ)*x_t + c_out(σ)*F_b8(c_in(σ)*x_t, t)
    ///
    /// The model core predicts `F_b8` and the block returns `x0_hat`.
    EdmX0,
}

/// Diffusion noise scheduler that manages variance schedules and cumulative products
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NoiseScheduler {
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

        let mut posterior_variance = Array1::zeros(num_timesteps);
        if num_timesteps > 0 {
            posterior_variance[0] = 0.0;
        }
        for t in 1..num_timesteps {
            let beta = betas[t - 1].clamp(0.0, 1.0);
            let alpha_bar_prev = alphas_cumprod[t - 1].clamp(0.0, 1.0);
            let alpha_bar_t = alphas_cumprod[t].clamp(0.0, 1.0);
            let denom = (1.0 - alpha_bar_t).max(1e-12);
            posterior_variance[t] = (beta * (1.0 - alpha_bar_prev) / denom).clamp(0.0, 1.0);
        }

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
        if num_timesteps == 0 {
            return Array1::zeros(0);
        }
        if num_timesteps == 1 {
            return Array1::zeros(1);
        }
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
            NoiseSchedule::Karras {
                sigma_min,
                sigma_max,
                rho,
            } => {
                // Build σ(t) increasing, then map ᾱ(t)=1/(1+σ(t)^2) and derive β.
                let tmax = (num_timesteps - 1).max(1) as f32;
                let rho = rho.max(1e-3);
                let smin = sigma_min.max(1e-6);
                let smax = sigma_max.max(smin);
                let smin_r = smin.powf(1.0 / rho);
                let smax_r = smax.powf(1.0 / rho);

                let mut alpha_bar = Array1::<f32>::zeros(num_timesteps + 1);
                alpha_bar[0] = 1.0;
                for t in 1..=num_timesteps {
                    let frac = (t - 1) as f32 / tmax;
                    let sigma = (smin_r + frac * (smax_r - smin_r)).powf(rho);
                    let ab = 1.0 / (1.0 + sigma * sigma);
                    alpha_bar[t] = ab.clamp(1e-12, 1.0);
                }

                let mut betas = Array1::<f32>::zeros(num_timesteps);
                for t in 0..num_timesteps {
                    let alpha_t = (alpha_bar[t + 1] / alpha_bar[t]).clamp(1e-12, 1.0);
                    betas[t] = (1.0 - alpha_t).clamp(1e-8, 0.999);
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

    /// DDIM sampling: Get previous sample using DDIM formula
    /// x_{t-1} = √(ᾱ_{t-1}/ᾱ_t) * x_t - √((1-ᾱ_{t-1})/ᾱ_t) * ε_θ + √(1-ᾱ_{t-1}) * z (if eta > 0)
    pub fn ddim_step(
        &self,
        x_t: &Array2<f32>,
        t: usize,
        pred_epsilon: &Array2<f32>,
        eta: f32,
        random_sample: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let t_prev = t.saturating_sub(1);
        self.ddim_step_between(x_t, t, t_prev, pred_epsilon, eta, random_sample)
    }

    /// DDIM step generalized to an arbitrary previous timestep.
    ///
    /// This is required for numerically-correct reduced-step samplers (DDIM/PNDM/DPM-Solver).
    pub fn ddim_step_between(
        &self,
        x_t: &Array2<f32>,
        t: usize,
        t_prev: usize,
        pred_epsilon: &Array2<f32>,
        eta: f32,
        random_sample: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let alpha_cumprod_t = self.sqrt_alpha_cumprod(t).powi(2);
        let alpha_cumprod_prev = self.sqrt_alpha_cumprod(t_prev).powi(2);

        let sqrt_alpha_cumprod_prev = alpha_cumprod_prev.sqrt();
        let sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt();

        let sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod(t);
        let sqrt_one_minus_alpha_cumprod_prev = self.sqrt_one_minus_alpha_cumprod(t_prev);

        // Coefficients for DDIM
        let coeff1 = sqrt_alpha_cumprod_prev / sqrt_alpha_cumprod_t;
        let coeff2 = sqrt_one_minus_alpha_cumprod_prev / sqrt_alpha_cumprod_t;

        // Deterministic component
        let mut x_prev = coeff1 * x_t - coeff2 * pred_epsilon;

        // Stochastic component (if eta > 0)
        if eta > 0.0
            && let Some(z) = random_sample
        {
            let sigma_t = eta * sqrt_one_minus_alpha_cumprod_t / sqrt_alpha_cumprod_t;
            x_prev = x_prev + sigma_t * z;
        }

        x_prev
    }

    /// P2 loss weighting from Nichol & Dhariwal 2021
    /// w(t) = (1 - ᾱ_t) / (1 - ᾱ_{t-1}) * (1 - ᾱ_{t-1}) / (1 - ᾱ_t) = 1.0
    /// Actually: w(t) = (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    pub fn p2_weight(&self, t: usize) -> f32 {
        if t == 0 {
            return 1.0;
        }
        let one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod(t).powi(2);
        let one_minus_alpha_cumprod_t_minus_1 = self.sqrt_one_minus_alpha_cumprod(t - 1).powi(2);

        if one_minus_alpha_cumprod_t < 1e-6 {
            return 1.0;
        }

        (one_minus_alpha_cumprod_t_minus_1 / one_minus_alpha_cumprod_t).clamp(0.0, 10.0)
    }

    /// SNR loss weighting: w(t) = SNR(t) = α_t / (1 - α_t)
    pub fn snr_weight(&self, t: usize) -> f32 {
        let alpha_t = self.alpha(t);
        if alpha_t >= 1.0 - 1e-6 {
            return 1.0;
        }
        (alpha_t / (1.0 - alpha_t)).clamp(0.0, 10.0)
    }

    /// Adaptive loss weighting combining P2 and SNR
    pub fn adaptive_weight(&self, _t: usize, p2_weight: f32, snr_weight: f32) -> f32 {
        // Simple combination: geometric mean
        (p2_weight * snr_weight).sqrt().clamp(0.1, 10.0)
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

        if t == 0 {
            return x_t.clone();
        }

        let beta = self
            .betas
            .get(t - 1)
            .copied()
            .unwrap_or(0.0)
            .clamp(0.0, 1.0);
        let alpha_t = (1.0 - beta).clamp(1e-12, 1.0);
        let sqrt_alpha_t = alpha_t.sqrt();
        let sqrt_recip_alpha_t = 1.0 / sqrt_alpha_t.max(1e-12);
        let alpha_bar_t = self.sqrt_alphas_cumprod[t].powi(2).clamp(1e-12, 1.0);
        let sqrt_one_minus_alpha_bar_t = (1.0 - alpha_bar_t).max(1e-12).sqrt();

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
        if t == 0 {
            return x_0.clone();
        }

        // Compute predicted noise: ε = (x_t - √ᾱ_t * x_0) / √(1-ᾱ_t)
        let sqrt_alpha_cumprod = self.sqrt_alpha_cumprod(t);
        let sqrt_one_minus_cumprod = self.sqrt_one_minus_alpha_cumprod(t);
        if !(sqrt_one_minus_cumprod.is_finite()) || sqrt_one_minus_cumprod.abs() < 1e-12 {
            return x_0.clone();
        }
        let predicted_noise = (x_t - &(x_0 * sqrt_alpha_cumprod)) / sqrt_one_minus_cumprod;

        let mean = self.posterior_mean(x_t, t, &predicted_noise);
        let variance = self.posterior_variance(t).max(0.0);

        if variance == 0.0 {
            // Deterministic case (t = 0)
            mean
        } else {
            // Add noise: x_{t-1} = μ + √σ_t² * ε
            &mean + &(noise * variance.sqrt())
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionCachedIntermediates {
    pub input_original: Arc<Array2<f32>>,
    pub input_used: Arc<Array2<f32>>,
    pub time_embed: Arc<Array1<f32>>,
    pub gamma_beta: Arc<Array1<f32>>,
    pub norm1_out: Arc<Array2<f32>>,
    pub norm1_mod: Arc<Array2<f32>>,
    pub attn_out: Arc<Array2<f32>>,
    pub residual1: Arc<Array2<f32>>,
    pub norm2_out: Arc<Array2<f32>>,
    pub norm2_mod: Arc<Array2<f32>>,
    pub ffn_out: Arc<Array2<f32>>,
    pub output: Arc<Array2<f32>>,
    pub h_vec: Arc<Array1<f32>>,
    pub gamma_attn: Arc<Array2<f32>>,
    pub beta_attn: Arc<Array2<f32>>,
    pub gamma_ffn: Arc<Array2<f32>>,
    pub beta_ffn: Arc<Array2<f32>>,
    pub timestep: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DiffusionParamPartitions {
    pub temporal_mixing: usize,
    pub feedforward: usize,
    pub pre_ffn_norm: usize,
    pub pre_attention_norm: usize,
    pub similarity_context_strength: usize,
    pub time_conditioner: usize,
    pub time_embedding: usize,
    // Adaptive residual parameter partitions (9 optimizers total)
    pub adaptive_residual_similarity: usize,
    pub adaptive_residual_affinity: usize,
    pub adaptive_residual_attention: usize,
    pub adaptive_residual_channel: usize,
    pub adaptive_residual_scales_attention: usize,
    pub adaptive_residual_scales_ffn: usize,
    // Theorem 4 extension partitions
    pub adaptive_residual_positional_qkv: usize,
    pub adaptive_residual_positional_cope: usize,
    pub adaptive_residual_positional_weights: usize,
}

impl DiffusionParamPartitions {
    fn total(&self) -> usize {
        self.temporal_mixing
            + self.feedforward
            + self.pre_ffn_norm
            + self.pre_attention_norm
            + self.similarity_context_strength
            + self.time_conditioner
            + self.time_embedding
            + self.adaptive_residual_similarity
            + self.adaptive_residual_affinity
            + self.adaptive_residual_attention
            + self.adaptive_residual_channel
            + self.adaptive_residual_scales_attention
            + self.adaptive_residual_scales_ffn
            + self.adaptive_residual_positional_qkv
            + self.adaptive_residual_positional_cope
            + self.adaptive_residual_positional_weights
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DiffusionBlock {
    pub config: DiffusionBlockConfig,
    #[serde(alias = "attention")]
    pub temporal_mixing: SharedTemporalProcessing,
    pub feedforward: SharedFeedforward,
    pub pre_attention_norm: RichardsNorm,
    pub pre_ffn_norm: RichardsNorm,
    pub(crate) time_embedding: TimeEmbedding,
    pub time_conditioner: TimeConditioner,
    pub(crate) noise_scheduler: NoiseScheduler,
    #[serde(skip)]
    pub cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
    #[serde(skip)]
    pub discrete_scheduler:
        Option<crate::domain::layers::diffusion::discrete::DiscreteMaskScheduler>,
    pub current_window_size: Option<usize>,
    pub win_max: usize,
    pub win_min: usize,
    pub win_step_up: usize,
    pub win_step_down: usize,
    pub pred_up: f32,
    pub pred_down: f32,
    pub adaptive_window_on: bool,
    pub enable_dropout: bool,
    pub dropout_rate: f32,
    pub use_ema_for_sampling: bool,
    pub ema_decay: f32,
    pub current_timestep: usize,
    #[serde(skip)]
    pub param_partitions: RwLock<Option<DiffusionParamPartitions>>,
    #[serde(skip)]
    pub adaptive_residuals: Option<AdaptiveResiduals>,
    #[serde(skip_serializing, skip_deserializing)]
    activation_similarity_matrix: Array2<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    incoming_similarity_context: Option<Array2<f32>>,
    #[serde(default = "default_similarity_context_strength")]
    similarity_context_strength: Array2<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    similarity_update_rate: f32,
    #[serde(skip)]
    pub context: SharedAttentionContext,
    #[serde(skip)]
    pub opt_similarity_context_strength: Adam,
    #[serde(skip)]
    pub film_modulation: SharedFilmModulation,
    #[serde(skip)]
    titan_memory_workspace: TitanMemoryWorkspace,
    #[serde(skip)]
    unified_workspace: UnifiedLayerWorkspace,
    #[serde(skip, default)]
    compute_backend: ComputeBackend,
}

impl DiffusionBlock {
    pub fn new(config: DiffusionBlockConfig) -> Self {
        let embed_dim = config.embed_dim;
        let hidden_dim = config.hidden_dim;
        let time_embed_dim = config.time_embed_dim;
        let num_timesteps = config.num_timesteps;
        let max_pos = config.max_pos;
        let window_size = config.window_size;
        let use_adaptive_window = config.use_adaptive_window;
        let use_advanced_adaptive_residuals = config.use_advanced_adaptive_residuals;
        let noise_schedule = config.noise_schedule.clone();

        let common_config = CommonLayerConfig::from(&config);
        let core_layers = build_shared_block_core(&common_config, window_size, use_adaptive_window);

        let time_embedding = TimeEmbedding::new(time_embed_dim);
        // Output dim of time conditioner = 4 * embed_dim (gamma_attn, beta_attn, gamma_ffn,
        // beta_ffn)
        let time_conditioner = TimeConditioner::new(time_embed_dim, hidden_dim, embed_dim * 4);
        let noise_scheduler = NoiseScheduler::new(noise_schedule, num_timesteps);

        let discrete_scheduler = if config.discrete_masked {
            Some(
                crate::domain::layers::diffusion::discrete::DiscreteMaskScheduler::new(
                    num_timesteps,
                ),
            )
        } else {
            None
        };

        let mut context = SharedAttentionContext::new();
        context.similarity_context_strength = default_similarity_context_strength();
        context.similarity_update_rate = 0.01;

        let opt_similarity_context_strength = Adam::new((1, 1));

        let mut film_modulation = SharedFilmModulation::new(embed_dim);
        film_modulation.scale_gamma = 0.1;
        film_modulation.scale_beta = 0.1;

        Self {
            config,
            temporal_mixing: core_layers.temporal_mixing,
            feedforward: core_layers.feedforward,
            pre_attention_norm: core_layers.pre_attention_norm,
            pre_ffn_norm: core_layers.pre_ffn_norm,
            time_embedding,
            time_conditioner,
            noise_scheduler,
            cached_intermediates: RwLock::new(None),
            discrete_scheduler,
            current_window_size: window_size,
            win_max: max_pos,
            win_min: 16,
            win_step_up: 16,
            win_step_down: 16,
            pred_up: 1.2,
            pred_down: 0.8,
            adaptive_window_on: use_adaptive_window,
            enable_dropout: false,
            dropout_rate: 0.0,
            use_ema_for_sampling: false,
            ema_decay: 0.999,
            current_timestep: 0,
            param_partitions: RwLock::new(None),
            adaptive_residuals: if use_advanced_adaptive_residuals {
                let mut residuals = AdaptiveResiduals::new_minimal(embed_dim);
                residuals.max_seq_len = num_timesteps.min(2048);
                Some(residuals)
            } else {
                None
            },
            activation_similarity_matrix: Array2::zeros((embed_dim, embed_dim)),
            incoming_similarity_context: None,
            similarity_context_strength: default_similarity_context_strength(),
            similarity_update_rate: 0.01,
            context,
            opt_similarity_context_strength,
            film_modulation,
            titan_memory_workspace: TitanMemoryWorkspace::default(),
            unified_workspace: {
                let mut ws = UnifiedLayerWorkspace::new();
                // Enable diffusion-specific buffers (time embedding, FiLM parameters, etc.)
                ws.set_diffusion_buffers_enabled(true);
                ws
            },
            compute_backend: ComputeBackend::Cpu,
        }
    }

    pub fn max_seq_len(&self) -> usize {
        self.config.max_pos.saturating_add(1)
    }

    pub fn activation_similarity_matrix(&self) -> Option<&Array2<f32>> {
        self.context.get_outgoing_context()
    }

    pub fn set_incoming_similarity_context(&mut self, context: Option<&Array2<f32>>) {
        self.context
            .set_incoming_context_checked_reuse(context, self.config.embed_dim);
    }

    /// Get the cached intermediates
    pub fn get_cache(&self) -> Option<DiffusionCachedIntermediates> {
        self.cached_intermediates.read().unwrap().clone()
    }

    /// Set the cached intermediates
    pub fn set_cache(&self, cache: Option<DiffusionCachedIntermediates>) {
        *self.cached_intermediates.write().unwrap() = cache;
    }

    pub fn set_timestep(&mut self, t: usize) {
        self.current_timestep = t;
    }

    pub fn set_use_ema_for_sampling(&mut self, use_ema: bool) {
        self.use_ema_for_sampling = use_ema;
    }

    pub fn set_causal_attention(&mut self, causal: bool) {
        self.config.causal_attention = causal;
    }

    /// Get reference to temporal mixing layer.
    #[inline]
    pub fn temporal_mixing(&self) -> &SharedTemporalProcessing {
        &self.temporal_mixing
    }

    /// Get mutable reference to temporal mixing layer.
    #[inline]
    pub fn temporal_mixing_mut(&mut self) -> &mut SharedTemporalProcessing {
        &mut self.temporal_mixing
    }

    /// Get reference to feedforward layer.
    #[inline]
    pub fn feedforward(&self) -> &FeedForwardVariant {
        self.feedforward.variant()
    }

    /// Get mutable reference to feedforward layer.
    #[inline]
    pub fn feedforward_mut(&mut self) -> &mut FeedForwardVariant {
        self.feedforward.variant_mut()
    }

    pub fn set_training_mode(&mut self, is_training: bool) {
        if let Some(moe) = self.feedforward_mut().as_moe_mut() {
            moe.set_training_mode(is_training);
        }
    }

    /// Set runtime compute backend for this block and shared components.
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set DiffusionBlock backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    ///
    /// When a GPU backend is selected this eagerly validates shared component
    /// GPU readiness, surfacing unsupported GPU variants immediately.
    pub fn set_compute_backend_checked(&mut self, compute_backend: ComputeBackend) -> Result<()> {
        self.compute_backend = compute_backend;
        self.unified_workspace.set_compute_backend(compute_backend);
        self.temporal_mixing
            .set_compute_backend_checked(compute_backend)?;
        self.feedforward
            .set_compute_backend_checked(compute_backend)?;
        Ok(())
    }

    /// Resolve and apply strict auto-GPU backend preference.
    ///
    /// Uses automatic GPU detection with strict no-fallback behavior when a GPU
    /// is detected but not usable for the selected layer variants.
    pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        self.set_compute_backend_checked(backend)
    }

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    pub fn min_snr_weight(&self, t: usize, gamma: f32) -> f32 {
        // SNR = ᾱ / (1-ᾱ) for the VP forward process.
        // Use Min-SNR weighting (Chen, 2023) with parameterization-specific variants.
        let alpha_cumprod = self
            .noise_scheduler
            .sqrt_alpha_cumprod(t)
            .powi(2)
            .clamp(1e-12, 1.0 - 1e-12);
        let snr = (alpha_cumprod / (1.0 - alpha_cumprod)).max(1e-12);
        let gamma = gamma.max(1e-12);
        let snr_clipped = snr.min(gamma);

        match self.config.prediction_target {
            // ε-objective: w = min(snr, γ) / snr
            DiffusionPredictionTarget::Epsilon => snr_clipped / snr,
            // v-objective: w = min(snr, γ) / (snr + 1)
            DiffusionPredictionTarget::VPrediction => snr_clipped / (snr + 1.0),
            // x0-objective: w = min(snr, γ)
            DiffusionPredictionTarget::Sample | DiffusionPredictionTarget::EdmX0 => snr_clipped,
        }
    }

    /// EDM-style loss weight for denoised (x0) objective when using `EdmX0`.
    ///
    /// This is only meaningful for denoising-in-x0 losses; we keep it separate from
    /// Min-SNR weighting so callers can combine them if desired.
    pub fn edm_loss_weight(&self, t: usize) -> f32 {
        let sigma = self.sigma_from_timestep(t).max(1e-6);
        edm::loss_weight_from_sigma(sigma, self.config.edm_sigma_data)
    }

    #[inline]
    fn sigma_from_timestep(&self, t: usize) -> f32 {
        // VP-style mapping: c3^2 = (1-b1c4)/b1c4, where b1c4 = b1bar(t).
        let alpha_bar = self
            .noise_scheduler
            .sqrt_alpha_cumprod(t)
            .powi(2)
            .clamp(1e-12, 1.0);
        edm::sigma_from_alpha_bar(alpha_bar)
    }

    #[inline]
    fn edm_precond_scales(&self, t: usize) -> (f32, f32, f32) {
        // Returns (c_in, c_skip, c_out)
        let sigma = self.sigma_from_timestep(t);
        edm::precond_scales_from_sigma(sigma, self.config.edm_sigma_data)
    }

    pub fn is_discrete_masked(&self) -> bool {
        self.config.discrete_masked
    }

    pub fn mask_token_id(&self) -> Option<usize> {
        self.config.mask_token_id
    }

    pub fn training_target(&self, x0: &Array2<f32>, noise: &Array2<f32>, t: usize) -> Array2<f32> {
        match self.config.prediction_target {
            DiffusionPredictionTarget::Epsilon => noise.clone(),
            DiffusionPredictionTarget::Sample => x0.clone(),
            DiffusionPredictionTarget::EdmX0 => x0.clone(),
            DiffusionPredictionTarget::VPrediction => {
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                (sqrt_alpha * noise) - (sqrt_one_minus_alpha * x0)
            }
        }
    }

    fn sanitize_tensor(_name: &str, tensor: &mut Array2<f32>) {
        tensor.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });
    }

    fn apply_dropout_inplace(input: &mut Array2<f32>, rate: f32) {
        let _rng = get_rng();
        let scale = 1.0 / (1.0 - rate);
        input.mapv_inplace(|x| {
            if rand::random::<f32>() > rate {
                x * scale
            } else {
                0.0
            }
        });
    }

    fn convert_prediction_to_epsilon(
        &self,
        x_t: &Array2<f32>,
        output: &Array2<f32>,
        t: usize,
    ) -> Array2<f32> {
        match self.config.prediction_target {
            DiffusionPredictionTarget::Epsilon => output.clone(),
            DiffusionPredictionTarget::Sample => {
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                if sqrt_one_minus_alpha.is_finite() && sqrt_one_minus_alpha > 0.0 {
                    (x_t - (output * sqrt_alpha)) / sqrt_one_minus_alpha
                } else {
                    Array2::<f32>::zeros(x_t.raw_dim())
                }
            }
            DiffusionPredictionTarget::EdmX0 => {
                // EdmX0 returns x0_hat, so conversion is identical to Sample.
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                if sqrt_one_minus_alpha.is_finite() && sqrt_one_minus_alpha > 0.0 {
                    (x_t - (output * sqrt_alpha)) / sqrt_one_minus_alpha
                } else {
                    Array2::<f32>::zeros(x_t.raw_dim())
                }
            }
            DiffusionPredictionTarget::VPrediction => {
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                // For v-prediction (Salimans & Ho): eps = sqrt(1-ᾱ_t) * x_t + sqrt(ᾱ_t) * v
                (x_t * sqrt_one_minus_alpha) + (output * sqrt_alpha)
            }
        }
    }

    /// Predict epsilon regardless of the configured target (epsilon/v/x0/EDM x0).
    #[inline]
    pub fn predict_epsilon_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) -> Array2<f32> {
        let pred = self.forward_with_timestep(x_t, t);
        self.convert_prediction_to_epsilon(x_t, &pred, t)
    }

    /// Forward pass through diffusion transformer block.
    ///
    /// Returns the model prediction in the configured parameterization
    /// (`Epsilon`, `VPrediction`, `Sample`, or `EdmX0`).
    pub fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) -> Array2<f32> {
        // GPU execution is handled by sub-components (temporal_mixing, feedforward)
        // which have their own GPU forward paths.
        self.unified_workspace.ensure_exact_execution_capacity(
            x_t.nrows(),
            x_t.ncols(),
            self.config.embed_dim,
        );
        if self.current_window_size != self.config.window_size {
            self.config.window_size = self.current_window_size;
        }
        let time_embed = self.time_embedding.forward(t, self.config.num_timesteps);
        let (gamma_beta, h) = self
            .time_conditioner
            .forward(&time_embed, self.use_ema_for_sampling);

        // Update FiLM modulation parameters using the shared component
        self.film_modulation
            .update(gamma_beta.as_slice().unwrap(), self.config.embed_dim);

        let (x_model_in, c_skip, c_out, edm_on) =
            if self.config.prediction_target == DiffusionPredictionTarget::EdmX0 {
                let (c_in, c_skip, c_out) = self.edm_precond_scales(t);
                (x_t * c_in, c_skip, c_out, true)
            } else {
                (x_t.clone(), 0.0, 1.0, false)
            };

        let input_original_arc = Arc::new(x_model_in);
        let input_used_arc: Arc<Array2<f32>> = if self.context.has_context() {
            let input_used_buf = self
                .unified_workspace
                .residual1_mut()
                .expect("residual1 workspace buffer must be allocated");
            let applied = self
                .context
                .apply_context_into(input_original_arc.as_ref(), input_used_buf)
                .unwrap_or_else(|err| {
                    panic!("DiffusionBlock context apply_context_into failed: {err}")
                });
            if applied {
                Arc::new(input_used_buf.clone())
            } else {
                input_original_arc.clone()
            }
        } else {
            input_original_arc.clone()
        };

        // Pre-attention norm into workspace buffer.
        {
            let norm1_out_buf = self
                .unified_workspace
                .norm1_out_mut()
                .expect("norm1_out workspace buffer must be allocated");
            self.pre_attention_norm
                .normalize_into(input_used_arc.as_ref(), norm1_out_buf);
        }
        let norm1_out = self
            .unified_workspace
            .norm1_out()
            .expect("norm1_out workspace buffer must be allocated")
            .clone();

        // FiLM attention conditioning into reusable workspace buffer.
        {
            let norm1_mod_buf = self
                .unified_workspace
                .ffn_intermediate_mut()
                .expect("ffn_intermediate workspace buffer must be allocated");
            self.film_modulation
                .apply_attn_conditioning_into(&norm1_out, norm1_mod_buf)
                .unwrap_or_else(|err| {
                    panic!("DiffusionBlock FiLM attn conditioning failed: {err}")
                });
        }
        let norm1_mod = self
            .unified_workspace
            .ffn_intermediate()
            .expect("ffn_intermediate workspace buffer must be allocated")
            .clone();

        // Update window size if needed
        self.temporal_mixing
            .set_window_size(self.current_window_size);

        // Forward with conditioning (although Attention might not use it directly in this variant,
        // we use the film-modulated input).
        // For SharedTemporalProcessing, we can pass None for conditioning if we already modulated the input,
        // OR we can pass the modulation vectors if the layer supports internal modulation.
        // Given we manually modulated `norm1_mod` above, we pass that as input.
        // However, `SharedTemporalProcessing`'s `forward_with_film` might be cleaner if we remove the manual modulation above.
        // Let's stick to the existing pattern: Manual modulation -> Layer.
        // But wait, the goal is to use `SharedTemporalProcessing`.
        // Let's check `SharedTemporalProcessing`. It has `forward` and `forward_with_film`.
        // If we use `forward_with_film`, it expects `gamma` and `beta`.
        // The previous code did: norm1_mod = apply_film(norm1_out, gamma, beta); attn_out = temporal_mixing.forward(norm1_mod)
        // So we will keep that pattern for now to match the "Manual Modulation" style of DiffusionBlock,
        // unless we want to push modulation *into* the layer.
        // For now, let's keep manual modulation using the helper, then call the layer.

        {
            let attn_out_buf = self
                .unified_workspace
                .temporal_out_mut()
                .expect("temporal_out workspace buffer must be allocated");
            self.temporal_mixing
                .forward_with_titan_fusion_into(
                    &norm1_mod,
                    attn_out_buf,
                    self.config.causal_attention,
                    &self.config.titan_memory,
                    &mut self.titan_memory_workspace,
                )
                .unwrap_or_else(|err| panic!("DiffusionBlock temporal forward_into failed: {err}"));
        }
        let mut attn_out = self
            .unified_workspace
            .temporal_out()
            .expect("temporal_out workspace buffer must be allocated")
            .clone();
        if self.enable_dropout && self.dropout_rate > 0.0 {
            Self::apply_dropout_inplace(&mut attn_out, self.dropout_rate);
        }
        self.context.update_outgoing_context(
            &input_used_arc.as_ref().view(),
            &attn_out.view(),
            self.config.embed_dim,
        );

        let head_activity = self.temporal_mixing.head_activity_summary();
        let head_activity_ratio = head_activity.ratio;
        let head_activity_vec = head_activity.head_activity;
        let token_head_activity_vec = head_activity.token_head_activity;
        let residual1 = if let Some(ref mut adaptive_residuals) = self.adaptive_residuals {
            adaptive_residuals.apply_attention_residual_with_moh(
                input_used_arc.as_ref(),
                &attn_out,
                Some(head_activity_ratio),
                head_activity_vec,
            )
        } else {
            input_used_arc.as_ref() + &attn_out
        };
        {
            let norm2_out_buf = self
                .unified_workspace
                .norm2_out_mut()
                .expect("norm2_out workspace buffer must be allocated");
            self.pre_ffn_norm.normalize_into(&residual1, norm2_out_buf);
        }
        let norm2_out = self
            .unified_workspace
            .norm2_out()
            .expect("norm2_out workspace buffer must be allocated")
            .clone();

        // FFN Modulation
        // We can use the new `forward_with_film` in `SharedFeedforward`!
        // But first let's see if we want to modulate *before* passing or pass parameters.
        // SharedFeedforward::forward(input) just runs the FFN.
        // If we want FiLM, we should use `forward_with_film` OR modulate manually.
        // The previous code did manual modulation: `norm2_mod = apply_film(...)`.
        // Let's switch to using `SharedFeedforward::forward_with_film` if possible,
        // OR just keep manual modulation + standard forward for consistency with Attention.
        // Actually, `SharedFeedforward` has `forward_with_film`.
        // However, `SharedTemporalProcessing` does NOT have `forward_with_film` exposed in the same way (it delegates).
        // To be consistent, let's do manual modulation here using `SharedFilmModulation` helper, then standard forward.

        {
            let norm2_mod_buf = self
                .unified_workspace
                .residual1_mut()
                .expect("residual1 workspace buffer must be allocated");
            self.film_modulation
                .apply_ffn_conditioning_into(&norm2_out, norm2_mod_buf)
                .unwrap_or_else(|err| panic!("DiffusionBlock FiLM ffn conditioning failed: {err}"));
        }
        let norm2_mod = self
            .unified_workspace
            .residual1()
            .expect("residual1 workspace buffer must be allocated")
            .clone();

        {
            let ffn_out_buf = self
                .unified_workspace
                .ffn_out_mut()
                .expect("ffn_out workspace buffer must be allocated");
            self.feedforward
                .forward_with_token_head_activity_into(
                    &norm2_mod,
                    ffn_out_buf,
                    Some(head_activity_ratio),
                    head_activity_vec,
                    token_head_activity_vec,
                )
                .unwrap_or_else(|err| {
                    panic!("DiffusionBlock feedforward forward_into failed: {err}")
                });
        }
        let mut ffn_out = self
            .unified_workspace
            .ffn_out()
            .expect("ffn_out workspace buffer must be allocated")
            .clone();

        if self.enable_dropout && self.dropout_rate > 0.0 {
            Self::apply_dropout_inplace(&mut ffn_out, self.dropout_rate);
        }
        // Apply advanced adaptive residuals for FFN residual connection if enabled
        let output = if let Some(ref mut adaptive_residuals) = self.adaptive_residuals {
            adaptive_residuals.apply_ffn_residual(&residual1, &ffn_out)
        } else {
            // Standard residual connection
            &residual1 + &ffn_out
        };

        let prediction = if edm_on {
            (x_t * c_skip) + (&output * c_out)
        } else {
            output
        };
        if prediction.iter().any(|v| !v.is_finite()) {
            panic!("DiffusionBlock forward produced non-finite prediction");
        }

        // Store intermediates Arc-backed so cache clones are shallow (important for LRM replay).
        let h_vec = h.row(0).to_owned();
        let cached_output = prediction.clone();

        *self.cached_intermediates.write().unwrap() = Some(DiffusionCachedIntermediates {
            input_original: input_original_arc,
            input_used: input_used_arc,
            time_embed: Arc::new(time_embed),
            norm1_out: Arc::new(norm1_out),
            norm1_mod: Arc::new(norm1_mod),
            residual1: Arc::new(residual1),
            norm2_out: Arc::new(norm2_out),
            norm2_mod: Arc::new(norm2_mod),
            h_vec: Arc::new(h_vec),
            gamma_attn: Arc::new(self.film_modulation.gamma_attn.clone()),
            beta_attn: Arc::new(self.film_modulation.beta_attn.clone()),
            gamma_ffn: Arc::new(self.film_modulation.gamma_ffn.clone()),
            beta_ffn: Arc::new(self.film_modulation.beta_ffn.clone()),
            gamma_beta: Arc::new(gamma_beta),
            attn_out: Arc::new(attn_out),
            ffn_out: Arc::new(ffn_out),
            output: Arc::new(cached_output),
            timestep: t,
        });
        if self.adaptive_window_on {
            let current = self.current_window_size.unwrap_or(self.win_max);
            if let Some(ws) = self.temporal_mixing.update_attention_window_from_pred_norm(
                current,
                self.win_min,
                self.win_max,
                self.win_step_up,
                self.win_step_down,
                self.pred_up,
                self.pred_down,
            ) {
                self.current_window_size = Some(ws);
            }
        }
        prediction
    }

    /// Capture a clone of the cached intermediates from the most recent forward pass
    #[allow(dead_code)]
    pub(crate) fn cache_snapshot(&self) -> Option<DiffusionCachedIntermediates> {
        self.cached_intermediates.read().unwrap().clone()
    }

    /// Restore cached intermediates so downstream gradient consumers can reuse them
    #[allow(dead_code)]
    pub(crate) fn restore_cache(&self, cache: DiffusionCachedIntermediates) {
        *self.cached_intermediates.write().unwrap() = Some(cache);
    }

    /// Sample from the reverse diffusion process (generative sampling)
    pub fn sample(&mut self, shape: (usize, usize), steps: Option<usize>) -> Array2<f32> {
        // Delegate to the sampler-aware implementation (DDPM/DDIM/PNDM/DPM-Solver++).
        // Note: for DDPM we always run the full discrete chain (the posterior is defined
        // for adjacent timesteps), so `steps` is only meaningful for reduced-step solvers.
        let guidance = self.config.guidance.clone();
        self.sample_with_guidance(shape, steps, guidance.as_ref(), None)
    }

    pub fn sample_ddim(&mut self, shape: (usize, usize), steps: Option<usize>) -> Array2<f32> {
        let total = self.noise_scheduler.num_timesteps().max(1);
        let k = steps.unwrap_or(total).max(1);
        let mut x_t = Array2::zeros(shape);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = get_rng();
        if let Some(slice) = x_t.as_slice_mut() {
            slice.par_iter_mut().for_each(|v| {
                *v = normal.sample(&mut get_rng()) as f32;
            });
        } else {
            for v in x_t.iter_mut() {
                *v = normal.sample(&mut rng) as f32;
            }
        }

        let timesteps =
            crate::domain::layers::diffusion::solvers::make_discrete_timesteps(k, total);
        for i in 0..(timesteps.len() - 1) {
            let t = timesteps[i];
            let t_prev = timesteps[i + 1];
            self.set_timestep(t);
            let pred = self.forward_with_timestep(&x_t, t);
            let eps_hat = crate::domain::layers::diffusion::solvers::epsilon_from_prediction_target(
                pred,
                &x_t,
                t,
                self.config.prediction_target.clone(),
                &self.noise_scheduler,
            );
            x_t = self
                .noise_scheduler
                .ddim_step_between(&x_t, t, t_prev, &eps_hat, 0.0, None);
        }
        x_t
    }

    pub fn set_noise_schedule(
        &mut self,
        schedule_type: NoiseSchedule,
        num_timesteps: Option<usize>,
    ) {
        let nt = num_timesteps.unwrap_or(self.config.num_timesteps);
        self.noise_scheduler = NoiseScheduler::new(schedule_type.clone(), nt);
        self.config.noise_schedule = schedule_type;
        self.config.num_timesteps = nt;
        if self.config.discrete_masked {
            self.discrete_scheduler =
                Some(crate::domain::layers::diffusion::discrete::DiscreteMaskScheduler::new(nt));
        }
    }

    pub fn prediction_target(&self) -> DiffusionPredictionTarget {
        self.config.prediction_target.clone()
    }

    pub fn timestep_strategy(&self) -> DiffusionTimestepStrategy {
        self.config.timestep_strategy
    }

    pub fn noise_schedule(&self) -> &NoiseSchedule {
        &self.config.noise_schedule
    }

    /// Speculative sampling using draft model to accelerate reverse diffusion.
    ///
    /// # Mathematical Invariant
    /// Unbiased sampling approximation: accept draft chain if first-step noise MSE < tau.
    /// Expected speedup ~ gamma / (1 + reject_rate), reject_rate ~ 0.5 empirically.
    ///
    /// Literature: Speculative Diffusion Sampling (arXiv)
    pub fn speculative_sample(
        &mut self,
        draft: &mut DiffusionBlock,
        shape: (usize, usize),
        steps: Option<usize>,
        config: &crate::domain::layers::transformer::speculative::SpeculativeSamplingConfig,
    ) -> Array2<f32> {
        let total = self.noise_scheduler.num_timesteps().max(1);
        let gamma = config.gamma;
        let tau = config.tau;

        let mut steps_left = steps.unwrap_or(total).max(gamma.max(1));
        let mut x_t = Array2::zeros(shape);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = get_rng();
        x_t.mapv_inplace(|_| normal.sample(&mut rng) as f32);
        let mut t = total.saturating_sub(1);
        while t > 0 && steps_left > 0 {
            if gamma == 0 || t < gamma {
                let pred = self.predict_epsilon_with_timestep(&x_t, t);
                x_t = self.noise_scheduler.ddim_step(&x_t, t, &pred, 0.0, None);
                t = t.saturating_sub(1);
                steps_left = steps_left.saturating_sub(1);
                continue;
            }

            let pred = self.predict_epsilon_with_timestep(&x_t, t);
            let draft_pred = draft.predict_epsilon_with_timestep(&x_t, t);
            let mse = pred
                .iter()
                .zip(draft_pred.iter())
                .map(|(a, b)| {
                    let diff = a - b;
                    diff * diff
                })
                .sum::<f32>()
                / pred.len().max(1) as f32;

            if mse > tau {
                // Reject draft proposal and advance baseline chain by one step.
                x_t = self.noise_scheduler.ddim_step(&x_t, t, &pred, 0.0, None);
                t = t.saturating_sub(1);
                steps_left = steps_left.saturating_sub(1);
                continue;
            }

            // Accept speculative proposal: reuse draft to leap gamma steps ahead.
            let mut x_draft = self
                .noise_scheduler
                .ddim_step(&x_t, t, &draft_pred, 0.0, None);
            let mut t_d = t.saturating_sub(1);
            let mut accepted = 1usize;
            for _ in 1..gamma {
                if t_d == 0 {
                    break;
                }
                let pred_d = draft.predict_epsilon_with_timestep(&x_draft, t_d);
                x_draft = self
                    .noise_scheduler
                    .ddim_step(&x_draft, t_d, &pred_d, 0.0, None);
                t_d = t_d.saturating_sub(1);
                accepted += 1;
            }

            x_t = x_draft;
            t = t_d;
            steps_left = steps_left.saturating_sub(accepted);
        }

        x_t
    }
}

impl DiffusionBlock {
    /// Apply Classifier-Free Guidance (CFG)
    ///
    /// unconditional_pred: Prediction from unconditional model (ε or v)
    /// conditional_pred: Prediction from conditional model (ε or v)
    /// guidance_scale: Scale factor (typically 1.0-10.0)
    ///
    /// Returns: Guided prediction ε_guided = unconditional + guidance_scale * (conditional -
    /// unconditional)
    pub fn apply_classifier_free_guidance(
        &self,
        unconditional_pred: &Array2<f32>,
        conditional_pred: &Array2<f32>,
        guidance_scale: f32,
    ) -> Array2<f32> {
        let mut guided = conditional_pred.clone();
        guided -= unconditional_pred;
        guided *= guidance_scale;
        guided += unconditional_pred;
        guided
    }

    /// Apply adaptive guidance with dynamic scale.
    pub fn apply_adaptive_guidance(
        &self,
        unconditional_pred: &Array2<f32>,
        conditional_pred: &Array2<f32>,
        t: usize,
    ) -> Array2<f32> {
        let t_normalized = t as f32 / self.config.num_timesteps as f32;
        let base_scale = self.config.min_guidance_scale
            + (self.config.max_guidance_scale - self.config.min_guidance_scale) * t_normalized;

        let diff = conditional_pred - unconditional_pred;
        let diff_norm = diff.mapv(|x| x.abs()).mean().unwrap_or(1.0);
        let adaptive_scale = base_scale / (1.0 + diff_norm).sqrt();
        unconditional_pred + adaptive_scale * diff
    }

    /// Enhanced sampling with guidance support.
    pub fn sample_with_guidance(
        &mut self,
        shape: (usize, usize),
        steps: Option<usize>,
        guidance_config: Option<&GuidanceConfig>,
        unconditional_input: Option<&Array2<f32>>,
    ) -> Array2<f32> {
        let total = self.noise_scheduler.num_timesteps().max(1);
        let steps = steps.unwrap_or(self.config.num_timesteps).max(1);
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Start with pure noise
        let mut x_t = Array2::from_shape_fn(shape, |_| normal.sample(&mut rng) as f32);

        match self.config.sampler {
            DiffusionSampler::DDPM => {
                // DDPM posterior updates are defined for consecutive steps, so we always run
                // the full discrete chain.
                for t in (1..total).rev() {
                    self.set_timestep(t);

                    let conditional_pred = self.predict_epsilon_with_timestep(&x_t, t);
                    let pred_epsilon = if let Some(guidance) = guidance_config {
                        if let Some(uncond_input) = unconditional_input {
                            let uncond_pred = self.predict_epsilon_with_timestep(uncond_input, t);
                            match guidance.guidance_type {
                                GuidanceType::Cfg | GuidanceType::CG => self
                                    .apply_classifier_free_guidance(
                                        &uncond_pred,
                                        &conditional_pred,
                                        guidance.scale,
                                    ),
                                GuidanceType::Adaptive => {
                                    self.apply_adaptive_guidance(&uncond_pred, &conditional_pred, t)
                                }
                            }
                        } else {
                            conditional_pred
                        }
                    } else {
                        conditional_pred
                    };

                    let noise =
                        Array2::from_shape_fn(x_t.raw_dim(), |_| normal.sample(&mut rng) as f32);
                    let sa = self.noise_scheduler.sqrt_alpha_cumprod(t).max(1e-6);
                    let soa = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                    let x0_hat = (&x_t - &(pred_epsilon * soa)) / sa;
                    x_t = self
                        .noise_scheduler
                        .posterior_sample(&x_t, &x0_hat, t, &noise);
                }
            }

            DiffusionSampler::DDIM { eta } => {
                let timesteps = crate::domain::layers::diffusion::solvers::make_discrete_timesteps(
                    steps, total,
                );
                for i in 0..(timesteps.len() - 1) {
                    let t = timesteps[i];
                    let t_prev = timesteps[i + 1];
                    self.set_timestep(t);

                    let conditional_pred = self.predict_epsilon_with_timestep(&x_t, t);
                    let pred_epsilon = if let Some(guidance) = guidance_config {
                        if let Some(uncond_input) = unconditional_input {
                            let uncond_pred = self.predict_epsilon_with_timestep(uncond_input, t);
                            match guidance.guidance_type {
                                GuidanceType::Cfg | GuidanceType::CG => self
                                    .apply_classifier_free_guidance(
                                        &uncond_pred,
                                        &conditional_pred,
                                        guidance.scale,
                                    ),
                                GuidanceType::Adaptive => {
                                    self.apply_adaptive_guidance(&uncond_pred, &conditional_pred, t)
                                }
                            }
                        } else {
                            conditional_pred
                        }
                    } else {
                        conditional_pred
                    };

                    let noise = if eta > 0.0 {
                        Some(Array2::from_shape_fn(x_t.raw_dim(), |_| {
                            normal.sample(&mut rng) as f32
                        }))
                    } else {
                        None
                    };

                    x_t = self.noise_scheduler.ddim_step_between(
                        &x_t,
                        t,
                        t_prev,
                        &pred_epsilon,
                        eta,
                        noise.as_ref(),
                    );
                }
            }

            DiffusionSampler::PNDM => {
                let timesteps = crate::domain::layers::diffusion::solvers::make_discrete_timesteps(
                    steps, total,
                );
                let scheduler = self.noise_scheduler.clone();
                let mut model_eps = |x: &Array2<f32>, t: usize| -> Array2<f32> {
                    self.set_timestep(t);
                    let conditional_pred = self.predict_epsilon_with_timestep(x, t);
                    if let Some(guidance) = guidance_config {
                        if let Some(uncond_input) = unconditional_input {
                            let uncond_pred = self.predict_epsilon_with_timestep(uncond_input, t);
                            match guidance.guidance_type {
                                GuidanceType::Cfg | GuidanceType::CG => self
                                    .apply_classifier_free_guidance(
                                        &uncond_pred,
                                        &conditional_pred,
                                        guidance.scale,
                                    ),
                                GuidanceType::Adaptive => {
                                    self.apply_adaptive_guidance(&uncond_pred, &conditional_pred, t)
                                }
                            }
                        } else {
                            conditional_pred
                        }
                    } else {
                        conditional_pred
                    }
                };

                x_t = crate::domain::layers::diffusion::solvers::pndm_plms_sample(
                    x_t,
                    &timesteps,
                    &scheduler,
                    &mut model_eps,
                );
            }

            DiffusionSampler::DPMSolver => {
                let scheduler = self.noise_scheduler.clone();
                let alpha_start = scheduler.sqrt_alpha_cumprod(total - 1).max(1e-12);
                let sigma_start = scheduler.sqrt_one_minus_alpha_cumprod(total - 1).max(1e-12);
                let alpha_end = scheduler.sqrt_alpha_cumprod(0).max(1e-12);
                let sigma_end = scheduler.sqrt_one_minus_alpha_cumprod(0).max(1e-12);
                let lambda_start = alpha_start.ln() - sigma_start.ln();
                let lambda_end = alpha_end.ln() - sigma_end.ln();
                let lambda_range = (lambda_end - lambda_start).abs().max(1e-3);

                let cfg = crate::domain::layers::diffusion::solvers::DpmSolverAdaptiveConfig {
                    h_init: (lambda_range / steps as f32).clamp(1e-4, 1.0),
                    ..Default::default()
                };

                let mut model_x0 = |x: &Array2<f32>, t: usize| -> Array2<f32> {
                    self.set_timestep(t);
                    let conditional_pred = self.predict_epsilon_with_timestep(x, t);
                    let eps = if let Some(guidance) = guidance_config {
                        if let Some(uncond_input) = unconditional_input {
                            let uncond_pred = self.predict_epsilon_with_timestep(uncond_input, t);
                            match guidance.guidance_type {
                                GuidanceType::Cfg | GuidanceType::CG => self
                                    .apply_classifier_free_guidance(
                                        &uncond_pred,
                                        &conditional_pred,
                                        guidance.scale,
                                    ),
                                GuidanceType::Adaptive => {
                                    self.apply_adaptive_guidance(&uncond_pred, &conditional_pred, t)
                                }
                            }
                        } else {
                            conditional_pred
                        }
                    } else {
                        conditional_pred
                    };

                    // Convert eps -> x0 at this discrete timestep.
                    crate::domain::layers::diffusion::solvers::x0_from_prediction_target(
                        eps,
                        x,
                        t,
                        DiffusionPredictionTarget::Epsilon,
                        &scheduler,
                    )
                };

                x_t = crate::domain::layers::diffusion::solvers::dpmsolverpp_adaptive_sample(
                    x_t,
                    &scheduler,
                    &mut model_x0,
                    cfg,
                );
            }
        }

        x_t
    }

    /// Enhanced loss calculation with P2/SNR weighting.
    pub fn compute_weighted_loss(
        &self,
        pred: &Array2<f32>,
        target: &Array2<f32>,
        t: usize,
    ) -> (Array2<f32>, f32) {
        let diff = pred - target;

        let weight = if self.config.use_p2_weighting {
            self.noise_scheduler.p2_weight(t)
        } else if self.config.use_snr_weighting {
            self.noise_scheduler.snr_weight(t)
        } else {
            match self.config.loss_weighting {
                LossWeighting::Uniform => 1.0,
                LossWeighting::P2 => self.noise_scheduler.p2_weight(t),
                LossWeighting::Snr => self.noise_scheduler.snr_weight(t),
                LossWeighting::Adaptive => {
                    let p2_w = self.noise_scheduler.p2_weight(t);
                    let snr_w = self.noise_scheduler.snr_weight(t);
                    self.noise_scheduler.adaptive_weight(t, p2_w, snr_w)
                }
            }
        };

        let weighted_diff = diff.mapv(|x| x * weight.sqrt());
        let weighted_loss = weighted_diff.mapv(|x| x * x).mean().unwrap_or(0.0);
        (weighted_diff, weighted_loss)
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

    fn set_training_progress(&mut self, progress: f64) {
        self.temporal_mixing.set_training_progress(progress);
    }

    #[allow(dead_code)]
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.pre_attention_norm.parameters()
            + self.temporal_mixing.parameters()
            + self.pre_ffn_norm.parameters()
            + self.feedforward.parameters()
            + 5
            + self
                .adaptive_residuals
                .as_ref()
                .map(|r| r.parameter_count())
                .unwrap_or(0)
    }

    fn weight_norm(&self) -> f32 {
        let residual_norm = self
            .adaptive_residuals
            .as_ref()
            .map(|r| r.weight_norm())
            .unwrap_or(0.0);
        (self.pre_attention_norm.weight_norm().powi(2)
            + self.temporal_mixing.weight_norm().powi(2)
            + self.pre_ffn_norm.weight_norm().powi(2)
            + self.feedforward.weight_norm().powi(2)
            + self.time_conditioner.weight_norm().powi(2)
            + residual_norm.powi(2))
        .sqrt()
    }

    /// Compute analytical gradients using cached forward intermediates
    /// Ensures full-gradient propagation across residual connections
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Validate input gradients before processing
        if !output_grads.iter().all(|&x| x.is_finite()) {
            tracing::error!("Non-finite gradients passed to DiffusionBlock::compute_gradients");
            return (Array2::zeros(output_grads.raw_dim()), Vec::new());
        }
        let cache_guard = self.cached_intermediates.read().unwrap();
        if let Some(cache) = &*cache_guard {
            let input_original: &Array2<f32> = cache.input_original.as_ref();
            let input_used: &Array2<f32> = cache.input_used.as_ref();
            let time_embed: &Array1<f32> = cache.time_embed.as_ref();
            let norm1_out: &Array2<f32> = cache.norm1_out.as_ref();
            let norm1_mod: &Array2<f32> = cache.norm1_mod.as_ref();
            let residual1: &Array2<f32> = cache.residual1.as_ref();
            let norm2_out: &Array2<f32> = cache.norm2_out.as_ref();
            let norm2_mod: &Array2<f32> = cache.norm2_mod.as_ref();
            let attn_out: &Array2<f32> = cache.attn_out.as_ref();
            let ffn_out: &Array2<f32> = cache.ffn_out.as_ref();
            let h_vec: &Array1<f32> = cache.h_vec.as_ref();
            let gamma_attn_vec: &Array2<f32> = cache.gamma_attn.as_ref();
            let beta_attn_vec: &Array2<f32> = cache.beta_attn.as_ref();
            let gamma_ffn_vec: &Array2<f32> = cache.gamma_ffn.as_ref();
            let beta_ffn_vec: &Array2<f32> = cache.beta_ffn.as_ref();
            let timestep = cache.timestep;
            let mut all_param_grads = Vec::new();

            let (block_grads_scale, input_extra_scale) = if self.config.prediction_target
                == DiffusionPredictionTarget::VPrediction
            {
                let sqrt_alpha_bar = self.noise_scheduler.sqrt_alpha_cumprod(timestep).max(1e-6);
                let sqrt_one_minus_alpha_bar = self
                    .noise_scheduler
                    .sqrt_one_minus_alpha_cumprod(timestep)
                    .max(1e-6);
                // Clamp to prevent extreme gradient scaling that can cause NaN
                let scale = sqrt_alpha_bar.clamp(1e-3, 1.0);
                (scale, Some(sqrt_one_minus_alpha_bar.clamp(1e-3, 1.0)))
            } else {
                (1.0f32, None)
            };

            // If the forward returned EDM x0_hat, map upstream grads back to the internal
            // residual-stack output via the preconditioning coefficients.
            let (scaled_output_grads, edm_skip_grad, edm_c_in) =
                if self.config.prediction_target == DiffusionPredictionTarget::EdmX0 {
                    let alpha_bar = self
                        .noise_scheduler
                        .sqrt_alpha_cumprod(timestep)
                        .powi(2)
                        .clamp(1e-12, 1.0);
                    let sigma = (((1.0 - alpha_bar) / alpha_bar).max(0.0)).sqrt();
                    let sigma_data = self.config.edm_sigma_data.max(1e-6);
                    let denom = (sigma * sigma + sigma_data * sigma_data).max(1e-12);
                    let c_in = 1.0 / denom.sqrt();
                    let c_skip = (sigma_data * sigma_data) / denom;
                    let c_out = (sigma * sigma_data) / denom.sqrt();
                    (
                        output_grads * (block_grads_scale * c_out),
                        Some(output_grads * (block_grads_scale * c_skip)),
                        Some(c_in),
                    )
                } else {
                    (output_grads * block_grads_scale, None, None)
                };
            // Sanitize after scaling to catch any NaN from the scaling operation
            let mut safe_scaled_grads = scaled_output_grads;
            Self::sanitize_tensor("scaled_output_grads", &mut safe_scaled_grads);

            // Compute gradients through the transformer block layers
            // This follows the same pattern as TransformerBlock but with timestep conditioning

            // Output = residual1 + ffn_out, so gradients split between residual1 and ffn_out.
            // Both branches receive the same upstream grads; avoid cloning.

            // Get feedforward gradients
            let (ffn_input_grad_mod, ffn_param_grads) =
                self.feedforward.backward(norm2_mod, &safe_scaled_grads);

            let (norm2_grad, grad_gamma_ffn_1d, grad_beta_ffn_1d) = self
                .film_modulation
                .film_backward(&ffn_input_grad_mod, norm2_out, gamma_ffn_vec);

            let (residual1_from_ffn, pre_ffn_param_grads) =
                self.pre_ffn_norm.compute_gradients(residual1, &norm2_grad);

            // Combine residual gradients
            let residual1_total_grads = &safe_scaled_grads + &residual1_from_ffn;

            // residual1 = input + attn_out: propagate full upstream gradient to both branches
            let attn_out_grads = &residual1_total_grads;

            let (mut attn_input_grad_mod, attn_param_grads) = self
                .temporal_mixing
                .compute_gradients(norm1_mod, attn_out_grads);
            if self.temporal_mixing.needs_external_titan_memory() {
                self.config
                    .titan_memory
                    .add_input_grads_from_output_grads_into(
                        attn_out_grads,
                        &mut attn_input_grad_mod,
                    );
            }

            let (norm1_grad, grad_gamma_attn, grad_beta_attn) =
                self.film_modulation
                    .film_backward(&attn_input_grad_mod, norm1_out, gamma_attn_vec);

            let (input_from_norm, pre_attn_param_grads) = self
                .pre_attention_norm
                .compute_gradients(input_used, &norm1_grad);

            // Gradients w.r.t. the mixed input used by this block: dX'.
            let final_input_used_grads = &residual1_total_grads + &input_from_norm;

            let mut similarity_strength_grad = Array2::zeros((1, 1));
            if let Some(ctx) = self.context.incoming_context.as_ref()
                && ctx.nrows() == self.config.embed_dim
                && ctx.ncols() == self.config.embed_dim
            {
                let d = (self.config.embed_dim.max(1)) as f32;
                let mixed = input_original.dot(ctx);
                let mut acc = 0.0f64;
                for (&g, &m) in final_input_used_grads.iter().zip(mixed.iter()) {
                    let gs: f64 = if g.is_finite() { g as f64 } else { 0.0 };
                    let ms: f64 = if m.is_finite() { m as f64 } else { 0.0 };
                    acc += gs * ms;
                }
                similarity_strength_grad[[0, 0]] = (acc as f32) / d;
            }

            let mut final_input_grads = final_input_used_grads;
            if let Some(ctx) = self.context.incoming_context.as_ref()
                && ctx.nrows() == self.config.embed_dim
                && ctx.ncols() == self.config.embed_dim
            {
                let d = (self.config.embed_dim.max(1)) as f32;
                let s = self.context.similarity_context_strength[[0, 0]];
                let s = if s.is_finite() { s } else { 0.0 };
                let k = s / d;
                if k != 0.0 {
                    let corr = final_input_grads.dot(&ctx.t());
                    final_input_grads.zip_mut_with(&corr, |g, &c| {
                        let cs = if c.is_finite() { c } else { 0.0 };
                        *g += k * cs;
                    });
                }
            }

            if let Some(extra_scale) = input_extra_scale {
                final_input_grads += &(output_grads * extra_scale);
            }

            // EDM: x_model_in = c_in * x_t, plus a skip path x0_hat includes c_skip * x_t.
            if let Some(c_in) = edm_c_in {
                final_input_grads *= c_in;
            }
            if let Some(skip) = edm_skip_grad {
                final_input_grads += &skip;
            }

            let attn_grad_count = attn_param_grads.len();
            let ffn_grad_count = ffn_param_grads.len();
            let pre_ffn_grad_count = pre_ffn_param_grads.len();
            let pre_attn_grad_count = pre_attn_param_grads.len();

            all_param_grads.extend(attn_param_grads);
            all_param_grads.extend(ffn_param_grads);
            all_param_grads.extend(pre_ffn_param_grads);
            all_param_grads.extend(pre_attn_param_grads);
            all_param_grads.push(similarity_strength_grad);

            let embed = self.config.embed_dim;

            // Calculate backprop for FiLM MLP using SharedFilmModulation helper
            let grad_gamma_beta_1d = self.film_modulation.compute_mlp_gradients(
                &grad_gamma_attn,
                &grad_beta_attn,
                &grad_gamma_ffn_1d,
                &grad_beta_ffn_1d,
                gamma_attn_vec,
                beta_attn_vec,
                gamma_ffn_vec,
                beta_ffn_vec,
                embed,
            );

            let h_mat = h_vec.view().to_shape((1, h_vec.len())).unwrap().to_owned();
            let (_, time_grads) =
                self.time_conditioner
                    .backward(&grad_gamma_beta_1d, &h_mat, time_embed);
            all_param_grads.extend(time_grads);

            let adaptive_param_grads = if let Some(residuals) = self.adaptive_residuals.as_ref() {
                residuals.compute_gradients(
                    input_used,
                    attn_out,
                    &residual1_total_grads,
                    ffn_out,
                    &safe_scaled_grads,
                )
            } else {
                Vec::new()
            };
            let adaptive_grad_count = adaptive_param_grads.len();
            all_param_grads.extend(adaptive_param_grads);

            let partitions = DiffusionParamPartitions {
                temporal_mixing: attn_grad_count,
                feedforward: ffn_grad_count,
                pre_ffn_norm: pre_ffn_grad_count,
                pre_attention_norm: pre_attn_grad_count,
                similarity_context_strength: 1,
                time_conditioner: 4,
                time_embedding: 0,
                adaptive_residual_similarity: 0,
                adaptive_residual_affinity: 0,
                adaptive_residual_attention: 0,
                adaptive_residual_channel: 0,
                adaptive_residual_scales_attention: adaptive_grad_count.min(1),
                adaptive_residual_scales_ffn: adaptive_grad_count.saturating_sub(1).min(1),
                // Theorem 4 extension partitions
                adaptive_residual_positional_qkv: 0,
                adaptive_residual_positional_cope: 0,
                adaptive_residual_positional_weights: 0,
            };
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = Some(partitions);
            }

            (final_input_grads, all_param_grads)
        } else {
            tracing::warn!(
                "DiffusionBlock::compute_gradients called without cached intermediates. Call forward() first."
            );
            if let Ok(mut guard) = self.param_partitions.write() {
                *guard = None;
            }
            (output_grads.clone(), Vec::new())
        }
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if param_grads.is_empty() {
            return Ok(());
        }

        let mut router = GradientRouter::new(param_grads, 5.0);

        let cached_partitions = self
            .param_partitions
            .read()
            .map(|guard| guard.clone())
            .unwrap_or(None);
        let partitions =
            cached_partitions.ok_or_else(|| crate::common::errors::ModelError::GradientError {
                message: "DiffusionBlock::apply_gradients missing partition metadata".to_string(),
            })?;

        let expected = partitions.total();
        if expected != router.total() {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "DiffusionBlock::apply_gradients gradient count mismatch: expected {}, got {}",
                    expected,
                    router.total()
                ),
            });
        }

        let temporal_weight_norm = self.temporal_mixing.weight_norm();
        router.route_lars_to_closure(
            partitions.temporal_mixing,
            temporal_weight_norm,
            lr,
            |grads, lr| self.temporal_mixing.apply_gradients(grads, lr),
        )?;

        let feedforward_weight_norm = self.feedforward.weight_norm();
        router.route_lars_to_closure(
            partitions.feedforward,
            feedforward_weight_norm,
            lr,
            |grads, lr| self.feedforward.apply_gradients(grads, lr),
        )?;

        router.route_to_closure(partitions.pre_ffn_norm, |grads| {
            self.pre_ffn_norm.apply_gradients_ref(grads, lr)
        })?;

        router.route_to_closure(partitions.pre_attention_norm, |grads| {
            self.pre_attention_norm.apply_gradients_ref(grads, lr)
        })?;

        router.route_exact_ref_to_closure::<1, _>(
            partitions.similarity_context_strength,
            |grads| {
                self.opt_similarity_context_strength.step(
                    &mut self.context.similarity_context_strength,
                    grads[0].as_ref(),
                    lr,
                );
                Ok(())
            },
        )?;

        router.route_exact_ref_to_closure::<4, _>(partitions.time_conditioner, |grads| {
            self.time_conditioner
                .apply_gradients(grads.as_slice(), lr, self.ema_decay);
            Ok(())
        })?;

        let adaptive_count =
            partitions.adaptive_residual_scales_attention + partitions.adaptive_residual_scales_ffn;
        let has_adaptive_residuals = self.adaptive_residuals.is_some();
        router.route_exact_ref_to_closure_if::<2, _>(
            adaptive_count,
            has_adaptive_residuals,
            |grads| {
                if let Some(ref mut residuals) = self.adaptive_residuals {
                    residuals.apply_gradients_ref((grads[0].as_ref(), grads[1].as_ref()), lr)?;
                }
                Ok(())
            },
        )?;

        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = None;
        }
        Ok(())
    }

    fn zero_gradients(&mut self) {
        // DiffusionBlock doesn't maintain internal gradient state beyond cached intermediates
        // Reset cached intermediates and partitions to free memory
        if let Ok(mut guard) = self.cached_intermediates.write() {
            *guard = None;
        }
        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = None;
        }
    }
}

impl From<TransformerBlockConfig> for DiffusionBlockConfig {
    fn from(t: TransformerBlockConfig) -> Self {
        Self {
            embed_dim: t.embed_dim,
            hidden_dim: t.hidden_dim,
            num_heads: t.num_heads,
            num_timesteps: 1000,
            noise_schedule: NoiseSchedule::default(),
            prediction_target: DiffusionPredictionTarget::default(),
            edm_sigma_data: edm::EDM_SIGMA_DATA_DEFAULT,
            timestep_strategy: DiffusionTimestepStrategy::Uniform,
            causal_attention: false,
            window_size: t.window_size,
            use_adaptive_window: t.use_adaptive_window,
            discrete_masked: false,
            poly_degree: t.poly_degree,
            max_pos: t.max_pos,
            use_moe: t.use_moe,
            moe_config: t.moe_config,
            head_selection: t.head_selection,
            moh_threshold_modulation: t.moh_threshold_modulation,
            titan_memory: t.titan_memory,
            time_embed_dim: t.embed_dim * 4,
            mask_token_id: None,
            temporal_mixing: t.temporal_mixing,
            use_advanced_adaptive_residuals: t.use_advanced_adaptive_residuals,
            sampler: DiffusionSampler::default(),
            guidance: None,
            loss_weighting: LossWeighting::default(),
            use_p2_weighting: false,
            use_snr_weighting: false,
            adaptive_guidance: false,
            min_guidance_scale: default_min_guidance(),
            max_guidance_scale: default_max_guidance(),
            ddim_steps_policy: Default::default(),
        }
    }
}

// Unified workspace management for memory optimization
impl crate::domain::layers::components::workspace_managed::WorkspaceManaged for DiffusionBlock {
    /// Ensure workspace buffers are allocated with correct capacity.
    ///
    /// This consolidates the diffusion-specific buffer allocation patterns (time embeddings,
    /// FiLM parameters, input/output caches) into a single unified interface.
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        self.unified_workspace
            .ensure_capacity(batch_size, seq_len, embed_dim);
    }

    /// Clear all workspace buffers to free memory.
    fn clear_workspace(&mut self) {
        self.unified_workspace.clear_workspace();
    }

    /// Get memory statistics for all managed buffers.
    fn workspace_stats(
        &self,
    ) -> crate::domain::layers::components::workspace_managed::WorkspaceStats {
        self.unified_workspace.workspace_stats()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::Array1;

    use super::*;

    #[test]
    fn test_scheduler_handles_single_timestep() {
        let sched = NoiseScheduler::new(
            NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            1,
        );
        assert_eq!(sched.num_timesteps(), 1);
        assert!(sched.beta(0).is_finite());
        assert_relative_eq!(sched.sqrt_alpha_cumprod(0), 1.0, epsilon = 1e-6);
        assert_relative_eq!(sched.sqrt_one_minus_alpha_cumprod(0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_q_sample_t0_is_identity() {
        let sched = NoiseScheduler::new(
            NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            8,
        );
        let x0 = Array2::from_elem((2, 3), 0.25);
        let noise = Array2::from_elem((2, 3), -0.75);
        let xt = sched.q_sample(&x0, 0, &noise);
        for (a, b) in xt.iter().zip(x0.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_posterior_sample_t0_returns_x0() {
        let sched = NoiseScheduler::new(
            NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            8,
        );
        let x0 = Array2::from_elem((2, 3), 1.25);
        let xt = x0.clone();
        let noise = Array2::from_elem((2, 3), 0.5);
        let x_prev = sched.posterior_sample(&xt, &x0, 0, &noise);
        for (a, b) in x_prev.iter().zip(x0.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_karras_schedule_produces_reasonable_betas() {
        let sched = NoiseScheduler::new(
            NoiseSchedule::Karras {
                sigma_min: 0.002,
                sigma_max: 10.0,
                rho: 7.0,
            },
            64,
        );
        for t in 0..sched.num_timesteps() {
            let b = sched.beta(t);
            assert!(b.is_finite());
            assert!(b > 0.0 && b < 1.0);
        }
        // alpha_bar should be non-increasing.
        let mut prev = 1.0f32;
        for t in 0..sched.num_timesteps() {
            let ab = sched.sqrt_alpha_cumprod(t).powi(2);
            assert!(ab <= prev + 1e-6);
            prev = ab;
        }
    }

    #[test]
    fn test_linear_and_cosine_schedules_produce_monotone_alpha_bar() {
        let schedules = [
            NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            NoiseSchedule::Cosine { s: 0.008 },
            NoiseSchedule::Quadratic {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
        ];

        for schedule in schedules {
            let sched = NoiseScheduler::new(schedule, 128);
            let mut prev = 1.0f32;
            for t in 0..sched.num_timesteps() {
                let ab = sched.sqrt_alpha_cumprod(t).powi(2);
                assert!(ab.is_finite());
                assert!(ab > 0.0 && ab <= 1.0);
                assert!(ab <= prev + 1e-6);
                prev = ab;
            }
        }
    }

    #[test]
    fn test_q_sample_matches_closed_form() {
        let sched = NoiseScheduler::new(
            NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            32,
        );
        let x0 = Array2::from_elem((2, 3), 2.0);
        let noise = Array2::from_elem((2, 3), -1.0);
        let t = 7;
        let xt = sched.q_sample(&x0, t, &noise);
        let sa = sched.sqrt_alpha_cumprod(t);
        let soa = sched.sqrt_one_minus_alpha_cumprod(t);
        let expected = (&x0 * sa) + (&noise * soa);
        for (a, b) in xt.iter().zip(expected.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-6);
        }
    }

    fn make_test_diffusion_block(prediction_target: DiffusionPredictionTarget) -> DiffusionBlock {
        let config = DiffusionBlockConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_timesteps: 64,
            noise_schedule: NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            prediction_target,
            timestep_strategy: DiffusionTimestepStrategy::Uniform,
            causal_attention: false,
            window_size: None,
            use_adaptive_window: false,
            discrete_masked: false,
            poly_degree: 1,
            max_pos: 32,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: TitanMemoryConfig::default(),
            time_embed_dim: 8,
            mask_token_id: None,
            temporal_mixing: TemporalMixingType::Attention,
            use_advanced_adaptive_residuals: false,
            edm_sigma_data: edm::EDM_SIGMA_DATA_DEFAULT,
            sampler: DiffusionSampler::DDPM,
            guidance: None,
            loss_weighting: LossWeighting::default(),
            use_p2_weighting: false,
            use_snr_weighting: false,
            adaptive_guidance: false,
            min_guidance_scale: 1.0,
            max_guidance_scale: 10.0,
            ddim_steps_policy: Default::default(),
        };
        DiffusionBlock::new(config)
    }

    #[test]
    fn test_min_snr_weight_bounds_across_targets() {
        let gamma = 3.0;
        let t = 10;

        let block_eps = make_test_diffusion_block(DiffusionPredictionTarget::Epsilon);
        let w_eps = block_eps.min_snr_weight(t, gamma);
        assert!(w_eps.is_finite());
        assert!(w_eps > 0.0 && w_eps <= 1.0 + 1e-6);

        let block_v = make_test_diffusion_block(DiffusionPredictionTarget::VPrediction);
        let w_v = block_v.min_snr_weight(t, gamma);
        assert!(w_v.is_finite());
        assert!(w_v > 0.0 && w_v < 1.0);

        let block_x0 = make_test_diffusion_block(DiffusionPredictionTarget::Sample);
        let w_x0 = block_x0.min_snr_weight(t, gamma);
        assert!(w_x0.is_finite());
        assert!(w_x0 > 0.0 && w_x0 <= gamma + 1e-6);
    }

    #[test]
    fn test_edm_loss_weight_is_finite_and_positive() {
        let block = make_test_diffusion_block(DiffusionPredictionTarget::EdmX0);
        for t in 0..block.noise_scheduler.num_timesteps() {
            let w = block.edm_loss_weight(t);
            assert!(w.is_finite());
            assert!(w > 0.0);
        }
    }

    #[test]
    fn test_time_conditioner_shapes() {
        let input_dim = 16;
        let hidden_dim = 32;
        let output_dim = 64;
        let conditioner = TimeConditioner::new(input_dim, hidden_dim, output_dim);

        let input = Array1::zeros(input_dim);
        let (output, hidden) = conditioner.forward(&input, false);

        assert_eq!(output.shape(), &[output_dim]);
        assert_eq!(hidden.shape(), &[1, hidden_dim]);
    }

    #[test]
    fn test_adaptive_residuals_diffusion_creation() {
        let embed_dim = 64;
        let num_timesteps = 1000;

        let mut residuals = AdaptiveResiduals::new_minimal(embed_dim);
        residuals.max_seq_len = num_timesteps.min(2048);

        assert_eq!(residuals.activation_similarity_diag.shape(), [embed_dim, 1]);
        assert_eq!(
            residuals.activation_similarity_off_abs_mean.shape(),
            [embed_dim, 1]
        );
        assert_eq!(residuals.max_seq_len, num_timesteps);
    }

    #[test]
    fn test_adaptive_residuals_forward() {
        let embed_dim = 32;
        let seq_len = 8;
        let num_timesteps = 100;

        let mut residuals = AdaptiveResiduals::new_minimal(embed_dim);
        residuals.max_seq_len = num_timesteps.min(2048);

        let input = Array2::from_elem((seq_len, embed_dim), 1.0);
        let attn_out = Array2::from_elem((seq_len, embed_dim), 0.5);

        let result = residuals.apply_attention_residual(&input, &attn_out);

        assert_eq!(result.shape(), [seq_len, embed_dim]);

        // Should produce reasonable residual values
        let mean_result = result.mean().unwrap_or(0.0);
        assert!(mean_result > 1.0); // Should be greater than input due to residual addition
        assert!(mean_result < 5.0); // Should be reasonable (not exploding)
    }

    #[test]
    fn test_diffusion_adaptive_residual() {
        let embed_dim = 16;
        let seq_len = 4;
        let num_timesteps = 100;

        let mut residuals = AdaptiveResiduals::new_minimal(embed_dim);
        residuals.max_seq_len = num_timesteps.min(2048);

        let input = Array2::from_elem((seq_len, embed_dim), 0.1);
        let attn_out = Array2::from_elem((seq_len, embed_dim), 0.2);

        // Test with different effective timestep scaling (residual implementation is shared)
        let early_scale = 1.0 + (10.0 / num_timesteps as f32) * 0.5;
        let late_scale = 1.0 + (80.0 / num_timesteps as f32) * 0.5;
        let attn_early = attn_out.mapv(|v| v * early_scale);
        let attn_late = attn_out.mapv(|v| v * late_scale);
        let mut residuals_early = residuals.clone();
        let mut residuals_late = residuals.clone();
        let result_early = residuals_early.apply_attention_residual(&input, &attn_early);
        let result_late = residuals_late.apply_attention_residual(&input, &attn_late);

        assert_eq!(result_early.shape(), [seq_len, embed_dim]);
        assert_eq!(result_late.shape(), [seq_len, embed_dim]);

        // Both should produce finite, reasonable values
        assert!(result_early.iter().all(|x: &f32| x.is_finite()));
        assert!(result_late.iter().all(|x: &f32| x.is_finite()));
    }

    #[test]
    fn test_snr_weighted_residuals() {
        let embed_dim = 8;
        let seq_len = 2;
        let num_timesteps = 100;

        let mut residuals = AdaptiveResiduals::new_minimal(embed_dim);
        residuals.max_seq_len = num_timesteps.min(2048);

        let input = Array2::from_elem((seq_len, embed_dim), 1.0);
        let attn_out = Array2::from_elem((seq_len, embed_dim), 0.5);

        // Test with different SNR weights by scaling the attention contribution
        let attn_low = attn_out.mapv(|v| v * 0.5);
        let attn_high = attn_out.mapv(|v| v * 2.0);
        let mut residuals_low = residuals.clone();
        let mut residuals_high = residuals.clone();
        let result_low_snr = residuals_low.apply_attention_residual(&input, &attn_low);
        let result_high_snr = residuals_high.apply_attention_residual(&input, &attn_high);

        assert_eq!(result_low_snr.shape(), [seq_len, embed_dim]);
        assert_eq!(result_high_snr.shape(), [seq_len, embed_dim]);

        // High SNR should amplify residuals more than low SNR
        let mean_low = result_low_snr.mean().unwrap_or(0.0);
        let mean_high = result_high_snr.mean().unwrap_or(0.0);
        assert!(
            mean_high > mean_low,
            "High SNR should produce stronger residuals"
        );
    }

    #[test]
    fn test_residual_parameter_count() {
        let embed_dim = 16;
        let residuals = AdaptiveResiduals::new_minimal(embed_dim);

        let param_count = residuals.parameter_count();
        let expected = 2 * embed_dim;
        assert_eq!(param_count, expected);
    }

    #[test]
    fn test_ddpm_sampling_produces_finite_output() {
        let config = DiffusionBlockConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            num_timesteps: 8,
            noise_schedule: NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            prediction_target: DiffusionPredictionTarget::Epsilon,
            timestep_strategy: DiffusionTimestepStrategy::Uniform,
            causal_attention: false,
            window_size: None,
            use_adaptive_window: false,
            discrete_masked: false,
            poly_degree: 1,
            max_pos: 32,
            use_moe: false,
            moe_config: None,
            head_selection: HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: TitanMemoryConfig::default(),
            time_embed_dim: 8,
            mask_token_id: None,
            temporal_mixing: TemporalMixingType::Attention,
            use_advanced_adaptive_residuals: false,
            edm_sigma_data: edm::EDM_SIGMA_DATA_DEFAULT,
            sampler: DiffusionSampler::DDPM,
            guidance: None,
            loss_weighting: LossWeighting::default(),
            use_p2_weighting: false,
            use_snr_weighting: false,
            adaptive_guidance: false,
            min_guidance_scale: 1.0,
            max_guidance_scale: 10.0,
            ddim_steps_policy: Default::default(),
        };
        let mut block = DiffusionBlock::new(config);
        let out = block.sample_with_guidance((4, 8), None, None, None);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_diffusion_block_set_compute_backend_checked_cpu() {
        let mut block = make_test_diffusion_block(DiffusionPredictionTarget::Epsilon);

        block
            .set_compute_backend_checked(crate::domain::compute_backend::ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted");

        assert_eq!(
            block.compute_backend(),
            crate::domain::compute_backend::ComputeBackend::Cpu
        );
    }
}

/// Sampling method for diffusion models
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub enum DiffusionSampler {
    /// Original DDPM sampling (stochastic)
    #[default]
    DDPM,
    /// DDIM sampling (deterministic when eta=0, stochastic when eta>0)
    DDIM { eta: f32 },
    /// PNDM sampling (pseudo numerical methods)
    PNDM,
    /// DPM-Solver (fast ODE solver)
    DPMSolver,
}

/// Guidance method for diffusion models
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GuidanceConfig {
    /// Guidance scale (typically 1.0-10.0)
    pub scale: f32,
    /// Guidance type
    pub guidance_type: GuidanceType,
}

/// Type of guidance to apply
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub enum GuidanceType {
    /// Classifier-Free Guidance (CFG)
    #[serde(rename = "CFG")]
    #[default]
    Cfg,
    /// Classifier Guidance (CG)
    CG,
    /// Adaptive Guidance
    Adaptive,
}

/// Loss weighting strategy
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Default)]
pub enum LossWeighting {
    /// Uniform weighting (original)
    #[default]
    Uniform,
    /// P2 weighting from Nichol & Dhariwal 2021
    P2,
    /// SNR weighting (signal-to-noise ratio)
    #[serde(rename = "SNR")]
    Snr,
    /// Adaptive weighting
    Adaptive,
}

impl GuidanceConfig {
    pub fn new_cfg(scale: f32) -> Self {
        Self {
            scale,
            guidance_type: GuidanceType::Cfg,
        }
    }

    pub fn new_adaptive(scale: f32) -> Self {
        Self {
            scale,
            guidance_type: GuidanceType::Adaptive,
        }
    }
}
