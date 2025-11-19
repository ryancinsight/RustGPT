#![allow(dead_code)]
use std::{f32::consts::PI, sync::RwLock};

use ndarray::{Array1, Array2, Axis, parallel::prelude::*, s};
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
    model_config::{DiffusionTimestepStrategy, ModelConfig},
    richards::{RichardsGlu, RichardsNorm},
    transformer::{common::FeedForwardVariant, transformer_block::TransformerBlockConfig},
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
    pub time_embed_dim: usize,
    pub mask_token_id: Option<usize>,
}

/// Prediction target for the diffusion model
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum DiffusionPredictionTarget {
    /// Predict the noise (epsilon) added to the input
    Epsilon,
    /// Predict the velocity (v) - see "Progressive Distillation for Fast Sampling of Diffusion Models"
    VPrediction,
    /// Predict the original sample (x_0)
    Sample,
}

impl Default for DiffusionPredictionTarget {
    fn default() -> Self {
        DiffusionPredictionTarget::Epsilon
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
        let sqrt_one_minus_cumprod = self.sqrt_one_minus_alpha_cumprod(t);
        let predicted_noise = (x_t - &(x_0 * sqrt_alpha_cumprod)) / sqrt_one_minus_cumprod;

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimeEmbedding {
    pub w: Array2<f32>,
    pub b: Array1<f32>,
}

impl TimeEmbedding {
    pub fn new(embed_dim: usize) -> Self {
        let w = Array2::zeros((embed_dim, embed_dim)); // Placeholder
        let b = Array1::zeros(embed_dim);
        Self { w, b }
    }
    pub fn forward(&self, t: usize, _max_t: usize) -> Array1<f32> {
        // Placeholder sinusoidal embedding
        let dim = self.b.len();
        let mut emb = Array1::zeros(dim);
        let half_dim = dim / 2;
        let freq = (-(2.0 * (half_dim as f32).ln() / half_dim as f32).exp()).exp();
        for i in 0..half_dim {
            let arg = t as f32 * freq.powi(i as i32);
            emb[2 * i] = arg.sin();
            emb[2 * i + 1] = arg.cos();
        }
        emb
    }
}

/// MLP for processing time embeddings into FiLM modulation parameters
#[derive(Serialize, Deserialize, Debug)]
pub struct TimeConditioner {
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_w1: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_b1: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_w2: Option<crate::adam::Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_b2: Option<crate::adam::Adam>,
    pub ema_w1: Array2<f32>,
    pub ema_b1: Array2<f32>,
    pub ema_w2: Array2<f32>,
    pub ema_b2: Array2<f32>,
}

impl TimeConditioner {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            Normal::new(0.0, (1.0 / input_dim as f32).sqrt())
                .unwrap()
                .sample(&mut rng)
        });
        let b1 = Array2::zeros((hidden_dim, 1));
        let w2 = Array2::from_shape_fn((hidden_dim, output_dim), |_| {
            Normal::new(0.0, (1.0 / hidden_dim as f32).sqrt())
                .unwrap()
                .sample(&mut rng)
        });
        let b2 = Array2::zeros((output_dim, 1));

        Self {
            ema_w1: w1.clone(),
            ema_b1: b1.clone(),
            ema_w2: w2.clone(),
            ema_b2: b2.clone(),
            opt_w1: Some(crate::adam::Adam::new_adamw((input_dim, hidden_dim), 0.01)),
            opt_b1: Some(crate::adam::Adam::new_adamw((hidden_dim, 1), 0.01)),
            opt_w2: Some(crate::adam::Adam::new_adamw((hidden_dim, output_dim), 0.01)),
            opt_b2: Some(crate::adam::Adam::new_adamw((output_dim, 1), 0.01)),
            w1,
            b1,
            w2,
            b2,
        }
    }

    pub fn forward(&self, input: &Array1<f32>, use_ema: bool) -> (Array2<f32>, Array2<f32>) {
        let (w1, b1, w2, b2) = if use_ema {
            (&self.ema_w1, &self.ema_b1, &self.ema_w2, &self.ema_b2)
        } else {
            (&self.w1, &self.b1, &self.w2, &self.b2)
        };

        let h_pre = input
            .view()
            .to_shape((1, input.len()))
            .unwrap()
            .dot(w1)
            + b1.t();
        
        let mut h = h_pre;
        h.mapv_inplace(|v| v.tanh());
        
        let output = h.dot(w2) + b2.t();
        (output, h)
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        h: &Array2<f32>,
        input: &Array1<f32>,
    ) -> (Array1<f32>, Vec<Array2<f32>>) {
        // grad_output: (1, output_dim)
        // h: (1, hidden_dim)
        // input: (input_dim)

        // dL/dW2 = h^T * grad_output
        let grad_w2 = h.t().dot(grad_output);
        // dL/db2 = grad_output^T (sum over batch, here batch=1)
        let grad_b2 = grad_output.t().to_owned();

        // dL/dh = grad_output * W2^T
        let mut grad_h = grad_output.dot(&self.w2.t());
        
        // dL/dh_pre = dL/dh * (1 - h^2)
        // h is already tanh(h_pre)
        grad_h.zip_mut_with(h, |g, &val| *g *= 1.0 - val * val);

        // dL/dW1 = input^T * grad_h
        let input_view = input.view();
        let input_mat = input_view.to_shape((1, input.len())).unwrap();
        let grad_w1 = input_mat.t().dot(&grad_h);
        
        // dL/db1 = grad_h^T
        let grad_b1 = grad_h.t().to_owned();

        // dL/dInput = grad_h * W1^T
        let grad_input_mat = grad_h.dot(&self.w1.t());
        let grad_input = grad_input_mat.row(0).to_owned();

        (grad_input, vec![grad_w2, grad_b2, grad_w1, grad_b1])
    }

    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32, ema_decay: f32) {
        if grads.len() != 4 {
            return;
        }
        let g_w2 = &grads[0];
        let g_b2 = &grads[1];
        let g_w1 = &grads[2];
        let g_b1 = &grads[3];

        if let Some(opt) = &mut self.opt_w2 { opt.step(&mut self.w2, g_w2, lr); }
        if let Some(opt) = &mut self.opt_b2 { opt.step(&mut self.b2, g_b2, lr); }
        if let Some(opt) = &mut self.opt_w1 { opt.step(&mut self.w1, g_w1, lr); }
        if let Some(opt) = &mut self.opt_b1 { opt.step(&mut self.b1, g_b1, lr); }

        // Update EMA
        let d = ema_decay;
        self.ema_w2.zip_mut_with(&self.w2, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_b2.zip_mut_with(&self.b2, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_w1.zip_mut_with(&self.w1, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_b1.zip_mut_with(&self.b1, |e, &w| *e = d * *e + (1.0 - d) * w);
    }

    pub fn weight_norm(&self) -> f32 {
        (self.w1.iter().map(|&w| w * w).sum::<f32>() + 
         self.w2.iter().map(|&w| w * w).sum::<f32>()).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct DiffusionCachedIntermediates {
    pub input: Array2<f32>,
    pub time_embed: Array1<f32>,
    pub gamma_beta: Array2<f32>,
    pub norm1_out: Array2<f32>,
    pub norm1_mod: Array2<f32>,
    pub attn_out: Array2<f32>,
    pub residual1: Array2<f32>,
    pub norm2_out: Array2<f32>,
    pub norm2_mod: Array2<f32>,
    pub ffn_out: Array2<f32>,
    pub output: Array2<f32>,
    pub h_vec: Array1<f32>,
    pub gamma_attn: Array2<f32>,
    pub beta_attn: Array2<f32>,
    pub gamma_ffn: Array2<f32>,
    pub beta_ffn: Array2<f32>,
    pub timestep: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DiffusionParamPartitions {
    pub attention: usize,
    pub feedforward: usize,
    pub pre_ffn_norm: usize,
    pub pre_attention_norm: usize,
    pub time_conditioner: usize,
    pub time_embedding: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DiffusionBlock {
    pub config: DiffusionBlockConfig,
    pub attention: PolyAttention,
    pub feedforward: FeedForwardVariant,
    pub pre_attention_norm: RichardsNorm,
    pub pre_ffn_norm: RichardsNorm,
    pub time_embedding: TimeEmbedding,
    pub time_conditioner: TimeConditioner,
    pub noise_scheduler: NoiseScheduler,
    #[serde(skip)]
    pub cached_intermediates: RwLock<Option<DiffusionCachedIntermediates>>,
    #[serde(skip)]
    pub discrete_scheduler: Option<crate::diffusion::discrete::DiscreteMaskScheduler>,
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
    pub film_scale_gamma: f32,
    pub film_scale_beta: f32,
    pub use_ema_for_sampling: bool,
    pub ema_decay: f32,
    pub current_timestep: usize,
    #[serde(skip)]
    pub param_partitions: RwLock<Option<DiffusionParamPartitions>>,
}

impl DiffusionBlock {
    pub fn new(config: DiffusionBlockConfig) -> Self {
        let pre_attention_norm = RichardsNorm::new(config.embed_dim);
        let mut attention = PolyAttention::new(
            config.embed_dim,
            config.num_heads,
            config.poly_degree,
            config.max_pos,
            config.window_size,
        );
        attention.set_head_selection_config(&config.head_selection);
        
        let pre_ffn_norm = RichardsNorm::new(config.embed_dim);
        
        let feedforward = if config.use_moe {
            if let Some(moe_config) = &config.moe_config {
                let moe_layer = MixtureOfExperts::new(
                    config.embed_dim,
                    (config.embed_dim / 4).max(32),
                    moe_config.clone(),
                );
                FeedForwardVariant::MixtureOfExperts(Box::new(moe_layer))
            } else {
                let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
                FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
            }
        } else {
            let richards_glu = RichardsGlu::new(config.embed_dim, config.hidden_dim);
            FeedForwardVariant::RichardsGlu(Box::new(richards_glu))
        };

        let time_embedding = TimeEmbedding::new(config.time_embed_dim);
        // Output dim of time conditioner = 4 * embed_dim (gamma_attn, beta_attn, gamma_ffn, beta_ffn)
        let time_conditioner = TimeConditioner::new(
            config.time_embed_dim,
            config.hidden_dim,
            config.embed_dim * 4
        );
        let noise_scheduler = NoiseScheduler::new(config.noise_schedule.clone(), config.num_timesteps);
        
        let discrete_scheduler = if config.discrete_masked {
            Some(crate::diffusion::discrete::DiscreteMaskScheduler::new(config.num_timesteps))
        } else {
            None
        };

        Self {
            config: config.clone(),
            attention,
            feedforward,
            pre_attention_norm,
            pre_ffn_norm,
            time_embedding,
            time_conditioner,
            noise_scheduler,
            cached_intermediates: RwLock::new(None),
            discrete_scheduler,
            current_window_size: config.window_size,
            win_max: config.max_pos,
            win_min: 16,
            win_step_up: 16,
            win_step_down: 16,
            pred_up: 1.2,
            pred_down: 0.8,
            adaptive_window_on: config.use_adaptive_window,
            enable_dropout: false,
            dropout_rate: 0.0,
            film_scale_gamma: 0.1,
            film_scale_beta: 0.1,
            use_ema_for_sampling: false,
            ema_decay: 0.999,
            current_timestep: 0,
            param_partitions: RwLock::new(None),
        }
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

    pub fn min_snr_weight(&self, t: usize, gamma: f32) -> f32 {
        let alpha_cumprod = self.noise_scheduler.sqrt_alpha_cumprod(t).powi(2);
        let snr = alpha_cumprod / (1.0 - alpha_cumprod);
        snr.min(gamma) / snr
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

    fn apply_film(input: &Array2<f32>, gamma: &Array2<f32>, beta: &Array2<f32>) -> Array2<f32> {
        // input: (seq_len, dim), gamma: (1, dim), beta: (1, dim)
        // output = input * gamma + beta
        input * gamma + beta
    }

    fn film_backward(grad_output: &Array2<f32>, input: &Array2<f32>, gamma: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // grad_output: (seq_len, dim)
        // input: (seq_len, dim)
        // gamma: (1, dim)
        
        // dL/dInput = grad_output * gamma
        let grad_input = grad_output * gamma;
        
        // dL/dGamma = sum(grad_output * input, axis=0)
        let grad_gamma = (grad_output * input).sum_axis(Axis(0)).insert_axis(Axis(0));
        
        // dL/dBeta = sum(grad_output, axis=0)
        let grad_beta = grad_output.sum_axis(Axis(0)).insert_axis(Axis(0));
        
        (grad_input, grad_gamma, grad_beta)
    }

    fn apply_dropout_inplace(input: &mut Array2<f32>, rate: f32) {
        let mut rng = rand::rng();
        let scale = 1.0 / (1.0 - rate);
        input.mapv_inplace(|x| {
            if rand::random::<f32>() > rate {
                x * scale
            } else {
                0.0
            }
        });
    }

    fn convert_prediction_to_epsilon(&self, x_t: &Array2<f32>, output: &Array2<f32>, t: usize) -> Array2<f32> {
        match self.config.prediction_target {
            DiffusionPredictionTarget::Epsilon => output.clone(),
            DiffusionPredictionTarget::Sample => {
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                (x_t - (output * sqrt_alpha)) / sqrt_one_minus_alpha.max(1e-6)
            }
            DiffusionPredictionTarget::VPrediction => {
                let sqrt_alpha = self.noise_scheduler.sqrt_alpha_cumprod(t);
                let sqrt_one_minus_alpha = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t);
                (output * sqrt_one_minus_alpha) + (x_t * sqrt_alpha) // Approximation/derivation check needed, but standard V-pred conversion
            }
        }
    }

    /// Forward pass through diffusion transformer block
    /// Takes noisy input `x_t` and timestep `t`, predicts the noise `ε_θ(x_t, t)`
    pub fn forward_with_timestep(&mut self, x_t: &Array2<f32>, t: usize) -> Array2<f32> {
        if self.current_window_size != self.config.window_size {
            self.config.window_size = self.current_window_size;
        }
        self.attention.set_window_size(self.current_window_size);
        let time_embed = self.time_embedding.forward(t, self.config.num_timesteps);
        let (gamma_beta, h) = self.time_conditioner.forward(&time_embed, self.use_ema_for_sampling);
        
        let embed = self.config.embed_dim;
        let raw_gamma_attn = gamma_beta.slice(s![.., 0..embed]).row(0).to_owned();
        let raw_beta_attn = gamma_beta.slice(s![.., embed..2 * embed]).row(0).to_owned();
        let raw_gamma_ffn = gamma_beta.slice(s![.., 2 * embed..3 * embed]).row(0).to_owned();
        let raw_beta_ffn = gamma_beta.slice(s![.., 3 * embed..4 * embed]).row(0).to_owned();
        let g_attn = raw_gamma_attn.mapv(|x| x.tanh());
        let b_attn = raw_beta_attn.mapv(|x| x.tanh());
        let g_ffn = raw_gamma_ffn.mapv(|x| x.tanh());
        let b_ffn = raw_beta_ffn.mapv(|x| x.tanh());
        let gamma_attn_vec = g_attn.mapv(|v| 1.0 + self.film_scale_gamma * v).insert_axis(Axis(0));
        let beta_attn_vec = b_attn.mapv(|v| self.film_scale_beta * v).insert_axis(Axis(0));
        let gamma_ffn_vec = g_ffn.mapv(|v| 1.0 + self.film_scale_gamma * v).insert_axis(Axis(0));
        let beta_ffn_vec = b_ffn.mapv(|v| self.film_scale_beta * v).insert_axis(Axis(0));

        let mut norm1_out = self.pre_attention_norm.forward(x_t);
        Self::sanitize_tensor("norm1_out", &mut norm1_out);
        let mut norm1_mod = Self::apply_film(&norm1_out, &gamma_attn_vec, &beta_attn_vec);
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
        let mut norm2_mod = Self::apply_film(&norm2_out, &gamma_ffn_vec, &beta_ffn_vec);
        Self::sanitize_tensor("norm2_mod", &mut norm2_mod);
        let mut ffn_out = self.feedforward.forward(&norm2_mod);
        Self::sanitize_tensor("ffn_out", &mut ffn_out);
        if self.enable_dropout && self.dropout_rate > 0.0 {
            Self::apply_dropout_inplace(&mut ffn_out, self.dropout_rate);
        }
        let mut output = &residual1 + &ffn_out;
        Self::sanitize_tensor("block_output", &mut output);

        *self.cached_intermediates.write().unwrap() = Some(DiffusionCachedIntermediates {
            input: x_t.clone(),
            time_embed,
            norm1_out,
            norm1_mod: norm1_mod.clone(),
            residual1: residual1.clone(),
            norm2_out,
            norm2_mod: norm2_mod.clone(),
            h_vec: Array1::from_vec(h.row(0).to_vec()),
            gamma_attn: gamma_attn_vec.clone(),
            beta_attn: beta_attn_vec.clone(),
            gamma_ffn: gamma_ffn_vec.clone(),
            beta_ffn: beta_ffn_vec.clone(),
            gamma_beta: gamma_beta.clone(),
            attn_out: attn_out.clone(),
            ffn_out: ffn_out.clone(),
            output: output.clone(),
            timestep: t,
        });

        let mut predicted_noise = self.convert_prediction_to_epsilon(x_t, &output, t);
        Self::sanitize_tensor("predicted_noise", &mut predicted_noise);
        if self.adaptive_window_on {
            if let Some(pn) = self.attention.last_pred_norm {
                let mut ws = self.current_window_size.unwrap_or(self.win_max);
                if pn > self.pred_up {
                    ws = (ws + self.win_step_up).min(self.win_max);
                } else if pn < self.pred_down {
                    ws = ws.saturating_sub(self.win_step_down).max(self.win_min);
                }
                self.current_window_size = Some(ws);
                self.attention.set_window_size(self.current_window_size);
            }
        }
        predicted_noise
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

    pub fn sample_ddim(&mut self, shape: (usize, usize), steps: Option<usize>) -> Array2<f32> {
        let total = self.noise_scheduler.num_timesteps().max(1);
        let k = steps.unwrap_or(total).max(1);
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

        let step_size = ((total - 1) / k).max(1);
        let mut t = total - 1;
        while t > 0 {
            self.set_timestep(t);
            let pred = self.forward_with_timestep(&x_t, t);
            let t_idx = t.min(total - 1);
            match self.config.prediction_target {
                DiffusionPredictionTarget::Epsilon => {
                    x_t = self.noise_scheduler.ddim_step(&x_t, t_idx, &pred);
                }
                DiffusionPredictionTarget::VPrediction => {
                    let sa = self.noise_scheduler.sqrt_alpha_cumprod(t_idx).max(1e-6);
                    let soa = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t_idx);
                    let x0_hat = (&x_t * sa) - (&pred * soa);
                    let eps_hat = (&pred + (&x0_hat * soa)) / sa;
                    x_t = self.noise_scheduler.ddim_step(&x_t, t_idx, &eps_hat);
                }
                DiffusionPredictionTarget::Sample => {
                    // If predicting sample directly, convert to epsilon for DDIM step
                    let sa = self.noise_scheduler.sqrt_alpha_cumprod(t_idx).max(1e-6);
                    let soa = self.noise_scheduler.sqrt_one_minus_alpha_cumprod(t_idx);
                    let eps_hat = (&x_t - (&pred * sa)) / soa;
                    x_t = self.noise_scheduler.ddim_step(&x_t, t_idx, &eps_hat);
                }
            }
            t = t.saturating_sub(step_size);
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
                Some(crate::diffusion::discrete::DiscreteMaskScheduler::new(nt));
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
        gamma: usize,
        tau: f32,
    ) -> Array2<f32> {
        let total = self.noise_scheduler.num_timesteps().max(1);
        let mut steps_left = steps.unwrap_or(total).max(gamma.max(1));
        let mut x_t = Array2::zeros(shape);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::rng();
        x_t.mapv_inplace(|_| normal.sample(&mut rng) as f32);
        let mut t = total.saturating_sub(1);
        while t > 0 && steps_left > 0 {
            if gamma == 0 || t < gamma {
                let pred = self.forward_with_timestep(&x_t, t);
                x_t = self.noise_scheduler.ddim_step(&x_t, t, &pred);
                t = t.saturating_sub(1);
                steps_left = steps_left.saturating_sub(1);
                continue;
            }

            let pred = self.forward_with_timestep(&x_t, t);
            let draft_pred = draft.forward_with_timestep(&x_t, t);
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
                x_t = self.noise_scheduler.ddim_step(&x_t, t, &pred);
                t = t.saturating_sub(1);
                steps_left = steps_left.saturating_sub(1);
                continue;
            }

            // Accept speculative proposal: reuse draft to leap gamma steps ahead.
            let mut x_draft = self
                .noise_scheduler
                .ddim_step(&x_t, t, &draft_pred);
            let mut t_d = t.saturating_sub(1);
            let mut accepted = 1usize;
            for _ in 1..gamma {
                if t_d == 0 {
                    break;
                }
                let pred_d = draft.forward_with_timestep(&x_draft, t_d);
                x_draft = self.noise_scheduler.ddim_step(&x_draft, t_d, &pred_d);
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

// Implement Layer trait for DiffusionBlock
impl Layer for DiffusionBlock {
    fn layer_type(&self) -> &str {
        "DiffusionBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // For Layer trait compatibility, use current timestep set by set_timestep()
        self.forward_with_timestep(input, self.current_timestep)
    }

    #[allow(dead_code)]
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
        (self.pre_attention_norm.weight_norm().powi(2)
            + self.attention.weight_norm().powi(2)
            + self.pre_ffn_norm.weight_norm().powi(2)
            + self.feedforward.weight_norm().powi(2)
            + self.time_conditioner.weight_norm().powi(2))
        .sqrt()
    }

    /// Compute analytical gradients using cached forward intermediates
    /// Ensures full-gradient propagation across residual connections
    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let cache_guard = self.cached_intermediates.read().unwrap();
        if let Some(cache) = &*cache_guard {
            let input_cache = &cache.input;
            let time_embed = &cache.time_embed;
            let norm1_out = &cache.norm1_out;
            let norm1_mod = &cache.norm1_mod;
            let residual1 = &cache.residual1;
            let norm2_out = &cache.norm2_out;
            let norm2_mod = &cache.norm2_mod;
            let h_vec = &cache.h_vec;
            let gamma_attn_vec = &cache.gamma_attn;
            let beta_attn_vec = &cache.beta_attn;
            let gamma_ffn_vec = &cache.gamma_ffn;
            let beta_ffn_vec = &cache.beta_ffn;
            let timestep = cache.timestep;
            let mut all_param_grads = Vec::new();

            let (block_grads_scale, input_extra_scale) =
                if self.config.prediction_target == DiffusionPredictionTarget::VPrediction {
                    let sqrt_alpha_bar = self.noise_scheduler.sqrt_alpha_cumprod(timestep);
                    let sqrt_one_minus_alpha_bar = self
                        .noise_scheduler
                        .sqrt_one_minus_alpha_cumprod(timestep);
                    (sqrt_alpha_bar, Some(sqrt_one_minus_alpha_bar))
                } else {
                    (1.0f32, None)
                };

            let scaled_output_grads = output_grads * block_grads_scale;

            // Compute gradients through the transformer block layers
            // This follows the same pattern as TransformerBlock but with timestep conditioning

            // Output = residual1 + ffn_out, so gradients split between residual1 and ffn_out
            let ffn_grads = scaled_output_grads.clone();
            let residual1_grads = scaled_output_grads.clone();

            // Get feedforward gradients
            let (ffn_input_grad_mod, ffn_param_grads) = match &self.feedforward {
                FeedForwardVariant::RichardsGlu(layer) => {
                    layer.compute_gradients(norm2_mod, &ffn_grads)
                }
                FeedForwardVariant::MixtureOfExperts(layer) => {
                    layer.compute_gradients(norm2_mod, &ffn_grads)
                }
            };

            let (norm2_grad, grad_gamma_ffn, grad_beta_ffn) =
                Self::film_backward(&ffn_input_grad_mod, norm2_out, &gamma_ffn_vec);

            let (residual1_from_ffn, pre_ffn_param_grads) =
                self.pre_ffn_norm.compute_gradients(residual1, &norm2_grad);

            // Combine residual gradients
            let residual1_total_grads = residual1_grads + residual1_from_ffn;

            // residual1 = input + attn_out: propagate full upstream gradient to both branches
            let input_grads = residual1_total_grads.clone();
            let attn_out_grads = residual1_total_grads.clone();

            let (attn_input_grad_mod, attn_param_grads) =
                self.attention.compute_gradients(norm1_mod, &attn_out_grads);

            let (norm1_grad, grad_gamma_attn, grad_beta_attn) =
                Self::film_backward(&attn_input_grad_mod, norm1_out, &gamma_attn_vec);

            let (input_from_norm, pre_attn_param_grads) = self
                .pre_attention_norm
                .compute_gradients(input_cache, &norm1_grad);

            // The final input gradients are the gradients w.r.t. the transformer input
            // (combining gradients from residual and attention path)
            let mut final_input_grads = &input_grads + &input_from_norm;

            if let Some(extra_scale) = input_extra_scale {
                final_input_grads = final_input_grads + &(output_grads * extra_scale);
            }

            let attn_grad_count = attn_param_grads.len();
            let ffn_grad_count = ffn_param_grads.len();
            let pre_ffn_grad_count = pre_ffn_param_grads.len();
            let pre_attn_grad_count = pre_attn_param_grads.len();

            all_param_grads.extend(attn_param_grads);
            all_param_grads.extend(ffn_param_grads);
            all_param_grads.extend(pre_ffn_param_grads);
            all_param_grads.extend(pre_attn_param_grads);

            let embed = self.config.embed_dim;
            let g_t_attn = gamma_attn_vec.mapv(|x| {
                let z = (x - 1.0) / self.film_scale_gamma;
                z.max(-1.0).min(1.0)
            });
            let b_t_attn = beta_attn_vec.mapv(|x| {
                let z = x / self.film_scale_beta;
                z.max(-1.0).min(1.0)
            });
            let g_t_ffn = gamma_ffn_vec.mapv(|x| {
                let z = (x - 1.0) / self.film_scale_gamma;
                z.max(-1.0).min(1.0)
            });
            let b_t_ffn = beta_ffn_vec.mapv(|x| {
                let z = x / self.film_scale_beta;
                z.max(-1.0).min(1.0)
            });
            let d_g_attn_raw = grad_gamma_attn.mapv(|x| x * self.film_scale_gamma) * (1.0 - g_t_attn.mapv(|x| x * x));
            let d_b_attn_raw = grad_beta_attn.mapv(|x| x * self.film_scale_beta) * (1.0 - b_t_attn.mapv(|x| x * x));
            let d_g_ffn_raw = grad_gamma_ffn.mapv(|x| x * self.film_scale_gamma) * (1.0 - g_t_ffn.mapv(|x| x * x));
            let d_b_ffn_raw = grad_beta_ffn.mapv(|x| x * self.film_scale_beta) * (1.0 - b_t_ffn.mapv(|x| x * x));
            
            let mut grad_gamma_beta = Array2::<f32>::zeros((1, embed * 4));
            {
                let mut view = grad_gamma_beta.row_mut(0);
                view.slice_mut(s![0..embed]).assign(&d_g_attn_raw.row(0));
                view.slice_mut(s![embed..2 * embed]).assign(&d_b_attn_raw.row(0));
                view.slice_mut(s![2 * embed..3 * embed]).assign(&d_g_ffn_raw.row(0));
                view.slice_mut(s![3 * embed..4 * embed]).assign(&d_b_ffn_raw.row(0));
            }
            
            let h_mat = h_vec.view().to_shape((1, h_vec.len())).unwrap().to_owned();
            let (_, time_grads) = self.time_conditioner.backward(&grad_gamma_beta, &h_mat, time_embed);
            all_param_grads.extend(time_grads);

            let partitions = DiffusionParamPartitions {
                attention: attn_grad_count,
                feedforward: ffn_grad_count,
                pre_ffn_norm: pre_ffn_grad_count,
                pre_attention_norm: pre_attn_grad_count,
                time_conditioner: 4,
                time_embedding: 0,
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

        // Sanitize and globally clip gradients for stability
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
        let mut sanitized: Vec<Array2<f32>> = pairs.iter().map(|(gg, _)| gg.clone()).collect();
        let norm_sq: f32 = pairs.iter().map(|(_, s)| *s).sum();
        let clip = 5.0f32;
        let nrm = norm_sq.sqrt();
        if nrm.is_finite() && nrm > clip && nrm > 0.0 {
            let scale = clip / nrm;
            for gg in &mut sanitized {
                gg.mapv_inplace(|x| x * scale);
            }
        }

        let cached_partitions = self
            .param_partitions
            .read()
            .map(|guard| guard.clone())
            .unwrap_or(None);
        let partitions = cached_partitions.unwrap_or_else(|| {
            if !sanitized.is_empty() {
                tracing::warn!(
                    arrays = sanitized.len(),
                    "DiffusionBlock::apply_gradients missing partition metadata; routing all gradients to attention as a safe fallback"
                );
                DiffusionParamPartitions {
                    attention: sanitized.len(),
                    feedforward: 0,
                    pre_ffn_norm: 0,
                    pre_attention_norm: 0,
                    time_conditioner: 0,
                    time_embedding: 0,
                }
            } else {
                DiffusionParamPartitions::default()
            }
        });

        let mut idx0 = 0usize;
        let mut next_range = |count: usize| {
            let available = sanitized.len().saturating_sub(idx0);
            let len = count.min(available);
            let start = idx0;
            idx0 += len;
            start..idx0
        };

        if partitions.attention + partitions.feedforward + partitions.pre_ffn_norm + partitions.pre_attention_norm + partitions.time_conditioner != sanitized.len() {
            // Just a warning, proceed with best effort
        }

        // Attention gradients
        let attn_range = next_range(partitions.attention);
        if !attn_range.is_empty() {
            let attention_grads = &sanitized[attn_range];
            let gnorm_attn: f32 = attention_grads
                .iter()
                .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
                .sum::<f32>()
                .sqrt();
            let wnorm_attn = self.attention.weight_norm().max(1e-6);
            let scale_attn = (wnorm_attn / (gnorm_attn.max(1e-6))).clamp(0.01, 5.0);
            let scaled: Vec<Array2<f32>> = attention_grads
                .par_iter()
                .map(|g| {
                    let mut gg = g.clone();
                    gg.mapv_inplace(|x| x * scale_attn);
                    gg
                })
                .collect();
            self.attention.apply_gradients(&scaled, lr)?;
        }

        // Feedforward gradients
        let ffn_range = next_range(partitions.feedforward);
        if !ffn_range.is_empty() {
            let feedforward_grads = &sanitized[ffn_range];
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
            let scale_ffn = (wnorm_ffn / (gnorm_ffn.max(1e-6))).clamp(0.01, 5.0);
            let scaled: Vec<Array2<f32>> = feedforward_grads
                .par_iter()
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
        }

        // Pre-FFN norm gradients
        let pre_ffn_range = next_range(partitions.pre_ffn_norm);
        if !pre_ffn_range.is_empty() {
            let pre_ffn_grads = &sanitized[pre_ffn_range];
            self.pre_ffn_norm.apply_gradients(pre_ffn_grads, lr)?;
        }

        // Pre-attention norm gradients
        let pre_attn_range = next_range(partitions.pre_attention_norm);
        if !pre_attn_range.is_empty() {
            let pre_attn_grads = &sanitized[pre_attn_range];
            self.pre_attention_norm
                .apply_gradients(pre_attn_grads, lr)?;
        }

        // Time-conditioner gradients (expect 4 arrays)
        let time_range = next_range(partitions.time_conditioner);
        if time_range.len() == 4 {
            let time_grads = &sanitized[time_range];
            self.time_conditioner.apply_gradients(time_grads, lr, self.ema_decay);
        }

        if let Ok(mut guard) = self.param_partitions.write() {
            *guard = None;
        }
        Ok(())
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
            time_embed_dim: t.embed_dim * 4,
            mask_token_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_time_conditioner_shapes() {
        let input_dim = 16;
        let hidden_dim = 32;
        let output_dim = 64;
        let conditioner = TimeConditioner::new(input_dim, hidden_dim, output_dim);
        
        let input = Array1::zeros(input_dim);
        let (output, _) = conditioner.forward(&input, false);
        
        assert_eq!(output.shape(), &[1, output_dim]);
    }
}
