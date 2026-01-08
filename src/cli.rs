use clap::{Parser, ValueEnum};

use crate::{
    layers::diffusion::{DiffusionPredictionTarget, NoiseSchedule},
    model_config::{DiffusionTimestepStrategy, TemporalMixingType},
};

/// CLI argument parsing for the LLM training and inference tool
#[derive(Parser)]
#[command(name = "llm")]
#[command(about = "Train and run a language model")]
pub struct Args {
    /// Enable interactive prompt after training
    #[arg(short)]
    pub interactive: bool,

    /// Random seed for reproducible training.
    /// When set, all random operations use deterministic sequences.
    /// Use the same seed to get identical results across runs.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Use hard head selection (top-k) instead of soft gating for MoH
    /// Hard mode: Only compute attention for selected heads (saves computation)
    /// Soft mode (default): Compute all heads and apply soft gating weights
    #[arg(long)]
    pub hard_heads: bool,

    /// Continue training from an existing model file (skips pre-training)
    #[arg(long)]
    pub continue_from: Option<String>,

    /// Use E-prop (Eligibility Propagation) training instead of standard backpropagation
    /// E-prop is a biologically plausible online learning algorithm for spiking neural networks
    /// with O(N) complexity vs O(N²) for standard e-prop
    #[arg(long)]
    pub eprop: bool,

    #[arg(long)]
    pub diffusion: bool,

    #[arg(long)]
    pub trm: bool,

    #[arg(long, default_value_t = 0.5)]
    pub diffusion_ce_weight: f32,

    #[arg(long, default_value_t = 3.0)]
    pub diffusion_min_snr_gamma: f32,

    #[arg(long, value_enum, default_value_t = DiffusionTargetCli::Epsilon)]
    pub diffusion_prediction_target: DiffusionTargetCli,

    /// Noise schedule used by diffusion blocks (cosine, linear, quadratic)
    #[arg(long, value_enum, default_value_t = DiffusionScheduleCli::Cosine)]
    pub diffusion_noise_schedule: DiffusionScheduleCli,

    /// Timestep sampling strategy for diffusion training
    #[arg(long, value_enum, default_value_t = DiffusionTimestepCli::Uniform)]
    pub diffusion_timestep_strategy: DiffusionTimestepCli,

    /// Enable speculative sampling (diffusion or transformer) with a cheaper draft chain
    #[arg(long)]
    pub speculative: bool,

    /// Speculative sampling mode: "diffusion" or "transformer"
    /// If not specified, auto-detected from model type:
    /// - With --diffusion: uses diffusion speculation
    /// - Without --diffusion: uses transformer speculation
    #[arg(long)]
    pub speculative_mode: Option<String>,

    /// Number of draft steps per speculative proposal
    #[arg(long, default_value_t = 4)]
    pub speculative_gamma: usize,

    /// Acceptance threshold (tau) for speculative verification
    #[arg(long, default_value_t = 0.001)]
    pub speculative_tau: f32,

    /// Number of layers to use for the speculative draft pass
    #[arg(long)]
    pub speculative_draft_layers: Option<usize>,

    #[arg(long)]
    pub ddim_steps: Option<usize>,

    #[arg(long, default_value_t = 0.10)]
    pub validation_ratio: f32,

    #[arg(long)]
    pub trm_recursions: Option<usize>,

    #[arg(long)]
    pub trm_supervision_steps: Option<usize>,

    #[arg(long)]
    pub trm_inference_steps: Option<usize>,

    #[arg(long)]
    pub trm_latent_moh: Option<bool>,

    #[arg(long, default_value_t = 0.6)]
    pub trm_latent_moh_top_p_min: f32,

    #[arg(long, default_value_t = 0.95)]
    pub trm_latent_moh_top_p_max: f32,

    /// Number of epochs to run during pre-training (default 100)
    #[arg(long, default_value_t = 100)]
    pub pretrain_epochs: usize,

    /// Number of epochs to run during instruction tuning (default 100)
    #[arg(long, default_value_t = 100)]
    pub instruction_epochs: usize,

    /// Enable Mixture-of-Experts (MoE) for feedforward layers
    /// When enabled, replaces standard feedforward layers with sparse MoE layers
    /// Each MoE layer contains multiple expert networks with learned routing
    #[arg(long)]
    pub moe: bool,

    /// Enable/disable learned router temperature for MoE (log-space parameterization).
    ///
    /// If not set, defaults to enabled when MoE is enabled.
    #[arg(long)]
    pub moe_learned_temperature: Option<bool>,

    /// Initial router log-temperature for MoE (temperature = exp(logT)).
    ///
    /// If not set, defaults to 0.0 (T=1).
    #[arg(long)]
    pub moe_router_log_temperature_init: Option<f32>,

    /// Learning-rate multiplier for MoE router log-temperature updates.
    ///
    /// If not set, defaults to a small multiplier (e.g. 0.05).
    #[arg(long)]
    pub moe_router_temperature_lr_mult: Option<f32>,

    /// Enable/disable MoH head-conditioned router temperature (logT_eff = logT + head_scale * h).
    ///
    /// If not set, defaults to enabled.
    #[arg(long)]
    pub moe_head_conditioned_temperature: Option<bool>,

    /// Initial scale for head-conditioned log-temperature.
    ///
    /// If not set, defaults to 0.0.
    #[arg(long)]
    pub moe_router_log_temperature_head_scale_init: Option<f32>,

    /// Learning-rate multiplier for head-conditioned log-temperature scale.
    ///
    /// If not set, defaults to 0.05.
    #[arg(long)]
    pub moe_router_temperature_head_lr_mult: Option<f32>,

    /// Enable/disable MoE router exploration noise injection during training.
    ///
    /// If not set, defaults to enabled.
    #[arg(long)]
    pub moe_router_use_noise: Option<bool>,

    /// Initial log-standard-deviation for MoE router exploration noise.
    ///
    /// If not set, defaults to -2.0 (σ ≈ 0.135).
    #[arg(long)]
    pub moe_router_log_noise_std_init: Option<f32>,

    /// Learning-rate multiplier for MoE router noise log-std updates.
    ///
    /// If not set, defaults to 0.05.
    #[arg(long)]
    pub moe_router_noise_lr_mult: Option<f32>,

    /// Enable/disable MoH head-conditioned router noise scale.
    ///
    /// If not set, defaults to enabled.
    #[arg(long)]
    pub moe_head_conditioned_noise: Option<bool>,

    /// Initial scale for head-conditioned router noise.
    ///
    /// If not set, defaults to 0.0.
    #[arg(long)]
    pub moe_router_log_noise_head_scale_init: Option<f32>,

    /// Learning-rate multiplier for head-conditioned router noise scale.
    ///
    /// If not set, defaults to 0.05.
    #[arg(long)]
    pub moe_router_noise_head_lr_mult: Option<f32>,

    /// Temporal mixing mechanism (attention vs SSM-style RG-LRU)
    #[arg(long, value_enum, default_value_t = TemporalMixingCli::Attention)]
    pub temporal_mixing: TemporalMixingCli,

    /// Auxiliary residual decorrelation weight (VICReg/Barlow-Twins style redundancy reduction).
    ///
    /// When > 0, adds a loss term that penalizes off-diagonal covariance of the residual stream
    /// right before the OutputProjection, encouraging features to be distinct ("what it is") and
    /// less confusable ("what it is not").
    #[arg(long, default_value_t = 0.01)]
    pub residual_decorrelation_weight: f32,

    /// If set, scales residual decorrelation strength up on harder examples (higher CE/SCE).
    #[arg(long, default_value_t = true)]
    pub residual_decorrelation_adaptive: bool,

    /// Auxiliary hard-negative residual repulsion weight (cosine-based, memory-bank hard negatives).
    ///
    /// When > 0, penalizes residual representations that are too similar to recent representations
    /// from other examples, using hard-negative top-k mining. This explicitly teaches “what it is
    /// not” by pushing away confusable states.
    #[arg(long, default_value_t = 0.005)]
    pub residual_hardneg_weight: f32,

    /// If set, scales hard-negative repulsion up on harder examples (higher CE/SCE).
    #[arg(long, default_value_t = true)]
    pub residual_hardneg_adaptive: bool,

    /// Number of hard negatives (top-k by cosine similarity) to use from the memory bank.
    #[arg(long, default_value_t = 8)]
    pub residual_hardneg_k: usize,

    /// Cosine similarity margin; similarities above this are penalized.
    #[arg(long, default_value_t = 0.2)]
    pub residual_hardneg_margin: f32,

    /// Temperature for the softplus penalty on (sim - margin).
    #[arg(long, default_value_t = 0.07)]
    pub residual_hardneg_temperature: f32,

    /// Maximum number of pooled residual vectors stored in the hard-negative memory bank.
    #[arg(long, default_value_t = 512)]
    pub residual_hardneg_bank_size: usize,
}

/// CLI representation of temporal mixing types
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum TemporalMixingCli {
    /// Use attention for temporal mixing (default)
    Attention,
    /// Use RG-LRU recurrent temporal mixing (SSM-style)
    #[value(alias = "rglru", alias = "rg-lru", alias = "ssm")]
    RgLru,

    /// Use Mamba selective SSM
    #[value(alias = "mamba")]
    Mamba,

    /// Use Mamba-2 style selective SSM
    #[value(alias = "mamba2", alias = "mamba-2")]
    Mamba2,
}

impl From<TemporalMixingCli> for TemporalMixingType {
    fn from(arg: TemporalMixingCli) -> Self {
        match arg {
            TemporalMixingCli::Attention => TemporalMixingType::Attention,
            TemporalMixingCli::RgLru => TemporalMixingType::RgLru,
            TemporalMixingCli::Mamba => TemporalMixingType::Mamba,
            TemporalMixingCli::Mamba2 => TemporalMixingType::Mamba2,
        }
    }
}

/// CLI representation of diffusion prediction targets
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionTargetCli {
    #[value(alias = "eps")]
    Epsilon,
    #[value(alias = "v", alias = "vpred")]
    VPrediction,

    /// EDM-style preconditioned x0 prediction
    #[value(alias = "edm", alias = "edmx0", alias = "edm-x0")]
    EdmX0,
}

impl From<DiffusionTargetCli> for DiffusionPredictionTarget {
    fn from(arg: DiffusionTargetCli) -> Self {
        match arg {
            DiffusionTargetCli::Epsilon => DiffusionPredictionTarget::Epsilon,
            DiffusionTargetCli::VPrediction => DiffusionPredictionTarget::VPrediction,
            DiffusionTargetCli::EdmX0 => DiffusionPredictionTarget::EdmX0,
        }
    }
}

/// CLI representation of diffusion noise schedules
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionScheduleCli {
    Cosine,
    Linear,
    Quadratic,
    /// Karras/EDM-inspired sigma schedule mapped to VP betas
    Karras,
}

impl From<DiffusionScheduleCli> for NoiseSchedule {
    fn from(arg: DiffusionScheduleCli) -> Self {
        match arg {
            DiffusionScheduleCli::Cosine => NoiseSchedule::Cosine { s: 0.008 },
            DiffusionScheduleCli::Linear => NoiseSchedule::Linear {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            DiffusionScheduleCli::Quadratic => NoiseSchedule::Quadratic {
                beta_min: 1e-4,
                beta_max: 0.02,
            },
            DiffusionScheduleCli::Karras => NoiseSchedule::Karras {
                sigma_min: 0.002,
                sigma_max: 80.0,
                rho: 7.0,
            },
        }
    }
}

/// CLI representation of diffusion timestep strategies
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionTimestepCli {
    Uniform,
    #[value(alias = "minsnr", alias = "min-snr")]
    MinSnr,

    /// EDM-style log-normal sigma sampling (best with Karras schedule)
    #[value(alias = "edm", alias = "edm-lognormal", alias = "log-sigma")]
    EdmLogNormal,
}

impl From<DiffusionTimestepCli> for DiffusionTimestepStrategy {
    fn from(arg: DiffusionTimestepCli) -> Self {
        match arg {
            DiffusionTimestepCli::Uniform => DiffusionTimestepStrategy::Uniform,
            DiffusionTimestepCli::MinSnr => DiffusionTimestepStrategy::MinSnr,
            DiffusionTimestepCli::EdmLogNormal => DiffusionTimestepStrategy::EdmLogNormal,
        }
    }
}
