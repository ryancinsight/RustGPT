use clap::{Parser, ValueEnum};
use crate::{
    model_config::DiffusionTimestepStrategy,
    transformer::diffusion_block::{DiffusionPredictionTarget, NoiseSchedule},
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
}

/// CLI representation of diffusion prediction targets
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionTargetCli {
    #[value(alias = "eps")]
    Epsilon,
    #[value(alias = "v", alias = "vpred")]
    VPrediction,
}

impl From<DiffusionTargetCli> for DiffusionPredictionTarget {
    fn from(arg: DiffusionTargetCli) -> Self {
        match arg {
            DiffusionTargetCli::Epsilon => DiffusionPredictionTarget::Epsilon,
            DiffusionTargetCli::VPrediction => DiffusionPredictionTarget::VPrediction,
        }
    }
}

/// CLI representation of diffusion noise schedules
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionScheduleCli {
    Cosine,
    Linear,
    Quadratic,
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
        }
    }
}

/// CLI representation of diffusion timestep strategies
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum DiffusionTimestepCli {
    Uniform,
    #[value(alias = "minsnr", alias = "min-snr")]
    MinSnr,
}

impl From<DiffusionTimestepCli> for DiffusionTimestepStrategy {
    fn from(arg: DiffusionTimestepCli) -> Self {
        match arg {
            DiffusionTimestepCli::Uniform => DiffusionTimestepStrategy::Uniform,
            DiffusionTimestepCli::MinSnr => DiffusionTimestepStrategy::MinSnr,
        }
    }
}

