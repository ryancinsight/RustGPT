use std::num::NonZeroUsize;

use clap::{ArgAction, Parser, ValueEnum};

use crate::domain::{
    compute_backend::ComputeBackendPreference,
    layers::diffusion::{DiffusionPredictionTarget, NoiseSchedule},
    models::config::{DiffusionTimestepStrategy, TemporalMixingType},
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

    #[arg(long)]
    pub diffusion: bool,

    #[arg(long, default_value_t = 0.5)]
    pub diffusion_ce_weight: f32,

    /// Use adaptive (Richards curve) modulation for ce_weight.
    /// If enabled, ce_weight acts as the peak value, modulated by a sigmoid schedule.
    #[arg(long)]
    pub diffusion_ce_weight_adaptive: bool,

    /// Richards curve midpoint (0.0-1.0) for ce_weight adaptive modulation.
    #[arg(long, default_value_t = 0.5)]
    pub diffusion_ce_weight_curve_m: f32,

    /// Richards curve steepness for ce_weight adaptive modulation.
    #[arg(long, default_value_t = 5.0)]
    pub diffusion_ce_weight_curve_k: f32,

    #[arg(long, default_value_t = 3.0)]
    pub diffusion_min_snr_gamma: f32,

    /// Use adaptive (Richards curve) modulation for min_snr_gamma.
    /// If enabled, min_snr_gamma acts as the peak value, modulated by a sigmoid schedule.
    #[arg(long)]
    pub diffusion_min_snr_gamma_adaptive: bool,

    /// Richards curve midpoint (0.0-1.0) for min_snr_gamma adaptive modulation.
    #[arg(long, default_value_t = 0.5)]
    pub diffusion_min_snr_gamma_curve_m: f32,

    /// Richards curve steepness for min_snr_gamma adaptive modulation.
    #[arg(long, default_value_t = 5.0)]
    pub diffusion_min_snr_gamma_curve_k: f32,

    /// Base value for MoH threshold modulation (default: 1.0)
    #[arg(long, default_value_t = 1.0)]
    pub moh_threshold_modulation: f32,

    /// Use adaptive (Richards curve) modulation for MoH threshold.
    /// If enabled, threshold_modulation acts as the peak value, modulated by a sigmoid schedule.
    #[arg(long)]
    pub moh_threshold_modulation_adaptive: bool,

    /// Richards curve midpoint (0.0-1.0) for MoH threshold adaptive modulation.
    #[arg(long, default_value_t = 0.5)]
    pub moh_threshold_modulation_curve_m: f32,

    /// Richards curve steepness for MoH threshold adaptive modulation.
    #[arg(long, default_value_t = 5.0)]
    pub moh_threshold_modulation_curve_k: f32,

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

    /// Save a versioned model checkpoint every N epochs during training.
    #[arg(long)]
    pub save_every: Option<NonZeroUsize>,

    /// Directory where checkpoints are saved.
    #[arg(long, default_value = "models")]
    pub checkpoint_dir: String,

    /// Number of epochs to run during pre-training (default 100)
    #[arg(long, default_value_t = 100)]
    pub pretrain_epochs: usize,

    /// Number of epochs to run during instruction tuning (default 100)
    #[arg(long, default_value_t = 100)]
    pub instruction_epochs: usize,

    /// Automatically tune batch size and gradient accumulation from dataset size and available RAM.
    ///
    /// Explicit `--*-batch-size` / `--*-gradient-accumulation-steps` values always take priority.
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
    pub auto_tune_batching: bool,

    /// Optional memory budget (GiB) for auto-tuned training batch sizing.
    ///
    /// When set, auto-tuning will not assume more than this memory budget.
    #[arg(long)]
    pub memory_budget_gb: Option<f32>,

    /// Batch size for pre-training (standard transformer path)
    #[arg(long)]
    pub pretrain_batch_size: Option<usize>,

    /// Number of micro-batches to accumulate before each optimizer update during pre-training
    #[arg(long)]
    pub pretrain_gradient_accumulation_steps: Option<usize>,

    /// Batch size for instruction tuning (standard transformer path)
    #[arg(long)]
    pub instruction_batch_size: Option<usize>,

    /// Number of micro-batches to accumulate before each optimizer update during instruction tuning
    #[arg(long)]
    pub instruction_gradient_accumulation_steps: Option<usize>,

    /// Batch size for diffusion CE training (both pretrain and instruction stages)
    #[arg(long)]
    pub diffusion_batch_size: Option<usize>,

    /// Number of micro-batches to accumulate before each optimizer update during diffusion training
    #[arg(long)]
    pub diffusion_gradient_accumulation_steps: Option<usize>,

    /// Enable Mixture-of-Experts (MoE) for feedforward layers
    /// When enabled, replaces standard feedforward layers with sparse MoE layers
    /// Each MoE layer contains multiple expert networks with learned routing
    #[arg(long, default_value_t = true, action = ArgAction::Set)]
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

    /// Compute backend selection.
    ///
    /// `auto-gpu` prefers GPU and falls back to CPU only when no GPU is detected.
    /// If a GPU is detected but not compiled in, startup fails so the GPU issue is visible.
    /// `npu` requires Intel NPU adapter selection via WGPU/Vulkan (no fallback).
    #[arg(long, value_enum, default_value_t = ComputeBackendCli::AutoGpu)]
    pub compute_backend: ComputeBackendCli,

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

    /// Auxiliary hard-negative residual repulsion weight (cosine-based, memory-bank hard
    /// negatives).
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

    /// Enable continual learning during inference (online learning from user feedback)
    #[arg(long, default_value_t = false)]
    pub continual_learning: bool,

    /// User ID for continual learning personalization
    #[arg(long)]
    pub user_id: Option<String>,

    /// Learning rate for online updates during continual learning (typically smaller than training LR)
    #[arg(long, default_value_t = 1e-5)]
    pub online_learning_rate: f32,

    /// EWC (Elastic Weight Consolidation) regularization strength for continual learning (0 to disable)
    #[arg(long, default_value_t = 100.0)]
    pub ewc_lambda: f32,

    /// Maximum number of past interactions to store per user for continual learning
    #[arg(long, default_value_t = 1000)]
    pub max_user_memory_size: usize,

    /// Number of replay samples to use for each continual learning update
    #[arg(long, default_value_t = 32)]
    pub replay_buffer_size: usize,

    /// Path to save/load user memories for continual learning
    #[arg(long)]
    pub user_memories_path: Option<String>,
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

/// CLI representation of compute backend preferences.
#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ComputeBackendCli {
    /// Prefer automatic GPU detection; resolves to CPU only when no GPU backend is available.
    #[value(alias = "auto", alias = "autogpu")]
    AutoGpu,
    /// Force CPU execution.
    Cpu,
    /// Require CUDA backend.
    Cuda,
    /// Require Metal backend.
    Metal,
    /// Require Vulkan backend.
    Vulkan,
    /// Require Intel NPU through WGPU/Vulkan adapter selection.
    #[value(alias = "intel-npu", alias = "intel_npu")]
    Npu,
}

impl From<ComputeBackendCli> for ComputeBackendPreference {
    fn from(arg: ComputeBackendCli) -> Self {
        match arg {
            ComputeBackendCli::AutoGpu => ComputeBackendPreference::AutoGpu,
            ComputeBackendCli::Cpu => ComputeBackendPreference::Cpu,
            ComputeBackendCli::Cuda => ComputeBackendPreference::Cuda,
            ComputeBackendCli::Metal => ComputeBackendPreference::Metal,
            ComputeBackendCli::Vulkan => ComputeBackendPreference::Vulkan,
            ComputeBackendCli::Npu => ComputeBackendPreference::Npu,
        }
    }
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
