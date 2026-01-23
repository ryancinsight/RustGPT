use serde::{Deserialize, Serialize};

use crate::{
    layers::diffusion::{DiffusionPredictionTarget, NoiseSchedule},
    mixtures::{moe::ExpertRouter, moh::HeadSelectionStrategy},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitanMemoryConfig {
    #[serde(default = "titan_memory_enabled_default")]
    pub enabled: bool,
    #[serde(default = "titan_memory_scale_default")]
    pub scale: f32,
    #[serde(default = "titan_memory_eta_default")]
    pub eta: f32,
    #[serde(default = "titan_memory_decay_default")]
    pub decay: f32,
    #[serde(default = "titan_memory_segment_len_default")]
    pub segment_len: usize,
    #[serde(default = "titan_memory_persistent_len_default")]
    pub persistent_len: usize,
    #[serde(default = "titan_memory_hidden_dim_default")]
    pub hidden_dim: usize,
    #[serde(default = "titan_memory_engram_enabled_default")]
    pub engram_enabled: bool,
    #[serde(default = "titan_memory_engram_scale_default")]
    pub engram_scale: f32,
    #[serde(default = "titan_memory_engram_ngram_order_default")]
    pub engram_ngram_order: usize,
    #[serde(default = "titan_memory_engram_num_heads_default")]
    pub engram_num_heads: usize,
}

impl Default for TitanMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: titan_memory_enabled_default(),
            scale: titan_memory_scale_default(),
            eta: titan_memory_eta_default(),
            decay: titan_memory_decay_default(),
            segment_len: titan_memory_segment_len_default(),
            persistent_len: titan_memory_persistent_len_default(),
            hidden_dim: titan_memory_hidden_dim_default(),
            engram_enabled: titan_memory_engram_enabled_default(),
            engram_scale: titan_memory_engram_scale_default(),
            engram_ngram_order: titan_memory_engram_ngram_order_default(),
            engram_num_heads: titan_memory_engram_num_heads_default(),
        }
    }
}

/// Architecture type for model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Autoregressive sequence model (Transformer-style residual stack).
    ///
    /// Important: the *temporal mixing* inside each block is configured separately via
    /// `temporal_mixing` (Attention/RG-LRU/Mamba/Mamba2). This variant describes the
    /// outer training/generation paradigm (next-token prediction), not the mixer.
    #[serde(alias = "Transformer")]
    Autoregressive,

    /// Tiny Recursive Model (LRM) - recursive reasoning with shared weights
    TRM,

    /// Diffusion Transformer - generative model using denoising diffusion process
    Diffusion,
}

/// Strategy for adapting sliding window size dynamically
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowAdaptationStrategy {
    /// Fixed window size (no adaptation)
    Fixed,

    /// Adapt based on sequence length: window_size = min(max, max(min, seq_len / 2))
    /// Simple and stable, scales window with input length
    SequenceLengthBased,
    /// Adapt based on attention entropy: larger windows when attention is diffuse
    /// More sophisticated, responds to attention patterns
    /// - Used in LLaMA, PaLM, GPT-NeoX, Mistral
    AttentionEntropy,

    /// Adapt based on prediction perplexity: larger windows when uncertain
    /// Most advanced, but requires perplexity computation
    PerplexityBased,
}

/// Attention mechanism selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard scaled dot-product self-attention
    SelfAttention,
    /// Polynomial attention layer with odd degree p (e.g., p=3)
    PolyAttention { degree_p: usize },
}

/// Temporal mixing mechanism selection (attention vs recurrent/SSM-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TemporalMixingType {
    /// Attention-based temporal mixing (default)
    #[default]
    Attention,
    /// Recurrent RG-LRU temporal mixing (Hawk/Griffin-style)
    RgLru,

    /// Mamba selective SSM (reference implementation)
    Mamba,

    /// Mamba-2 style selective SSM (reference implementation)
    Mamba2,

    /// Titans MAC (Memory As Context)
    Titans,
}

/// Strategy for sampling diffusion timesteps during training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffusionTimestepStrategy {
    /// Uniformly sample timesteps
    Uniform,
    /// Min-SNR weighting/sampling strategy
    MinSnr,
    /// EDM-style log-normal sigma sampling
    EdmLogNormal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of architecture to use
    pub architecture: ArchitectureType,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Hidden dimension for feedforward/channel mixing layers
    pub hidden_dim: usize,

    /// Number of transformer/hypermixer blocks
    pub num_layers: usize,

    /// Hidden dimension for hypernetwork (only used in HyperMixer)
    /// If None, defaults to embedding_dim / 4
    pub hypernetwork_hidden_dim: Option<usize>,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Number of attention heads for multi-head attention (used in both Transformer and
    /// HyperMixer) If None, defaults to 8 (same as standard transformers)
    pub num_heads: Option<usize>,

    /// Use DynamicTanhNorm for normalization
    /// Default: false (disabled by default)
    pub use_dynamic_tanh_norm: bool,

    /// Maximum position value for CoPE positional encoding
    /// Default: 64 (works well for context length 1024)
    pub cope_max_pos: usize,

    /// Number of key-value heads for Group-Query Attention (GQA)
    /// If None, uses standard Multi-Head Attention (MHA) with num_heads KV heads
    /// If Some(n), uses GQA with n KV heads shared across query heads
    /// Example: num_heads=8, num_kv_heads=Some(4) → 2 query heads per KV head
    /// Default: None (use MHA for backward compatibility)
    pub num_kv_heads: Option<usize>,

    /// Enable E-Prop (Eligibility Propagation) trace-based adaptation
    /// This adds an EPropAdaptor to each transformer block
    #[serde(default)]
    pub eprop_enabled: bool,

    /// Configuration for neurons used in E-Prop adaptor
    /// If None, defaults to LIF neurons
    #[serde(default)]
    pub eprop_neuron_config: Option<crate::eprop::config::NeuronConfig>,

    /// Sliding window size for attention (Sliding Window Attention)
    ///
    /// If None, uses full attention (all tokens attend to all previous tokens)
    /// If Some(w), each token only attends to the last w tokens (sliding window)
    /// Example: window_size=Some(4096) → Mistral 7B style (32k context efficient)
    ///
    /// Benefits:
    ///
    /// - Reduces attention complexity from O(N²) to O(N × window_size)
    /// - Enables longer context windows (32k+ tokens) efficiently
    /// - Minimal quality degradation (local context often sufficient)
    ///
    /// Default: None (use full attention for backward compatibility)
    pub window_size: Option<usize>,

    /// Enable adaptive window sizing (Phase 4 enhancement)
    ///
    /// If true, window size adapts dynamically based on the chosen strategy
    /// If false, uses fixed window_size (Phase 3 behavior)
    ///
    /// Default: false (use fixed window for backward compatibility)
    pub use_adaptive_window: bool,

    /// Minimum window size for adaptive window sizing
    ///
    /// Only used when use_adaptive_window = true
    /// Ensures window never shrinks below this value
    ///
    /// Default: 512 (reasonable minimum for most tasks)
    pub min_window_size: usize,

    /// Maximum window size for adaptive window sizing
    ///
    /// Only used when use_adaptive_window = true
    /// Ensures window never grows beyond this value
    ///
    /// Default: 4096 (Mistral 7B style)
    pub max_window_size: usize,

    /// Strategy for adapting window size
    ///
    /// Only used when use_adaptive_window = true
    /// Determines how window size changes based on context
    ///
    /// Default: SequenceLengthBased (simplest and most stable)
    pub window_adaptation_strategy: WindowAdaptationStrategy,

    #[serde(default = "entropy_ema_alpha_default_model")]
    pub entropy_ema_alpha: f32,

    /// Strategy for selecting which attention heads to activate
    ///
    /// Only `Learned` gating is supported: complexity-aware dynamic component selection
    /// where all heads are candidates and the number of active heads per token
    /// is determined by learned predictors.
    ///
    /// Default: `Learned` gating with adaptive component selection
    pub head_selection: HeadSelectionStrategy,

    /// Attention mechanism selection (SelfAttention vs PolyAttention)
    pub attention: AttentionType,

    /// Temporal mixing type selection (attention vs RG-LRU)
    #[serde(default)]
    pub temporal_mixing: TemporalMixingType,

    /// Enable Mixture-of-Experts (MoE) for feedforward layers
    ///
    /// When enabled, replaces standard feedforward layers with sparse MoE layers.
    /// Each MoE layer contains multiple expert networks with learned routing.
    ///
    /// Default: None (use standard feedforward)
    pub moe_router: Option<ExpertRouter>,

    #[serde(default)]
    pub titan_memory: TitanMemoryConfig,

    #[serde(default)]
    pub spiking_neuron_model: Option<crate::eprop::NeuronModel>,

    /// Use diffusion-conditioned blocks inside TRM when architecture=TRM
    pub trm_use_diffusion: bool,

    pub trm_num_recursions: Option<usize>,
    pub trm_max_supervision_steps: Option<usize>,
    pub trm_max_inference_steps: Option<usize>,
    pub trm_latent_update_alpha: Option<f32>,
    pub trm_latent_moh_enabled: Option<bool>,
    pub trm_latent_moh_top_p_min: Option<f32>,
    pub trm_latent_moh_top_p_max: Option<f32>,

    /// Target parameterization for diffusion blocks (ε vs v prediction)
    pub diffusion_prediction_target: DiffusionPredictionTarget,

    /// Min-SNR gamma cap used when weighting diffusion losses
    pub diffusion_min_snr_gamma: f32,

    /// Noise schedule controlling β_t across diffusion timesteps
    #[serde(default = "diffusion_noise_schedule_default")]
    pub diffusion_noise_schedule: NoiseSchedule,

    /// Strategy for sampling diffusion timesteps during training
    #[serde(default = "diffusion_timestep_strategy_default")]
    pub diffusion_timestep_strategy: DiffusionTimestepStrategy,

    /// Auxiliary residual decorrelation loss weight.
    ///
    /// This is a redundancy-reduction objective on residual streams (VICReg/Barlow-Twins style)
    /// that penalizes off-diagonal covariance of the hidden state right before the output
    /// projection.
    #[serde(default = "residual_decorrelation_weight_default")]
    pub residual_decorrelation_weight: f32,

    /// If true, increase decorrelation pressure on harder examples.
    #[serde(default = "residual_decorrelation_adaptive_default")]
    pub residual_decorrelation_adaptive: bool,

    /// Auxiliary hard-negative residual repulsion weight.
    #[serde(default = "residual_hardneg_weight_default")]
    pub residual_hardneg_weight: f32,

    /// If true, increase hard-negative pressure on harder examples.
    #[serde(default = "residual_hardneg_adaptive_default")]
    pub residual_hardneg_adaptive: bool,

    /// Number of hard negatives (top-k) to use.
    #[serde(default = "residual_hardneg_k_default")]
    pub residual_hardneg_k: usize,

    /// Cosine similarity margin.
    #[serde(default = "residual_hardneg_margin_default")]
    pub residual_hardneg_margin: f32,

    /// Temperature for hard-negative softplus penalty.
    #[serde(default = "residual_hardneg_temperature_default")]
    pub residual_hardneg_temperature: f32,

    /// Memory bank size.
    #[serde(default = "residual_hardneg_bank_size_default")]
    pub residual_hardneg_bank_size: usize,
}

impl ModelConfig {
    /// Create a new autoregressive configuration with modern defaults.
    ///
    /// Backward compatibility: `transformer(...)` remains as an alias.
    pub fn autoregressive(
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: Option<usize>,
        num_heads: Option<usize>,
    ) -> Self {
        Self::transformer(
            embedding_dim,
            hidden_dim,
            num_layers,
            max_seq_len,
            hypernetwork_hidden_dim,
            num_heads,
        )
    }

    /// Create a new Transformer configuration with modern defaults
    ///
    /// Note: this constructs an `ArchitectureType::Autoregressive` model.
    pub fn transformer(
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: Option<usize>,
        num_heads: Option<usize>,
    ) -> Self {
        let default_num_heads = num_heads.unwrap_or(8).max(1);
        Self {
            architecture: ArchitectureType::Autoregressive,
            embedding_dim,
            hidden_dim,
            num_layers,
            hypernetwork_hidden_dim,
            max_seq_len,
            num_heads,
            use_dynamic_tanh_norm: true, // Use DynamicTanhNorm
            cope_max_pos: 64,
            num_kv_heads: None,
            window_size: Some(16),
            use_adaptive_window: false,
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            entropy_ema_alpha: 0.2,
            head_selection: HeadSelectionStrategy::Learned {
                num_active: default_num_heads,
                load_balance_weight: 0.01,
                complexity_loss_weight: 0.005,
                sparsity_weight: 0.001,
                importance_loss_weight: 0.0,
                switch_balance_weight: 0.0,
                training_mode: crate::mixtures::gating::GatingTrainingMode::Coupled,
            },
            attention: AttentionType::SelfAttention,
            temporal_mixing: TemporalMixingType::Attention,
            moe_router: None, // Default: no MoE (standard feedforward)
            titan_memory: TitanMemoryConfig::default(),
            spiking_neuron_model: None,
            trm_use_diffusion: false,
            trm_num_recursions: None,
            trm_max_supervision_steps: None,
            trm_max_inference_steps: None,
            trm_latent_update_alpha: None,
            trm_latent_moh_enabled: Some(true),
            trm_latent_moh_top_p_min: Some(0.6),
            trm_latent_moh_top_p_max: Some(0.95),
            diffusion_prediction_target: DiffusionPredictionTarget::Epsilon,
            diffusion_min_snr_gamma: 3.0,
            diffusion_noise_schedule: NoiseSchedule::Cosine { s: 0.008 },
            diffusion_timestep_strategy: DiffusionTimestepStrategy::Uniform,
            residual_decorrelation_weight: residual_decorrelation_weight_default(),
            residual_decorrelation_adaptive: residual_decorrelation_adaptive_default(),
            residual_hardneg_weight: residual_hardneg_weight_default(),
            residual_hardneg_adaptive: residual_hardneg_adaptive_default(),
            residual_hardneg_k: residual_hardneg_k_default(),
            residual_hardneg_margin: residual_hardneg_margin_default(),
            residual_hardneg_temperature: residual_hardneg_temperature_default(),
            residual_hardneg_bank_size: residual_hardneg_bank_size_default(),
            eprop_enabled: false,
            eprop_neuron_config: None,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::transformer(128, 256, 3, 80, None, Some(4))
    }
}

// Provide serde default value for entropy_ema_alpha
fn entropy_ema_alpha_default_model() -> f32 {
    0.2
}

fn diffusion_noise_schedule_default() -> NoiseSchedule {
    NoiseSchedule::Cosine { s: 0.008 }
}

fn diffusion_timestep_strategy_default() -> DiffusionTimestepStrategy {
    DiffusionTimestepStrategy::Uniform
}

fn titan_memory_enabled_default() -> bool {
    true
}

fn titan_memory_scale_default() -> f32 {
    0.1
}

fn titan_memory_eta_default() -> f32 {
    0.2
}

fn titan_memory_decay_default() -> f32 {
    0.001
}

fn titan_memory_segment_len_default() -> usize {
    128
}

fn titan_memory_persistent_len_default() -> usize {
    32
}

fn titan_memory_hidden_dim_default() -> usize {
    64
}

fn titan_memory_engram_enabled_default() -> bool {
    true
}

fn titan_memory_engram_scale_default() -> f32 {
    0.05
}

fn titan_memory_engram_ngram_order_default() -> usize {
    3
}

fn titan_memory_engram_num_heads_default() -> usize {
    4
}

fn residual_decorrelation_weight_default() -> f32 {
    0.01
}

fn residual_decorrelation_adaptive_default() -> bool {
    true
}

fn residual_hardneg_weight_default() -> f32 {
    0.005
}

fn residual_hardneg_adaptive_default() -> bool {
    true
}

fn residual_hardneg_k_default() -> usize {
    8
}

fn residual_hardneg_margin_default() -> f32 {
    0.2
}

fn residual_hardneg_temperature_default() -> f32 {
    0.07
}

fn residual_hardneg_bank_size_default() -> usize {
    512
}

impl ModelConfig {
    pub fn get_num_heads(&self) -> usize {
        self.num_heads.unwrap_or(8)
    }

    pub fn get_num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.get_num_heads())
    }

    pub fn get_hypernetwork_hidden_dim(&self) -> usize {
        // Provide a reasonable default if not specified.
        self.hypernetwork_hidden_dim
            .unwrap_or(self.embedding_dim / 4)
    }

    pub fn get_recursive_depth(&self) -> usize {
        // In recursive models, num_layers stores the recursive depth
        self.num_layers
    }

    /// Get polynomial degree `p` for `PolyAttention`.
    /// Defaults to 3 if attention is not explicitly set to PolyAttention.
    pub fn get_poly_degree_p(&self) -> usize {
        match self.attention {
            AttentionType::PolyAttention { degree_p } => degree_p,
            _ => 3,
        }
    }
}
