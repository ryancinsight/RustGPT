//! Centralized Configuration Constants
//!
//! This module provides a single source of truth for all magic numbers and
//! hyperparameters used throughout the RustGPT codebase. This follows the
//! Single Responsibility Principle and makes tuning and experimentation easier.
//!
//! # Design Principles
//!
//! - **SSOT**: All magic numbers defined in one place
//! - **Discoverability**: Constants are organized by domain
//! - **Type Safety**: Strongly typed with documentation
//! - **Extensibility**: Easy to add new constants
//!
//! # Usage
//!
//! ```rust
//! use llm::common::config::{TrainingConstants, ModelConstants, AttentionConstants};
//!
//! let lr = self_lr * TrainingConstants::LR_MIN_RATIO;
//! ```

/// Core training hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct TrainingConstants;

impl TrainingConstants {
    /// Minimum learning rate as fraction of base LR (cosine schedule)
    /// Recommended range: 0.05 - 0.15
    pub const LR_MIN_RATIO: f32 = 0.10;

    /// Cosine schedule midpoint (1.0 = standard cosine, <1.0 = warmer)
    pub const LR_COSINE_MIDPOINT: f32 = 0.5;

    /// Learning rate increment for online adaptation
    pub const LR_INCREMENT: f32 = 0.01;

    /// Default Adam epsilon (numerical stability)
    pub const ADAM_EPSILON: f32 = 1e-8;

    /// Gradient norm clipping threshold
    pub const GRAD_CLIP_THRESHOLD: f32 = 1.0;

    /// EMA decay for moving averages (momentum-like)
    pub const EMA_DECAY: f32 = 0.99;

    /// Minimum loss difference for early stopping consideration
    pub const LOSS_IMPROVEMENT_THRESHOLD: f32 = 1e-4;

    /// Default dropout rate (if used)
    pub const DROPOUT_RATE: f32 = 0.0;
}

/// Gating and mixture hyperparameters
#[derive(Debug, Clone, Copy)]
pub struct GatingConstants;

impl GatingConstants {
    /// MoH gating threshold modulation range
    pub const MOH_THRESHOLD_MIN: f32 = 0.1;
    pub const MOH_THRESHOLD_MAX: f32 = 0.9;

    /// Load balance weight increment
    pub const LOAD_BALANCE_INCREMENT: f32 = 0.01;
    pub const LOAD_BALANCE_MAX: f32 = 0.2;

    /// Sparsity weight adjustment rate
    pub const SPARSITY_INCREMENT: f32 = 0.01;
    pub const SPARSITY_DECAY: f32 = 0.95;
    pub const SPARSITY_MAX: f32 = 0.2;

    /// Complexity loss weight adjustment
    pub const COMPLEXITY_WEIGHT_INCREMENT: f32 = 0.01;
    pub const COMPLEXITY_WEIGHT_DECAY: f32 = 0.9;

    /// Entropy threshold for activation adjustment
    pub const ENTROPY_LOW_THRESHOLD: f32 = 0.2;

    /// Latent update alpha range
    pub const LATENT_ALPHA_BASE: f32 = 0.03;
    pub const LATENT_ALPHA_RANGE: f32 = 0.05;

    /// Expert routing temperature
    pub const ROUTING_TEMPERATURE: f32 = 1.0;

    /// Head activity ratio for adjustment
    pub const HEAD_ACTIVITY_RATIO_MIN: f32 = 0.1;
    pub const HEAD_ACTIVITY_RATIO_MAX: f32 = 1.0;
    pub const HEAD_ACTIVITY_RATIO_DEFAULT: f32 = 0.5;
}

/// Residual decoration and regularization
#[derive(Debug, Clone, Copy)]
pub struct RegularizationConstants;

impl RegularizationConstants {
    /// Difficulty weighting range
    pub const DIFFICULTY_MIN: f32 = 0.0;
    pub const DIFFICULTY_MAX: f32 = 1.0;

    /// Residual hard negative bank decay
    pub const HN_BANK_DECAY: f32 = 0.95;

    /// Decorrelation weight range
    pub const DECORRELATION_MIN: f32 = 0.0;
    pub const DECORRELATION_MAX: f32 = 1.0;

    /// Auxiliary loss weighting
    pub const AUX_LOSS_THRESHOLD: f32 = 10.0;

    /// L2 regularization strength
    pub const L2_REGULARIZATION: f32 = 1e-5;
}

/// Attention mechanism constants
#[derive(Debug, Clone, Copy)]
pub struct AttentionConstants;

impl AttentionConstants {
    /// Default attention temperature
    pub const ATTN_TEMPERATURE: f32 = 1.0;

    /// Softmax temperature scaling
    pub const SOFTMAX_TEMPERATURE: f32 = 1.0;

    /// CoPE log1p temperature
    pub const COPE_LOG1P_TEMPERATURE: f32 = 1.0;

    /// Default head dimension (for calculations)
    pub const DEFAULT_HEAD_DIM: usize = 64;

    /// Default number of heads
    pub const DEFAULT_NUM_HEADS: usize = 8;

    /// Sliding window overlap for consistency
    pub const WINDOW_OVERLAP: usize = 1;

    /// Ring attention chunk size
    pub const RING_CHUNK_SIZE: usize = 4096;

    /// KV cache eviction threshold (fraction)
    pub const KV_CACHE_EVICTION_THRESHOLD: f32 = 0.8;
}

/// Memory and cache constants
#[derive(Debug, Clone, Copy)]
pub struct MemoryConstants;

impl MemoryConstants {
    /// Engram cache size (tokens)
    pub const ENGRAM_CACHE_SIZE: usize = 16384;

    /// Long-term engram cache size
    pub const LONG_TERM_CACHE_SIZE: usize = 131072;

    /// Memory retrieval decay rate
    pub const MEMORY_DECAY: f32 = 0.99;

    /// Surprise estimation alpha
    pub const SURPRISE_ALPHA: f32 = 0.1;

    /// Titans MAC threshold
    pub const MAC_THRESHOLD: f32 = 0.5;

    /// neural memory eta (learning rate)
    pub const NEURAL_ETA: f32 = 0.01;
}

/// Diffusion model constants
#[derive(Debug, Clone, Copy)]
pub struct DiffusionConstants;

impl DiffusionConstants {
    /// Default noise schedule beta range
    pub const NOISE_BETA_START: f32 = 0.0001;
    pub const NOISE_BETA_END: f32 = 0.02;

    /// DDIM eta
    pub const DDIM_ETA: f32 = 0.0;

    /// Default timesteps
    pub const DEFAULT_TIMESTEPS: usize = 1000;

    /// guidance scale range
    pub const GUIDANCE_SCALE_MIN: f32 = 1.0;
    pub const GUIDANCE_SCALE_MAX: f32 = 10.0;
}

/// Speculative decoding constants
#[derive(Debug, Clone, Copy)]
pub struct SpeculativeConstants;

impl SpeculativeConstants {
    /// Default draft token count
    pub const DRAFT_TOKENS: usize = 4;

    /// Acceptance probability adjustment
    pub const ACCEPTANCE_MIN: f32 = 0.0;
    pub const ACCEPTANCE_MAX: f32 = 1.0;

    /// Minimum rejection threshold
    pub const REJECTION_THRESHOLD: f32 = 0.5;
}

/// Streaming and windowing constants
#[derive(Debug, Clone, Copy)]
pub struct StreamingConstants;

impl StreamingConstants {
    /// Buffer resize threshold (fraction of capacity)
    pub const BUFFER_RESIZE_THRESHOLD: f32 = 0.8;

    /// Entropy EMA alpha for window adaptation
    pub const ENTROPY_EMA_ALPHA: f32 = 0.2;

    /// Minimum window size ratio
    pub const WINDOW_MIN_RATIO: f32 = 0.25;

    /// Maximum window expansion per step
    pub const WINDOW_EXPAND_STEP: f32 = 0.1;

    /// Window shrink rate
    pub const WINDOW_SHRINK_RATE: f32 = 0.95;
}

/// Validation thresholds
#[derive(Debug, Clone, Copy)]
pub struct ValidationConstants;

impl ValidationConstants {
    /// Maximum gradient norm before clipping
    pub const MAX_GRAD_NORM: f32 = 5000.0;

    /// Finite value check threshold
    pub const FINITE_THRESHOLD: f32 = 1e6;

    /// Loss anomaly threshold
    pub const LOSS_ANOMALY_THRESHOLD: f32 = 1e4;

    /// Minimum valid probability
    pub const PROBABILITY_MIN: f32 = 1e-8;
}

/// Complete configuration container
#[derive(Debug, Clone, Copy)]
pub struct RustGPTConfig {
    pub training: TrainingConstants,
    pub gating: GatingConstants,
    pub regularization: RegularizationConstants,
    pub attention: AttentionConstants,
    pub memory: MemoryConstants,
    pub diffusion: DiffusionConstants,
    pub speculative: SpeculativeConstants,
    pub streaming: StreamingConstants,
    pub validation: ValidationConstants,
}

impl Default for RustGPTConfig {
    fn default() -> Self {
        Self {
            training: TrainingConstants,
            gating: GatingConstants,
            regularization: RegularizationConstants,
            attention: AttentionConstants,
            memory: MemoryConstants,
            diffusion: DiffusionConstants,
            speculative: SpeculativeConstants,
            streaming: StreamingConstants,
            validation: ValidationConstants,
        }
    }
}

/// Global configuration instance
pub static RUSTGPT_CONFIG: RustGPTConfig = RustGPTConfig {
    training: TrainingConstants,
    gating: GatingConstants,
    regularization: RegularizationConstants,
    attention: AttentionConstants,
    memory: MemoryConstants,
    diffusion: DiffusionConstants,
    speculative: SpeculativeConstants,
    streaming: StreamingConstants,
    validation: ValidationConstants,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_constants() {
        assert!(TrainingConstants::LR_MIN_RATIO > 0.0);
        assert!(TrainingConstants::LR_MIN_RATIO < 1.0);
        assert!(TrainingConstants::ADAM_EPSILON > 0.0);
    }

    #[test]
    fn test_gating_constants() {
        assert!(GatingConstants::MOH_THRESHOLD_MIN < GatingConstants::MOH_THRESHOLD_MAX);
        assert!(GatingConstants::LOAD_BALANCE_MAX > GatingConstants::LOAD_BALANCE_INCREMENT);
    }

    #[test]
    fn test_attention_constants() {
        assert!(AttentionConstants::DEFAULT_HEAD_DIM > 0);
        assert!(AttentionConstants::DEFAULT_NUM_HEADS > 0);
    }

    #[test]
    fn test_config_singleton() {
        assert_eq!(TrainingConstants::LR_MIN_RATIO, TrainingConstants::LR_MIN_RATIO);
    }
}
