use serde::{Deserialize, Serialize};

/// Configuration for speculative sampling
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct SpeculativeSamplingConfig {
    /// Number of speculative steps to take (gamma)
    pub gamma: usize,
    /// Acceptance threshold (tau) - interpretation depends on the sampler (MSE for diffusion, probability for AR)
    pub tau: f32,
    /// Number of layers in the draft model (if applicable/configurable)
    pub draft_layers: usize,
}

/// Speculative sampling mode - determines which type of model uses speculative sampling
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum SpeculativeMode {
    /// Speculative sampling for diffusion models (existing implementation)
    Diffusion,
    /// Speculative sampling for transformer models (new implementation)
    Transformer,
}

impl Default for SpeculativeMode {
    fn default() -> Self {
        SpeculativeMode::Diffusion
    }
}

impl Default for SpeculativeSamplingConfig {
    fn default() -> Self {
        Self {
            gamma: 4,
            tau: 0.01,
            draft_layers: 2,
        }
    }
}

/// Trait for models that support speculative sampling
pub trait SpeculativeSampler<DraftModel, Input, Output> {
    /// Perform speculative sampling using a draft model
    fn speculative_sample(
        &mut self,
        draft: &mut DraftModel,
        input: &Input,
        config: &SpeculativeSamplingConfig,
    ) -> Output;
}
