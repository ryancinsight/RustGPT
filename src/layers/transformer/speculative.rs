use std::{
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

use serde::{Deserialize, Serialize};

/// Configuration for speculative sampling
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct SpeculativeSamplingConfig {
    /// Number of speculative steps to take (gamma)
    pub gamma: usize,
    /// Acceptance threshold (tau) - interpretation depends on the sampler (MSE for diffusion,
    /// probability for AR)
    pub tau: f32,
    /// Number of layers in the draft model (if applicable/configurable)
    pub draft_layers: usize,
    /// Temperature for sampling (1.0 = no modification, < 1.0 = sharper, > 1.0 = softer)
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Nucleus sampling threshold (top-p). Set to 1.0 to disable.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
}

fn default_temperature() -> f32 {
    1.0
}
fn default_top_p() -> f32 {
    1.0
}

/// Speculative sampling mode - determines which type of model uses speculative sampling
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
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

impl fmt::Display for SpeculativeMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeculativeMode::Diffusion => write!(f, "Diffusion"),
            SpeculativeMode::Transformer => write!(f, "Transformer"),
        }
    }
}

impl Default for SpeculativeSamplingConfig {
    fn default() -> Self {
        Self {
            gamma: 4,
            tau: 0.01,
            draft_layers: 2,
            temperature: 1.0,
            top_p: 1.0,
        }
    }
}

impl SpeculativeSamplingConfig {
    /// Create a new config with the given parameters
    pub fn new(gamma: usize, tau: f32, draft_layers: usize) -> Self {
        Self {
            gamma: gamma.max(1),
            tau: tau.max(1e-6),
            draft_layers: draft_layers.max(1),
            temperature: 1.0,
            top_p: 1.0,
        }
    }

    /// Set sampling temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature.max(0.01);
        self
    }

    /// Set nucleus sampling threshold (top-p)
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p.clamp(0.0, 1.0);
        self
    }

    /// Get a description string for display
    pub fn description(&self) -> String {
        format!(
            "γ={}, τ={:.4}, layers={}",
            self.gamma, self.tau, self.draft_layers
        )
    }
}

impl fmt::Display for SpeculativeSamplingConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Speculative({})", self.description())
    }
}

/// Statistics tracker for speculative decoding performance
#[derive(Debug, Default)]
pub struct SpeculativeStats {
    /// Total tokens generated
    total_tokens: AtomicUsize,
    /// Tokens accepted from draft model
    accepted_tokens: AtomicUsize,
    /// Tokens rejected (fell back to target model)
    rejected_tokens: AtomicUsize,
    /// Total draft tokens proposed
    draft_proposals: AtomicUsize,
}

impl SpeculativeStats {
    /// Create a new stats tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a token generation event
    pub fn record_token(&self, accepted: bool) {
        self.total_tokens.fetch_add(1, Ordering::Relaxed);
        if accepted {
            self.accepted_tokens.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rejected_tokens.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record draft proposals
    pub fn record_draft_proposals(&self, count: usize) {
        self.draft_proposals.fetch_add(count, Ordering::Relaxed);
    }

    /// Get acceptance rate (0.0 to 1.0)
    pub fn acceptance_rate(&self) -> f32 {
        let total = self.total_tokens.load(Ordering::Relaxed);
        let accepted = self.accepted_tokens.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            accepted as f32 / total as f32
        }
    }

    /// Get total tokens generated
    pub fn total_tokens(&self) -> usize {
        self.total_tokens.load(Ordering::Relaxed)
    }

    /// Get accepted token count
    pub fn accepted_tokens(&self) -> usize {
        self.accepted_tokens.load(Ordering::Relaxed)
    }

    /// Get rejected token count
    pub fn rejected_tokens(&self) -> usize {
        self.rejected_tokens.load(Ordering::Relaxed)
    }

    /// Get draft proposal count
    pub fn draft_proposals(&self) -> usize {
        self.draft_proposals.load(Ordering::Relaxed)
    }

    /// Reset all statistics
    pub fn reset(&self) {
        self.total_tokens.store(0, Ordering::Relaxed);
        self.accepted_tokens.store(0, Ordering::Relaxed);
        self.rejected_tokens.store(0, Ordering::Relaxed);
        self.draft_proposals.store(0, Ordering::Relaxed);
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Speculative Stats: {} total, {} accepted, {} rejected, {:.1}% acceptance rate",
            self.total_tokens(),
            self.accepted_tokens(),
            self.rejected_tokens(),
            self.acceptance_rate() * 100.0
        )
    }
}

impl Clone for SpeculativeStats {
    fn clone(&self) -> Self {
        Self {
            total_tokens: AtomicUsize::new(self.total_tokens.load(Ordering::Relaxed)),
            accepted_tokens: AtomicUsize::new(self.accepted_tokens.load(Ordering::Relaxed)),
            rejected_tokens: AtomicUsize::new(self.rejected_tokens.load(Ordering::Relaxed)),
            draft_proposals: AtomicUsize::new(self.draft_proposals.load(Ordering::Relaxed)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_builder() {
        let config = SpeculativeSamplingConfig::new(8, 0.05, 3)
            .with_temperature(0.8)
            .with_top_p(0.9);

        assert_eq!(config.gamma, 8);
        assert!((config.tau - 0.05).abs() < 1e-6);
        assert_eq!(config.draft_layers, 3);
        assert!((config.temperature - 0.8).abs() < 1e-6);
        assert!((config.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_speculative_config_clamps_invalid() {
        let config = SpeculativeSamplingConfig::new(0, -1.0, 0);

        assert_eq!(config.gamma, 1);
        assert!(config.tau >= 1e-6);
        assert_eq!(config.draft_layers, 1);
    }

    #[test]
    fn test_speculative_stats() {
        let stats = SpeculativeStats::new();

        stats.record_token(true);
        stats.record_token(true);
        stats.record_token(false);

        assert_eq!(stats.total_tokens(), 3);
        assert_eq!(stats.accepted_tokens(), 2);
        assert_eq!(stats.rejected_tokens(), 1);
        assert!((stats.acceptance_rate() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_speculative_mode_display() {
        assert_eq!(format!("{}", SpeculativeMode::Transformer), "Transformer");
        assert_eq!(format!("{}", SpeculativeMode::Diffusion), "Diffusion");
    }

    #[test]
    fn test_speculative_config_display() {
        let config = SpeculativeSamplingConfig::new(4, 0.001, 2);
        let desc = format!("{}", config);
        assert!(desc.contains("Speculative"));
        assert!(desc.contains("γ=4"));
    }
}
