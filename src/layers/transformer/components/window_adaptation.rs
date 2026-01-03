//! Window Adaptation Component
//!
//! Handles dynamic window size adaptation for attention mechanisms.
//! This component encapsulates the complex logic for adjusting attention windows
//! based on different strategies (entropy, sequence length, etc.).

use serde::{Deserialize, Serialize};

use crate::{
    layers::components::common::TemporalMixingLayer,
    model_config::WindowAdaptationStrategy,
};

/// Window adaptation configuration
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct WindowAdaptationConfig {
    pub use_adaptive_window: bool,
    pub window_adaptation_strategy: WindowAdaptationStrategy,
    pub base_window_size: usize,
    pub min_window_size: usize,
    pub max_window_size: usize,
    pub entropy_ema_alpha: f32,
}

impl WindowAdaptationConfig {
    pub fn new(
        use_adaptive_window: bool,
        window_adaptation_strategy: WindowAdaptationStrategy,
        base_window_size: usize,
        min_window_size: usize,
        max_window_size: usize,
        entropy_ema_alpha: f32,
    ) -> Self {
        Self {
            use_adaptive_window,
            window_adaptation_strategy,
            base_window_size,
            min_window_size,
            max_window_size,
            entropy_ema_alpha,
        }
    }
}

/// Window adaptation state
#[derive(Serialize, Deserialize, Debug)]
pub struct WindowAdaptationState {
    window_entropy_ema: f32,
}

impl WindowAdaptationState {
    pub fn new() -> Self {
        Self {
            window_entropy_ema: 0.0,
        }
    }
}

/// Window adaptation component
#[derive(Serialize, Deserialize, Debug)]
pub struct WindowAdaptation {
    config: WindowAdaptationConfig,
    state: WindowAdaptationState,
}

impl WindowAdaptation {
    pub fn new(config: WindowAdaptationConfig) -> Self {
        Self {
            config,
            state: WindowAdaptationState::new(),
        }
    }

    /// Calculate the adaptive window size
    pub fn calculate_window_size(
        &mut self,
        seq_len: usize,
        temporal_mixing: &TemporalMixingLayer,
    ) -> usize {
        if !self.config.use_adaptive_window {
            return self.config.base_window_size.min(seq_len.max(1));
        }

        let min_w = self.config.min_window_size.max(1);
        let max_w = self.config.max_window_size.max(min_w);
        let base_w = self.config.base_window_size;

        // Adaptive window is attention-specific
        if !matches!(temporal_mixing, TemporalMixingLayer::Attention(_)) {
            return base_w.min(seq_len.max(1));
        }

        match self.config.window_adaptation_strategy {
            WindowAdaptationStrategy::Fixed => {
                base_w.min(seq_len.max(1))
            }
            WindowAdaptationStrategy::SequenceLengthBased => {
                let w = (seq_len / 2).max(min_w).min(max_w);
                w
            }
            WindowAdaptationStrategy::AttentionEntropy => {
                let (tau_span, pred_rms) = self.extract_attention_metrics(temporal_mixing);
                let signal = (0.7 * tau_span + 0.3 * pred_rms).clamp(0.0, 1.0);
                
                let alpha = self.config.entropy_ema_alpha.clamp(0.0, 1.0);
                self.state.window_entropy_ema = 
                    alpha * signal + (1.0 - alpha) * self.state.window_entropy_ema;
                
                let w = min_w as f32
                    + self.state.window_entropy_ema * (max_w.saturating_sub(min_w) as f32);
                w.round() as usize
            }
            WindowAdaptationStrategy::PerplexityBased => {
                base_w.min(seq_len.max(1))
            }
        }
        .min(seq_len.max(1))
        .clamp(min_w, max_w)
    }

    /// Extract attention metrics from temporal mixing layer
    fn extract_attention_metrics(
        &self,
        temporal_mixing: &TemporalMixingLayer,
    ) -> (f32, f32) {
        match temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                let tau_span = if let Some((tmin, tmax)) = attn.last_tau_metrics {
                    (tmax - tmin).abs().max(0.0)
                } else {
                    0.0
                };
                let pred_rms = attn.last_pred_norm.unwrap_or(0.0).max(0.0);
                (tau_span, pred_rms)
            }
            _ => (0.0, 0.0),
        }
    }

    /// Get the current window entropy EMA value
    pub fn window_entropy_ema(&self) -> f32 {
        self.state.window_entropy_ema
    }

    /// Reset the window adaptation state
    pub fn reset_state(&mut self) {
        self.state.window_entropy_ema = 0.0;
    }
}