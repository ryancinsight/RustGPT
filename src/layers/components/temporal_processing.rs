//! Shared Temporal Processing Component
//!
//! This component provides a unified interface for temporal processing
//! (attention, RG-LRU, Mamba) that can be used by multiple architectures.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{errors::Result, layers::components::common::TemporalMixingLayer, network::Layer};

/// Shared temporal processing component
#[derive(Serialize, Deserialize, Debug)]
pub struct SharedTemporalProcessing {
    /// The underlying temporal mixing layer
    pub temporal_mixing: TemporalMixingLayer,
    /// Window size for attention-based mixing
    pub window_size: Option<usize>,
    /// Use adaptive window sizing
    pub use_adaptive_window: bool,
}

impl SharedTemporalProcessing {
    /// Create a new shared temporal processing component
    pub fn new(
        temporal_mixing: TemporalMixingLayer,
        window_size: Option<usize>,
        use_adaptive_window: bool,
    ) -> Self {
        Self {
            temporal_mixing,
            window_size,
            use_adaptive_window,
        }
    }

    /// Forward pass through the temporal processing layer
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Set window size if using adaptive window and it's attention-based
        if self.use_adaptive_window
            && let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing
            && let Some(window_size) = self.window_size
        {
            attn.set_window_size(Some(window_size));
        }

        // Forward through the underlying layer
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.forward(input),
            TemporalMixingLayer::RgLru(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba2(layer) => layer.forward(input),
            TemporalMixingLayer::RgLruMoH(layer) => layer.forward(input),
            TemporalMixingLayer::MambaMoH(layer) => layer.forward(input),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.forward(input),
            TemporalMixingLayer::Titans(layer) => layer.forward(input),
        }
    }

    /// Backward pass through the temporal processing layer
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLru(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba2(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLruMoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::MambaMoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Titans(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    /// Apply gradients to the temporal processing layer
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::RgLru(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Mamba(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Mamba2(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::RgLruMoH(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::MambaMoH(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Titans(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    /// Get the number of parameters
    pub fn parameters(&self) -> usize {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.parameters(),
            TemporalMixingLayer::RgLru(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba2(layer) => layer.parameters(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.parameters(),
            TemporalMixingLayer::MambaMoH(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.parameters(),
            TemporalMixingLayer::Titans(layer) => layer.parameters(),
        }
    }

    /// Get the weight norm
    pub fn weight_norm(&self) -> f32 {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLru(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba2(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::MambaMoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.weight_norm(),
            TemporalMixingLayer::Titans(layer) => layer.weight_norm(),
        }
    }

    /// Zero out gradients
    pub fn zero_gradients(&mut self) {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.zero_gradients(),
            TemporalMixingLayer::RgLru(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Mamba(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Mamba2(layer) => layer.zero_gradients(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.zero_gradients(),
            TemporalMixingLayer::MambaMoH(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Mamba2MoH(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Titans(layer) => layer.zero_gradients(),
        }
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(_) => "Attention",
            TemporalMixingLayer::RgLru(_) => "RG-LRU",
            TemporalMixingLayer::Mamba(_) => "Mamba",
            TemporalMixingLayer::Mamba2(_) => "Mamba2",
            TemporalMixingLayer::RgLruMoH(_) => "RG-LRU-MoH",
            TemporalMixingLayer::MambaMoH(_) => "Mamba-MoH",
            TemporalMixingLayer::Mamba2MoH(_) => "Mamba2-MoH",
            TemporalMixingLayer::Titans(_) => "TitansMAC",
        }
    }

    /// Set window size for attention-based temporal mixing
    pub fn set_window_size(&mut self, window_size: Option<usize>) {
        self.window_size = window_size;
        if let TemporalMixingLayer::Attention(layer) = &mut self.temporal_mixing {
            layer.set_window_size(window_size);
        }
    }

    /// Get head activity metrics if available (for attention-based mixing)
    pub fn get_head_activity_metrics(&self) -> (Option<f32>, Option<&[f32]>) {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                let ratio = if let Some(avg) = attn.last_avg_active_heads {
                    let num_heads = attn.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                };
                (ratio, attn.last_head_activity_vec.as_deref())
            }
            TemporalMixingLayer::RgLruMoH(rglru) => {
                let ratio = if let Some(avg) = rglru.last_avg_active_heads {
                    let num_heads = rglru.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                };
                (ratio, rglru.last_head_activity_vec.as_deref())
            }
            TemporalMixingLayer::MambaMoH(m) => {
                let ratio = if let Some(avg) = m.last_avg_active_heads {
                    let num_heads = m.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                };
                (ratio, m.last_head_activity_vec.as_deref())
            }
            TemporalMixingLayer::Mamba2MoH(m) => {
                let ratio = if let Some(avg) = m.last_avg_active_heads {
                    let num_heads = m.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                };
                (ratio, m.last_head_activity_vec.as_deref())
            }
            TemporalMixingLayer::Titans(mac) => {
                let ratio = if let Some(avg) = mac.core.last_avg_active_heads {
                    let num_heads = mac.core.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                };
                (ratio, mac.core.last_head_activity_vec.as_deref())
            }
            _ => (Some(1.0), None),
        }
    }

    pub fn get_token_head_activity_vec(&self) -> Option<&[f32]> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => attn.last_token_head_activity_vec.as_deref(),
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.last_token_head_activity_vec.as_deref(),
            TemporalMixingLayer::MambaMoH(m) => m.last_token_head_activity_vec.as_deref(),
            TemporalMixingLayer::Mamba2MoH(m) => m.last_token_head_activity_vec.as_deref(),
            TemporalMixingLayer::Titans(mac) => mac.core.last_token_head_activity_vec.as_deref(),
            _ => None,
        }
    }

    /// Get window entropy metrics if available (for attention-based mixing)
    pub fn get_window_entropy(&self) -> Option<f32> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                if let Some((tmin, tmax)) = attn.last_tau_metrics {
                    let tau_span = (tmax - tmin).abs().max(0.0);
                    let pred_rms = attn.last_pred_norm.unwrap_or(0.0).max(0.0);
                    Some((0.7 * tau_span + 0.3 * pred_rms).clamp(0.0, 1.0))
                } else {
                    Some(0.0)
                }
            }
            _ => None,
        }
    }
}
