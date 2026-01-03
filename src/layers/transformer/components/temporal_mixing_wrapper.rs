//! Temporal Mixing Wrapper Component
//!
//! Wraps temporal mixing layers (attention, RG-LRU, Mamba) with additional functionality.
//! Handles window size management and head activity tracking.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{layers::components::common::TemporalMixingLayer, network::Layer};

/// Temporal mixing wrapper component
#[derive(Serialize, Deserialize, Debug)]
pub struct TemporalMixingWrapper {
    pub temporal_mixing: TemporalMixingLayer,
}

impl TemporalMixingWrapper {
    pub fn new(temporal_mixing: TemporalMixingLayer) -> Self {
        Self { temporal_mixing }
    }

    /// Forward pass through the temporal mixing layer
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => Layer::forward(layer, input),
            TemporalMixingLayer::RgLru(layer) => Layer::forward(layer, input),
            TemporalMixingLayer::Mamba(layer) => Layer::forward(layer, input),
            TemporalMixingLayer::Mamba2(layer) => Layer::forward(layer, input),
            TemporalMixingLayer::RgLruMoH(layer) => Layer::forward(layer, input),
        }
    }

    /// Set window size for attention-based temporal mixing
    pub fn set_window_size(&mut self, window_size: Option<usize>) {
        if let TemporalMixingLayer::Attention(layer) = &mut self.temporal_mixing {
            layer.set_window_size(window_size);
        }
    }

    /// Get head activity ratio from attention layer
    pub fn get_head_activity_ratio(&self) -> Option<f32> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                if let Some(avg) = attn.last_avg_active_heads {
                    let num_heads = attn.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                }
            }
            TemporalMixingLayer::RgLruMoH(rglru) => {
                if let Some(avg) = rglru.last_avg_active_heads {
                    let num_heads = rglru.num_heads as f32;
                    Some((avg / num_heads.max(1.0)).clamp(0.0, 1.0))
                } else {
                    Some(1.0)
                }
            }
            _ => Some(1.0),
        }
    }

    /// Get head activity vector from attention layer
    pub fn get_head_activity_vec(&self) -> Option<&[f32]> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => attn.last_head_activity_vec.as_deref(),
            TemporalMixingLayer::RgLruMoH(rglru) => rglru.last_head_activity_vec.as_deref(),
            _ => None,
        }
    }

    /// Get window entropy EMA from attention layer
    pub fn get_window_entropy_ema(&self) -> Option<f32> {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(attn) => {
                let (tau_span, pred_rms) = if let Some((tmin, tmax)) = attn.last_tau_metrics {
                    let tau_span = (tmax - tmin).abs().max(0.0);
                    let pred_rms = attn.last_pred_norm.unwrap_or(0.0).max(0.0);
                    (tau_span, pred_rms)
                } else {
                    (0.0, 0.0)
                };
                Some((0.7 * tau_span + 0.3 * pred_rms).clamp(0.0, 1.0))
            }
            _ => None,
        }
    }

    /// Backward pass through the temporal mixing layer
    pub fn backward(&mut self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLru(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::Mamba2(layer) => layer.compute_gradients(input, output_grads),
            TemporalMixingLayer::RgLruMoH(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    /// Apply gradients to the temporal mixing layer
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> crate::errors::Result<()> {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::RgLru(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Mamba(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::Mamba2(layer) => layer.apply_gradients(param_grads, lr),
            TemporalMixingLayer::RgLruMoH(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    /// Get the number of parameters in the temporal mixing layer
    pub fn parameters(&self) -> usize {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.parameters(),
            TemporalMixingLayer::RgLru(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba(layer) => layer.parameters(),
            TemporalMixingLayer::Mamba2(layer) => layer.parameters(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.parameters(),
        }
    }

    /// Get the weight norm of the temporal mixing layer
    pub fn weight_norm(&self) -> f32 {
        match &self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLru(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba(layer) => layer.weight_norm(),
            TemporalMixingLayer::Mamba2(layer) => layer.weight_norm(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.weight_norm(),
        }
    }

    /// Zero out the gradients in the temporal mixing layer
    pub fn zero_gradients(&mut self) {
        match &mut self.temporal_mixing {
            TemporalMixingLayer::Attention(layer) => layer.zero_gradients(),
            TemporalMixingLayer::RgLru(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Mamba(layer) => layer.zero_gradients(),
            TemporalMixingLayer::Mamba2(layer) => layer.zero_gradients(),
            TemporalMixingLayer::RgLruMoH(layer) => layer.zero_gradients(),
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
        }
    }
}