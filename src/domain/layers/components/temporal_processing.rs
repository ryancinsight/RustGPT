//! Shared Temporal Processing Component
//!
//! This component provides a unified interface for temporal processing
//! (attention, RG-LRU, Mamba) that can be used by multiple architectures.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::Result,
    domain::{
        layers::components::common::TemporalMixingLayer,
        network::Layer,
    },
};

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
    /// 
    /// Uses the Layer trait for zero-cost abstraction, eliminating
    /// redundant match statements across all temporal mixing variants.
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Set window size if using adaptive window and it's attention-based
        if self.use_adaptive_window
            && let TemporalMixingLayer::Attention(attn) = &mut self.temporal_mixing
            && let Some(window_size) = self.window_size
        {
            attn.set_window_size(Some(window_size));
        }

        // Use Layer trait method - zero-cost abstraction
        self.temporal_mixing.forward(input)
    }

    /// Backward pass through the temporal processing layer
    /// 
    /// Uses compute_gradients from Layer trait for consistent
    /// gradient computation across all temporal mixing variants.
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.temporal_mixing.compute_gradients(input, output_grads)
    }

    /// Apply gradients to the temporal processing layer
    /// 
    /// Uses Layer trait method for zero-cost delegation.
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        self.temporal_mixing.apply_gradients(param_grads, lr)
    }

    /// Get the number of parameters
    /// 
    /// Uses Layer trait method for zero-cost delegation.
    pub fn parameters(&self) -> usize {
        self.temporal_mixing.parameters()
    }

    /// Get the weight norm
    /// 
    /// Uses Layer trait method for zero-cost delegation.
    pub fn weight_norm(&self) -> f32 {
        self.temporal_mixing.weight_norm()
    }

    /// Zero out gradients
    /// 
    /// Uses Layer trait method for zero-cost delegation.
    pub fn zero_gradients(&mut self) {
        self.temporal_mixing.zero_gradients()
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &'static str {
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

    /// Get head activity metrics if available (for MoH-based mixing)
    /// 
    /// Uses shared accessor pattern with type-specific field access.
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

    /// Get token head activity vector if available
    /// 
    /// Uses shared accessor pattern with zero-copy view returns.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::models::config::TemporalMixingType;

    #[test]
    fn test_shared_temporal_processing_layer_type() {
        // This test verifies that the SharedTemporalProcessing correctly
        // delegates to Layer trait methods
        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 16,
            hidden_dim: 32,
            num_heads: 4,
            poly_degree: 2,
            max_pos: 32,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let stp = SharedTemporalProcessing::new(
            layers.temporal_mixing,
            None,
            false,
        );

        assert_eq!(stp.layer_type(), "Attention");
        assert!(stp.parameters() > 0);
    }

    #[test]
    fn test_layer_trait_delegation() {
        // Test that Layer trait methods are correctly delegated
        let config = crate::domain::layers::components::common::CommonLayerConfig {
            embed_dim: 8,
            hidden_dim: 16,
            num_heads: 2,
            poly_degree: 2,
            max_pos: 16,
            window_size: None,
            use_moe: false,
            moe_config: None,
            head_selection: crate::domain::mixtures::HeadSelectionStrategy::Fixed { num_active: 2 },
            moh_threshold_modulation: crate::domain::richards::adaptive::AdaptiveScalar::default(),
            titan_memory: crate::domain::models::config::TitanMemoryConfig::default(),
            temporal_mixing: TemporalMixingType::Attention,
        };

        let layers = crate::domain::layers::components::common::CommonLayers::new(&config);
        let mut stp = SharedTemporalProcessing::new(
            layers.temporal_mixing,
            None,
            false,
        );

        // Test forward pass through Layer trait
        let input = Array2::zeros((2, 8));
        let output = stp.forward(&input);
        assert_eq!(output.dim(), (2, 8));

        // Test that parameters() returns consistent value
        let params = stp.parameters();
        assert!(params > 0);

        // Test weight_norm through Layer trait
        let norm = stp.weight_norm();
        assert!(norm >= 0.0);
    }
}
