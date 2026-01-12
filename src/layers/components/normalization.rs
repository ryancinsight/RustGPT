//! Shared Normalization Component
//!
//! This component provides a unified normalization interface that can be used
//! by multiple architectures (Transformer, Diffusion, SSM).

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{network::Layer, richards::RichardsNorm};

/// Shared normalization component
#[derive(Serialize, Deserialize, Debug)]
pub struct SharedNormalization {
    /// The underlying Richards normalization layer
    pub norm: RichardsNorm,
}

impl SharedNormalization {
    /// Create a new shared normalization component
    pub fn new(embed_dim: usize) -> Self {
        Self {
            norm: RichardsNorm::new(embed_dim),
        }
    }

    /// Forward pass through the normalization layer
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        Layer::forward(&mut self.norm, input)
    }

    /// Backward pass through the normalization layer
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.norm.compute_gradients(input, output_grads)
    }

    /// Apply gradients to the normalization layer
    pub fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        self.norm.apply_gradients(param_grads, lr)
    }

    /// Get the number of parameters
    pub fn parameters(&self) -> usize {
        self.norm.parameters()
    }

    /// Get the weight norm
    pub fn weight_norm(&self) -> f32 {
        self.norm.weight_norm()
    }

    /// Zero out gradients
    pub fn zero_gradients(&mut self) {
        // RichardsNorm doesn't have gradients to zero in the current implementation
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        "RichardsNorm"
    }

    /// Get normalization statistics
    pub fn get_statistics(&self) -> (f32, f32) {
        // In a full implementation, this would return mean/variance statistics
        (0.0, 1.0)
    }

    /// Reset normalization statistics
    pub fn reset_statistics(&mut self) {
        // In a full implementation, this would reset running statistics
    }
}
