//! Normalization Layer Component
//!
//! Encapsulates the normalization functionality for transformer blocks.
//! Provides a clean interface for pre-attention and pre-feedforward normalization.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{network::Layer, richards::RichardsNorm};

/// Normalization layer component
#[derive(Serialize, Deserialize, Debug)]
pub struct NormalizationLayer {
    norm: RichardsNorm,
}

impl NormalizationLayer {
    pub fn new(norm: RichardsNorm) -> Self {
        Self { norm }
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

    /// Get the number of parameters in the normalization layer
    pub fn parameters(&self) -> usize {
        self.norm.parameters()
    }

    /// Get the weight norm of the normalization layer
    pub fn weight_norm(&self) -> f32 {
        self.norm.weight_norm()
    }

    /// Zero out the gradients in the normalization layer
    pub fn zero_gradients(&mut self) {
        // RichardsNorm doesn't have gradients to zero in the current implementation
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        "NormalizationLayer"
    }
}
