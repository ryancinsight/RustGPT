//! Shared Feedforward Component
//!
//! This component provides a unified feedforward interface that can be used
//! by multiple architectures (Transformer, Diffusion, SSM).

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{errors::Result, layers::components::common::FeedForwardVariant, network::Layer};

/// Shared feedforward component
#[derive(Serialize, Deserialize, Debug)]
pub struct SharedFeedforward {
    /// The underlying feedforward variant
    pub feedforward: FeedForwardVariant,
}

impl SharedFeedforward {
    /// Create a new shared feedforward component
    pub fn new(feedforward: FeedForwardVariant) -> Self {
        Self { feedforward }
    }

    /// Forward pass through the feedforward network
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.feedforward.forward(input)
    }

    /// Backward pass through the feedforward network
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            FeedForwardVariant::MixtureOfExperts(layer) => {
                layer.compute_gradients(input, output_grads)
            }
        }
    }

    /// Apply gradients to the feedforward network
    pub fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.apply_gradients(param_grads, lr),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.apply_gradients(param_grads, lr),
        }
    }

    /// Get the number of parameters
    pub fn parameters(&self) -> usize {
        self.feedforward.parameters()
    }

    /// Get the weight norm
    pub fn weight_norm(&self) -> f32 {
        self.feedforward.weight_norm()
    }

    /// Zero out gradients
    pub fn zero_gradients(&mut self) {
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(layer) => layer.zero_gradients(),
            FeedForwardVariant::MixtureOfExperts(layer) => layer.zero_gradients(),
        }
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => "RichardsGlu",
            FeedForwardVariant::MixtureOfExperts(_) => "MixtureOfExperts",
        }
    }
}
