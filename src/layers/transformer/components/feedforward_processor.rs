//! Feedforward Processor Component
//!
//! Handles feedforward network processing in transformer blocks.
//! Supports both RichardsGlu and MixtureOfExperts feedforward variants.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::layers::components::common::FeedForwardVariant;

/// Feedforward processor component
#[derive(Serialize, Deserialize, Debug)]
pub struct FeedforwardProcessor {
    feedforward: FeedForwardVariant,
}

impl FeedforwardProcessor {
    pub fn new(feedforward: FeedForwardVariant) -> Self {
        Self { feedforward }
    }

    /// Forward pass through the feedforward network
    /// Note: This is a simplified implementation for demonstration purposes
    /// In a full implementation, this would properly delegate to the underlying
    /// feedforward variant with proper trait objects or enum matching
    pub fn forward(&mut self, input: &Array2<f32>, _head_activity_ratio: Option<f32>, _head_activity_vec: Option<&[f32]>) -> Array2<f32> {
        // For now, implement a simple identity transformation
        // This demonstrates the modular structure while keeping the example simple
        input.clone()
    }

    /// Backward pass through the feedforward network
    /// Note: This is a simplified implementation for demonstration purposes
    pub fn backward(&mut self, input: &Array2<f32>, _output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        // For now, implement a simple identity gradient
        // In a full implementation, this would properly delegate to the underlying
        // feedforward variant with proper trait objects or enum matching
        (input.clone(), Vec::new())
    }

    /// Apply gradients to the feedforward network
    /// Note: This is a simplified implementation for demonstration purposes
    pub fn apply_gradients(&mut self, _param_grads: &[Array2<f32>], _lr: f32) -> crate::errors::Result<()> {
        // For now, implement a no-op
        // In a full implementation, this would properly delegate to the underlying
        // feedforward variant with proper trait objects or enum matching
        Ok(())
    }

    /// Get the number of parameters in the feedforward network
    pub fn parameters(&self) -> usize {
        self.feedforward.parameters()
    }

    /// Get the weight norm of the feedforward network
    pub fn weight_norm(&self) -> f32 {
        self.feedforward.weight_norm()
    }

    /// Zero out the gradients in the feedforward network
    /// Note: This is a simplified implementation for demonstration purposes
    pub fn zero_gradients(&mut self) {
        // For now, implement a no-op
        // In a full implementation, this would properly delegate to the underlying
        // feedforward variant with proper trait objects or enum matching
        match &mut self.feedforward {
            FeedForwardVariant::RichardsGlu(_layer) => {}
            FeedForwardVariant::MixtureOfExperts(_layer) => {}
        }
    }

    /// Get the layer type name
    pub fn layer_type(&self) -> &str {
        match &self.feedforward {
            FeedForwardVariant::RichardsGlu(_) => "RichardsGlu",
            FeedForwardVariant::MixtureOfExperts(_) => "MixtureOfExperts",
        }
    }

    /// Get head activity metrics if available
    pub fn get_head_activity_metrics(&self) -> (Option<f32>, Option<&[f32]>) {
        match &self.feedforward {
            FeedForwardVariant::MixtureOfExperts(_layer) => {
                // For MoE, we could return expert activity metrics
                (None, None)
            }
            _ => (None, None),
        }
    }
}