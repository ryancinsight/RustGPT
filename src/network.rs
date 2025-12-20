use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    embeddings::TokenEmbeddings,
    layers::{recurrence::LRM, transformer::TransformerBlock},
    output_projection::OutputProjection,
    richards::{RichardsGlu, RichardsNorm},
};

/// Layer trait for neural network components
pub trait Layer {
    fn layer_type(&self) -> &str;
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
    fn parameters(&self) -> usize;
    /// Frobenius norm of all learnable weights in the layer
    /// Used by LARS trust-ratio to balance update magnitude
    fn weight_norm(&self) -> f32;
    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>);
    /// Apply gradients to layer parameters
    /// Returns GradientError if param_grads has incorrect length
    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()>;
    fn zero_gradients(&mut self);
}

/// Enumeration of all possible layer types in the network
#[derive(Serialize, Deserialize, Debug)]
pub enum LayerEnum {
    TokenEmbeddings(TokenEmbeddings),
    // Removed SelfAttention variant
    // Removed FeedForward variant; RichardsGlu is the only FFN
    RichardsGlu(Box<RichardsGlu>),
    MixtureOfExperts(Box<crate::mixtures::moe::MixtureOfExperts>),

    DynamicTanhNorm(RichardsNorm),
    OutputProjection(OutputProjection),

    // Removed TRMBlock variant
    PolyAttention(Box<crate::attention::poly_attention::PolyAttention>),
    TransformerBlock(Box<TransformerBlock>),
    DiffusionBlock(Box<crate::layers::diffusion::DiffusionBlock>),
    LRM(Box<LRM>),
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.layer_type(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.layer_type(),
            LayerEnum::MixtureOfExperts(layer) => layer.layer_type(),

            LayerEnum::DynamicTanhNorm(layer) => layer.layer_type(),
            LayerEnum::OutputProjection(layer) => layer.layer_type(),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.layer_type(),
            LayerEnum::TransformerBlock(layer) => layer.layer_type(),
            LayerEnum::DiffusionBlock(layer) => layer.layer_type(),
            LayerEnum::LRM(layer) => layer.layer_type(),
        }
    }

    fn parameters(&self) -> usize {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.parameters(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.parameters(),
            LayerEnum::MixtureOfExperts(layer) => layer.parameters(),

            LayerEnum::DynamicTanhNorm(layer) => layer.parameters(),
            LayerEnum::OutputProjection(layer) => layer.parameters(),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.parameters(),
            LayerEnum::TransformerBlock(layer) => layer.parameters(),
            LayerEnum::DiffusionBlock(layer) => layer.parameters(),
            LayerEnum::LRM(layer) => layer.parameters(),
        }
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.forward(input),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.forward(input),
            LayerEnum::MixtureOfExperts(layer) => layer.forward(input),

            LayerEnum::DynamicTanhNorm(layer) => layer.forward(input),
            LayerEnum::OutputProjection(layer) => layer.forward(input),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.forward(input),
            LayerEnum::TransformerBlock(layer) => layer.forward(input),
            LayerEnum::DiffusionBlock(layer) => layer.forward(input),
            LayerEnum::LRM(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.backward(grads, lr),
            LayerEnum::RichardsGlu(layer) => layer.backward(grads, lr),
            LayerEnum::MixtureOfExperts(layer) => layer.backward(grads, lr),
            LayerEnum::DynamicTanhNorm(layer) => layer.backward(grads, lr),
            LayerEnum::OutputProjection(layer) => layer.backward(grads, lr),
            LayerEnum::PolyAttention(layer) => layer.backward(grads, lr),
            LayerEnum::TransformerBlock(layer) => layer.backward(grads, lr),
            LayerEnum::DiffusionBlock(layer) => layer.backward(grads, lr),
            LayerEnum::LRM(layer) => layer.backward(grads, lr),
        }
    }

    fn weight_norm(&self) -> f32 {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.weight_norm(),
            LayerEnum::RichardsGlu(layer) => layer.weight_norm(),
            LayerEnum::MixtureOfExperts(layer) => layer.weight_norm(),
            LayerEnum::DynamicTanhNorm(layer) => layer.weight_norm(),
            LayerEnum::OutputProjection(layer) => layer.weight_norm(),
            LayerEnum::PolyAttention(layer) => layer.weight_norm(),
            LayerEnum::TransformerBlock(layer) => layer.weight_norm(),
            LayerEnum::DiffusionBlock(layer) => layer.weight_norm(),
            LayerEnum::LRM(layer) => layer.weight_norm(),
        }
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.compute_gradients(input, output_grads),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::MixtureOfExperts(layer) => layer.compute_gradients(input, output_grads),

            LayerEnum::DynamicTanhNorm(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::OutputProjection(layer) => layer.compute_gradients(input, output_grads),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::TransformerBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::DiffusionBlock(layer) => layer.compute_gradients(input, output_grads),
            LayerEnum::LRM(layer) => layer.compute_gradients(input, output_grads),
        }
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.apply_gradients(gradients, learning_rate),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.apply_gradients(gradients, learning_rate),
            LayerEnum::MixtureOfExperts(layer) => layer.apply_gradients(gradients, learning_rate),

            LayerEnum::DynamicTanhNorm(layer) => layer.apply_gradients(gradients, learning_rate),
            LayerEnum::OutputProjection(layer) => layer.apply_gradients(gradients, learning_rate),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.apply_gradients(gradients, learning_rate),
            LayerEnum::TransformerBlock(layer) => layer.apply_gradients(gradients, learning_rate),
            LayerEnum::DiffusionBlock(layer) => layer.apply_gradients(gradients, learning_rate),
            LayerEnum::LRM(layer) => layer.apply_gradients(gradients, learning_rate),
        }
    }

    fn zero_gradients(&mut self) {
        match self {
            LayerEnum::TokenEmbeddings(layer) => layer.zero_gradients(),
            // Removed SelfAttention arm
            // Removed FeedForward arm
            LayerEnum::RichardsGlu(layer) => layer.zero_gradients(),
            LayerEnum::MixtureOfExperts(layer) => layer.zero_gradients(),

            LayerEnum::DynamicTanhNorm(layer) => layer.zero_gradients(),
            LayerEnum::OutputProjection(layer) => layer.zero_gradients(),

            // Removed TRMBlock arm
            LayerEnum::PolyAttention(layer) => layer.zero_gradients(),
            LayerEnum::TransformerBlock(layer) => layer.zero_gradients(),
            LayerEnum::DiffusionBlock(layer) => layer.zero_gradients(),
            LayerEnum::LRM(layer) => layer.zero_gradients(),
        }
    }
}
