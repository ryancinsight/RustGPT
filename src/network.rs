use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    embeddings::TokenEmbeddings,
    layers::{
        recurrence::LRM,
        spiking::{AlifLayer, LifLayer},
        transformer::TransformerBlock,
    },
    memory::titans::NeuralMemory,
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

    DynamicTanhNorm(Box<RichardsNorm>),
    OutputProjection(OutputProjection),

    // Removed TRMBlock variant
    PolyAttention(Box<crate::attention::poly_attention::PolyAttention>),
    TransformerBlock(Box<TransformerBlock>),
    DiffusionBlock(Box<crate::layers::diffusion::DiffusionBlock>),
    LRM(Box<LRM>),
    TitansMemory(Box<NeuralMemory>),
    LifLayer(Box<LifLayer>),
    AlifLayer(Box<AlifLayer>),
}

/// Macro to reduce boilerplate in LayerEnum trait implementations
macro_rules! delegate_to_variant {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            LayerEnum::TokenEmbeddings(layer) => layer.$method($($arg),*),
            LayerEnum::RichardsGlu(layer) => layer.$method($($arg),*),
            LayerEnum::MixtureOfExperts(layer) => layer.$method($($arg),*),
            LayerEnum::DynamicTanhNorm(layer) => layer.$method($($arg),*),
            LayerEnum::OutputProjection(layer) => layer.$method($($arg),*),
            LayerEnum::PolyAttention(layer) => layer.$method($($arg),*),
            LayerEnum::TransformerBlock(layer) => layer.$method($($arg),*),
            LayerEnum::DiffusionBlock(layer) => layer.$method($($arg),*),
            LayerEnum::LRM(layer) => layer.$method($($arg),*),
            LayerEnum::TitansMemory(layer) => layer.$method($($arg),*),
            LayerEnum::LifLayer(layer) => layer.$method($($arg),*),
            LayerEnum::AlifLayer(layer) => layer.$method($($arg),*),
        }
    };
}

impl Layer for LayerEnum {
    fn layer_type(&self) -> &str {
        delegate_to_variant!(self, layer_type)
    }

    fn parameters(&self) -> usize {
        delegate_to_variant!(self, parameters)
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        delegate_to_variant!(self, forward, input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        delegate_to_variant!(self, backward, grads, lr)
    }

    fn weight_norm(&self) -> f32 {
        delegate_to_variant!(self, weight_norm)
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        delegate_to_variant!(self, compute_gradients, input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        delegate_to_variant!(self, apply_gradients, gradients, learning_rate)
    }

    fn zero_gradients(&mut self) {
        delegate_to_variant!(self, zero_gradients)
    }
}
