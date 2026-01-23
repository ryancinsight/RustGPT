use serde::{Deserialize, Serialize};

use super::neural::NeuralMemory;
use crate::attention::sliding_window_attention::SlidingWindowAttention;

/// Memory As Layer (MAL) Architecture
///
/// "Uses the neural Memory As a Layer (MAL) of a deep neural network."
/// Sequential: Memory -> Attention.
#[derive(Serialize, Deserialize, Debug)]
pub struct TitansMAL {
    pub memory: NeuralMemory,
    pub attention: SlidingWindowAttention,
}

use ndarray::Array2;

use crate::network::Layer;

impl TitansMAL {
    pub fn new(
        input_dim: usize,
        key_dim: usize,
        val_dim: usize,
        memory_hidden_dim: usize,
        window_size: usize,
    ) -> Self {
        Self {
            memory: NeuralMemory::new(input_dim, key_dim, val_dim, memory_hidden_dim),
            attention: SlidingWindowAttention::new(val_dim, window_size),
        }
    }

    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let memory_output = self.memory.forward(input);
        self.attention.forward(&memory_output)
    }
}
