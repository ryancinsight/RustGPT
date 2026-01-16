use serde::{Deserialize, Serialize};

/// Memory As Layer (MAL) Architecture
///
/// "Uses the neural Memory As a Layer (MAL) of a deep neural network."
/// Sequential: Memory -> Attention.
#[derive(Serialize, Deserialize, Debug)]
pub struct TitansMAL {
    // TODO: Contain NeuralMemory.
    // TODO: Contain Sliding Window Attention (SWA).
}

impl TitansMAL {
    // TODO: Implement forward logic:
    // 1. Input -> NeuralMemory -> y
    // 2. y -> SWA -> Output
}
