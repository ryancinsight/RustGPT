pub use crate::domain::memory::EngramMemory;
pub use crate::domain::memory::titans::{NeuralMemory, TitansMAC, TitansMAG, TitansMAL, TitansMemory};

pub mod memory {
    pub use crate::domain::memory::EngramMemory;
    pub use crate::domain::memory::titans::{
        MemoryWeights, NeuralMemory, TitansMAC, TitansMAG, TitansMAL, TitansMemory,
    };
}
