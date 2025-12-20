pub mod adam;
pub mod attention;
pub mod cli;
pub mod config_builder;
pub mod evaluator;
pub mod inference;
pub mod interactive;
pub mod layers;
pub mod network;
pub mod persistence;
pub mod rng;
pub mod trainer;
pub mod training;

pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod loss;
pub mod metrics;
pub mod pade;
pub mod richards;
pub mod soft;

// removed: pub mod head_router;
pub mod llm;

pub mod mixtures;
pub mod model_builder;
pub mod model_config;
pub mod model_persistence;
pub mod output_projection;

pub mod decoding;
pub mod encoding;

// Define crate-level constants used across modules
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
pub const MAX_SEQ_LEN: usize = 256;
pub const MAX_VOCAB_SIZE: usize = 50_000;
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB
pub const MAX_INPUT_LENGTH: usize = 10_000;
pub const GRADIENT_ANOMALY_THRESHOLD: f32 = 5000.0;

// Re-export key structs for easier access
pub use adam::Adam;
pub use dataset_loader::{Dataset, DatasetType};
// Also re-export decoding types for convenience
pub use decoding::GreedyDecoder;
pub use embeddings::TokenEmbeddings as Embeddings;
// Also re-export encoding types for convenience
pub use encoding::{SimpleTokenizer, Vocab};
pub use errors::{ModelError, Result};
// TRM is implemented via the recursive layer(s) under `layers::recurrence`.

// Re-export core LLM functionality
pub use evaluator::Evaluator;
pub use inference::InferenceEngine;
pub use llm::LLM;
// removed head_router re-exports
// pub use head_router::{RouterType, FullyAdaptiveHeadRouter};
// Also re-export mixture types for convenience
pub use mixtures::{
    ExpertRouter, ExpertRouterConfig, HeadSelectionConfig, HeadSelectionStrategy, MixtureOfExperts,
    ThresholdPredictor,
};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{ArchitectureType, AttentionType, ModelConfig, WindowAdaptationStrategy};
pub use network::{Layer, LayerEnum};
pub use persistence::ModelPersistence;
// Also re-export RichardsGlu
pub use richards::RichardsGlu;
// Also re-export RichardsNorm as DynamicTanhNorm for compatibility
pub use richards::RichardsNorm as DynamicTanhNorm;
pub use rng::{get_rng, get_seed, is_seeded, set_seed};
pub use trainer::Trainer;

// TRM tests removed
