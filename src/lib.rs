pub mod adam;
pub mod attention;
pub mod cli;
pub mod config_builder;
pub mod evaluator;
pub mod interactive;
pub mod layers;
pub mod network;
pub mod rng;

pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod loss;
pub mod metrics;
pub mod pade;
pub mod richards;
pub mod soft;

pub mod mixtures;
pub mod model;
#[path = "model/builder.rs"]
pub mod model_builder;
#[path = "model/config.rs"]
pub mod model_config;
#[path = "model/persistence.rs"]
mod model_persistence;
pub mod output_projection;

pub mod decoding;
pub mod encoding;

// New Architecture Structure
pub mod inference;
pub mod models;
pub mod training;

pub mod eprop;

// Define crate-level constants used across modules
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
pub const MAX_SEQ_LEN: usize = 256;
pub const MAX_VOCAB_SIZE: usize = 50_000;
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB
pub const MAX_INPUT_LENGTH: usize = 10_000;
pub const GRADIENT_ANOMALY_THRESHOLD: f32 = 5000.0;

// Re-exports for backward compatibility
pub use adam::Adam;
pub use dataset_loader::{Dataset, DatasetType};
pub use decoding::GreedyDecoder;
pub use embeddings::TokenEmbeddings as Embeddings;
pub use encoding::{SimpleTokenizer, Vocab};
pub use errors::{ModelError, Result};
pub use evaluator::Evaluator;
pub use inference::engine::InferenceEngine; // adjusted path
pub use mixtures::{
    ExpertRouter, ExpertRouterConfig, HeadSelectionConfig, HeadSelectionStrategy, MixtureOfExperts,
    ThresholdPredictor,
};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{ArchitectureType, AttentionType, ModelConfig, WindowAdaptationStrategy};
// Keep module aliases if necessary
pub use models::llm;
pub use models::llm::LLM; // adjusted path
pub use network::{Layer, LayerEnum};
pub use richards::{RichardsGlu, RichardsNorm as DynamicTanhNorm};
pub use rng::{get_rng, get_seed, is_seeded, set_seed};
pub use training::trainer::Trainer; // adjusted path
