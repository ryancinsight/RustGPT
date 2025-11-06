pub mod adam;

// pub mod cop; // removed: integrated CoPE into PolyAttention
pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod richards;

// removed: pub mod head_router;
pub mod llm;

pub mod model_builder;
pub mod model_config;
pub mod model_persistence;
pub mod output_projection;
// removed: pub mod sigmoid_poly;
// removed: pub mod routing;
// removed: pub mod self_attention;
pub mod mixtures;
pub mod poly_attention;

// removed: pub mod trm;
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
pub use embeddings::TokenEmbeddings as Embeddings;
pub use errors::{ModelError, Result};
// removed head_router re-exports
// pub use head_router::{RouterType, FullyAdaptiveHeadRouter};
pub use llm::{LLM, Layer, LayerEnum};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{
    ArchitectureType, AttentionType, ModelConfig, WindowAdaptationStrategy,
};
// Also re-export encoding types for convenience
pub use encoding::{Vocab, SimpleTokenizer};
// Also re-export decoding types for convenience
pub use decoding::{AutoDeco, AutoDecoConfig, GreedyDecoder, TemperatureHead, TopPHead};
// Also re-export mixture types for convenience
pub use mixtures::{HeadSelectionStrategy, HeadSelectionConfig, ThresholdPredictor};
// Also re-export RichardsNorm as DynamicTanhNorm for compatibility
pub use richards::RichardsNorm as DynamicTanhNorm;
// Also re-export RichardsGlu
pub use richards::RichardsGlu;
