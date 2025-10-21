pub mod activations;
pub mod adam;


pub mod cop;
pub mod dataset_loader;
pub mod embeddings;
pub mod errors;

pub mod gradient_clipping;
pub mod head_router;
pub mod layer_norm;
pub mod llm;

pub mod model_builder;
pub mod model_config;
pub mod model_persistence;
pub mod output_projection;
pub mod rms_norm;
pub mod dynamic_tanh_norm;
pub mod rope;
pub mod routing;
pub mod self_attention;
pub mod swiglu;

pub mod trm;
pub mod vocab;

// Define crate-level constants used across modules
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
pub const MAX_SEQ_LEN: usize = 128;
pub const MAX_VOCAB_SIZE: usize = 50_000;
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB
pub const MAX_INPUT_LENGTH: usize = 10_000;
pub const GRADIENT_ANOMALY_THRESHOLD: f32 = 2000.0;

// Re-export key structs for easier access
pub use adam::Adam;


pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use errors::{ModelError, Result};
pub use gradient_clipping::{
    AdaptiveClippingConfig, AdaptiveGradientClipping, GradientClipping, L2GradientClipping,
};
pub use head_router::{RouterType, HeadRouterStandard, FullyAdaptiveHeadRouter};



pub use llm::{LLM, Layer, LayerEnum};

pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{
    ArchitectureType,
    HeadSelectionStrategy,
    ModelConfig,
    PositionalEncodingType,
    WindowAdaptationStrategy,
};
// Also re-export Vocab for convenience
pub use vocab::Vocab;
