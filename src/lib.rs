pub mod adam;
pub mod beam_search;
pub mod channel_mixing;
pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod feed_forward;
pub mod swiglu;
pub mod gradient_clipping;
pub mod hrm;
pub mod hrm_high_level;
pub mod hrm_low_level;
pub mod hypernetwork;
pub mod hypermixer;
pub mod layer_norm;
pub mod rms_norm;
pub mod rope;
pub mod cop;
pub mod llm;
pub mod model_builder;
pub mod model_config;
pub mod output_projection;
pub mod self_attention;
pub mod token_mixing;
pub mod transformer;
pub mod vocab;
// Re-export key structs for easier access
pub use adam::Adam;
pub use beam_search::{BeamHypothesis, BeamSearchConfig, BeamSearchState};
pub use channel_mixing::ChannelMixingMLP;
pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use errors::{ModelError, Result};
pub use gradient_clipping::{
    AdaptiveClippingConfig, AdaptiveGradientClipping, GradientClipping, L2GradientClipping,
};
pub use hrm::HRMBlock;
pub use hrm_high_level::HighLevelModule;
pub use hrm_low_level::LowLevelModule;
pub use hypernetwork::Hypernetwork;
pub use hypermixer::HyperMixerBlock;
pub use llm::{LLM, Layer, LayerEnum};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{ArchitectureType, ModelConfig, PositionalEncodingType, WindowAdaptationStrategy};
pub use token_mixing::TokenMixingMLP;
pub use vocab::Vocab;

// Constants
pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;

// Security validation constants
pub const MAX_INPUT_LENGTH: usize = 10_000; // Maximum input text length
pub const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024; // 100MB max file size
pub const MAX_VOCAB_SIZE: usize = 50_000; // Maximum vocabulary size
pub const GRADIENT_ANOMALY_THRESHOLD: f32 = 1000.0; // Threshold for gradient anomaly detection