pub mod adam;
pub mod channel_mixing;
pub mod dataset_loader;
pub mod embeddings;
pub mod errors;
pub mod feed_forward;
pub mod gradient_clipping;
pub mod hypernetwork;
pub mod hypermixer;
pub mod layer_norm;
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
pub use channel_mixing::ChannelMixingMLP;
pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use errors::{ModelError, Result};
pub use gradient_clipping::{
    AdaptiveClippingConfig, AdaptiveGradientClipping, GradientClipping, L2GradientClipping,
};
pub use hypernetwork::Hypernetwork;
pub use hypermixer::HyperMixerBlock;
pub use llm::{LLM, Layer, LayerEnum};
pub use model_builder::{build_network, print_architecture_summary};
pub use model_config::{ArchitectureType, ModelConfig};
pub use token_mixing::TokenMixingMLP;
pub use vocab::Vocab;

// Constants
pub const MAX_SEQ_LEN: usize = 80;
pub const EMBEDDING_DIM: usize = 128;
pub const HIDDEN_DIM: usize = 256;
