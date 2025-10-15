use serde::{Deserialize, Serialize};

/// Architecture type for model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Standard Transformer with self-attention mechanism
    Transformer,
    /// HyperMixer with dynamic token mixing via hypernetworks
    HyperMixer,
}

/// Configuration for model architecture and hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type of architecture to use
    pub architecture: ArchitectureType,
    
    /// Embedding dimension
    pub embedding_dim: usize,
    
    /// Hidden dimension for feedforward/channel mixing layers
    pub hidden_dim: usize,
    
    /// Number of transformer/hypermixer blocks
    pub num_layers: usize,
    
    /// Hidden dimension for hypernetwork (only used in HyperMixer)
    /// If None, defaults to embedding_dim / 4
    pub hypernetwork_hidden_dim: Option<usize>,
    
    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Number of attention heads for multi-head attention (used in both Transformer and HyperMixer)
    /// If None, defaults to 8 (same as standard transformers)
    pub num_heads: Option<usize>,
}

impl ModelConfig {
    /// Create a new Transformer configuration
    pub fn transformer(embedding_dim: usize, hidden_dim: usize, num_layers: usize, max_seq_len: usize, hypernetwork_hidden_dim: Option<usize>, num_heads: Option<usize>) -> Self {
        Self {
            architecture: ArchitectureType::Transformer,
            embedding_dim,
            hidden_dim,
            num_layers,
            hypernetwork_hidden_dim,
            max_seq_len,
            num_heads,
        }
    }
    
    /// Create a new HyperMixer configuration
    pub fn hypermixer(
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: Option<usize>,
        num_heads: Option<usize>,
    ) -> Self {
        Self {
            architecture: ArchitectureType::HyperMixer,
            embedding_dim,
            hidden_dim,
            num_layers,
            hypernetwork_hidden_dim,
            max_seq_len,
            num_heads,
        }
    }
    
    /// Get the hypernetwork hidden dimension, using default if not specified
    pub fn get_hypernetwork_hidden_dim(&self) -> usize {
        self.hypernetwork_hidden_dim.unwrap_or(self.embedding_dim / 4)
    }

    /// Get the number of attention heads, using default if not specified
    pub fn get_num_heads(&self) -> usize {
        self.num_heads.unwrap_or(8) // Same as standard transformers
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::transformer(128, 256, 3, 80, None, Some(8))
    }
}

