use serde::{Deserialize, Serialize};

/// Architecture type for model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Standard Transformer with self-attention mechanism
    Transformer,
    /// HyperMixer with dynamic token mixing via hypernetworks
    HyperMixer,
    /// Hierarchical Reasoning Model with two-level recurrent architecture
    HRM,
}

/// Strategy for adapting sliding window size dynamically
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowAdaptationStrategy {
    /// Fixed window size (no adaptation)
    Fixed,

    /// Adapt based on sequence length: window_size = min(max, max(min, seq_len / 2))
    /// Simple and stable, scales window with input length
    SequenceLengthBased,

    /// Adapt based on attention entropy: larger windows when attention is diffuse
    /// More sophisticated, responds to attention patterns
    AttentionEntropy,

    /// Adapt based on prediction perplexity: larger windows when uncertain
    /// Most advanced, but requires perplexity computation
    PerplexityBased,
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

    /// Use RMSNorm instead of LayerNorm (modern LLM practice)
    /// Default: false (use LayerNorm for backward compatibility)
    pub use_rms_norm: bool,

    /// Use SwiGLU instead of ReLU-based FeedForward (modern LLM practice)
    /// Default: false (use FeedForward for backward compatibility)
    pub use_swiglu: bool,

    /// Use RoPE (Rotary Positional Encoding) instead of learned positional embeddings
    /// Default: false (use learned embeddings for backward compatibility)
    pub use_rope: bool,

    /// Number of key-value heads for Group-Query Attention (GQA)
    /// If None, uses standard Multi-Head Attention (MHA) with num_heads KV heads
    /// If Some(n), uses GQA with n KV heads shared across query heads
    /// Example: num_heads=8, num_kv_heads=Some(4) → 2 query heads per KV head
    /// Default: None (use MHA for backward compatibility)
    pub num_kv_heads: Option<usize>,

    /// Sliding window size for attention (Sliding Window Attention)
    ///
    /// If None, uses full attention (all tokens attend to all previous tokens)
    /// If Some(w), each token only attends to the last w tokens (sliding window)
    /// Example: window_size=Some(4096) → Mistral 7B style (32k context efficient)
    ///
    /// Benefits:
    ///
    /// - Reduces attention complexity from O(N²) to O(N × window_size)
    /// - Enables longer context windows (32k+ tokens) efficiently
    /// - Minimal quality degradation (local context often sufficient)
    ///
    /// Default: None (use full attention for backward compatibility)
    pub window_size: Option<usize>,

    /// Enable adaptive window sizing (Phase 4 enhancement)
    ///
    /// If true, window size adapts dynamically based on the chosen strategy
    /// If false, uses fixed window_size (Phase 3 behavior)
    ///
    /// Default: false (use fixed window for backward compatibility)
    pub use_adaptive_window: bool,

    /// Minimum window size for adaptive window sizing
    ///
    /// Only used when use_adaptive_window = true
    /// Ensures window never shrinks below this value
    ///
    /// Default: 512 (reasonable minimum for most tasks)
    pub min_window_size: usize,

    /// Maximum window size for adaptive window sizing
    ///
    /// Only used when use_adaptive_window = true
    /// Ensures window never grows beyond this value
    ///
    /// Default: 4096 (Mistral 7B style)
    pub max_window_size: usize,

    /// Strategy for adapting window size
    ///
    /// Only used when use_adaptive_window = true
    /// Determines how window size changes based on context
    ///
    /// Default: SequenceLengthBased (simplest and most stable)
    pub window_adaptation_strategy: WindowAdaptationStrategy,
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
            use_rms_norm: false, // Default to LayerNorm for backward compatibility
            use_swiglu: false,   // Default to FeedForward for backward compatibility
            use_rope: false,     // Default to learned embeddings for backward compatibility
            num_kv_heads: None,  // Default to MHA for backward compatibility
            window_size: None,   // Default to full attention for backward compatibility
            use_adaptive_window: false, // Default to fixed window for backward compatibility
            min_window_size: 512,       // Reasonable minimum
            max_window_size: 4096,      // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
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
            use_rms_norm: false, // Default to LayerNorm for backward compatibility
            use_swiglu: false,   // Default to FeedForward for backward compatibility
            use_rope: false,     // Default to learned embeddings for backward compatibility
            num_kv_heads: None,  // Default to MHA for backward compatibility
            window_size: None,   // Default to full attention for backward compatibility
            use_adaptive_window: false, // Default to fixed window for backward compatibility
            min_window_size: 512,       // Reasonable minimum
            max_window_size: 4096,      // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
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

    /// Get the number of key-value heads for GQA
    /// If None, returns num_heads (standard MHA)
    /// If Some(n), returns n (GQA with n KV heads)
    pub fn get_num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or_else(|| self.get_num_heads())
    }

    /// Create a new HRM configuration
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of embeddings (e.g., 128)
    /// * `hidden_dim` - Hidden dimension for Transformer layers (e.g., 192)
    /// * `num_high_cycles` - Number of high-level cycles N (e.g., 2)
    /// * `low_steps_per_cycle` - Low-level steps per cycle T (e.g., 2)
    /// * `max_seq_len` - Maximum sequence length (e.g., 80)
    ///
    /// # Returns
    ///
    /// A new `ModelConfig` for HRM architecture
    ///
    /// # Note
    ///
    /// - `num_layers` stores N (number of high-level cycles)
    /// - `hypernetwork_hidden_dim` stores T (low-level steps per cycle)
    pub fn hrm(
        embedding_dim: usize,
        hidden_dim: usize,
        num_high_cycles: usize,
        low_steps_per_cycle: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            architecture: ArchitectureType::HRM,
            embedding_dim,
            hidden_dim,
            num_layers: num_high_cycles,
            hypernetwork_hidden_dim: Some(low_steps_per_cycle),
            max_seq_len,
            num_heads: None,
            use_rms_norm: false, // Default to LayerNorm for backward compatibility
            use_swiglu: false,   // Default to FeedForward for backward compatibility
            use_rope: false,     // Default to learned embeddings for backward compatibility
            num_kv_heads: None,  // Default to MHA for backward compatibility
            window_size: None,   // Default to full attention for backward compatibility
            use_adaptive_window: false, // Default to fixed window for backward compatibility
            min_window_size: 512,       // Reasonable minimum
            max_window_size: 4096,      // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
        }
    }

    /// Get the number of high-level cycles (N) for HRM
    pub fn get_num_high_cycles(&self) -> usize {
        self.num_layers
    }

    /// Get the number of low-level steps per cycle (T) for HRM
    pub fn get_low_steps_per_cycle(&self) -> usize {
        self.hypernetwork_hidden_dim.unwrap_or(2)
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::transformer(128, 256, 3, 80, None, Some(8))
    }
}

