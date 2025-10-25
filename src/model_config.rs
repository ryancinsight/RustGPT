use serde::{Deserialize, Serialize};

/// Architecture type for model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Standard Transformer with self-attention mechanism
    Transformer,
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
    /// - Used in LLaMA, PaLM, GPT-NeoX, Mistral
    AttentionEntropy,

    /// Adapt based on prediction perplexity: larger windows when uncertain
    /// Most advanced, but requires perplexity computation
    PerplexityBased,
}

/// Strategy for selecting which attention heads to activate
///
/// Implements Mixture-of-Heads (MoH) for dynamic head selection per token.
/// Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeadSelectionStrategy {
    /// Fully Adaptive Mixture-of-Heads: complexity-aware dynamic head selection
    ///
    /// This is the only supported strategy. All heads are candidates and the number
    /// of active heads per token is determined by learned predictors.
    FullyAdaptiveMoH {
        /// Minimum number of heads to activate (safety constraint)
        min_heads: usize,
        /// Maximum number of heads to activate (efficiency constraint)
        max_heads: usize,
        /// Weight for load balance loss (prevents routing collapse)
        load_balance_weight: f32,
        /// Weight for complexity alignment loss (aligns head usage with predicted complexity)
        complexity_loss_weight: f32,
        /// Weight for sparsity loss (encourages minimal head usage)
        sparsity_weight: f32,
    },
}

/// Attention mechanism selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    /// Standard scaled dot-product self-attention
    SelfAttention,
    /// Polynomial attention layer with odd degree p (e.g., p=3)
    PolyAttention { degree_p: usize },
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

    /// Number of attention heads for multi-head attention (used in both Transformer and
    /// HyperMixer) If None, defaults to 8 (same as standard transformers)
    pub num_heads: Option<usize>,

    /// Use DynamicTanhNorm for normalization
    /// Default: false (disabled by default)
    pub use_dynamic_tanh_norm: bool,

    /// Maximum position value for CoPE positional encoding
    /// Default: 64 (works well for context length 1024)
    pub cope_max_pos: usize,

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

    #[serde(default = "entropy_ema_alpha_default_model")]
    pub entropy_ema_alpha: f32,

    /// Strategy for selecting which attention heads to activate
    ///
    /// Only `FullyAdaptiveMoH` is supported: complexity-aware dynamic head selection
    /// where all heads are candidates and the number of active heads per token
    /// is determined by learned predictors.
    ///
    /// Default: `FullyAdaptiveMoH`
    pub head_selection: HeadSelectionStrategy,

    /// Attention mechanism selection (SelfAttention vs PolyAttention)
    pub attention: AttentionType,
}

impl ModelConfig {
    /// Create a new Transformer configuration with modern defaults
    pub fn transformer(
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: Option<usize>,
        num_heads: Option<usize>,
    ) -> Self {
        Self {
            architecture: ArchitectureType::Transformer,
            embedding_dim,
            hidden_dim,
            num_layers,
            hypernetwork_hidden_dim,
            max_seq_len,
            num_heads,
            use_dynamic_tanh_norm: true, // Use DynamicTanhNorm
            cope_max_pos: 64,
            num_kv_heads: None,
            window_size: None,
            use_adaptive_window: false,
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            entropy_ema_alpha: 0.2,
            head_selection: HeadSelectionStrategy::FullyAdaptiveMoH {
                min_heads: 1,
                max_heads: num_heads.unwrap_or(8),
                load_balance_weight: 0.01,
                complexity_loss_weight: 0.01,
                sparsity_weight: 0.001,
            },
            attention: AttentionType::SelfAttention,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::transformer(128, 256, 3, 80, None, Some(8))
    }
}

// Provide serde default value for entropy_ema_alpha
fn entropy_ema_alpha_default_model() -> f32 {
    0.2
}

impl ModelConfig {
    pub fn get_num_heads(&self) -> usize {
        self.num_heads.unwrap_or(8)
    }

    pub fn get_num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.get_num_heads())
    }

    pub fn get_hypernetwork_hidden_dim(&self) -> usize {
        // Provide a reasonable default if not specified.
        self.hypernetwork_hidden_dim
            .unwrap_or(self.embedding_dim / 4)
    }

    pub fn get_recursive_depth(&self) -> usize {
        // In recursive models, num_layers stores the recursive depth
        self.num_layers
    }

    /// Get polynomial degree `p` for `PolyAttention`.
    /// Defaults to 3 if attention is not explicitly set to PolyAttention.
    pub fn get_poly_degree_p(&self) -> usize {
        match self.attention {
            AttentionType::PolyAttention { degree_p } => degree_p,
            _ => 3,
        }
    }
}
