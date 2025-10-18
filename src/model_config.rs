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

/// Positional encoding type for attention mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PositionalEncodingType {
    /// Learned positional embeddings (standard absolute positional embeddings)
    /// - Parameters: max_seq_len × embedding_dim learned weights
    /// - Used in original Transformer, GPT-2, GPT-3
    /// - Simple and effective for fixed-length contexts
    #[default]
    Learned,

    /// RoPE (Rotary Positional Encoding): Geometric position encoding
    /// - Parameters: Zero (no learned weights)
    /// - Encodes relative position through rotation matrices
    /// - Better length extrapolation (handles longer sequences)
    /// - Used in LLaMA, PaLM, GPT-NeoX, Mistral
    RoPE,

    /// CoPE (Contextual Position Encoding): Context-aware position encoding
    /// - Parameters: max_pos × head_dim learned position embeddings (max_pos << max_seq_len)
    /// - Positions conditioned on context via gating mechanism
    /// - Can count abstract units (words, sentences, specific tokens)
    /// - Better OOD generalization and perplexity than RoPE
    /// - Used in research (Meta FAIR 2024)
    CoPE {
        /// Maximum position value (can be much smaller than sequence length)
        /// For example, max_pos=64 works well for context length 1024
        max_pos: usize,
    },
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

/// Strategy for selecting which attention heads to activate
///
/// Implements Mixture-of-Heads (MoH) for dynamic head selection per token.
/// Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum HeadSelectionStrategy {
    /// All heads always active (standard multi-head attention)
    /// - Zero overhead
    /// - Backward compatible
    /// - Use for baseline comparisons
    #[default]
    AllHeads,

    /// Mixture-of-Heads: dynamic head selection per token
    /// - Learned routing via router network
    /// - Shared heads (always active) + routed heads (Top-K)
    /// - Weighted summation for flexibility
    /// - Load balance loss to prevent routing collapse
    /// - Recommended for 5-8% speedup with minimal memory overhead
    MixtureOfHeads {
        /// Number of shared heads (always active, capture common knowledge)
        /// Recommended: 25% of total heads (e.g., 2 out of 8)
        num_shared_heads: usize,

        /// Number of routed heads to activate per token (Top-K)
        /// Recommended: 50-75% of routed heads
        /// Example: 8 total, 2 shared, 6 routed → activate 3-4 routed
        num_active_routed_heads: usize,

        /// Weight for load balance loss (β in paper)
        /// Recommended: 0.01 (prevents routing collapse)
        load_balance_weight: f32,
    },

    /// Static head pruning: use only first K heads (for ablation studies)
    /// - No routing overhead
    /// - Fixed head selection
    /// - Useful for comparing against dynamic routing
    StaticPruning {
        /// Number of heads to keep active
        num_active_heads: usize,
    },
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

    /// Positional encoding type to use
    /// Default: CoPE (modern default for best performance)
    pub positional_encoding: PositionalEncodingType,

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

    /// Strategy for selecting which attention heads to activate
    ///
    /// Controls dynamic head selection (Mixture-of-Heads):
    /// - AllHeads: All heads active (standard MHA, backward compatible)
    /// - MixtureOfHeads: Dynamic head selection (5-8% speedup, <1% memory)
    /// - StaticPruning: First K heads (for ablation studies)
    ///
    /// Default: AllHeads (backward compatible)
    pub head_selection: HeadSelectionStrategy,

    /// Use Mixture of Experts (MoE) instead of standard feedforward layers
    ///
    /// If true, replaces SwiGLU/FeedForward with sparse MoE layers
    /// Each MoE layer contains multiple expert networks with learned routing
    ///
    /// Benefits:
    /// - Increased model capacity without proportional compute increase
    /// - Sparse activation (only k out of N experts active per token)
    /// - Better specialization through expert routing
    ///
    /// Default: false (use standard feedforward for backward compatibility)
    pub use_moe: bool,

    /// Number of expert networks in each MoE layer
    ///
    /// Only used when use_moe = true
    /// Typical values: 4, 8, 16
    ///
    /// Default: 4 (balance between capacity and complexity)
    pub num_experts: usize,

    /// Number of experts to activate per token (k in top-k routing)
    ///
    /// Only used when use_moe = true
    /// Typical values: 1 (Switch Transformers), 2 (Mixtral)
    ///
    /// Default: 2 (Mixtral-style, more stable than top-1)
    pub num_active_experts: usize,

    /// Hidden dimension for each expert network
    ///
    /// Only used when use_moe = true
    /// Should be smaller than hidden_dim to maintain parameter count
    /// Typical: hidden_dim / num_experts or hidden_dim / (num_experts / 2)
    ///
    /// Default: hidden_dim / 4 (for 4 experts, maintains ~same params)
    pub expert_hidden_dim: usize,

    /// Weight for MoE load balance loss
    ///
    /// Only used when use_moe = true
    /// Encourages uniform expert utilization
    ///
    /// Default: 0.01 (standard in literature)
    pub moe_load_balance_weight: f32,

    /// Weight for MoE router z-loss
    ///
    /// Only used when use_moe = true
    /// Prevents routing logits from growing too large
    ///
    /// Default: 0.001 (standard in literature)
    pub moe_router_z_loss_weight: f32,
}

impl ModelConfig {
    /// Create a new Transformer configuration with modern defaults
    ///
    /// Modern defaults (as of 2024):
    /// - RMSNorm (faster, more stable than LayerNorm)
    /// - SwiGLU (better than ReLU/GELU)
    /// - CoPE positional encoding (better than RoPE/Learned)
    /// - MHA (can be changed to GQA via num_kv_heads)
    /// - Full attention (can be changed via window_size)
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
            use_rms_norm: true, // Modern default: RMSNorm
            use_swiglu: true,   // Modern default: SwiGLU
            positional_encoding: PositionalEncodingType::CoPE { max_pos: 64 }, // Modern default: CoPE
            num_kv_heads: None,         // Default to MHA (can be changed to GQA)
            window_size: None, // Default to full attention (can be changed to sliding window)
            use_adaptive_window: false, // Default to fixed window
            min_window_size: 512, // Reasonable minimum
            max_window_size: 4096, // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            head_selection: HeadSelectionStrategy::AllHeads, // Default to all heads (can be changed to MoH)
            use_moe: false, // Default to standard feedforward (can be changed to MoE)
            num_experts: 4, // 4 experts (balance between capacity and complexity)
            num_active_experts: 2, // Top-2 routing (Mixtral-style)
            expert_hidden_dim: hidden_dim / 4, // Maintain parameter count
            moe_load_balance_weight: 0.01, // Standard load balance weight
            moe_router_z_loss_weight: 0.001, // Standard router z-loss weight
        }
    }

    /// Create a new HyperMixer configuration with modern defaults
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
            use_rms_norm: true, // Modern default: RMSNorm
            use_swiglu: true,   // Modern default: SwiGLU
            positional_encoding: PositionalEncodingType::CoPE { max_pos: 64 }, // Modern default: CoPE
            num_kv_heads: None,                                                // Default to MHA
            window_size: None,          // Default to full attention
            use_adaptive_window: false, // Default to fixed window
            min_window_size: 512,       // Reasonable minimum
            max_window_size: 4096,      // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            head_selection: HeadSelectionStrategy::AllHeads, // Default to all heads
            use_moe: false, // Default to standard feedforward
            num_experts: 4,
            num_active_experts: 2,
            expert_hidden_dim: hidden_dim / 4,
            moe_load_balance_weight: 0.01,
            moe_router_z_loss_weight: 0.001,
        }
    }

    /// Get the hypernetwork hidden dimension, using default if not specified
    pub fn get_hypernetwork_hidden_dim(&self) -> usize {
        self.hypernetwork_hidden_dim
            .unwrap_or(self.embedding_dim / 4)
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
            use_rms_norm: true, // Modern default: RMSNorm
            use_swiglu: true,   // Modern default: SwiGLU
            positional_encoding: PositionalEncodingType::CoPE { max_pos: 64 }, // Modern default: CoPE
            num_kv_heads: None,                                                // Default to MHA
            window_size: None,          // Default to full attention
            use_adaptive_window: false, // Default to fixed window
            min_window_size: 512,       // Reasonable minimum
            max_window_size: 4096,      // Mistral 7B style
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            head_selection: HeadSelectionStrategy::AllHeads, // Default to all heads
            use_moe: false, // Default to standard feedforward
            num_experts: 4,
            num_active_experts: 2,
            expert_hidden_dim: hidden_dim / 4,
            moe_load_balance_weight: 0.01,
            moe_router_z_loss_weight: 0.001,
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
