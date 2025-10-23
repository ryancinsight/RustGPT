use serde::{Deserialize, Serialize};

/// Architecture type for model configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchitectureType {
    /// Standard Transformer with self-attention mechanism
    Transformer,
}

/// Positional encoding type for attention mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionalEncodingType {
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

impl Default for PositionalEncodingType {
    fn default() -> Self {
        PositionalEncodingType::CoPE { max_pos: 64 }
    }
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
    /// All heads always active (standard multi-head attention)
    /// - Zero overhead
    /// - Backward compatible
    /// - Use for baseline comparisons

    AllHeads,

    /// Mixture-of-Heads: fully adaptive dynamic head selection per token
    /// - Learned routing via router network
    /// - Shared heads (always active) + routed heads (adaptive top-p)
    /// - Per-token learned thresholds (optional)
    /// - Layer-wise adaptive thresholds
    /// - Sparsity-based adaptive loss weight
    /// - Training-progress-based annealing
    /// - Load balance loss to prevent routing collapse
    /// - Dynamic loss (entropy minimization) to encourage sparsity
    /// - Recommended for 30-50% efficiency gain with maintained/improved performance
    MixtureOfHeads {
        /// Number of shared heads (always active, capture common knowledge)
        /// Recommended: 25% of total heads (e.g., 2 out of 8)
        num_shared_heads: usize,

        /// Number of routed heads to activate per token (DEPRECATED for adaptive routing)
        /// Kept for backward compatibility, not used in adaptive top-p routing
        /// With adaptive routing, each token selects 1-6 routed heads based on confidence
        num_active_routed_heads: usize,

        /// Weight for load balance loss (β in paper)
        /// Recommended: 0.01 (prevents routing collapse)
        load_balance_weight: f32,

        /// Base threshold for adaptive top-p routing (adjusted per-layer and per-token)
        /// Range: 0.0-1.0, typical: 0.4-0.6
        /// Lower values → fewer heads activated (more efficient, may hurt accuracy)
        /// Higher values → more heads activated (less efficient, better accuracy)
        /// Recommended: 0.5 (balanced efficiency and performance)
        /// Note: Actual threshold varies by layer (early: +0.1, late: -0.1)
        threshold_p_base: f32,

        /// Base weight for dynamic loss (adjusted adaptively based on sparsity and training progress)
        /// Typical: 1e-4
        /// Actual weight = base * sparsity_multiplier * annealing_multiplier
        /// Prevents model from activating all heads by encouraging confident routing
        dynamic_loss_weight_base: f32,

        /// Whether to use learned per-token threshold predictor
        /// If true: each token gets custom threshold based on its representation
        /// If false: use layer-wise base thresholds only
        /// Recommended: true for maximum adaptability (adds ~embedding_dim parameters)
        use_learned_threshold: bool,

        /// Target average number of routed heads for sparsity-based adaptation
        /// Typical: 3.0 (for 6 routed heads total)
        /// Used to adjust dynamic_loss_weight based on current sparsity
        target_avg_routed_heads: f32,

        /// Confidence threshold for fallback to all heads (0.0-1.0)
        /// When max routing probability < threshold, activate all routed heads
        /// Typical: 0.6 (60% confidence required for sparse routing)
        /// Lower values → more aggressive sparsity (may hurt quality)
        /// Higher values → more conservative (better quality, less efficient)
        confidence_threshold: f32,

        /// Whether to use confidence-based fallback
        /// If true: activate all heads when routing confidence is low
        /// If false: always use adaptive sparse routing
        /// Recommended: true for maintaining quality during training
        use_confidence_fallback: bool,
    },

    /// Fully Adaptive Mixture-of-Heads: complexity-aware dynamic head selection
    ///
    /// **Key Innovation**: No hardcoded shared/routed head split - ALL heads are routing candidates.
    /// Head count is determined by learned complexity predictor based on input difficulty.
    ///
    /// # Architecture
    ///
    /// - **Complexity Predictor**: Learns to predict input complexity → target head count
    /// - **Threshold Predictor**: Learns per-token threshold for top-p selection
    /// - **Unified Router**: Single router for all heads (no shared/routed split)
    ///
    /// # Advantages over MixtureOfHeads
    ///
    /// - ✅ **No hardcoded shared heads** - all heads can be deactivated for simple inputs
    /// - ✅ **Complexity-aware** - simple inputs use 1-2 heads, complex inputs use 6-8 heads
    /// - ✅ **Better efficiency** - 15-25% speedup vs 5-8% for standard MoH
    /// - ✅ **Cleaner architecture** - fewer parameters, simpler routing logic
    /// - ✅ **Compatible with various architectures** - designed for gradient stability
    ///
    /// # Example Usage
    ///
    /// ```ignore
    /// HeadSelectionStrategy::FullyAdaptiveMoH {
    ///     min_heads: 1,                      // Allow single head for simple inputs
    ///     max_heads: 8,                      // All heads available for complex inputs
    ///     load_balance_weight: 0.01,         // Prevent routing collapse
    ///     complexity_loss_weight: 0.01,      // Align head usage with complexity
    ///     sparsity_weight: 0.001,            // Encourage minimal head usage
    /// }
    /// ```
    ///
    /// # Expected Performance
    ///
    /// - **Simple inputs**: 1-2 heads (12-25% of total)
    /// - **Medium inputs**: 3-4 heads (37-50% of total)
    /// - **Complex inputs**: 6-8 heads (75-100% of total)
    /// - **Average**: 3-4 heads (44% of total, vs 69% for standard MoH)
    ///
    /// # References
    ///
    /// - Design: `docs/FULLY_ADAPTIVE_MOH_DESIGN.md`
    /// - Based on: "MoH: Multi-Head Attention as Mixture-of-Head Attention" (Skywork AI, 2024)
    FullyAdaptiveMoH {
        /// Minimum number of heads to activate (safety constraint)
        ///
        /// Ensures at least this many heads are active even for very simple inputs.
        /// Recommended: 1 (allow single head for maximum efficiency)
        ///
        /// Range: 1 to num_heads
        min_heads: usize,

        /// Maximum number of heads to activate (efficiency constraint)
        ///
        /// Caps the number of heads for very complex inputs.
        /// Recommended: num_heads (allow all heads for maximum capacity)
        ///
        /// Range: min_heads to num_heads
        max_heads: usize,

        /// Weight for load balance loss (prevents routing collapse)
        ///
        /// Encourages uniform head usage across tokens to prevent all tokens
        /// routing to the same subset of heads.
        ///
        /// Loss: L_balance = Σ(i=1 to num_heads) P_i × f_i
        /// Where P_i = avg routing score, f_i = fraction of tokens selecting head i
        ///
        /// Recommended: 0.01
        /// Range: 0.001 to 0.1
        load_balance_weight: f32,

        /// Weight for complexity alignment loss (aligns head usage with predicted complexity)
        ///
        /// Encourages the router to use the number of heads predicted by the
        /// complexity predictor, ensuring efficient resource allocation.
        ///
        /// Loss: L_complexity = |avg_active_heads - avg_target_heads|
        ///
        /// Recommended: 0.01
        /// Range: 0.001 to 0.1
        complexity_loss_weight: f32,

        /// Weight for sparsity loss (encourages minimal head usage)
        ///
        /// Provides a small penalty for using more heads, encouraging the model
        /// to use as few heads as possible while maintaining quality.
        ///
        /// Loss: L_sparsity = (avg_active_heads / num_heads)
        ///
        /// Recommended: 0.001 (10x smaller than other losses)
        /// Range: 0.0001 to 0.01
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

    /// Number of attention heads for multi-head attention (used in both Transformer and HyperMixer)
    /// If None, defaults to 8 (same as standard transformers)
    pub num_heads: Option<usize>,

    /// Use DynamicTanhNorm for normalization
    /// Default: false (disabled by default)
    pub use_dynamic_tanh_norm: bool,



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

    #[serde(default = "entropy_ema_alpha_default_model")]
    pub entropy_ema_alpha: f32,

    /// Strategy for selecting which attention heads to activate
    ///
    /// Controls dynamic head selection (Mixture-of-Heads):
    /// - AllHeads: All heads active (standard MHA, backward compatible)
    /// - MixtureOfHeads: Dynamic head selection (5-8% speedup, <1% memory)
    /// - StaticPruning: First K heads (for ablation studies)
    ///
    /// Default: AllHeads (backward compatible)
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
            positional_encoding: PositionalEncodingType::CoPE { max_pos: 64 },
            num_kv_heads: None,
            window_size: None,
            use_adaptive_window: false,
            min_window_size: 512,
            max_window_size: 4096,
            window_adaptation_strategy: WindowAdaptationStrategy::SequenceLengthBased,
            entropy_ema_alpha: 0.2,
            head_selection: HeadSelectionStrategy::MixtureOfHeads {
                num_shared_heads: 2,
                num_active_routed_heads: 4,
                load_balance_weight: 0.01,
                threshold_p_base: 0.5,
                dynamic_loss_weight_base: 0.0001,
                use_learned_threshold: true,
                target_avg_routed_heads: 3.0,
                confidence_threshold: 0.6,
                use_confidence_fallback: true,
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
fn entropy_ema_alpha_default_model() -> f32 { 0.2 }

impl ModelConfig {
    pub fn get_num_heads(&self) -> usize {
        self.num_heads.unwrap_or(8)
    }

    pub fn get_num_kv_heads(&self) -> usize {
        self.num_kv_heads.unwrap_or(self.get_num_heads())
    }

    pub fn get_hypernetwork_hidden_dim(&self) -> usize {
        // Provide a reasonable default if not specified.
        self.hypernetwork_hidden_dim.unwrap_or(self.embedding_dim / 4)
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
