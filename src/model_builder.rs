use crate::{
    embeddings::Embeddings,
    feed_forward::FeedForward,
    hrm::HRMBlock,
    hypermixer::HyperMixerBlock,
    layer_norm::LayerNorm,
    llm::{Layer, LayerEnum},
    model_config::{ArchitectureType, ModelConfig},
    output_projection::OutputProjection,
    rms_norm::RMSNorm,
    self_attention::SelfAttention,
    swiglu::SwiGLU,
    vocab::Vocab,
};

/// Build a network based on the provided configuration
///
/// This function constructs Transformer, HyperMixer, or HRM architecture
/// based on the configuration, allowing for easy A/B comparison between
/// different approaches.
///
/// # Arguments
/// * `config` - Model configuration specifying architecture and hyperparameters
/// * `vocab` - Vocabulary for embeddings and output projection
///
/// # Returns
/// Vector of layers that form the complete network
pub fn build_network(config: &ModelConfig, vocab: &Vocab) -> Vec<LayerEnum> {
    let mut layers = Vec::new();

    // Add embedding layer (common to all architectures)
    layers.push(LayerEnum::Embeddings(Embeddings::new(vocab.clone())));

    // Build architecture-specific layers
    match config.architecture {
        ArchitectureType::Transformer => {
            build_transformer_layers(&mut layers, config);
        }
        ArchitectureType::HyperMixer => {
            build_hypermixer_layers(&mut layers, config);
        }
        ArchitectureType::HRM => {
            build_hrm_layers(&mut layers, config);
        }
    }

    // Add output projection layer (common to all architectures)
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(
        config.embedding_dim,
        vocab.words.len(),
    )));

    layers
}

/// Build Transformer architecture layers
///
/// Creates the Pre-LN transformer architecture with:
/// - Self-attention layers (with optional RoPE/CoPE)
/// - Layer normalization or RMSNorm (based on config)
/// - Feedforward networks or SwiGLU (based on config)
/// - Residual connections (handled within layers)
/// - Final normalization layer (required for Pre-LN stability)
///
/// Reference: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
fn build_transformer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let num_heads = config.get_num_heads();
    let num_kv_heads = config.get_num_kv_heads();

    for _ in 0..config.num_layers {
        // Self-attention layer (with positional encoding, GQA, Sliding Window, and Adaptive Window)
        let mut attention = SelfAttention::new_with_positional_encoding(
            config.embedding_dim,
            num_heads,
            num_kv_heads,
            &config.positional_encoding,
            config.max_seq_len,
            if config.use_adaptive_window {
                None
            } else {
                config.window_size
            },
        );

        // Enable adaptive window if configured
        if config.use_adaptive_window {
            attention.enable_adaptive_window(
                config.min_window_size,
                config.max_window_size,
                config.window_adaptation_strategy,
            );
        }

        // Set head selection strategy (MoH, AllHeads, or StaticPruning)
        attention.set_head_selection(config.head_selection.clone());

        // Gradient clipping is handled globally in the training loop via AdaptiveGradientClipper
        // Per-layer gradient clipping is disabled to avoid double-clipping and maintain
        // consistent gradient flow across all layers

        layers.push(LayerEnum::SelfAttention(Box::new(attention)));

        // Normalization after attention (LayerNorm or RMSNorm based on config)
        if config.use_rms_norm {
            layers.push(LayerEnum::RMSNorm(RMSNorm::new(config.embedding_dim)));
        } else {
            layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
        }

        // Feedforward layer (FeedForward or SwiGLU based on config)
        if config.use_swiglu {
            layers.push(LayerEnum::SwiGLU(Box::new(SwiGLU::new(
                config.embedding_dim,
                config.hidden_dim,
            ))));
        } else {
            layers.push(LayerEnum::FeedForward(Box::new(FeedForward::new(
                config.embedding_dim,
                config.hidden_dim,
            ))));
        }

        // Normalization after feedforward (LayerNorm or RMSNorm based on config)
        if config.use_rms_norm {
            layers.push(LayerEnum::RMSNorm(RMSNorm::new(config.embedding_dim)));
        } else {
            layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
        }
    }

    // Final normalization layer (critical for Pre-LN Transformer stability)
    // Reference: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    // Pre-LN requires a final norm before the output projection to stabilize gradients
    if config.use_rms_norm {
        layers.push(LayerEnum::RMSNorm(RMSNorm::new(config.embedding_dim)));
    } else {
        layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
    }
}

/// Build HyperMixer architecture layers
///
/// Creates the HyperMixer architecture with:
/// - Token mixing via hypernetworks (replaces attention)
/// - Channel mixing (similar to feedforward)
/// - Layer normalization
/// - Residual connections (handled within HyperMixerBlock)
fn build_hypermixer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let hypernetwork_hidden_dim = config.get_hypernetwork_hidden_dim();
    let num_heads = config.get_num_heads();

    for _ in 0..config.num_layers {
        // HyperMixer block (combines token mixing + channel mixing + norms)
        let hypermixer_block = HyperMixerBlock::new(
            config.embedding_dim,
            config.hidden_dim,
            config.max_seq_len,
            hypernetwork_hidden_dim,
            num_heads,
        );

        // Enable gradient clipping for training stability (higher threshold)
        // hypermixer_block.enable_gradient_clipping(50.0); // Commented out - handled at LLM level

        layers.push(LayerEnum::HyperMixerBlock(Box::new(hypermixer_block)));
    }
}

/// Print architecture summary
///
/// Displays information about the constructed network for debugging
/// and comparison purposes.
pub fn print_architecture_summary(config: &ModelConfig, layers: &[LayerEnum]) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║          MODEL ARCHITECTURE SUMMARY                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n📐 Base Configuration:");
    println!("  Architecture Type: {:?}", config.architecture);
    println!("  Embedding Dimension: {}", config.embedding_dim);
    println!("  Hidden Dimension: {}", config.hidden_dim);
    println!("  Number of Layers: {}", config.num_layers);
    println!("  Max Sequence Length: {}", config.max_seq_len);

    if config.architecture == ArchitectureType::HyperMixer {
        println!(
            "  Hypernetwork Hidden Dim: {}",
            config.get_hypernetwork_hidden_dim()
        );
    }

    // Modern LLM Enhancements
    println!("\n🚀 Modern LLM Enhancements:");

    // Normalization
    if config.use_rms_norm {
        println!("  ✓ RMSNorm (50% param reduction vs LayerNorm)");
    } else {
        println!("  • LayerNorm (standard)");
    }

    // Activation
    if config.use_swiglu {
        println!("  ✓ SwiGLU (gated activation, no bias)");
    } else {
        println!("  • FeedForward with ReLU (standard)");
    }

    // Positional Encoding
    use crate::model_config::PositionalEncodingType;
    match &config.positional_encoding {
        PositionalEncodingType::Learned => {
            println!("  • Learned Positional Embeddings (standard)");
        }
        PositionalEncodingType::RoPE => {
            println!("  ✓ RoPE (zero params, better extrapolation)");
        }
        PositionalEncodingType::CoPE { max_pos } => {
            println!(
                "  ✓ CoPE (context-aware, max_pos={}, best performance)",
                max_pos
            );
        }
    }

    // Group-Query Attention
    let num_heads = config.get_num_heads();
    let num_kv_heads = config.get_num_kv_heads();
    if num_kv_heads < num_heads {
        let reduction = ((num_heads - num_kv_heads) as f32 / num_heads as f32 * 100.0) as usize;
        println!("  ✓ Group-Query Attention (GQA)");
        println!("    - Query Heads: {}", num_heads);
        println!("    - KV Heads: {}", num_kv_heads);
        println!("    - Queries per KV: {}", num_heads / num_kv_heads);
        println!("    - KV Cache Reduction: ~{}%", reduction);
    } else {
        println!("  • Multi-Head Attention (MHA)");
        println!("    - Heads: {}", num_heads);
    }

    // Sliding Window Attention
    if config.use_adaptive_window {
        println!("  ✓ Adaptive Sliding Window Attention (Phase 4)");
        println!("    - Strategy: {:?}", config.window_adaptation_strategy);
        println!(
            "    - Window Range: {} - {}",
            config.min_window_size, config.max_window_size
        );
        println!(
            "    - Base Window: {}",
            config
                .window_size
                .map_or("None".to_string(), |w| w.to_string())
        );
        println!("    - Adapts dynamically based on context");
    } else if let Some(window_size) = config.window_size {
        println!("  ✓ Sliding Window Attention");
        println!("    - Window Size: {}", window_size);
        println!("    - Complexity: O(N × {})", window_size);
        println!("    - Enables efficient long-context processing");
    } else {
        println!("  • Full Attention");
        println!("    - Complexity: O(N²)");
    }

    // Head Selection Strategy (Mixture-of-Heads)
    use crate::model_config::HeadSelectionStrategy;
    match &config.head_selection {
        HeadSelectionStrategy::AllHeads => {
            println!("  • All Heads Active (standard MHA)");
        }
        HeadSelectionStrategy::MixtureOfHeads {
            num_shared_heads,
            num_active_routed_heads,
            load_balance_weight,
        } => {
            let num_routed_heads = num_heads - num_shared_heads;
            let total_active = num_shared_heads + num_active_routed_heads;
            let compute_savings =
                ((num_heads - total_active) as f32 / num_heads as f32 * 100.0) as usize;
            println!("  ✓ Mixture-of-Heads (MoH) - Dynamic Head Selection");
            println!("    - Total Heads: {}", num_heads);
            println!("    - Shared Heads: {} (always active)", num_shared_heads);
            println!(
                "    - Routed Heads: {} (Top-{} selection)",
                num_routed_heads, num_active_routed_heads
            );
            println!(
                "    - Active per Token: {}/{} heads",
                total_active, num_heads
            );
            println!("    - Compute Savings: ~{}%", compute_savings);
            println!("    - Load Balance Weight: {}", load_balance_weight);
            println!("    - Expected Speedup: 5-8%");
        }
        HeadSelectionStrategy::StaticPruning { num_active_heads } => {
            let compute_savings =
                ((num_heads - num_active_heads) as f32 / num_heads as f32 * 100.0) as usize;
            println!("  • Static Head Pruning (ablation study)");
            println!("    - Active Heads: {}/{}", num_active_heads, num_heads);
            println!("    - Compute Savings: ~{}%", compute_savings);
        }
    }

    // Architecture Alignment
    println!("\n🎯 Architecture Alignment:");
    let has_rms = config.use_rms_norm;
    let has_swiglu = config.use_swiglu;
    let has_rope = matches!(config.positional_encoding, PositionalEncodingType::RoPE);
    let has_cope = matches!(
        config.positional_encoding,
        PositionalEncodingType::CoPE { .. }
    );
    let has_gqa = num_kv_heads < num_heads;
    let has_window = config.window_size.is_some();

    if has_rms && has_swiglu && has_rope && !has_gqa && !has_window {
        println!("  Matches: LLaMA 1/2 7B, PaLM");
    } else if has_rms && has_swiglu && has_rope && has_gqa && !has_window {
        println!("  Matches: LLaMA 2 70B");
    } else if has_rms && has_swiglu && has_rope && has_gqa && has_window {
        println!("  Matches: Mistral 7B ⭐");
    } else if !has_rms && !has_swiglu && !has_rope && !has_gqa && !has_window {
        println!("  Matches: Original Transformer, GPT-2");
    } else if has_cope {
        println!("  Custom Configuration with CoPE (Research/Experimental) 🔬");
    } else {
        println!("  Custom Configuration");
    }

    println!("\n📊 Layer Stack:");
    for (idx, layer) in layers.iter().enumerate() {
        println!("  {}: {}", idx, layer.layer_type());
    }

    let total_params: usize = layers.iter().map(|l| l.parameters()).sum();
    println!("\n💾 Total Parameters: {}", total_params);
    println!("\n════════════════════════════════════════════════════════════════\n");
}

/// Build HRM architecture layers
///
/// Creates a single HRM block that internally contains N×T effective depth
/// through hierarchical convergence.
///
/// # Arguments
/// * `layers` - Mutable vector to append layers to
/// * `config` - Model configuration with HRM parameters
///
/// # HRM Configuration
/// - `num_layers` stores N (number of high-level cycles)
/// - `hypernetwork_hidden_dim` stores T (low-level steps per cycle)
/// - `hidden_dim` is used for Transformer blocks within HRM modules
fn build_hrm_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let num_high_cycles = config.get_num_high_cycles();
    let low_steps_per_cycle = config.get_low_steps_per_cycle();

    // Create single HRM block (contains N×T effective depth internally)
    let hrm_block = HRMBlock::new(
        config.embedding_dim,
        config.hidden_dim,
        num_high_cycles,
        low_steps_per_cycle,
        config.max_seq_len,
    );

    layers.push(LayerEnum::HRMBlock(Box::new(hrm_block)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_transformer_network() {
        let vocab = Vocab::new(vec!["a", "b", "c"]);
        let config = ModelConfig::transformer(128, 256, 2, 80, None, Some(8));

        let layers = build_network(&config, &vocab);

        // Should have: Embeddings + (Attention + Norm + FF + Norm) * 2 + OutputProjection
        // = 1 + 4*2 + 1 = 10 layers
        assert_eq!(layers.len(), 10);

        // Check first and last layers
        assert_eq!(layers[0].layer_type(), "Embeddings");
        assert_eq!(layers[layers.len() - 1].layer_type(), "OutputProjection");
    }

    #[test]
    fn test_build_hypermixer_network() {
        let vocab = Vocab::new(vec!["a", "b", "c"]);
        let config = ModelConfig::hypermixer(128, 256, 2, 80, Some(32), Some(8));

        let layers = build_network(&config, &vocab);

        // Should have: Embeddings + HyperMixerBlock * 2 + OutputProjection
        // = 1 + 2 + 1 = 4 layers
        assert_eq!(layers.len(), 4);

        // Check first and last layers
        assert_eq!(layers[0].layer_type(), "Embeddings");
        assert_eq!(layers[1].layer_type(), "HyperMixerBlock");
        assert_eq!(layers[2].layer_type(), "HyperMixerBlock");
        assert_eq!(layers[layers.len() - 1].layer_type(), "OutputProjection");
    }
}
