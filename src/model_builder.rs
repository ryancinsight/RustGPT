use crate::{
    embeddings::Embeddings,
    // feed_forward::FeedForward, // Removed: using SwiGLU exclusively

    llm::{Layer, LayerEnum},
    model_config::{ArchitectureType, ModelConfig},
    output_projection::OutputProjection,
    swiglu::SwiGLU,
    vocab::Vocab,
    dynamic_tanh_norm::DynamicTanhNorm,
    poly_attention::PolyAttention,
};

/// Build a network based on the provided configuration
///
/// This function constructs Transformer architecture
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
    // CoPE handles positions inside attention, so disable additive positional embeddings
    let use_positional = false;
    layers.push(LayerEnum::Embeddings(Embeddings::new_with_positional(vocab.clone(), use_positional)));

    // Build architecture-specific layers
    match config.architecture {
        ArchitectureType::Transformer => {
            build_transformer_layers(&mut layers, config);
        }
    }

    // Add output projection layer (common to all architectures)
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(
        config.embedding_dim,
        vocab.size(),
    )));

    layers
}

/// Build Transformer architecture layers
///
/// Creates the Pre-LN transformer architecture with:
/// - Self-attention layers (with optional RoPE/CoPE)
/// - DynamicTanhNorm normalization
/// - SwiGLU feedforward (exclusively)
/// - Residual connections (handled within layers)
/// - Final normalization layer (required for Pre-LN stability)
fn build_transformer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let num_heads = config.get_num_heads();

    for _layer_idx in 0..config.num_layers {
        // Always build PolyAttention layers
    let max_pos = match config.positional_encoding { crate::model_config::PositionalEncodingType::CoPE { max_pos } => max_pos };
        let poly = PolyAttention::new(
            config.embedding_dim,
            num_heads,
            config.get_poly_degree_p(),
            max_pos,
            config.window_size,
        );
        layers.push(LayerEnum::PolyAttention(Box::new(poly)));

        // Normalization after attention (always DynamicTanhNorm)
        layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));

        // Feedforward layer (SwiGLU only)
        {
            let swiglu = SwiGLU::new(
                config.embedding_dim,
                config.hidden_dim,
            );

            layers.push(LayerEnum::SwiGLU(Box::new(swiglu)));
        }

        // Normalization after feedforward (always DynamicTanhNorm)
        layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));
    }

    // Final normalization layer (critical for Pre-LN Transformer stability)
    // Avoid duplicate norm if the last layer is already a normalization
    let last_is_norm = matches!(
        layers.last(),
        Some(LayerEnum::DynamicTanhNorm(_))
    );
    if !last_is_norm {
        layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));
    }

}


/// Print architecture summary
///
/// Displays information about the constructed network for debugging
/// and comparison purposes.
pub fn print_architecture_summary(config: &ModelConfig, layers: &[LayerEnum]) {
    println!("\n╔════════════════════════════════════════════════════════════════╝");
    println!("║          MODEL ARCHITECTURE SUMMARY                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n📐 Base Configuration:");
    println!("  Architecture Type: {:?}", config.architecture);
    println!("  Embedding Dimension: {}", config.embedding_dim);
    println!("  Hidden Dimension: {}", config.hidden_dim);
    println!("  Number of Layers: {}", config.num_layers);
    println!("  Max Sequence Length: {}", config.max_seq_len);


    // Modern LLM Enhancements
    println!("\n🚀 Modern LLM Enhancements:");

    // Normalization
    println!("  ✓ DynamicTanhNorm (adaptive, tanh-based)");

    // Activation
    println!("  ✓ SwiGLU (gated activation, no bias)");

    // Positional Encoding
    use crate::model_config::PositionalEncodingType;
    match &config.positional_encoding {
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
        println!("    - Entropy EMA Alpha: {}", config.entropy_ema_alpha);
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
            println!("  • All Heads Active (Standard Multi-Head Attention)");
            println!("    - No efficiency gains");
        }

        HeadSelectionStrategy::MixtureOfHeads {
            num_shared_heads,
            num_active_routed_heads: _,
            load_balance_weight,
            threshold_p_base,
            dynamic_loss_weight_base,
            use_learned_threshold,
            target_avg_routed_heads,
            confidence_threshold,
            use_confidence_fallback,
        } => {
            let num_routed_heads = num_heads - num_shared_heads;
            // With adaptive routing, estimate average active heads based on threshold_p_base
            // Lower threshold → fewer heads, higher threshold → more heads
            let estimated_avg_routed = (num_routed_heads as f32 * threshold_p_base * 1.2).min(num_routed_heads as f32);
            let estimated_total_active = *num_shared_heads as f32 + estimated_avg_routed;
            let compute_savings =
                ((num_heads as f32 - estimated_total_active) / num_heads as f32 * 100.0) as usize;
            println!("  ✓ Mixture-of-Heads (MoH) - Fully Adaptive Routing");
            println!("    - Total Heads: {}", num_heads);
            println!("    - Shared Heads: {} (always active)", num_shared_heads);
            println!(
                "    - Routed Heads: {} (adaptive top-p selection)",
                num_routed_heads
            );
            println!(
                "    - Target Avg Routed: {:.1} heads",
                target_avg_routed_heads
            );
            println!(
                "    - Estimated Active per Token: {:.1}/{} heads",
                estimated_total_active, num_heads
            );
            println!("    - Base Threshold P: {} (layer-wise adjusted)", threshold_p_base);
            if *use_learned_threshold {
                println!("    - Learned Per-Token Thresholds: ENABLED");
                println!("      → Each token gets custom threshold [0.3-0.7]");
            } else {
                println!("    - Learned Per-Token Thresholds: DISABLED");
                println!("      → Using layer-wise base thresholds only");
            }
            println!("    - Layer-Wise Adaptation:");
            println!("      → Early layers (L0-L3): p = {:.2} (more heads)", threshold_p_base + 0.1);
            println!("      → Middle layers (L4-L7): p = {:.2}", threshold_p_base);
            println!("      → Late layers (L8+): p = {:.2} (fewer heads)", threshold_p_base - 0.1);
            println!("    - Estimated Compute Savings: ~{}%", compute_savings);
            println!("    - Load Balance Weight: {}", load_balance_weight);
            println!("    - Base Dynamic Loss Weight: {}", dynamic_loss_weight_base);
            println!("      → Adjusted by sparsity & training progress");
            if *use_confidence_fallback {
                println!("    - Confidence-Based Fallback: ENABLED");
                println!("      → Confidence Threshold: {:.2} ({}% required)", confidence_threshold, confidence_threshold * 100.0);
                println!("      → Activates all heads when routing confidence < threshold");
            } else {
                println!("    - Confidence-Based Fallback: DISABLED");
            }
            println!("    - Expected Efficiency Gain: 30-50%");
        }

        HeadSelectionStrategy::FullyAdaptiveMoH {
            min_heads,
            max_heads,
            load_balance_weight,
            complexity_loss_weight,
            sparsity_weight,
        } => {
            let estimated_avg_heads = (min_heads + max_heads) as f32 / 2.0;
            let compute_savings = ((num_heads as f32 - estimated_avg_heads) / num_heads as f32 * 100.0) as usize;
            println!("  ✓ Fully Adaptive Mixture-of-Heads - Complexity-Aware Routing");
            println!("    - Total Heads: {} (all are routing candidates)", num_heads);
            println!("    - Min Heads: {} (for simple inputs)", min_heads);
            println!("    - Max Heads: {} (for complex inputs)", max_heads);
            println!("    - Estimated Avg Active: {:.1}/{} heads", estimated_avg_heads, num_heads);
            println!("    - Complexity Predictor: ENABLED");
            println!("      → Learns to predict input complexity → target head count");
            println!("    - Threshold Predictor: ENABLED");
            println!("      → Learns per-token threshold for top-p selection");
            println!("    - Load Balance Weight: {}", load_balance_weight);
            println!("    - Complexity Loss Weight: {}", complexity_loss_weight);
            println!("    - Sparsity Weight: {}", sparsity_weight);
            println!("    - Estimated Compute Savings: ~{}%", compute_savings);
            println!("    - Expected Efficiency Gain: 15-25% (vs 5-8% for standard MoH)");
        }
    }

    // Architecture Alignment
    println!("\n🎯 Architecture Alignment:");
    let has_cope = matches!(
        config.positional_encoding,
        PositionalEncodingType::CoPE { .. }
    );

    if has_cope {
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

/// Legacy note: HRM architecture removed
///
/// This section previously described HRM-specific layer construction, which
/// has been removed. Supported architectures: Transformer.

// TRM architecture removed; only Transformer is supported now.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_transformer_network() {
        let vocab = Vocab::new(vec!["a", "b", "c"]);
        let config = ModelConfig::transformer(128, 256, 2, 80, None, Some(8));

        let layers = build_network(&config, &vocab);

        // Should have: Embeddings + (Attention + Norm + FF + Norm) * 2 + OutputProjection
        // Final norm deduped when last layer is already normalization
        // = 1 + 4*2 + 1 = 10 layers
        assert_eq!(layers.len(), 10);

        // Check first and last layers
        assert_eq!(layers[0].layer_type(), "Embeddings");
        assert_eq!(layers[layers.len() - 1].layer_type(), "OutputProjection");
    }

}


#[cfg(any())]
pub fn build_network(config: &ModelConfig, vocab_size: usize) -> Vec<LayerEnum> {
    match config.architecture {
        ArchitectureType::Transformer => build_transformer_layers(config, vocab_size),
    }
}

#[cfg(any())]
fn build_transformer_layers(config: &ModelConfig, vocab_size: usize) -> Vec<LayerEnum> {
    let mut layers: Vec<LayerEnum> = Vec::new();
    layers.push(LayerEnum::Embeddings(Embeddings::new(vocab_size, config.embedding_dim)));

    // Positional encoding could be inserted here based on config.positional_encoding
    match config.positional_encoding {
        PositionalEncodingType::Learned => {
            // TODO: implement learned positional embeddings if needed
        }
        PositionalEncodingType::CoPE { .. } => {
            // CoPE is integrated within attention modules as needed
        }
    }

    // Build attention + FFN blocks
    for _ in 0..config.num_layers {
        match config.attention {
            AttentionType::SelfAttention => {
                layers.push(LayerEnum::SelfAttention(SelfAttention::new(
                    config.get_num_heads(),
                    config.embedding_dim,
                    config.num_kv_heads,
                    config.window_size,
                    config.use_adaptive_window,
                    config.min_window_size,
                    config.max_window_size,
                    config.window_adaptation_strategy,
                    config.entropy_ema_alpha,
                    &config.head_selection,
                )));
            }
            AttentionType::PolyAttention { degree_p } => {
        let max_pos = match config.positional_encoding { crate::model_config::PositionalEncodingType::CoPE { max_pos } => max_pos };
                layers.push(LayerEnum::PolyAttention(PolyAttention::new(
                    config.embedding_dim,
                    config.get_num_heads(),
                    degree_p,
                    max_pos,
                    config.window_size,
                )));
            }
        }

        if config.use_dynamic_tanh_norm {
            layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));
        }

        layers.push(LayerEnum::SwiGLU(SwiGLU::new(config.embedding_dim, config.hidden_dim)));
    }

    // Output projection to vocab size
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(config.embedding_dim, vocab_size)));

    layers
}


