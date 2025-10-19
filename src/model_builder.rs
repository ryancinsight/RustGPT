use crate::{
    attention_moe::AttentionMoELayer,
    embeddings::Embeddings,
    feed_forward::FeedForward,
    hrm::HRMBlock,
    hypermixer::HyperMixerBlock,
    layer_norm::LayerNorm,
    llm::{Layer, LayerEnum},
    moe::MoELayer,
    model_config::{ArchitectureType, ModelConfig},
    output_projection::OutputProjection,
    rms_norm::RMSNorm,
    self_attention::SelfAttention,
    swiglu::SwiGLU,
    trm::TinyRecursiveModel,
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
        ArchitectureType::TRM => {
            build_trm_layers(&mut layers, config);
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

    for layer_idx in 0..config.num_layers {
        // Check if we should use hierarchical MoH-in-MoE for attention
        if config.use_moh_in_experts {
            // Use AttentionMoE: MoE of attention experts, each with MoH
            let attention_moe = AttentionMoELayer::new(
                config.embedding_dim,
                config.num_experts,
                config.num_active_experts,
                config.expert_moh_num_shared_heads,
                config.expert_moh_num_routed_heads,
                num_kv_heads,
                config.expert_moh_use_learned_threshold,
                config.moe_load_balance_weight,
                config.moe_router_z_loss_weight,
            );

            layers.push(LayerEnum::AttentionMoE(Box::new(attention_moe)));
        } else {
            // Standard self-attention layer (with positional encoding, GQA, Sliding Window, and Adaptive Window)
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
            // Pass layer_idx for layer-wise adaptive thresholds
            attention.set_head_selection(config.head_selection.clone(), layer_idx);

            // Gradient clipping is handled globally in the training loop via AdaptiveGradientClipper
            // Per-layer gradient clipping is disabled to avoid double-clipping and maintain
            // consistent gradient flow across all layers

            layers.push(LayerEnum::SelfAttention(Box::new(attention)));
        }

        // Normalization after attention (LayerNorm or RMSNorm based on config)
        if config.use_rms_norm {
            layers.push(LayerEnum::RMSNorm(RMSNorm::new(config.embedding_dim)));
        } else {
            layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
        }

        // Feedforward layer (MoE, SwiGLU, or FeedForward based on config)
        if config.use_moe {
            // Use Mixture of Experts
            layers.push(LayerEnum::MoE(Box::new(MoELayer::new(
                config.embedding_dim,
                config.expert_hidden_dim,
                config.num_experts,
                config.num_active_experts,
                config.moe_load_balance_weight,
                config.moe_router_z_loss_weight,
            ))));
        } else if config.use_swiglu {
            // Use SwiGLU
            layers.push(LayerEnum::SwiGLU(Box::new(SwiGLU::new(
                config.embedding_dim,
                config.hidden_dim,
            ))));
        } else {
            // Use standard FeedForward
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

/// Build TRM (Tiny Recursive Model) architecture layers
///
/// Creates a parameter-efficient architecture with weight sharing across depth.
/// A single transformer block is applied recursively multiple times.
///
/// # Architecture
/// - Single transformer block (attention + FFN + norms)
/// - Applied recursively D times (recursive depth)
/// - Adaptive residual scaling per step (learned)
/// - Per-step gradient tracking
///
/// # TRM Configuration
/// - `num_layers` stores recursive depth (D)
/// - `use_swiglu` determines FFN type (SwiGLU vs FeedForward)
/// - `use_rms_norm` determines normalization type (RMSNorm vs LayerNorm)
///
/// # Gradient Stability
/// - Adaptive residual scaling prevents vanishing/exploding gradients
/// - Per-step gradient tracking for analysis
/// - No gradient clipping required (handled by adaptive mechanisms)
fn build_trm_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let recursive_depth = config.get_recursive_depth();
    let num_heads = config.get_num_heads();
    let num_kv_heads = config.get_num_kv_heads();

    // Create single TRM block (contains recursive depth internally)
    let trm_block = TinyRecursiveModel::new(
        config.embedding_dim,
        config.hidden_dim,
        num_heads,
        Some(num_kv_heads),
        recursive_depth,
        config.use_swiglu,
        config.max_seq_len,
    );

    layers.push(LayerEnum::TRMBlock(Box::new(trm_block)));

    // Add final normalization layer (critical for Pre-LN stability)
    if config.use_rms_norm {
        layers.push(LayerEnum::RMSNorm(RMSNorm::new(config.embedding_dim)));
    } else {
        layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
    }
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
