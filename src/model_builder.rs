use crate::{
    embeddings::Embeddings,
    llm::{Layer, LayerEnum},
    model_config::{ArchitectureType, AttentionType, ModelConfig, PositionalEncodingType},
    output_projection::OutputProjection,
    poly_attention::PolyAttention,
    swiglu::SwiGLU,
    vocab::Vocab,
    dynamic_tanh_norm::DynamicTanhNorm,
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

/// Build Transformer layers (Pre-LN): Norm -> Attn -> Norm -> SwiGLU, with final Norm
pub fn build_transformer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    // Build attention + FFN blocks for num_layers
    for _ in 0..config.num_layers {
        // Pre-LN before attention
        layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));

        match config.attention {
            AttentionType::SelfAttention => {
                panic!("SelfAttention not supported in this build; use PolyAttention");
            }
            AttentionType::PolyAttention { degree_p } => {
                // Use CoPE positional encoding integrated in PolyAttention; max_pos is ignored at runtime
                let max_pos = match config.positional_encoding { PositionalEncodingType::CoPE { max_pos } => max_pos };
                layers.push(LayerEnum::PolyAttention(Box::new(PolyAttention::new(
                    config.embedding_dim,
                    config.get_num_heads(),
                    degree_p,
                    max_pos,
                    config.window_size,
                ))));
            }
        }

        // Pre-LN before feed-forward
        layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));
        layers.push(LayerEnum::SwiGLU(Box::new(SwiGLU::new(config.embedding_dim, config.hidden_dim))));
    }

    // Final normalization before output projection (Pre-LN stability)
    layers.push(LayerEnum::DynamicTanhNorm(DynamicTanhNorm::new(config.embedding_dim)));
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
    match &config.positional_encoding {
        PositionalEncodingType::CoPE { .. } => {
            println!("  ✓ CoPE (Contextual Position Encoding)");
            let derived_span = config.window_size.unwrap_or(128);
            println!(
                "    - CoPE size: derived from {} (span = {})",
                if config.window_size.is_some() { "sliding window" } else { "tile" },
                derived_span
            );
        }
    }

    println!("\n🧠 Attention:");
    match &config.attention {
        AttentionType::PolyAttention { degree_p } => {
            println!("  ✓ Polynomial Attention (p = {})", degree_p);
            println!("    - Grouped-query heads: {}", config.get_num_heads());
            println!("    - Sliding window: {}", config.window_size.map(|w| w.to_string()).unwrap_or_else(|| "disabled".to_string()));
        }
        AttentionType::SelfAttention => {
            println!("  ✓ Scaled Dot-Product Self-Attention");
        }
    }

    println!("\n🧱 Layer Stack:");
    for (i, layer) in layers.iter().enumerate() {
        println!("  {}: {}", i, layer.layer_type());
    }

    // Parameter count summary
    let params: usize = layers.iter().map(|l| l.parameters()).sum();
    println!("\n🧮 Total Parameters: {}", params);
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
        let mut config = ModelConfig::transformer(128, 256, 2, 80, None, Some(8));
        config.attention = AttentionType::PolyAttention { degree_p: 3 };

        let layers = build_network(&config, &vocab);

        // Should have: Embeddings + (Norm + Attention + Norm + FF) * 2 + Final Norm + OutputProjection
        // = 1 + 4*2 + 1 + 1 = 11 layers
        assert_eq!(layers.len(), 11);

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


