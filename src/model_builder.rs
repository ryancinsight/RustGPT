use crate::{
    embeddings::Embeddings,
    feed_forward::FeedForward,
    hypermixer::HyperMixerBlock,
    layer_norm::LayerNorm,
    llm::{Layer, LayerEnum},
    model_config::{ArchitectureType, ModelConfig},
    output_projection::OutputProjection,
    self_attention::SelfAttention,
    vocab::Vocab,
};

/// Build a network based on the provided configuration
/// 
/// This function constructs either a Transformer or HyperMixer architecture
/// based on the configuration, allowing for easy A/B comparison between
/// the two approaches.
/// 
/// # Arguments
/// * `config` - Model configuration specifying architecture and hyperparameters
/// * `vocab` - Vocabulary for embeddings and output projection
/// 
/// # Returns
/// Vector of layers that form the complete network
pub fn build_network(config: &ModelConfig, vocab: &Vocab) -> Vec<LayerEnum> {
    let mut layers = Vec::new();
    
    // Add embedding layer (common to both architectures)
    layers.push(LayerEnum::Embeddings(Embeddings::new(vocab.clone())));
    
    // Build architecture-specific layers
    match config.architecture {
        ArchitectureType::Transformer => {
            build_transformer_layers(&mut layers, config);
        }
        ArchitectureType::HyperMixer => {
            build_hypermixer_layers(&mut layers, config);
        }
    }
    
    // Add output projection layer (common to both architectures)
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(
        config.embedding_dim,
        vocab.words.len(),
    )));
    
    layers
}

/// Build Transformer architecture layers
/// 
/// Creates the standard transformer architecture with:
/// - Self-attention layers
/// - Layer normalization
/// - Feedforward networks
/// - Residual connections (handled within layers)
fn build_transformer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let num_heads = config.get_num_heads();

    for _ in 0..config.num_layers {
        // Self-attention layer
        layers.push(LayerEnum::SelfAttention(Box::new(SelfAttention::new_with_heads(
            config.embedding_dim,
            num_heads,
        ))));
        
        // Layer normalization after attention
        layers.push(LayerEnum::LayerNorm(LayerNorm::new(config.embedding_dim)));
        
        // Feedforward layer
        layers.push(LayerEnum::FeedForward(Box::new(FeedForward::new(
            config.embedding_dim,
            config.hidden_dim,
        ))));
        
        // Layer normalization after feedforward
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
        let mut hypermixer_block = HyperMixerBlock::new(
            config.embedding_dim,
            config.hidden_dim,
            config.max_seq_len,
            hypernetwork_hidden_dim,
            num_heads,
        );

        // Enable gradient clipping for training stability
        hypermixer_block.enable_gradient_clipping(5.0);

        layers.push(LayerEnum::HyperMixerBlock(Box::new(hypermixer_block)));
    }
}

/// Print architecture summary
/// 
/// Displays information about the constructed network for debugging
/// and comparison purposes.
pub fn print_architecture_summary(config: &ModelConfig, layers: &[LayerEnum]) {
    println!("\n=== Model Architecture Summary ===");
    println!("Architecture Type: {:?}", config.architecture);
    println!("Embedding Dimension: {}", config.embedding_dim);
    println!("Hidden Dimension: {}", config.hidden_dim);
    println!("Number of Layers: {}", config.num_layers);
    println!("Max Sequence Length: {}", config.max_seq_len);
    
    if config.architecture == ArchitectureType::HyperMixer {
        println!(
            "Hypernetwork Hidden Dim: {}",
            config.get_hypernetwork_hidden_dim()
        );
    }
    
    println!("\nLayer Stack:");
    for (idx, layer) in layers.iter().enumerate() {
        println!("  {}: {}", idx, layer.layer_type());
    }
    
    let total_params: usize = layers.iter().map(|l| l.parameters()).sum();
    println!("\nTotal Parameters: {}", total_params);
    println!("==================================\n");
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

