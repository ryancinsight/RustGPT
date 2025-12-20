use crate::{
    embeddings::TokenEmbeddings,
    encoding::Vocab,
    layers::{
        diffusion::{DiffusionBlock, DiffusionBlockConfig, EDM_SIGMA_DATA_DEFAULT, NoiseSchedule},
        recurrence::LRM,
        transformer::TransformerBlock,
    },
    model_config::{ArchitectureType, ModelConfig},
    network::{Layer, LayerEnum},
    output_projection::OutputProjection,
    richards::RichardsNorm,
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
    // Position embeddings are handled inside attention (CoPE), so only token embeddings
    layers.push(LayerEnum::TokenEmbeddings(TokenEmbeddings::new(
        vocab.clone(),
    )));

    // Build architecture-specific layers
    match config.architecture {
        ArchitectureType::Transformer => {
            build_transformer_layers(&mut layers, config);
        }
        ArchitectureType::TRM => {
            build_trm_layers(&mut layers, config);
        }
        ArchitectureType::Diffusion => {
            build_diffusion_layers(&mut layers, config, vocab);
        }
    }

    // Add output projection layer (common to all architectures)
    layers.push(LayerEnum::OutputProjection(OutputProjection::new(
        config.embedding_dim,
        vocab.size(),
    )));

    // Set TRM/LRM layers to inference mode by default for speed
    for layer in &mut layers {
        if let LayerEnum::LRM(lrm) = layer {
            lrm.set_training_mode(false);
        }
    }

    layers
}

/// Build Diffusion Transformer architecture layers
///
/// Creates a diffusion-based transformer architecture where each layer
/// is a DiffusionBlock that performs denoising conditioned on timestep.
/// The architecture follows the same structure as standard transformers
/// but predicts noise instead of next tokens.
fn build_diffusion_layers(
    layers: &mut Vec<LayerEnum>,
    config: &ModelConfig,
    vocab: &crate::encoding::Vocab,
) {
    for _layer_idx in 0..config.num_layers {
        // Build LLaDA-style masked diffusion block config
        let max_pos = if config.use_adaptive_window {
            config.max_window_size
        } else if let Some(w) = config.window_size {
            w
        } else {
            config.max_seq_len
        }
        .saturating_sub(1);

        let mask_id = vocab
            .encode("<mask>")
            .or_else(|| vocab.encode_or_unknown("<mask>"))
            .unwrap_or_else(|| vocab.encode_or_unknown("<unk>").unwrap_or(0));

        let block_cfg = DiffusionBlockConfig {
            embed_dim: config.embedding_dim,
            hidden_dim: config.hidden_dim,
            num_heads: config.get_num_heads(),
            poly_degree: config.get_poly_degree_p(),
            max_pos,
            window_size: config.window_size,
            use_moe: config.moe_router.is_some(),
            moe_config: config
                .moe_router
                .as_ref()
                .map(|router| crate::mixtures::moe::ExpertRouterConfig::from_router(router)),
            head_selection: config.head_selection.clone(),
            time_embed_dim: config.embedding_dim,
            num_timesteps: 1000,
            noise_schedule: config.diffusion_noise_schedule.clone(),
            causal_attention: false,
            discrete_masked: true,
            use_adaptive_window: config.use_adaptive_window,
            mask_token_id: Some(mask_id),
            prediction_target: config.diffusion_prediction_target.clone(),
            edm_sigma_data: EDM_SIGMA_DATA_DEFAULT,
            timestep_strategy: config.diffusion_timestep_strategy,
            temporal_mixing: config.temporal_mixing,
            use_advanced_adaptive_residuals: true, // Enable by default for diffusion blocks
        };

        let diffusion_block = DiffusionBlock::new(block_cfg);
        layers.push(LayerEnum::DiffusionBlock(Box::new(diffusion_block)));
    }

    // Final normalization layer prior to logits projection (typical Pre-LN pattern)
    layers.push(LayerEnum::DynamicTanhNorm(
        crate::richards::RichardsNorm::new(config.embedding_dim),
    ));
}

/// Build Transformer architecture layers
///
/// Creates a Pre-LN-style transformer architecture using consolidated TransformerBlock components.
/// Each TransformerBlock encapsulates:
/// - Pre-attention normalization
/// - Attention mechanism (PolyAttention with CoPE)
/// - Pre-feedforward normalization
/// - Feedforward network (RichardsGlu or MixtureOfExperts)
/// - Residual connections
fn build_transformer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    for layer_idx in 0..config.num_layers {
        // Create a complete transformer block that encapsulates all components
        let transformer_block = TransformerBlock::from_model_config(config, layer_idx);
        layers.push(LayerEnum::TransformerBlock(Box::new(transformer_block)));
    }

    // Final normalization layer prior to logits projection (typical Pre-LN pattern)
    layers.push(LayerEnum::DynamicTanhNorm(RichardsNorm::new(
        config.embedding_dim,
    )));
}

/// Build TRM (Tiny Recursive Model) layers
///
/// Creates a single TRM layer that handles recursive reasoning internally.
/// TRM uses shared weights across recursive operations for efficient reasoning.
fn build_trm_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let lrm = LRM::from_model_config(config);
    layers.push(LayerEnum::LRM(Box::new(lrm)));
    layers.push(LayerEnum::DynamicTanhNorm(RichardsNorm::new(
        config.embedding_dim,
    )));
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

    match config.architecture {
        ArchitectureType::Transformer => {
            println!("  Number of Layers: {}", config.num_layers);
        }
        ArchitectureType::TRM => {
            println!("  Recursions per Step: {}", 2); // From TRM config
            println!("  Max Supervision Steps: {}", 16); // Training mode
            println!("  Max Inference Steps: {}", 3); // Inference mode (much faster)
            println!(
                "  TRM Mode: {}",
                if config.trm_use_diffusion {
                    "Diffusion"
                } else {
                    "Autoregressive"
                }
            );
        }
        ArchitectureType::Diffusion => {
            println!("  Number of Layers: {}", config.num_layers);
            println!("  Diffusion Timesteps: 1000");
            let schedule_label = match &config.diffusion_noise_schedule {
                NoiseSchedule::Cosine { .. } => "Cosine (Improved DDPM)",
                NoiseSchedule::Linear { .. } => "Linear",
                NoiseSchedule::Quadratic { .. } => "Quadratic",
                NoiseSchedule::Karras { .. } => "Karras (σ-schedule mapped to VP)",
            };
            println!("  Noise Schedule: {}", schedule_label);
            println!(
                "  Timestep Sampling: {:?}",
                config.diffusion_timestep_strategy
            );
        }
    }

    println!("  Max Sequence Length: {}", config.max_seq_len);

    // Modern LLM Enhancements
    println!("\n🚀 Modern LLM Enhancements:");

    // Normalization
    println!("  ✓ DynamicTanhNorm (adaptive, tanh-based)");

    // Activation
    println!("  ✓ RichardsGlu (learned Richards gated activation, no bias)");

    // Positional Encoding (CoPE always on; max_pos derived from window)
    let effective_window = if config.use_adaptive_window {
        config.max_window_size
    } else if let Some(w) = config.window_size {
        w
    } else {
        config.max_seq_len
    };
    let cope_max_pos = effective_window.saturating_sub(1);
    println!("  ✓ CoPE (Contextual Position Encoding)");
    println!("    - Max Position (derived): {}", cope_max_pos);

    println!("\n🧠 Attention:");
    use crate::model_config::AttentionType;
    match &config.attention {
        AttentionType::PolyAttention { degree_p } => {
            println!("  ✓ Polynomial Attention (p = {})", degree_p);
            println!("    - Grouped-query heads: {}", config.get_num_heads());
            println!(
                "    - Sliding window: {}",
                config
                    .window_size
                    .map(|w: usize| w.to_string())
                    .unwrap_or_else(|| "disabled".to_string())
            );
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
    use crate::{
        layers::diffusion::{DiffusionPredictionTarget, NoiseSchedule},
        model_config::DiffusionTimestepStrategy,
    };

    #[test]
    fn test_build_transformer_network() {
        let vocab = Vocab::new(vec!["a", "b", "c"]);
        let config = ModelConfig::transformer(128, 256, 1, 80, None, Some(8));

        let layers = build_network(&config, &vocab);

        // Should have: Embeddings + TransformerBlock * 1 + Final Norm + OutputProjection
        // = 1 + 1 + 1 + 1 = 4 layers
        assert_eq!(layers.len(), 4);

        // Check first and last layers
        assert_eq!(layers[0].layer_type(), "TokenEmbeddings");
        assert_eq!(layers[1].layer_type(), "TransformerBlock");
        assert_eq!(layers[2].layer_type(), "RichardsNorm");
        assert_eq!(layers[layers.len() - 1].layer_type(), "OutputProjection");
    }

    #[test]
    fn test_build_diffusion_network_uses_prediction_target() {
        let vocab = Vocab::new(vec!["<pad>", "<mask>", "hello"]);
        let mut config = ModelConfig::transformer(64, 128, 1, 64, None, Some(4));
        config.architecture = ArchitectureType::Diffusion;
        config.diffusion_prediction_target = DiffusionPredictionTarget::VPrediction;

        let layers = build_network(&config, &vocab);
        let prediction = layers
            .iter()
            .find_map(|layer| match layer {
                LayerEnum::DiffusionBlock(block) => Some(block.prediction_target()),
                _ => None,
            })
            .expect("diffusion block not found");

        assert_eq!(prediction, DiffusionPredictionTarget::VPrediction);
    }

    #[test]
    fn test_diffusion_network_inherits_schedule_and_sampling() {
        let vocab = Vocab::new(vec!["<pad>", "<mask>", "world"]);
        let mut config = ModelConfig::transformer(32, 64, 1, 32, None, Some(4));
        config.architecture = ArchitectureType::Diffusion;
        config.diffusion_noise_schedule = NoiseSchedule::Linear {
            beta_min: 1e-4,
            beta_max: 0.02,
        };
        config.diffusion_timestep_strategy = DiffusionTimestepStrategy::MinSnr;

        let layers = build_network(&config, &vocab);
        let mut found = false;
        for layer in &layers {
            if let LayerEnum::DiffusionBlock(block) = layer {
                match block.noise_schedule() {
                    NoiseSchedule::Linear { beta_min, beta_max } => {
                        assert!((*beta_min - 1e-4).abs() < f32::EPSILON);
                        assert!((*beta_max - 0.02).abs() < f32::EPSILON);
                    }
                    other => panic!("unexpected schedule: {:?}", other),
                }
                assert_eq!(block.timestep_strategy(), DiffusionTimestepStrategy::MinSnr);
                found = true;
            }
        }
        assert!(found, "diffusion block not constructed");
    }
}
