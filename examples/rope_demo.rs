use llm::domain::models::llm::LLM;
use llm::domain::models::config::ModelConfig;
use llm::application::encoding::Vocab;
use llm::domain::models::builder::{build_network, print_architecture_summary};

/// Demonstrate the Transformer model architecture available in RustGPT
///
/// This example shows the Transformer architecture with self-attention
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️ RustGPT Architecture Comparison Demo");
    println!("======================================\n");

    // Create configuration
    let base_config = ModelConfig::default();
    let config_transformer = ModelConfig::transformer(
        base_config.embedding_dim,
        base_config.hidden_dim,
        2,
        base_config.max_seq_len,
        base_config.hypernetwork_hidden_dim,
        base_config.num_heads,
    );

    println!("Configuration:");
    println!("-------------");
    println!("Architecture: {:?}", config_transformer.architecture);
    println!("Embedding Dim: {}", config_transformer.embedding_dim);
    println!("Hidden Dim: {}", config_transformer.hidden_dim);
    println!("Num Layers: {}", config_transformer.num_layers);
    println!();

    // Use default vocab which includes necessary tokens like </s>
    let vocab = Vocab::default();

    // Build network
    println!("Building Network:");
    println!("-----------------");
    let network_transformer = build_network(&config_transformer, &vocab);
    println!("Network: {} layers", network_transformer.len());
    println!();

    // Print architecture details
    println!("Architecture Details:");
    println!("---------------------");
    print_architecture_summary(&config_transformer, &network_transformer);
    println!();

    // Create LLM for testing
    let mut llm_transformer = LLM::new(vocab, network_transformer);

    // Test with different prompts to show architecture differences
    let test_prompts = vec![
        "hello world",
        "the sun rises",
        "water flows",
        "mountains are tall",
    ];

    println!("Generation Comparison:");
    println!("======================");

    for prompt in &test_prompts {
        println!("Prompt: \"{}\"", prompt);

        // Generate with the model
        let output_transformer = llm_transformer.predict(prompt);

        println!("Output: {}", output_transformer);
        println!();
    }

    println!("🏗️ Architecture:");
    println!("================");
    println!("• Transformer: Uses self-attention for token relationships");
    println!("• Supports multi-head attention and layer normalization");

    Ok(())
}
