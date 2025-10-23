use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, ModelConfig, Vocab, build_network, print_architecture_summary,
};

/// Demonstrate the Transformer model architecture available in RustGPT
///
/// This example shows the Transformer architecture with self-attention
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️ RustGPT Architecture Comparison Demo");
    println!("======================================\n");

    // Create configuration
    let config_transformer =
        ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 2, 80, None, Some(8));

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
