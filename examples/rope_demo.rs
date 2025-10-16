use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, ModelConfig,
    build_network, print_architecture_summary, Vocab, LLM
};

/// Demonstrate different model architectures available in RustGPT
///
/// This example shows the current model architectures supported:
/// - Standard Transformer with self-attention
/// - HyperMixer with dynamic token mixing via hypernetworks
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️ RustGPT Architecture Comparison Demo");
    println!("======================================\n");

    // Create configurations for comparison
    let config_transformer = ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 2, 80, None, Some(8));
    let config_hypermixer = ModelConfig::hypermixer(EMBEDDING_DIM, HIDDEN_DIM, 2, 80, None, Some(8));

    println!("Configuration Comparison:");
    println!("------------------------");
    println!("Transformer: {:?}", config_transformer.architecture);
    println!("HyperMixer:  {:?}", config_hypermixer.architecture);
    println!("Embedding Dim: {}", config_transformer.embedding_dim);
    println!("Hidden Dim: {}", config_transformer.hidden_dim);
    println!("Num Layers: {}", config_transformer.num_layers);
    println!();

    // Use default vocab which includes necessary tokens like </s>
    let vocab = Vocab::default();

    // Build networks
    println!("Building Networks:");
    println!("------------------");
    let network_transformer = build_network(&config_transformer, &vocab);
    println!("Transformer Network: {} layers", network_transformer.len());

    let network_hypermixer = build_network(&config_hypermixer, &vocab);
    println!("HyperMixer Network:  {} layers", network_hypermixer.len());
    println!();

    // Print architecture details
    println!("Architecture Details (Transformer):");
    println!("-----------------------------------");
    print_architecture_summary(&config_transformer, &network_transformer);
    println!();

    println!("Architecture Details (HyperMixer):");
    println!("----------------------------------");
    print_architecture_summary(&config_hypermixer, &network_hypermixer);
    println!();

    // Create LLMs for testing
    let mut llm_transformer = LLM::new(vocab.clone(), network_transformer);
    let mut llm_hypermixer = LLM::new(vocab, network_hypermixer);

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

        // Generate with both models
        let output_transformer = llm_transformer.predict(prompt);
        let output_hypermixer = llm_hypermixer.predict(prompt);

        println!("Transformer: {}", output_transformer);
        println!("HyperMixer:  {}", output_hypermixer);
        println!();
    }

    println!("🏗️ Architecture Comparison:");
    println!("===========================");
    println!("• Transformer: Uses self-attention for token relationships");
    println!("• HyperMixer: Uses hypernetworks for dynamic token mixing");
    println!("• Both support multi-head attention and layer normalization");
    println!("• Choose based on your specific use case and performance needs");

    Ok(())
}
