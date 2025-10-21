use llm::{
    EMBEDDING_DIM, HIDDEN_DIM, LLM, ModelConfig, Vocab, build_network, print_architecture_summary,
};

/// Demonstrate different model architectures available in RustGPT
///
/// This example shows the current model architectures supported:
/// - Standard Transformer with self-attention
/// - TRM (Tiny Recursive Model) with weight sharing across depth
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🏗️ RustGPT Architecture Comparison Demo");
    println!("======================================\n");

    // Create configurations for comparison
    let config_transformer =
        ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 2, 80, None, Some(8));
    let config_trm =
        ModelConfig::trm(EMBEDDING_DIM, HIDDEN_DIM, 5, 80, Some(8));

    println!("Configuration Comparison:");
    println!("------------------------");
    println!("Transformer: {:?}", config_transformer.architecture);
    println!("TRM:         {:?}", config_trm.architecture);
    println!("Transformer Embedding Dim: {}", config_transformer.embedding_dim);
    println!("Transformer Hidden Dim: {}", config_transformer.hidden_dim);
    println!("Transformer Num Layers: {}", config_transformer.num_layers);
    println!("TRM Embedding Dim: {}", config_trm.embedding_dim);
    println!("TRM Hidden Dim: {}", config_trm.hidden_dim);
    println!("TRM Recursive Depth: {}", config_trm.num_layers);
    println!();

    // Use default vocab which includes necessary tokens like </s>
    let vocab = Vocab::default();

    // Build networks
    println!("Building Networks:");
    println!("------------------");
    let network_transformer = build_network(&config_transformer, &vocab);
    println!("Transformer Network: {} layers", network_transformer.len());

    let network_trm = build_network(&config_trm, &vocab);
    println!("TRM Network:         {} layers", network_trm.len());
    println!();

    // Print architecture details
    println!("Architecture Details (Transformer):");
    println!("-----------------------------------");
    print_architecture_summary(&config_transformer, &network_transformer);
    println!();

    println!("Architecture Details (TRM):");
    println!("----------------------------");
    print_architecture_summary(&config_trm, &network_trm);
    println!();

    // Create LLMs for testing
    let mut llm_transformer = LLM::new(vocab.clone(), network_transformer);
    let mut llm_trm = LLM::new(vocab, network_trm);

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
        let output_trm = llm_trm.predict(prompt);

        println!("Transformer: {}", output_transformer);
        println!("TRM:         {}", output_trm);
        println!();
    }

    println!("🏗️ Architecture Comparison:");
    println!("===========================");
    println!("• Transformer: Uses self-attention for token relationships");
    println!("• TRM: Uses a single block recursively with weight sharing");
    println!("• Both support multi-head attention and layer normalization");
    println!("• Choose based on your specific use case and performance needs");

    Ok(())
}
