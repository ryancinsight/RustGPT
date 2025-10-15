use std::io::Write;

use llm::{
    build_network, print_architecture_summary, ArchitectureType, Dataset, DatasetType,
    ModelConfig, LLM, Vocab, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN,
};

fn main() -> llm::Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // ============================================================================
    // ARCHITECTURE CONFIGURATION
    // ============================================================================
    // Toggle between Transformer and HyperMixer architectures for comparison
    //
    // Transformer: Standard self-attention based architecture
    //   - Uses Q, K, V matrices for token mixing
    //   - Quadratic complexity O(n²) in sequence length
    //   - Well-established, proven architecture
    //
    // HyperMixer: MLP-based architecture with dynamic token mixing
    //   - Uses hypernetworks to generate token-mixing weights dynamically
    //   - Linear complexity O(n) in sequence length
    //   - More parameter efficient than transformers
    //   - Better than static MLPMixer due to input-dependent mixing
    // ============================================================================

    // Choose architecture: Transformer or HyperMixer
    let architecture = ArchitectureType::HyperMixer; // Change to HyperMixer for comparison

    // Create model configuration
    let config = match architecture {
        ArchitectureType::Transformer => {
            ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 3, MAX_SEQ_LEN)
        }
        ArchitectureType::HyperMixer => {
            // HyperMixer with hypernetwork hidden dim = embedding_dim / 4
            ModelConfig::hypermixer(EMBEDDING_DIM, HIDDEN_DIM, 3, MAX_SEQ_LEN, None)
        }
    };

    // Mock input - test conversational format
    let string = String::from("User: How do mountains form?");

    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    )?;

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();

    // Process all training examples for vocabulary
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort(); // Sort for deterministic ordering
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s: &String| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // Build network based on configuration
    let network = build_network(&config, &vocab);

    // Print architecture summary
    print_architecture_summary(&config, &network);

    // Create LLM with the configured network
    let mut llm = LLM::new(vocab, network);

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!("Total parameters: {}", llm.total_parameters());

    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));

    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        100,
        0.0005
    );

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.train_with_batch_size(pretraining_examples, 100, 0.0005, 4);

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    llm.train_with_batch_size(chat_training_examples, 100, 0.0001, 4); // Much lower learning rate for stability

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input
    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        // Clear the input string
        input.clear();

        // Prompt for user input
        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();

        // Read user input
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        // Trim whitespace and check for exit command
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        // Generate prediction based on user input with "User:" prefix
        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict(&formatted_input);
        println!("Model output: {}", prediction);
    }

    Ok(())
}
