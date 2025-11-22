use clap::Parser;
use llm::{
    cli::Args,
    config_builder::build_model_config,
    dataset_loader::{Dataset, DatasetType},
    encoding::Vocab,
    errors::Result,
    interactive::run_interactive_mode,
    llm::LLM,
    model_builder::{build_network, print_architecture_summary},
    training::run_training_pipeline,
};

fn main() -> crate::Result<()> {
    let args = Args::parse();

    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Load dataset and build vocabulary
    let pre_path = String::from("data/pretraining_data.json");
    let chat_path = String::from("data/chat_training_data.json");
    let dataset = Dataset::new(pre_path.clone(), chat_path.clone(), DatasetType::JSON)?;

    let mut all_texts = Vec::new();
    all_texts.extend(dataset.pretraining_data.iter().cloned());
    all_texts.extend(dataset.chat_training_data.iter().cloned());
    let vocab = Vocab::build_from_texts(all_texts.iter());

    // Build model configuration
    let config = build_model_config(&args);

    // Build network based on configuration
    let network = build_network(&config, &vocab);

    // Print architecture summary
    print_architecture_summary(&config, &network);

    // Create or load LLM
    let mut llm = if let Some(model_path) = &args.continue_from {
        println!("\n=== LOADING EXISTING MODEL ===");
        println!("Loading model from: {}", model_path);
        LLM::load_versioned(model_path)?
    } else {
        LLM::new(vocab.clone(), network)
    };

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!("Total parameters: {}", llm.total_parameters());

    // Test prediction before training
    let test_input = "User: How do mountains form?";
    println!("\n=== BEFORE TRAINING ===");
    println!("Input: {}", test_input);
    println!("Output: {}", llm.predict(test_input));

    // Run training pipeline
    llm = run_training_pipeline(&args, &dataset, &vocab, &config, llm)?;

    // Save trained model to disk for inference
    std::fs::create_dir_all("models").ok();
    let save_path = "models/rustgpt.bin";
    llm.save_versioned(save_path, Some("RustGPT trained model".to_string()))?;
    println!("Saved model to {}", save_path);

    // Test prediction after training
    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", test_input);
    let result = llm.predict(test_input);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input (only if -i flag is provided)
    if args.interactive {
        run_interactive_mode(&mut llm)?;
    }

    Ok(())
}
