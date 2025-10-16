use std::io::Write;

use clap::Parser;
use llm::{
    build_network, print_architecture_summary, ArchitectureType, BeamSearchConfig, Dataset, DatasetType,
    ModelConfig, WindowAdaptationStrategy, LLM, Vocab, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN,
};

#[derive(Parser)]
#[command(name = "llm")]
#[command(about = "Train and run a language model")]
struct Args {
    /// Enable interactive prompt after training
    #[arg(short)]
    interactive: bool,
}

fn main() -> llm::Result<()> {
    let args = Args::parse();

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
    //let architecture = ArchitectureType::HyperMixer; // Change to HyperMixer for comparison
    let architecture = ArchitectureType::Transformer; // Change to HyperMixer for comparison

    // ============================================================================
    // NORMALIZATION CONFIGURATION
    // ============================================================================
    // Toggle between LayerNorm and RMSNorm for comparison
    //
    // LayerNorm: Standard normalization (mean centering + std normalization)
    //   - Parameters: gamma (scale) + beta (bias)
    //   - Used in original Transformer
    //
    // RMSNorm: Modern normalization (RMS normalization only)
    //   - Parameters: gamma (scale) only (50% reduction)
    //   - ~10-15% faster than LayerNorm
    //   - Better training stability
    //   - Used in LLaMA, PaLM, Mistral, GPT-NeoX
    // ============================================================================

    let use_rms_norm = true; // Set to true to use RMSNorm, false for LayerNorm

    // ============================================================================
    // FEEDFORWARD CONFIGURATION
    // ============================================================================
    // Toggle between FeedForward (ReLU) and SwiGLU for comparison
    //
    // FeedForward: Standard ReLU-based feedforward
    //   - Activation: ReLU (x → max(0, x))
    //   - Parameters: 2 weight matrices + 2 bias vectors
    //   - Can suffer from dead neurons
    //
    // SwiGLU: Modern gated linear unit with Swish activation
    //   - Activation: Swish (x → x * sigmoid(x))
    //   - Parameters: 3 weight matrices, no biases
    //   - Better gradient flow, enhanced capacity through gating
    //   - Used in LLaMA, PaLM, Mistral
    // ============================================================================

    let use_swiglu = true; // Set to true to use SwiGLU, false for FeedForward

    // ============================================================================
    // POSITIONAL ENCODING CONFIGURATION
    // ============================================================================
    // Toggle between learned positional embeddings and RoPE
    //
    // Learned Embeddings: Standard absolute positional embeddings
    //   - Parameters: max_seq_len × embedding_dim learned weights
    //   - Used in original Transformer, GPT-2, GPT-3
    //
    // RoPE (Rotary Positional Encoding): Geometric position encoding
    //   - Parameters: Zero (no learned weights)
    //   - Encodes relative position through rotation matrices
    //   - Better length extrapolation (handles longer sequences)
    //   - Used in LLaMA, PaLM, GPT-NeoX, Mistral
    // ============================================================================

    let use_rope = true; // Set to true to use RoPE, false for learned embeddings

    // ============================================================================
    // GROUP-QUERY ATTENTION (GQA) CONFIGURATION
    // ============================================================================
    // Toggle between Multi-Head Attention (MHA) and Group-Query Attention (GQA)
    //
    // MHA (Multi-Head Attention): Standard attention with num_heads KV heads
    //   - num_kv_heads = None (defaults to num_heads)
    //   - Each query head has its own key/value head
    //   - Used in original Transformer, GPT-2, GPT-3
    //
    // GQA (Group-Query Attention): Grouped attention with fewer KV heads
    //   - num_kv_heads = Some(n) where n < num_heads
    //   - Multiple query heads share the same key/value heads
    //   - Example: 8 query heads, 4 KV heads → 2 queries per KV head
    //   - Benefits:
    //     * Reduced KV cache size (e.g., 2x reduction with 8→4 heads)
    //     * Faster inference (smaller memory bandwidth)
    //     * Lower memory usage during generation
    //     * Minimal quality degradation vs MHA
    //   - Used in LLaMA 2 70B, Mistral 7B
    //
    // MQA (Multi-Query Attention): Extreme case with 1 KV head
    //   - num_kv_heads = Some(1)
    //   - All query heads share a single key/value head
    //   - Maximum KV cache reduction but potential quality loss
    // ============================================================================

    let num_kv_heads: Option<usize> = Some(4); // None for MHA, Some(4) for GQA, Some(1) for MQA

    // ============================================================================
    // SLIDING WINDOW ATTENTION CONFIGURATION
    // ============================================================================
    // Toggle between full attention and sliding window attention
    //
    // Full Attention: Standard attention (all tokens attend to all previous tokens)
    //   - window_size = None
    //   - Complexity: O(N²) where N is sequence length
    //   - Used in original Transformer, GPT-2, GPT-3, LLaMA 1/2
    //
    // Sliding Window Attention: Local attention with fixed window
    //   - window_size = Some(W) where W is the window size
    //   - Each token only attends to the last W tokens
    //   - Complexity: O(N × W) - much faster for long sequences
    //   - Benefits:
    //     * 2-10x faster for long sequences (depending on window size)
    //     * Enables 32k+ token context windows efficiently
    //     * Reduced memory usage: O(N × W) instead of O(N²)
    //     * Minimal quality degradation (local context often sufficient)
    //   - Used in Mistral 7B (window_size = 4096)
    //
    // Recommended configurations:
    //   - None: Full attention (baseline, best quality)
    //   - Some(4096): Mistral 7B style (32k context efficient)
    //   - Some(2048): Balanced (good for 16k contexts)
    //   - Some(1024): Aggressive (very fast, local context only)
    // ============================================================================

    let window_size: Option<usize> = Some(4096); // None for full attention, Some(4096) for Mistral-style

    // ============================================================================
    // ADAPTIVE WINDOW ATTENTION CONFIGURATION (Phase 4)
    // ============================================================================
    // Enable dynamic window sizing that adapts based on context
    //
    // When enabled, the window size automatically adjusts for each forward pass
    // based on the chosen strategy, within [min_window_size, max_window_size].
    //
    // Strategies:
    //   - SequenceLengthBased: window = seq_len / 2 (simple, stable, recommended)
    //   - AttentionEntropy: Adapts based on attention distribution
    //   - PerplexityBased: Adapts based on prediction confidence (future)
    //   - Fixed: Use configured window_size (Phase 3 behavior)
    //
    // Benefits:
    //   - Better resource utilization (smaller windows for short sequences)
    //   - Improved quality (larger windows when needed)
    //   - Automatic tuning (no manual window size selection)
    //
    // Recommended configurations:
    //   - use_adaptive_window = false: Phase 3 behavior (fixed window)
    //   - use_adaptive_window = true + SequenceLengthBased: General purpose
    //   - use_adaptive_window = true + AttentionEntropy: Advanced (context-aware)
    // ============================================================================

    let use_adaptive_window: bool = true; // Enable adaptive window sizing
    let min_window_size: usize = 512;      // Minimum window size
    let max_window_size: usize = 4096;     // Maximum window size
    let window_adaptation_strategy = WindowAdaptationStrategy::SequenceLengthBased;

    // ============================================================================
    // BEAM SEARCH CONFIGURATION (Phase 4 - Secondary Objective)
    // ============================================================================
    // Enable beam search for higher quality text generation
    //
    // Beam search explores multiple hypotheses in parallel, which can produce
    // better quality output than greedy decoding (which only picks the most
    // likely token at each step).
    //
    // Configuration options:
    //   - use_beam_search: Enable beam search (false = greedy decoding)
    //   - beam_width: Number of hypotheses to maintain (higher = better quality, slower)
    //   - use_adaptive_beam: Dynamically adjust beam width based on confidence
    //   - min_beam_width: Minimum beam width for adaptive beam search
    //   - max_beam_width: Maximum beam width for adaptive beam search
    //   - max_length: Maximum generation length
    //   - temperature: Sampling temperature (1.0 = no change, <1.0 = more confident)
    //
    // Recommended configurations:
    //   - use_beam_search = false: Greedy decoding (fastest, baseline)
    //   - beam_width = 4: Good balance of quality and speed
    //   - beam_width = 8: Higher quality, slower
    //   - use_adaptive_beam = true: Automatic beam width adjustment
    //
    // Benefits:
    //   - Better generation quality (explores multiple hypotheses)
    //   - Adaptive beam width reduces computation when model is confident
    //   - Configurable trade-off between quality and speed
    // ============================================================================

    let use_beam_search: bool = true;     // Enable beam search (false = greedy decoding)
    let beam_width: usize = 4;             // Number of hypotheses to maintain
    let use_adaptive_beam: bool = true;   // Enable adaptive beam width
    let min_beam_width: usize = 1;         // Minimum beam width
    let max_beam_width: usize = 8;         // Maximum beam width
    let beam_max_length: usize = 100;      // Maximum generation length
    let beam_temperature: f32 = 1.0;       // Sampling temperature

    // Create model configuration
    let mut config = match architecture {
        ArchitectureType::Transformer => {
            ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 3, MAX_SEQ_LEN, None, Some(8))
        }
        ArchitectureType::HyperMixer => {
            // HyperMixer with hypernetwork hidden dim = embedding_dim / 4 and 8 heads
            ModelConfig::hypermixer(EMBEDDING_DIM, HIDDEN_DIM, 3, MAX_SEQ_LEN, None, Some(8))
        }
        ArchitectureType::HRM => {
            // HRM with N=2 cycles, T=2 steps per cycle, hidden_dim=192 for parameter efficiency
            ModelConfig::hrm(EMBEDDING_DIM, 192, 2, 2, MAX_SEQ_LEN)
        }
    };

    // Apply modern LLM enhancements configuration
    config.use_rms_norm = use_rms_norm;
    config.use_swiglu = use_swiglu;
    config.use_rope = use_rope;
    config.num_kv_heads = num_kv_heads;
    config.window_size = window_size;
    config.use_adaptive_window = use_adaptive_window;
    config.min_window_size = min_window_size;
    config.max_window_size = max_window_size;
    config.window_adaptation_strategy = window_adaptation_strategy;

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

    llm.train_with_batch_size(pretraining_examples, 100, 0.0001, 4)?;

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    llm.train_with_batch_size(chat_training_examples, 100, 0.0001, 4)?; // Much lower learning rate for stability

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input (only if -i flag is provided)
    if args.interactive {
        println!("\n--- Interactive Mode ---");
        println!("Type a prompt and press Enter to generate text.");
        if use_beam_search {
            println!("Using beam search (beam_width={}, adaptive={})", beam_width, use_adaptive_beam);
        } else {
            println!("Using greedy decoding");
        }
        println!("Type 'exit' to quit.");

        // Create beam search config if enabled
        let beam_config = if use_beam_search {
            Some(BeamSearchConfig::new()
                .with_beam_width(beam_width)
                .with_adaptive_beam(use_adaptive_beam)
                .with_beam_range(min_beam_width, max_beam_width)
                .with_max_length(beam_max_length)
                .with_temperature(beam_temperature))
        } else {
            None
        };

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

            let prediction = if let Some(ref config) = beam_config {
                // Use beam search
                llm.generate_with_beam_search(&formatted_input, config)
            } else {
                // Use greedy decoding
                llm.predict(&formatted_input)
            };

            println!("Model output: {}", prediction);
        }
    }

    Ok(())
}
