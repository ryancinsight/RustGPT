use std::io::Write;

use clap::Parser;
use llm::{
    ArchitectureType, AttentionType, Dataset, DatasetType, EMBEDDING_DIM, HIDDEN_DIM,
    ExpertRouter, HeadSelectionStrategy, LLM, MAX_SEQ_LEN, ModelConfig, Vocab,
    WindowAdaptationStrategy, build_network, print_architecture_summary,
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
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
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

    // Choose architecture: Transformer

    // let architecture = ArchitectureType::Transformer; // Standard transformer - TESTING FULLY
    // TRM (Tiny Recursive Model) - recursive reasoning with shared weights

    let architecture = ArchitectureType::TRM; // Tiny Recursive Model (weight sharing)

    let use_dynamic_tanh_norm = true;

    // ============================================================================
    // FEEDFORWARD CONFIGURATION
    // ============================================================================
    // Toggle between FeedForward (ReLU) and RichardsGlu for comparison
    //
    // FeedForward: Standard ReLU-based feedforward
    //   - Activation: ReLU (x → max(0, x))
    //   - Parameters: 2 weight matrices + 2 bias vectors
    //   - Can suffer from dead neurons
    //
    // RichardsGlu: Advanced gated linear unit with learned Richards activations
    //   - Activation: Learned Richards curves for both value and gate functions
    //   - Parameters: 3 weight matrices + Richards parameters, no biases
    //   - Adaptive activation functions that learn optimal gating behavior
    //   - Enhanced gradient flow and numerical stability
    // ============================================================================

    // ============================================================================
    // POSITIONAL ENCODING CONFIGURATION
    // ============================================================================
    // CoPE (Contextual Position Encoding): Context-aware position encoding
    // - Parameters: max_pos × head_dim learned position embeddings
    // - Positions conditioned on context via gating mechanism
    // - Can count abstract units (words, sentences, specific tokens)
    // - Better OOD generalization and perplexity than RoPE
    // - Used in research (Meta FAIR 2024)
    // ============================================================================

    // CoPE max_pos derives from sliding window size.

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
    let min_window_size: usize = 512; // Minimum window size
    let max_window_size: usize = 4096; // Maximum window size
    let window_adaptation_strategy = WindowAdaptationStrategy::AttentionEntropy;

    // ============================================================================
    // MIXTURE-OF-HEADS (MoH) CONFIGURATION
    // ============================================================================
    // Enable dynamic head selection for efficient attention computation
    //
    // Mixture-of-Heads (MoH) dynamically selects which attention heads to activate
    // per token using a learned routing mechanism. This reduces computation while
    // maintaining model quality.
    //
    // Based on "MoH: Multi-Head Attention as Mixture-of-Head Attention"
    // (Skywork AI, Oct 2024, arXiv:2410.11842)
    //
    // Architecture:
    //   - Shared Heads: Always active, capture common knowledge (25% of heads)
    //   - Routed Heads: Top-K selection per token, specialize for patterns (75% of heads)
    //   - Router Network: Learns to select which routed heads to activate
    //   - Load Balance Loss: Prevents routing collapse (all tokens → same heads)
    //
    // Configuration:
    //   - num_shared_heads: Number of shared heads (always active)
    //     * Recommended: 25% of total heads (e.g., 2 out of 8)
    //   - num_active_routed_heads: Number of routed heads to activate (Top-K)
    //     * Recommended: 50-75% of routed heads (e.g., 4 out of 6 routed)
    //   - load_balance_weight: Weight for load balance loss (β in paper)
    //     * Recommended: 0.01 (prevents routing collapse)
    //
    // Benefits:
    //   - 5-8% inference speedup (25% compute savings in attention)
    //   - <1% memory overhead (router parameters)
    //   - Minimal quality degradation (proven on ViT, DiT, LLMs)
    //   - Parameter-neutral design (router overhead compensated by efficiency)
    //
    // Parameter Budget (for 8 heads, 3 layers, embedding_dim=128):
    //   - Baseline: 573,440 parameters
    //   - Router: 3,840 parameters (+0.67%)
    //   - Total: 577,280 parameters (within ±2% budget)
    //
    // Recommended configurations:
    //   - AllHeads: Standard MHA (baseline, backward compatible)
    //   - MixtureOfHeads: Dynamic routing (5-8% speedup, recommended)
    //   - StaticPruning: Fixed head selection (ablation studies only)
    // ============================================================================

    // Head selection strategy (MoH vs standard MHA) - DISABLED for TRM
    let head_selection = {
        // ============================================================================
        // MIXTURE-OF-HEADS DISABLED for TRM
        // ============================================================================
        // TRM uses single shared transformer weights for recursive reasoning,
        // so MoH dynamic head selection is disabled to focus on core TRM functionality
        // ============================================================================
        HeadSelectionStrategy::Fixed {
            num_active: 8, // Use all heads for TRM stability
        }

        // Alternative: MoH configurations (uncomment to re-enable):
        // HeadSelectionStrategy::SoftTopP {
        //     top_p: 0.9,                  // Use 90% probability mass for head selection
        // HeadSelectionStrategy::MixtureOfHeads {
        //     num_shared_heads: 2,
        //     num_active_routed_heads: 4,
        //     load_balance_weight: 0.01,
        //     threshold_p_base: 0.4,
        //     dynamic_loss_weight_base: 5e-5,
        //     use_learned_threshold: true,
        //     target_avg_routed_heads: 3.5,
        //     confidence_threshold: 0.4,
        //     use_confidence_fallback: false,
        // }
    };

    // Alternative configurations: (Mixture-of-Heads variants only)

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

    // Create model configuration
    let mut config = match architecture {
        ArchitectureType::Transformer => {
            ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 1, MAX_SEQ_LEN, None, Some(8))
        }
        ArchitectureType::TRM => {
            let mut config = ModelConfig::transformer(EMBEDDING_DIM, HIDDEN_DIM, 1, MAX_SEQ_LEN, None, Some(8));
            config.architecture = ArchitectureType::TRM;
            config
        }
    };

    // Apply modern LLM enhancements configuration
    config.use_dynamic_tanh_norm = use_dynamic_tanh_norm;

    config.num_kv_heads = num_kv_heads;
    config.window_size = window_size;
    config.use_adaptive_window = use_adaptive_window;
    config.min_window_size = min_window_size;
    config.max_window_size = max_window_size;
    config.window_adaptation_strategy = window_adaptation_strategy;
    config.head_selection = head_selection;

    // Set attention mechanism: use PolyAttention as an alternative to SelfAttention
    config.attention = AttentionType::PolyAttention { degree_p: 3 };

    // ============================================================================
    // MIXTURE OF EXPERTS (MoE) CONFIGURATION
    // ============================================================================
    // Enable sparse MoE for increased model capacity
    //
    // When enabled, replaces standard feedforward layers with sparse MoE layers
    // Each MoE layer contains multiple expert networks with learned routing
    // ============================================================================

    // ============================================================================
    // MIXTURE OF EXPERTS (MoE) - DISABLED for TRM
    // ============================================================================
    // TRM uses single shared transformer weights for recursive reasoning,
    // so MoE/MoH complexity is disabled to focus on core TRM functionality
    // ============================================================================

    // MoE disabled for TRM - TRM uses single shared weights
    config.moe_router = None;

    // Uncomment to re-enable MoE (not recommended with TRM):
    // config.moe_router = Some(ExpertRouter::LearnedMoE {
    //     num_experts: 4,                    // 4 experts (balanced capacity)
    //     num_active_experts: 2,             // Top-2 routing (Mixtral-style)
    //     expert_hidden_dim: 64,             // Smaller experts (128/2 = 64)
    //     load_balance_weight: 0.01,         // Prevent expert collapse
    //     sparsity_weight: 0.001,            // Encourage minimal activation
    //     diversity_weight: 0.005,           // Encourage expert specialization
    // });


    // Mock input - test conversational format
    let string = String::from("User: How do mountains form?");

    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    )?;

    // Build vocabulary from training data
    let all_texts = dataset.pretraining_data.iter()
        .chain(dataset.chat_training_data.iter())
        .cloned();

    let vocab = Vocab::build_from_texts(all_texts);

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

    llm.train_with_batch_size(pretraining_examples, 100, 0.001, 256)?;

    println!("\n=== INSTRUCTION TUNING ===");
    // Use same LR as pre-training
    let instruction_lr = 0.002;
    let instruction_epochs = 100;
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        instruction_epochs,
        instruction_lr
    );

    // Use warmup + cosine annealing for instruction tuning stability
    let warmup_epochs = 25;
    llm.train_with_warmup(
        chat_training_examples,
        instruction_epochs,
        instruction_lr,
        64,
        warmup_epochs,
    )?;

    // Save trained model to disk for inference
    std::fs::create_dir_all("models").ok();
    let save_path = "models/rustgpt.bin";
    llm.save_versioned(save_path, Some("RustGPT trained model".to_string()))?;
    println!("Saved model to {}", save_path);

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    let result = llm.predict(&string);
    println!("Output: {}", result);
    println!("======================\n");

    // Interactive mode for user input (only if -i flag is provided)
    if args.interactive {
        println!("\n--- Interactive Mode ---");
        println!("Type a prompt and press Enter to generate text.");
        println!("Using greedy decoding");
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
    }

    Ok(())
}
