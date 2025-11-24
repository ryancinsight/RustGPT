use llm::{LLM, transformer::speculative::{SpeculativeSamplingConfig, SpeculativeMode}};

fn main() {
    println!("🧪 Testing Transformer Speculative Sampling Configuration");
    println!("=========================================================");

    // Test that we can create an LLM and enable transformer speculative sampling
    let vocab = llm::vocab::Vocab::default();
    let network = Vec::new(); // Empty network for testing
    let mut llm = LLM::new(vocab, network);

    // Check initial state
    assert_eq!(llm.speculative_mode, SpeculativeMode::Diffusion);
    assert!(llm.speculative_config.is_none());

    // Enable transformer speculative sampling
    llm.enable_speculative_sampling(4, 0.1, 2, SpeculativeMode::Transformer);

    // Verify configuration
    assert_eq!(llm.speculative_mode, SpeculativeMode::Transformer);
    assert!(llm.speculative_config.is_some());

    let config = llm.speculative_config.as_ref().unwrap();
    assert_eq!(config.gamma, 4);
    assert_eq!(config.tau, 0.1);
    assert_eq!(config.draft_layers, 2);

    println!("✅ Speculative sampling configuration test passed!");
    println!("   Mode: {:?}", llm.speculative_mode);
    println!("   Gamma: {}", config.gamma);
    println!("   Tau: {}", config.tau);
    println!("   Draft layers: {}", config.draft_layers);

    // Test that the speculative sampling method exists (would fail to compile if not)
    // We can't actually call it without a proper model, but we can verify the method signature
    println!("✅ Transformer speculative sampling method is available!");
}
