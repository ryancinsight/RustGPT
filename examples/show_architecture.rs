/// Example: Display RustGPT Architecture Summary
///
/// This example demonstrates the modern LLM architecture configurations
/// available in RustGPT and displays detailed architecture summaries.
use llm::{ModelConfig, PositionalEncodingType, Vocab, build_network, print_architecture_summary};

fn main() {
    println!("\n🦀 RustGPT Architecture Showcase\n");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create a simple vocabulary for demonstration
    let vocab = Vocab::new(vec!["<pad>", "hello", "world"]);

    // Configuration 1: Original Transformer (Baseline)
    println!("📋 Configuration 1: Original Transformer (Baseline)\n");
    let mut config1 = ModelConfig::transformer(512, 2048, 6, 512, None, Some(8));
    config1.use_rms_norm = false;
    config1.use_swiglu = false;
    config1.positional_encoding = PositionalEncodingType::Learned;
    config1.num_kv_heads = None;
    config1.window_size = None;
    let network1 = build_network(&config1, &vocab);
    print_architecture_summary(&config1, &network1);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Configuration 2: LLaMA 1/2 7B Style
    println!("📋 Configuration 2: LLaMA 1/2 7B Style\n");
    let mut config2 = ModelConfig::transformer(512, 2048, 6, 2048, None, Some(8));
    config2.use_rms_norm = true;
    config2.use_swiglu = true;
    config2.positional_encoding = PositionalEncodingType::RoPE;
    config2.num_kv_heads = None; // MHA
    config2.window_size = None; // Full attention
    let network2 = build_network(&config2, &vocab);
    print_architecture_summary(&config2, &network2);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Configuration 3: LLaMA 2 70B Style (with GQA)
    println!("📋 Configuration 3: LLaMA 2 70B Style (with GQA)\n");
    let mut config3 = ModelConfig::transformer(512, 2048, 6, 4096, None, Some(8));
    config3.use_rms_norm = true;
    config3.use_swiglu = true;
    config3.positional_encoding = PositionalEncodingType::RoPE;
    config3.num_kv_heads = Some(4); // GQA
    config3.window_size = None; // Full attention
    let network3 = build_network(&config3, &vocab);
    print_architecture_summary(&config3, &network3);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Configuration 4: Mistral 7B Style (Complete Modern Stack)
    println!("📋 Configuration 4: Mistral 7B Style ⭐ (Complete Modern Stack)\n");
    let mut config4 = ModelConfig::transformer(512, 2048, 6, 8192, None, Some(8));
    config4.use_rms_norm = true;
    config4.use_swiglu = true;
    config4.positional_encoding = PositionalEncodingType::RoPE;
    config4.num_kv_heads = Some(4); // GQA
    config4.window_size = Some(4096); // Sliding Window
    let network4 = build_network(&config4, &vocab);
    print_architecture_summary(&config4, &network4);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Configuration 5: Aggressive Efficiency
    println!("📋 Configuration 5: Aggressive Efficiency (Maximum Speed)\n");
    let mut config5 = ModelConfig::transformer(512, 2048, 6, 4096, None, Some(8));
    config5.use_rms_norm = true;
    config5.use_swiglu = true;
    config5.positional_encoding = PositionalEncodingType::RoPE;
    config5.num_kv_heads = Some(2); // Aggressive GQA (4x reduction)
    config5.window_size = Some(1024); // Small window (very fast)
    let network5 = build_network(&config5, &vocab);
    print_architecture_summary(&config5, &network5);

    println!("\n═══════════════════════════════════════════════════════════════\n");
    println!("✅ All configurations displayed successfully!");
    println!("\n🎉 RustGPT supports the complete modern LLM stack!");
    println!("   - Phase 1: RMSNorm, SwiGLU, RoPE, No Bias");
    println!("   - Phase 2: Group-Query Attention (GQA)");
    println!("   - Phase 3: Sliding Window Attention");
    println!("\n🚀 Ready for production use!\n");
}
