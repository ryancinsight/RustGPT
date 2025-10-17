//! Example demonstrating adaptive gradient clipping functionality
//!
//! This example shows how to configure and use different gradient clipping strategies
//! in the RustGPT LLM, including the new adaptive techniques.

use llm::{AdaptiveClippingConfig, AdaptiveGradientClipping, L2GradientClipping, LLM};

fn main() {
    println!("=== Adaptive Gradient Clipping Demo ===\n");

    // Create a default LLM (which uses adaptive clipping by default)
    let mut llm = LLM::default();

    println!("1. Default LLM uses Adaptive Gradient Clipping with AGC + Centralization");
    println!("   Configuration: AGC threshold = 0.01, Centralization = enabled, AGC = enabled\n");

    // Demonstrate different clipping configurations
    demonstrate_clipping_strategies();

    // Show how to configure custom clipping
    println!("2. Configuring custom gradient clipping strategies:");

    // Example 1: Pure L2 clipping (legacy behavior)
    let l2_clipper = Box::new(L2GradientClipping::new(5.0));
    llm.set_gradient_clipping(l2_clipper);
    println!("   - Set to L2 clipping with threshold 5.0");

    // Example 2: Adaptive clipping with custom config
    let custom_config = AdaptiveClippingConfig {
        agc_threshold: 0.02, // More aggressive AGC
        use_centralization: true,
        use_agc: true,
        l2_threshold: 10.0, // Higher fallback
    };
    let adaptive_clipper = Box::new(AdaptiveGradientClipping::new(custom_config));
    llm.set_gradient_clipping(adaptive_clipper);
    println!("   - Set to Adaptive clipping with custom config (AGC threshold = 0.02)");

    // Example 3: Disable clipping entirely
    llm.disable_gradient_clipping();
    println!("   - Disabled gradient clipping entirely");

    println!("\n3. Benefits of Adaptive Gradient Clipping:");
    println!("   - AGC (Adaptive Gradient Clipping): Scales based on gradient-to-parameter ratios");
    println!("   - Gradient Centralization: Centers gradients around zero mean per feature");
    println!("   - Better stability for different model sizes and training conditions");
    println!("   - Literature-validated techniques from NFNet and other papers");

    println!("\nDemo completed successfully!");
}

fn demonstrate_clipping_strategies() {
    // For demo purposes, we'll show the concepts without ndarray dependency
    // In a real implementation, you'd use ndarray::Array2

    println!("Gradient clipping strategies available:");
    println!("- L2GradientClipping: Traditional L2 norm clipping with fixed threshold");
    println!("- AdaptiveGradientClipping: Advanced clipping with AGC and centralization");
    println!("- GradientClipping trait: Extensible interface for custom strategies");

    println!("\nKey features:");
    println!("- AGC scales gradients based on their magnitude relative to parameter norms");
    println!("- Gradient centralization centers each feature dimension around zero mean");
    println!("- Configurable thresholds and strategies for different training scenarios");
}
