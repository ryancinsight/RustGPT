/// Test to trace GPU execution path - minimal training loop
use llm::domain::models::llm::LLM;
use ndarray::Array2;

fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(true)
        .init();

    println!("=== GPU Execution Tracing Test ===");
    println!("Feature flags: gpu-wgpu={}", cfg!(feature = "gpu-wgpu"));
    println!("Feature flags: gpu-cuda={}", cfg!(feature = "gpu-cuda"));
    println!();

    let mut llm = LLM::default();
    println!("Created LLM with {} layers", llm.network_depth());
    println!();

    // Create minimal batch
    let test_data = vec!["hello world"; 2]; // Tiny dataset
    println!("Running minimal training loop with {} examples", test_data.len());
    println!();

    // Run minimal training - this should trigger GPU paths if enabled
    match llm.train_with_warmup_with_accumulation(
        test_data,
        1,      // 1 epoch
        0.0005, // learning rate
        2,      // batch size
        0,      // warmup epochs
        1,      // no gradient accumulation
    ) {
        Ok(()) => println!("✅ Training completed successfully"),
        Err(e) => eprintln!("❌ Training failed: {}", e),
    }

    println!();
    println!("=== Test Complete ===");
}
