/// GPU Diagnostic Tool
/// Run with: RUST_LOG=debug cargo run --release --features gpu-wgpu --bin diagnose_gpu
use llm::domain::models::llm::LLM;

fn main() {
    // Setup logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .with_level(true)
        .init();

    println!("╔════════════════════════════════════════════╗");
    println!("║      GPU Utilization Diagnostic Tool       ║");
    println!("╚════════════════════════════════════════════╝");
    println!();

    // Check compile-time feature flags
    println!("[*] Feature Flags:");
    println!("    gpu-wgpu: {}", cfg!(feature = "gpu-wgpu"));
    println!("    gpu-cuda: {}", cfg!(feature = "gpu-cuda"));
    println!("    gpu-metal: {}", cfg!(feature = "gpu-metal"));
    println!();

    // Initialize LLM
    println!("[*] Creating LLM instance...");
    let mut llm = LLM::default();
    println!("    Network depth: {} layers", llm.network_depth());
    println!();

    // List layers
    println!("[*] Layer Composition:");
    for (i, layer_type) in llm.network_iter().enumerate() {
        println!("    [{}] {}", i, layer_type);
    }
    println!();

    // Try minimal training
    println!("[*] Starting minimal training loop (1 epoch, batch_size=2)...");
    println!("    WATCH GPU METRICS DURING THIS SECTION");
    println!();

    let test_data = vec!["test input"; 2];
    match llm.train_with_warmup_with_accumulation(
        test_data,
        1,      // 1 epoch only
        0.0005, // learning rate
        2,      // batch size
        0,      // no warmup
        1,      // no gradient accumulation
    ) {
        Ok(()) => {
            println!();
            println!("✅ Training completed successfully");
            println!("   If you saw GPU utilization, GPU dispatch IS working");
        }
        Err(e) => {
            println!();
            println!("❌ Training failed: {}", e);
            println!("   Error details above should indicate GPU issue");
        }
    }
    println!();
}

trait LlmExt {
    fn network_iter(&self) -> impl Iterator<Item = &str>;
}

impl LlmExt for LLM {
    fn network_iter(&self) -> impl Iterator<Item = &str> {
        std::iter::empty()
    }
}
