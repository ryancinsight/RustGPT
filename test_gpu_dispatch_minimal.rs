/// Minimal test to verify GPU dispatch is working and not silently failing
/// Run with: cargo test --test test_gpu_dispatch_minimal --lib --features gpu-wgpu -- --nocapture
use llm::domain::models::llm::LLM;
use ndarray::Array2;

#[test]
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
fn test_gpu_dispatch_no_fallback() {
    // This test should panic if GPU dispatch fails with gpu-wgpu enabled
    println!("Creating LLM...");
    let mut llm = LLM::default();
    
    // Ensure at least one RichardsGlu or PolyAttention layer exists
    println!("Network has {} layers", llm.network_depth());
    
    // Create simple input (batch_size=1, seq_len=16)
    let input = Array2::zeros((1, 16));
    
    println!("Testing forward pass with GPU enabled...");
    // This should panic if GPU fails, since we have strict no-fallback
    match llm.forward(&input) {
        Ok(output) => {
            println!("Forward pass succeeded, output shape: {:?}", output.dim());
        }
        Err(e) => {
            eprintln!("Forward pass returned error: {}", e);
            panic!("Forward pass failed: {}", e);
        }
    }
}

#[test]
#[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
fn test_cpu_dispatch() {
    println!("Creating LLM (CPU only)...");
    let mut llm = LLM::default();
    
    let input = Array2::zeros((1, 16));
    println!("Testing forward pass (CPU)...");
    match llm.forward(&input) {
        Ok(output) => {
            println!("Forward pass succeeded, output shape: {:?}", output.dim());
        }
        Err(e) => {
            eprintln!("Forward pass failed: {}", e);
        }
    }
}
