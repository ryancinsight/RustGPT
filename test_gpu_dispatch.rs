/// Quick test to verify GPU dispatch is working in training loop
/// Run with: cargo test --test gpu_dispatch --features gpu-wgpu -- --nocapture --test-threads=1
#[cfg(test)]
mod tests {
    use llm::domain::{
        compute::GpuComponent,
        models::config::ModelConfig,
        richards::RichardsGlu,
    };
    use ndarray::Array2;

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
    fn test_gpu_dispatch_richards_glu() {
        // Create a simple RichardsGlu layer
        let mut layer = RichardsGlu::new(512, 1024, Default::default());

        // Try GPU initialization
        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!("✓ GPU initialized successfully");
                println!("  Backend: {}", layer.gpu_backend_name().unwrap_or("Unknown"));
                assert!(layer.is_gpu_ready());

                // Test GPU forward pass
                let input = Array2::zeros((1, 512));
                match layer.forward_gpu(&input) {
                    Ok(output) => {
                        println!("✓ GPU forward pass successful");
                        println!("  Output shape: {}x{}", output.nrows(), output.ncols());
                        assert_eq!(output.ncols(), 1024);
                    }
                    Err(e) => {
                        println!("✗ GPU forward failed: {}", e);
                        panic!("GPU forward should work when GPU is initialized");
                    }
                }
            }
            Err(e) => {
                println!("⚠ GPU not available: {}", e);
                println!("  This is expected if no GPU is present");
            }
        }
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
    fn test_training_gpu_dispatch() {
        println!("GPU dispatch integration test");
        println!("================================");
        println!("Training loop will now:");
        println!("1. Call enable_gpu_auto_detect() on all layers");
        println!("2. Dispatch RichardsGlu/PolyAttention to GPU");
        println!("3. Fall back to CPU if GPU unavailable");
        println!("");
        println!("To verify GPU usage:");
        println!("  - Windows: Open Task Manager → Performance → GPU");
        println!("  - Linux: watch nvidia-smi");
        println!("  - Mac: Activity Monitor → GPU");
    }
}
