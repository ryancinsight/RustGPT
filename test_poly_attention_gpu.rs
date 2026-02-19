/// Test PolyAttention GPU forward implementation
/// This test verifies that PolyAttention::forward_gpu correctly integrates
/// with the attention GPU kernel and unblocks SharedTemporalProcessing GPU execution.

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use llm::domain::attention::poly_attention::PolyAttention;
    use llm::domain::compute::GpuComponent;
    use llm::domain::layers::components::common::TemporalMixingLayer;
    use llm::domain::layers::components::shared_temporal_processing::SharedTemporalProcessing;

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_poly_attention_gpu_forward() {
        // Create PolyAttention layer
        let mut attention = PolyAttention::new(512, 8, 2048);

        // Try to enable GPU with auto-detection
        match attention.enable_gpu_auto_detect() {
            Ok(()) => {
                println!("✅ GPU auto-detection successful: {:?}", attention.gpu_backend_name());

                // Create test input: (seq_len, embed_dim)
                let batch_size = 2;
                let seq_len = 4;
                let embed_dim = 512;
                let input = Array2::zeros((batch_size * seq_len, embed_dim));

                // Call GPU forward
                match attention.forward_gpu(&input) {
                    Ok(output) => {
                        assert_eq!(output.dim(), input.dim(), "Output shape mismatch");
                        println!("✅ PolyAttention GPU forward successful");
                        println!("   Input shape: {:?}", input.dim());
                        println!("   Output shape: {:?}", output.dim());
                    }
                    Err(e) => {
                        panic!("GPU forward failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  GPU not available, skipping GPU forward test: {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_temporal_mixing_layer_gpu_forward() {
        // Create TemporalMixingLayer::Attention variant
        let attention = PolyAttention::new(512, 8, 2048);
        let mut layer = TemporalMixingLayer::Attention(Box::new(attention));

        // Try to enable GPU with auto-detection
        match layer.ensure_gpu_device_auto_detect() {
            Ok(()) => {
                println!("✅ GPU auto-detection successful for TemporalMixingLayer");

                // Create test input
                let batch_size = 2;
                let seq_len = 4;
                let embed_dim = 512;
                let input = Array2::zeros((batch_size * seq_len, embed_dim));

                // Call GPU forward via TemporalMixingLayer
                match layer.forward_gpu(&input) {
                    Ok(output) => {
                        assert_eq!(output.dim(), input.dim(), "Output shape mismatch");
                        println!("✅ TemporalMixingLayer GPU forward successful");
                        println!("   Input shape: {:?}", input.dim());
                        println!("   Output shape: {:?}", output.dim());
                    }
                    Err(e) => {
                        panic!("TemporalMixingLayer GPU forward failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("⚠️  GPU not available, skipping GPU forward test: {}", e);
            }
        }
    }

    #[test]
    fn test_poly_attention_gpu_weights_cache() {
        // Create PolyAttention layer
        let mut attention = PolyAttention::new(512, 8, 2048);

        // Verify GPU weights cache is initially None
        assert!(attention.gpu_weights.is_none(), "GPU weights should be None initially");

        // Enable GPU
        #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            if attention.enable_gpu_auto_detect().is_ok() {
                // Create small test input to trigger GPU execution
                let input = Array2::zeros((1, 512));
                let _ = attention.forward_gpu(&input);

                // Verify GPU weights cache is populated
                assert!(
                    attention.gpu_weights.is_some(),
                    "GPU weights should be cached after forward"
                );
                println!("✅ GPU weights cache verified");
            }
        }
    }
}

fn main() {
    println!("PolyAttention GPU integration tests");
}
