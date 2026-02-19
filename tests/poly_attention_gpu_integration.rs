//! Integration tests for PolyAttention GPU forward implementation
//!
//! Phase 5.6: GPU Consolidation - PolyAttention GPU Kernel Integration
//!
//! This test verifies that PolyAttention::forward_gpu correctly:
//! 1. Integrates with the existing attention_gpu_kernel
//! 2. Manages GPU weights cache (upload, reuse)
//! 3. Unblocks SharedTemporalProcessing GPU execution for Transformers

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
mod tests {
    use llm::domain::attention::poly_attention::PolyAttention;
    use llm::domain::attention::position::config::CoPEConfig;
    use llm::domain::compute::GpuComponent;
    use llm::domain::layers::components::common::TemporalMixingLayer;
    use ndarray::Array2;

    // Helper to skip tests if GPU is unavailable
    fn has_gpu() -> bool {
        llm::domain::compute::GpuDevice::auto_detect().is_ok()
    }

    #[test]
    fn test_poly_attention_forward_gpu_basic() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        let cope_config = CoPEConfig::default();
        let mut attention = PolyAttention::new(512, 8, 2048, cope_config);

        // Enable GPU
        assert!(
            attention.enable_gpu_auto_detect().is_ok(),
            "GPU auto-detection should succeed"
        );

        // Create test input
        let batch_size = 2;
        let seq_len = 4;
        let embed_dim = 512;
        let input = Array2::zeros((batch_size * seq_len, embed_dim));

        // Execute GPU forward
        match attention.forward_gpu(&input) {
            Ok(output) => {
                assert_eq!(output.dim(), input.dim(), "Output shape should match input");
                println!(
                    "✅ PolyAttention GPU forward successful: {:?} -> {:?}",
                    input.dim(),
                    output.dim()
                );
            }
            Err(e) => {
                panic!("GPU forward failed: {}", e);
            }
        }
    }

    #[test]
    fn test_poly_attention_gpu_weights_cache() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        let cope_config = CoPEConfig::default();
        let mut attention = PolyAttention::new(256, 4, 1024, cope_config);

        // Enable GPU
        attention
            .enable_gpu_auto_detect()
            .expect("GPU should be available");

        // Verify weights cache is initially empty
        assert!(
            attention.gpu_weights.is_none(),
            "GPU weights should be None before forward"
        );

        // Run forward once to populate cache
        let input1 = Array2::zeros((8, 256));
        let _ = attention.forward_gpu(&input1);

        // Verify weights cache is now populated
        assert!(
            attention.gpu_weights.is_some(),
            "GPU weights should be cached after forward"
        );

        // Run forward again - should reuse cached weights
        let input2 = Array2::zeros((4, 256));
        match attention.forward_gpu(&input2) {
            Ok(output) => {
                assert_eq!(
                    output.dim(),
                    input2.dim(),
                    "Output shape should match input"
                );
                println!("✅ GPU weights cache reuse successful");
            }
            Err(e) => {
                panic!("Second forward with cached weights failed: {}", e);
            }
        }
    }

    #[test]
    fn test_poly_attention_gpu_different_batch_sizes() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        let cope_config = CoPEConfig::default();
        let mut attention = PolyAttention::new(256, 4, 1024, cope_config);
        attention
            .enable_gpu_auto_detect()
            .expect("GPU should be available");

        let batch_sizes = vec![1, 2, 4, 8];
        let seq_len = 4;
        let embed_dim = 256;

        for batch_size in batch_sizes {
            let input = Array2::zeros((batch_size * seq_len, embed_dim));

            match attention.forward_gpu(&input) {
                Ok(output) => {
                    assert_eq!(
                        output.dim(),
                        (batch_size * seq_len, embed_dim),
                        "Output shape mismatch for batch_size={}",
                        batch_size
                    );
                    println!("✅ Batch size {}: GPU forward successful", batch_size);
                }
                Err(e) => {
                    panic!("GPU forward failed for batch_size={}: {}", batch_size, e);
                }
            }
        }
    }

    #[test]
    fn test_temporal_mixing_layer_gpu_forward() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        // Create TemporalMixingLayer::Attention variant
        let cope_config = CoPEConfig::default();
        let attention = PolyAttention::new(512, 8, 2048, cope_config);
        let mut layer = TemporalMixingLayer::Attention(Box::new(attention));

        // Enable GPU
        assert!(
            layer.ensure_gpu_device_auto_detect().is_ok(),
            "GPU auto-detection should succeed"
        );

        // Create test input
        let batch_size = 2;
        let seq_len = 4;
        let embed_dim = 512;
        let input = Array2::zeros((batch_size * seq_len, embed_dim));

        // Execute GPU forward via TemporalMixingLayer
        match layer.forward_gpu(&input) {
            Ok(output) => {
                assert_eq!(output.dim(), input.dim(), "Output shape should match input");
                println!(
                    "✅ TemporalMixingLayer GPU forward successful: {:?} -> {:?}",
                    input.dim(),
                    output.dim()
                );
            }
            Err(e) => {
                panic!("TemporalMixingLayer GPU forward failed: {}", e);
            }
        }
    }

    #[test]
    fn test_poly_attention_gpu_ready_status() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        let cope_config = CoPEConfig::default();
        let mut attention = PolyAttention::new(256, 4, 1024, cope_config);

        // Before GPU setup
        assert!(
            !attention.is_gpu_ready(),
            "GPU should not be ready before setup"
        );

        // After enable_gpu_auto_detect
        attention
            .enable_gpu_auto_detect()
            .expect("GPU should be available");
        assert!(
            attention.is_gpu_ready() || !has_gpu(),
            "GPU should be ready if available"
        );

        // After forward (weights cached)
        let input = Array2::zeros((4, 256));
        let _ = attention.forward_gpu(&input);
        assert!(
            attention.is_gpu_ready(),
            "GPU should be ready after forward with cached weights"
        );

        println!("✅ GPU ready status correctly tracked");
    }

    #[test]
    fn test_poly_attention_gpu_backend_name() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        let cope_config = CoPEConfig::default();
        let mut attention = PolyAttention::new(256, 4, 1024, cope_config);
        attention
            .enable_gpu_auto_detect()
            .expect("GPU should be available");

        // Verify backend name is accessible
        if let Some(backend_name) = attention.gpu_backend_name() {
            println!("✅ GPU backend: {}", backend_name);
            assert!(!backend_name.is_empty(), "Backend name should not be empty");
        } else {
            panic!("Backend name should be available after GPU setup");
        }
    }

    #[test]
    fn test_poly_attention_gpu_caches_intermediates() {
        if !has_gpu() {
            println!("⚠️  GPU not available, skipping test");
            return;
        }

        // With window_size set in CoPEConfig, we get proper batch/seq separation
        let mut cope_config = CoPEConfig::default();
        cope_config.window_size = Some(8); // Set window_size for proper batch/seq handling
        let mut attention = PolyAttention::new(256, 4, 127, cope_config);
        attention
            .enable_gpu_auto_detect()
            .expect("GPU should be available");

        let batch_size = 2;
        let seq_len = 8;
        let embed_dim = 256;
        let input = Array2::zeros((batch_size * seq_len, embed_dim));

        // Execute GPU forward
        let _ = attention
            .forward_gpu(&input)
            .expect("GPU forward should succeed");

        // Verify all required caches are populated
        assert!(
            attention.cached_input.is_some(),
            "cached_input should be populated after forward_gpu"
        );
        assert!(
            attention.cached_q.is_some(),
            "cached_q should be populated after forward_gpu"
        );
        assert!(
            attention.cached_k.is_some(),
            "cached_k should be populated after forward_gpu"
        );
        assert!(
            attention.cached_v.is_some(),
            "cached_v should be populated after forward_gpu"
        );
        assert!(
            attention.cached_attn_weights.is_some(),
            "cached_attn_weights should be populated after forward_gpu"
        );

        // Verify cache dimensions match expectations
        let cached_q = attention.cached_q.as_ref().unwrap();
        assert_eq!(
            cached_q.dim(),
            (batch_size * seq_len, embed_dim),
            "cached_q should have shape (N, embed_dim)"
        );

        let cached_k = attention.cached_k.as_ref().unwrap();
        assert_eq!(
            cached_k.dim(),
            (batch_size * seq_len, embed_dim),
            "cached_k should have shape (N, embed_dim)"
        );

        let cached_v = attention.cached_v.as_ref().unwrap();
        assert_eq!(
            cached_v.dim(),
            (batch_size * seq_len, embed_dim),
            "cached_v should have shape (N, embed_dim)"
        );

        let cached_attn_weights = attention.cached_attn_weights.as_ref().unwrap();
        // Verify attention weights are cached (exact shape depends on GPU kernel implementation)
        let (rows, cols) = cached_attn_weights.dim();
        assert!(
            rows > 0 && cols > 0,
            "cached_attn_weights should be non-empty: got ({}, {})",
            rows,
            cols
        );
        // Total elements should be batch_size * num_heads * seq_len * seq_len
        let total_elements = batch_size * attention.num_heads * seq_len * seq_len;
        assert_eq!(
            rows * cols,
            total_elements,
            "cached_attn_weights should have {} total elements, got {} * {} = {}",
            total_elements,
            rows,
            cols,
            rows * cols
        );

        println!("✅ All GPU intermediate caches correctly populated");
    }
}

#[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
mod tests {
    #[test]
    fn test_gpu_features_disabled() {
        println!(
            "⚠️  GPU features disabled - compile with --features gpu-wgpu, gpu-cuda, or gpu-metal to run GPU tests"
        );
    }
}
