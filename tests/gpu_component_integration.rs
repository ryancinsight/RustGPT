//! GPU Component Integration Tests (Phase 5.6)
//!
//! Tests for GpuComponent trait implementation across shared layer components.
//! Validates automatic GPU detection, device attachment, and capacity management.

#![allow(cfg_attr_crate_type)]

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
mod gpu_component_tests {
    use llm::domain::{
        compute::GpuComponent,
        layers::components::{
            attention_context::SharedAttentionContext,
            common::{CommonLayers, FeedForwardVariant, TemporalMixingType},
            feedforward::SharedFeedforward,
            temporal_processing::SharedTemporalProcessing,
        },
        models::config::ModelConfig,
        richards::RichardsGlu,
    };

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            embedding_dim: 64,
            sequence_length: 128,
            num_layers: 2,
            temporal_mixing: TemporalMixingType::Attention,
            ..Default::default()
        }
    }

    #[test]
    fn test_richards_glu_gpu_component_auto_detect() {
        let mut glu = RichardsGlu::new(64, 128);

        match glu.enable_gpu_auto_detect() {
            Ok(()) => {
                assert!(glu.is_gpu_ready(), "GPU should be ready after auto-detect");
                let backend_name = glu.gpu_backend_name();
                assert!(backend_name.is_some(), "Backend name should be available");
                println!("RichardsGLU GPU backend: {:?}", backend_name);
            }
            Err(e) => {
                println!(
                    "No GPU available for test (expected on CPU-only systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_shared_feedforward_gpu_component_auto_detect() {
        let glu = RichardsGlu::new(64, 128);
        let mut feedforward =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(glu)));

        match feedforward.enable_gpu_auto_detect() {
            Ok(()) => {
                assert!(
                    feedforward.is_gpu_ready(),
                    "GPU should be ready after auto-detect"
                );
                let backend_name = feedforward.gpu_backend_name();
                assert!(backend_name.is_some(), "Backend name should be available");
                println!("SharedFeedforward GPU backend: {:?}", backend_name);
            }
            Err(e) => {
                println!(
                    "No GPU available for test (expected on CPU-only systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_shared_temporal_processing_gpu_component_auto_detect() {
        let config = create_test_config();
        let common_layers = CommonLayers::new(&config);
        let mut temporal =
            SharedTemporalProcessing::new(common_layers.temporal_mixing, None, false);

        match temporal.enable_gpu_auto_detect() {
            Ok(()) => {
                assert!(
                    temporal.is_gpu_ready(),
                    "GPU should be ready after auto-detect"
                );
                let backend_name = temporal.gpu_backend_name();
                assert!(backend_name.is_some(), "Backend name should be available");
                println!("SharedTemporalProcessing GPU backend: {:?}", backend_name);
            }
            Err(e) => {
                println!(
                    "No GPU available for test (expected on CPU-only systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_shared_attention_context_gpu_component_auto_detect() {
        let mut ctx = SharedAttentionContext::new();

        match ctx.enable_gpu_auto_detect() {
            Ok(()) => {
                assert!(ctx.is_gpu_ready(), "GPU should be ready after auto-detect");
                let backend_name = ctx.gpu_backend_name();
                assert!(backend_name.is_some(), "Backend name should be available");
                println!("SharedAttentionContext GPU backend: {:?}", backend_name);
            }
            Err(e) => {
                println!(
                    "No GPU available for test (expected on CPU-only systems): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_shared_feedforward_ensure_capacity() {
        let glu = RichardsGlu::new(64, 128);
        let mut feedforward =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(glu)));

        match feedforward.enable_gpu_auto_detect() {
            Ok(()) => {
                let result = feedforward.ensure_capacity(32, 64, 128);
                assert!(
                    result.is_ok(),
                    "ensure_capacity should succeed: {:?}",
                    result
                );
                println!("SharedFeedforward capacity ensured for batch_size=32, embed_dim=64");
            }
            Err(e) => {
                println!("No GPU available for test: {}", e);
            }
        }
    }

    #[test]
    fn test_shared_temporal_processing_ensure_capacity() {
        let config = create_test_config();
        let common_layers = CommonLayers::new(&config);
        let mut temporal =
            SharedTemporalProcessing::new(common_layers.temporal_mixing, None, false);

        match temporal.enable_gpu_auto_detect() {
            Ok(()) => {
                let result = temporal.ensure_capacity(32, 64, 128);
                assert!(
                    result.is_ok(),
                    "ensure_capacity should succeed: {:?}",
                    result
                );
                println!(
                    "SharedTemporalProcessing capacity ensured for batch_size=32, embed_dim=64, seq_len=128"
                );
            }
            Err(e) => {
                println!("No GPU available for test: {}", e);
            }
        }
    }

    #[test]
    fn test_shared_attention_context_ensure_capacity() {
        let mut ctx = SharedAttentionContext::new();

        match ctx.enable_gpu_auto_detect() {
            Ok(()) => {
                let result = ctx.ensure_capacity(32, 64, 128);
                assert!(
                    result.is_ok(),
                    "ensure_capacity should succeed: {:?}",
                    result
                );
                println!("SharedAttentionContext capacity ensured for batch_size=32, embed_dim=64");
            }
            Err(e) => {
                println!("No GPU available for test: {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_device_attachment_without_auto_detect_fails() {
        let glu = RichardsGlu::new(64, 128);
        let mut feedforward =
            SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(glu)));

        // Try to ensure capacity without attaching GPU device
        let result = feedforward.ensure_capacity(32, 64, 128);
        assert!(
            result.is_err(),
            "ensure_capacity should fail without GPU device"
        );

        match result {
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("enable_gpu_auto_detect"),
                    "Error message should mention enable_gpu_auto_detect"
                );
                println!("Expected error: {}", msg);
            }
            Ok(()) => panic!("Should have failed without GPU device"),
        }
    }
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
mod no_gpu_tests {
    #[test]
    fn test_no_gpu_features_enabled() {
        println!(
            "No GPU features enabled. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal to run GPU tests."
        );
    }
}
