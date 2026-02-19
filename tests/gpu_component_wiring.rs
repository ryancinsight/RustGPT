//! GPU Component Wiring Tests (Phase 5.6)
//!
//! Verifies that all shared components properly implement GpuComponent trait
//! with automatic GPU detection (strict no-fallback semantics).

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]

use llm::domain::compute::GpuComponent;
use llm::domain::layers::components::common::FeedForwardVariant;
use llm::domain::layers::components::feedforward::SharedFeedforward;
use llm::domain::richards::RichardsGlu;

#[test]
fn test_shared_feedforward_gpu_auto_detect() {
    let richards_glu = RichardsGlu::new(8, 16);
    let mut feedforward =
        SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

    // GPU auto-detect should either succeed (GPU available) or fail gracefully
    match feedforward.enable_gpu_auto_detect() {
        Ok(()) => {
            assert!(
                feedforward.is_gpu_ready(),
                "GPU should be ready after enable"
            );
            let backend = feedforward.gpu_backend_name();
            println!("✓ GPU enabled: {:?}", backend);
            assert!(backend.is_some(), "Backend name should be available");
        }
        Err(e) => {
            println!("ℹ GPU not available (expected in CI): {}", e);
            assert!(
                !feedforward.is_gpu_ready(),
                "GPU should not be ready after failed enable"
            );
        }
    }
}

#[test]
fn test_shared_feedforward_ensure_capacity() {
    let richards_glu = RichardsGlu::new(8, 16);
    let mut feedforward =
        SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

    // Enable GPU if available
    let _ = feedforward.enable_gpu_auto_detect();

    // Ensure capacity should always succeed
    let result = feedforward.ensure_capacity(2, 8, 128);
    assert!(
        result.is_ok(),
        "ensure_capacity should not fail, got: {:?}",
        result
    );

    // After ensure_capacity, dimensions should be tracked
    let (batch_size, embed_dim) = feedforward.workspace_info();
    assert_eq!(batch_size, Some(2), "Batch size should be tracked");
    assert_eq!(embed_dim, Some(8), "Embed dim should be tracked");
}

#[test]
fn test_shared_feedforward_gpu_device_attachment() {
    let richards_glu = RichardsGlu::new(8, 16);
    let mut feedforward =
        SharedFeedforward::new(FeedForwardVariant::RichardsGlu(Box::new(richards_glu)));

    // Initially, no GPU device should be attached
    assert!(
        feedforward.gpu_device().is_none(),
        "GPU device should not be attached initially"
    );

    // Try to enable GPU
    if feedforward.enable_gpu_auto_detect().is_ok() {
        // GPU device should now be attached
        assert!(
            feedforward.gpu_device().is_some(),
            "GPU device should be attached after enable"
        );

        // Multiple enables should not fail
        let result = feedforward.enable_gpu_auto_detect();
        assert!(
            result.is_ok(),
            "Re-enabling GPU should succeed, got: {:?}",
            result
        );
    }
}
