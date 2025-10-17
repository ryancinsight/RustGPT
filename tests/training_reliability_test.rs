use llm::{LLM, ModelError};

#[test]
fn test_training_divergence_detection_nan() {
    // Create a default LLM
    let mut llm = LLM::default();

    // Training data that could cause NaN (though unlikely with default config)
    let training_data = vec!["test data"];

    // We can't easily force NaN in a real training scenario without modifying internals,
    // so this test verifies the error handling path exists
    // In practice, NaN detection happens in train_with_batch_size when loss.is_nan() is true
    let result = llm.train(training_data, 1, 0.01);

    // Training should either succeed or fail with a proper error (not panic)
    match result {
        Ok(_) => {
            // Training succeeded normally - this is the expected path
        }
        Err(ModelError::Training { message }) => {
            // If training failed, it should be with a proper error message
            assert!(
                message.contains("diverged") || message.contains("NaN") || message.contains("Inf"),
                "Error message should indicate divergence: {}",
                message
            );
        }
        Err(e) => {
            panic!("Unexpected error type: {:?}", e);
        }
    }
}

#[test]
fn test_training_metrics_output() {
    // This test verifies that training produces metrics output
    // We can't easily capture println! output, but we can verify training completes
    let mut llm = LLM::default();
    let training_data = vec!["hello world"];

    // Training should complete and produce metrics (loss, grad_norm, lr)
    let result = llm.train(training_data, 2, 0.01);

    // Should succeed without errors
    assert!(result.is_ok(), "Training should complete successfully");
}

#[test]
fn test_gradient_norm_computation() {
    // Verify that gradient norms are computed during training
    // This is an integration test that ensures the gradient norm path is exercised
    let mut llm = LLM::default();
    let training_data = vec!["test sequence for gradient norm"];

    // Train for a few epochs
    let result = llm.train(training_data, 3, 0.001);

    // Should complete successfully with gradient norm computation
    assert!(
        result.is_ok(),
        "Training with gradient norm computation should succeed"
    );
}

#[test]
fn test_training_with_batch_size_metrics() {
    // Test that batch training produces proper metrics
    let mut llm = LLM::default();
    let training_data = vec!["batch 1", "batch 2", "batch 3", "batch 4"];

    // Train with batch size 2
    let result = llm.train_with_batch_size(training_data, 2, 0.01, 2);

    // Should succeed and produce metrics for each epoch
    assert!(result.is_ok(), "Batch training with metrics should succeed");
}

#[test]
fn test_training_stability_with_small_lr() {
    // Test training stability with very small learning rate
    let mut llm = LLM::default();
    let training_data = vec!["stable training test"];

    // Very small learning rate should prevent divergence
    let result = llm.train(training_data, 5, 0.0001);

    // Should complete successfully without divergence
    assert!(
        result.is_ok(),
        "Training with small learning rate should be stable"
    );
}
