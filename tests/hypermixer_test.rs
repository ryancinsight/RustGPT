use llm::{HyperMixerBlock, Layer, model_builder::build_network, model_config::ModelConfig};
use ndarray::Array2;

#[test]
fn test_hypermixer_gradient_stability() {
    // Create a small HyperMixer block for testing
    let mut hypermixer = HyperMixerBlock::new(64, 128, 10, 16, true);

    // Create some test input
    let input = Array2::from_elem((8, 64), 0.1);

    // Run forward pass
    let output = hypermixer.forward(&input);

    // Check output is finite
    assert!(
        output.iter().all(|&x| x.is_finite()),
        "Forward pass produced NaN/inf values"
    );

    // Create gradients for backward pass
    let grads = Array2::from_elem((8, 64), 0.01);

    // Run backward pass
    let input_grads = hypermixer.backward(&grads, 0.001);

    // Check gradients are finite
    assert!(
        input_grads.iter().all(|&x| x.is_finite()),
        "Backward pass produced NaN/inf gradients"
    );

    // Check gradient stability method
    assert!(
        hypermixer.check_gradient_stability(&input_grads),
        "Gradients are not stable"
    );
}

#[test]
fn test_hypermixer_training_stability() {
    // Create a small HyperMixer block
    let mut hypermixer = HyperMixerBlock::new(32, 64, 5, 8, true);

    // Training loop to check for stability
    let input = Array2::from_elem((4, 32), 0.1);

    for step in 0..10 {
        // Forward pass
        let output = hypermixer.forward(&input);

        // Check output is finite
        assert!(
            output.iter().all(|&x| x.is_finite()),
            "Forward pass produced NaN/inf at step {}",
            step
        );

        // Compute loss (simple MSE-like)
        let target = Array2::from_elem((4, 32), 0.0);
        let loss_grads = 2.0 * (&output - &target);

        // Backward pass
        let input_grads = hypermixer.backward(&loss_grads, 0.01);

        // Check gradients are finite
        assert!(
            input_grads.iter().all(|&x| x.is_finite()),
            "Backward pass produced NaN/inf at step {}",
            step
        );

        // Check gradient magnitudes are reasonable (not exploding)
        let grad_norm = input_grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(
            grad_norm < 100.0,
            "Gradient explosion detected at step {}: norm = {}",
            step,
            grad_norm
        );
    }
}

#[test]
fn test_build_hypermixer_network() {
    let config = ModelConfig::hypermixer(64, 128, 2, 10, Some(16), Some(4));
    let network = build_network(&config, &llm::Vocab::default());

    // Check that network was built successfully
    assert!(!network.is_empty());

    // Check parameter count is reasonable
    let total_params = network.iter().map(|l| l.parameters()).sum::<usize>();
    assert!(
        total_params > 1000,
        "Parameter count seems too low: {}",
        total_params
    );
    assert!(
        total_params < 1_000_000,
        "Parameter count seems too high: {}",
        total_params
    );
}
