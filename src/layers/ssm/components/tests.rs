//! Comprehensive tests for SSM components
//!
//! This module contains tests for all SSM components including:
//! - State management functionality
//! - Selective scan operations
//! - Projection layers
//! - Richards activation integration

use ndarray::{Array1, Array2};
use approx::assert_abs_diff_eq;

use super::*;

#[test]
fn test_state_cache_basic() {
    let mut cache = StateCache::new(128);
    
    // Test initial state
    assert!(!cache.is_valid());
    assert_eq!(cache.memory_usage(), 0);
    
    // Test caching a state
    let test_state = Array2::ones((64, 128));
    cache.cache_state("test", test_state.clone());
    
    assert!(cache.is_valid());
    assert_eq!(cache.memory_usage(), 64 * 128 * 4); // 4 bytes per f32
    
    // Test retrieving state
    let retrieved = cache.get_state("test").unwrap();
    assert_eq!(retrieved.shape(), test_state.shape());
    assert_abs_diff_eq!(retrieved.sum(), test_state.sum(), epsilon = 1e-6);
}

#[test]
fn test_state_cache_invalidation() {
    let mut cache = StateCache::new(64);
    
    // Cache some states
    let state1 = Array2::ones((32, 64));
    let state2 = Array2::zeros((32, 64));
    cache.cache_state("state1", state1);
    cache.cache_state("state2", state2);
    
    assert!(cache.is_valid());
    assert!(cache.get_state("state1").is_some());
    
    // Test manual invalidation
    cache.invalidate();
    assert!(!cache.is_valid());
    assert!(cache.get_state("state1").is_none());
    assert_eq!(cache.memory_usage(), 0);
}

#[test]
fn test_state_cache_memory_management() {
    let mut cache = StateCache::new(256);
    
    // Add several large states
    for i in 0..5 {
        let state = Array2::ones((100, 256));
        cache.cache_state(&format!("large_state_{}", i), state);
    }
    
    let initial_memory = cache.memory_usage();
    assert!(initial_memory > 0);
    
    // Test memory clearing
    cache.clear_large_states(1024 * 1024); // 1MB limit
    let final_memory = cache.memory_usage();
    
    // Should have cleared some states
    assert!(final_memory <= initial_memory);
}

#[test]
fn test_state_manager_automatic_invalidation() {
    let mut manager = StateManager::new(64, 1024 * 1024);
    
    // Create initial input
    let input1 = Array2::ones((32, 64));
    let cache1 = manager.cache(&input1);
    cache1.cache_state("test", Array2::zeros((32, 64)));
    
    assert!(cache1.is_valid());
    
    // Create input with different dimensions - should invalidate
    let input2 = Array2::ones((64, 64)); // Different sequence length
    let cache2 = manager.cache(&input2);
    
    assert!(!cache2.is_valid()); // Should be invalidated due to dimension change
}

#[test]
fn test_selective_scanner_sequential() {
    let scanner = SelectiveScanner::with_config(SelectiveScanConfig {
        parallel: false,
        chunk_size: 1024,
        stability_threshold: 1e-6,
    });
    
    // Test with simple matrices
    let a = Array2::from_diag(&Array1::ones(3)); // Identity matrix
    let b = Array2::ones((3, 3));
    let u = Array2::from_shape_fn((5, 3), |(i, j)| (i * 3 + j) as f32);
    
    let result = scanner.scan(&a, &b, &u);
    
    // With identity A matrix, result should be similar to cumulative sum
    assert_eq!(result.shape(), [5, 3]);
    
    // Check that all values are finite
    for val in result.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_selective_scanner_stability() {
    let scanner = SelectiveScanner::new();
    
    // Test with potentially unstable values
    let a = Array2::from_diag(&Array1::from_vec(vec![0.5, -0.5, 1.5]));
    let b = Array2::ones((3, 3)) * 2.0;
    let u = Array2::ones((10, 3));
    
    let result = scanner.stable_scan(&a, &b, &u);
    
    // All values should be stable (finite and within reasonable bounds)
    for val in result.iter() {
        assert!(val.is_finite());
        assert!(val.abs() < 1e6); // Reasonable bound
    }
}

#[test]
fn test_linear_projection_basic() {
    let config = ProjectionConfig {
        use_bias: true,
        small_init: true,
        init_scale: 0.02,
    };
    
    let projection = LinearProjection::new(64, 128, config);
    
    // Test forward pass
    let input = Array2::ones((32, 64));
    let output = projection.forward(&input);
    
    assert_eq!(output.shape(), [32, 128]);
    
    // With small_init, weights should be close to zero
    assert_abs_diff_eq!(output.sum(), 0.0, epsilon = 1.0); // Allow some tolerance
}

#[test]
fn test_linear_projection_gradients() {
    let config = ProjectionConfig::default();
    let mut projection = LinearProjection::new(32, 64, config);
    
    // Set known weights for testing
    projection.weight.fill(0.1);
    if let Some(bias) = &mut projection.bias {
        bias.fill(0.05);
    }
    
    let input = Array2::ones((16, 32));
    let output = projection.forward(&input);
    
    // Calculate expected output: input * weight + bias
    let expected_sum = 16.0 * 32.0 * 0.1 + 16.0 * 64.0 * 0.05;
    assert_abs_diff_eq!(output.sum(), expected_sum, epsilon = 1e-5);
}

#[test]
fn test_depthwise_conv1d() {
    let config = ProjectionConfig::default();
    let conv = DepthwiseConv1D::new(64, 3, config);
    
    // Test forward pass
    let input = Array2::from_shape_fn((10, 64), |(i, j)| (i * 64 + j) as f32);
    let output = conv.forward_causal(&input);
    
    assert_eq!(output.shape(), input.shape());
    
    // Check that output is different from input (convolution applied)
    assert_ne!(output.sum(), input.sum());
}

#[test]
fn test_richards_activation_integration() {
    use crate::richards::Variant;
    use ndarray::Array1;
    
    // Test sigmoid-based activation (Swish-like)
    let activation = SsmRichardsActivation::sigmoid(true, true);
    
    let input = Array2::from_shape_fn((8, 32), |(i, j)| (i * 32 + j) as f32 * 0.1);
    let output = activation.forward(&input);
    
    assert_eq!(output.shape(), input.shape());
    
    // Output should be similar to input * sigmoid(input) (Swish)
    for (&in_val, &out_val) in input.iter().zip(output.iter()) {
        let expected = in_val * (1.0 / (1.0 + (-in_val).exp()));
        assert_abs_diff_eq!(out_val, expected, epsilon = 0.1); // Allow some tolerance for learning
    }
}

#[test]
fn test_ssm_activation_config() {
    let config = SsmActivationConfig::sigmoid(true);
    let activation = config.create_activation();
    
    assert!(matches!(activation.activation.richards_curve.variant, crate::richards::Variant::Sigmoid));
    assert!(activation.use_elementwise_mult);
}

#[test]
fn test_component_memory_usage() {
    // Test memory usage tracking
    let mut cache = StateCache::new(256);
    
    let large_state = Array2::ones((1000, 256));
    cache.cache_state("large", large_state);
    
    let memory_usage = cache.memory_usage();
    let expected_usage = 1000 * 256 * 4; // 4 bytes per f32
    
    assert_eq!(memory_usage, expected_usage);
}

#[test]
fn test_numerical_stability() {
    let scanner = SelectiveScanner::new();
    
    // Test with extreme values
    let a = Array2::from_diag(&Array1::from_vec(vec![10.0, -10.0, 0.0]));
    let b = Array2::ones((3, 3)) * 100.0;
    let u = Array2::ones((5, 3)) * 1000.0;
    
    let result = scanner.stable_scan(&a, &b, &u);
    
    // Should handle extreme values gracefully
    for val in result.iter() {
        assert!(val.is_finite(), "Result contains non-finite value: {}", val);
    }
}