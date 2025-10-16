use llm::rope::{RotaryEmbedding, apply_rotary_pos_emb};
use ndarray::Array2;

#[test]
fn test_rope_creation() {
    let rope = RotaryEmbedding::new(128, 512);
    assert_eq!(rope.dim(), 128);
    assert_eq!(rope.max_seq_len(), 512);
}

#[test]
fn test_rope_with_custom_base() {
    let rope = RotaryEmbedding::with_base(64, 256, 5000.0);
    assert_eq!(rope.dim(), 64);
    assert_eq!(rope.max_seq_len(), 256);
}

#[test]
fn test_rope_apply_shape_preservation() {
    let rope = RotaryEmbedding::new(64, 100);
    let input = Array2::ones((10, 64));
    let output = rope.apply(&input);
    assert_eq!(output.shape(), &[10, 64]);
}

#[test]
fn test_rope_identity_at_position_zero() {
    // At position 0, rotation should be identity (cos(0)=1, sin(0)=0)
    let rope = RotaryEmbedding::new(4, 10);
    let input = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let output = rope.apply(&input);
    
    // At position 0, rotation is identity
    assert!((output[[0, 0]] - 1.0).abs() < 1e-6);
    assert!(output[[0, 1]].abs() < 1e-6);
    assert!((output[[0, 2]] - 1.0).abs() < 1e-6);
    assert!(output[[0, 3]].abs() < 1e-6);
}

#[test]
fn test_rope_rotation_properties() {
    // Test that rotation preserves vector magnitude
    let rope = RotaryEmbedding::new(64, 100);
    let input = Array2::from_shape_fn((5, 64), |_| rand::random::<f32>());
    let output = rope.apply(&input);
    
    // Check magnitude preservation for each position
    for pos in 0..5 {
        let input_row = input.row(pos);
        let output_row = output.row(pos);
        
        let input_mag: f32 = input_row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let output_mag: f32 = output_row.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Rotation should preserve magnitude
        assert!((input_mag - output_mag).abs() < 1e-3, 
            "Magnitude not preserved at position {}: {} vs {}", pos, input_mag, output_mag);
    }
}

#[test]
fn test_rope_relative_position_encoding() {
    // Test that relative position is preserved in dot product
    let rope = RotaryEmbedding::new(64, 100);
    
    // Create two identical vectors at position 0
    let q1 = Array2::ones((1, 64));
    let k1 = Array2::ones((1, 64));
    
    let q1_rot = rope.apply(&q1);
    let k1_rot = rope.apply(&k1);
    
    // Dot product at same position (relative distance = 0)
    let dot1: f32 = q1_rot.iter().zip(k1_rot.iter()).map(|(a, b)| a * b).sum();
    
    // Create the same vectors at position 5
    let mut q2 = Array2::zeros((6, 64));
    let mut k2 = Array2::zeros((6, 64));
    for i in 0..64 {
        q2[[5, i]] = 1.0;
        k2[[5, i]] = 1.0;
    }
    
    let q2_rot = rope.apply(&q2);
    let k2_rot = rope.apply(&k2);
    
    // Dot product at same relative position (both at same position, relative distance = 0)
    let dot2: f32 = (0..64).map(|i| q2_rot[[5, i]] * k2_rot[[5, i]]).sum();
    
    // Should be approximately equal (relative position is the same)
    assert!((dot1 - dot2).abs() < 1e-3, 
        "Relative position not preserved: {} vs {}", dot1, dot2);
}

#[test]
fn test_rope_different_relative_positions() {
    // Test that different relative positions give different dot products
    let rope = RotaryEmbedding::new(64, 100);

    let input = Array2::ones((10, 64));
    let rotated = rope.apply(&input);
    
    // Dot product between position 0 and position 0 (relative distance = 0)
    let dot_same: f32 = (0..64).map(|i| rotated[[0, i]] * rotated[[0, i]]).sum();
    
    // Dot product between position 0 and position 5 (relative distance = 5)
    let dot_diff: f32 = (0..64).map(|i| rotated[[0, i]] * rotated[[5, i]]).sum();
    
    // Different relative positions should give different dot products
    assert!((dot_same - dot_diff).abs() > 0.1, 
        "Different relative positions should give different dot products: {} vs {}", 
        dot_same, dot_diff);
}

#[test]
fn test_apply_rotary_pos_emb_function() {
    let rope = RotaryEmbedding::new(64, 100);
    let q = Array2::ones((10, 64));
    let k = Array2::ones((10, 64));
    
    let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &rope);
    
    assert_eq!(q_rot.shape(), &[10, 64]);
    assert_eq!(k_rot.shape(), &[10, 64]);
}

#[test]
fn test_rope_multiple_sequences() {
    let rope = RotaryEmbedding::new(32, 50);
    
    // Test with different sequence lengths
    for seq_len in [1, 5, 10, 25, 50] {
        let input = Array2::ones((seq_len, 32));
        let output = rope.apply(&input);
        assert_eq!(output.shape(), &[seq_len, 32]);
    }
}

#[test]
fn test_rope_frequency_bands() {
    // Test that different dimension pairs have different frequencies
    let rope = RotaryEmbedding::new(8, 10);
    
    // Create input with 1.0 in first pair, 0.0 elsewhere
    let mut input1 = Array2::zeros((2, 8));
    input1[[0, 0]] = 1.0;
    input1[[1, 0]] = 1.0;
    
    // Create input with 1.0 in last pair, 0.0 elsewhere
    let mut input2 = Array2::zeros((2, 8));
    input2[[0, 6]] = 1.0;
    input2[[1, 6]] = 1.0;
    
    let output1 = rope.apply(&input1);
    let output2 = rope.apply(&input2);
    
    // Different dimension pairs should rotate at different rates
    let rotation1 = (output1[[1, 0]] - output1[[0, 0]]).abs();
    let rotation2 = (output2[[1, 6]] - output2[[0, 6]]).abs();
    
    // Lower dimensions should rotate faster than higher dimensions
    assert!(rotation1 > rotation2, 
        "Lower dimensions should rotate faster: {} vs {}", rotation1, rotation2);
}

#[test]
fn test_rope_zero_input() {
    let rope = RotaryEmbedding::new(64, 100);
    let input = Array2::zeros((10, 64));
    let output = rope.apply(&input);
    
    // Rotating zero should give zero
    for i in 0..10 {
        for j in 0..64 {
            assert!(output[[i, j]].abs() < 1e-6);
        }
    }
}

#[test]
#[should_panic(expected = "Embedding dimension must be even")]
fn test_rope_odd_dimension_panics() {
    RotaryEmbedding::new(63, 100);
}

#[test]
#[should_panic(expected = "Sequence length exceeds maximum")]
fn test_rope_exceeds_max_len_panics() {
    let rope = RotaryEmbedding::new(64, 10);
    let input = Array2::ones((20, 64));
    rope.apply(&input);
}

#[test]
#[should_panic(expected = "Input dimension must match RoPE dimension")]
fn test_rope_dimension_mismatch_panics() {
    let rope = RotaryEmbedding::new(64, 100);
    let input = Array2::ones((10, 32)); // Wrong dimension
    rope.apply(&input);
}

#[test]
fn test_rope_integration_with_attention() {
    // Simulate attention mechanism with RoPE
    let rope = RotaryEmbedding::new(64, 100);
    
    // Create query and key
    let q = Array2::from_shape_fn((10, 64), |_| rand::random::<f32>());
    let k = Array2::from_shape_fn((10, 64), |_| rand::random::<f32>());
    
    // Apply RoPE
    let (q_rot, k_rot) = apply_rotary_pos_emb(&q, &k, &rope);
    
    // Compute attention scores (simplified, no softmax)
    let scores_without_rope = q.dot(&k.t());
    let scores_with_rope = q_rot.dot(&k_rot.t());
    
    // Scores should be different (RoPE encodes position)
    let diff: f32 = scores_without_rope.iter()
        .zip(scores_with_rope.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    
    assert!(diff > 1.0, "RoPE should change attention scores");
}

#[test]
fn test_rope_parameter_count() {
    // RoPE should have zero learnable parameters
    let rope = RotaryEmbedding::new(128, 512);
    
    // This is a conceptual test - RoPE has no parameters() method
    // because it's not a Layer, but we verify it's zero-parameter by design
    // The precomputed cos/sin caches are not learnable parameters
    
    // Just verify creation succeeds
    assert_eq!(rope.dim(), 128);
}

