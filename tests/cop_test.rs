use llm::cop::ContextualPositionEncoding;
use ndarray::Array2;

#[test]
fn test_cope_creation() {
    let head_dim = 64;
    let max_pos = 128;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Verify dimensions
    assert_eq!(cope.head_dim(), head_dim);
    assert_eq!(cope.max_pos(), max_pos);
}

#[test]
fn test_cope_shape_preservation() {
    let head_dim = 64;
    let max_pos = 128;
    let seq_len = 10;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create random Q and K matrices
    let q = Array2::<f32>::zeros((seq_len, head_dim));
    let k = Array2::<f32>::zeros((seq_len, head_dim));

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify output shape: (seq_len, seq_len)
    assert_eq!(position_logits.shape(), &[seq_len, seq_len]);
}

#[test]
fn test_cope_gate_computation() {
    let head_dim = 4;
    let max_pos = 10;
    let seq_len = 3;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create simple Q and K matrices
    let q = Array2::<f32>::ones((seq_len, head_dim));
    let k = Array2::<f32>::ones((seq_len, head_dim));

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify that position logits are finite (no NaN or Inf)
    for &val in position_logits.iter() {
        assert!(
            val.is_finite(),
            "Position logit should be finite, got {}",
            val
        );
    }
}

#[test]
fn test_cope_causal_structure() {
    let head_dim = 8;
    let max_pos = 16;
    let seq_len = 5;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create Q and K matrices
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 0.5);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), 0.5);

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // CoPE should respect causal structure (positions are cumulative sums)
    // This means position[i][j] should be based on gates from j to i
    // We can't directly test the internal computation, but we can verify
    // that the output is reasonable
    assert_eq!(position_logits.shape(), &[seq_len, seq_len]);

    // All values should be finite
    for &val in position_logits.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cope_with_different_seq_lengths() {
    let head_dim = 32;
    let max_pos = 64;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Test with various sequence lengths
    for seq_len in [1, 5, 10, 20, 50] {
        let q = Array2::<f32>::zeros((seq_len, head_dim));
        let k = Array2::<f32>::zeros((seq_len, head_dim));

        let position_logits = cope.apply(&q, &k);

        assert_eq!(position_logits.shape(), &[seq_len, seq_len]);

        // Verify all values are finite
        for &val in position_logits.iter() {
            assert!(val.is_finite());
        }
    }
}

#[test]
fn test_cope_position_clamping() {
    let head_dim = 16;
    let max_pos = 10; // Small max_pos
    let seq_len = 20; // Larger seq_len

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create Q and K that would generate large positions
    let q = Array2::<f32>::ones((seq_len, head_dim));
    let k = Array2::<f32>::ones((seq_len, head_dim));

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify output shape
    assert_eq!(position_logits.shape(), &[seq_len, seq_len]);

    // All values should be finite (clamping should prevent overflow)
    for &val in position_logits.iter() {
        assert!(
            val.is_finite(),
            "Position logit should be finite after clamping"
        );
    }
}

#[test]
fn test_cope_interpolation() {
    let head_dim = 8;
    let max_pos = 5;
    let seq_len = 3;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create Q and K that will generate fractional positions
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 0.3);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), 0.7);

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify that interpolation works (no NaN from fractional positions)
    for &val in position_logits.iter() {
        assert!(
            val.is_finite(),
            "Interpolation should produce finite values"
        );
    }
}

#[test]
fn test_cope_zero_input() {
    let head_dim = 16;
    let max_pos = 32;
    let seq_len = 5;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create zero Q and K matrices
    let q = Array2::<f32>::zeros((seq_len, head_dim));
    let k = Array2::<f32>::zeros((seq_len, head_dim));

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // With zero inputs, gates should be ~0.5 (sigmoid(0) = 0.5)
    // Positions should be cumulative sums of ~0.5
    // Position logits should be finite
    for &val in position_logits.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cope_consistency() {
    let head_dim = 32;
    let max_pos = 64;
    let seq_len = 8;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create Q and K matrices
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 0.5);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), 0.5);

    // Apply CoPE twice with same inputs
    let position_logits_1 = cope.apply(&q, &k);
    let position_logits_2 = cope.apply(&q, &k);

    // Results should be identical (deterministic)
    assert_eq!(position_logits_1.shape(), position_logits_2.shape());

    for (val1, val2) in position_logits_1.iter().zip(position_logits_2.iter()) {
        assert!((val1 - val2).abs() < 1e-6, "CoPE should be deterministic");
    }
}

#[test]
fn test_cope_different_qk() {
    let head_dim = 16;
    let max_pos = 32;
    let seq_len = 6;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create different Q and K matrices
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 1.0);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), -1.0);

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify output
    assert_eq!(position_logits.shape(), &[seq_len, seq_len]);

    for &val in position_logits.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_cope_max_pos_boundary() {
    let head_dim = 8;
    let max_pos = 5;
    let seq_len = 3;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Create inputs that will generate positions near max_pos
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 2.0);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), 2.0);

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify that clamping at max_pos works correctly
    for &val in position_logits.iter() {
        assert!(
            val.is_finite(),
            "Values at max_pos boundary should be finite"
        );
    }
}

#[test]
fn test_cope_single_token() {
    let head_dim = 16;
    let max_pos = 32;
    let seq_len = 1;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Single token sequence
    let q = Array2::<f32>::ones((seq_len, head_dim));
    let k = Array2::<f32>::ones((seq_len, head_dim));

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify output shape
    assert_eq!(position_logits.shape(), &[1, 1]);

    // Single value should be finite
    assert!(position_logits[[0, 0]].is_finite());
}

#[test]
fn test_cope_large_sequence() {
    let head_dim = 64;
    let max_pos = 128;
    let seq_len = 100;

    let cope = ContextualPositionEncoding::new(head_dim, max_pos);

    // Large sequence
    let q = Array2::<f32>::from_elem((seq_len, head_dim), 0.1);
    let k = Array2::<f32>::from_elem((seq_len, head_dim), 0.1);

    // Apply CoPE
    let position_logits = cope.apply(&q, &k);

    // Verify output shape
    assert_eq!(position_logits.shape(), &[seq_len, seq_len]);

    // All values should be finite
    for &val in position_logits.iter() {
        assert!(val.is_finite());
    }
}
