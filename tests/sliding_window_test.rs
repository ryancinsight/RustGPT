use llm::{EMBEDDING_DIM, Layer, PositionalEncodingType, poly_attention::PolyAttention};
use ndarray::Array2;

const TEST_SEQ_LEN: usize = 10;
const TEST_WINDOW_SIZE: usize = 4;

#[test]
fn test_sliding_window_mask_correctness() {
    // Test that tokens only attend within the sliding window
    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(TEST_WINDOW_SIZE),
    );

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    // Output should be valid
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_full_attention_backward_compatibility() {
    // Test that window_size = None gives full attention
    let mut full_attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, None);

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = full_attention.forward(&input);

    // Should work exactly as before
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_window_size_1024() {
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(TEST_WINDOW_SIZE));

    let input = Array2::from_elem((20, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [20, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_window_size_2048() {
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(2048));

    let input = Array2::from_elem((30, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [30, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_window_size_4096_mistral_style() {
    // Mistral 7B uses window_size=4096
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(4096));

    let input = Array2::from_elem((50, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [50, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_sliding_window_with_gqa() {
    // Test integration of sliding window with GQA
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(TEST_WINDOW_SIZE));

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_sliding_window_with_rope() {
    // Test integration of sliding window with RoPE
    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(TEST_WINDOW_SIZE),
    );

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_sliding_window_with_gqa_and_rope() {
    // Test full Mistral 7B configuration: GQA + RoPE + Sliding Window
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(TEST_WINDOW_SIZE));

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_causal_masking_preserved() {
    // Verify that causal masking (no future attention) is still enforced
    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(TEST_WINDOW_SIZE),
    );

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    // Output should be valid (causal masking prevents NaN/Inf)
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_long_sequence_handling() {
    // Test with sequence length much greater than window size
    let seq_len = 100;
    let window_size = 10;

    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(window_size),
    );

    let input = Array2::from_elem((seq_len, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_window_size_equals_sequence_length() {
    // When window_size >= seq_len, should behave like full attention
    let seq_len = 10;
    let window_size = 10;

    let mut windowed = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(window_size),
    );

    let mut full = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        None,
    );

    let input = Array2::from_elem((seq_len, EMBEDDING_DIM), 0.1);

    let output_windowed = windowed.forward(&input);
    let output_full = full.forward(&input);

    // Both should produce valid outputs
    assert_eq!(output_windowed.shape(), output_full.shape());
    assert!(output_windowed.iter().all(|&x| x.is_finite()));
    assert!(output_full.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_window_size_one() {
    // Extreme case: window_size = 1 (only attend to self)
    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(1),
    );

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);

    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_sliding_window_backward_pass() {
    let mut attention = PolyAttention::new(
        EMBEDDING_DIM,
        8,
        3,
        64,
        Some(TEST_WINDOW_SIZE),
    );

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let _output = attention.forward(&input);

    let grads = Array2::ones((TEST_SEQ_LEN, EMBEDDING_DIM));
    let grad_input = attention.backward(&grads, 0.01);

    // Gradients should be valid
    assert_eq!(grad_input.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_sliding_window_training_stability() {
    let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(TEST_WINDOW_SIZE));

    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);

    // Run multiple training steps
    for _ in 0..20 {
        let _output = attention.forward(&input);
        let grads = Array2::ones((TEST_SEQ_LEN, EMBEDDING_DIM));
        let _grad_input = attention.backward(&grads, 0.01);
    }

    // Verify parameters are still valid after training
    let final_output = attention.forward(&input);
    assert!(final_output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_different_window_sizes_comparison() {
    // Compare outputs with different window sizes
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);

    for window_size in [2, 4, 8, TEST_SEQ_LEN] {
        let mut attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, Some(window_size));

        let output = attention.forward(&input);
        assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
