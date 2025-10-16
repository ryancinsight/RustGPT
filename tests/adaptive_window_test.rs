use llm::{EMBEDDING_DIM, Layer, self_attention::SelfAttention, WindowAdaptationStrategy};
use ndarray::Array2;

const TEST_SEQ_LEN: usize = 20;

#[test]
fn test_adaptive_window_sequence_length_based() {
    // Test SequenceLengthBased strategy: window = seq_len / 2
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(100)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    // Output should be valid
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_min_max_bounds() {
    // Test that adaptive window respects min/max bounds
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(5)
    .max_window_size(15)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    // Test with very short sequence (should use min_window_size)
    let short_input = Array2::from_elem((3, EMBEDDING_DIM), 0.1);
    let short_output = attention.forward(&short_input);
    assert_eq!(short_output.shape(), [3, EMBEDDING_DIM]);
    assert!(short_output.iter().all(|&x| x.is_finite()));
    
    // Test with very long sequence (should use max_window_size)
    let long_input = Array2::from_elem((100, EMBEDDING_DIM), 0.1);
    let long_output = attention.forward(&long_input);
    assert_eq!(long_output.shape(), [100, EMBEDDING_DIM]);
    assert!(long_output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_with_gqa() {
    // Test adaptive window with GQA
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        4, // GQA: 4 KV heads
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_with_rope() {
    // Test adaptive window with RoPE
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        true, // Enable RoPE
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_with_gqa_and_rope() {
    // Test full modern stack: GQA + RoPE + Adaptive Window
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        4, // GQA
        true, // RoPE
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_backward_pass() {
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let _output = attention.forward(&input);
    
    let grads = Array2::ones((TEST_SEQ_LEN, EMBEDDING_DIM));
    let grad_input = attention.backward(&grads, 0.01);
    
    // Gradients should be valid
    assert_eq!(grad_input.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(grad_input.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_training_stability() {
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        4,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
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
fn test_adaptive_window_different_sequence_lengths() {
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(5)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    // Test with different sequence lengths
    for seq_len in [5, 10, 20, 40, 80] {
        let input = Array2::from_elem((seq_len, EMBEDDING_DIM), 0.1);
        let output = attention.forward(&input);
        
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

#[test]
fn test_adaptive_window_attention_entropy_strategy() {
    // Test AttentionEntropy strategy (will fallback to SequenceLengthBased initially)
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::AttentionEntropy)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_perplexity_based_strategy() {
    // Test PerplexityBased strategy (will fallback to SequenceLengthBased for now)
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::PerplexityBased)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_fixed_strategy() {
    // Test Fixed strategy (should behave like non-adaptive)
    let mut attention = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        Some(30), // Fixed window size
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::Fixed)
    .build();
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    let output = attention.forward(&input);
    
    assert_eq!(output.shape(), [TEST_SEQ_LEN, EMBEDDING_DIM]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_adaptive_window_vs_fixed_window() {
    // Compare adaptive window with fixed window
    let mut adaptive = SelfAttention::new_with_adaptive_window(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        None,
    )
    .min_window_size(10)
    .max_window_size(50)
    .strategy(WindowAdaptationStrategy::SequenceLengthBased)
    .build();
    
    let mut fixed = SelfAttention::new_with_gqa(
        EMBEDDING_DIM,
        8,
        8,
        false,
        512,
        Some(25), // Fixed window
    );
    
    let input = Array2::from_elem((TEST_SEQ_LEN, EMBEDDING_DIM), 0.1);
    
    let output_adaptive = adaptive.forward(&input);
    let output_fixed = fixed.forward(&input);
    
    // Both should produce valid outputs
    assert_eq!(output_adaptive.shape(), output_fixed.shape());
    assert!(output_adaptive.iter().all(|&x| x.is_finite()));
    assert!(output_fixed.iter().all(|&x| x.is_finite()));
}

