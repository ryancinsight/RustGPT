use llm::{EMBEDDING_DIM, Layer, poly_attention::PolyAttention};
use ndarray::Array2;

#[test]
fn test_self_attention_forward() {
    // Create self-attention module
    let mut self_attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, None);

    // Create input tensor (batch_size=1, seq_len=3, embedding_dim=EMBEDDING_DIM)
    let input = Array2::ones((3, EMBEDDING_DIM));

    // Test forward pass
    let output = self_attention.forward(&input);

    // Check output shape - should be same as input
    assert_eq!(output.shape(), input.shape());
}

#[test]
fn test_self_attention_with_different_sequence_lengths() {
    // Create self-attention module
    let mut self_attention = PolyAttention::new(EMBEDDING_DIM, 8, 3, 64, None);

    // Test with different sequence lengths
    for seq_len in 1..5 {
        // Create input tensor
        let input = Array2::ones((seq_len, EMBEDDING_DIM));

        // Test forward pass
        let output = self_attention.forward(&input);

        // Check output shape
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}
