use llm::{EMBEDDING_DIM, Layer, PositionalEncodingType, self_attention::SelfAttention};
use ndarray::Array2;

#[test]
fn test_gqa_creation() {
    let attention = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(attention.num_heads, 8);
    assert_eq!(attention.num_kv_heads, 4);
    assert_eq!(attention.embedding_dim, EMBEDDING_DIM);
}

#[test]
fn test_mha_backward_compatibility() {
    let attention_mha = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        8,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(attention_mha.num_heads, 8);
    assert_eq!(attention_mha.num_kv_heads, 8);
    let attention_standard = SelfAttention::new(EMBEDDING_DIM);
    assert_eq!(attention_mha.num_heads, attention_standard.num_heads);
    assert_eq!(attention_mha.num_kv_heads, attention_standard.num_kv_heads);
}

#[test]
fn test_mqa_extreme_case() {
    let attention = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        1,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(attention.num_heads, 8);
    assert_eq!(attention.num_kv_heads, 1);
}

#[test]
#[should_panic(expected = "num_heads must be divisible by num_kv_heads")]
fn test_gqa_invalid_grouping() {
    SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        3,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
}

#[test]
#[should_panic(expected = "num_heads must be divisible by num_kv_heads")]
fn test_gqa_invalid_kv_heads() {
    SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        4,
        8,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
}

#[test]
fn test_gqa_parameter_reduction() {
    let head_dim = EMBEDDING_DIM / 8;
    let cope_params = 8 * (64 + 1) * head_dim;
    let mha = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        8,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let mha_params = mha.parameters();
    let expected_mha = 8 * head_dim * head_dim + 8 * 2 * head_dim * head_dim + cope_params;
    assert_eq!(mha_params, expected_mha);
    let gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let gqa_params = gqa.parameters();
    let expected_gqa = 8 * head_dim * head_dim + 4 * 2 * head_dim * head_dim + cope_params;
    assert_eq!(gqa_params, expected_gqa);
    let reduction = mha_params - gqa_params;
    let expected_reduction = 4 * 2 * head_dim * head_dim;
    assert_eq!(reduction, expected_reduction);
    let mqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        1,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let mqa_params = mqa.parameters();
    let expected_mqa = 8 * head_dim * head_dim + 2 * head_dim * head_dim + cope_params;
    assert_eq!(mqa_params, expected_mqa);
}

#[test]
fn test_gqa_forward_pass() {
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let input = Array2::ones((5, EMBEDDING_DIM));
    let output = gqa.forward(&input);
    assert_eq!(output.shape(), [5, EMBEDDING_DIM]);
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_gqa_with_rope() {
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::CoPE { max_pos: 64 },
        512,
        None,
    );
    let input = Array2::ones((5, EMBEDDING_DIM));
    let output = gqa.forward(&input);
    assert_eq!(output.shape(), [5, EMBEDDING_DIM]);
    assert!(output.iter().any(|&x| x != 0.0));
}

#[test]
fn test_gqa_backward_pass() {
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let input = Array2::ones((5, EMBEDDING_DIM));
    let _output = gqa.forward(&input);
    let grads = Array2::ones((5, EMBEDDING_DIM));
    let grad_input = gqa.backward(&grads, 0.01);
    assert_eq!(grad_input.shape(), [5, EMBEDDING_DIM]);
    assert!(grad_input.iter().any(|&x| x != 0.0));
}

#[test]
fn test_gqa_different_sequence_lengths() {
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    for seq_len in [1, 3, 5, 10] {
        let input = Array2::ones((seq_len, EMBEDDING_DIM));
        let output = gqa.forward(&input);
        assert_eq!(output.shape(), [seq_len, EMBEDDING_DIM]);
    }
}

#[test]
fn test_gqa_vs_mha_output_similarity() {
    let mut mha = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        8,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let input = Array2::from_elem((5, EMBEDDING_DIM), 0.1);
    let output_mha = mha.forward(&input);
    let output_gqa = gqa.forward(&input);
    assert_eq!(output_mha.shape(), output_gqa.shape());
    assert!(output_mha.iter().any(|&x| x != 0.0));
    assert!(output_gqa.iter().any(|&x| x != 0.0));
}

#[test]
fn test_gqa_kv_cache_size_reduction() {
    let head_dim = EMBEDDING_DIM / 8;
    let _mha = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        8,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let mha_kv_params = 8 * 2 * head_dim * head_dim;
    let _gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let gqa_kv_params = 4 * 2 * head_dim * head_dim;
    assert_eq!(mha_kv_params, gqa_kv_params * 2);
    let _mqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        1,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let mqa_kv_params = 2 * head_dim * head_dim;
    assert_eq!(mha_kv_params, mqa_kv_params * 8);
}

#[test]
fn test_gqa_training_stability() {
    let mut gqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let input = Array2::from_elem((5, EMBEDDING_DIM), 0.1);
    for _ in 0..10 {
        let _output = gqa.forward(&input);
        let grads = Array2::ones((5, EMBEDDING_DIM));
        let _grad_input = gqa.backward(&grads, 0.01);
    }
    let final_output = gqa.forward(&input);
    assert!(final_output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_gqa_with_rope_integration() {
    let mut gqa_with_rope = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::CoPE { max_pos: 64 },
        512,
        None,
    );
    let mut gqa_without_rope = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    let input = Array2::from_elem((5, EMBEDDING_DIM), 0.1);
    let output_with_rope = gqa_with_rope.forward(&input);
    let output_without_rope = gqa_without_rope.forward(&input);
    assert_eq!(output_with_rope.shape(), output_without_rope.shape());
    assert!(output_with_rope.iter().all(|&x| x.is_finite()));
    assert!(output_without_rope.iter().all(|&x| x.is_finite()));
    let diff: f32 = output_with_rope
        .iter()
        .zip(output_without_rope.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff > 0.0, "CoPE should change the output");
}

#[test]
fn test_gqa_grouping_correctness() {
    let gqa_4kv = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        4,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(gqa_4kv.num_heads / gqa_4kv.num_kv_heads, 2);
    let gqa_2kv = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        2,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(gqa_2kv.num_heads / gqa_2kv.num_kv_heads, 4);
    let mqa = SelfAttention::new_with_positional_encoding(
        EMBEDDING_DIM,
        8,
        1,
        &PositionalEncodingType::Learned,
        512,
        None,
    );
    assert_eq!(mqa.num_heads / mqa.num_kv_heads, 8);
}

#[test]
fn test_gqa_parameter_count_consistency() {
    let head_dim = EMBEDDING_DIM / 8;
    let cope_params = 8 * (64 + 1) * head_dim;
    for num_kv_heads in [1, 2, 4, 8] {
        let gqa = SelfAttention::new_with_positional_encoding(
            EMBEDDING_DIM,
            8,
            num_kv_heads,
            &PositionalEncodingType::Learned,
            512,
            None,
        );
        let expected_params = 8 * head_dim * head_dim + num_kv_heads * 2 * head_dim * head_dim + cope_params;
        assert_eq!(
            gqa.parameters(),
            expected_params,
            "Parameter count mismatch for num_kv_heads={}",
            num_kv_heads
        );
    }
}
