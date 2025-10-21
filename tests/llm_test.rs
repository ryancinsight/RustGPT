use llm::{
    EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM, LayerEnum, MAX_SEQ_LEN, Vocab,
    layer_norm::LayerNorm, output_projection::OutputProjection, self_attention::SelfAttention,
    swiglu::SwiGLU,
};
use ndarray::{Array2, Axis};
use proptest::prelude::*;

// Test fixture: creates a default vocabulary for testing
fn create_test_vocab() -> Vocab {
    Vocab::default()
}

// Test fixture: creates a test vocabulary with custom words
fn create_custom_vocab(words: Vec<&str>) -> Vocab {
    Vocab::new(words)
}

// =========================================================================
// Basic Functionality Tests
// =========================================================================

#[test]
fn test_llm_tokenize() {
    let vocab = create_test_vocab();
    let vocab_size = vocab.words.len();
    let llm = LLM::new(
        vocab,
        vec![LayerEnum::OutputProjection(OutputProjection::new(
            EMBEDDING_DIM,
            vocab_size,
        ))],
    );

    // Test tokenization produces expected tokens
    let tokens = llm.tokenize("hello world");
    assert_eq!(tokens.len(), 2, "Expected 2 tokens for 'hello world'");

    // Verify specific token mappings
    assert_eq!(tokens[0], llm.vocab.encode("hello").unwrap());
    assert_eq!(tokens[1], llm.vocab.encode("world").unwrap());

    // Test that all tokens can be decoded back
    tokens.iter().for_each(|&token| {
        assert!(
            llm.vocab.decode(token).is_some(),
            "Token {} should be decodable",
            token
        );
    });
}

#[test]
fn test_llm_predict() {
    // Use default LLM which has proper layer structure
    let mut llm = LLM::default();

    // Test prediction produces non-empty output
    let input_text = "hello world this is rust";
    let result = llm.predict(input_text);

    assert!(!result.is_empty(), "Prediction should produce output");

    // Note: Default model is untrained, so we don't enforce stop token ending
}

#[test]
fn test_llm_train() {
    // Use default LLM for training test
    let mut llm = LLM::default();

    let training_data = vec!["<pad> <unk> </s>"];

    // Training should complete without panicking
    let _ = llm.train(training_data, 10, 0.01);
}

// =========================================================================
// Edge Case Tests
// =========================================================================

#[test]
fn test_llm_tokenize_empty_input() {
    // Use default LLM for tokenization test
    let llm = LLM::default();

    let tokens = llm.tokenize("");
    assert!(tokens.is_empty(), "Empty input should produce no tokens");
}

#[test]
fn test_llm_tokenize_unknown_words() {
    let vocab = create_custom_vocab(vec!["hello", "world", "</s>"]);
    let vocab_size = vocab.words.len();
    let llm = LLM::new(
        vocab,
        vec![LayerEnum::OutputProjection(OutputProjection::new(
            EMBEDDING_DIM,
            vocab_size,
        ))],
    );

    // "unknown" is not in vocab, should be skipped
    let tokens = llm.tokenize("hello unknown world");
    assert_eq!(tokens.len(), 2, "Unknown words should be filtered out");
    assert_eq!(tokens[0], llm.vocab.encode("hello").unwrap());
    assert_eq!(tokens[1], llm.vocab.encode("world").unwrap());
}

#[test]
fn test_llm_tokenize_punctuation() {
    let vocab = create_custom_vocab(vec!["hello", "world", ".", "!", "</s>"]);
    let vocab_size = vocab.words.len();
    let llm = LLM::new(
        vocab,
        vec![LayerEnum::OutputProjection(OutputProjection::new(
            EMBEDDING_DIM,
            vocab_size,
        ))],
    );

    let tokens = llm.tokenize("hello world.");
    assert_eq!(tokens.len(), 3, "Punctuation should be separate token");
    assert_eq!(tokens[0], llm.vocab.encode("hello").unwrap());
    assert_eq!(tokens[1], llm.vocab.encode("world").unwrap());
    assert_eq!(tokens[2], llm.vocab.encode(".").unwrap());
}

#[test]
fn test_llm_predict_empty_input() {
    // Use default LLM
    let mut llm = LLM::default();

    let result = llm.predict("");
    assert!(result.is_empty(), "Empty input should produce empty output");
}

#[test]
fn test_llm_train_empty_data() {
    let mut llm = LLM::default();

    // Training with empty data should not panic
    let _ = llm.train(vec![], 5, 0.01);
}

#[test]
fn test_llm_train_single_token() {
    let mut llm = LLM::default();

    // Single token sequences should be skipped (need at least 2 for input/target)
    let _ = llm.train(vec!["hello"], 5, 0.01);
}

// =========================================================================
// Parameter Count Consistency with SwiGLU
// =========================================================================

#[test]
fn test_parameter_count_consistency() {
    // Test that parameter count is consistent across multiple instantiations
    let vocab = create_test_vocab();
    let vocab_size = vocab.encode.len();

    let llm1 = LLM::new(
        vocab.clone(),
        vec![
            LayerEnum::Embeddings(Embeddings::new(vocab.clone())),
            LayerEnum::SelfAttention(Box::new(SelfAttention::new(EMBEDDING_DIM))),
            LayerEnum::LayerNorm(LayerNorm::new(EMBEDDING_DIM)),
            LayerEnum::SwiGLU(Box::new(SwiGLU::new(EMBEDDING_DIM, HIDDEN_DIM))),
            LayerEnum::LayerNorm(LayerNorm::new(EMBEDDING_DIM)),
            LayerEnum::OutputProjection(OutputProjection::new(EMBEDDING_DIM, vocab_size)),
        ],
    );

    let llm2 = LLM::new(
        vocab.clone(),
        vec![
            LayerEnum::Embeddings(Embeddings::new(vocab.clone())),
            LayerEnum::SelfAttention(Box::new(SelfAttention::new(EMBEDDING_DIM))),
            LayerEnum::LayerNorm(LayerNorm::new(EMBEDDING_DIM)),
            LayerEnum::SwiGLU(Box::new(SwiGLU::new(EMBEDDING_DIM, HIDDEN_DIM))),
            LayerEnum::LayerNorm(LayerNorm::new(EMBEDDING_DIM)),
            LayerEnum::OutputProjection(OutputProjection::new(EMBEDDING_DIM, vocab_size)),
        ],
    );

    assert_eq!(
        llm1.total_parameters(),
        llm2.total_parameters(),
        "Parameter count should be deterministic",
    );
}

// =========================================================================
// Helper Functions for Testing Mathematical Properties
// =========================================================================

fn manual_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = logits.clone();

    for mut row in result.rows_mut() {
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();

        for (i, &exp_val) in exp_values.iter().enumerate() {
            row[i] = exp_val / sum_exp;
        }
    }

    result
}

fn manual_greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
    probs
        .map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a
                    .partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap()
        })
        .to_vec()
}

// =========================================================================
// Mathematical Property Tests
// =========================================================================

#[test]
fn test_softmax_properties() {
    // Test that softmax produces valid probability distributions
    let logits = Array2::from_shape_vec(
        (2, 4),
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0],
    )
    .unwrap();

    let softmax_result = manual_softmax(&logits);

    // Property 1: All values should be in [0, 1]
    for &val in softmax_result.iter() {
        assert!((0.0..=1.0).contains(&val), "Softmax value {} not in [0,1]", val);
    }

    // Property 2: Each row should sum to 1.0
    for row in softmax_result.rows() {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax row sum {} != 1.0", sum);
    }

    // Property 3: Largest logit should produce largest probability
    for (row_idx, row) in logits.rows().into_iter().enumerate() {
        let max_logit_idx = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let max_prob_idx = softmax_result
            .row(row_idx)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        assert_eq!(
            max_logit_idx, max_prob_idx,
            "Max logit index should match max probability index",
        );
    }
}

#[test]
fn test_softmax_numerical_stability() {
    // Test with very large values (should not overflow)
    let large_logits = Array2::from_shape_vec((1, 3), vec![1000.0, 1001.0, 999.0]).unwrap();
    let result = manual_softmax(&large_logits);

    for &val in result.iter() {
        assert!(val.is_finite(), "Softmax should handle large values without overflow");
        assert!((0.0..=1.0).contains(&val), "Softmax value {} not in [0,1]", val);
    }

    // Test with very small values (should not underflow to zero)
    let small_logits = Array2::from_shape_vec((1, 3), vec![-1000.0, -999.0, -1001.0]).unwrap();
    let result = manual_softmax(&small_logits);

    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "Softmax should handle small values, sum={}", sum);
}

#[test]
fn test_greedy_decode_properties() {
    // Test that greedy decode selects maximum probability
    let probs = Array2::from_shape_vec(
        (2, 4),
        vec![
            0.1, 0.2, 0.6, 0.1, // max at index 2
            0.4, 0.1, 0.1, 0.4, // max at index 0 or 3
        ],
    )
    .unwrap();

    let decoded = manual_greedy_decode(&probs);

    assert_eq!(decoded.len(), 2, "Should decode one token per row");
    assert_eq!(decoded[0], 2, "Should select index with max probability");
    assert!(decoded[1] == 0 || decoded[1] == 3, "Should select one of the max indices");
}

// =========================================================================
// Persistence Round-Trip
// =========================================================================

#[test]
fn test_llm_save_load() {
    use std::fs;
    use tempfile::NamedTempFile;

    let original_llm = LLM::default();

    // Save to a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap();
    original_llm.save(path).expect("Failed to save LLM");

    // Load from file
    let loaded_llm = LLM::load(path).expect("Failed to load LLM");

    // Verify that the loaded LLM has the same structure
    assert_eq!(original_llm.vocab.encode.len(), loaded_llm.vocab.encode.len());
    assert_eq!(original_llm.network.len(), loaded_llm.network.len());
    assert_eq!(original_llm.total_parameters(), loaded_llm.total_parameters());

    // Clean up
    fs::remove_file(path).unwrap();
}

// =========================================================================
// Property-Based Tests
// =========================================================================

proptest! {
    #[test]
    fn prop_tokenize_produces_valid_indices(s in "[a-z ]{1,50}") {
        let llm = LLM::default();
        let vocab_size = llm.vocab.words.len();

        let tokens = llm.tokenize(&s);

        for &token in &tokens {
            prop_assert!(token < vocab_size, "Token {} exceeds vocab size {}", token, vocab_size);
            prop_assert!(llm.vocab.decode(token).is_some(), "Token {} should be decodable", token);
        }
    }

    #[test]
    fn prop_tokenize_length_bounded(s in "[a-z ]{1,100}") {
        let llm = LLM::default();
        let tokens = llm.tokenize(&s);
        let word_count = s.split_whitespace().count();

        prop_assert!(tokens.len() <= word_count * 2,
            "Token count {} should not exceed 2x word count {}",
            tokens.len(), word_count);
    }
}
