use ndarray::Array2;

use llm::{network::Layer, Vocab};
use llm::embeddings::TokenEmbeddings;

#[test]
fn token_embeddings_forward_clamps_and_sanitizes_token_ids() {
    let vocab = Vocab::default();
    let vocab_size = vocab.size();

    let mut emb = TokenEmbeddings::new(vocab);
    // Make embeddings deterministic for assertions.
    emb.token_embeddings = Array2::from_shape_fn((vocab_size, llm::EMBEDDING_DIM), |(i, j)| {
        (i * 1000 + j) as f32
    });

    let input = Array2::from_shape_vec((1, 3), vec![-1.0, f32::NAN, 999.0]).unwrap();
    let out = emb.forward(&input);

    // -1 and NaN map to 0, huge id maps to vocab_size-1.
    let last = vocab_size - 1;

    assert_eq!(out[[0, 0]], 0.0);
    assert_eq!(out[[0, llm::EMBEDDING_DIM - 1]], (llm::EMBEDDING_DIM - 1) as f32);

    assert_eq!(out[[1, 0]], 0.0);
    assert_eq!(out[[1, llm::EMBEDDING_DIM - 1]], (llm::EMBEDDING_DIM - 1) as f32);

    assert_eq!(out[[2, 0]], (last * 1000) as f32);
    assert_eq!(
        out[[2, llm::EMBEDDING_DIM - 1]],
        (last * 1000 + (llm::EMBEDDING_DIM - 1)) as f32
    );
}

#[test]
fn token_embeddings_compute_gradients_accumulates_repeated_tokens() {
    let vocab = Vocab::default();
    let vocab_size = vocab.size();

    let emb = TokenEmbeddings::new(vocab);

    // token ids: [1, 1, 2]
    let input = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 2.0]).unwrap();

    // grads per position: row0=1, row1=2, row2=3
    let mut output_grads = Array2::<f32>::zeros((3, llm::EMBEDDING_DIM));
    for j in 0..llm::EMBEDDING_DIM {
        output_grads[[0, j]] = 1.0;
        output_grads[[1, j]] = 2.0;
        output_grads[[2, j]] = 3.0;
    }

    let (input_grads, param_grads) = emb.compute_gradients(&input, &output_grads);

    // No gradients into token ids.
    assert_eq!(input_grads.dim(), (1, 3));
    assert!(input_grads.iter().all(|&x| x == 0.0));

    assert_eq!(param_grads.len(), 1);
    let token_grads = &param_grads[0];
    assert_eq!(token_grads.dim(), (vocab_size, llm::EMBEDDING_DIM));

    // token 1 accumulates rows 0 and 1 => 3.0, token 2 accumulates row 2 => 3.0
    assert_eq!(token_grads[[1, 0]], 3.0);
    assert_eq!(token_grads[[1, llm::EMBEDDING_DIM - 1]], 3.0);
    assert_eq!(token_grads[[2, 0]], 3.0);
    assert_eq!(token_grads[[2, llm::EMBEDDING_DIM - 1]], 3.0);

    // token 0 should be untouched.
    assert_eq!(token_grads[[0, 0]], 0.0);
}
