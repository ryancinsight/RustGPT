use llm::Vocab;

#[test]
fn test_vocab_encode_decode() {
    let words = vec!["hello", "world", "this", "is", "rust", "</s>"];
    let vocab = Vocab::new(words);

    // Test encoding
    assert_eq!(vocab.encode("hello"), Some(0));
    assert_eq!(vocab.encode("world"), Some(1));
    assert_eq!(vocab.encode("unknown"), None);

    // Test decoding
    assert_eq!(vocab.decode(0), Some("hello"));
    assert_eq!(vocab.decode(1), Some("world"));
    assert_eq!(vocab.decode(999), None);

    // Test size
    assert_eq!(vocab.size(), 6);

    // Test contains
    assert!(vocab.contains("hello"));
    assert!(!vocab.contains("unknown"));

    // Test batch operations
    let batch_encode = vocab.encode_batch(&["hello", "world", "unknown"]);
    assert_eq!(batch_encode, vec![Some(0), Some(1), None]);

    let batch_decode = vocab.decode_batch(vec![0, 1, 999]);
    assert_eq!(batch_decode, vec![Some("hello"), Some("world"), None]);

    // Test iterator
    let words: Vec<&str> = vocab.iter_words().collect();
    assert_eq!(words, vec!["hello", "world", "this", "is", "rust", "</s>"]);

    // Test unknown token
    assert_eq!(vocab.encode_or_unknown("unknown"), None);
    assert_eq!(vocab.unknown_token(), None);
}

#[test]
fn test_vocab_default() {
    let vocab = Vocab::default();

    // Test that default vocab contains expected words
    assert!(vocab.encode("hello").is_some());
    assert!(vocab.encode("world").is_some());
    assert!(vocab.encode("</s>").is_some());
}
