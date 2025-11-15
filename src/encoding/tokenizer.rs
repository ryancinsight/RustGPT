//! # Tokenizer Module
//!
//! Core tokenization algorithms for converting raw text into token sequences.
//! This module provides the SimpleTokenizer which handles word-level tokenization
//! with punctuation splitting and unknown token handling.

/// Simple word-level tokenizer that splits on whitespace and punctuation
#[derive(Clone, Debug)]
pub struct SimpleTokenizer;

impl SimpleTokenizer {
    /// Create a new simple tokenizer
    pub fn new() -> Self {
        Self
    }

    /// Tokenize text into words by splitting on whitespace and punctuation
    pub fn tokenize(&self, text: &str, vocab: &super::Vocab) -> Vec<usize> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            // Special case for end token
            if word == "</s>" {
                if let Some(token_id) = vocab.encode(word) {
                    tokens.push(token_id);
                }
                continue;
            }

            // Split on punctuation and keep both the word parts and punctuation
            let word_tokens = word
                .split(|c: char| c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .chain(
                    word.chars()
                        .filter(|c| c.is_ascii_punctuation())
                        .map(|c| c.to_string()),
                );

            for token in word_tokens {
                if let Some(token_id) = vocab.encode(&token) {
                    tokens.push(token_id);
                } else if let Some(unk_id) = vocab.unknown_token().and_then(|unk| vocab.encode(unk))
                {
                    tokens.push(unk_id);
                }
            }
        }

        tokens
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vocab;

    #[test]
    fn test_simple_tokenization() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = Vocab::new(vec!["hello", "world", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello world", &vocab);
        assert_eq!(tokens.len(), 2);
        assert_eq!(vocab.decode(tokens[0]), Some("hello"));
        assert_eq!(vocab.decode(tokens[1]), Some("world"));
    }

    #[test]
    fn test_punctuation_splitting() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = Vocab::new(vec!["hello", ",", "world", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello, world", &vocab);
        assert_eq!(tokens.len(), 3); // hello, ,, world
        assert_eq!(vocab.decode(tokens[0]), Some("hello"));
        assert_eq!(vocab.decode(tokens[1]), Some(","));
        assert_eq!(vocab.decode(tokens[2]), Some("world"));
    }

    #[test]
    fn test_unknown_token() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = Vocab::new(vec!["hello", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello unknown", &vocab);
        assert_eq!(tokens.len(), 2); // hello, <unk>
        assert_eq!(vocab.decode(tokens[0]), Some("hello"));
        assert_eq!(vocab.decode(tokens[1]), Some("<unk>"));
    }

    #[test]
    fn test_end_token_special_handling() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = Vocab::new(vec!["hello", "world", "</s>", "<unk>"]);

        // Test that </s> is treated as a single token
        let tokens = tokenizer.tokenize("hello world </s>", &vocab);
        assert_eq!(tokens.len(), 3); // hello, world, </s>
        assert_eq!(vocab.decode(tokens[0]), Some("hello"));
        assert_eq!(vocab.decode(tokens[1]), Some("world"));
        assert_eq!(vocab.decode(tokens[2]), Some("</s>"));
    }
}
