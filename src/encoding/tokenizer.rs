//! # Tokenizer Module
//!
//! Core tokenization algorithms for converting raw text into token sequences.
//! This module provides the SimpleTokenizer which handles word-level tokenization
//! with punctuation splitting and unknown token handling.

use std::collections::HashSet;

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
            // Split on punctuation and keep both the word parts and punctuation
            let word_tokens = word
                .split(|c: char| c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .chain(
                    word.chars()
                        .filter(|c| c.is_ascii_punctuation())
                        .map(|c| c.to_string())
                );

            for token in word_tokens {
                if let Some(token_id) = vocab.encode(&token) {
                    tokens.push(token_id);
                } else if let Some(unk_id) = vocab.unknown_token()
                    .and_then(|unk| vocab.encode(unk)) {
                    tokens.push(unk_id);
                }
            }
        }

        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, token_ids: &[usize], vocab: &super::Vocab) -> String {
        let mut result = String::new();
        let mut first = true;

        for &token_id in token_ids {
            if let Some(word) = vocab.decode(token_id) {
                if !first && !word.chars().all(|c| c.is_ascii_punctuation()) {
                    result.push(' ');
                }
                result.push_str(word);
                first = false;
            }
        }

        result
    }

    /// Extract vocabulary from a stream of texts
    pub fn extract_vocab_from_texts<I, S>(texts: I) -> Vec<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut vocab_set = HashSet::new();

        // Always include special tokens
        vocab_set.insert("</s>".to_string());
        vocab_set.insert("<unk>".to_string());

        for text in texts.into_iter() {
            let s = text.as_ref();
            for word in s.split_whitespace() {
                // Split on punctuation and collect all parts
                let tokens = word
                    .split(|c: char| c.is_ascii_punctuation())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .chain(
                        word.chars()
                            .filter(|c| c.is_ascii_punctuation())
                            .map(|c| c.to_string())
                    );

                for token in tokens {
                    vocab_set.insert(token);
                }
            }
        }

        let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
        vocab_words.sort();
        vocab_words
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

    #[test]
    fn test_simple_tokenization() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = super::Vocab::new(vec!["hello", "world", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello world", &vocab);
        assert_eq!(tokens.len(), 2);

        let decoded = tokenizer.decode(&tokens, &vocab);
        assert_eq!(decoded, "hello world");
    }

    #[test]
    fn test_punctuation_splitting() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = super::Vocab::new(vec!["hello", ",", "world", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello, world", &vocab);
        assert_eq!(tokens.len(), 3); // hello, ,, world

        let decoded = tokenizer.decode(&tokens, &vocab);
        assert_eq!(decoded, "hello, world");
    }

    #[test]
    fn test_unknown_token() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = super::Vocab::new(vec!["hello", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("hello unknown", &vocab);
        assert_eq!(tokens.len(), 2); // hello, <unk>

        let decoded = tokenizer.decode(&tokens, &vocab);
        assert_eq!(decoded, "hello <unk>");
    }
}
