//! # Vocabulary Module
//!
//! Manages bidirectional mapping between tokens and their unique IDs.
//! Provides efficient storage and lookup for token vocabularies with
//! contiguous string buffers for memory efficiency.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Vocabulary management with efficient token-to-ID mapping
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    words_buffer: String,
    word_ranges: Vec<(usize, usize)>, // (start, len)
    unknown_token: Option<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    /// Create a new vocabulary from an iterator of token strings
    pub fn new<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut encode = HashMap::new();
        let mut words_buffer = String::new();
        let mut word_ranges = Vec::new();
        for (i, word_str) in words.into_iter().take(crate::MAX_VOCAB_SIZE).enumerate() {
            let word = word_str.as_ref();
            let start = words_buffer.len();
            words_buffer.push_str(word);
            let len = word.len();
            word_ranges.push((start, len));
            encode.insert(word.to_string(), i);
        }

        Vocab {
            encode,
            words_buffer,
            word_ranges,
            unknown_token: Some("<unk>".to_string()),
        }
    }

    /// Convert a word to its token index
    #[inline]
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// Convert a word to its token index, using unknown token if not found
    pub fn encode_or_unknown(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied().or_else(|| {
            self.unknown_token
                .as_ref()
                .and_then(|unk| self.encode.get(unk).copied())
        })
    }

    /// Check if a word is in the vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.encode.contains_key(word)
    }

    /// Convert a token index back to a word
    #[inline]
    pub fn decode(&self, token_id: usize) -> Option<&str> {
        self.word_ranges
            .get(token_id)
            .map(|&(start, len)| &self.words_buffer[start..start + len])
    }

    /// Get the size of the vocabulary
    pub fn size(&self) -> usize {
        self.word_ranges.len()
    }

    /// Set the unknown token
    pub fn set_unknown_token(&mut self, token: String) {
        self.unknown_token = Some(token);
    }

    /// Get the unknown token
    pub fn unknown_token(&self) -> Option<&str> {
        self.unknown_token.as_deref()
    }

    /// Get a reference to the words vector (for compatibility)
    pub fn words(&self) -> Vec<&str> {
        self.word_ranges
            .iter()
            .map(|&(start, len)| &self.words_buffer[start..start + len])
            .collect()
    }

    /// Encode multiple words at once (returns iterator for zero-copy)
    pub fn encode_batch<'a, I, S>(&'a self, words: I) -> impl Iterator<Item = Option<usize>> + 'a
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
        <I as IntoIterator>::IntoIter: 'a,
    {
        words
            .into_iter()
            .map(move |word| self.encode(word.as_ref()))
    }

    /// Decode multiple token IDs at once (returns iterator for zero-copy)
    pub fn decode_batch<'a, I>(&'a self, token_ids: I) -> impl Iterator<Item = Option<&'a str>> + 'a
    where
        I: IntoIterator<Item = usize>,
        <I as IntoIterator>::IntoIter: 'a,
    {
        token_ids.into_iter().map(move |id| self.decode(id))
    }

    /// Iterate over all words in the vocabulary
    pub fn iter_words(&self) -> impl Iterator<Item = &str> {
        self.word_ranges
            .iter()
            .map(|&(start, len)| &self.words_buffer[start..start + len])
    }

    /// Default words for testing and initialization
    pub fn default_words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "</s>"]
    }

    /// Convenience: build a Vocab from a text stream
    pub fn build_from_stream<I, S>(texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let vocab_words = super::tokenizer::SimpleTokenizer::extract_vocab_from_texts(texts);
        Vocab::new(vocab_words)
    }

    /// Tokenize text using simple word-level tokenization
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let tokenizer = super::tokenizer::SimpleTokenizer::new();
        tokenizer.tokenize(text, self)
    }
}

impl From<Vocab> for String {
    fn from(val: Vocab) -> Self {
        String::from_iter(
            val.word_ranges
                .iter()
                .enumerate()
                .map(|(i, &(start, len))| {
                    let word = &val.words_buffer[start..start + len];
                    format!("({i},{word}),")
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_creation() {
        let vocab = Vocab::new(vec!["hello", "world", "</s>"]);
        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.encode("hello"), Some(0));
        assert_eq!(vocab.encode("world"), Some(1));
        assert_eq!(vocab.encode("</s>"), Some(2));
    }

    #[test]
    fn test_vocab_decode() {
        let vocab = Vocab::new(vec!["hello", "world", "</s>"]);
        assert_eq!(vocab.decode(0), Some("hello"));
        assert_eq!(vocab.decode(1), Some("world"));
        assert_eq!(vocab.decode(2), Some("</s>"));
        assert_eq!(vocab.decode(3), None);
    }

    #[test]
    fn test_unknown_token() {
        let vocab = Vocab::new(vec!["hello", "world", "</s>", "<unk>"]);
        assert_eq!(vocab.encode("unknown"), None);
        assert_eq!(vocab.encode_or_unknown("unknown"), Some(3)); // <unk> token
    }

    #[test]
    fn test_vocab_iteration() {
        let vocab = Vocab::new(vec!["hello", "world", "</s>"]);
        let words: Vec<&str> = vocab.iter_words().collect();
        assert_eq!(words, vec!["hello", "world", "</s>"]);
    }
}
