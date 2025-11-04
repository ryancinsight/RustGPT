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


    /// Tokenize text using simple word-level tokenization
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let tokenizer = super::tokenizer::SimpleTokenizer::new();
        tokenizer.tokenize(text, self)
    }


    /// Build vocabulary from a stream of texts
    /// This is the primary method for creating vocabularies from training data
    pub fn build_from_texts<I, S>(texts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut vocab_set = std::collections::HashSet::new();

        // Always include special tokens
        vocab_set.insert("</s>".to_string());
        vocab_set.insert("<unk>".to_string());

        // Process each text to extract tokens
        for text in texts {
            Self::process_text_tokens(text.as_ref(), &mut vocab_set);
        }

        // Convert to sorted vector for deterministic ordering
        let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
        vocab_words.sort();

        Self::new(vocab_words.iter().map(|s| s.as_str()))
    }

    /// Process a single text to extract tokens and add them to the vocabulary set
    fn process_text_tokens(text: &str, vocab_set: &mut std::collections::HashSet<String>) {
        for word in text.split_whitespace() {
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
