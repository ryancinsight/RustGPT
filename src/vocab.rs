use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

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
    pub fn new<I, S>(words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut encode = HashMap::new();
        let mut words_buffer = String::new();
        let mut word_ranges = Vec::new();
        let mut total_len = 0;

        for (i, word_str) in words.into_iter().take(crate::MAX_VOCAB_SIZE).enumerate() {
            let word = word_str.as_ref();
            tracing::debug!(word = word, index = i, "Adding word to encoding");
            let start = words_buffer.len();
            words_buffer.push_str(word);
            let len = word.len();
            word_ranges.push((start, len));
            encode.insert(word.to_string(), i);
            total_len += len;
        }

        let vocab_size = word_ranges.len();

        // Pre-allocate capacity for words_buffer (though already built)
        words_buffer.reserve(total_len.saturating_sub(words_buffer.len()));

        tracing::info!(vocab_size = vocab_size, "Vocabulary initialized");
        Vocab {
            encode,
            words_buffer,
            word_ranges,
            unknown_token: None,
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
            self.unknown_token.as_ref().and_then(|unk| self.encode.get(unk).copied())
        })
    }

    /// Check if a word is in the vocabulary
    pub fn contains(&self, word: &str) -> bool {
        self.encode.contains_key(word)
    }

    /// Convert a token index back to a word
    #[inline]
    pub fn decode(&self, token_id: usize) -> Option<&str> {
        self.word_ranges.get(token_id).map(|&(start, len)| {
            &self.words_buffer[start..start + len]
        })
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
        self.word_ranges.iter().map(|&(start, len)| {
            &self.words_buffer[start..start + len]
        }).collect()
    }

    /// Encode multiple words at once
    pub fn encode_batch<I, S>(&self, words: I) -> Vec<Option<usize>>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        words.into_iter().map(|word| self.encode(word.as_ref())).collect()
    }

    /// Decode multiple token IDs at once
    pub fn decode_batch<I>(&self, token_ids: I) -> Vec<Option<&str>>
    where
        I: IntoIterator<Item = usize>,
    {
        token_ids.into_iter().map(|id| self.decode(id)).collect()
    }

    /// Iterate over all words in the vocabulary
    pub fn iter_words(&self) -> impl Iterator<Item = &str> {
        self.word_ranges.iter().map(|&(start, len)| {
            &self.words_buffer[start..start + len]
        })
    }

    pub fn default_words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "</s>"]
    }

    /// Process text data to extract vocabulary words and add them to the vocabulary set
    pub fn process_text_for_vocab(texts: &[String], vocab_set: &mut HashSet<String>) {
        // Add end of sequence token
        vocab_set.insert("</s>".to_string());

        // Process all training examples for vocabulary using iterators
        texts
            .iter()
            .flat_map(|text| text.split_whitespace())
            .flat_map(|word| {
                word.split(|c: char| c.is_ascii_punctuation())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .chain(
                        word.chars()
                            .filter(|c| c.is_ascii_punctuation())
                            .map(|c| c.to_string())
                    )
            })
            .for_each(|token| {
                vocab_set.insert(token);
            });
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
