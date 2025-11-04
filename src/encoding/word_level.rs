//! # Word-Level Processing Utilities
//!
//! Helper functions for word-level tokenization and vocabulary extraction.
//! This module provides utilities that are shared between tokenization
//! and vocabulary building processes.

use std::collections::HashSet;

/// Tokenize text at the word level, splitting on whitespace and punctuation
pub fn tokenize_word_level(text: &str) -> Vec<String> {
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

        tokens.extend(word_tokens);
    }

    tokens
}

/// Extract vocabulary from a stream of texts using word-level tokenization
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

/// Count word frequencies in a stream of texts
pub fn count_word_frequencies<I, S>(texts: I) -> std::collections::HashMap<String, usize>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut frequencies = std::collections::HashMap::new();

    for text in texts.into_iter() {
        let tokens = tokenize_word_level(text.as_ref());
        for token in tokens {
            *frequencies.entry(token).or_insert(0) += 1;
        }
    }

    frequencies
}

/// Filter vocabulary by minimum frequency
pub fn filter_vocab_by_frequency(
    vocab: &[String],
    min_frequency: usize,
    texts: &[String],
) -> Vec<String> {
    let frequencies = count_word_frequencies(texts.iter().map(|s| s.as_str()));

    vocab
        .iter()
        .filter(|word| {
            // Always keep special tokens
            *word == "</s>" || *word == "<unk>" ||
            *frequencies.get(*word).unwrap_or(&0) >= min_frequency
        })
        .cloned()
        .collect()
}

/// Process text data to extract vocabulary tokens and add them to a vocabulary set
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
                        .map(|c| c.to_string()),
                )
        })
        .for_each(|token| {
            vocab_set.insert(token);
        });
}

/// Process a streaming iterator of texts to extract vocabulary tokens
pub fn process_stream_for_vocab<I, S>(texts: I, vocab_set: &mut HashSet<String>)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    // Add end of sequence token
    vocab_set.insert("</s>".to_string());

    for text in texts.into_iter() {
        let s = text.as_ref();
        for word in s.split_whitespace() {
            // Build owned tokens from the word to avoid borrowing across iterator boundaries
            let tokens = word
                .split(|c: char| c.is_ascii_punctuation())
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .chain(
                    word.chars()
                        .filter(|c| c.is_ascii_punctuation())
                        .map(|c| c.to_string()),
                );
            for token in tokens {
                vocab_set.insert(token);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_word_level() {
        let text = "Hello, world! How are you?";
        let tokens = tokenize_word_level(text);
        assert_eq!(tokens, vec!["Hello", ",", "world", "!", "How", "are", "you", "?"]);
    }

    #[test]
    fn test_extract_vocab_from_texts() {
        let texts = vec!["hello world", "hello rust", "world peace"];
        let vocab = extract_vocab_from_texts(texts);
        assert!(vocab.contains(&"hello".to_string()));
        assert!(vocab.contains(&"world".to_string()));
        assert!(vocab.contains(&"rust".to_string()));
        assert!(vocab.contains(&"peace".to_string()));
        assert!(vocab.contains(&"</s>".to_string()));
        assert!(vocab.contains(&"<unk>".to_string()));
    }

    #[test]
    fn test_count_word_frequencies() {
        let texts = vec!["hello world", "hello rust", "world world"];
        let frequencies = count_word_frequencies(texts);
        assert_eq!(*frequencies.get("hello").unwrap(), 2);
        assert_eq!(*frequencies.get("world").unwrap(), 3);
        assert_eq!(*frequencies.get("rust").unwrap(), 1);
    }

    #[test]
    fn test_process_text_for_vocab() {
        let texts = vec!["hello world".to_string(), "hello rust".to_string()];
        let mut vocab_set = HashSet::new();
        process_text_for_vocab(&texts, &mut vocab_set);

        assert!(vocab_set.contains("hello"));
        assert!(vocab_set.contains("world"));
        assert!(vocab_set.contains("rust"));
        assert!(vocab_set.contains("</s>"));
    }
}
