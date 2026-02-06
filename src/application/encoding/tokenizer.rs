//! # Tokenizer Module
//!
//! Core tokenization algorithms for converting raw text into token sequences.
//! This module provides the SimpleTokenizer which handles word-level tokenization
//! with punctuation splitting and unknown token handling.

/// Simple word-level tokenizer that splits on whitespace and punctuation
#[derive(Clone, Debug)]
pub struct SimpleTokenizer;

#[inline]
fn is_ascii_ws(b: u8) -> bool {
    b.is_ascii_whitespace()
}

#[inline]
fn is_ascii_punct(b: u8) -> bool {
    b.is_ascii_punctuation()
}

/// Scan `text` and emit tokens as `&str` slices.
///
/// Token definition:
/// - Skip ASCII whitespace
/// - ASCII punctuation becomes its own 1-byte token
/// - Otherwise emit maximal spans of non-ws, non-punct bytes
/// - If a substring beginning with '<' matches a vocab entry up to the next '>', emit it as a
///   single token (to support special tokens like `</s>`, `<unk>`, `<mask>`).
fn for_each_token_with_vocab<'a>(
    text: &'a str,
    vocab: &super::Vocab,
    mut emit: impl FnMut(&'a str),
) {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    'outer: while i < bytes.len() {
        let b = bytes[i];
        if is_ascii_ws(b) {
            i += 1;
            continue;
        }

        // Special-token fast path: if we see '<', try to match a vocab token like "</s>".
        if b == b'<' {
            let mut j = i + 1;
            // Bound the scan to avoid pathological long searches.
            // Typical special tokens are tiny (e.g. "<unk>", "</s>").
            let max_len = 32usize;
            while j < bytes.len() && (j - i) <= max_len {
                if bytes[j] == b'>' {
                    let candidate = &text[i..=j];
                    if vocab.contains(candidate) {
                        emit(candidate);
                        i = j + 1;
                        // We consumed a token; restart from the new position.
                        continue 'outer;
                    }
                    // Not a known special token: fall through and treat '<' as punctuation.
                    break;
                }
                if is_ascii_ws(bytes[j]) {
                    break;
                }
                j += 1;
            }
        }

        if is_ascii_punct(b) {
            emit(&text[i..i + 1]);
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() {
            let nb = bytes[i];
            if is_ascii_ws(nb) || is_ascii_punct(nb) || nb == b'<' {
                break;
            }
            i += 1;
        }
        emit(&text[start..i]);
    }
}

/// Scan `text` and emit tokens as `&str` slices without consulting a vocabulary.
///
/// This is the canonical segmentation used for vocabulary building.
pub(crate) fn for_each_token<'a>(text: &'a str, mut emit: impl FnMut(&'a str)) {
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if is_ascii_ws(b) {
            i += 1;
            continue;
        }
        if is_ascii_punct(b) {
            emit(&text[i..i + 1]);
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < bytes.len() {
            let nb = bytes[i];
            if is_ascii_ws(nb) || is_ascii_punct(nb) {
                break;
            }
            i += 1;
        }
        emit(&text[start..i]);
    }
}

impl SimpleTokenizer {
    /// Create a new simple tokenizer
    pub fn new() -> Self {
        Self
    }

    /// Tokenize text into token IDs.
    ///
    /// This is allocation-minimal: it does not allocate per-token strings and only grows `Vec` for
    /// the returned token IDs.
    pub fn tokenize(&self, text: &str, vocab: &super::Vocab) -> Vec<usize> {
        let mut tokens = Vec::with_capacity((text.len() / 8).saturating_add(8));
        self.tokenize_into(text, vocab, &mut tokens);
        tokens
    }

    /// In-place variant of [`Self::tokenize`], useful for reusing buffers.
    pub fn tokenize_into(&self, text: &str, vocab: &super::Vocab, out: &mut Vec<usize>) {
        out.clear();
        let unknown_id = vocab.unknown_id();
        for_each_token_with_vocab(text, vocab, |tok| {
            if let Some(id) = vocab.encode(tok) {
                out.push(id);
            } else if let Some(unk) = unknown_id {
                out.push(unk);
            }
        });
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
    use crate::application::encoding::Vocab;

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
    fn test_punctuation_order_within_word() {
        let tokenizer = SimpleTokenizer::new();
        let vocab = Vocab::new(vec!["a", ",", "b", "</s>", "<unk>"]);

        let tokens = tokenizer.tokenize("a,b", &vocab);
        assert_eq!(tokens.len(), 3); // a, ,, b (in order)
        assert_eq!(vocab.decode(tokens[0]), Some("a"));
        assert_eq!(vocab.decode(tokens[1]), Some(","));
        assert_eq!(vocab.decode(tokens[2]), Some("b"));
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
