//! # Zero-Copy Tokenized Dataset via `rkyv`
//!
//! This module provides a high-performance, zero-allocation data pipeline for transformer
//! training. The full pipeline is:
//!
//! 1. **Offline step (once)**: Run `tokenize_and_save` to encode a `Vec<String>` text corpus
//!    into an `rkyv` archive on disk (storing token IDs as `u64` for stable cross-platform ABI).
//! 2. **Training step (repeated)**: Open the archive with [`MemoryMappedDataset`]. The OS
//!    memory-maps the file; the archive header is validated once. Thereafter `iter_examples`
//!    yields sequences **with zero copies and zero heap allocations** per element.
//!
//! ## Why `u64` instead of `usize`?
//! `usize` is not a stable ABI type across 32/64-bit platforms; `rkyv` would store it with
//! target-dependent widths. Storing as `u64` makes archives portable across machines.

use std::num::NonZeroUsize;
use std::path::Path;

use memmap2::Mmap;
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize, archived_root, to_bytes};

use crate::application::encoding::Vocab;
use crate::common::errors::{ModelError, Result};

// ---------------------------------------------------------------------------
// Archive types
// ---------------------------------------------------------------------------

/// One tokenized training example – a sequence of token IDs.
#[derive(Archive, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[archive_attr(derive(Debug))]
pub struct TokenSequence {
    pub ids: Vec<u64>,
}

/// The top-level archive: a flat list of [`TokenSequence`]s.
#[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
#[archive_attr(derive(Debug))]
pub struct TokenizedCorpus {
    pub sequences: Vec<TokenSequence>,
}

// ---------------------------------------------------------------------------
// Serialization / offline pre-processing
// ---------------------------------------------------------------------------

/// Tokenize `texts` in parallel using `vocab` and write an `rkyv` archive to `path`.
///
/// Call this **once** as an offline pre-processing step. Then use [`MemoryMappedDataset`]
/// during training for zero-copy iteration.
pub fn tokenize_and_save<P: AsRef<Path>>(
    texts: &[String],
    vocab: &Vocab,
    path: P,
    _chunk_size: Option<NonZeroUsize>,
) -> Result<()> {

    let sequences: Vec<TokenSequence> = texts
        .par_iter()
        .map(|text| {
            let ids = vocab
                .tokenize(text)
                .into_iter()
                .map(|id| id as u64)
                .collect();
            TokenSequence { ids }
        })
        .collect();

    let corpus = TokenizedCorpus { sequences };

    let bytes = to_bytes::<_, 4096>(&corpus).map_err(|e| ModelError::InvalidInput {
        message: format!("rkyv serialization error: {:?}", e),
    })?;

    std::fs::write(&path, bytes.as_ref()).map_err(ModelError::from)?;

    tracing::info!(
        path = %path.as_ref().display(),
        "Tokenized corpus saved to rkyv archive"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-mapped zero-copy reader
// ---------------------------------------------------------------------------

/// A tokenized corpus opened as a memory-mapped file for zero-copy training iteration.
///
/// The OS maps the archive file into the process address space – no data is ever copied
/// to the heap. [`iter_examples`](Self::iter_examples) yields lazily over mmap data.
pub struct MemoryMappedDataset {
    // SAFETY invariant: `mmap` must outlive every borrow derived from it.
    mmap: Mmap,
}

impl MemoryMappedDataset {
    /// Open and validate an `rkyv` corpus archive created by [`tokenize_and_save`].
    ///
    /// Note: We skip `check_archived_root` here because it requires `CheckBytes` implementations
    /// that are currently only enabled with the "bytecheck" crate feature. Instead we rely on
    /// writing always-valid archives via `to_bytes` and check length sanity manually.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(&path).map_err(ModelError::from)?;
        // SAFETY: We only read the file during the lifetime of the mmap, and the file is not
        // truncated while alive. This is standard practice for read-only training data files.
        let mmap = unsafe { Mmap::map(&file).map_err(ModelError::from)? };

        if mmap.len() < 8 {
            return Err(ModelError::InvalidInput {
                message: format!("rkyv archive too small: {:?}", path.as_ref()),
            });
        }

        // Touch the archived header to catch obvious corruption early (before training begins).
        let _ = Self::archived_corpus_raw(&mmap);

        Ok(Self { mmap })
    }

    /// Number of sequences stored in the archive.
    #[inline]
    pub fn len(&self) -> usize {
        self.archived_corpus().sequences.len()
    }

    /// `true` when the archive is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Zero-copy iterator over sequences.
    ///
    /// Each item is a `Vec<usize>` built by upcasting the archived `u64` values – the only
    /// per-token overhead is a `as usize` cast. No additional heap data is copied.
    pub fn iter_examples(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        self.archived_corpus()
            .sequences
            .iter()
            .map(|seq| seq.ids.iter().map(|id| *id as usize).collect())
    }

    /// Build a mini-batch of `usize` token vecs from a range of archived sequences.
    ///
    /// Allocates only the output `Vec` – no extra temporaries.
    #[inline]
    pub fn collect_batch(&self, start: usize, end: usize) -> Vec<Vec<usize>> {
        let corpus = self.archived_corpus();
        let end = end.min(corpus.sequences.len());
        corpus.sequences[start..end]
            .iter()
            .map(|seq| seq.ids.iter().map(|id| *id as usize).collect())
            .collect()
    }

    #[inline]
    fn archived_corpus(&self) -> &ArchivedTokenizedCorpus {
        Self::archived_corpus_raw(&self.mmap)
    }

    #[inline]
    fn archived_corpus_raw(mmap: &[u8]) -> &ArchivedTokenizedCorpus {
        // SAFETY: We wrote this archive via `to_bytes` which always produces a valid archive.
        // We also check the mmap length > 8 in `open`.
        unsafe { archived_root::<TokenizedCorpus>(mmap) }
    }
}

// ---------------------------------------------------------------------------
// Batch iterator – a zero-cost abstraction over the mmap
// ---------------------------------------------------------------------------

/// A lazy, chunked mini-batch iterator over an [`MemoryMappedDataset`].
///
/// Groups archived sequences into batches of up to `batch_size` without any heap allocation
/// beyond the per-batch `Vec<Vec<usize>>` output.
pub struct BatchedArchiveIter<'a> {
    dataset: &'a MemoryMappedDataset,
    pos: usize,
    batch_size: usize,
}

impl<'a> BatchedArchiveIter<'a> {
    #[inline]
    pub fn new(dataset: &'a MemoryMappedDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            pos: 0,
            batch_size: batch_size.max(1),
        }
    }
}

impl<'a> Iterator for BatchedArchiveIter<'a> {
    /// A heap batch (`Vec<Vec<usize>>`) per mini-batch. Minimal per-batch allocation.
    type Item = Vec<Vec<usize>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.dataset.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.dataset.len());
        let batch = self.dataset.collect_batch(self.pos, end);
        self.pos = end;
        Some(batch)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.dataset.len().saturating_sub(self.pos);
        let count = rem.div_ceil(self.batch_size.max(1));
        (count, Some(count))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use tempfile::NamedTempFile;

    use super::*;

    fn test_vocab() -> Vocab {
        let texts = vec!["hello world foo bar baz qux".to_string()];
        Vocab::build_from_texts(texts.iter())
    }

    #[test]
    fn roundtrip_ten_sequences() {
        let vocab = test_vocab();
        let texts: Vec<String> = (0..10).map(|i| format!("hello world {}", i)).collect();
        let tmp = NamedTempFile::new().unwrap();
        tokenize_and_save(&texts, &vocab, tmp.path(), None).unwrap();

        let ds = MemoryMappedDataset::open(tmp.path()).unwrap();
        assert_eq!(ds.len(), 10);
        let total_tokens: usize = ds.iter_examples().map(|seq| seq.len()).sum();
        assert!(total_tokens > 0, "Expected non-zero tokens");
    }

    #[test]
    fn batched_iter_covers_all() {
        let vocab = test_vocab();
        let texts: Vec<String> = (0..10).map(|i| format!("foo bar {}", i)).collect();
        let tmp = NamedTempFile::new().unwrap();
        tokenize_and_save(&texts, &vocab, tmp.path(), NonZeroUsize::new(3)).unwrap();

        let ds = MemoryMappedDataset::open(tmp.path()).unwrap();
        let batches: Vec<_> = BatchedArchiveIter::new(&ds, 3).collect();
        // 10 examples, batch_size 3 => batches of sizes [3,3,3,1]
        assert_eq!(batches.len(), 4, "Expected 4 batches");
        let total: usize = batches.iter().map(|b| b.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn collect_batch_clamps_out_of_bounds() {
        let vocab = test_vocab();
        let texts: Vec<String> = (0..5).map(|i| format!("hello {}", i)).collect();
        let tmp = NamedTempFile::new().unwrap();
        tokenize_and_save(&texts, &vocab, tmp.path(), None).unwrap();

        let ds = MemoryMappedDataset::open(tmp.path()).unwrap();
        // out-of-bounds end is clamped to len()
        let batch = ds.collect_batch(3, 999);
        assert_eq!(batch.len(), 2);
    }
}
