//! # Encoding Module
//!
//! This module provides text encoding functionality for the RustGPT language model,
//! organized with clear separation of concerns and hierarchical structure.
//!
//! ## Architecture
//!
//! ```text
//! encoding/
//! ├── mod.rs              # Main module exports and coordination
//! ├── tokenizer.rs        # Core tokenization algorithms (SimpleTokenizer)
//! ├── vocabulary.rs       # Vocabulary management and token-ID mapping
//! └── word_level.rs       # Word-level tokenization utilities
//! ```
//!
//! ## Key Components
//!
//! - **Tokenizer**: Converts raw text into token sequences
//! - **Vocabulary**: Manages bidirectional mapping between tokens and IDs
//! - **Word-Level Utils**: Helper functions for word-level processing
//!
//! ## Design Principles
//!
//! - **Separation of Concerns**: Each submodule handles one aspect
//! - **Hierarchical Organization**: Clear dependency structure
//! - **Zero-Copy Operations**: Efficient string handling where possible
//! - **Extensible Design**: Easy to add new tokenization methods

pub mod tokenizer;
pub mod vocabulary;
pub mod word_level;

// Re-export main types for convenience
pub use tokenizer::SimpleTokenizer;
pub use vocabulary::Vocab;

// Re-export utility functions that might be commonly used
pub use word_level::{tokenize_word_level, extract_vocab_from_texts};
