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
//! └── vocabulary.rs       # Vocabulary management and token-ID mapping
//! ```
//!
//! ## Key Components
//!
//! - **Tokenizer**: Converts raw text into token sequences
//! - **Vocabulary**: Manages bidirectional mapping between tokens and IDs
//!
//! ## Design Principles
//!
//! - **Separation of Concerns**: Each submodule handles one aspect
//! - **Hierarchical Organization**: Clear dependency structure
//! - **Zero-Copy Operations**: Efficient string handling where possible
//! - **Extensible Design**: Easy to add new tokenization methods

pub mod tokenizer;
pub mod vocabulary;

// Re-export main types for convenience
pub use tokenizer::SimpleTokenizer;
pub use vocabulary::Vocab;

