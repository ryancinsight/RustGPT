//! # Decoding Module
//!
//! This module provides text decoding functionality for the RustGPT language model,
//! organized with clear separation of concerns and hierarchical structure.
//!
//! ## Architecture
//!
//! ```text
//! decoding/
//! ├── mod.rs              # Main module exports and coordination
//! └── greedy.rs           # Greedy decoding implementation
//! ```
//!
//! ## Key Components
//!
//! - **GreedyDecoder**: Simple greedy token selection
//!
//! ## Design Principles
//!
//! - **Separation of Concerns**: Each submodule handles one decoding strategy
//! - **Hierarchical Organization**: Clear dependency structure
//! - **Performance-Oriented**: Zero-cost abstractions where possible
//! - **Extensible Design**: Easy to add new decoding methods

pub mod greedy;

// Re-export main types for convenience
pub use greedy::GreedyDecoder;
