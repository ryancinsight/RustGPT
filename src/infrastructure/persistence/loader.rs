//! Generic dataset loader trait.
//!
//! Defines a common interface for loading different types of datasets
//! (audio, image, text, etc.) from storage.

use crate::common::errors::Result;
use std::path::Path;

/// A generic trait for loading datasets from a source path.
///
/// This trait abstracts the loading mechanism, allowing different
/// implementations for different dataset types while providing a
/// consistent API.
pub trait DatasetLoader {
    /// The type of item produced by the loader.
    ///
    /// This is typically a vector of examples (e.g., `Vec<SpeechExample>`)
    /// or a tuple of data and labels.
    type Item;

    /// Load the dataset from the given source path.
    ///
    /// # Arguments
    ///
    /// * `source` - The path to the dataset source (directory or file).
    ///
    /// # Returns
    ///
    /// The loaded dataset items or an error if loading fails.
    fn load<P: AsRef<Path>>(&self, source: P) -> Result<Self::Item>;
}
