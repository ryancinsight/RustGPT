//! Model storage abstraction for persistence operations
//!
//! Defines a trait for model storage backends that can be used
//! by the web UI and other components to load and save models.

use std::path::Path;

use crate::domain::models::llm::LLM;
use crate::common::errors::Result;

/// Trait for model storage backends
///
/// This abstraction allows different storage implementations
/// (file system, cloud storage, database, etc.) to be used
/// interchangeably.
pub trait ModelStorage: Send + Sync {
    /// Save a model to storage
    ///
    /// # Arguments
    ///
    /// * `model` - The model to save
    /// * `name` - The name/identifier for the model
    ///
    /// # Returns
    ///
    /// The path or identifier where the model was saved
    fn save(&self, model: &LLM, name: &str) -> Result<String>;

    /// Load a model from storage
    ///
    /// # Arguments
    ///
    /// * `name` - The name/identifier of the model to load
    ///
    /// # Returns
    ///
    /// The loaded model
    fn load(&self, name: &str) -> Result<LLM>;

    /// List all available models
    ///
    /// # Returns
    ///
    /// A list of model names/identifiers
    fn list_models(&self) -> Result<Vec<String>>;

    /// Check if a model exists
    ///
    /// # Arguments
    ///
    /// * `name` - The name/identifier to check
    ///
    /// # Returns
    ///
    /// True if the model exists, false otherwise
    fn exists(&self, name: &str) -> bool;

    /// Delete a model from storage
    ///
    /// # Arguments
    ///
    /// * `name` - The name/identifier of the model to delete
    ///
    /// # Returns
    ///
    /// True if the model was deleted, false if it didn't exist
    fn delete(&self, name: &str) -> Result<bool>;

    /// Get model metadata
    ///
    /// # Arguments
    ///
    /// * `name` - The name/identifier of the model
    ///
    /// # Returns
    ///
    /// Model metadata including size, creation time, etc.
    fn get_metadata(&self, name: &str) -> Result<ModelMetadata>;
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model name/identifier
    pub name: String,
    /// Model file path or storage key
    pub path: String,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Creation timestamp
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Last modified timestamp
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Model version (if versioned)
    pub version: Option<String>,
}

/// File-based model storage implementation
pub struct FileModelStorage {
    /// Base directory for model files
    base_dir: std::path::PathBuf,
}

impl FileModelStorage {
    /// Create a new file-based model storage
    ///
    /// # Arguments
    ///
    /// * `base_dir` - The base directory where models are stored
    pub fn new(base_dir: impl AsRef<Path>) -> Self {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir).ok();
        Self { base_dir }
    }

    /// Get the full path for a model file
    fn model_path(&self, name: &str) -> std::path::PathBuf {
        self.base_dir.join(format!("{}.bin", name))
    }
}

impl FileModelStorage {
    /// Get the path for a model file with specific extension
    fn model_path_with_ext(&self, name: &str, ext: &str) -> std::path::PathBuf {
        self.base_dir.join(format!("{}.{}", name, ext))
    }
}

impl ModelStorage for FileModelStorage {
    fn save(&self, model: &LLM, name: &str) -> Result<String> {
        let path = self.model_path(name);
        let path_str = path.to_string_lossy();
        model.save_binary(&path_str)?;
        Ok(path_str.to_string())
    }

    fn load(&self, name: &str) -> Result<LLM> {
        // Try binary format first (default)
        let bin_path = self.model_path(name);
        if bin_path.exists() {
            let path_str = bin_path.to_string_lossy();
            match LLM::load_binary(&path_str) {
                Ok(model) => return Ok(model),
                Err(e) => {
                    // If binary fails, check if it's actually a JSON file
                    let json_path = self.model_path_with_ext(name, "json");
                    if json_path.exists() {
                        return LLM::load_json(&json_path.to_string_lossy());
                    }
                    return Err(e);
                }
            }
        }
        
        // Try JSON format if binary doesn't exist
        let json_path = self.model_path_with_ext(name, "json");
        if json_path.exists() {
            return LLM::load_json(&json_path.to_string_lossy());
        }
        
        // Try auto-detect format from any extension
        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if let Some(file_stem) = entry.path().file_stem().and_then(|s| s.to_str()) {
                    if file_stem == name {
                        let path = entry.path();
                        let path_str = path.to_string_lossy();
                        if path.extension().map(|e| e == "json").unwrap_or(false) {
                            return LLM::load_json(&path_str);
                        } else {
                            return LLM::load_binary(&path_str);
                        }
                    }
                }
            }
        }
        
        Err(crate::common::errors::ModelError::Generic(format!(
            "Model not found: {}. Looked for {}.bin and {}.json",
            name, name, name
        )))
    }

    fn list_models(&self) -> Result<Vec<String>> {
        let mut models = Vec::new();
        let mut seen = std::collections::HashSet::new();

        if let Ok(entries) = std::fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry
                    .path()
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
                {
                    if seen.insert(name.clone()) {
                        models.push(name);
                    }
                }
            }
        }

        Ok(models)
    }

    fn exists(&self, name: &str) -> bool {
        self.model_path(name).exists() || 
        self.model_path_with_ext(name, "json").exists()
    }

    fn delete(&self, name: &str) -> Result<bool> {
        let mut deleted = false;
        
        // Try to delete binary version
        let bin_path = self.model_path(name);
        if bin_path.exists() {
            std::fs::remove_file(&bin_path)?;
            deleted = true;
        }
        
        // Try to delete JSON version
        let json_path = self.model_path_with_ext(name, "json");
        if json_path.exists() {
            std::fs::remove_file(&json_path)?;
            deleted = true;
        }
        
        Ok(deleted)
    }

    fn get_metadata(&self, name: &str) -> Result<ModelMetadata> {
        // Try binary first, then JSON
        let path = if self.model_path(name).exists() {
            self.model_path(name)
        } else {
            self.model_path_with_ext(name, "json")
        };

        if !path.exists() {
            return Err(crate::common::errors::ModelError::Generic(format!(
                "Model not found: {}",
                name
            )));
        }

        let metadata = std::fs::metadata(&path)?;
        let size_bytes = metadata.len();

        let created_at = metadata
            .created()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| chrono::DateTime::UNIX_EPOCH + chrono::Duration::from_std(d).unwrap_or_default());

        let modified_at = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| chrono::DateTime::UNIX_EPOCH + chrono::Duration::from_std(d).unwrap_or_default());

        Ok(ModelMetadata {
            name: name.to_string(),
            path: path.to_string_lossy().to_string(),
            size_bytes,
            created_at,
            modified_at,
            version: None,
        })
    }
}
