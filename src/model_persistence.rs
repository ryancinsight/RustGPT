use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;

use crate::errors::{ModelError, Result};
use crate::llm::LLM;

/// Current model format version
/// Increment this when making breaking changes to the serialization format
const MODEL_VERSION: u32 = 1;

/// Versioned model container with integrity checking
#[derive(Serialize, Deserialize, Clone)]
pub struct VersionedModel {
    /// Format version for backward compatibility
    pub version: u32,
    /// SHA256 checksum of the serialized model data (hex string)
    pub checksum: String,
    /// Serialized model data (JSON or binary)
    pub data: Vec<u8>,
    /// Metadata for debugging and tracking
    pub metadata: ModelMetadata,
}

/// Metadata about the model
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModelMetadata {
    /// Timestamp when model was saved (ISO 8601 format)
    pub saved_at: String,
    /// Model architecture type (e.g., "Transformer" or "TRM")
    pub architecture: String,
    /// Number of parameters
    pub num_parameters: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Optional description
    pub description: Option<String>,
}

impl VersionedModel {
    /// Create a new versioned model from an LLM instance
    ///
    /// # Arguments
    /// * `llm` - The LLM instance to serialize
    /// * `format` - Serialization format ("json" or "binary")
    /// * `description` - Optional description for metadata
    ///
    /// # Errors
    /// Returns `ModelError::Serialization` if serialization fails
    pub fn from_llm(llm: &LLM, format: &str, description: Option<String>) -> Result<Self> {
        // Serialize the model
        let data = match format {
            "json" => serde_json::to_vec_pretty(llm).map_err(|e| ModelError::Serialization {
                source: Box::new(e),
            })?,
            "binary" => {
                let config = bincode::config::standard();
                bincode::serde::encode_to_vec(llm, config).map_err(|e| {
                    ModelError::Serialization {
                        source: Box::new(e),
                    }
                })?
            }
            _ => {
                return Err(ModelError::InvalidInput {
                    message: format!("Unsupported format: {}", format),
                });
            }
        };

        // Compute checksum
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let checksum = format!("{:x}", hasher.finalize());

        // Extract metadata from LLM
        let metadata = ModelMetadata {
            saved_at: chrono::Utc::now().to_rfc3339(),
            architecture: llm.get_architecture_name(),
            num_parameters: llm.count_parameters(),
            embedding_dim: llm.get_embedding_dim(),
            num_layers: llm.network.len(),
            description,
        };

        Ok(VersionedModel {
            version: MODEL_VERSION,
            checksum,
            data,
            metadata,
        })
    }

    /// Validate the checksum of the model data
    ///
    /// # Errors
    /// Returns `ModelError::Serialization` if checksum validation fails
    pub fn validate_checksum(&self) -> Result<()> {
        let mut hasher = Sha256::new();
        hasher.update(&self.data);
        let computed_checksum = format!("{:x}", hasher.finalize());

        if computed_checksum != self.checksum {
            return Err(ModelError::Serialization {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Checksum mismatch: expected {}, got {}",
                        self.checksum, computed_checksum
                    ),
                )),
            });
        }

        Ok(())
    }

    /// Validate the model version
    ///
    /// # Errors
    /// Returns `ModelError::Serialization` if version is incompatible
    pub fn validate_version(&self) -> Result<()> {
        if self.version > MODEL_VERSION {
            return Err(ModelError::Serialization {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Model version {} is newer than supported version {}. Please upgrade the library.",
                        self.version, MODEL_VERSION
                    ),
                )),
            });
        }

        // Future: Handle backward compatibility for older versions
        if self.version < MODEL_VERSION {
            tracing::warn!(
                "Loading model with older version {} (current: {}). Some features may not be available.",
                self.version,
                MODEL_VERSION
            );
        }

        Ok(())
    }

    /// Deserialize the model data into an LLM instance
    ///
    /// # Arguments
    /// * `format` - Serialization format ("json" or "binary")
    ///
    /// # Errors
    /// Returns `ModelError::Serialization` if deserialization fails
    pub fn to_llm(&self, format: &str) -> Result<LLM> {
        // Validate before deserializing
        self.validate_version()?;
        self.validate_checksum()?;

        // Deserialize
        let llm = match format {
            "json" => {
                serde_json::from_slice(&self.data).map_err(|e| ModelError::Serialization {
                    source: Box::new(e),
                })?
            }
            "binary" => {
                let config = bincode::config::standard();
                let (llm, _): (LLM, usize) = bincode::serde::decode_from_slice(&self.data, config)
                    .map_err(|e| ModelError::Serialization {
                        source: Box::new(e),
                    })?;
                llm
            }
            _ => {
                return Err(ModelError::InvalidInput {
                    message: format!("Unsupported format: {}", format),
                });
            }
        };

        Ok(llm)
    }

    /// Save the versioned model to a file
    ///
    /// # Errors
    /// Returns `ModelError::Serialization` if file write fails
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| ModelError::Serialization {
            source: Box::new(e),
        })?;
        fs::write(path, json).map_err(ModelError::from)?;
        Ok(())
    }

    /// Load a versioned model from a file
    ///
    /// # Errors
    /// Returns `ModelError` if file read or deserialization fails
    pub fn load_from_file(path: &str) -> Result<Self> {
        let data = fs::read_to_string(path).map_err(ModelError::from)?;
        let versioned_model: VersionedModel =
            serde_json::from_str(&data).map_err(|e| ModelError::Serialization {
                source: Box::new(e),
            })?;
        Ok(versioned_model)
    }
}

/// Extension methods for LLM to support versioned serialization
impl LLM {
    /// Save model with versioning and integrity checking
    ///
    /// # Arguments
    /// * `path` - File path (extension determines format: .json or .bin)
    /// * `description` - Optional description for metadata
    ///
    /// # Errors
    /// Returns `ModelError` if serialization or file write fails
    pub fn save_versioned(&self, path: &str, description: Option<String>) -> Result<()> {
        let format = if path.ends_with(".json") {
            "json"
        } else {
            "binary"
        };

        let versioned = VersionedModel::from_llm(self, format, description)?;
        versioned.save_to_file(path)?;

        tracing::info!(
            path = path,
            version = MODEL_VERSION,
            checksum = &versioned.checksum[..16], // Log first 16 chars
            architecture = &versioned.metadata.architecture,
            "Model saved with versioning and integrity check"
        );

        Ok(())
    }

    /// Load model with versioning and integrity checking
    ///
    /// # Errors
    /// Returns `ModelError` if file read, validation, or deserialization fails
    pub fn load_versioned(path: &str) -> Result<Self> {
        let versioned = VersionedModel::load_from_file(path)?;

        tracing::info!(
            path = path,
            version = versioned.version,
            checksum = &versioned.checksum[..16], // Log first 16 chars
            architecture = &versioned.metadata.architecture,
            "Loading model with version {} (saved at {})",
            versioned.version,
            versioned.metadata.saved_at
        );

        let format = if path.ends_with(".json") {
            "json"
        } else {
            "binary"
        };

        versioned.to_llm(format)
    }

    /// Get the architecture name for metadata
    fn get_architecture_name(&self) -> String {
        // Detect architecture from network layers
        let has_self_attention = self
            .network
            .iter()
            .any(|l| matches!(l, crate::llm::LayerEnum::SelfAttention(_)));
        let has_trm = self
            .network
            .iter()
            .any(|l| matches!(l, crate::llm::LayerEnum::TRMBlock(_)));

        if has_trm {
            "TRM".to_string()
        } else if has_self_attention {
            "Transformer".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    /// Get the embedding dimension
    fn get_embedding_dim(&self) -> usize {
        // Extract from first embeddings layer
        for layer in &self.network {
            if let crate::llm::LayerEnum::Embeddings(emb) = layer {
                // Get embedding dimension from token_embeddings shape
                return emb.token_embeddings.shape()[1];
            }
        }
        0
    }

    /// Count total parameters in the model by traversing all layers
    fn count_parameters(&self) -> usize {
        // Delegate to LLM's total_parameters() which properly sums parameters across all layers
        self.total_parameters()
    }
}
