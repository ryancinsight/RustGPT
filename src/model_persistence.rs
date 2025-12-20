use std::fs;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    errors::{ModelError, Result},
    llm::LLM,
};

/// Current model format version
/// Increment this when making breaking changes to the serialization format
const MODEL_VERSION: u32 = 2;

fn default_data_format() -> Option<String> {
    // New saves always set this explicitly.
    None
}

/// Versioned model container with integrity checking
#[derive(Serialize, Deserialize, Clone)]
pub struct VersionedModel {
    /// Format version for backward compatibility
    pub version: u32,
    /// SHA256 checksum of the serialized model data (hex string)
    pub checksum: String,
    /// Payload codec used for `data` (e.g., "json", "msgpack", "bincode2")
    #[serde(default = "default_data_format")]
    pub data_format: Option<String>,
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
    /// Model architecture type (e.g., "Transformer")
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
        let (data_format, data) = match format {
            "json" => (
                Some("json".to_string()),
                serde_json::to_vec_pretty(llm).map_err(|e| ModelError::Serialization {
                    source: Box::new(e),
                })?,
            ),
            "binary" => (
                Some("msgpack".to_string()),
                rmp_serde::to_vec_named(llm).map_err(|e| ModelError::Serialization {
                    source: Box::new(e),
                })?,
            ),
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
            data_format,
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

        // Prefer the stored payload codec if present.
        let effective_format = self.data_format.as_deref().unwrap_or(format);

        // Deserialize
        let llm = match effective_format {
            "json" => {
                serde_json::from_slice(&self.data).map_err(|e| ModelError::Serialization {
                    source: Box::new(e),
                })?
            }
            "msgpack" | "binary" => {
                rmp_serde::from_slice(&self.data).map_err(|e| ModelError::Serialization {
                    source: Box::new(e),
                })?
            }
            // Legacy payload codec for MODEL_VERSION=1 files.
            "bincode2" => {
                let config = bincode::config::standard();
                let (llm, _): (LLM, usize) = bincode::serde::decode_from_slice(&self.data, config)
                    .map_err(|e| ModelError::Serialization {
                        source: Box::new(e),
                    })?;
                llm
            }
            _ => {
                return Err(ModelError::InvalidInput {
                    message: format!("Unsupported format: {}", effective_format),
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
            data_format = versioned.data_format.as_deref().unwrap_or(format),
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

        let requested_format = if path.ends_with(".json") {
            "json"
        } else {
            "binary"
        };

        // Back-compat: older v1 files used bincode v2 for the payload but didn't store a codec tag.
        if versioned.version == 1 && versioned.data_format.is_none() && requested_format == "binary"
        {
            let mut v = versioned;
            v.data_format = Some("bincode2".to_string());
            return v.to_llm(requested_format);
        }

        versioned.to_llm(requested_format)
    }

    /// Get the architecture name for metadata
    fn get_architecture_name(&self) -> String {
        // In PolyAttention-only refactor, any presence of PolyAttention implies Transformer
        let has_poly_attention = self
            .network
            .iter()
            .any(|l| matches!(l, crate::LayerEnum::PolyAttention(_)));

        if has_poly_attention {
            "Transformer".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    /// Get the embedding dimension
    fn get_embedding_dim(&self) -> usize {
        // Extract from first embeddings layer
        for layer in &self.network {
            if let crate::LayerEnum::TokenEmbeddings(emb) = layer {
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
