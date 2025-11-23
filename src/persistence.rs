use crate::{
    errors::Result,
    llm::LLM,
};

/// Model persistence functionality (save/load operations)
pub struct ModelPersistence;

impl ModelPersistence {
    /// Save model in JSON format
    pub fn save_json(llm: &LLM, path: &str) -> Result<()> {
        llm.save_json(path)
    }

    /// Load model from JSON format
    pub fn load_json(path: &str) -> Result<LLM> {
        LLM::load_json(path)
    }

    /// Save model in binary format
    pub fn save_binary(llm: &LLM, path: &str) -> Result<()> {
        llm.save_binary(path)
    }

    /// Load model from binary format
    pub fn load_binary(path: &str) -> Result<LLM> {
        LLM::load_binary(path)
    }

    /// Save model with automatic format detection
    pub fn save(llm: &LLM, path: &str) -> Result<()> {
        llm.save(path)
    }

    /// Load model with automatic format detection
    pub fn load(path: &str) -> Result<LLM> {
        LLM::load(path)
    }

    /// Save model with versioning metadata
    pub fn save_versioned(llm: &LLM, path: &str, description: Option<String>) -> Result<()> {
        llm.save_versioned(path, description)
    }

    /// Load model with versioning support
    pub fn load_versioned(path: &str) -> Result<LLM> {
        LLM::load_versioned(path)
    }
}

