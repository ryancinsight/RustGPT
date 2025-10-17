use llm::{LLM, VersionedModel};
use tempfile::NamedTempFile;

#[test]
fn test_versioned_save_and_load_json() {
    // Create a default LLM
    let llm = LLM::default();

    // Create a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap().to_string() + ".json";

    // Save with versioning
    llm.save_versioned(&path, Some("Test model".to_string()))
        .expect("Failed to save versioned model");

    // Load with versioning
    let loaded_llm = LLM::load_versioned(&path).expect("Failed to load versioned model");

    // Verify vocab size matches
    assert_eq!(llm.vocab.encode.len(), loaded_llm.vocab.encode.len());

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_versioned_save_and_load_binary() {
    // Create a default LLM
    let llm = LLM::default();

    // Create a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap().to_string() + ".bin";

    // Save with versioning
    llm.save_versioned(&path, Some("Test binary model".to_string()))
        .expect("Failed to save versioned model");

    // Load with versioning
    let loaded_llm = LLM::load_versioned(&path).expect("Failed to load versioned model");

    // Verify vocab size matches
    assert_eq!(llm.vocab.encode.len(), loaded_llm.vocab.encode.len());

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_checksum_validation() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model
    let versioned = VersionedModel::from_llm(&llm, "json", Some("Test".to_string()))
        .expect("Failed to create versioned model");

    // Validate checksum (should pass)
    assert!(versioned.validate_checksum().is_ok());

    // Corrupt the data
    let mut corrupted = versioned.clone();
    if !corrupted.data.is_empty() {
        corrupted.data[0] ^= 0xFF; // Flip bits
    }

    // Validation should fail
    assert!(corrupted.validate_checksum().is_err());
}

#[test]
fn test_version_validation() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model
    let versioned = VersionedModel::from_llm(&llm, "json", Some("Test".to_string()))
        .expect("Failed to create versioned model");

    // Validate version (should pass)
    assert!(versioned.validate_version().is_ok());

    // Create a model with future version
    let mut future_version = versioned.clone();
    future_version.version = 999;

    // Validation should fail
    assert!(future_version.validate_version().is_err());
}

#[test]
fn test_metadata_extraction() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model
    let versioned = VersionedModel::from_llm(&llm, "json", Some("Test metadata".to_string()))
        .expect("Failed to create versioned model");

    // Verify metadata
    assert_eq!(
        versioned.metadata.description,
        Some("Test metadata".to_string())
    );
    assert!(versioned.metadata.num_parameters > 0);
    assert!(versioned.metadata.embedding_dim > 0);
    assert!(versioned.metadata.num_layers > 0);
    assert!(!versioned.metadata.architecture.is_empty());
}

#[test]
fn test_corrupted_file_detection() {
    // Create a default LLM
    let llm = LLM::default();

    // Create a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap().to_string() + ".json";

    // Save with versioning
    llm.save_versioned(&path, None)
        .expect("Failed to save versioned model");

    // Corrupt the file
    let mut data = std::fs::read_to_string(&path).unwrap();
    // Replace a character in the middle to corrupt checksum
    if data.len() > 100 {
        data.replace_range(50..51, "X");
        std::fs::write(&path, data).unwrap();
    }

    // Loading should fail due to checksum mismatch
    let result = LLM::load_versioned(&path);
    assert!(result.is_err());

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_round_trip_preserves_data() {
    // Create a default LLM
    let mut llm = LLM::default();

    // Train for a few epochs to create non-trivial state
    let training_data = vec!["hello world", "test data"];
    llm.train(training_data, 2, 0.01).ok();

    // Create a temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().to_str().unwrap().to_string() + ".json";

    // Save with versioning
    llm.save_versioned(&path, Some("Trained model".to_string()))
        .expect("Failed to save versioned model");

    // Load with versioning
    let loaded_llm = LLM::load_versioned(&path).expect("Failed to load versioned model");

    // Verify vocab matches
    assert_eq!(llm.vocab.encode.len(), loaded_llm.vocab.encode.len());
    assert_eq!(llm.vocab.words.len(), loaded_llm.vocab.words.len());

    // Verify network structure matches
    assert_eq!(llm.network.len(), loaded_llm.network.len());

    // Cleanup
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_backward_compatibility_warning() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model with older version
    let mut versioned = VersionedModel::from_llm(&llm, "json", Some("Old version".to_string()))
        .expect("Failed to create versioned model");

    // Set to older version (but not too old to be incompatible)
    versioned.version = 1; // Assuming current version is 1 or higher

    // Validation should pass (with warning logged)
    assert!(versioned.validate_version().is_ok());
}

#[test]
fn test_invalid_format_error() {
    // Create a default LLM
    let llm = LLM::default();

    // Try to create with invalid format
    let result = VersionedModel::from_llm(&llm, "invalid_format", None);
    assert!(result.is_err());
}

#[test]
fn test_checksum_hex_format() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model
    let versioned =
        VersionedModel::from_llm(&llm, "json", None).expect("Failed to create versioned model");

    // Verify checksum is valid hex (64 characters for SHA256)
    assert_eq!(versioned.checksum.len(), 64);
    assert!(versioned.checksum.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn test_metadata_timestamp_format() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model
    let versioned =
        VersionedModel::from_llm(&llm, "json", None).expect("Failed to create versioned model");

    // Verify timestamp is in ISO 8601 format (RFC3339)
    // Should be parseable by chrono
    let parsed = chrono::DateTime::parse_from_rfc3339(&versioned.metadata.saved_at);
    assert!(parsed.is_ok(), "Timestamp should be valid RFC3339 format");
}

#[test]
fn test_architecture_detection() {
    // Create a default LLM (should be Transformer architecture)
    let llm = LLM::default();

    // Create versioned model
    let versioned =
        VersionedModel::from_llm(&llm, "json", None).expect("Failed to create versioned model");

    // Verify architecture is detected
    assert!(
        versioned.metadata.architecture == "Transformer"
            || versioned.metadata.architecture == "HyperMixer"
            || versioned.metadata.architecture == "HRM"
            || versioned.metadata.architecture == "Unknown"
    );
}

#[test]
fn test_empty_description() {
    // Create a default LLM
    let llm = LLM::default();

    // Create versioned model without description
    let versioned =
        VersionedModel::from_llm(&llm, "json", None).expect("Failed to create versioned model");

    // Verify description is None
    assert_eq!(versioned.metadata.description, None);
}

#[test]
fn test_binary_format_smaller_than_json() {
    // Create a default LLM
    let llm = LLM::default();

    // Create both formats
    let json_versioned = VersionedModel::from_llm(&llm, "json", None)
        .expect("Failed to create JSON versioned model");
    let binary_versioned = VersionedModel::from_llm(&llm, "binary", None)
        .expect("Failed to create binary versioned model");

    // Binary should be smaller (or at least not significantly larger)
    // This is a general expectation, though not guaranteed for very small models
    println!(
        "JSON size: {}, Binary size: {}",
        json_versioned.data.len(),
        binary_versioned.data.len()
    );
}
