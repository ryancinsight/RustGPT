use std::fs;

use llm::LLM;
use tempfile::NamedTempFile;

#[test]
fn test_llm_save_load_json() {
    let original_llm = LLM::default();

    // Save to JSON
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().with_extension("json");
    let path_str = path.to_str().unwrap();

    original_llm
        .save_json(path_str)
        .expect("Failed to save LLM as JSON");

    // Load from JSON
    let loaded_llm = LLM::load_json(path_str).expect("Failed to load LLM from JSON");

    // Verify structure
    assert_eq!(
        original_llm.vocab.encode.len(),
        loaded_llm.vocab.encode.len()
    );
    assert_eq!(original_llm.network.len(), loaded_llm.network.len());
    assert_eq!(
        original_llm.total_parameters(),
        loaded_llm.total_parameters()
    );

    // Cleanup
    fs::remove_file(path_str).ok();
}

#[test]
fn test_llm_save_load_binary() {
    let original_llm = LLM::default();

    // Save to binary
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path().with_extension("bin");
    let path_str = path.to_str().unwrap();

    original_llm
        .save_binary(path_str)
        .expect("Failed to save LLM as binary");

    // Load from binary
    let loaded_llm = LLM::load_binary(path_str).expect("Failed to load LLM from binary");

    // Verify structure
    assert_eq!(
        original_llm.vocab.encode.len(),
        loaded_llm.vocab.encode.len()
    );
    assert_eq!(original_llm.network.len(), loaded_llm.network.len());
    assert_eq!(
        original_llm.total_parameters(),
        loaded_llm.total_parameters()
    );

    // Cleanup
    fs::remove_file(path_str).ok();
}

#[test]
fn test_llm_save_load_auto_detect() {
    let original_llm = LLM::default();

    // Test JSON auto-detection
    let json_path = "test_auto.json";
    original_llm.save(json_path).expect("Failed to save JSON");
    let loaded_json = LLM::load(json_path).expect("Failed to load JSON");
    assert_eq!(original_llm.network.len(), loaded_json.network.len());
    fs::remove_file(json_path).ok();

    // Test binary auto-detection
    let bin_path = "test_auto.bin";
    original_llm.save(bin_path).expect("Failed to save binary");
    let loaded_bin = LLM::load(bin_path).expect("Failed to load binary");
    assert_eq!(original_llm.network.len(), loaded_bin.network.len());
    fs::remove_file(bin_path).ok();
}

#[test]
fn test_binary_smaller_than_json() {
    let llm = LLM::default();

    let json_path = "test_size.json";
    let bin_path = "test_size.bin";

    llm.save_json(json_path).expect("Failed to save JSON");
    llm.save_binary(bin_path).expect("Failed to save binary");

    let json_size = fs::metadata(json_path).unwrap().len();
    let bin_size = fs::metadata(bin_path).unwrap().len();

    // Binary should be significantly smaller (typically 50-70% smaller)
    assert!(
        bin_size < json_size,
        "Binary size ({}) should be smaller than JSON size ({})",
        bin_size,
        json_size
    );

    fs::remove_file(json_path).ok();
    fs::remove_file(bin_path).ok();
}

#[test]
fn test_save_load_preserves_vocab() {
    let llm = LLM::default();
    let original_words: Vec<&str> = llm.vocab.words();

    let path = "test_vocab.bin";
    llm.save_binary(path).expect("Failed to save");
    let loaded = LLM::load_binary(path).expect("Failed to load");

    assert_eq!(
        original_words,
        loaded.vocab.words(),
        "Vocabulary words should be preserved"
    );

    fs::remove_file(path).ok();
}

#[test]
fn test_load_nonexistent_file() {
    let result = LLM::load_json("nonexistent.json");
    assert!(result.is_err(), "Loading nonexistent file should fail");

    let result = LLM::load_binary("nonexistent.bin");
    assert!(result.is_err(), "Loading nonexistent file should fail");
}

#[test]
fn test_json_is_human_readable() {
    let llm = LLM::default();
    let path = "test_readable.json";

    llm.save_json(path).expect("Failed to save");

    let content = fs::read_to_string(path).expect("Failed to read file");

    assert!(content.contains("vocab"), "JSON should contain vocab field");
    assert!(
        content.contains("network"),
        "JSON should contain network field"
    );
    assert!(content.contains("words"), "JSON should contain words field");

    fs::remove_file(path).ok();
}
