use llm::infrastructure::persistence::loader::DatasetLoader;
use llm::infrastructure::persistence::mnist_loader::{MnistDatasetType, MnistLoader};
use llm::infrastructure::persistence::speech_loader::{SpeechConfig, SpeechLoader};
use std::path::PathBuf;

#[test]
fn test_speech_loader_trait_impl() {
    let config = SpeechConfig::default();
    let loader = SpeechLoader::new(config, None);
    // Just verifying that it implements the trait and compiles.
    // It will return an error because the path doesn't exist, but that's expected.
    let result = loader.load(PathBuf::from("dummy/path"));
    assert!(result.is_err());
}

#[test]
fn test_mnist_loader_trait_impl() {
    let loader = MnistLoader::new(MnistDatasetType::Train);
    // Just verifying that it implements the trait and compiles.
    let result = loader.load(PathBuf::from("dummy/path"));
    assert!(result.is_err());
}
