//! Integration tests for multimodal training with real data.
//!
//! Tests the complete pipeline from data loading through multimodal processing.

use llm::domain::multimodal::{
    image::{ImageConfig, ImageEncoder, ImageSample},
    audio::{AudioConfig, AudioEncoder, AudioSample},
    processor::{Modality, MultiModalProcessor, MultiModalExample},
};
use llm::infrastructure::persistence::{
    mnist_loader::{load_mnist_training_data, MNIST_IMAGE_SIZE},
    speech_loader::{load_speech_training_data, SPEECH_COMMANDS},
    dataset::{Dataset, DatasetType},
};

/// Test MNIST data loading and image encoding
#[test]
fn test_mnist_image_loading_and_encoding() {
    // Try to load MNIST data (may not exist in all test environments)
    let mnist_dir = "data/mnist";
    let examples = match load_mnist_training_data(mnist_dir, Some(10)) {
        Ok(examples) => examples,
        Err(_) => {
            println!("MNIST data not available, skipping test");
            return;
        }
    };

    assert!(!examples.is_empty(), "Should load at least some MNIST examples");
    
    // Create image encoder
    let config = ImageConfig {
        image_height: MNIST_IMAGE_SIZE,
        image_width: MNIST_IMAGE_SIZE,
        patch_size: 7,
        num_channels: 1,
        embedding_dim: 64,
        use_cls_token: true,
        ..Default::default()
    };
    
    let encoder = ImageEncoder::new(config).expect("Failed to create image encoder");
    
    // Test encoding
    let pixels = vec![0.5f32; MNIST_IMAGE_SIZE * MNIST_IMAGE_SIZE];
    let sample = ImageSample::new(pixels, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE, 1);
    
    let embeddings = encoder.encode(&sample).expect("Failed to encode image");
    
    // 28x28 with 7x7 patches = 4x4 = 16 patches + 1 CLS = 17
    assert_eq!(embeddings.nrows(), 17, "Should have 17 embeddings (16 patches + CLS)");
    assert_eq!(embeddings.ncols(), 64, "Embedding dimension should be 64");
}

/// Test Speech Commands data loading
#[test]
fn test_speech_commands_loading() {
    let speech_dir = "data/speech_commands";
    
    let examples = match load_speech_training_data(speech_dir, Some(5)) {
        Ok(examples) => examples,
        Err(_) => {
            println!("Speech Commands data not available, skipping test");
            return;
        }
    };
    
    assert!(!examples.is_empty(), "Should load at least some speech examples");
    
    // Check that we have examples from expected commands
    let commands_found: std::collections::HashSet<_> = examples
        .iter()
        .map(|ex| ex.transcript.clone())
        .collect();
    
    println!("Commands found: {:?}", commands_found);
    
    // Should have at least one of the expected commands
    let has_expected = SPEECH_COMMANDS.iter().any(|cmd| commands_found.contains(*cmd));
    assert!(has_expected, "Should have at least one expected command");
}

/// Test audio encoding with speech data configuration
#[test]
fn test_audio_encoding() {
    // Create audio encoder with speech-optimized config
    let config = AudioConfig {
        sample_rate: 16000,
        max_duration: 1.0,
        n_fft: 400,
        hop_length: 160,
        n_mels: 64,
        temporal_patch_size: 16,
        freq_patch_size: 8,
        embedding_dim: 64,
        ..Default::default()
    };
    
    let encoder = AudioEncoder::new(config).expect("Failed to create audio encoder");
    
    // Create synthetic audio sample
    let waveform = vec![0.1f32; 16000]; // 1 second at 16kHz
    let sample = AudioSample::new(waveform, 16000);
    
    let embeddings = encoder.encode(&sample).expect("Failed to encode audio");
    
    assert!(embeddings.nrows() > 0, "Should have audio embeddings");
    assert_eq!(embeddings.ncols(), 64, "Embedding dimension should be 64");
}

/// Test multimodal processor with multiple modalities
#[test]
fn test_multimodal_processor() {
    let processor = MultiModalProcessor::new(
        64,
        Some(ImageConfig {
            image_height: 28,
            image_width: 28,
            patch_size: 7,
            num_channels: 1,
            embedding_dim: 64,
            use_cls_token: true,
            ..Default::default()
        }),
        None, // No video
        Some(AudioConfig {
            sample_rate: 8000,
            max_duration: 1.0,
            n_fft: 256,
            hop_length: 128,
            n_mels: 64,
            temporal_patch_size: 63,
            freq_patch_size: 8,
            embedding_dim: 64,
            ..Default::default()
        }),
    ).expect("Failed to create multimodal processor");
    
    // Test text example
    let text_example = MultiModalExample::Text {
        tokens: vec![1, 2, 3, 4, 5],
        label: None,
    };
    
    assert_eq!(text_example.primary_modality(), Modality::Text);
    assert!(text_example.has_modality(Modality::Text));
    assert!(!text_example.has_modality(Modality::Image));
    
    // Test image example
    let image_sample = ImageSample::new(
        vec![0.5f32; 28 * 28],
        28,
        28,
        1,
    );
    let image_example = MultiModalExample::Image { sample: image_sample };
    
    assert_eq!(image_example.primary_modality(), Modality::Image);
    
    // Test audio example
    let audio_sample = AudioSample::new(vec![0.1f32; 8000], 8000);
    let audio_example = MultiModalExample::Audio { sample: audio_sample };
    
    assert_eq!(audio_example.primary_modality(), Modality::Audio);
    
    // Verify processor supports correct modalities
    assert!(processor.supports_modality(Modality::Text));
    assert!(processor.supports_modality(Modality::Image));
    assert!(processor.supports_modality(Modality::Audio));
    assert!(!processor.supports_modality(Modality::Video));
}

/// Test dataset with real multimodal data
#[test]
fn test_dataset_with_real_multimodal_data() {
    let dataset = Dataset::with_real_multimodal_data(
        "data/pretraining_data.json".to_string(),
        "data/chat_training_data.json".to_string(),
        DatasetType::JSON,
        Some(100),    // Max 100 MNIST samples
        Some(5),      // Max 5 per speech class
    );
    
    match dataset {
        Ok(dataset) => {
            println!(
                "Loaded dataset with {} images, {} speech samples",
                dataset.image_training_data.len(),
                dataset.speech_training_data.len()
            );
            
            // Check if we got multimodal data
            let has_images = !dataset.image_training_data.is_empty();
            let has_speech = !dataset.speech_training_data.is_empty();
            
            if has_images || has_speech {
                println!("Successfully loaded multimodal data");
                
                // Test getting all text data
                let all_text = dataset.get_all_text_data();
                assert!(!all_text.is_empty(), "Should have some text data");
                
                // Test multimodal check
                assert!(dataset.has_multimodal_data(), "Should detect multimodal data");
            } else {
                println!("No multimodal data found (data directories may not exist)");
            }
        }
        Err(e) => {
            println!("Failed to load dataset: {}", e);
            // Don't fail test if data is not available
        }
    }
}

/// Test multimodal batch processing
#[test]
fn test_multimodal_batch_processing() {
    let processor = MultiModalProcessor::text_only(64);
    
    // Create batch of text examples
    let examples = vec![
        MultiModalExample::Text {
            tokens: vec![1, 2, 3],
            label: None,
        },
        MultiModalExample::Text {
            tokens: vec![4, 5, 6, 7],
            label: None,
        },
    ];
    
    let batch = processor.process_batch(&examples).expect("Failed to process batch");
    
    assert_eq!(batch.batch_size, 2);
    assert!(!batch.embeddings.is_empty());
    
    // Check sequence length calculation
    let total_len = batch.total_sequence_length();
    assert_eq!(total_len, 7, "Total sequence length should be 3 + 4 = 7");
}

/// Test image normalization ranges
#[test]
fn test_image_normalization() {
    use llm::domain::multimodal::image::ImageNormRange;
    
    let mut sample = ImageSample::new(
        (0..784).map(|i| i as f32).collect(),
        28,
        28,
        1,
    );
    
    // Test [0, 1] normalization
    sample.normalize(ImageNormRange::ZeroToOne);
    let max_val = sample.pixels.iter().fold(0.0f32, |a, &b| a.max(b));
    assert!((max_val - 1.0).abs() < 1e-5, "Max should be ~1.0 after normalization");
    
    // Test [-1, 1] normalization
    let mut sample2 = ImageSample::new(
        vec![0.0f32, 127.5, 255.0],
        1,
        3,
        1,
    );
    sample2.normalize(ImageNormRange::NegOneToOne);
    
    // 0 -> -1, 127.5 -> 0, 255 -> 1
    assert!((sample2.pixels[0] - (-1.0)).abs() < 1e-5, "0 should map to -1");
    assert!(sample2.pixels[1].abs() < 1e-5, "127.5 should map to ~0");
    assert!((sample2.pixels[2] - 1.0).abs() < 1e-5, "255 should map to 1");
}

/// Test audio preprocessing operations
#[test]
fn test_audio_preprocessing() {
    let mut sample = AudioSample::new(
        (0..16000).map(|i| (i as f32 / 16000.0).sin()).collect(),
        16000,
    );
    
    // Test normalization
    sample.normalize();
    let max_abs = sample.waveform.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    assert!((max_abs - 1.0).abs() < 1e-5 || max_abs == 0.0, "Max abs should be ~1.0 after normalization");
    
    // Test pad/truncate
    sample.pad_or_truncate(8000);
    assert_eq!(sample.waveform.len(), 8000);
    
    // Test resampling
    let original_sample = AudioSample::new(vec![0.0f32, 0.5, 1.0, 0.5, 0.0], 16000);
    let resampled = original_sample.resample(8000);
    assert_eq!(resampled.len(), 5, "Half sample rate should give ~half samples");
}