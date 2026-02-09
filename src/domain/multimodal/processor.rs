//! Multi-modal processor for coordinating different modality encoders.
//!
//! Provides unified batching and processing of mixed-modality data,
//! enabling training on text, image, video, and audio samples.
//!
//! # Features
//!
//! - **Modality Token Type Embeddings**: Adds learnable type embeddings to distinguish
//!   between different modalities in a unified sequence.
//! - **Modality Dropout**: Randomly drops modalities during training for robustness.
//! - **Cross-Modal Attention Masks**: Builds attention masks for cross-modal attention.
//! - **Data Augmentation**: Applies modality-specific augmentation during training.

use ndarray::{concatenate, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::common::errors::Result;
use crate::domain::multimodal::{
    audio::{AudioConfig, AudioEncoder, AudioSample},
    image::{ImageConfig, ImageEncoder, ImageSample},
    video::{VideoConfig, VideoEncoder, VideoSample},
    CrossModalMaskBuilder, ModalityDropout, ModalityDropoutConfig, ModalityTokenType,
    ModalityTypeEmbeddings,
};

/// Supported modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

impl Modality {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Text => "text",
            Modality::Image => "image",
            Modality::Video => "video",
            Modality::Audio => "audio",
        }
    }
}

/// A single training example that can contain any modality
#[derive(Debug, Clone)]
pub enum MultiModalExample {
    /// Text example with tokens
    Text { tokens: Vec<usize>, label: Option<String> },
    /// Image example with pixels
    Image { sample: ImageSample },
    /// Video example with frames
    Video { sample: VideoSample },
    /// Audio example with waveform
    Audio { sample: AudioSample },
    /// Mixed example with multiple modalities
    Mixed {
        text: Option<Vec<usize>>,
        image: Option<ImageSample>,
        video: Option<VideoSample>,
        audio: Option<AudioSample>,
        ordering: Vec<Modality>,
    },
}

impl MultiModalExample {
    /// Get the primary modality of this example
    pub fn primary_modality(&self) -> Modality {
        match self {
            MultiModalExample::Text { .. } => Modality::Text,
            MultiModalExample::Image { .. } => Modality::Image,
            MultiModalExample::Video { .. } => Modality::Video,
            MultiModalExample::Audio { .. } => Modality::Audio,
            MultiModalExample::Mixed { ordering, .. } => {
                ordering.first().copied().unwrap_or(Modality::Text)
            }
        }
    }

    /// Check if this example has a specific modality
    pub fn has_modality(&self, modality: Modality) -> bool {
        match self {
            MultiModalExample::Text { .. } => modality == Modality::Text,
            MultiModalExample::Image { .. } => modality == Modality::Image,
            MultiModalExample::Video { .. } => modality == Modality::Video,
            MultiModalExample::Audio { .. } => modality == Modality::Audio,
            MultiModalExample::Mixed { ordering, .. } => ordering.contains(&modality),
        }
    }
}

/// A batch of multi-modal data with enhanced features for training
#[derive(Debug, Clone)]
pub struct MultiModalBatch {
    /// Encoded embeddings for each modality
    pub embeddings: Vec<(Modality, Array2<f32>)>,
    /// Attention masks (if applicable)
    pub attention_masks: Vec<Option<Array2<f32>>>,
    /// Labels for supervised learning
    pub labels: Vec<Option<Vec<usize>>>,
    /// Batch size
    pub batch_size: usize,
    /// Modality token types for each position in the sequence
    pub token_types: Vec<ModalityTokenType>,
}

impl MultiModalBatch {
    /// Create a new empty batch
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            attention_masks: Vec::new(),
            labels: Vec::new(),
            batch_size: 0,
            token_types: Vec::new(),
        }
    }

    /// Concatenate all embeddings along the batch dimension
    pub fn concat_embeddings(&self) -> Option<Array2<f32>> {
        if self.embeddings.is_empty() {
            return None;
        }

        // Group by modality
        let mut text_embeddings = Vec::new();
        let mut image_embeddings = Vec::new();
        let mut video_embeddings = Vec::new();
        let mut audio_embeddings = Vec::new();

        for (modality, emb) in &self.embeddings {
            match modality {
                Modality::Text => text_embeddings.push(emb.view()),
                Modality::Image => image_embeddings.push(emb.view()),
                Modality::Video => video_embeddings.push(emb.view()),
                Modality::Audio => audio_embeddings.push(emb.view()),
            }
        }

        // Concatenate each modality group
        let mut results = Vec::new();

        if !text_embeddings.is_empty() {
            results.push(concatenate(Axis(0), &text_embeddings).ok()?);
        }
        if !image_embeddings.is_empty() {
            results.push(concatenate(Axis(0), &image_embeddings).ok()?);
        }
        if !video_embeddings.is_empty() {
            results.push(concatenate(Axis(0), &video_embeddings).ok()?);
        }
        if !audio_embeddings.is_empty() {
            results.push(concatenate(Axis(0), &audio_embeddings).ok()?);
        }

        if results.is_empty() {
            return None;
        }

        // Concatenate all modalities along sequence dimension
        concatenate(Axis(0), &results.iter().map(|a| a.view()).collect::<Vec<_>>()).ok()
    }

    /// Get the total sequence length across all modalities
    pub fn total_sequence_length(&self) -> usize {
        self.embeddings.iter().map(|(_, emb)| emb.nrows()).sum()
    }

    /// Build token types array for the entire batch
    pub fn build_token_types(&self) -> Vec<ModalityTokenType> {
        self.embeddings
            .iter()
            .flat_map(|(modality, emb)| {
                let token_type = ModalityTokenType::from(*modality);
                vec![token_type; emb.nrows()]
            })
            .collect()
    }

    /// Build cross-modal attention mask
    pub fn build_attention_mask(&self, cross_attention_enabled: bool) -> Option<Array2<f32>> {
        let token_types = self.build_token_types();
        if token_types.is_empty() {
            return None;
        }

        let builder = CrossModalMaskBuilder::new(cross_attention_enabled);
        Some(builder.build_mask(&token_types))
    }

    /// Build causal attention mask (for autoregressive generation)
    pub fn build_causal_mask(&self) -> Option<Array2<f32>> {
        let token_types = self.build_token_types();
        if token_types.is_empty() {
            return None;
        }

        let builder = CrossModalMaskBuilder::default();
        Some(builder.build_causal_mask(&token_types))
    }

    /// Add modality type embeddings to the concatenated embeddings
    pub fn add_modality_type_embeddings(
        &self,
        type_embeddings: &ModalityTypeEmbeddings,
    ) -> Option<Array2<f32>> {
        let mut concatenated = self.concat_embeddings()?;
        let token_types = self.build_token_types();

        if token_types.len() != concatenated.nrows() {
            return None;
        }

        let type_emb_array = type_embeddings.get_sequence(&token_types);
        
        // Add type embeddings to each position
        for (i, row) in concatenated.rows_mut().into_iter().enumerate() {
            for (j, &val) in type_emb_array.row(i).iter().enumerate() {
                row[j] += val;
            }
        }

        Some(concatenated)
    }
}

impl Default for MultiModalBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-modal processor that coordinates encoding of different modalities
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MultiModalProcessor {
    /// Image encoder (if enabled)
    pub image_encoder: Option<ImageEncoder>,
    /// Video encoder (if enabled)
    pub video_encoder: Option<VideoEncoder>,
    /// Audio encoder (if enabled)
    pub audio_encoder: Option<AudioEncoder>,
    /// Target embedding dimension
    pub embedding_dim: usize,
}

impl MultiModalProcessor {
    /// Create a new processor with the specified encoders enabled
    pub fn new(
        embedding_dim: usize,
        image_config: Option<ImageConfig>,
        video_config: Option<VideoConfig>,
        audio_config: Option<AudioConfig>,
    ) -> Result<Self> {
        let image_encoder = if let Some(cfg) = image_config {
            let mut cfg = cfg;
            cfg.embedding_dim = embedding_dim;
            Some(ImageEncoder::new(cfg)?)
        } else {
            None
        };

        let video_encoder = if let Some(cfg) = video_config {
            let mut cfg = cfg;
            cfg.embedding_dim = embedding_dim;
            Some(VideoEncoder::new(cfg)?)
        } else {
            None
        };

        let audio_encoder = if let Some(cfg) = audio_config {
            let mut cfg = cfg;
            cfg.embedding_dim = embedding_dim;
            Some(AudioEncoder::new(cfg)?)
        } else {
            None
        };

        Ok(Self {
            image_encoder,
            video_encoder,
            audio_encoder,
            embedding_dim,
        })
    }

    /// Create a processor with all modalities enabled using default configs
    pub fn with_all_modalities(embedding_dim: usize) -> Result<Self> {
        Self::new(
            embedding_dim,
            Some(ImageConfig::default()),
            Some(VideoConfig::default()),
            Some(AudioConfig::default()),
        )
    }

    /// Create a processor with only text (no multi-modal)
    pub fn text_only(embedding_dim: usize) -> Self {
        Self {
            image_encoder: None,
            video_encoder: None,
            audio_encoder: None,
            embedding_dim,
        }
    }

    /// Check if a modality is supported
    pub fn supports_modality(&self, modality: Modality) -> bool {
        match modality {
            Modality::Text => true, // Text is always supported
            Modality::Image => self.image_encoder.is_some(),
            Modality::Video => self.video_encoder.is_some(),
            Modality::Audio => self.audio_encoder.is_some(),
        }
    }

    /// Process a single example and return embeddings
    pub fn process_example(&self, example: &MultiModalExample) -> Result<Vec<(Modality, Array2<f32>)>> {
        let mut results = Vec::new();

        match example {
            MultiModalExample::Text { tokens, .. } => {
                // Text tokens need to be embedded by the token embedding layer
                // For now, return placeholder that will be processed by TokenEmbeddings
                let placeholder = Array2::zeros((tokens.len(), self.embedding_dim));
                results.push((Modality::Text, placeholder));
            }
            MultiModalExample::Image { sample } => {
                if let Some(ref encoder) = self.image_encoder {
                    let emb = encoder.encode(sample)?;
                    results.push((Modality::Image, emb));
                } else {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: "Image encoder not initialized".to_string(),
                    });
                }
            }
            MultiModalExample::Video { sample } => {
                if let Some(ref encoder) = self.video_encoder {
                    let emb = encoder.encode(sample)?;
                    results.push((Modality::Video, emb));
                } else {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: "Video encoder not initialized".to_string(),
                    });
                }
            }
            MultiModalExample::Audio { sample } => {
                if let Some(ref encoder) = self.audio_encoder {
                    let emb = encoder.encode(sample)?;
                    results.push((Modality::Audio, emb));
                } else {
                    return Err(crate::common::errors::ModelError::InvalidInput {
                        message: "Audio encoder not initialized".to_string(),
                    });
                }
            }
            MultiModalExample::Mixed { text, image, video, audio, ordering } => {
                // Process each modality in the specified order
                for modality in ordering {
                    match modality {
                        Modality::Text => {
                            if let Some(tokens) = text {
                                let placeholder = Array2::zeros((tokens.len(), self.embedding_dim));
                                results.push((Modality::Text, placeholder));
                            }
                        }
                        Modality::Image => {
                            if let Some(sample) = image {
                                if let Some(ref encoder) = self.image_encoder {
                                    let emb = encoder.encode(sample)?;
                                    results.push((Modality::Image, emb));
                                }
                            }
                        }
                        Modality::Video => {
                            if let Some(sample) = video {
                                if let Some(ref encoder) = self.video_encoder {
                                    let emb = encoder.encode(sample)?;
                                    results.push((Modality::Video, emb));
                                }
                            }
                        }
                        Modality::Audio => {
                            if let Some(sample) = audio {
                                if let Some(ref encoder) = self.audio_encoder {
                                    let emb = encoder.encode(sample)?;
                                    results.push((Modality::Audio, emb));
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Process a batch of examples
    pub fn process_batch(&self, examples: &[MultiModalExample]) -> Result<MultiModalBatch> {
        let mut batch = MultiModalBatch::new();
        batch.batch_size = examples.len();

        for example in examples {
            let embeddings = self.process_example(example)?;
            for (modality, emb) in embeddings {
                batch.embeddings.push((modality, emb));
            }
        }

        Ok(batch)
    }

    /// Get the number of parameters across all encoders
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;
        if let Some(ref enc) = self.image_encoder {
            total += enc.patch_embed.num_parameters();
        }
        if let Some(ref enc) = self.video_encoder {
            total += enc.spatiotemporal_embed.num_parameters();
        }
        if let Some(ref enc) = self.audio_encoder {
            total += enc.patch_embed.num_parameters();
        }
        total
    }
}

/// Dataset loader for multi-modal data
#[derive(Debug, Clone)]
pub struct MultiModalDataset {
    /// Examples in the dataset
    pub examples: Vec<MultiModalExample>,
    /// Whether to shuffle on each epoch
    pub shuffle: bool,
}

impl MultiModalDataset {
    /// Create a new dataset
    pub fn new(examples: Vec<MultiModalExample>) -> Self {
        Self {
            examples,
            shuffle: true,
        }
    }

    /// Create a dataset from text data using a tokenization function
    pub fn from_text<F>(texts: Vec<String>, tokenize: F) -> Self
    where
        F: Fn(&str) -> Vec<usize>,
    {
        let examples: Vec<MultiModalExample> = texts
            .into_iter()
            .map(|text| {
                let tokens = tokenize(&text);
                MultiModalExample::Text {
                    tokens,
                    label: None,
                }
            })
            .collect();
        Self::new(examples)
    }

    /// Get batch iterator
    pub fn batches(&self, batch_size: usize) -> MultiModalBatchIterator<'_> {
        MultiModalBatchIterator {
            dataset: self,
            batch_size,
            current_idx: 0,
        }
    }

    /// Get the number of examples
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

/// Iterator over batches in a multi-modal dataset
pub struct MultiModalBatchIterator<'a> {
    dataset: &'a MultiModalDataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for MultiModalBatchIterator<'a> {
    type Item = &'a [MultiModalExample];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        let batch = &self.dataset.examples[self.current_idx..end_idx];
        self.current_idx = end_idx;

        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::multimodal::image::{ImageNormRange, ImageSample};

    fn create_test_image() -> ImageSample {
        let pixels: Vec<f32> = (0..32 * 32 * 3).map(|i| (i % 256) as f32).collect();
        ImageSample::new(pixels, 32, 32, 3)
    }

    #[test]
    fn test_multimodal_processor_creation() {
        // Use compatible audio config to avoid divisibility issues
        let audio_config = AudioConfig {
            max_duration: 1.0,
            sample_rate: 8000,
            n_fft: 256,
            hop_length: 128,
            n_mels: 64,
            temporal_patch_size: 63, // 63 time frames divides evenly
            freq_patch_size: 8,
            embedding_dim: 128,
            ..Default::default()
        };

        let processor = MultiModalProcessor::new(
            128,
            Some(ImageConfig::default()),
            Some(VideoConfig::default()),
            Some(audio_config),
        ).unwrap();
        assert!(processor.supports_modality(Modality::Text));
        assert!(processor.supports_modality(Modality::Image));
        assert!(processor.supports_modality(Modality::Video));
        assert!(processor.supports_modality(Modality::Audio));
    }

    #[test]
    fn test_text_only_processor() {
        let processor = MultiModalProcessor::text_only(128);
        assert!(processor.supports_modality(Modality::Text));
        assert!(!processor.supports_modality(Modality::Image));
    }

    #[test]
    fn test_process_text_example() {
        let processor = MultiModalProcessor::text_only(64);
        let example = MultiModalExample::Text {
            tokens: vec![1, 2, 3, 4, 5],
            label: None,
        };

        let result = processor.process_example(&example).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, Modality::Text);
        assert_eq!(result[0].1.nrows(), 5);
    }

    #[test]
    fn test_process_image_example() {
        let config = ImageConfig {
            image_height: 32,
            image_width: 32,
            patch_size: 16,
            num_channels: 3,
            embedding_dim: 64,
            use_position_embeddings: false,
            ..Default::default()
        };

        let processor = MultiModalProcessor::new(
            64,
            Some(config),
            None,
            None,
        )
        .unwrap();

        let mut sample = create_test_image();
        sample.normalize(ImageNormRange::ZeroToOne);

        let example = MultiModalExample::Image { sample };
        let result = processor.process_example(&example).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, Modality::Image);
        // 32x32 with 16x16 patches = 4 patches + 1 CLS = 5
        assert_eq!(result[0].1.nrows(), 5);
    }

    #[test]
    fn test_multimodal_batch() {
        let mut batch = MultiModalBatch::new();
        batch.embeddings.push((Modality::Text, Array2::zeros((10, 64))));
        batch.embeddings.push((Modality::Image, Array2::zeros((5, 64))));

        assert_eq!(batch.total_sequence_length(), 15);
    }

    #[test]
    fn test_mixed_example() {
        let example = MultiModalExample::Mixed {
            text: Some(vec![1, 2, 3]),
            image: None,
            video: None,
            audio: None,
            ordering: vec![Modality::Text],
        };

        assert_eq!(example.primary_modality(), Modality::Text);
        assert!(example.has_modality(Modality::Text));
        assert!(!example.has_modality(Modality::Image));
    }
}
