//! Multi-modal processing module for image, video, and audio/speech training.
//!
//! This module provides unified interfaces for processing different modalities
//! and converting them into embeddings compatible with the transformer architecture.
//!
//! # Features
//!
//! - **Modality Token Type Embeddings**: Learnable embeddings to distinguish between
//!   different modalities (text, image, video, audio) in a unified sequence.
//! - **Data Augmentation**: Built-in augmentation for images (flip, crop, color jitter),
//!   video (temporal jitter, frame dropout), and audio (noise, time stretch, pitch shift).
//! - **Modality Dropout**: Randomly drop modalities during training for robustness.
//! - **Cross-Modal Attention**: Support for attention across different modalities.

pub mod audio;
pub mod image;
pub mod patch;
pub mod processor;
pub mod video;

pub use audio::{AudioConfig, AudioEncoder, AudioSample};
pub use image::{ImageConfig, ImageEncoder, ImageSample};
pub use patch::{PatchEmbed, PatchMerge};
pub use processor::{Modality, MultiModalBatch, MultiModalProcessor};
pub use video::{VideoConfig, VideoEncoder, VideoSample};

use crate::common::errors::Result;
use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Modality token type identifiers for cross-modal attention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ModalityTokenType {
    /// Text token
    Text = 0,
    /// Image patch token
    Image = 1,
    /// Video spatio-temporal patch token
    Video = 2,
    /// Audio spectrogram patch token
    Audio = 3,
    /// Padding token (should be masked in attention)
    Padding = 4,
}

impl ModalityTokenType {
    /// Get the number of modality token types
    pub fn num_types() -> usize {
        5
    }

    /// Convert to index for embedding lookup
    pub fn to_index(self) -> usize {
        self as usize
    }

    /// Convert from index
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::Text),
            1 => Some(Self::Image),
            2 => Some(Self::Video),
            3 => Some(Self::Audio),
            4 => Some(Self::Padding),
            _ => None,
        }
    }
}

impl From<Modality> for ModalityTokenType {
    fn from(modality: Modality) -> Self {
        match modality {
            Modality::Text => Self::Text,
            Modality::Image => Self::Image,
            Modality::Video => Self::Video,
            Modality::Audio => Self::Audio,
        }
    }
}

/// Learnable modality token type embeddings
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ModalityTypeEmbeddings {
    /// Embedding matrix (num_types x embedding_dim)
    pub embeddings: Array2<f32>,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl ModalityTypeEmbeddings {
    /// Create new modality type embeddings with random initialization
    pub fn new(embedding_dim: usize) -> Result<Self> {
        if embedding_dim == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "embedding_dim must be > 0".to_string(),
            });
        }

        let mut rng = crate::common::rng::get_rng();
        let std = 0.02;
        let normal = Normal::new(0.0, std).unwrap();

        let embeddings =
            Array2::from_shape_fn((ModalityTokenType::num_types(), embedding_dim), |_| {
                normal.sample(&mut rng)
            });

        Ok(Self {
            embeddings,
            embedding_dim,
        })
    }

    /// Get embedding for a specific modality token type
    pub fn get(&self, token_type: ModalityTokenType) -> Array1<f32> {
        self.embeddings.row(token_type.to_index()).to_owned()
    }

    /// Get embedding as a vector (copies data)
    pub fn get_vec(&self, token_type: ModalityTokenType) -> Vec<f32> {
        self.embeddings.row(token_type.to_index()).to_vec()
    }

    /// Get embeddings for a sequence of modality types
    pub fn get_sequence(&self, token_types: &[ModalityTokenType]) -> Array2<f32> {
        Array2::from_shape_vec(
            (token_types.len(), self.embedding_dim),
            token_types.iter().flat_map(|&t| self.get_vec(t)).collect(),
        )
        .unwrap_or_else(|_| Array2::zeros((token_types.len(), self.embedding_dim)))
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.embeddings.len()
    }
}

/// Configuration for modality dropout during training
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModalityDropoutConfig {
    /// Probability of dropping each modality (independent)
    pub text_drop_prob: f32,
    pub image_drop_prob: f32,
    pub video_drop_prob: f32,
    pub audio_drop_prob: f32,
    /// Minimum number of modalities to keep
    pub min_modalities: usize,
    /// Whether to drop entire modalities or just mask tokens
    pub drop_entire_modality: bool,
}

impl Default for ModalityDropoutConfig {
    fn default() -> Self {
        Self {
            text_drop_prob: 0.0, // Text is usually kept
            image_drop_prob: 0.1,
            video_drop_prob: 0.15,
            audio_drop_prob: 0.1,
            min_modalities: 1,
            drop_entire_modality: true,
        }
    }
}

impl ModalityDropoutConfig {
    /// Create a new dropout config with uniform drop probability
    pub fn uniform(prob: f32) -> Self {
        Self {
            text_drop_prob: prob,
            image_drop_prob: prob,
            video_drop_prob: prob,
            audio_drop_prob: prob,
            ..Default::default()
        }
    }

    /// No dropout
    pub fn none() -> Self {
        Self {
            text_drop_prob: 0.0,
            image_drop_prob: 0.0,
            video_drop_prob: 0.0,
            audio_drop_prob: 0.0,
            min_modalities: 1,
            drop_entire_modality: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.text_drop_prob < 0.0 || self.text_drop_prob > 1.0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "text_drop_prob must be in [0, 1]".to_string(),
            });
        }
        if self.image_drop_prob < 0.0 || self.image_drop_prob > 1.0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "image_drop_prob must be in [0, 1]".to_string(),
            });
        }
        if self.video_drop_prob < 0.0 || self.video_drop_prob > 1.0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "video_drop_prob must be in [0, 1]".to_string(),
            });
        }
        if self.audio_drop_prob < 0.0 || self.audio_drop_prob > 1.0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "audio_drop_prob must be in [0, 1]".to_string(),
            });
        }
        Ok(())
    }
}

/// Applies modality dropout during training for robustness
#[derive(Debug, Clone)]
pub struct ModalityDropout {
    config: ModalityDropoutConfig,
}

impl ModalityDropout {
    /// Create a new modality dropout layer
    pub fn new(config: ModalityDropoutConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Apply dropout to determine which modalities to keep
    pub fn apply_dropout(&self, available_modalities: &[Modality]) -> Vec<Modality> {
        use rand::Rng;

        let mut rng = crate::common::rng::get_rng();
        let mut kept = Vec::new();

        for &modality in available_modalities {
            let drop_prob = match modality {
                Modality::Text => self.config.text_drop_prob,
                Modality::Image => self.config.image_drop_prob,
                Modality::Video => self.config.video_drop_prob,
                Modality::Audio => self.config.audio_drop_prob,
            };

            if rng.random::<f32>() >= drop_prob {
                kept.push(modality);
            }
        }

        // Ensure minimum modalities are kept
        if kept.len() < self.config.min_modalities && !available_modalities.is_empty() {
            // Add back modalities until we reach minimum
            for &modality in available_modalities {
                if !kept.contains(&modality) && kept.len() < self.config.min_modalities {
                    kept.push(modality);
                }
            }
        }

        kept
    }

    /// Get the configuration
    pub fn config(&self) -> &ModalityDropoutConfig {
        &self.config
    }
}

/// Cross-modal attention mask builder
#[derive(Debug, Clone)]
pub struct CrossModalMaskBuilder {
    /// Whether modalities can attend to each other
    pub cross_attention_enabled: bool,
    /// Modalities that should not be attended to (e.g., padding)
    pub masked_modalities: Vec<ModalityTokenType>,
}

impl Default for CrossModalMaskBuilder {
    fn default() -> Self {
        Self {
            cross_attention_enabled: true,
            masked_modalities: vec![ModalityTokenType::Padding],
        }
    }
}

impl CrossModalMaskBuilder {
    /// Create a new mask builder
    pub fn new(cross_attention_enabled: bool) -> Self {
        Self {
            cross_attention_enabled,
            ..Default::default()
        }
    }

    /// Build attention mask for a sequence of modality types
    /// Returns a 2D array where 1.0 means attend, 0.0 means mask
    pub fn build_mask(&self, token_types: &[ModalityTokenType]) -> Array2<f32> {
        let seq_len = token_types.len();
        let mut mask = Array2::ones((seq_len, seq_len));

        for (i, &query_type) in token_types.iter().enumerate() {
            for (j, &key_type) in token_types.iter().enumerate() {
                // Mask padding tokens in key positions
                if self.masked_modalities.contains(&key_type) {
                    mask[[i, j]] = 0.0;
                }
                // If cross-attention is disabled, only attend to same modality
                else if !self.cross_attention_enabled && query_type != key_type {
                    mask[[i, j]] = 0.0;
                }
            }
        }

        mask
    }

    /// Build causal mask (for autoregressive generation)
    pub fn build_causal_mask(&self, token_types: &[ModalityTokenType]) -> Array2<f32> {
        let seq_len = token_types.len();
        let mut mask = self.build_mask(token_types);

        // Apply causal mask
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[i, j]] = 0.0;
            }
        }

        mask
    }
}

/// Trait for modality-specific encoders
pub trait ModalityEncoder: Send + Sync {
    /// Encode input data into embeddings
    fn encode(&self, input: &ModalityInput) -> Result<Array2<f32>>;

    /// Get the output embedding dimension
    fn output_dim(&self) -> usize;

    /// Get the modality type
    fn modality(&self) -> Modality;
}

/// Input data for different modalities
#[derive(Debug, Clone)]
pub enum ModalityInput {
    /// Text tokens
    Text(Vec<usize>),
    /// Image pixels (flattened or tensor)
    Image(Vec<f32>),
    /// Video frames (sequence of image data)
    Video(Vec<Vec<f32>>),
    /// Audio waveform or spectrogram
    Audio(Vec<f32>),
}

impl ModalityInput {
    /// Get the modality type for this input
    pub fn modality(&self) -> Modality {
        match self {
            ModalityInput::Text(_) => Modality::Text,
            ModalityInput::Image(_) => Modality::Image,
            ModalityInput::Video(_) => Modality::Video,
            ModalityInput::Audio(_) => Modality::Audio,
        }
    }
}

/// Configuration for multi-modal training
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MultiModalConfig {
    /// Enable image processing
    #[serde(default = "default_true")]
    pub enable_image: bool,
    /// Enable video processing
    #[serde(default = "default_true")]
    pub enable_video: bool,
    /// Enable audio processing
    #[serde(default = "default_true")]
    pub enable_audio: bool,
    /// Image patch size (square patches)
    #[serde(default = "image_patch_size_default")]
    pub image_patch_size: usize,
    /// Video patch size (spatio-temporal patches)
    #[serde(default = "video_patch_size_default")]
    pub video_patch_size: usize,
    /// Audio patch size (temporal patches)
    #[serde(default = "audio_patch_size_default")]
    pub audio_patch_size: usize,
    /// Number of frames to sample from videos
    #[serde(default = "video_num_frames_default")]
    pub video_num_frames: usize,
    /// Audio sample rate in Hz
    #[serde(default = "audio_sample_rate_default")]
    pub audio_sample_rate: usize,
    /// Maximum audio duration in seconds
    #[serde(default = "max_audio_duration_default")]
    pub max_audio_duration: f32,
    /// Target embedding dimension (must match model)
    pub embedding_dim: usize,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            enable_image: true,
            enable_video: true,
            enable_audio: true,
            image_patch_size: 16,
            video_patch_size: 2,
            audio_patch_size: 400,
            video_num_frames: 8,
            audio_sample_rate: 16000,
            max_audio_duration: 30.0,
            embedding_dim: 128,
        }
    }
}

impl MultiModalConfig {
    /// Create a new config with the specified embedding dimension
    pub fn with_embedding_dim(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.image_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "image_patch_size must be > 0".to_string(),
            });
        }
        if self.video_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "video_patch_size must be > 0".to_string(),
            });
        }
        if self.audio_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "audio_patch_size must be > 0".to_string(),
            });
        }
        if self.embedding_dim == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "embedding_dim must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

fn default_true() -> bool {
    true
}

fn image_patch_size_default() -> usize {
    16
}

fn video_patch_size_default() -> usize {
    2
}

fn audio_patch_size_default() -> usize {
    400
}

fn video_num_frames_default() -> usize {
    8
}

fn audio_sample_rate_default() -> usize {
    16000
}

fn max_audio_duration_default() -> f32 {
    30.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_config_default() {
        let config = MultiModalConfig::default();
        assert!(config.enable_image);
        assert!(config.enable_video);
        assert!(config.enable_audio);
        assert_eq!(config.image_patch_size, 16);
        assert_eq!(config.embedding_dim, 128);
    }

    #[test]
    fn test_modality_input_types() {
        let text = ModalityInput::Text(vec![1, 2, 3]);
        let image = ModalityInput::Image(vec![0.5; 100]);
        let video = ModalityInput::Video(vec![vec![0.5; 100]; 8]);
        let audio = ModalityInput::Audio(vec![0.1; 1000]);

        assert_eq!(text.modality(), Modality::Text);
        assert_eq!(image.modality(), Modality::Image);
        assert_eq!(video.modality(), Modality::Video);
        assert_eq!(audio.modality(), Modality::Audio);
    }

    #[test]
    fn test_modality_token_type_conversions() {
        // Test index conversions
        assert_eq!(ModalityTokenType::Text.to_index(), 0);
        assert_eq!(ModalityTokenType::Image.to_index(), 1);
        assert_eq!(ModalityTokenType::Video.to_index(), 2);
        assert_eq!(ModalityTokenType::Audio.to_index(), 3);
        assert_eq!(ModalityTokenType::Padding.to_index(), 4);

        // Test from_index
        assert_eq!(
            ModalityTokenType::from_index(0),
            Some(ModalityTokenType::Text)
        );
        assert_eq!(ModalityTokenType::from_index(5), None);

        // Test from Modality
        assert_eq!(
            ModalityTokenType::from(Modality::Text),
            ModalityTokenType::Text
        );
        assert_eq!(
            ModalityTokenType::from(Modality::Image),
            ModalityTokenType::Image
        );
    }

    #[test]
    fn test_modality_type_embeddings() {
        let embeddings = ModalityTypeEmbeddings::new(64).unwrap();
        assert_eq!(embeddings.embedding_dim, 64);
        assert_eq!(
            embeddings.embeddings.nrows(),
            ModalityTokenType::num_types()
        );
        assert_eq!(embeddings.embeddings.ncols(), 64);

        // Test getting single embedding
        let text_emb = embeddings.get(ModalityTokenType::Text);
        assert_eq!(text_emb.len(), 64);

        // Test getting sequence
        let seq_types = vec![
            ModalityTokenType::Text,
            ModalityTokenType::Image,
            ModalityTokenType::Text,
        ];
        let seq_emb = embeddings.get_sequence(&seq_types);
        assert_eq!(seq_emb.shape(), &[3, 64]);

        // Test parameter count
        assert_eq!(embeddings.num_parameters(), 5 * 64);
    }

    #[test]
    fn test_modality_dropout_config() {
        let config = ModalityDropoutConfig::default();
        assert_eq!(config.text_drop_prob, 0.0);
        assert!(config.image_drop_prob > 0.0);

        let uniform = ModalityDropoutConfig::uniform(0.2);
        assert_eq!(uniform.text_drop_prob, 0.2);
        assert_eq!(uniform.image_drop_prob, 0.2);

        let none = ModalityDropoutConfig::none();
        assert_eq!(none.text_drop_prob, 0.0);
        assert_eq!(none.image_drop_prob, 0.0);

        // Validation
        assert!(config.validate().is_ok());
        let invalid = ModalityDropoutConfig {
            text_drop_prob: -0.1,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_modality_dropout() {
        let config = ModalityDropoutConfig::none();
        let dropout = ModalityDropout::new(config).unwrap();

        // With no dropout, all modalities should be kept
        let modalities = vec![Modality::Text, Modality::Image];
        let kept = dropout.apply_dropout(&modalities);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_cross_modal_mask_builder() {
        let builder = CrossModalMaskBuilder::new(true);

        let token_types = vec![
            ModalityTokenType::Text,
            ModalityTokenType::Text,
            ModalityTokenType::Image,
            ModalityTokenType::Padding,
        ];

        let mask = builder.build_mask(&token_types);
        assert_eq!(mask.shape(), &[4, 4]);

        // Padding should be masked in all key positions
        for i in 0..4 {
            assert_eq!(mask[[i, 3]], 0.0); // Padding column should be 0
        }

        // Non-padding should be attendable
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 2]], 1.0); // Cross-modal attention enabled
    }

    #[test]
    fn test_cross_modal_mask_builder_no_cross_attention() {
        let builder = CrossModalMaskBuilder::new(false);

        let token_types = vec![
            ModalityTokenType::Text,
            ModalityTokenType::Text,
            ModalityTokenType::Image,
        ];

        let mask = builder.build_mask(&token_types);

        // Text can attend to text
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[0, 1]], 1.0);
        // Text cannot attend to image (cross-attention disabled)
        assert_eq!(mask[[0, 2]], 0.0);
        // Image can attend to image
        assert_eq!(mask[[2, 2]], 1.0);
    }

    #[test]
    fn test_causal_mask() {
        let builder = CrossModalMaskBuilder::default();
        let token_types = vec![
            ModalityTokenType::Text,
            ModalityTokenType::Text,
            ModalityTokenType::Image,
        ];

        let mask = builder.build_causal_mask(&token_types);

        // Causal: can only attend to past and current
        assert_eq!(mask[[0, 0]], 1.0); // Can attend to self
        assert_eq!(mask[[0, 1]], 0.0); // Cannot attend to future
        assert_eq!(mask[[1, 0]], 1.0); // Can attend to past
        assert_eq!(mask[[1, 1]], 1.0); // Can attend to self
        assert_eq!(mask[[1, 2]], 0.0); // Cannot attend to future
    }
}
