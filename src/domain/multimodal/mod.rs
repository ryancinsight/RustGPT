//! Multi-modal processing module for image, video, and audio/speech training.
//!
//! This module provides unified interfaces for processing different modalities
//! and converting them into embeddings compatible with the transformer architecture.

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
use ndarray::Array2;

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
}
