//! Video processing module for multi-modal training.
//!
//! Implements video understanding through spatio-temporal patch embedding,
//! treating video as a sequence of frame patches with temporal dimension.

use ndarray::{Array1, Array2, Array4};
use serde::{Deserialize, Serialize};

use crate::common::errors::Result;
use crate::domain::multimodal::{
    image::{ImageConfig, ImageSample},
    patch::{PatchEmbed, get_1d_sincos_pos_embed},
};

/// Configuration for video processing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VideoConfig {
    /// Number of frames to sample from video
    pub num_frames: usize,
    /// Frame height in pixels
    pub frame_height: usize,
    /// Frame width in pixels
    pub frame_width: usize,
    /// Temporal patch size (number of frames per temporal patch)
    pub temporal_patch_size: usize,
    /// Spatial patch size (square patches)
    pub spatial_patch_size: usize,
    /// Number of color channels
    pub num_channels: usize,
    /// Target embedding dimension
    pub embedding_dim: usize,
    /// Use temporal position embeddings
    #[serde(default = "default_true")]
    pub use_temporal_embeddings: bool,
    /// Sampling strategy for frames
    #[serde(default)]
    pub frame_sampling: FrameSamplingStrategy,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            num_frames: 8,
            frame_height: 224,
            frame_width: 224,
            temporal_patch_size: 2,
            spatial_patch_size: 16,
            num_channels: 3,
            embedding_dim: 768,
            use_temporal_embeddings: true,
            frame_sampling: FrameSamplingStrategy::Uniform,
        }
    }
}

impl VideoConfig {
    /// Calculate number of spatial patches per frame
    pub fn num_spatial_patches(&self) -> usize {
        (self.frame_height / self.spatial_patch_size) * (self.frame_width / self.spatial_patch_size)
    }

    /// Calculate number of temporal patches
    pub fn num_temporal_patches(&self) -> usize {
        self.num_frames / self.temporal_patch_size
    }

    /// Calculate total number of spatio-temporal patches
    pub fn total_patches(&self) -> usize {
        self.num_spatial_patches() * self.num_temporal_patches()
    }

    /// Calculate the flattened patch dimension
    pub fn patch_dim(&self) -> usize {
        self.spatial_patch_size
            * self.spatial_patch_size
            * self.num_channels
            * self.temporal_patch_size
    }

    /// Get effective image config for frame processing
    pub fn to_image_config(&self) -> ImageConfig {
        ImageConfig {
            patch_size: self.spatial_patch_size,
            image_height: self.frame_height,
            image_width: self.frame_width,
            num_channels: self.num_channels,
            embedding_dim: self.embedding_dim,
            use_position_embeddings: false, // We handle our own position embeddings
            normalize_pixels: true,
            use_cls_token: false,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.num_frames == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "num_frames must be > 0".to_string(),
            });
        }
        if self.temporal_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "temporal_patch_size must be > 0".to_string(),
            });
        }
        if self.num_frames % self.temporal_patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "num_frames ({}) must be divisible by temporal_patch_size ({})",
                    self.num_frames, self.temporal_patch_size
                ),
            });
        }
        if self.spatial_patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "spatial_patch_size must be > 0".to_string(),
            });
        }
        if self.frame_height % self.spatial_patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "frame_height ({}) must be divisible by spatial_patch_size ({})",
                    self.frame_height, self.spatial_patch_size
                ),
            });
        }
        if self.frame_width % self.spatial_patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "frame_width ({}) must be divisible by spatial_patch_size ({})",
                    self.frame_width, self.spatial_patch_size
                ),
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

/// Frame sampling strategy for videos
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameSamplingStrategy {
    /// Uniform sampling across the video
    Uniform,
    /// Random sampling
    Random,
    /// Sample from the start of the video
    Start,
    /// Sample from the end of the video
    End,
    /// Sample from the middle of the video
    Middle,
}

impl Default for FrameSamplingStrategy {
    fn default() -> Self {
        FrameSamplingStrategy::Uniform
    }
}

/// A single video sample
#[derive(Debug, Clone)]
pub struct VideoSample {
    /// Frame data (flattened pixels for each frame)
    pub frames: Vec<Vec<f32>>,
    /// Number of frames
    pub num_frames: usize,
    /// Frame height
    pub height: usize,
    /// Frame width
    pub width: usize,
    /// Number of channels
    pub channels: usize,
    /// Frames per second (if known)
    pub fps: Option<f32>,
    /// Optional caption or description
    pub caption: Option<String>,
}

impl VideoSample {
    /// Create a new video sample
    pub fn new(frames: Vec<Vec<f32>>, height: usize, width: usize, channels: usize) -> Self {
        let num_frames = frames.len();
        Self {
            frames,
            num_frames,
            height,
            width,
            channels,
            fps: None,
            caption: None,
        }
    }

    /// Create a new video sample with caption
    pub fn with_caption(
        frames: Vec<Vec<f32>>,
        height: usize,
        width: usize,
        channels: usize,
        caption: String,
    ) -> Self {
        let num_frames = frames.len();
        Self {
            frames,
            num_frames,
            height,
            width,
            channels,
            fps: None,
            caption: Some(caption),
        }
    }

    /// Sample frames according to the specified strategy
    pub fn sample_frames(
        &self,
        num_frames: usize,
        strategy: FrameSamplingStrategy,
    ) -> Vec<Vec<f32>> {
        if self.num_frames <= num_frames {
            return self.frames.clone();
        }

        match strategy {
            FrameSamplingStrategy::Uniform => {
                let step = self.num_frames as f32 / num_frames as f32;
                (0..num_frames)
                    .map(|i| {
                        let idx = ((i as f32 * step) as usize).min(self.num_frames - 1);
                        self.frames[idx].clone()
                    })
                    .collect()
            }
            FrameSamplingStrategy::Random => {
                use crate::common::rng::get_rng;
                use rand::seq::SliceRandom;
                let mut rng = get_rng();
                let mut indices: Vec<usize> = (0..self.num_frames).collect();
                indices.shuffle(&mut rng);
                indices.truncate(num_frames);
                indices.sort_unstable();
                indices.iter().map(|&i| self.frames[i].clone()).collect()
            }
            FrameSamplingStrategy::Start => self.frames.iter().take(num_frames).cloned().collect(),
            FrameSamplingStrategy::End => self
                .frames
                .iter()
                .skip(self.num_frames.saturating_sub(num_frames))
                .cloned()
                .collect(),
            FrameSamplingStrategy::Middle => {
                let start = (self.num_frames - num_frames) / 2;
                self.frames
                    .iter()
                    .skip(start)
                    .take(num_frames)
                    .cloned()
                    .collect()
            }
        }
    }

    /// Extract a single frame as an ImageSample
    pub fn get_frame(&self, frame_idx: usize) -> Option<ImageSample> {
        self.frames
            .get(frame_idx)
            .map(|pixels| ImageSample::new(pixels.clone(), self.height, self.width, self.channels))
    }
}

/// Video encoder using spatio-temporal patch embedding
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VideoEncoder {
    pub config: VideoConfig,
    pub spatiotemporal_embed: PatchEmbed,
    #[serde(skip, default)]
    pub temporal_position_embeddings: Option<Array2<f32>>,
}

impl VideoEncoder {
    /// Create a new video encoder
    pub fn new(config: VideoConfig) -> Result<Self> {
        config.validate()?;

        let patch_dim = config.patch_dim();
        let spatiotemporal_embed = PatchEmbed::new(patch_dim, config.embedding_dim)?;

        let temporal_position_embeddings = if config.use_temporal_embeddings {
            let num_temporal_tokens = config.num_temporal_patches();
            Some(get_1d_sincos_pos_embed(
                config.embedding_dim,
                num_temporal_tokens,
            ))
        } else {
            None
        };

        Ok(Self {
            config,
            spatiotemporal_embed,
            temporal_position_embeddings,
        })
    }

    /// Encode a video sample into embeddings
    pub fn encode(&self, sample: &VideoSample) -> Result<Array2<f32>> {
        // Sample frames according to config
        let sampled_frames =
            sample.sample_frames(self.config.num_frames, self.config.frame_sampling);

        // Extract spatio-temporal patches
        let st_patches = self.extract_spatiotemporal_patches(&sampled_frames)?;

        // Embed patches
        let embeddings = self.spatiotemporal_embed.forward(&st_patches);

        // Add temporal position embeddings if enabled
        if let Some(ref temp_pos) = self.temporal_position_embeddings {
            // Reshape to apply temporal position per spatial position
            let num_spatial = self.config.num_spatial_patches();
            let num_temporal = self.config.num_temporal_patches();

            let mut reshaped = embeddings;
            for t in 0..num_temporal {
                for s in 0..num_spatial {
                    let idx = t * num_spatial + s;
                    let mut row = reshaped.row_mut(idx);
                    for (i, &pos_val) in temp_pos.row(t).iter().enumerate() {
                        row[i] += pos_val;
                    }
                }
            }
            return Ok(reshaped);
        }

        Ok(embeddings)
    }

    /// Extract spatio-temporal patches from frame sequence
    fn extract_spatiotemporal_patches(&self, frames: &[Vec<f32>]) -> Result<Array2<f32>> {
        if frames.len() != self.config.num_frames {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Expected {} frames, got {}",
                    self.config.num_frames,
                    frames.len()
                ),
            });
        }

        let num_spatial_patches = self.config.num_spatial_patches();
        let num_temporal_patches = self.config.num_temporal_patches();
        let total_patches = num_spatial_patches * num_temporal_patches;
        let patch_dim = self.config.patch_dim();

        let mut patches = Array2::zeros((total_patches, patch_dim));

        let p_s = self.config.spatial_patch_size;
        let p_t = self.config.temporal_patch_size;
        let c = self.config.num_channels;
        let h = self.config.frame_height;
        let w = self.config.frame_width;
        let grid_h = h / p_s;
        let grid_w = w / p_s;

        // Group frames into temporal chunks
        for (t_idx, temporal_chunk) in frames.chunks(p_t).enumerate() {
            // For each spatial position
            for ph in 0..grid_h {
                for pw in 0..grid_w {
                    let spatial_idx = ph * grid_w + pw;
                    let patch_idx = t_idx * num_spatial_patches + spatial_idx;

                    // Extract spatio-temporal patch data
                    let mut patch_data = Vec::with_capacity(patch_dim);

                    // Iterate over temporal dimension (frames in chunk)
                    for frame in temporal_chunk {
                        let frame_array = Array4::from_shape_vec((1, h, w, c), frame.clone())
                            .map_err(|e| crate::common::errors::ModelError::InvalidInput {
                                message: format!("Failed to reshape frame: {}", e),
                            })?;

                        // Extract spatial patch from this frame
                        for i in 0..p_s {
                            for j in 0..p_s {
                                for ch in 0..c {
                                    let y = ph * p_s + i;
                                    let x = pw * p_s + j;
                                    if y < h && x < w {
                                        patch_data.push(frame_array[[0, y, x, ch]]);
                                    }
                                }
                            }
                        }
                    }

                    if patch_data.len() == patch_dim {
                        patches
                            .row_mut(patch_idx)
                            .assign(&Array1::from_vec(patch_data));
                    }
                }
            }
        }

        Ok(patches)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get sequence length (total patches)
    pub fn sequence_length(&self) -> usize {
        self.config.total_patches()
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_video(
        num_frames: usize,
        height: usize,
        width: usize,
        channels: usize,
    ) -> VideoSample {
        let frames: Vec<Vec<f32>> = (0..num_frames)
            .map(|f| {
                (0..height * width * channels)
                    .map(|i| ((f * 10 + i) % 256) as f32)
                    .collect()
            })
            .collect();
        VideoSample::new(frames, height, width, channels)
    }

    #[test]
    fn test_video_config_validation() {
        let config = VideoConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = VideoConfig {
            temporal_patch_size: 3,
            num_frames: 8,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_video_encoder_creation() {
        let config = VideoConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            spatial_patch_size: 8,
            temporal_patch_size: 2,
            embedding_dim: 64,
            ..Default::default()
        };

        let encoder = VideoEncoder::new(config).unwrap();
        assert_eq!(encoder.config.num_spatial_patches(), 16); // 4x4 per frame
        assert_eq!(encoder.config.num_temporal_patches(), 2); // 4 frames / 2
        assert_eq!(encoder.config.total_patches(), 32); // 16 * 2
    }

    #[test]
    fn test_frame_sampling() {
        let video = create_test_video(20, 32, 32, 3);

        let uniform = video.sample_frames(4, FrameSamplingStrategy::Uniform);
        assert_eq!(uniform.len(), 4);

        let start = video.sample_frames(4, FrameSamplingStrategy::Start);
        assert_eq!(start.len(), 4);

        let end = video.sample_frames(4, FrameSamplingStrategy::End);
        assert_eq!(end.len(), 4);
    }

    #[test]
    fn test_video_encoding() {
        let config = VideoConfig {
            num_frames: 4,
            frame_height: 32,
            frame_width: 32,
            spatial_patch_size: 16,
            temporal_patch_size: 2,
            embedding_dim: 64,
            ..Default::default()
        };

        let encoder = VideoEncoder::new(config).unwrap();
        let video = create_test_video(4, 32, 32, 3);

        let embeddings = encoder.encode(&video).unwrap();

        // 4 frames, 16x16 spatial patches = 2x2 = 4 patches per frame
        // temporal size 2, so 2 temporal groups
        // total: 2 * 4 = 8 patches
        assert_eq!(embeddings.shape()[0], 8);
        assert_eq!(embeddings.shape()[1], 64);
    }
}
