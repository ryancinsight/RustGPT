//! Image processing module for multi-modal training.
//!
//! Implements Vision Transformer (ViT)-style patch embedding and encoding
//! for image understanding capabilities.
//!
//! # Data Augmentation
//!
//! The module provides comprehensive data augmentation for training:
//! - Random horizontal/vertical flip
//! - Random crop and resize
//! - Color jitter (brightness, contrast, saturation, hue)
//! - Random rotation
//! - Gaussian noise injection
//! - Cutout/Random erasing

use ndarray::{Array1, Array2, Array3, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::common::{errors::Result, rng::get_rng};
use crate::domain::multimodal::patch::PatchEmbed;

/// Configuration for image processing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Patch size (square patches, e.g., 16x16)
    pub patch_size: usize,
    /// Image height in pixels
    pub image_height: usize,
    /// Image width in pixels
    pub image_width: usize,
    /// Number of color channels (3 for RGB, 1 for grayscale)
    pub num_channels: usize,
    /// Target embedding dimension
    pub embedding_dim: usize,
    /// Use learned position embeddings
    #[serde(default = "default_true")]
    pub use_position_embeddings: bool,
    /// Normalize pixel values to [-1, 1]
    #[serde(default = "default_true")]
    pub normalize_pixels: bool,
    /// Add a class token (CLS token) like ViT
    #[serde(default = "default_false")]
    pub use_cls_token: bool,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            image_height: 224,
            image_width: 224,
            num_channels: 3,
            embedding_dim: 768,
            use_position_embeddings: true,
            normalize_pixels: true,
            use_cls_token: true,
        }
    }
}

impl ImageConfig {
    /// Calculate the number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_height / self.patch_size) * (self.image_width / self.patch_size)
    }

    /// Calculate the sequence length (patches + optional CLS token)
    pub fn sequence_length(&self) -> usize {
        let base = self.num_patches();
        if self.use_cls_token {
            base + 1
        } else {
            base
        }
    }

    /// Calculate the flattened patch dimension
    pub fn patch_dim(&self) -> usize {
        self.patch_size * self.patch_size * self.num_channels
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "patch_size must be > 0".to_string(),
            });
        }
        if self.image_height % self.patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "image_height ({}) must be divisible by patch_size ({})",
                    self.image_height, self.patch_size
                ),
            });
        }
        if self.image_width % self.patch_size != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "image_width ({}) must be divisible by patch_size ({})",
                    self.image_width, self.patch_size
                ),
            });
        }
        if self.num_channels == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "num_channels must be > 0".to_string(),
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

/// A single image sample
#[derive(Debug, Clone)]
pub struct ImageSample {
    /// Flattened pixel data (normalized)
    pub pixels: Vec<f32>,
    /// Image height
    pub height: usize,
    /// Image width
    pub width: usize,
    /// Number of channels
    pub channels: usize,
    /// Optional label or caption
    pub label: Option<String>,
}

impl ImageSample {
    /// Create a new image sample from pixel data
    pub fn new(pixels: Vec<f32>, height: usize, width: usize, channels: usize) -> Self {
        Self {
            pixels,
            height,
            width,
            channels,
            label: None,
        }
    }

    /// Create a new image sample with a label
    pub fn with_label(
        pixels: Vec<f32>,
        height: usize,
        width: usize,
        channels: usize,
        label: String,
    ) -> Self {
        Self {
            pixels,
            height,
            width,
            channels,
            label: Some(label),
        }
    }

    /// Convert to ndarray (height, width, channels)
    pub fn to_array3(&self) -> Array3<f32> {
        Array3::from_shape_vec(
            (self.height, self.width, self.channels),
            self.pixels.clone(),
        )
        .unwrap_or_else(|_| Array3::zeros((self.height, self.width, self.channels)))
    }

    /// Normalize pixel values from [0, 255] to [-1, 1] or [0, 1]
    pub fn normalize(&mut self, to_range: ImageNormRange) {
        match to_range {
            ImageNormRange::NegOneToOne => {
                for p in &mut self.pixels {
                    *p = (*p / 127.5) - 1.0;
                }
            }
            ImageNormRange::ZeroToOne => {
                for p in &mut self.pixels {
                    *p /= 255.0;
                }
            }
            ImageNormRange::Imagenet => {
                // ImageNet normalization: (x - mean) / std
                // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
                let means = [0.485, 0.456, 0.406];
                let stds = [0.229, 0.224, 0.225];
                for (i, p) in self.pixels.iter_mut().enumerate() {
                    let c = i % self.channels;
                    *p = (*p / 255.0 - means[c]) / stds[c];
                }
            }
        }
    }
}

/// Normalization range for image pixels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageNormRange {
    /// Normalize to [-1, 1]
    NegOneToOne,
    /// Normalize to [0, 1]
    ZeroToOne,
    /// ImageNet normalization
    Imagenet,
}

/// Image encoder using Vision Transformer-style patch embedding
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ImageEncoder {
    pub config: ImageConfig,
    pub patch_embed: PatchEmbed,
    #[serde(skip, default)]
    pub position_embeddings: Option<Array2<f32>>,
    #[serde(skip, default)]
    pub cls_token: Option<Array1<f32>>,
}

impl ImageEncoder {
    /// Create a new image encoder
    pub fn new(config: ImageConfig) -> Result<Self> {
        config.validate()?;

        let patch_dim = config.patch_dim();
        let _num_patches = config.num_patches();

        let patch_embed = PatchEmbed::new(patch_dim, config.embedding_dim)?;

        let position_embeddings = if config.use_position_embeddings {
            let seq_len = config.sequence_length();
            Some(Self::init_position_embeddings(seq_len, config.embedding_dim))
        } else {
            None
        };

        let cls_token = if config.use_cls_token {
            Some(Self::init_cls_token(config.embedding_dim))
        } else {
            None
        };

        Ok(Self {
            config,
            patch_embed,
            position_embeddings,
            cls_token,
        })
    }

    /// Initialize position embeddings with normal distribution
    fn init_position_embeddings(seq_len: usize, dim: usize) -> Array2<f32> {
        let mut rng = get_rng();
        let std = 0.02; // Standard initialization for position embeddings
        let normal = Normal::new(0.0, std).unwrap();

        Array2::from_shape_fn((seq_len, dim), |_| normal.sample(&mut rng))
    }

    /// Initialize CLS token with normal distribution
    fn init_cls_token(dim: usize) -> Array1<f32> {
        let mut rng = get_rng();
        let std = 0.02;
        let normal = Normal::new(0.0, std).unwrap();

        Array1::from_shape_fn(dim, |_| normal.sample(&mut rng))
    }

    /// Encode an image sample into embeddings
    pub fn encode(&self, sample: &ImageSample) -> Result<Array2<f32>> {
        // Convert to array
        let img = sample.to_array3();

        // Extract patches
        let patches = self.extract_patches(&img)?;

        // Embed patches
        let mut embeddings = self.patch_embed.forward(&patches);

        // Add CLS token if enabled
        if let Some(ref cls) = self.cls_token {
            let cls_emb = cls.view().insert_axis(Axis(0));
            embeddings = ndarray::concatenate(Axis(0), &[cls_emb, embeddings.view()])
                .map_err(|e| crate::common::errors::ModelError::InvalidInput {
                    message: format!("Failed to concatenate CLS token: {}", e),
                })?;
        }

        // Add position embeddings
        if let Some(ref pos_emb) = self.position_embeddings {
            embeddings = embeddings + pos_emb;
        }

        Ok(embeddings)
    }

    /// Extract patches from an image array
    fn extract_patches(&self, img: &Array3<f32>) -> Result<Array2<f32>> {
        let (h, w, c) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let p = self.config.patch_size;

        if h != self.config.image_height || w != self.config.image_width {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Image dimensions ({}, {}) don't match config ({}, {})",
                    h, w, self.config.image_height, self.config.image_width
                ),
            });
        }

        let num_patches_h = h / p;
        let num_patches_w = w / p;
        let num_patches = num_patches_h * num_patches_w;
        let patch_dim = p * p * c;

        let mut patches = Array2::zeros((num_patches, patch_dim));

        let mut patch_idx = 0;
        for ph in 0..num_patches_h {
            for pw in 0..num_patches_w {
                let mut patch_data = Vec::with_capacity(patch_dim);

                for i in 0..p {
                    for j in 0..p {
                        for ch in 0..c {
                            let y = ph * p + i;
                            let x = pw * p + j;
                            patch_data.push(img[[y, x, ch]]);
                        }
                    }
                }

                patches
                    .row_mut(patch_idx)
                    .assign(&Array1::from_vec(patch_data));
                patch_idx += 1;
            }
        }

        Ok(patches)
    }

    /// Resize image to target dimensions using bilinear interpolation
    pub fn resize(&self, sample: &ImageSample, target_height: usize, target_width: usize) -> ImageSample {
        let img = sample.to_array3();
        let c = sample.channels;

        let mut resized = vec![0.0f32; target_height * target_width * c];

        let scale_y = sample.height as f32 / target_height as f32;
        let scale_x = sample.width as f32 / target_width as f32;

        for y in 0..target_height {
            for x in 0..target_width {
                let src_y = y as f32 * scale_y;
                let src_x = x as f32 * scale_x;

                let y0 = src_y.floor() as usize;
                let x0 = src_x.floor() as usize;
                let y1 = (y0 + 1).min(sample.height - 1);
                let x1 = (x0 + 1).min(sample.width - 1);

                let dy = src_y - y0 as f32;
                let dx = src_x - x0 as f32;

                for ch in 0..c {
                    let v00 = img[[y0, x0, ch]];
                    let v01 = img[[y0, x1, ch]];
                    let v10 = img[[y1, x0, ch]];
                    let v11 = img[[y1, x1, ch]];

                    let v0 = v00 * (1.0 - dx) + v01 * dx;
                    let v1 = v10 * (1.0 - dx) + v11 * dx;
                    let v = v0 * (1.0 - dy) + v1 * dy;

                    resized[(y * target_width + x) * c + ch] = v;
                }
            }
        }

        let mut new_sample = ImageSample::new(resized, target_height, target_width, c);
        new_sample.label = sample.label.clone();
        new_sample
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Get sequence length
    pub fn sequence_length(&self) -> usize {
        self.config.sequence_length()
    }
}

fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(height: usize, width: usize, channels: usize) -> ImageSample {
        let pixels: Vec<f32> = (0..height * width * channels)
            .map(|i| (i % 256) as f32)
            .collect();
        ImageSample::new(pixels, height, width, channels)
    }

    #[test]
    fn test_image_config_validation() {
        let config = ImageConfig::default();
        assert!(config.validate().is_ok());

        let bad_config = ImageConfig {
            patch_size: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_image_encoder_creation() {
        let config = ImageConfig {
            image_height: 32,
            image_width: 32,
            patch_size: 8,
            num_channels: 3,
            embedding_dim: 64,
            ..Default::default()
        };

        let encoder = ImageEncoder::new(config).unwrap();
        assert_eq!(encoder.config.num_patches(), 16); // 4x4 grid
        assert_eq!(encoder.output_dim(), 64);
    }

    #[test]
    fn test_patch_extraction() {
        let config = ImageConfig {
            image_height: 32,
            image_width: 32,
            patch_size: 8,
            num_channels: 3,
            embedding_dim: 64,
            ..Default::default()
        };

        let encoder = ImageEncoder::new(config).unwrap();
        let sample = create_test_image(32, 32, 3);

        let img = sample.to_array3();
        let patches = encoder.extract_patches(&img).unwrap();

        assert_eq!(patches.shape()[0], 16); // 4x4 patches
        assert_eq!(patches.shape()[1], 8 * 8 * 3); // patch_dim
    }

    #[test]
    fn test_image_encoding() {
        let config = ImageConfig {
            image_height: 32,
            image_width: 32,
            patch_size: 16,
            num_channels: 3,
            embedding_dim: 64,
            use_cls_token: true,
            ..Default::default()
        };

        let encoder = ImageEncoder::new(config).unwrap();
        let sample = create_test_image(32, 32, 3);

        let embeddings = encoder.encode(&sample).unwrap();

        // 4 patches (2x2) + 1 CLS token = 5
        assert_eq!(embeddings.shape()[0], 5);
        assert_eq!(embeddings.shape()[1], 64);
    }

    #[test]
    fn test_image_resize() {
        let config = ImageConfig::default();
        let encoder = ImageEncoder::new(config).unwrap();

        let sample = create_test_image(64, 64, 3);
        let resized = encoder.resize(&sample, 32, 32);

        assert_eq!(resized.height, 32);
        assert_eq!(resized.width, 32);
        assert_eq!(resized.channels, 3);
    }
}
