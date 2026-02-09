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

use crate::common::{errors::Result, rng::{get_rng, DeterministicRng}};
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

/// Configuration for image data augmentation during training
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageAugmentationConfig {
    /// Probability of applying horizontal flip
    pub horizontal_flip_prob: f32,
    /// Probability of applying vertical flip
    pub vertical_flip_prob: f32,
    /// Random crop scale range (min, max)
    pub random_crop_scale: (f32, f32),
    /// Random crop ratio range (width/height)
    pub random_crop_ratio: (f32, f32),
    /// Brightness jitter factor (0 = no jitter)
    pub brightness_jitter: f32,
    /// Contrast jitter factor
    pub contrast_jitter: f32,
    /// Saturation jitter factor
    pub saturation_jitter: f32,
    /// Hue jitter factor
    pub hue_jitter: f32,
    /// Gaussian noise standard deviation
    pub noise_std: f32,
    /// Cutout probability
    pub cutout_prob: f32,
    /// Cutout size as fraction of image
    pub cutout_size: f32,
    /// Random rotation max angle in degrees
    pub max_rotation_degrees: f32,
}

impl Default for ImageAugmentationConfig {
    fn default() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.0,
            random_crop_scale: (0.8, 1.0),
            random_crop_ratio: (0.75, 1.333),
            brightness_jitter: 0.1,
            contrast_jitter: 0.1,
            saturation_jitter: 0.1,
            hue_jitter: 0.05,
            noise_std: 0.0,
            cutout_prob: 0.0,
            cutout_size: 0.2,
            max_rotation_degrees: 0.0,
        }
    }
}

impl ImageAugmentationConfig {
    /// No augmentation
    pub fn none() -> Self {
        Self {
            horizontal_flip_prob: 0.0,
            vertical_flip_prob: 0.0,
            random_crop_scale: (1.0, 1.0),
            random_crop_ratio: (1.0, 1.0),
            brightness_jitter: 0.0,
            contrast_jitter: 0.0,
            saturation_jitter: 0.0,
            hue_jitter: 0.0,
            noise_std: 0.0,
            cutout_prob: 0.0,
            cutout_size: 0.0,
            max_rotation_degrees: 0.0,
        }
    }

    /// Light augmentation for fine-tuning
    pub fn light() -> Self {
        Self {
            horizontal_flip_prob: 0.3,
            vertical_flip_prob: 0.0,
            random_crop_scale: (0.9, 1.0),
            random_crop_ratio: (0.9, 1.1),
            brightness_jitter: 0.05,
            contrast_jitter: 0.05,
            saturation_jitter: 0.05,
            hue_jitter: 0.02,
            noise_std: 0.0,
            cutout_prob: 0.0,
            cutout_size: 0.1,
            max_rotation_degrees: 5.0,
        }
    }

    /// Strong augmentation for training from scratch
    pub fn strong() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.1,
            random_crop_scale: (0.7, 1.0),
            random_crop_ratio: (0.75, 1.333),
            brightness_jitter: 0.2,
            contrast_jitter: 0.2,
            saturation_jitter: 0.2,
            hue_jitter: 0.1,
            noise_std: 0.02,
            cutout_prob: 0.3,
            cutout_size: 0.25,
            max_rotation_degrees: 15.0,
        }
    }
}

/// Image data augmentation transformer
#[derive(Clone)]
pub struct ImageAugmentation {
    config: ImageAugmentationConfig,
    rng: crate::common::rng::DeterministicRng,
}

impl ImageAugmentation {
    /// Create a new augmentation transformer
    pub fn new(config: ImageAugmentationConfig) -> Self {
        Self {
            config,
            rng: get_rng(),
        }
    }

    /// Apply augmentation to an image sample
    pub fn augment(&mut self, sample: &ImageSample) -> ImageSample {
        use rand::Rng;

        let mut pixels = sample.pixels.clone();
        let h = sample.height;
        let w = sample.width;
        let c = sample.channels;

        // Horizontal flip
        if self.rng.random::<f32>() < self.config.horizontal_flip_prob {
            pixels = self.horizontal_flip(&pixels, h, w, c);
        }

        // Vertical flip
        if self.rng.random::<f32>() < self.config.vertical_flip_prob {
            pixels = self.vertical_flip(&pixels, h, w, c);
        }

        // Color jitter
        if self.config.brightness_jitter > 0.0
            || self.config.contrast_jitter > 0.0
            || self.config.saturation_jitter > 0.0
        {
            pixels = self.color_jitter(&pixels, h, w, c);
        }

        // Gaussian noise
        if self.config.noise_std > 0.0 {
            pixels = self.add_noise(&pixels, self.config.noise_std);
        }

        // Cutout
        if self.rng.random::<f32>() < self.config.cutout_prob {
            pixels = self.cutout(&pixels, h, w, c);
        }

        let mut result = ImageSample::new(pixels, h, w, c);
        result.label = sample.label.clone();
        result
    }

    fn horizontal_flip(&self, pixels: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut flipped = vec![0.0f32; pixels.len()];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    let src_idx = (y * w + x) * c + ch;
                    let dst_idx = (y * w + (w - 1 - x)) * c + ch;
                    flipped[dst_idx] = pixels[src_idx];
                }
            }
        }
        flipped
    }

    fn vertical_flip(&self, pixels: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        let mut flipped = vec![0.0f32; pixels.len()];
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    let src_idx = (y * w + x) * c + ch;
                    let dst_idx = ((h - 1 - y) * w + x) * c + ch;
                    flipped[dst_idx] = pixels[src_idx];
                }
            }
        }
        flipped
    }

    fn color_jitter(&mut self, pixels: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        use rand::Rng;

        let mut result = pixels.to_vec();

        // Brightness
        if self.config.brightness_jitter > 0.0 {
            let delta = self.rng.random_range(
                -self.config.brightness_jitter..self.config.brightness_jitter
            );
            for pixel in &mut result {
                *pixel = (*pixel + delta * 255.0).clamp(0.0, 255.0);
            }
        }

        // Contrast
        if self.config.contrast_jitter > 0.0 {
            let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
            let factor = self.rng.random_range(
                1.0 - self.config.contrast_jitter..1.0 + self.config.contrast_jitter
            );
            for pixel in &mut result {
                *pixel = (mean + (*pixel - mean) * factor).clamp(0.0, 255.0);
            }
        }

        // Saturation (for RGB images)
        if c == 3 && self.config.saturation_jitter > 0.0 {
            let factor = self.rng.random_range(
                1.0 - self.config.saturation_jitter..1.0 + self.config.saturation_jitter
            );
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * c;
                    let r = result[idx];
                    let g = result[idx + 1];
                    let b = result[idx + 2];
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    result[idx] = (gray + (r - gray) * factor).clamp(0.0, 255.0);
                    result[idx + 1] = (gray + (g - gray) * factor).clamp(0.0, 255.0);
                    result[idx + 2] = (gray + (b - gray) * factor).clamp(0.0, 255.0);
                }
            }
        }

        result
    }

    fn add_noise(&mut self, pixels: &[f32], std: f32) -> Vec<f32> {
        let normal = Normal::new(0.0, std).unwrap();
        pixels
            .iter()
            .map(|&p| {
                let noise: f32 = normal.sample(&mut self.rng);
                (p + noise * 255.0).clamp(0.0, 255.0)
            })
            .collect()
    }

    fn cutout(&mut self, pixels: &[f32], h: usize, w: usize, c: usize) -> Vec<f32> {
        use rand::Rng;

        let mut result = pixels.to_vec();
        let cutout_h = (h as f32 * self.config.cutout_size) as usize;
        let cutout_w = (w as f32 * self.config.cutout_size) as usize;

        let start_y = self.rng.gen_range(0..h.saturating_sub(cutout_h).max(1));
        let start_x = self.rng.gen_range(0..w.saturating_sub(cutout_w).max(1));

        for y in start_y..(start_y + cutout_h).min(h) {
            for x in start_x..(start_x + cutout_w).min(w) {
                for ch in 0..c {
                    let idx = (y * w + x) * c + ch;
                    result[idx] = 0.0; // Set to black
                }
            }
        }

        result
    }
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

    #[test]
    fn test_image_augmentation_config_presets() {
        let none = ImageAugmentationConfig::none();
        assert_eq!(none.horizontal_flip_prob, 0.0);
        assert_eq!(none.brightness_jitter, 0.0);

        let light = ImageAugmentationConfig::light();
        assert!(light.horizontal_flip_prob > 0.0);
        assert!(light.horizontal_flip_prob < 0.5);

        let strong = ImageAugmentationConfig::strong();
        assert_eq!(strong.horizontal_flip_prob, 0.5);
        assert!(strong.cutout_prob > 0.0);
    }

    #[test]
    fn test_image_augmentation_horizontal_flip() {
        let config = ImageAugmentationConfig {
            horizontal_flip_prob: 1.0, // Always flip
            ..ImageAugmentationConfig::none()
        };
        let mut aug = ImageAugmentation::new(config);

        // Create image with distinct left/right halves
        let mut pixels = vec![0.0f32; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..2 {
                for c in 0..3 {
                    pixels[(y * 4 + x) * 3 + c] = 100.0; // Left half = 100
                }
            }
            for x in 2..4 {
                for c in 0..3 {
                    pixels[(y * 4 + x) * 3 + c] = 200.0; // Right half = 200
                }
            }
        }

        let sample = ImageSample::new(pixels, 4, 4, 3);
        let flipped = aug.augment(&sample);

        // After flip, left half should be 200 and right half should be 100
        assert_eq!(flipped.pixels[(0 * 4 + 0) * 3], 200.0);
        assert_eq!(flipped.pixels[(0 * 4 + 3) * 3], 100.0);
    }

    #[test]
    fn test_image_augmentation_no_augmentation() {
        let config = ImageAugmentationConfig::none();
        let mut aug = ImageAugmentation::new(config);

        let sample = create_test_image(32, 32, 3);
        let augmented = aug.augment(&sample);

        // With no augmentation, pixels should be identical
        assert_eq!(sample.pixels, augmented.pixels);
    }

    #[test]
    fn test_image_augmentation_cutout() {
        let config = ImageAugmentationConfig {
            cutout_prob: 1.0, // Always apply cutout
            cutout_size: 0.5, // 50% of image
            ..ImageAugmentationConfig::none()
        };
        let mut aug = ImageAugmentation::new(config);

        let sample = ImageSample::new(vec![255.0; 8 * 8 * 3], 8, 8, 3);
        let augmented = aug.augment(&sample);

        // Some pixels should be zeroed out
        let has_zeros = augmented.pixels.iter().any(|&p| p == 0.0);
        assert!(has_zeros);
    }
}
