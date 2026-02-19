//! Patch embedding and merging utilities for multi-modal processing.
//!
//! Provides linear projection from patch pixels to embeddings,
//! used by both image and video encoders.

use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::common::{errors::Result, rng::get_rng};

/// Linear embedding from patch pixels to embedding dimension
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PatchEmbed {
    /// Weight matrix (patch_dim x embedding_dim)
    pub weight: Array2<f32>,
    /// Bias vector (embedding_dim)
    pub bias: Array1<f32>,
    /// Input dimension (flattened patch size)
    pub in_features: usize,
    /// Output dimension (embedding dimension)
    pub out_features: usize,
}

impl PatchEmbed {
    /// Create a new patch embedding layer
    pub fn new(in_features: usize, out_features: usize) -> Result<Self> {
        if in_features == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "in_features must be > 0".to_string(),
            });
        }
        if out_features == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "out_features must be > 0".to_string(),
            });
        }

        let (weight, bias) = Self::init_weights(in_features, out_features);

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    /// Initialize weights using truncated normal distribution
    /// Following ViT initialization: std = 0.02
    fn init_weights(in_features: usize, out_features: usize) -> (Array2<f32>, Array1<f32>) {
        let mut rng = get_rng();
        let std = 0.02;
        let normal = Normal::new(0.0, std).unwrap();

        let weight = Array2::from_shape_fn((in_features, out_features), |_| {
            // Truncated normal: reject values outside [-2*std, 2*std]
            loop {
                let v: f32 = normal.sample(&mut rng);
                if v.abs() <= 2.0 * std {
                    return v;
                }
            }
        });

        let bias = Array1::zeros(out_features);

        (weight, bias)
    }

    /// Forward pass: project patches to embeddings
    pub fn forward(&self, patches: &Array2<f32>) -> Array2<f32> {
        // patches: (num_patches, patch_dim)
        // weight: (patch_dim, embedding_dim)
        // output: (num_patches, embedding_dim)
        patches.dot(&self.weight) + &self.bias
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

/// Patch merging layer for hierarchical architectures
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PatchMerge {
    /// Weight matrix for merging
    pub weight: Array2<f32>,
    /// Bias vector
    pub bias: Array1<f32>,
    /// Merge factor (e.g., 2 means 2x2 patches merged into 1)
    pub merge_factor: usize,
    /// Input dimension per patch
    pub in_features: usize,
    /// Output dimension per merged patch
    pub out_features: usize,
}

impl PatchMerge {
    /// Create a new patch merging layer
    pub fn new(merge_factor: usize, in_features: usize, out_features: usize) -> Result<Self> {
        if merge_factor == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "merge_factor must be > 0".to_string(),
            });
        }
        if in_features == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "in_features must be > 0".to_string(),
            });
        }
        if out_features == 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: "out_features must be > 0".to_string(),
            });
        }

        let (weight, bias) =
            PatchEmbed::init_weights(in_features * merge_factor * merge_factor, out_features);

        Ok(Self {
            weight,
            bias,
            merge_factor,
            in_features,
            out_features,
        })
    }

    /// Forward pass: merge adjacent patches
    pub fn forward(
        &self,
        patches: &Array2<f32>,
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Array2<f32>> {
        let num_patches = patches.nrows();
        let expected_patches = grid_h * grid_w;

        if num_patches != expected_patches {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Expected {} patches for grid {}x{}, got {}",
                    expected_patches, grid_h, grid_w, num_patches
                ),
            });
        }

        if grid_h % self.merge_factor != 0 || grid_w % self.merge_factor != 0 {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Grid dimensions ({}x{}) must be divisible by merge_factor {}",
                    grid_h, grid_w, self.merge_factor
                ),
            });
        }

        let new_grid_h = grid_h / self.merge_factor;
        let new_grid_w = grid_w / self.merge_factor;
        let new_num_patches = new_grid_h * new_grid_w;

        let mut merged = Vec::with_capacity(new_num_patches);

        for mh in 0..new_grid_h {
            for mw in 0..new_grid_w {
                // Collect patches in the merge window
                let mut window_patches = Vec::new();

                for i in 0..self.merge_factor {
                    for j in 0..self.merge_factor {
                        let h = mh * self.merge_factor + i;
                        let w = mw * self.merge_factor + j;
                        let idx = h * grid_w + w;
                        window_patches.push(patches.row(idx).to_vec());
                    }
                }

                // Concatenate all patches in the window
                let concatenated: Vec<f32> = window_patches.into_iter().flatten().collect();
                merged.push(concatenated);
            }
        }

        // Convert to array and apply linear projection
        let merged_array = Array2::from_shape_vec(
            (
                new_num_patches,
                self.in_features * self.merge_factor * self.merge_factor,
            ),
            merged.into_iter().flatten().collect(),
        )
        .map_err(|e| crate::common::errors::ModelError::InvalidInput {
            message: format!("Failed to create merged array: {}", e),
        })?;

        Ok(merged_array.dot(&self.weight) + &self.bias)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

/// Compute 2D sinusoidal position embeddings
pub fn get_2d_sincos_pos_embed(embed_dim: usize, grid_size: usize) -> Array2<f32> {
    let grid_h = Array1::from_iter(0..grid_size);
    let grid_w = Array1::from_iter(0..grid_size);

    let mut grid = Vec::with_capacity(grid_size * grid_size);
    for h in grid_h.iter() {
        for w in grid_w.iter() {
            grid.push((*h as f32, *w as f32));
        }
    }

    let num_patches = grid.len();
    let mut pos_embed = Array2::zeros((num_patches, embed_dim));

    for (idx, (h, w)) in grid.iter().enumerate() {
        for dim in 0..embed_dim {
            let freq = 1.0 / 10000f32.powf(2.0 * (dim / 2) as f32 / embed_dim as f32);
            if dim % 2 == 0 {
                pos_embed[[idx, dim]] = (h * freq).sin();
            } else {
                pos_embed[[idx, dim]] = (w * freq).sin();
            }
        }
    }

    pos_embed
}

/// Compute 1D sinusoidal position embeddings
pub fn get_1d_sincos_pos_embed(embed_dim: usize, length: usize) -> Array2<f32> {
    let mut pos_embed = Array2::zeros((length, embed_dim));

    for pos in 0..length {
        for dim in 0..embed_dim {
            let angle = pos as f32 / 10000f32.powf(2.0 * (dim / 2) as f32 / embed_dim as f32);
            if dim % 2 == 0 {
                pos_embed[[pos, dim]] = angle.sin();
            } else {
                pos_embed[[pos, dim]] = angle.cos();
            }
        }
    }

    pos_embed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patch_embed_forward() {
        let embed = PatchEmbed::new(64, 128).unwrap();
        let patches = Array2::zeros((16, 64));

        let output = embed.forward(&patches);

        assert_eq!(output.shape()[0], 16);
        assert_eq!(output.shape()[1], 128);
    }

    #[test]
    fn test_patch_merge() {
        let merge = PatchMerge::new(2, 64, 128).unwrap();
        let patches = Array2::zeros((16, 64)); // 4x4 grid

        let output = merge.forward(&patches, 4, 4).unwrap();

        // 4x4 grid with merge_factor 2 -> 2x2 grid
        assert_eq!(output.shape()[0], 4);
        assert_eq!(output.shape()[1], 128);
    }

    #[test]
    fn test_2d_sincos_pos_embed() {
        let pos_embed = get_2d_sincos_pos_embed(128, 4);

        // 4x4 grid = 16 positions
        assert_eq!(pos_embed.shape()[0], 16);
        assert_eq!(pos_embed.shape()[1], 128);

        // Check that embeddings are not all zeros
        assert!(pos_embed.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_1d_sincos_pos_embed() {
        let pos_embed = get_1d_sincos_pos_embed(128, 16);

        assert_eq!(pos_embed.shape()[0], 16);
        assert_eq!(pos_embed.shape()[1], 128);

        // Check that embeddings vary across positions
        let first = pos_embed.row(0).to_vec();
        let last = pos_embed.row(15).to_vec();
        assert_ne!(first, last);
    }
}
