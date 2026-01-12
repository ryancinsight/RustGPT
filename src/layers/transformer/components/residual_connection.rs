//! Residual Connection Component
//!
//! Handles residual connections in transformer blocks.
//! Provides efficient in-place residual addition and similarity context application.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Residual connection component
#[derive(Serialize, Deserialize, Debug)]
pub struct ResidualConnection {
    /// Similarity context strength for attention-based residual mixing
    similarity_context_strength: Array2<f32>,
    /// Similarity update rate for EMA updates
    similarity_update_rate: f32,
    /// Current activation similarity matrix
    activation_similarity_matrix: Array2<f32>,
}

impl ResidualConnection {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            similarity_context_strength: Array2::zeros((1, 1)),
            similarity_update_rate: 0.01,
            activation_similarity_matrix: Array2::zeros((embed_dim, embed_dim)),
        }
    }

    /// Apply similarity context to input
    pub fn apply_similarity_context(
        &self,
        input: &Array2<f32>,
        context: &Array2<f32>,
    ) -> Array2<f32> {
        let strength = self.similarity_context_strength[[0, 0]];
        let embed_dim = input.ncols();

        if strength == 0.0 || embed_dim == 0 {
            return input.clone();
        }

        let mut result = input.clone();
        let scale = strength / embed_dim as f32;

        // Apply context mixing: X' = X + (strength / embed_dim) * X·S
        for i in 0..input.nrows() {
            for j in 0..embed_dim {
                let mut sum = 0.0;
                for k in 0..embed_dim {
                    sum += input[[i, k]] * context[[k, j]];
                }
                result[[i, j]] += scale * sum;
            }
        }

        result
    }

    /// Update activation similarity matrix
    pub fn update_activation_similarity_matrix(
        &mut self,
        input: &Array2<f32>,
        output: &Array2<f32>,
    ) {
        let rate = self.similarity_update_rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return;
        }

        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input
            .ncols()
            .min(output.ncols())
            .min(self.activation_similarity_matrix.ncols());
        if seq_len == 0 || embed_dim == 0 {
            return;
        }

        let sample = seq_len.min(32);
        let step = (seq_len / sample).max(1);

        let mut nx = vec![0.0f64; embed_dim];
        let mut ny = vec![0.0f64; embed_dim];

        // Compute norms for normalization
        for seq_idx in (0..seq_len).step_by(step).take(sample) {
            for j in 0..embed_dim {
                let x = input[[seq_idx, j]];
                let y = output[[seq_idx, j]];
                let xs = if x.is_finite() { x as f64 } else { 0.0 };
                let ys = if y.is_finite() { y as f64 } else { 0.0 };
                nx[j] += xs * xs;
                ny[j] += ys * ys;
            }
        }

        // Update similarity matrix with EMA
        for i in 0..embed_dim {
            for j in 0..embed_dim {
                let mut dot = 0.0f64;
                for seq_idx in (0..seq_len).step_by(step).take(sample) {
                    let x = input[[seq_idx, i]];
                    let y = output[[seq_idx, j]];
                    let xs = if x.is_finite() { x as f64 } else { 0.0 };
                    let ys = if y.is_finite() { y as f64 } else { 0.0 };
                    dot += xs * ys;
                }

                let denom_x = (nx[i] + 1e-6).sqrt();
                let denom_y = (ny[j] + 1e-6).sqrt();
                let cosine = (dot / (denom_x * denom_y + 1e-6)) as f32;

                // EMA update
                let current = self.activation_similarity_matrix[[i, j]];
                self.activation_similarity_matrix[[i, j]] = rate * cosine + (1.0 - rate) * current;
            }
        }
    }

    /// Perform in-place residual addition
    pub fn add_residual_inplace(output: &mut Array2<f32>, residual: &Array2<f32>) {
        *output += residual;
    }

    /// Get the activation similarity matrix
    pub fn activation_similarity_matrix(&self) -> &Array2<f32> {
        &self.activation_similarity_matrix
    }

    /// Set similarity context strength
    pub fn set_similarity_context_strength(&mut self, strength: f32) {
        self.similarity_context_strength[[0, 0]] = strength;
    }

    /// Get similarity context strength
    pub fn similarity_context_strength(&self) -> f32 {
        self.similarity_context_strength[[0, 0]]
    }
}
