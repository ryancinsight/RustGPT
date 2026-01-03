//! Shared Adaptive Residuals Component
//!
//! This component provides advanced adaptive residual connections that can be used
//! by multiple architectures (Transformer, Diffusion, SSM). It implements the
//! similarity-based residual scaling described in the adaptive residuals research.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::adam::Adam;

/// Configuration for adaptive residuals
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdaptiveResidualConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Similarity update rate for EMA
    pub similarity_update_rate: f32,
    /// Residual stability threshold
    pub residual_stability_threshold: f32,
    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,
}

impl Default for AdaptiveResidualConfig {
    fn default() -> Self {
        Self {
            embed_dim: 128,
            similarity_update_rate: 0.01,
            residual_stability_threshold: 0.1,
            max_seq_len: 256,
        }
    }
}

/// Adaptive residuals component
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AdaptiveResiduals {
    /// Compact weight similarity matrix (embed_dim × embed_dim)
    pub weight_similarity_matrix: Array2<f32>,

    /// Activation-derived similarity representation (embed_dim × embed_dim)
    /// Tracks how input channels align with output channels, bounded in [-1, 1]
    pub activation_similarity_matrix: Array2<f32>,

    /// Per-channel affinity scores (embed_dim × 1)
    pub layer_affinity_scores: Array2<f32>,

    /// Adaptive residual scaling for attention paths (embed_dim × 1)
    pub attention_residual_scales: Array2<f32>,

    /// Adaptive residual scaling for FFN paths (embed_dim × 1)
    pub ffn_residual_scales: Array2<f32>,

    /// Position-aware attention parameters for CoPE-based residual scaling
    pub positional_residual_qkv: Array2<f32>,

    /// CoPE positional embeddings (learned relative position encodings)
    pub cope_pos_embeddings: Array2<f32>,

    /// Position-aware residual scaling weights
    pub positional_residual_weights: Array2<f32>,

    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,

    /// Optimizers for all parameters
    opt_similarity: Adam,
    opt_affinity: Adam,
    opt_scales_attention: Adam,
    opt_scales_ffn: Adam,
    opt_positional_qkv: Adam,
    opt_cope_pos: Adam,
    opt_positional_weights: Adam,

    /// Configuration
    config: AdaptiveResidualConfig,

    /// Runtime statistics
    similarity_entropy: f32,
    residual_variance: f32,
    gradient_norm: f32,
}

impl AdaptiveResiduals {
    /// Create a new adaptive residuals component with full configuration
    pub fn new(config: AdaptiveResidualConfig) -> Self {
        let embed_dim = config.embed_dim;
        let max_seq_len = config.max_seq_len;

        // Initialize matrices with appropriate dimensions
        let weight_similarity = Array2::zeros((embed_dim, embed_dim));
        let activation_similarity = Array2::zeros((embed_dim, embed_dim));
        let affinity_scores = Array2::zeros((embed_dim, 1));
        let attn_scales = Array2::ones((embed_dim, 1));
        let ffn_scales = Array2::ones((embed_dim, 1));
        let pos_qkv = Array2::zeros((embed_dim, 3 * embed_dim));
        let cope_pos = Array2::zeros((max_seq_len, embed_dim));
        let pos_weights = Array2::zeros((embed_dim, embed_dim));

        // Initialize optimizers
        let opt_similarity = Adam::new((embed_dim, embed_dim));
        let opt_affinity = Adam::new((embed_dim, 1));
        let opt_scales_attention = Adam::new((embed_dim, 1));
        let opt_scales_ffn = Adam::new((embed_dim, 1));
        let opt_positional_qkv = Adam::new((embed_dim, 3 * embed_dim));
        let opt_cope_pos = Adam::new((max_seq_len, embed_dim));
        let opt_positional_weights = Adam::new((embed_dim, embed_dim));

        Self {
            weight_similarity_matrix: weight_similarity,
            activation_similarity_matrix: activation_similarity,
            layer_affinity_scores: affinity_scores,
            attention_residual_scales: attn_scales,
            ffn_residual_scales: ffn_scales,
            positional_residual_qkv: pos_qkv,
            cope_pos_embeddings: cope_pos,
            positional_residual_weights: pos_weights,
            max_seq_len: max_seq_len,
            opt_similarity,
            opt_affinity,
            opt_scales_attention,
            opt_scales_ffn,
            opt_positional_qkv,
            opt_cope_pos,
            opt_positional_weights,
            config,
            similarity_entropy: 0.0,
            residual_variance: 0.0,
            gradient_norm: 0.0,
        }
    }

    /// Create a new adaptive residuals component with minimal configuration
    pub fn new_minimal(embed_dim: usize) -> Self {
        let config = AdaptiveResidualConfig {
            embed_dim,
            similarity_update_rate: 0.01,
            residual_stability_threshold: 0.1,
            max_seq_len: 256, // Default value
        };
        Self::new(config)
    }

    /// Apply adaptive residual connection after attention
    pub fn apply_attention_residual(
        &mut self,
        input: &Array2<f32>,
        attn_out: &Array2<f32>,
    ) -> Array2<f32> {
        // Update similarity matrices
        self.update_similarity_matrices(input, attn_out);

        // Compute adaptive residual scaling
        let attn_scales = &self.attention_residual_scales;

        // Apply position-aware scaling
        let mut output = attn_out.clone();
        
        // Apply channel-wise scaling
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                let scale = attn_scales[[j, 0]];
                output[[i, j]] *= scale;
            }
        }

        // Add residual
        output += input;
        
        output
    }

    /// Apply adaptive residual connection after feedforward
    pub fn apply_ffn_residual(
        &mut self,
        residual1: &Array2<f32>,
        ffn_out: &Array2<f32>,
    ) -> Array2<f32> {
        // Update similarity matrices
        self.update_similarity_matrices(residual1, ffn_out);

        // Compute adaptive residual scaling
        let ffn_scales = &self.ffn_residual_scales;

        // Apply position-aware scaling
        let mut output = ffn_out.clone();
        
        // Apply channel-wise scaling
        for i in 0..output.nrows() {
            for j in 0..output.ncols() {
                let scale = ffn_scales[[j, 0]];
                output[[i, j]] *= scale;
            }
        }

        // Add residual
        output += residual1;
        
        output
    }

    /// Update similarity matrices based on input and output
    fn update_similarity_matrices(&mut self, input: &Array2<f32>, output: &Array2<f32>) {
        let rate = self.config.similarity_update_rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return;
        }

        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input.ncols().min(output.ncols()).min(self.config.embed_dim);
        if seq_len == 0 || embed_dim == 0 {
            return;
        }

        let sample = seq_len.min(32);
        let step = (seq_len / sample).max(1);

        // Compute channel-to-channel cosine similarity with EMA update
        let mut nx = vec![0.0f64; embed_dim];
        let mut ny = vec![0.0f64; embed_dim];
        
        // Compute norms
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

        // Update similarity matrix
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
                self.activation_similarity_matrix[[i, j]] = 
                    rate * cosine + (1.0 - rate) * current;
            }
        }

        // Update statistics
        self.update_statistics();
    }

    /// Update runtime statistics
    fn update_statistics(&mut self) {
        // Compute similarity entropy
        let mut entropy = 0.0f32;
        let mut count = 0;
        for &val in self.activation_similarity_matrix.iter() {
            if val.abs() > 1e-6 {
                let p = (val + 1.0) * 0.5; // Map [-1,1] to [0,1]
                entropy -= p * p.ln() + (1.0 - p) * (1.0 - p).ln();
                count += 1;
            }
        }
        if count > 0 {
            self.similarity_entropy = entropy / count as f32;
        }

        // Compute residual variance
        let mut variance = 0.0;
        let mut mean = 0.0;
        let mut n = 0;
        for &val in self.attention_residual_scales.iter() {
            mean += val;
            n += 1;
        }
        if n > 0 {
            mean /= n as f32;
            for &val in self.attention_residual_scales.iter() {
                variance += (val - mean) * (val - mean);
            }
            variance /= n as f32;
            self.residual_variance = variance;
        }
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let mut count = 0;
        count += self.attention_residual_scales.len();
        count += self.ffn_residual_scales.len();
        count += self.positional_residual_qkv.len();
        count += self.cope_pos_embeddings.len();
        count += self.positional_residual_weights.len();
        count
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> (f32, f32, f32) {
        (self.similarity_entropy, self.residual_variance, self.gradient_norm)
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.similarity_entropy = 0.0;
        self.residual_variance = 0.0;
        self.gradient_norm = 0.0;
    }

    /// Get activation similarity matrix
    pub fn activation_similarity_matrix(&self) -> &Array2<f32> {
        &self.activation_similarity_matrix
    }

    /// Get attention residual scales
    pub fn attention_residual_scales(&self) -> &Array2<f32> {
        &self.attention_residual_scales
    }

    /// Get FFN residual scales
    pub fn ffn_residual_scales(&self) -> &Array2<f32> {
        &self.ffn_residual_scales
    }

    /// Get residual stability threshold from config
    pub fn residual_stability_threshold(&self) -> f32 {
        self.config.residual_stability_threshold
    }

    /// Calculate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut bytes = 0;
        bytes += self.weight_similarity_matrix.len() * std::mem::size_of::<f32>();
        bytes += self.activation_similarity_matrix.len() * std::mem::size_of::<f32>();
        bytes += self.layer_affinity_scores.len() * std::mem::size_of::<f32>();
        bytes += self.attention_residual_scales.len() * std::mem::size_of::<f32>();
        bytes += self.ffn_residual_scales.len() * std::mem::size_of::<f32>();
        bytes += self.positional_residual_qkv.len() * std::mem::size_of::<f32>();
        bytes += self.cope_pos_embeddings.len() * std::mem::size_of::<f32>();
        bytes += self.positional_residual_weights.len() * std::mem::size_of::<f32>();
        bytes
    }

    /// Invalidate similarity cache (no-op for this implementation)
    pub fn invalidate_similarity_cache(&mut self) {
        // This implementation doesn't have a separate cache, so this is a no-op
    }

    /// Compute batch similarity matrix (placeholder implementation)
    pub fn compute_batch_similarity_matrix(&mut self, _attention_weights: &Array2<f32>, _ffn_weights: &Array2<f32>) -> &Array2<f32> {
        // Placeholder - return the current activation similarity matrix
        // In a full implementation, this would update based on batch data
        &self.activation_similarity_matrix
    }

    /// Compute gradients for adaptive residuals (placeholder)
    pub fn compute_gradients(&self, _input: &Array2<f32>, _attn_out: &Array2<f32>, _ffn_out: &Array2<f32>, _residual_grads: &Array2<f32>) -> Vec<Array2<f32>> {
        // Placeholder implementation - return zero gradients
        vec![
            Array2::zeros(self.weight_similarity_matrix.raw_dim()),
            Array2::zeros(self.layer_affinity_scores.raw_dim()),
            Array2::zeros(self.positional_residual_qkv.raw_dim()),
            Array2::zeros(self.positional_residual_weights.raw_dim()),
        ]
    }

    /// Apply gradients to adaptive residuals (placeholder)
    pub fn apply_gradients(&mut self, _param_grads: &[Array2<f32>], _lr: f32) -> crate::errors::Result<()> {
        // Placeholder implementation - would apply gradients to parameters
        Ok(())
    }
}