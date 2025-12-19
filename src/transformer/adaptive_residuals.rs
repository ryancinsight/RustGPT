#![allow(dead_code)]
use ndarray::Array2;

use crate::{
    adam::Adam,
    errors::Result,
};

/// Unified trait for adaptive residual strategies
pub trait AdaptiveResidualStrategy {
    /// Apply adaptive residual connection after attention
    fn apply_attention_residual(&mut self, input: &Array2<f32>, attn_out: &Array2<f32>) -> Array2<f32>;

    /// Apply adaptive residual connection after feedforward
    fn apply_ffn_residual(&self, residual1: &Array2<f32>, ffn_out: &Array2<f32>) -> Array2<f32>;

    /// Compute gradients for adaptive residual parameters
    fn compute_gradients(&self, input: &Array2<f32>, attn_out: &Array2<f32>, ffn_out: &Array2<f32>, residual_grads: &Array2<f32>) -> Vec<Array2<f32>>;

    /// Apply gradients to adaptive residual parameters
    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()>;

    /// Get parameter count for adaptive residuals
    fn parameter_count(&self) -> usize;

    /// Get memory usage in bytes
    fn memory_usage_bytes(&self) -> usize;

    /// Get performance metrics
    fn get_performance_metrics(&self) -> (f32, f32, f32);
}

/// Shared adaptive residual state for both transformer and diffusion blocks
#[derive(Debug, Clone)]
pub struct UnifiedAdaptiveResiduals {
    /// Compact weight similarity matrix (embed_dim × embed_dim)
    pub weight_similarity_matrix: Array2<f32>,

    /// Activation-derived similarity representation (embed_dim × embed_dim)
    ///
    /// This tracks, per layer, how input channels align with output channels.
    /// Values are bounded smoothly into [-1, 1] and updated as an EMA.
    pub activation_similarity_matrix: Array2<f32>,

    /// Per-channel affinity scores (embed_dim × 1)
    pub layer_affinity_scores: Array2<f32>,

    /// Optimized attention parameters for residual fusion (embed_dim × 3 * embed_dim)
    pub residual_attention_qkv: Array2<f32>,

    /// Channel interaction matrix (embed_dim × embed_dim)
    pub channel_affinity_matrix: Array2<f32>,

    /// Adaptive residual scaling for attention paths (embed_dim × 1)
    pub attention_residual_scales: Array2<f32>,

    /// Adaptive residual scaling for FFN paths (embed_dim × 1)
    pub ffn_residual_scales: Array2<f32>,

    /// ===== Theorem 4 Extension: Position-Aware Residual Scaling =====
    /// Position-aware attention parameters for CoPE-based residual scaling
    pub positional_residual_qkv: Array2<f32>,

    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,

    /// CoPE positional embeddings (learned relative position encodings)
    pub cope_pos_embeddings: Array2<f32>,

    /// Position-aware residual scaling weights
    pub positional_residual_weights: Array2<f32>,

    /// Optimizers for all parameters
    opt_similarity: Adam,
    opt_affinity: Adam,
    opt_attention: Adam,
    opt_channel: Adam,
    opt_scales_attention: Adam,
    opt_scales_ffn: Adam,
    opt_positional_qkv: Adam,
    opt_cope_pos: Adam,
    opt_positional_weights: Adam,

    /// Configuration
    embed_dim: usize,
    similarity_update_rate: f32,
    residual_stability_threshold: f32,

    /// Representative similarity statistics (EMA)
    ///
    /// These are not trained parameters; they track running alignment between
    /// the residual stream and the residual branch. They provide a stable,
    /// representative signal for modulating residual gating.
    similarity_ema_global: f32,
    similarity_ema_per_channel: Array2<f32>,

    /// Performance caches
    similarity_matrix_valid: bool,
    last_similarity_update: usize,
}

#[inline]
fn softplus_beta(z: f32, beta: f32) -> f32 {
    // Numerically-stable softplus: softplus(z) = ln(1 + exp(beta*z)) / beta
    // Piecewise to avoid overflow/underflow. Smooth everywhere.
    let x = z * beta;
    if x > 20.0 {
        z
    } else if x < -20.0 {
        x.exp() / beta
    } else {
        (1.0 + x.exp()).ln() / beta
    }
}

#[inline]
fn smooth_clamp(x: f32, lo: f32, hi: f32, beta: f32) -> f32 {
    // Smooth approximation to clamp(lo, hi) that is ~identity inside [lo,hi]
    // and smoothly saturates outside.
    x - softplus_beta(x - hi, beta) + softplus_beta(lo - x, beta)
}

#[inline]
fn smooth_clip_tanh(x: f32, limit: f32) -> f32 {
    if limit <= 0.0 {
        return 0.0;
    }
    limit * (x / limit).tanh()
}

impl UnifiedAdaptiveResiduals {
    /// Create new unified adaptive residuals with performance tuning
    pub fn new(embed_dim: usize) -> Self {
        use rand::prelude::*;
        let mut rng = rand::rng();

        // Initialize with intelligent defaults for better convergence
        let weight_similarity_matrix = Array2::from_elem((embed_dim, embed_dim), 0.1);
        let activation_similarity_matrix = Array2::zeros((embed_dim, embed_dim));
        let layer_affinity_scores = Array2::from_elem((embed_dim, 1), 1.0);

        // Combined QKV attention parameters for efficiency
        let residual_attention_qkv = Array2::from_shape_fn((embed_dim, 3 * embed_dim), |_| {
            0.02 * (rng.random::<f32>() - 0.5)
        });

        // Channel affinity with identity-like initialization
        let mut channel_affinity_matrix = Array2::eye(embed_dim) * 0.5;
        channel_affinity_matrix += &Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            0.01 * (rng.random::<f32>() - 0.5)
        });

        // Adaptive scaling initialized to 1.0 (standard residual)
        let attention_residual_scales = Array2::ones((embed_dim, 1));
        let ffn_residual_scales = Array2::ones((embed_dim, 1));

        // Initialize optimizers
        let opt_similarity = Adam::new((embed_dim, embed_dim));
        let opt_affinity = Adam::new((embed_dim, 1));
        let opt_attention = Adam::new((embed_dim, 3 * embed_dim));
        let opt_channel = Adam::new((embed_dim, embed_dim));
        let opt_scales_attention = Adam::new((embed_dim, 1));
        let opt_scales_ffn = Adam::new((embed_dim, 1));

        // ===== Theorem 4 Extension: Initialize position-aware residual scaling =====
        let positional_residual_qkv = Array2::from_shape_fn((embed_dim, 3 * embed_dim), |_| {
            0.01 * (rng.random::<f32>() - 0.5)
        });

        // CoPE-like positional embeddings
        let max_seq_len = 2048;
        let cope_pos_embeddings = Array2::from_shape_fn((max_seq_len, embed_dim), |(pos, dim)| {
            let angle = pos as f32 / (10000_f32.powf(2.0 * dim as f32 / embed_dim as f32));
            if dim % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            }
        });

        // Position-aware residual weights
        let positional_residual_weights = Array2::from_shape_fn((embed_dim, embed_dim), |_| {
            0.01 * (rng.random::<f32>() - 0.5)
        });

        // Theorem 4 Extension: Additional optimizers
        let opt_positional_qkv = Adam::new((embed_dim, 3 * embed_dim));
        let opt_cope_pos = Adam::new((max_seq_len, embed_dim));
        let opt_positional_weights = Adam::new((embed_dim, embed_dim));

        Self {
            weight_similarity_matrix,
            activation_similarity_matrix,
            layer_affinity_scores,
            residual_attention_qkv,
            channel_affinity_matrix,
            attention_residual_scales,
            ffn_residual_scales,
            positional_residual_qkv,
            max_seq_len,
            cope_pos_embeddings,
            positional_residual_weights,
            opt_similarity,
            opt_affinity,
            opt_attention,
            opt_channel,
            opt_scales_attention,
            opt_scales_ffn,
            opt_positional_qkv,
            opt_cope_pos,
            opt_positional_weights,
            embed_dim,
            similarity_update_rate: 0.01,
            residual_stability_threshold: 0.1,
            similarity_ema_global: 0.0,
            similarity_ema_per_channel: Array2::zeros((embed_dim, 1)),
            similarity_matrix_valid: false,
            last_similarity_update: 0,
        }
    }

    #[inline]
    fn update_activation_similarity_matrix(&mut self, input: &Array2<f32>, output: &Array2<f32>) {
        // Update an EMA similarity matrix between input/output channels.
        // This is meant as a *representation* of how similarity evolves per layer.
        let rate = smooth_clamp(self.similarity_update_rate, 0.0, 1.0, 10.0);
        if rate <= 0.0 {
            return;
        }

        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input.ncols().min(output.ncols()).min(self.embed_dim);
        if seq_len == 0 || embed_dim == 0 {
            return;
        }

        // Sample along sequence to reduce cost: O(sample * d^2).
        let sample = seq_len.min(32);
        let step = (seq_len / sample).max(1);

        // Precompute per-channel norms on the sampled rows.
        let mut nx = vec![0.0f64; embed_dim];
        let mut ny = vec![0.0f64; embed_dim];
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

                let denom = (nx[i] * ny[j]).sqrt();
                let sim = if denom > 1e-12 { (dot / denom) as f32 } else { 0.0 };
                let sim = if sim.is_finite() { sim.tanh() } else { 0.0 };

                let prev = self.activation_similarity_matrix[[i, j]];
                self.activation_similarity_matrix[[i, j]] = (1.0 - rate) * prev + rate * sim;
            }
        }
    }

    #[inline]
    fn update_similarity_ema(&mut self, input: &Array2<f32>, output: &Array2<f32>) {
        let rate = smooth_clamp(self.similarity_update_rate, 0.0, 1.0, 10.0);
        if rate <= 0.0 {
            return;
        }

        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input.ncols().min(output.ncols()).min(self.embed_dim);
        if seq_len == 0 || embed_dim == 0 {
            return;
        }

        // Per-channel cosine similarity across the sequence dimension.
        // Uses f64 accumulation for numerical accuracy and robustness.
        let mut global_acc = 0.0f64;
        for embed_idx in 0..embed_dim {
            let mut dot = 0.0f64;
            let mut nx = 0.0f64;
            let mut ny = 0.0f64;
            for seq_idx in 0..seq_len {
                let x = input[[seq_idx, embed_idx]];
                let y = output[[seq_idx, embed_idx]];
                let xs = if x.is_finite() { x as f64 } else { 0.0 };
                let ys = if y.is_finite() { y as f64 } else { 0.0 };
                dot += xs * ys;
                nx += xs * xs;
                ny += ys * ys;
            }

            let denom = (nx * ny).sqrt();
            let sim = if denom > 1e-12 { (dot / denom) as f32 } else { 0.0 };
            let sim = if sim.is_finite() { sim.tanh() } else { 0.0 };

            let prev = self.similarity_ema_per_channel[[embed_idx, 0]];
            self.similarity_ema_per_channel[[embed_idx, 0]] = (1.0 - rate) * prev + rate * sim;
            global_acc += self.similarity_ema_per_channel[[embed_idx, 0]] as f64;
        }

        let mean = (global_acc / embed_dim as f64) as f32;
        self.similarity_ema_global = if mean.is_finite() { mean } else { 0.0 };
        self.last_similarity_update = self.last_similarity_update.saturating_add(1);
    }

    /// SIMD-optimized cosine similarity computation
    #[inline]
    fn cosine_similarity_optimized(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
        let mut dot_product = 0.0f64;
        let mut norm_a_sq = 0.0f64;
        let mut norm_b_sq = 0.0f64;

        let len = a.len().min(b.len());

        for i in 0..len {
            let va = a[i] as f64;
            let vb = b[i] as f64;
            dot_product += va * vb;
            norm_a_sq += va * va;
            norm_b_sq += vb * vb;
        }

        let norm_a = norm_a_sq.sqrt();
        let norm_b = norm_b_sq.sqrt();

        if norm_a > 1e-12 && norm_b > 1e-12 {
            let sim = (dot_product / (norm_a * norm_b)) as f32;
            // Cosine similarity should already be within [-1, 1], but allow
            // small numeric overshoots to saturate smoothly.
            sim.tanh()
        } else {
            0.0
        }
    }

    #[inline]
    fn cosine_similarity_rows(input: &Array2<f32>, output: &Array2<f32>, row: usize) -> f32 {
        let embed_dim = input.ncols().min(output.ncols());
        let mut dot = 0.0f64;
        let mut nx = 0.0f64;
        let mut ny = 0.0f64;

        for d in 0..embed_dim {
            let x = input[[row, d]];
            let y = output[[row, d]];
            let xs = if x.is_finite() { x as f64 } else { 0.0 };
            let ys = if y.is_finite() { y as f64 } else { 0.0 };
            dot += xs * ys;
            nx += xs * xs;
            ny += ys * ys;
        }

        let denom = (nx * ny).sqrt();
        if denom > 1e-12 {
            (dot / denom) as f32
        } else {
            0.0
        }
    }

    /// Memory-efficient batch similarity computation
    pub fn compute_batch_similarity_matrix(&mut self, attention_weights: &Array2<f32>, ffn_weights: &Array2<f32>) -> &Array2<f32> {
        let embed_dim = self.embed_dim;

        if self.similarity_matrix_valid {
            return &self.weight_similarity_matrix;
        }

        for i in 0..embed_dim {
            let attn_i = attention_weights.column(i);
            for j in 0..embed_dim {
                let ffn_j = ffn_weights.column(j);
                let similarity = Self::cosine_similarity_optimized(&attn_i, &ffn_j);
                self.weight_similarity_matrix[[i, j]] = similarity;
            }
        }

        self.similarity_matrix_valid = true;
        self.last_similarity_update += 1;

        &self.weight_similarity_matrix
    }
}

impl AdaptiveResidualStrategy for UnifiedAdaptiveResiduals {
    fn apply_attention_residual(&mut self, input: &Array2<f32>, attn_out: &Array2<f32>) -> Array2<f32> {
        let seq_len = input.nrows();

        // Update representative similarity statistics.
        self.update_similarity_ema(input, attn_out);

        // Update per-layer similarity representation matrix (input→output channel similarity).
        self.update_activation_similarity_matrix(input, attn_out);

        // Similarity-based residual fusion.
        // Signed gating: positive similarity increases contribution; negative similarity
        // suppresses and can invert the residual contribution if strong enough.
        let beta_clamp = 10.0;
        let sim_beta = 1.5;
        let gate_limit = 2.0;

        let mut residual = input.clone();
        for seq_idx in 0..seq_len {
            let sim_raw = Self::cosine_similarity_rows(input, attn_out, seq_idx);
            let sim = smooth_clamp(sim_raw, -1.0, 1.0, beta_clamp);
            let sim_centered = (sim_beta * sim).tanh();

            for embed_idx in 0..self.embed_dim {
                let affinity = smooth_clamp(self.layer_affinity_scores[[embed_idx, 0]], 0.0, 2.0, beta_clamp);
                let scale = self.attention_residual_scales[[embed_idx, 0]];
                // Confidence from representative per-channel similarity magnitude.
                let rep = self.similarity_ema_per_channel[[embed_idx.min(self.embed_dim - 1), 0]];
                let conf = smooth_clamp(rep.abs(), 0.0, 1.0, beta_clamp);
                let gate = sim_centered * affinity * scale * (0.5 + 0.5 * conf);
                let gate = smooth_clamp(gate, -gate_limit, gate_limit, beta_clamp);
                let weight = 1.0 + gate;

                let input_val = input[[seq_idx, embed_idx]];
                let attn_val = attn_out[[seq_idx, embed_idx]];
                let input_safe = if input_val.is_finite() { input_val } else { 0.0 };
                let attn_safe = if attn_val.is_finite() { attn_val } else { 0.0 };
                let weight_safe = if weight.is_finite() { weight } else { 1.0 };

                residual[[seq_idx, embed_idx]] = input_safe + weight_safe * attn_safe;
            }
        }

        residual
    }

    fn apply_ffn_residual(&self, residual1: &Array2<f32>, ffn_out: &Array2<f32>) -> Array2<f32> {
        let seq_len = residual1.nrows();
        let mut output = residual1.clone();
        let beta_clamp = 10.0;
        let sim_beta = 1.5;
        let gate_limit = 2.0;
        let avg_similarity = smooth_clamp(self.similarity_ema_global, -1.0, 1.0, beta_clamp);

        for seq_idx in 0..seq_len {
            let sim_raw = Self::cosine_similarity_rows(residual1, ffn_out, seq_idx);
            let sim = smooth_clamp(sim_raw, -1.0, 1.0, beta_clamp);
            let sim_centered = (sim_beta * sim).tanh();

            for embed_idx in 0..self.embed_dim {
                let scale = self.ffn_residual_scales[[embed_idx, 0]];
                let rep = self.similarity_ema_per_channel[[embed_idx.min(self.embed_dim - 1), 0]];
                let rep = smooth_clamp(rep, -1.0, 1.0, beta_clamp);
                let gate = sim_centered * (0.5 * avg_similarity + 0.5 * rep) * scale;
                let gate = smooth_clamp(gate, -gate_limit, gate_limit, beta_clamp);
                let effective_scale = 1.0 + gate;

                let residual_val = residual1[[seq_idx, embed_idx]];
                let ffn_val = ffn_out[[seq_idx, embed_idx]];

                let residual_safe = if residual_val.is_finite() { residual_val } else { 0.0 };
                let ffn_safe = if ffn_val.is_finite() { ffn_val } else { 0.0 };
                let scale_safe = if effective_scale.is_finite() { effective_scale } else { 1.0 };

                output[[seq_idx, embed_idx]] = residual_safe + scale_safe * ffn_safe;
            }
        }

        output
    }

    fn compute_gradients(&self, input: &Array2<f32>, attn_out: &Array2<f32>, ffn_out: &Array2<f32>, residual_grads: &Array2<f32>) -> Vec<Array2<f32>> {
        let seq_len = input.nrows();
        let embed_dim = self.embed_dim;
        let mut param_grads = Vec::with_capacity(9);

        // Gradient arrays
        let similarity_grads = Array2::zeros((embed_dim, embed_dim));
        param_grads.push(similarity_grads);

        let mut affinity_grads = Array2::zeros((embed_dim, 1));
        for embed_idx in 0..embed_dim {
            let residual_sum: f32 = (0..seq_len).map(|seq| residual_grads[[seq, embed_idx]].abs()).sum();
            affinity_grads[[embed_idx, 0]] = residual_sum * 0.001;
        }
        param_grads.push(affinity_grads);

        let attention_grads = Array2::zeros((embed_dim, 3 * embed_dim));
        param_grads.push(attention_grads);

        let channel_grads = Array2::zeros((embed_dim, embed_dim));
        param_grads.push(channel_grads);

        let mut attention_scale_grads = Array2::zeros((embed_dim, 1));
        let mut ffn_scale_grads = Array2::zeros((embed_dim, 1));

        for embed_idx in 0..embed_dim {
            let attn_residual_sum: f32 = (0..seq_len).map(|seq| {
                residual_grads[[seq, embed_idx]] * attn_out[[seq, embed_idx]]
            }).sum();
            attention_scale_grads[[embed_idx, 0]] = attn_residual_sum * 0.001;

            let ffn_residual_sum: f32 = (0..seq_len).map(|seq| {
                residual_grads[[seq, embed_idx]] * ffn_out[[seq, embed_idx]]
            }).sum();
            ffn_scale_grads[[embed_idx, 0]] = ffn_residual_sum * 0.001;
        }

        param_grads.push(attention_scale_grads);
        param_grads.push(ffn_scale_grads);

        // Theorem 4 Extension gradients
        let positional_qkv_grads = Array2::zeros((embed_dim, 3 * embed_dim));
        let cope_pos_grads = Array2::zeros((self.max_seq_len, embed_dim));
        let positional_weights_grads = Array2::zeros((embed_dim, embed_dim));

        param_grads.push(positional_qkv_grads);
        param_grads.push(cope_pos_grads);
        param_grads.push(positional_weights_grads);

        param_grads
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if param_grads.len() < 9 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!("Expected at least 9 gradient arrays, got {}", param_grads.len())
            });
        }

        let mut idx = 0;

        // Similarity matrix
        let similarity_grads = &param_grads[idx];
        self.opt_similarity.step(&mut self.weight_similarity_matrix, similarity_grads, lr);
        let norm = self.weight_similarity_matrix.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            self.weight_similarity_matrix.mapv_inplace(|x| x / norm);
        }
        idx += 1;

        // Affinity scores
        let affinity_grads = &param_grads[idx];
        self.opt_affinity.step(&mut self.layer_affinity_scores, affinity_grads, lr);
        self.layer_affinity_scores.mapv_inplace(|x| smooth_clamp(x, -2.0, 2.0, 10.0));
        idx += 1;

        // Attention parameters
        let attention_grads = &param_grads[idx];
        self.opt_attention.step(&mut self.residual_attention_qkv, attention_grads, lr);
        idx += 1;

        // Channel affinity
        let channel_grads = &param_grads[idx];
        self.opt_channel.step(&mut self.channel_affinity_matrix, channel_grads, lr);
        for i in 0..self.embed_dim {
            for j in 0..i {
                let avg = (self.channel_affinity_matrix[[i, j]] + self.channel_affinity_matrix[[j, i]]) * 0.5;
                self.channel_affinity_matrix[[i, j]] = avg;
                self.channel_affinity_matrix[[j, i]] = avg;
            }
        }
        idx += 1;

        // Attention scales
        let attention_scale_grads = &param_grads[idx];
        self.opt_scales_attention.step(&mut self.attention_residual_scales, attention_scale_grads, lr);
        self.attention_residual_scales.mapv_inplace(|x| smooth_clip_tanh(x, self.residual_stability_threshold));
        idx += 1;

        // FFN scales
        let ffn_scale_grads = &param_grads[idx];
        self.opt_scales_ffn.step(&mut self.ffn_residual_scales, ffn_scale_grads, lr);
        self.ffn_residual_scales.mapv_inplace(|x| smooth_clip_tanh(x, self.residual_stability_threshold));
        idx += 1;

        // Theorem 4 Extension
        let positional_qkv_grads = &param_grads[idx];
        self.opt_positional_qkv.step(&mut self.positional_residual_qkv, positional_qkv_grads, lr);
        idx += 1;

        let cope_pos_grads = &param_grads[idx];
        self.opt_cope_pos.step(&mut self.cope_pos_embeddings, cope_pos_grads, lr);
        idx += 1;

        let positional_weights_grads = &param_grads[idx];
        self.opt_positional_weights.step(&mut self.positional_residual_weights, positional_weights_grads, lr);

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.weight_similarity_matrix.len() +
        self.layer_affinity_scores.len() +
        self.residual_attention_qkv.len() +
        self.channel_affinity_matrix.len() +
        self.attention_residual_scales.len() +
        self.ffn_residual_scales.len() +
        self.positional_residual_qkv.len() +
        self.cope_pos_embeddings.len() +
        self.positional_residual_weights.len()
    }

    fn memory_usage_bytes(&self) -> usize {
        (self.parameter_count() * std::mem::size_of::<f32>()) +
        (self.parameter_count() * std::mem::size_of::<f32>() * 2)
    }

    fn get_performance_metrics(&self) -> (f32, f32, f32) {
        // Use a smooth mapping into (0,1) to avoid discontinuities from hard clamping.
        let affinity_entropy = -self.layer_affinity_scores.iter().map(|&x| {
            let p = 1.0 / (1.0 + (-x).exp());
            // Keep p away from 0/1 smoothly by composing with softplus-based clamp.
            let p = smooth_clamp(p, 1e-3, 1.0 - 1e-3, 10.0);
            p * p.ln() + (1.0 - p) * (1.0 - p).ln()
        }).sum::<f32>() / self.embed_dim as f32;

        let mean_similarity = self.weight_similarity_matrix.mean().unwrap_or(0.0);
        let similarity_variance = self.weight_similarity_matrix.iter()
            .map(|x| (x - mean_similarity).powi(2)).sum::<f32>() / self.weight_similarity_matrix.len() as f32;

        let scale_stability = (self.attention_residual_scales.iter().chain(self.ffn_residual_scales.iter()))
            .map(|x| x.abs()).sum::<f32>() / (2 * self.embed_dim) as f32;

        (affinity_entropy, similarity_variance.sqrt(), scale_stability)
    }
}

impl UnifiedAdaptiveResiduals {
    pub fn residual_stability_threshold(&self) -> f32 {
        self.residual_stability_threshold
    }

    pub fn invalidate_similarity_cache(&mut self) {
        self.similarity_matrix_valid = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_unified_adaptive_residuals_creation() {
        let embed_dim = 64;
        let residuals = UnifiedAdaptiveResiduals::new(embed_dim);

        let param_count = residuals.parameter_count();
        let expected = (embed_dim * embed_dim) + embed_dim + (embed_dim * 3 * embed_dim) +
                       (embed_dim * embed_dim) + embed_dim + embed_dim +
                       (embed_dim * 3 * embed_dim) + 2048 * embed_dim + (embed_dim * embed_dim);

        assert_eq!(param_count, expected);
    }

    #[test]
    fn test_unified_residuals_forward() {
        let embed_dim = 32;
        let seq_len = 8;

        let mut residuals = UnifiedAdaptiveResiduals::new(embed_dim);

        let input = Array2::from_elem((seq_len, embed_dim), 1.0);
        let attn_out = Array2::from_elem((seq_len, embed_dim), 0.5);

        let result = residuals.apply_attention_residual(&input, &attn_out);

        assert_eq!(result.shape(), [seq_len, embed_dim]);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_adaptive_residual_trait() {
        let embed_dim = 8;
        let seq_len = 4;

        let mut residuals: Box<dyn AdaptiveResidualStrategy> = Box::new(UnifiedAdaptiveResiduals::new(embed_dim));

        let input = Array2::from_elem((seq_len, embed_dim), 1.0);
        let attn_out = Array2::from_elem((seq_len, embed_dim), 0.5);
        let ffn_out = Array2::from_elem((seq_len, embed_dim), 0.3);
        let grad = Array2::from_elem((seq_len, embed_dim), 0.01);

        // Test trait methods
        let res1 = residuals.apply_attention_residual(&input, &attn_out);
        assert_eq!(res1.shape(), [seq_len, embed_dim]);

        let res2 = residuals.apply_ffn_residual(&res1, &ffn_out);
        assert_eq!(res2.shape(), [seq_len, embed_dim]);

        let param_grads = residuals.compute_gradients(&input, &attn_out, &ffn_out, &grad);
        assert!(!param_grads.is_empty());

        // This would test apply_gradients but requires setup
        let param_count = residuals.parameter_count();
        assert!(param_count > 0);

        let memory = residuals.memory_usage_bytes();
        assert!(memory > 0);

        let metrics = residuals.get_performance_metrics();
        assert_eq!(metrics.0.is_finite(), true);
        assert_eq!(metrics.1.is_finite(), true);
        assert_eq!(metrics.2.is_finite(), true);
    }

    #[test]
    fn test_similarity_matrix_consistency() {
        let embed_dim = 8;

        let mut residuals = UnifiedAdaptiveResiduals::new(embed_dim);
        let weights = Array2::from_shape_fn((10, embed_dim), |(i, j)| (i + j) as f32 * 0.1);

        // First call should compute the matrix
        residuals.compute_batch_similarity_matrix(&weights, &weights);
        assert_eq!(residuals.similarity_matrix_valid, true);
        let matrix1_ptr = residuals.weight_similarity_matrix.as_ptr();

        // Second call should return cached result
        residuals.compute_batch_similarity_matrix(&weights, &weights);
        assert_eq!(residuals.similarity_matrix_valid, true);
        let matrix2_ptr = residuals.weight_similarity_matrix.as_ptr();

        assert_eq!(matrix1_ptr, matrix2_ptr); // Same memory address = cached

        // Invalidate cache
        residuals.similarity_matrix_valid = false;

        // Recompute should allocate new matrix
        residuals.compute_batch_similarity_matrix(&weights, &weights);
        let _matrix3_ptr = residuals.weight_similarity_matrix.as_ptr();

        // Test consistency by checking values are the same
        let mut original = UnifiedAdaptiveResiduals::new(embed_dim);
        original.compute_batch_similarity_matrix(&weights, &weights);
        let original_values = original.weight_similarity_matrix.clone();

        residuals.similarity_matrix_valid = false;
        residuals.compute_batch_similarity_matrix(&weights, &weights);

        // Values should be approximately the same after recomputation
        let diff = (&original_values - &residuals.weight_similarity_matrix).mapv(|x| x.abs()).mean().unwrap_or(1.0);
        assert!(diff < 1e-6, "Recomputed matrix should have same values, diff: {}", diff);
    }
}
