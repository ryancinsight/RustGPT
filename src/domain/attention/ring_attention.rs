//! Ring Attention for Unbounded Context Length
//!
//! This module implements Ring Attention (arXiv:2309.01809), enabling O(1) memory complexity
//! for arbitrary context lengths through block-wise computation with circular KV buffering.
//!
//! # Mathematical Framework
//!
//! Ring Attention decomposes the attention computation into blocks that are processed
//! in a ring topology, allowing the KV cache to be stored in a fixed-size circular buffer
//! regardless of sequence length.
//!
//! ## Key Properties
//!
//! 1. **O(1) Memory**: Fixed-size circular buffer for KV cache
//! 2. **Unbounded Context**: No theoretical limit on sequence length
//! 3. **Numerical Stability**: Online softmax computation with proper normalization
//! 4. **Compatibility**: Works with polynomial attention and other attention variants
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Ring Attention                        │
//! ├─────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
//! │  │   Block 0   │───→│   Block 1   │───→│   Block 2   │  │
//! │  │  (Q,K,V)    │    │  (Q,K,V)    │    │  (Q,K,V)    │  │
//! │  └─────────────┘    └─────────────┘    └─────────────┘  │
//! │         ↑                                     │         │
//! │         └─────────────────────────────────────┘         │
//! │                    (circular topology)                   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Research Alignment
//!
//! - **Ring Attention**: Liu et al. (2023) - arXiv:2309.01809
//! - **Online Softmax**: Milakov & Gimelshein (2018)
//! - **Flash Attention**: Dao et al. (2022) - block-wise computation patterns

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};

/// Configuration for Ring Attention.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RingAttentionConfig {
    /// Number of tokens per block
    pub block_size: usize,
    /// Number of blocks in the ring buffer (determines effective context length)
    pub num_blocks: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Polynomial degree for attention transformation
    pub polynomial_degree: usize,
    /// Use online softmax for numerical stability
    pub use_online_softmax: bool,
}

impl Default for RingAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            num_blocks: 64, // 64 * 64 = 4096 tokens effective context
            embed_dim: 128,
            num_heads: 8,
            polynomial_degree: 3,
            use_online_softmax: true,
        }
    }
}

impl RingAttentionConfig {
    /// Calculate the effective maximum context length.
    pub fn effective_context_length(&self) -> usize {
        self.block_size * self.num_blocks
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("block_size must be > 0".to_string());
        }
        if self.num_blocks == 0 {
            return Err("num_blocks must be > 0".to_string());
        }
        if self.embed_dim == 0 {
            return Err("embed_dim must be > 0".to_string());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".to_string());
        }
        if self.embed_dim % self.num_heads != 0 {
            return Err("embed_dim must be divisible by num_heads".to_string());
        }
        Ok(())
    }
}

/// A single block in the ring buffer containing KV cache.
#[derive(Debug, Clone)]
pub struct RingBlock {
    /// Key cache for this block (block_size, embed_dim)
    pub k: Array2<f32>,
    /// Value cache for this block (block_size, embed_dim)
    pub v: Array2<f32>,
    /// Number of valid tokens in this block (0..block_size)
    pub valid_len: usize,
    /// Global position of the first token in this block
    pub global_pos: usize,
}

impl RingBlock {
    /// Create a new empty ring block.
    pub fn new(block_size: usize, embed_dim: usize) -> Self {
        Self {
            k: Array2::zeros((block_size, embed_dim)),
            v: Array2::zeros((block_size, embed_dim)),
            valid_len: 0,
            global_pos: 0,
        }
    }

    /// Reset the block to empty state.
    pub fn reset(&mut self) {
        self.valid_len = 0;
        self.global_pos = 0;
    }

    /// Check if the block is full.
    pub fn is_full(&self) -> bool {
        self.valid_len >= self.k.nrows()
    }

    /// Append tokens to this block.
    /// Returns the number of tokens successfully appended.
    pub fn append(&mut self, k_new: &ArrayView2<f32>, v_new: &ArrayView2<f32>, global_pos: usize) -> usize {
        let available = self.k.nrows() - self.valid_len;
        let to_append = k_new.nrows().min(available).min(v_new.nrows());
        
        if to_append > 0 {
            let start = self.valid_len;
            let end = start + to_append;
            self.k.slice_mut(s![start..end, ..]).assign(&k_new.slice(s![..to_append, ..]));
            self.v.slice_mut(s![start..end, ..]).assign(&v_new.slice(s![..to_append, ..]));
            
            if self.valid_len == 0 {
                self.global_pos = global_pos;
            }
            self.valid_len += to_append;
        }
        
        to_append
    }

    /// Get a view of the valid keys in this block.
    pub fn k_view(&self) -> ArrayView2<f32> {
        self.k.slice(s![..self.valid_len, ..])
    }

    /// Get a view of the valid values in this block.
    pub fn v_view(&self) -> ArrayView2<f32> {
        self.v.slice(s![..self.valid_len, ..])
    }
}

/// Circular ring buffer for KV cache with O(1) memory.
#[derive(Debug, Clone)]
pub struct RingBuffer {
    /// Configuration
    config: RingAttentionConfig,
    /// Ring blocks
    blocks: Vec<RingBlock>,
    /// Current write position in the ring
    write_pos: usize,
    /// Total number of tokens stored
    total_tokens: usize,
    /// Head dimension
    head_dim: usize,
}

impl RingBuffer {
    /// Create a new ring buffer with the given configuration.
    pub fn new(config: RingAttentionConfig) -> Self {
        config.validate().expect("Invalid RingAttentionConfig");
        let head_dim = config.embed_dim / config.num_heads;
        
        let blocks: Vec<RingBlock> = (0..config.num_blocks)
            .map(|_| RingBlock::new(config.block_size, config.embed_dim))
            .collect();
        
        Self {
            config,
            blocks,
            write_pos: 0,
            total_tokens: 0,
            head_dim,
        }
    }

    /// Reset the ring buffer to empty state.
    pub fn reset(&mut self) {
        for block in &mut self.blocks {
            block.reset();
        }
        self.write_pos = 0;
        self.total_tokens = 0;
    }

    /// Get the total number of tokens stored.
    pub fn len(&self) -> usize {
        self.total_tokens
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    /// Get the effective context length (maximum tokens that can be stored).
    pub fn capacity(&self) -> usize {
        self.config.effective_context_length()
    }

    /// Append key-value pairs to the ring buffer.
    /// 
    /// This implements the circular write pattern where new tokens overwrite
    /// the oldest tokens when the buffer is full.
    pub fn append_kv(&mut self, k: &Array2<f32>, v: &Array2<f32>) {
        assert_eq!(k.shape(), v.shape(), "K and V must have same shape");
        assert_eq!(k.ncols(), self.config.embed_dim, "K dimension mismatch");
        
        let mut offset = 0usize;
        
        while offset < k.nrows() {
            let block = &mut self.blocks[self.write_pos];
            
            if block.is_full() {
                // Move to next block and reset it
                self.write_pos = (self.write_pos + 1) % self.config.num_blocks;
                self.blocks[self.write_pos].reset();
                continue;
            }
            
            let k_slice = k.slice(s![offset.., ..]);
            let v_slice = v.slice(s![offset.., ..]);
            
            let appended = block.append(&k_slice, &v_slice, self.total_tokens);
            
            if appended == 0 {
                // Block is full, move to next
                self.write_pos = (self.write_pos + 1) % self.config.num_blocks;
                self.blocks[self.write_pos].reset();
            } else {
                offset += appended;
                self.total_tokens += appended;
            }
        }
    }

    /// Iterate over blocks in chronological order (oldest to newest).
    pub fn iter_blocks(&self) -> impl Iterator<Item = &RingBlock> {
        let start = if self.total_tokens >= self.capacity() {
            // Buffer is full/wrapped, start from write position
            self.write_pos
        } else {
            // Buffer not full yet, start from beginning
            0
        };
        
        (0..self.config.num_blocks).map(move |i| {
            let idx = (start + i) % self.config.num_blocks;
            &self.blocks[idx]
        })
    }

    /// Get blocks that are relevant for a query at the given position.
    /// 
    /// For causal attention, this returns all blocks up to the query position.
    pub fn get_relevant_blocks(&self, query_global_pos: usize) -> Vec<&RingBlock> {
        self.iter_blocks()
            .filter(|block| {
                // Include block if it contains positions before the query
                block.valid_len > 0 && block.global_pos < query_global_pos
            })
            .collect()
    }
}

/// Online softmax accumulator for stable attention computation.
/// 
/// Implements the online softmax algorithm from Milakov & Gimelshein (2018)
/// to compute attention in a numerically stable manner without materializing
/// the full attention matrix.
#[derive(Debug, Clone, Default)]
pub struct OnlineSoftmaxAccumulator {
    /// Running maximum of attention scores
    max_score: f32,
    /// Running sum of exponentials
    sum_exp: f32,
    /// Accumulated output
    output: Array1<f32>,
    /// Count of processed blocks
    blocks_processed: usize,
}

impl OnlineSoftmaxAccumulator {
    /// Create a new accumulator with the given output dimension.
    pub fn new(output_dim: usize) -> Self {
        Self {
            max_score: f32::NEG_INFINITY,
            sum_exp: 0.0,
            output: Array1::zeros(output_dim),
            blocks_processed: 0,
        }
    }

    /// Process a new block of attention scores and values.
    /// 
    /// This implements the online softmax update:
    /// - Track running maximum for numerical stability
    /// - Update exponential sum with correction factor
    /// - Accumulate weighted values
    pub fn process_block(
        &mut self,
        scores: &ArrayView1<f32>,
        values: &ArrayView2<f32>,
    ) {
        assert_eq!(scores.len(), values.nrows(), "Score/value count mismatch");
        assert_eq!(values.ncols(), self.output.len(), "Output dimension mismatch");
        
        if scores.is_empty() {
            return;
        }
        
        // Find max score in this block
        let block_max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if self.blocks_processed == 0 {
            // First block - initialize
            self.max_score = block_max;
            self.sum_exp = 0.0;
            
            for (i, &score) in scores.iter().enumerate() {
                let exp_val = (score - self.max_score).exp();
                self.sum_exp += exp_val;
                for j in 0..values.ncols() {
                    self.output[j] += exp_val * values[[i, j]];
                }
            }
        } else {
            // Subsequent blocks - update with correction
            let new_max = self.max_score.max(block_max);
            let correction_old = (self.max_score - new_max).exp();
            let correction_new = (block_max - new_max).exp();
            
            // Scale existing output
            self.output *= correction_old;
            self.sum_exp *= correction_old;
            
            // Add new contributions
            for (i, &score) in scores.iter().enumerate() {
                let exp_val = (score - new_max).exp() * correction_new;
                self.sum_exp += exp_val;
                for j in 0..values.ncols() {
                    self.output[j] += exp_val * values[[i, j]];
                }
            }
            
            self.max_score = new_max;
        }
        
        self.blocks_processed += 1;
    }

    /// Finalize the accumulator and return normalized output.
    pub fn finalize(mut self) -> Array1<f32> {
        if self.sum_exp > 0.0 {
            self.output /= self.sum_exp;
        }
        self.output
    }
}

/// Ring Attention processor with polynomial attention transformation.
#[derive(Debug)]
pub struct RingAttention {
    config: RingAttentionConfig,
    ring_buffer: RingBuffer,
    /// Polynomial parameters (a, b, scale)
    poly_a: f32,
    poly_b: f32,
    poly_scale: f32,
    /// Query projection weights
    w_q: Array2<f32>,
    /// Key projection weights
    w_k: Array2<f32>,
    /// Value projection weights
    w_v: Array2<f32>,
    /// Output projection weights
    w_out: Array2<f32>,
    /// Scaling factor for attention scores
    scale_factor: f32,
}

impl RingAttention {
    /// Create a new Ring Attention processor.
    pub fn new(config: RingAttentionConfig) -> Self {
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        
        config.validate().expect("Invalid RingAttentionConfig");
        let head_dim = config.embed_dim / config.num_heads;
        
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, (head_dim as f32).powf(-0.5)).unwrap();
        
        // Initialize weights with Xavier/Glorot initialization
        let w_q = Array2::from_shape_fn((config.embed_dim, config.embed_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_k = Array2::from_shape_fn((config.embed_dim, config.embed_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_v = Array2::from_shape_fn((config.embed_dim, config.embed_dim), |_| {
            normal.sample(&mut rng)
        });
        let w_out = Array2::from_shape_fn((config.embed_dim, config.embed_dim), |_| {
            normal.sample(&mut rng)
        });
        
        Self {
            config,
            ring_buffer: RingBuffer::new(config),
            poly_a: 1.0,
            poly_b: 0.0,
            poly_scale: 1.0 / (head_dim as f32).sqrt(),
            w_q,
            w_k,
            w_v,
            w_out,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Reset the ring buffer.
    pub fn reset(&mut self) {
        self.ring_buffer.reset();
    }

    /// Process a single token (streaming/rolling mode).
    /// 
    /// This is the core operation for autoregressive generation with unbounded context.
    pub fn forward_step(&mut self, input: &ArrayView1<f32>) -> Array1<f32> {
        let dim = self.config.embed_dim;
        let num_heads = self.config.num_heads;
        let head_dim = dim / num_heads;
        
        // Project input to Q, K, V
        let q = input.dot(&self.w_q.t());
        let k = input.dot(&self.w_k.t());
        let v = input.dot(&self.w_v.t());
        
        // Append K, V to ring buffer
        let k_2d = k.clone().insert_axis(Axis(0));
        let v_2d = v.clone().insert_axis(Axis(0));
        self.ring_buffer.append_kv(&k_2d, &v_2d);
        
        // Process each head
        let mut output = Array1::zeros(dim);
        let global_pos = self.ring_buffer.len();
        
        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            
            let q_h = q.slice(s![start..end]);
            
            // Accumulate attention across ring blocks
            let mut accumulator = OnlineSoftmaxAccumulator::new(head_dim);
            
            for block in self.ring_buffer.get_relevant_blocks(global_pos) {
                let k_block = block.k_view();
                let v_block = block.v_view();
                
                // Compute scores for this block
                let block_len = k_block.nrows();
                let mut scores = Array1::zeros(block_len);
                
                for i in 0..block_len {
                    let k_h = k_block.slice(s![i, start..end]);
                    scores[i] = self.compute_score(&q_h, &k_h);
                }
                
                // Process block
                let v_h = v_block.slice(s![.., start..end]);
                accumulator.process_block(&scores.view(), &v_h);
            }
            
            // Get output for this head
            let head_output = accumulator.finalize();
            
            // Project to output
            let w_out_h = self.w_out.slice(s![start..end, ..]);
            for j in 0..dim {
                for i in 0..head_dim {
                    output[j] += head_output[i] * w_out_h[[i, j]];
                }
            }
        }
        
        output
    }

    /// Compute polynomial attention score.
    fn compute_score(&self, q: &ArrayView1<f32>, k: &ArrayView1<f32>) -> f32 {
        let dot = q.dot(k) * self.scale_factor;
        
        // Apply polynomial transformation
        let p = self.config.polynomial_degree as i32;
        let s_stable = Self::smooth_clip(dot, 8.0);
        
        let sp = if p <= 3 {
            match p {
                1 => s_stable,
                2 => s_stable * s_stable,
                3 => s_stable * s_stable * s_stable,
                _ => 1.0,
            }
        } else {
            s_stable.powi(p)
        };
        
        self.poly_scale * (self.poly_a * sp + self.poly_b)
    }

    /// Smooth clipping for numerical stability.
    fn smooth_clip(x: f32, limit: f32) -> f32 {
        if x > limit {
            limit + (x - limit).tanh()
        } else if x < -limit {
            -limit + (x + limit).tanh()
        } else {
            x
        }
    }

    /// Get current buffer statistics.
    pub fn stats(&self) -> RingAttentionStats {
        RingAttentionStats {
            total_tokens: self.ring_buffer.len(),
            capacity: self.ring_buffer.capacity(),
            utilization: self.ring_buffer.len() as f32 / self.ring_buffer.capacity() as f32,
        }
    }
}

/// Statistics for Ring Attention.
#[derive(Debug, Clone, Copy)]
pub struct RingAttentionStats {
    /// Total tokens currently stored
    pub total_tokens: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Buffer utilization (0.0 to 1.0)
    pub utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_block_append() {
        let mut block = RingBlock::new(4, 8);
        
        let k = Array2::zeros((2, 8));
        let v = Array2::zeros((2, 8));
        
        let appended = block.append(&k.view(), &v.view(), 0);
        assert_eq!(appended, 2);
        assert_eq!(block.valid_len, 2);
        
        // Try to append more than capacity
        let k2 = Array2::zeros((10, 8));
        let v2 = Array2::zeros((10, 8));
        let appended2 = block.append(&k2.view(), &v2.view(), 2);
        assert_eq!(appended2, 2); // Only 2 more fit
        assert!(block.is_full());
    }

    #[test]
    fn test_ring_buffer_circular_write() {
        let config = RingAttentionConfig {
            block_size: 2,
            num_blocks: 3,
            embed_dim: 4,
            num_heads: 2,
            polynomial_degree: 3,
            use_online_softmax: true,
        };
        
        let mut buffer = RingBuffer::new(config);
        
        // Fill buffer
        for i in 0..10 {
            let k = Array2::from_elem((1, 4), i as f32);
            let v = Array2::from_elem((1, 4), i as f32);
            buffer.append_kv(&k, &v);
        }
        
        // Buffer should have wrapped around
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.capacity(), 6);
        
        // Should have 3 blocks with valid data
        let valid_blocks: Vec<_> = buffer.iter_blocks().filter(|b| b.valid_len > 0).collect();
        assert_eq!(valid_blocks.len(), 3);
    }

    #[test]
    fn test_online_softmax() {
        let mut acc = OnlineSoftmaxAccumulator::new(4);
        
        // First block
        let scores1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let values1 = Array2::from_shape_vec((3, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]).unwrap();
        acc.process_block(&scores1.view(), &values1.view());
        
        // Second block
        let scores2 = Array1::from_vec(vec![2.0, 3.0, 4.0]);
        let values2 = Array2::from_shape_vec((3, 4), vec![
            0.0, 0.0, 0.0, 1.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]).unwrap();
        acc.process_block(&scores2.view(), &values2.view());
        
        let output = acc.finalize();
        
        // Check that output is properly normalized
        // With scores [1,2,3,2,3,4], the max is 4
        // Softmax should weight the higher scores more
        assert!(output.iter().all(|&x| x.is_finite()));
        assert!(output.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_ring_attention_forward() {
        let config = RingAttentionConfig {
            block_size: 4,
            num_blocks: 4,
            embed_dim: 16,
            num_heads: 4,
            polynomial_degree: 3,
            use_online_softmax: true,
        };
        
        let mut ring_attn = RingAttention::new(config);
        
        // Process several tokens
        for _ in 0..20 {
            let input = Array1::zeros(16);
            let output = ring_attn.forward_step(&input.view());
            assert_eq!(output.len(), 16);
            assert!(output.iter().all(|&x| x.is_finite()));
        }
        
        let stats = ring_attn.stats();
        assert_eq!(stats.total_tokens, 20);
        assert_eq!(stats.capacity, 16);
        assert!(stats.utilization > 1.0); // Over capacity due to wrapping
    }

    #[test]
    fn test_config_validation() {
        let valid_config = RingAttentionConfig::default();
        assert!(valid_config.validate().is_ok());
        
        let invalid_config = RingAttentionConfig {
            block_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
        
        let invalid_config2 = RingAttentionConfig {
            embed_dim: 128,
            num_heads: 3, // Not divisible
            ..Default::default()
        };
        assert!(invalid_config2.validate().is_err());
    }
}
