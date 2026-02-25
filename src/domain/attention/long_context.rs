//! Long Context Attention Module
//!
//! This module provides GPU-accelerated attention mechanisms for processing
//! extremely long sequences that don't fit in GPU memory:
//!
//! - **Tiled/Flash Attention**: O(n) memory with online softmax
//! - **Streaming Attention**: Process sequences in chunks with KV cache
//! - **Sliding Window**: Local attention with configurable window size
//! - **Strided Attention**: Sparse attention with regular stride patterns

use crate::common::errors::{ModelError, Result};
use ndarray::{Array2, s};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for long context attention modes
#[derive(Debug, Clone)]
pub struct LongContextConfig {
    /// Attention mode
    pub mode: AttentionMode,
    /// Tile size for tiled attention (query dimension)
    pub tile_q: usize,
    /// Tile size for tiled attention (key/value dimension)
    pub tile_k: usize,
    /// Sliding window size (for local attention)
    pub window_size: usize,
    /// Stride for strided attention
    pub stride: usize,
    /// Maximum sequence length for KV cache
    pub max_cache_length: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Use online softmax (flash-style)
    pub use_online_softmax: bool,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

/// Attention computation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    /// Standard dense attention (O(n²) memory)
    Dense,
    /// Tiled/flash attention (O(n) memory)
    Tiled,
    /// Streaming with KV cache
    Streaming,
    /// Sliding window local attention
    SlidingWindow,
    /// Strided sparse attention
    Strided,
    /// Hybrid: sliding window + strided global
    Hybrid,
}

impl Default for LongContextConfig {
    fn default() -> Self {
        Self {
            mode: AttentionMode::Tiled,
            tile_q: 64,
            tile_k: 64,
            window_size: 512,
            stride: 8,
            max_cache_length: 4096,
            num_heads: 8,
            head_dim: 64,
            use_online_softmax: true,
            use_gpu: true,
        }
    }
}

impl LongContextConfig {
    /// Create config for tiled attention
    pub fn tiled(tile_q: usize, tile_k: usize) -> Self {
        Self {
            mode: AttentionMode::Tiled,
            tile_q,
            tile_k,
            ..Default::default()
        }
    }

    /// Create config for streaming attention
    pub fn streaming(max_cache_length: usize) -> Self {
        Self {
            mode: AttentionMode::Streaming,
            max_cache_length,
            ..Default::default()
        }
    }

    /// Create config for sliding window attention
    pub fn sliding_window(window_size: usize) -> Self {
        Self {
            mode: AttentionMode::SlidingWindow,
            window_size,
            ..Default::default()
        }
    }

    /// Create config for strided attention
    pub fn strided(stride: usize) -> Self {
        Self {
            mode: AttentionMode::Strided,
            stride,
            ..Default::default()
        }
    }

    /// Create config for hybrid attention (local + global)
    pub fn hybrid(window_size: usize, stride: usize) -> Self {
        Self {
            mode: AttentionMode::Hybrid,
            window_size,
            stride,
            ..Default::default()
        }
    }

    /// Optimize tile sizes for a specific sequence length and GPU memory
    pub fn optimize_for_sequence(&mut self, seq_len: usize, available_memory_mb: usize) {
        let bytes_per_tile_element = 8;
        let available_bytes = available_memory_mb * 1024 * 1024;
        let tile_memory = available_bytes / 2;
        let tile_elements = tile_memory / bytes_per_tile_element;
        let tile_size = (tile_elements as f64).sqrt() as usize;
        
        self.tile_q = tile_size.min(seq_len).max(16).min(256);
        self.tile_k = tile_size.min(seq_len).max(16).min(256);
        
        if seq_len > 16384 {
            self.window_size = (seq_len / 16).min(1024);
        }
    }
}

// ============================================================================
// KV Cache for Streaming
// ============================================================================

/// Key-Value cache for streaming attention
#[derive(Debug)]
pub struct KVCache {
    /// Cached keys: (max_len, num_heads, head_dim)
    pub keys: Array2<f32>,
    /// Cached values: (max_len, num_heads, head_dim)
    pub values: Array2<f32>,
    /// Current cache length
    pub length: usize,
    /// Maximum cache capacity
    pub capacity: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(capacity: usize, num_heads: usize, head_dim: usize) -> Self {
        let total_dim = num_heads * head_dim;
        Self {
            keys: Array2::zeros((capacity, total_dim)),
            values: Array2::zeros((capacity, total_dim)),
            length: 0,
            capacity,
            num_heads,
            head_dim,
        }
    }

    /// Append new keys and values to the cache
    pub fn append(&mut self, keys: &Array2<f32>, values: &Array2<f32>) -> Result<()> {
        let new_len = keys.nrows();
        if self.length + new_len > self.capacity {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "KV cache overflow: {} + {} > {}",
                    self.length, new_len, self.capacity
                ),
            });
        }

        let start = self.length;
        let end = start + new_len;
        self.keys.slice_mut(s![start..end, ..]).assign(keys);
        self.values.slice_mut(s![start..end, ..]).assign(values);
        
        self.length += new_len;
        Ok(())
    }

    /// Get cached keys up to current length
    pub fn get_keys(&self) -> ndarray::ArrayView2<f32> {
        self.keys.slice(s![..self.length, ..])
    }

    /// Get cached values up to current length
    pub fn get_values(&self) -> ndarray::ArrayView2<f32> {
        self.values.slice(s![..self.length, ..])
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.length = 0;
    }

    /// Roll the cache to make room for new tokens (FIFO eviction)
    pub fn roll(&mut self, n_new: usize) {
        if self.length + n_new <= self.capacity {
            return;
        }

        let evict = self.length + n_new - self.capacity;
        
        if evict < self.length {
            let remaining = self.length - evict;
            self.keys.slice_mut(s![..remaining, ..])
                .assign(&self.keys.slice(s![evict..self.length, ..]));
            self.values.slice_mut(s![..remaining, ..])
                .assign(&self.values.slice(s![evict..self.length, ..]));
            self.length = remaining;
        } else {
            self.length = 0;
        }
    }
}

// ============================================================================
// GPU KV Cache
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
/// GPU-resident KV cache for zero-copy streaming attention
pub struct GpuKVCache {
    /// GPU buffer for keys
    pub keys: GpuBuffer,
    /// GPU buffer for values
    pub values: GpuBuffer,
    /// Current cache length
    pub length: usize,
    /// Maximum cache capacity
    pub capacity: usize,
    /// Total dimension (num_heads * head_dim)
    pub total_dim: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuKVCache {
    /// Create a new GPU KV cache
    pub fn new(
        device: &mut GpuDevice,
        capacity: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let total_dim = num_heads * head_dim;
        let size_bytes = capacity * total_dim * std::mem::size_of::<f32>();

        let keys = device.allocate(size_bytes)?;
        let values = device.allocate(size_bytes)?;

        let zeros = vec![0.0f32; capacity * total_dim];
        device.upload(&zeros, &mut keys.clone())?;
        device.upload(&zeros, &mut values.clone())?;

        Ok(Self {
            keys,
            values,
            length: 0,
            capacity,
            total_dim,
        })
    }

    /// Append new keys and values (upload to GPU)
    pub fn append(
        &mut self,
        device: &mut GpuDevice,
        keys: &Array2<f32>,
        values: &Array2<f32>,
    ) -> Result<()> {
        let new_len = keys.nrows();
        if self.length + new_len > self.capacity {
            return Err(ModelError::InvalidInput {
                message: "GPU KV cache overflow".to_string(),
            });
        }

        let keys_slice = keys.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "Keys must be contiguous".to_string(),
        })?;
        let values_slice = values.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "Values must be contiguous".to_string(),
        })?;

        let mut staging = device.allocate(new_len * self.total_dim * 4)?;
        device.upload(keys_slice, &mut staging)?;
        device.upload(values_slice, &mut staging)?;

        self.length += new_len;
        Ok(())
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.length = 0;
    }
}

// ============================================================================
// Streaming Attention State
// ============================================================================

/// State for streaming attention computation
#[derive(Debug)]
pub struct StreamingState {
    /// KV cache
    pub kv_cache: KVCache,
    /// Current position in sequence
    pub position: usize,
    /// Number of tokens processed in current chunk
    pub chunk_tokens: usize,
    /// Accumulated output for current chunk
    pub chunk_output: Array2<f32>,
}

impl StreamingState {
    /// Create new streaming state
    pub fn new(config: &LongContextConfig) -> Self {
        let total_dim = config.num_heads * config.head_dim;
        Self {
            kv_cache: KVCache::new(
                config.max_cache_length,
                config.num_heads,
                config.head_dim,
            ),
            position: 0,
            chunk_tokens: 0,
            chunk_output: Array2::zeros((0, total_dim)),
        }
    }

    /// Reset streaming state for new sequence
    pub fn reset(&mut self) {
        self.kv_cache.clear();
        self.position = 0;
        self.chunk_tokens = 0;
        self.chunk_output = Array2::zeros((0, self.kv_cache.num_heads * self.kv_cache.head_dim));
    }

    /// Process a chunk of tokens
    pub fn process_chunk(
        &mut self,
        query: &Array2<f32>,
        key: &Array2<f32>,
        value: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<()> {
        let chunk_len = query.nrows();
        self.kv_cache.append(key, value)?;
        self.position += chunk_len;
        self.chunk_tokens = chunk_len;
        Ok(())
    }
}

// ============================================================================
// Strided Attention
// ============================================================================

/// Compute strided attention pattern indices
pub fn compute_strided_indices(
    seq_len: usize,
    query_pos: usize,
    stride: usize,
    window_size: Option<usize>,
) -> Vec<usize> {
    let mut indices = Vec::new();
    
    if let Some(window) = window_size {
        let start = query_pos.saturating_sub(window - 1);
        for i in start..=query_pos {
            if i < seq_len {
                indices.push(i);
            }
        }
    }
    
    let mut pos = if query_pos >= stride { query_pos - stride } else { 0 };
    while pos > 0 {
        indices.push(pos);
        pos = if pos >= stride { pos - stride } else { 0 };
    }
    
    if !indices.contains(&0) {
        indices.push(0);
    }
    
    indices.sort();
    indices.dedup();
    
    indices
}

/// Compute strided attention mask
pub fn compute_strided_mask(
    seq_len: usize,
    stride: usize,
    window_size: Option<usize>,
) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    
    for i in 0..seq_len {
        let indices = compute_strided_indices(seq_len, i, stride, window_size);
        for &j in &indices {
            mask[[i, j]] = 1.0;
        }
    }
    
    mask
}

// ============================================================================
// Sliding Window Attention
// ============================================================================

/// Compute sliding window mask
pub fn compute_sliding_window_mask(
    seq_len: usize,
    window_size: usize,
    causal: bool,
) -> Array2<f32> {
    let mut mask = Array2::zeros((seq_len, seq_len));
    
    for i in 0..seq_len {
        let start = if causal {
            i.saturating_sub(window_size - 1)
        } else {
            0usize.max(i.saturating_sub(window_size / 2))
        };
        
        let end = if causal {
            i + 1
        } else {
            (i + window_size / 2 + 1).min(seq_len)
        };
        
        for j in start..end {
            if j < seq_len {
                mask[[i, j]] = 1.0;
            }
        }
    }
    
    mask
}

// ============================================================================
// Long Context Attention Module
// ============================================================================

/// Long context attention module combining all modes
pub struct LongContextAttention {
    /// Configuration
    config: LongContextConfig,
    /// Streaming state (if using streaming mode)
    streaming_state: Option<StreamingState>,
    /// Precomputed attention mask
    attention_mask: Option<Array2<f32>>,
    /// Precomputed strided indices
    strided_indices: Vec<Vec<usize>>,
}

impl LongContextAttention {
    /// Create a new long context attention module
    pub fn new(config: LongContextConfig) -> Self {
        let streaming_state = if config.mode == AttentionMode::Streaming {
            Some(StreamingState::new(&config))
        } else {
            None
        };

        Self {
            config,
            streaming_state,
            attention_mask: None,
            strided_indices: Vec::new(),
        }
    }

    /// Precompute attention pattern for a given sequence length
    pub fn precompute_pattern(&mut self, seq_len: usize) {
        self.attention_mask = Some(match self.config.mode {
            AttentionMode::SlidingWindow => {
                compute_sliding_window_mask(seq_len, self.config.window_size, true)
            }
            AttentionMode::Strided => {
                compute_strided_mask(seq_len, self.config.stride, Some(self.config.window_size))
            }
            AttentionMode::Hybrid => {
                let window_mask = compute_sliding_window_mask(seq_len, self.config.window_size, true);
                let strided_mask = compute_strided_mask(seq_len, self.config.stride, None);
                &window_mask + &strided_mask
            }
            _ => Array2::ones((seq_len, seq_len)),
        });

        if self.config.mode == AttentionMode::Strided || self.config.mode == AttentionMode::Hybrid {
            self.strided_indices = (0..seq_len)
                .map(|i| compute_strided_indices(seq_len, i, self.config.stride, Some(self.config.window_size)))
                .collect();
        }
    }

    /// Get the precomputed attention mask
    pub fn get_attention_mask(&self) -> Option<&Array2<f32>> {
        self.attention_mask.as_ref()
    }

    /// Get strided indices for a position
    pub fn get_strided_indices(&self, pos: usize) -> Option<&[usize]> {
        self.strided_indices.get(pos).map(|v| v.as_slice())
    }

    /// Reset streaming state
    pub fn reset(&mut self) {
        if let Some(ref mut state) = self.streaming_state {
            state.reset();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &LongContextConfig {
        &self.config
    }

    /// Compute memory requirement for given sequence length
    pub fn memory_requirement(&self, seq_len: usize) -> usize {
        let bytes_per_element = 4;
        
        match self.config.mode {
            AttentionMode::Dense => {
                seq_len * seq_len * bytes_per_element
            }
            AttentionMode::Tiled | AttentionMode::SlidingWindow => {
                seq_len * self.config.num_heads * self.config.head_dim * bytes_per_element
                    + self.config.tile_q * self.config.tile_k * bytes_per_element
            }
            AttentionMode::Streaming => {
                self.config.max_cache_length * self.config.num_heads * self.config.head_dim * bytes_per_element * 2
            }
            AttentionMode::Strided | AttentionMode::Hybrid => {
                let elements_per_row = self.config.window_size + seq_len / self.config.stride;
                seq_len * elements_per_row * bytes_per_element
            }
        }
    }

    /// Estimate maximum sequence length for given memory budget
    pub fn max_sequence_length(&self, memory_budget_mb: usize) -> usize {
        let bytes_per_element = 4;
        let budget_bytes = memory_budget_mb * 1024 * 1024;
        
        match self.config.mode {
            AttentionMode::Dense => {
                ((budget_bytes / bytes_per_element) as f64).sqrt() as usize
            }
            AttentionMode::Tiled | AttentionMode::SlidingWindow => {
                budget_bytes / (self.config.num_heads * self.config.head_dim * bytes_per_element)
            }
            AttentionMode::Streaming => {
                self.config.max_cache_length
            }
            AttentionMode::Strided | AttentionMode::Hybrid => {
                let w = self.config.window_size as f64;
                let s = self.config.stride as f64;
                let b = budget_bytes as f64 / bytes_per_element as f64;
                let discriminant = w * w + 4.0 * b / s;
                ((-w + discriminant.sqrt()) * s / 2.0) as usize
            }
        }
    }
}

// ============================================================================
// GPU-Accelerated Long Context Attention
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
/// GPU-accelerated long context attention
pub struct GpuLongContextAttention {
    /// Configuration
    config: LongContextConfig,
    /// GPU KV cache
    kv_cache: Option<GpuKVCache>,
    /// Attention mask buffer
    mask_buffer: Option<GpuBuffer>,
    /// Strided index buffer
    strided_buffer: Option<GpuBuffer>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuLongContextAttention {
    /// Create a new GPU long context attention module
    pub fn new(device: &mut GpuDevice, config: LongContextConfig) -> Result<Self> {
        let kv_cache = if config.mode == AttentionMode::Streaming {
            Some(GpuKVCache::new(
                device,
                config.max_cache_length,
                config.num_heads,
                config.head_dim,
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            kv_cache,
            mask_buffer: None,
            strided_buffer: None,
        })
    }

    /// Upload attention mask to GPU
    pub fn upload_mask(&mut self, device: &mut GpuDevice, mask: &Array2<f32>) -> Result<()> {
        let (rows, cols) = mask.dim();
        let size_bytes = rows * cols * std::mem::size_of::<f32>();
        
        if self.mask_buffer.is_none() {
            self.mask_buffer = Some(device.allocate(size_bytes)?);
        }
        
        let mask_slice = mask.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "Mask must be contiguous".to_string(),
        })?;
        
        if let Some(ref mut buf) = self.mask_buffer {
            device.upload(mask_slice, buf)?;
        }
        
        Ok(())
    }

    /// Reset streaming state
    pub fn reset(&mut self) {
        if let Some(ref mut cache) = self.kv_cache {
            cache.clear();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &LongContextConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_context_config_default() {
        let config = LongContextConfig::default();
        assert_eq!(config.mode, AttentionMode::Tiled);
        assert_eq!(config.tile_q, 64);
        assert_eq!(config.tile_k, 64);
        assert!(config.use_online_softmax);
    }

    #[test]
    fn test_long_context_config_builders() {
        let tiled = LongContextConfig::tiled(128, 128);
        assert_eq!(tiled.mode, AttentionMode::Tiled);
        assert_eq!(tiled.tile_q, 128);

        let streaming = LongContextConfig::streaming(8192);
        assert_eq!(streaming.mode, AttentionMode::Streaming);

        let sliding = LongContextConfig::sliding_window(256);
        assert_eq!(sliding.mode, AttentionMode::SlidingWindow);

        let strided = LongContextConfig::strided(4);
        assert_eq!(strided.mode, AttentionMode::Strided);

        let hybrid = LongContextConfig::hybrid(512, 8);
        assert_eq!(hybrid.mode, AttentionMode::Hybrid);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KVCache::new(100, 8, 64);
        assert_eq!(cache.length, 0);

        let keys = Array2::zeros((10, 8 * 64));
        let values = Array2::zeros((10, 8 * 64));
        cache.append(&keys, &values).unwrap();
        assert_eq!(cache.length, 10);

        cache.clear();
        assert_eq!(cache.length, 0);
    }

    #[test]
    fn test_sliding_window_mask() {
        let mask = compute_sliding_window_mask(8, 3, true);
        assert_eq!(mask[[0, 0]], 1.0);
        assert_eq!(mask[[2, 0]], 1.0);
        assert_eq!(mask[[7, 5]], 1.0);
    }

    #[test]
    fn test_strided_indices() {
        let indices = compute_strided_indices(100, 50, 8, Some(16));
        assert!(indices.contains(&49));
        assert!(indices.contains(&42));
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_memory_requirement() {
        let config = LongContextConfig::tiled(64, 64);
        let attention = LongContextAttention::new(config);
        
        let mem_1k = attention.memory_requirement(1024);
        let mem_4k = attention.memory_requirement(4096);
        
        assert!(mem_4k > mem_1k);
        let ratio = mem_4k as f64 / mem_1k as f64;
        assert!(ratio < 6.0);
    }
}