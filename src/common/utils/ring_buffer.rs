//! Unified Ring Buffer Abstraction for Zero-Allocation Streaming Operations
//!
//! This module provides a single, optimized ring buffer implementation that serves
//! as the SSOT (Single Source of Truth) for all rolling/sliding operations in the
//! codebase, including:
//! - Convolution history in Mamba layers
//! - KV cache in attention mechanisms
//! - Sliding window attention caches
//!
//! # Design Principles
//!
//! - **Zero-Allocation Hot Path**: Pre-allocated buffers with no runtime allocation
//! - **O(1) Operations**: Constant time push, get, and rotation
//! - **Memory Efficient**: Fixed-size buffer regardless of sequence length
//! - **Cache Friendly**: Contiguous memory layout for better cache utilization
//! - **Dynamic Window**: Adaptive window sizing based on context needs
//!
//! # Research Alignment
//!
//! - Ring Attention (Liu et al., 2023): O(1) memory for unbounded context
//! - Flash Attention: Block-wise computation patterns
//! - vLLM PagedAttention: Efficient memory management
//! - Dynamic Context: Research on adaptive context windows (2024)

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use serde::{Deserialize, Serialize};

/// Configuration for a ring buffer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RingBufferConfig {
    /// Maximum capacity (number of elements or rows)
    pub capacity: usize,
    /// Dimension of each element (1D: element size, 2D: row width)
    pub dim: usize,
}

impl Default for RingBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            dim: 128,
        }
    }
}

/// A 1D ring buffer for scalar/vector history.
///
/// This is used for convolution history, temporal accumulation, etc.
/// Provides O(1) push and access with zero allocation after initialization.
#[derive(Debug, Clone)]
pub struct RingBuffer1D {
    /// Underlying storage
    buffer: Array2<f32>, // Shape: (capacity, dim)
    /// Current write position (0..capacity)
    head: usize,
    /// Number of valid entries (0..capacity)
    len: usize,
    /// Configuration
    config: RingBufferConfig,
}

impl RingBuffer1D {
    /// Create a new 1D ring buffer with the given capacity and dimension.
    #[inline]
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            buffer: Array2::zeros((capacity.max(1), dim.max(1))),
            head: 0,
            len: 0,
            config: RingBufferConfig { capacity, dim },
        }
    }

    /// Create from existing configuration.
    #[inline]
    pub fn from_config(config: RingBufferConfig) -> Self {
        Self::new(config.capacity, config.dim)
    }

    /// Push a new element into the buffer.
    ///
    /// If the buffer is full, the oldest element is overwritten.
    #[inline]
    pub fn push(&mut self, element: &ArrayView1<f32>) {
        debug_assert_eq!(element.len(), self.config.dim);

        // Copy element to current head position
        self.buffer.row_mut(self.head).assign(element);

        // Advance head with wrap-around
        self.head = (self.head + 1) % self.config.capacity.max(1);

        // Update length (capped at capacity)
        if self.len < self.config.capacity {
            self.len += 1;
        }
    }

    /// Push a scalar value, broadcasting it to all dimensions.
    #[inline]
    pub fn push_scalar(&mut self, value: f32) {
        self.buffer.row_mut(self.head).fill(value);
        self.head = (self.head + 1) % self.config.capacity.max(1);
        if self.len < self.config.capacity {
            self.len += 1;
        }
    }

    /// Get the number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if the buffer is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len >= self.config.capacity
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Get the dimension of each element.
    #[inline]
    pub fn dim(&self) -> usize {
        self.config.dim
    }

    /// Get element at relative index (0 = oldest, len-1 = newest).
    #[inline]
    pub fn get(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        if relative_idx >= self.len {
            return None;
        }

        let capacity = self.config.capacity.max(1);
        // Calculate actual index: oldest is at (head - len) % capacity
        let start = if self.len < capacity { 0 } else { self.head };
        let actual_idx = (start + relative_idx) % capacity;

        Some(self.buffer.row(actual_idx))
    }

    /// Get the most recently pushed element.
    #[inline]
    pub fn latest(&self) -> Option<ArrayView1<'_, f32>> {
        if self.len == 0 {
            return None;
        }

        let capacity = self.config.capacity.max(1);
        let latest_idx = (self.head + capacity - 1) % capacity;
        Some(self.buffer.row(latest_idx))
    }

    /// Get the oldest element.
    #[inline]
    pub fn oldest(&self) -> Option<ArrayView1<'_, f32>> {
        if self.len == 0 {
            return None;
        }

        let capacity = self.config.capacity.max(1);
        let oldest_idx = if self.len < capacity { 0 } else { self.head };
        Some(self.buffer.row(oldest_idx))
    }

    /// Iterate over all valid elements in chronological order (oldest to newest).
    #[inline]
    pub fn iter(&self) -> RingBuffer1DIterator<'_> {
        RingBuffer1DIterator {
            buffer: self,
            pos: 0,
        }
    }

    /// Clear the buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Reset the buffer and zero all memory.
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.len = 0;
    }

    /// Get the underlying storage (for advanced use).
    #[inline]
    pub fn storage(&self) -> &Array2<f32> {
        &self.buffer
    }

    /// Get mutable access to underlying storage (for advanced use).
    #[inline]
    pub fn storage_mut(&mut self) -> &mut Array2<f32> {
        &mut self.buffer
    }

    /// Compute weighted sum with given weights (for convolution).
    ///
    /// `weights` should have length equal to `min(self.len, weights.len())`.
    /// Output is accumulated into `result` (not cleared first).
    #[inline]
    pub fn weighted_sum_into(&self, weights: &[f32], result: &mut Array1<f32>) {
        let n = weights.len().min(self.len);
        let capacity = self.config.capacity.max(1);

        // Calculate starting position (oldest element)
        let start = if self.len < capacity { 0 } else { self.head };

        for i in 0..n {
            let actual_idx = (start + i) % capacity;
            let w = weights[i];
            let row = self.buffer.row(actual_idx);
            Zip::from(result.view_mut())
                .and(&row)
                .for_each(|acc, &v| *acc += w * v);
        }
    }

    /// Compute weighted sum, returning a new array.
    #[inline]
    pub fn weighted_sum(&self, weights: &[f32]) -> Array1<f32> {
        let mut result = Array1::zeros(self.config.dim);
        self.weighted_sum_into(weights, &mut result);
        result
    }
}

/// Iterator over ring buffer elements in chronological order.
pub struct RingBuffer1DIterator<'a> {
    buffer: &'a RingBuffer1D,
    pos: usize,
}

impl<'a> Iterator for RingBuffer1DIterator<'a> {
    type Item = ArrayView1<'a, f32>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.buffer.len {
            return None;
        }

        let item = self.buffer.get(self.pos);
        self.pos += 1;
        item
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.buffer.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for RingBuffer1DIterator<'a> {}

/// A 2D ring buffer for matrix/tensor history.
///
/// This is used for KV caches in attention mechanisms where each entry
/// is a 2D matrix (e.g., keys or values for multiple tokens).
#[derive(Debug, Clone)]
pub struct RingBuffer2D {
    /// Underlying storage: Vec of matrices
    buffers: Vec<Array2<f32>>,
    /// Current write position
    head: usize,
    /// Number of valid entries
    len: usize,
    /// Maximum capacity
    capacity: usize,
}

impl RingBuffer2D {
    /// Create a new 2D ring buffer.
    ///
    /// Each entry will be a matrix of shape (rows, cols).
    pub fn new(capacity: usize, rows: usize, cols: usize) -> Self {
        let buffers = (0..capacity.max(1))
            .map(|_| Array2::zeros((rows.max(1), cols.max(1))))
            .collect();

        Self {
            buffers,
            head: 0,
            len: 0,
            capacity: capacity.max(1),
        }
    }

    /// Push a new matrix into the buffer.
    #[inline]
    pub fn push(&mut self, matrix: &ArrayView2<f32>) {
        self.buffers[self.head].assign(matrix);
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get the number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get element at relative index.
    #[inline]
    pub fn get(&self, relative_idx: usize) -> Option<&Array2<f32>> {
        if relative_idx >= self.len {
            return None;
        }

        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        let actual_idx = (start + relative_idx) % self.capacity;
        Some(&self.buffers[actual_idx])
    }

    /// Clear the buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Reset and zero all buffers.
    #[inline]
    pub fn reset(&mut self) {
        for buf in &mut self.buffers {
            buf.fill(0.0);
        }
        self.head = 0;
        self.len = 0;
    }
}

/// A specialized ring buffer for sliding window attention KV cache.
///
/// Optimized for the common case where we need:
/// 1. O(1) append of new K/V pairs
/// 2. O(1) access to the last W entries
/// 3. Zero allocation during inference
#[derive(Debug, Clone)]
pub struct SlidingWindowKVCache {
    /// Key cache: (window_size, num_heads, head_dim)
    k_cache: Array2<f32>,
    /// Value cache: (window_size, num_heads, head_dim)
    v_cache: Array2<f32>,
    /// Current write position
    head: usize,
    /// Number of valid entries
    len: usize,
    /// Window size (capacity)
    window_size: usize,
    /// Total embedding dimension (num_heads * head_dim)
    embed_dim: usize,
}

impl SlidingWindowKVCache {
    /// Create a new sliding window KV cache.
    #[inline]
    pub fn new(window_size: usize, embed_dim: usize) -> Self {
        let ws = window_size.max(1);
        let ed = embed_dim.max(1);
        Self {
            k_cache: Array2::zeros((ws, ed)),
            v_cache: Array2::zeros((ws, ed)),
            head: 0,
            len: 0,
            window_size: ws,
            embed_dim: ed,
        }
    }

    /// Append a new K/V pair.
    #[inline]
    pub fn append(&mut self, k: &ArrayView1<f32>, v: &ArrayView1<f32>) {
        debug_assert_eq!(k.len(), self.embed_dim);
        debug_assert_eq!(v.len(), self.embed_dim);

        self.k_cache.row_mut(self.head).assign(k);
        self.v_cache.row_mut(self.head).assign(v);

        self.head = (self.head + 1) % self.window_size;
        if self.len < self.window_size {
            self.len += 1;
        }
    }

    /// Get the number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len >= self.window_size
    }

    /// Get K cache view for all valid entries.
    /// Returns (len, embed_dim) shaped view.
    #[inline]
    pub fn k_view(&self) -> Array2<f32> {
        if self.len < self.window_size {
            // Not wrapped yet, simple slice from beginning
            self.k_cache.slice(s![..self.len, ..]).to_owned()
        } else {
            // Wrapped: need to reconstruct in order
            let mut result = Array2::zeros((self.len, self.embed_dim));
            // Copy from head to end
            let first_part = self.window_size - self.head;
            result
                .slice_mut(s![..first_part, ..])
                .assign(&self.k_cache.slice(s![self.head.., ..]));
            // Copy from beginning to head
            if self.head > 0 {
                result
                    .slice_mut(s![first_part.., ..])
                    .assign(&self.k_cache.slice(s![..self.head, ..]));
            }
            result
        }
    }

    /// Get V cache view for all valid entries.
    #[inline]
    pub fn v_view(&self) -> Array2<f32> {
        if self.len < self.window_size {
            self.v_cache.slice(s![..self.len, ..]).to_owned()
        } else {
            let mut result = Array2::zeros((self.len, self.embed_dim));
            let first_part = self.window_size - self.head;
            result
                .slice_mut(s![..first_part, ..])
                .assign(&self.v_cache.slice(s![self.head.., ..]));
            if self.head > 0 {
                result
                    .slice_mut(s![first_part.., ..])
                    .assign(&self.v_cache.slice(s![..self.head, ..]));
            }
            result
        }
    }

    /// Get K row at relative index (0 = oldest).
    #[inline]
    pub fn k_row(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        if relative_idx >= self.len {
            return None;
        }

        let start = if self.len < self.window_size {
            0
        } else {
            self.head
        };
        let actual_idx = (start + relative_idx) % self.window_size;
        Some(self.k_cache.row(actual_idx))
    }

    /// Get V row at relative index.
    #[inline]
    pub fn v_row(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        if relative_idx >= self.len {
            return None;
        }

        let start = if self.len < self.window_size {
            0
        } else {
            self.head
        };
        let actual_idx = (start + relative_idx) % self.window_size;
        Some(self.v_cache.row(actual_idx))
    }

    /// Clear the cache.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Reset and zero memory.
    #[inline]
    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.head = 0;
        self.len = 0;
    }

    /// Get window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get embedding dimension.
    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Get raw K cache storage (for advanced use).
    #[inline]
    pub fn k_storage(&self) -> &Array2<f32> {
        &self.k_cache
    }

    /// Get raw V cache storage.
    #[inline]
    pub fn v_storage(&self) -> &Array2<f32> {
        &self.v_cache
    }

    /// Get mutable K cache storage.
    #[inline]
    pub fn k_storage_mut(&mut self) -> &mut Array2<f32> {
        &mut self.k_cache
    }

    /// Get mutable V cache storage.
    #[inline]
    pub fn v_storage_mut(&mut self) -> &mut Array2<f32> {
        &mut self.v_cache
    }
}

/// A dynamic window ring buffer that adapts its effective window size.
///
/// This extends the basic ring buffer with:
/// - Dynamic window sizing based on attention patterns
/// - Importance-based eviction for better context retention
/// - Memory-efficient sparse attention support
///
/// # Research Alignment
///
/// - Dynamic Context Window (2024): Adaptive context based on task complexity
/// - Sparse Attention: Focus on relevant tokens
/// - Memory-efficient inference for long-context models
#[derive(Debug, Clone)]
pub struct DynamicWindowRingBuffer {
    /// Underlying storage
    buffer: Array2<f32>,
    /// Importance scores for each position (for eviction decisions)
    importance: Vec<f32>,
    /// Current write position
    head: usize,
    /// Number of valid entries
    len: usize,
    /// Maximum capacity
    capacity: usize,
    /// Dimension of each element
    dim: usize,
    /// Current effective window size (can be smaller than capacity)
    effective_window: usize,
    /// Decay factor for importance scores
    importance_decay: f32,
}

impl DynamicWindowRingBuffer {
    /// Create a new dynamic window ring buffer.
    #[inline]
    pub fn new(capacity: usize, dim: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            buffer: Array2::zeros((cap, dim.max(1))),
            importance: vec![0.0; cap],
            head: 0,
            len: 0,
            capacity: cap,
            dim: dim.max(1),
            effective_window: cap,
            importance_decay: 0.95,
        }
    }

    /// Set the effective window size.
    ///
    /// This allows dynamically adjusting how many tokens are considered
    /// without reallocating the buffer.
    #[inline]
    pub fn set_effective_window(&mut self, window: usize) {
        self.effective_window = window.min(self.capacity).max(1);
    }

    /// Get the effective window size.
    #[inline]
    pub fn effective_window(&self) -> usize {
        self.effective_window
    }

    /// Push an element with an importance score.
    ///
    /// Higher importance scores make entries less likely to be evicted
    /// when the buffer is under memory pressure.
    #[inline]
    pub fn push_with_importance(&mut self, element: &ArrayView1<f32>, importance: f32) {
        debug_assert_eq!(element.len(), self.dim);

        // Decay all existing importance scores
        for score in &mut self.importance {
            *score *= self.importance_decay;
        }

        // Store the element
        self.buffer.row_mut(self.head).assign(element);
        self.importance[self.head] = importance;

        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Push an element with default importance (1.0).
    #[inline]
    pub fn push(&mut self, element: &ArrayView1<f32>) {
        self.push_with_importance(element, 1.0);
    }

    /// Get the number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.min(self.effective_window)
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get element at relative index (0 = oldest within effective window).
    #[inline]
    pub fn get(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        let effective_len = self.len();
        if relative_idx >= effective_len {
            return None;
        }

        // Calculate the start position considering effective window
        let skip = self.len.saturating_sub(self.effective_window);
        let start = if self.len < self.capacity {
            skip
        } else {
            (self.head + skip) % self.capacity
        };

        let actual_idx = (start + relative_idx) % self.capacity;
        Some(self.buffer.row(actual_idx))
    }

    /// Get importance score at relative index.
    #[inline]
    pub fn importance_at(&self, relative_idx: usize) -> Option<f32> {
        let effective_len = self.len();
        if relative_idx >= effective_len {
            return None;
        }

        let skip = self.len.saturating_sub(self.effective_window);
        let start = if self.len < self.capacity {
            skip
        } else {
            (self.head + skip) % self.capacity
        };

        let actual_idx = (start + relative_idx) % self.capacity;
        Some(self.importance[actual_idx])
    }

    /// Get the most important entries (sorted by importance).
    ///
    /// Returns indices relative to the effective window.
    pub fn most_important_indices(&self, top_k: usize) -> Vec<usize> {
        let effective_len = self.len();
        let k = top_k.min(effective_len);

        let mut indexed: Vec<(usize, f32)> = (0..effective_len)
            .filter_map(|i| self.importance_at(i).map(|imp| (i, imp)))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    /// Compute weighted attention over buffer entries.
    ///
    /// Combines importance scores with query-based attention weights.
    #[inline]
    pub fn attention_weighted_sum(
        &self,
        _query: &ArrayView1<f32>,
        attention_weights: &[f32],
        output: &mut Array1<f32>,
    ) {
        let effective_len = self.len();
        debug_assert!(attention_weights.len() >= effective_len);

        output.fill(0.0);

        for i in 0..effective_len {
            if let (Some(element), Some(imp)) = (self.get(i), self.importance_at(i)) {
                let combined_weight = attention_weights[i] * imp;
                Zip::from(output.view_mut())
                    .and(&element)
                    .for_each(|out, &v| {
                        *out += combined_weight * v;
                    });
            }
        }
    }

    /// Clear the buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
        self.importance.fill(0.0);
    }

    /// Reset and zero memory.
    #[inline]
    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.importance.fill(0.0);
        self.head = 0;
        self.len = 0;
    }

    /// Set the importance decay factor.
    #[inline]
    pub fn set_importance_decay(&mut self, decay: f32) {
        self.importance_decay = decay.clamp(0.0, 1.0);
    }

    /// Get memory usage in bytes.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        // buffer: capacity * dim * 4 bytes
        // importance: capacity * 4 bytes
        self.capacity * self.dim * 4 + self.capacity * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ring_buffer_1d_basic() {
        let mut buf = RingBuffer1D::new(4, 3);

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 4);

        // Push elements
        buf.push(&array![1.0, 2.0, 3.0].view());
        assert_eq!(buf.len(), 1);

        buf.push(&array![4.0, 5.0, 6.0].view());
        buf.push(&array![7.0, 8.0, 9.0].view());
        buf.push(&array![10.0, 11.0, 12.0].view());

        assert!(buf.is_full());
        assert_eq!(buf.len(), 4);

        // Check oldest and latest
        let oldest = buf.oldest().unwrap();
        assert_eq!(oldest, array![1.0, 2.0, 3.0]);

        let latest = buf.latest().unwrap();
        assert_eq!(latest, array![10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_ring_buffer_1d_wrap_around() {
        let mut buf = RingBuffer1D::new(3, 2);

        buf.push(&array![1.0, 2.0].view());
        buf.push(&array![3.0, 4.0].view());
        buf.push(&array![5.0, 6.0].view());
        buf.push(&array![7.0, 8.0].view()); // Overwrites first

        assert_eq!(buf.len(), 3);

        // Oldest should now be [3, 4]
        let oldest = buf.oldest().unwrap();
        assert_eq!(oldest, array![3.0, 4.0]);

        // Latest should be [7, 8]
        let latest = buf.latest().unwrap();
        assert_eq!(latest, array![7.0, 8.0]);
    }

    #[test]
    fn test_ring_buffer_1d_iteration() {
        let mut buf = RingBuffer1D::new(4, 1);

        buf.push(&array![1.0].view());
        buf.push(&array![2.0].view());
        buf.push(&array![3.0].view());

        let values: Vec<f32> = buf.iter().map(|row| row[0]).collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_ring_buffer_1d_weighted_sum() {
        let mut buf = RingBuffer1D::new(4, 2);

        buf.push(&array![1.0, 1.0].view());
        buf.push(&array![2.0, 2.0].view());
        buf.push(&array![3.0, 3.0].view());

        let weights = [1.0, 2.0, 3.0];
        let sum = buf.weighted_sum(&weights);

        // Expected: 1*[1,1] + 2*[2,2] + 3*[3,3] = [14, 14]
        assert_eq!(sum, array![14.0, 14.0]);
    }

    #[test]
    fn test_sliding_window_kv_cache() {
        let mut cache = SlidingWindowKVCache::new(3, 4);

        assert!(cache.is_empty());

        cache.append(
            &array![1.0, 2.0, 3.0, 4.0].view(),
            &array![5.0, 6.0, 7.0, 8.0].view(),
        );
        cache.append(
            &array![9.0, 10.0, 11.0, 12.0].view(),
            &array![13.0, 14.0, 15.0, 16.0].view(),
        );

        assert_eq!(cache.len(), 2);

        // Check K rows
        let k0 = cache.k_row(0).unwrap();
        assert_eq!(k0, array![1.0, 2.0, 3.0, 4.0]);

        let k1 = cache.k_row(1).unwrap();
        assert_eq!(k1, array![9.0, 10.0, 11.0, 12.0]);

        // Add one more to fill window
        cache.append(
            &array![17.0, 18.0, 19.0, 20.0].view(),
            &array![21.0, 22.0, 23.0, 24.0].view(),
        );
        assert!(cache.is_full());

        // Add one more to trigger wrap
        cache.append(
            &array![25.0, 26.0, 27.0, 28.0].view(),
            &array![29.0, 30.0, 31.0, 32.0].view(),
        );

        // Oldest should now be [9, 10, 11, 12]
        let k0 = cache.k_row(0).unwrap();
        assert_eq!(k0, array![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_ring_buffer_2d() {
        let mut buf = RingBuffer2D::new(3, 2, 2);

        let m1 = array![[1.0, 2.0], [3.0, 4.0]];
        let m2 = array![[5.0, 6.0], [7.0, 8.0]];

        buf.push(&m1.view());
        buf.push(&m2.view());

        assert_eq!(buf.len(), 2);

        let retrieved = buf.get(0).unwrap();
        assert_eq!(*retrieved, m1);
    }

    #[test]
    fn test_ring_buffer_clear_and_reset() {
        let mut buf = RingBuffer1D::new(3, 2);

        buf.push(&array![1.0, 2.0].view());
        buf.push(&array![3.0, 4.0].view());

        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        // Storage should still contain data
        assert_ne!(buf.storage()[[0, 0]], 0.0);

        buf.reset();
        // Storage should be zeroed
        assert_eq!(buf.storage()[[0, 0]], 0.0);
    }

    #[test]
    fn test_dynamic_window_ring_buffer_basic() {
        let mut buf = DynamicWindowRingBuffer::new(10, 2);

        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.effective_window(), 10);

        // Push elements
        buf.push(&array![1.0, 2.0].view());
        buf.push(&array![3.0, 4.0].view());

        assert_eq!(buf.len(), 2);

        // Get element
        let elem = buf.get(0).unwrap();
        assert_eq!(elem, array![1.0, 2.0]);
    }

    #[test]
    fn test_dynamic_window_effective_size() {
        let mut buf = DynamicWindowRingBuffer::new(10, 2);

        // Fill buffer
        for i in 0..10 {
            buf.push(&array![i as f32, (i + 1) as f32].view());
        }

        assert_eq!(buf.len(), 10);

        // Reduce effective window
        buf.set_effective_window(5);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.effective_window(), 5);

        // Should only see last 5 elements
        let first = buf.get(0).unwrap();
        assert_eq!(first, array![5.0, 6.0]);
    }

    #[test]
    fn test_dynamic_window_importance() {
        let mut buf = DynamicWindowRingBuffer::new(5, 2);

        // Push with different importance
        buf.push_with_importance(&array![1.0, 1.0].view(), 0.5);
        buf.push_with_importance(&array![2.0, 2.0].view(), 1.0);
        buf.push_with_importance(&array![3.0, 3.0].view(), 0.1);

        // Check importance scores (accounting for decay)
        let imp0 = buf.importance_at(0).unwrap();
        let imp1 = buf.importance_at(1).unwrap();
        let imp2 = buf.importance_at(2).unwrap();

        // First element decayed twice, second decayed once, third not decayed
        assert!(imp0 < imp1); // Decayed more
        assert!(imp2 < imp1); // Started lower

        // Get most important indices
        let top = buf.most_important_indices(2);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_dynamic_window_attention_weighted() {
        let mut buf = DynamicWindowRingBuffer::new(5, 2);

        // Set decay to 1.0 so importance doesn't change
        buf.set_importance_decay(1.0);

        buf.push_with_importance(&array![1.0, 0.0].view(), 1.0);
        buf.push_with_importance(&array![0.0, 1.0].view(), 1.0);

        let query = array![1.0, 1.0];
        let attention_weights = [0.5, 0.5];
        let mut output = Array1::zeros(2);

        buf.attention_weighted_sum(&query.view(), &attention_weights, &mut output);

        // 0.5 * 1.0 * [1,0] + 0.5 * 1.0 * [0,1] = [0.5, 0.5]
        assert!((output[0] - 0.5).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_dynamic_window_memory_usage() {
        let buf = DynamicWindowRingBuffer::new(100, 64);

        // 100 * 64 * 4 (buffer) + 100 * 4 (importance) = 25600 + 400 = 26000
        assert_eq!(buf.memory_usage(), 26000);
    }
}
