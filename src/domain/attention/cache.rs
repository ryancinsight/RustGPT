//! KV Cache for Attention Mechanisms
//!
//! This module provides efficient key-value caching for attention layers with:
//! - O(1) memory complexity via ring buffer pattern
//! - Zero-allocation hot paths for streaming inference
//! - Sliding window support for long context handling
//!
//! # Design Principles
//!
//! - **SSOT**: Single cache implementation used across all attention variants
//! - **SRP**: Cache handles only storage, computation is separate
//! - **SOC**: Clear separation between allocation, access, and eviction

use ndarray::{Array2, ArrayView1, ArrayView2, Zip, s};

/// Cache for a single attention head storing Key and Value matrices.
///
/// Uses a ring buffer pattern for O(1) memory with automatic eviction
/// of oldest tokens when capacity is exceeded.
#[derive(Debug, Clone)]
pub struct HeadCache {
    /// Key matrix cache (capacity, head_dim)
    pub k: Array2<f32>,
    /// Value matrix cache (capacity, head_dim)
    pub v: Array2<f32>,
    /// Current number of tokens stored in the cache
    pub len: usize,
    /// Write position for ring buffer (0..capacity)
    head: usize,
    /// Maximum capacity (cached for fast access)
    capacity: usize,
    /// Head dimension (cached for fast access)
    head_dim: usize,
}

impl HeadCache {
    /// Create a new cache with specified capacity.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens to cache
    /// * `head_dim` - Dimension of each head
    #[inline]
    pub fn new(capacity: usize, head_dim: usize) -> Self {
        let cap = capacity.max(1);
        let dim = head_dim.max(1);
        Self {
            k: Array2::zeros((cap, dim)),
            v: Array2::zeros((cap, dim)),
            len: 0,
            head: 0,
            capacity: cap,
            head_dim: dim,
        }
    }

    /// Reset the cache without deallocating buffers.
    #[inline]
    pub fn reset(&mut self) {
        self.len = 0;
        self.head = 0;
    }

    /// Reset and zero all memory (for security-sensitive contexts).
    #[inline]
    pub fn reset_and_zero(&mut self) {
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.len = 0;
        self.head = 0;
    }

    /// Append new Key and Value states to the cache.
    ///
    /// Uses ring buffer pattern - when capacity is exceeded, oldest tokens
    /// are automatically overwritten.
    #[inline]
    pub fn append(&mut self, k: &Array2<f32>, v: &Array2<f32>) {
        let n = k.nrows();
        debug_assert_eq!(k.ncols(), self.head_dim, "Key dimension mismatch");
        debug_assert_eq!(v.ncols(), self.head_dim, "Value dimension mismatch");
        debug_assert_eq!(v.nrows(), n, "Key/Value row count mismatch");

        if n == 0 {
            return;
        }

        // Handle case where n > capacity (only keep last capacity tokens)
        let to_write = n.min(self.capacity);
        let skip = n.saturating_sub(self.capacity);

        for (i, row_idx) in (skip..n).enumerate() {
            let write_idx = (self.head + i) % self.capacity;
            self.k.row_mut(write_idx).assign(&k.row(row_idx));
            self.v.row_mut(write_idx).assign(&v.row(row_idx));
        }

        self.head = (self.head + to_write) % self.capacity;
        self.len = (self.len + to_write).min(self.capacity);
    }

    /// Append a single token (optimized for streaming inference).
    #[inline]
    pub fn append_single(&mut self, k: &ArrayView1<f32>, v: &ArrayView1<f32>) {
        debug_assert_eq!(k.len(), self.head_dim, "Key dimension mismatch");
        debug_assert_eq!(v.len(), self.head_dim, "Value dimension mismatch");

        self.k.row_mut(self.head).assign(k);
        self.v.row_mut(self.head).assign(v);

        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get valid Key slice (may allocate if buffer is wrapped).
    /// For zero-allocation access, use `iter_keys()` instead.
    pub fn key_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.k.slice(s![0..self.len, ..])
    }

    /// Get valid Value slice (may allocate if buffer is wrapped).
    /// For zero-allocation access, use `iter_values()` instead.
    pub fn value_view(&self) -> ndarray::ArrayView2<'_, f32> {
        self.v.slice(s![0..self.len, ..])
    }

    /// Get the number of valid entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if cache is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len >= self.capacity
    }

    /// Get the capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the head dimension.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get key row at relative index (0 = oldest, len-1 = newest).
    #[inline]
    pub fn key_row(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        if relative_idx >= self.len {
            return None;
        }
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        let actual_idx = (start + relative_idx) % self.capacity;
        Some(self.k.row(actual_idx))
    }

    /// Get value row at relative index.
    #[inline]
    pub fn value_row(&self, relative_idx: usize) -> Option<ArrayView1<'_, f32>> {
        if relative_idx >= self.len {
            return None;
        }
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        let actual_idx = (start + relative_idx) % self.capacity;
        Some(self.v.row(actual_idx))
    }

    /// Get the most recently added key.
    #[inline]
    pub fn latest_key(&self) -> Option<ArrayView1<'_, f32>> {
        if self.len == 0 {
            return None;
        }
        let latest_idx = (self.head + self.capacity - 1) % self.capacity;
        Some(self.k.row(latest_idx))
    }

    /// Get the most recently added value.
    #[inline]
    pub fn latest_value(&self) -> Option<ArrayView1<'_, f32>> {
        if self.len == 0 {
            return None;
        }
        let latest_idx = (self.head + self.capacity - 1) % self.capacity;
        Some(self.v.row(latest_idx))
    }

    /// Compute attention scores for a query against all cached keys.
    /// Writes results into the provided scores buffer.
    #[inline]
    pub fn compute_scores_into(&self, query: &ArrayView1<f32>, scores: &mut [f32]) {
        debug_assert!(scores.len() >= self.len, "Scores buffer too small");

        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };

        for i in 0..self.len {
            let actual_idx = (start + i) % self.capacity;
            let k_row = self.k.row(actual_idx);
            let dot: f32 = Zip::from(query)
                .and(&k_row)
                .fold(0.0, |acc, &q, &k| acc + q * k);
            scores[i] = dot;
        }
    }

    /// Compute weighted sum of values using attention weights.
    /// Writes results into the provided output buffer.
    #[inline]
    pub fn weighted_sum_into(&self, weights: &[f32], output: &mut [f32]) {
        debug_assert!(weights.len() >= self.len, "Weights buffer too small");
        debug_assert!(output.len() >= self.head_dim, "Output buffer too small");

        output.fill(0.0);

        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };

        for i in 0..self.len {
            let actual_idx = (start + i) % self.capacity;
            let w = weights[i];
            let v_row = self.v.row(actual_idx);
            for (j, &v) in v_row.iter().enumerate() {
                output[j] += w * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_head_cache_basic() {
        let mut cache = HeadCache::new(4, 2);

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 4);

        // Add single token
        let k = array![1.0, 2.0];
        let v = array![3.0, 4.0];
        cache.append_single(&k.view(), &v.view());

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_full());

        // Retrieve
        let retrieved_k = cache.key_row(0).unwrap();
        assert_eq!(retrieved_k, k);
    }

    #[test]
    fn test_head_cache_ring_buffer() {
        let mut cache = HeadCache::new(3, 2);

        // Add 5 tokens (should only keep last 3)
        for i in 0..5 {
            let k = array![i as f32, (i + 1) as f32];
            let v = array![(i + 2) as f32, (i + 3) as f32];
            cache.append_single(&k.view(), &v.view());
        }

        assert!(cache.is_full());
        assert_eq!(cache.len(), 3);

        // Oldest should be token 2 (0-indexed)
        let oldest_k = cache.key_row(0).unwrap();
        assert_eq!(oldest_k, array![2.0, 3.0]);

        // Newest should be token 4
        let latest_k = cache.latest_key().unwrap();
        assert_eq!(latest_k, array![4.0, 5.0]);
    }

    #[test]
    fn test_head_cache_compute_scores() {
        let mut cache = HeadCache::new(4, 2);

        cache.append_single(&array![1.0, 0.0].view(), &array![1.0, 0.0].view());
        cache.append_single(&array![0.0, 1.0].view(), &array![0.0, 1.0].view());

        let query = array![1.0, 1.0];
        let mut scores = vec![0.0; 4];

        cache.compute_scores_into(&query.view(), &mut scores);

        assert_eq!(scores[0], 1.0); // [1,0] dot [1,1] = 1
        assert_eq!(scores[1], 1.0); // [0,1] dot [1,1] = 1
    }

    #[test]
    fn test_head_cache_weighted_sum() {
        let mut cache = HeadCache::new(4, 2);

        cache.append_single(&array![1.0, 0.0].view(), &array![1.0, 2.0].view());
        cache.append_single(&array![0.0, 1.0].view(), &array![3.0, 4.0].view());

        let weights = [0.5, 0.5];
        let mut output = vec![0.0; 2];

        cache.weighted_sum_into(&weights, &mut output);

        // 0.5 * [1,2] + 0.5 * [3,4] = [2, 3]
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_head_cache_append_batch() {
        let mut cache = HeadCache::new(10, 2);

        let k = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let v = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        cache.append(&k, &v);

        assert_eq!(cache.len(), 3);

        let k0 = cache.key_row(0).unwrap();
        assert_eq!(k0, array![1.0, 2.0]);
    }
}
