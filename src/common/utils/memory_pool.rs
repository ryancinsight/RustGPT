//! Memory pool for reducing allocation pressure in hot paths.
//!
//! Provides thread-local and shared memory pools for reusable buffers,
//! reducing allocator pressure during training and inference.
//!
//! # Design
//!
//! - `ThreadLocalPool`: Per-thread buffers for maximum performance
//! - `BufferBucket`: Size-categorized buckets within pools
//! - `MemoryPool`: Configurable pool with capacity limits
//!
//! # Usage
//!
//! ```rust
//! use common::utils::memory_pool::MemoryPool;
//!
//! let mut pool = MemoryPool::new(1024 * 1024); // 1MB capacity
//!
//! // Acquire a buffer of specific size
//! let buf = pool.acquire::<f32>(1024);
//! // ... use buf ...
//! pool.release(buf);
//! ```

use std::collections::HashMap;
use std::mem;
use std::sync::{Arc, Mutex};

/// Thread-local buffer pool for per-thread allocations.
#[derive(Debug)]
pub struct ThreadLocalPool {
    /// Size-categorized buckets
    buckets: HashMap<usize, Vec<Vec<f32>>>,
    /// Maximum total bytes to retain
    capacity_bytes: usize,
    /// Current total bytes in pool
    current_bytes: usize,
}

impl Default for ThreadLocalPool {
    fn default() -> Self {
        Self::new(1024 * 1024) // 1MB default capacity
    }
}

impl ThreadLocalPool {
    /// Create a new thread-local pool with specified capacity.
    #[inline]
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            capacity_bytes,
            current_bytes: 0,
        }
    }

    /// Acquire a buffer of at least the requested size.
    #[inline]
    pub fn acquire_f32(&mut self, min_size: usize) -> Vec<f32> {
        // Round up to power of 2 for efficiency
        let size = min_size.next_power_of_two().max(64);

        if let Some(bucket) = self.buckets.get_mut(&size) {
            if let Some(buf) = bucket.pop() {
                // Buffer taken from pool - reduce current usage
                let bytes = buf.capacity() * mem::size_of::<f32>();
                self.current_bytes = self.current_bytes.saturating_sub(bytes);
                return buf;
            }
        }

        // Allocate new buffer - don't count it until it's returned
        Vec::with_capacity(size)
    }

    /// Release a buffer back to the pool.
    #[inline]
    pub fn release(&mut self, mut buf: Vec<f32>) {
        let size = buf.capacity();
        let bytes = size * mem::size_of::<f32>();

        // Check if we have capacity to retain this buffer
        if self.current_bytes + bytes <= self.capacity_bytes {
            buf.clear();
            self.buckets.entry(size).or_insert_with(Vec::new).push(buf);
            self.current_bytes += bytes;
        }
        // Otherwise: buffer is dropped and freed naturally
    }

    /// Get current memory usage.
    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.current_bytes
    }

    /// Clear all cached buffers.
    #[inline]
    pub fn clear(&mut self) {
        self.buckets.clear();
        self.current_bytes = 0;
    }
}

/// A buffer bucket containing buffers of uniform size.
#[derive(Debug)]
pub struct BufferBucket<T> {
    /// Buffers in this bucket
    buffers: Vec<Vec<T>>,
    /// Buffer capacity
    capacity: usize,
    /// Maximum buffers to retain
    max_buffers: usize,
}

impl<T> BufferBucket<T> {
    /// Create a new bucket.
    #[inline]
    pub fn new(capacity: usize, max_buffers: usize) -> Self {
        Self {
            buffers: Vec::with_capacity(max_buffers.min(64)),
            capacity,
            max_buffers,
        }
    }

    /// Acquire a buffer, creating new if empty.
    #[inline]
    pub fn acquire(&mut self) -> Vec<T> {
        self.buffers
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(self.capacity))
    }

    /// Release a buffer back to the bucket.
    #[inline]
    pub fn release(&mut self, mut buf: Vec<T>) {
        if self.buffers.len() < self.max_buffers && buf.capacity() == self.capacity {
            buf.clear();
            self.buffers.push(buf);
        }
        // Otherwise: let buffer be freed
    }

    /// Get current buffer count.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Check if bucket is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Shared memory pool with thread-safe access.
#[derive(Debug, Clone)]
pub struct MemoryPool {
    inner: Arc<Mutex<PoolInner>>,
    total_capacity: usize,
}

#[derive(Debug)]
struct PoolInner {
    buckets: HashMap<usize, BufferBucket<f32>>,
}

impl MemoryPool {
    /// Create a new memory pool.
    #[inline]
    pub fn new(capacity_bytes: usize) -> Self {
        let bucket_count = (capacity_bytes / 1024).max(16);

        Self {
            inner: Arc::new(Mutex::new(PoolInner {
                buckets: HashMap::with_capacity(bucket_count),
            })),
            total_capacity: capacity_bytes,
        }
    }

    /// Acquire a buffer of at least the requested size.
    #[inline]
    pub fn acquire(&self, min_size: usize) -> Vec<f32> {
        let size = min_size.next_power_of_two().max(64);

        let mut inner = self.inner.lock().unwrap();

        if let Some(bucket) = inner.buckets.get_mut(&size) {
            let buf = bucket.acquire();
            if !buf.is_empty() {
                return buf;
            }
        }

        Vec::with_capacity(size)
    }

    /// Release a buffer back to the pool.
    #[inline]
    pub fn release(&self, buf: Vec<f32>) {
        let size = buf.capacity();
        let bytes = size * mem::size_of::<f32>();

        if bytes > self.total_capacity / 16 {
            // Too large for pool, let it be freed
            return;
        }

        let mut inner = self.inner.lock().unwrap();

        let bucket = inner
            .buckets
            .entry(size)
            .or_insert_with(|| BufferBucket::new(size, 64));

        bucket.release(buf);
    }

    /// Get pool statistics.
    #[inline]
    pub fn stats(&self) -> PoolStats {
        let inner = self.inner.lock().unwrap();
        let mut total_buffers = 0;
        let mut total_capacity = 0;

        for (size, bucket) in &inner.buckets {
            total_buffers += bucket.len();
            total_capacity += bucket.len() * size * mem::size_of::<f32>();
        }

        PoolStats {
            total_buffers,
            total_capacity,
            total_capacity_bytes: self.total_capacity,
        }
    }
}

/// Statistics for a memory pool.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of cached buffers
    pub total_buffers: usize,
    /// Total capacity of cached data
    pub total_capacity: usize,
    /// Maximum capacity
    pub total_capacity_bytes: usize,
}

/// Get or create thread-local memory pool.
///
/// Note: This returns a mutable borrow that must be released within the same closure.
/// For most use cases, create a ThreadLocalPool instance directly.
#[inline]
pub fn with_tls_pool<R>(f: impl FnOnce(&mut ThreadLocalPool) -> R) -> R {
    thread_local! {
        static POOL: std::cell::RefCell<ThreadLocalPool> =
            std::cell::RefCell::new(ThreadLocalPool::default());
    }

    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        f(&mut pool)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_acquire_release() {
        let mut pool = ThreadLocalPool::new(1024 * 1024);

        // Acquire buffers
        let buf1 = pool.acquire_f32(256);
        assert!(buf1.capacity() >= 256);
        assert!(buf1.is_empty());

        let buf2 = pool.acquire_f32(128);
        assert!(buf2.capacity() >= 128);

        // Release and reacquire
        pool.release(buf1);
        let buf1_reacquired = pool.acquire_f32(256);
        assert!(buf1_reacquired.capacity() >= 256);
    }

    #[test]
    fn test_buffer_pool_capacity_limit() {
        let mut pool = ThreadLocalPool::new(1024); // 1KB limit

        // Acquire and release buffers
        for _ in 0..100 {
            let buf = pool.acquire_f32(512);
            pool.release(buf);
        }

        // After stabilization, pool should be at or near capacity
        // The pool rounds up sizes to power of 2, so 512 -> 512 or 1024
        // With 1KB limit, we can fit at most 2 buffers of 512 bytes each
        let usage = pool.memory_usage();
        assert!(
            usage <= 1024 * 4,
            "Pool usage {} exceeds reasonable limit",
            usage
        );
    }

    #[test]
    fn test_shared_pool_stats() {
        let pool = MemoryPool::new(1024 * 1024);

        let buf = pool.acquire(512);
        pool.release(buf);

        let stats = pool.stats();
        assert!(stats.total_buffers >= 1);
    }
}
