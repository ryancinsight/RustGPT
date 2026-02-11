//! PagedAttention - Memory-Efficient KV Cache with OS-like Paging
//!
//! This module implements PagedAttention as described in:
//! "Efficient Memory Management for Large Language Models Serving" (vLLM)
//! (Kwon et al., 2023)
//!
//! # Key Ideas
//!
//! - KV cache is stored in fixed-size "pages" (like OS virtual memory)
//! - Logical KV sequences map to non-contiguous physical pages
//! - Memory sharing between sequences through copy-on-write pages
//! - Significantly reduces memory fragmentation and waste
//!
//! # Architecture
//!
//! ```text
//! Logical Sequence:  [Token 0] [Token 1] [Token 2] [Token 3] [Token 4] ...
//!                    ├─────────┼─────────┼─────────┼─────────┼─────────┤
//! Physical Pages:    [Page 0]  [Page 1]  [Page 2]  [Page 3]  [Page 4]
//!
//! Mapping:           L0→P0, L1→P1, L2→P0(shared), L3→P3, L4→P5
//!
//! # Benefits
//!
//! - 2-4x memory efficiency vs contiguous allocation
//! - Memory sharing between requests
//! - Reduced fragmentation
//! - Better GPU memory utilization
//!
//! # Usage
//!
//! ```rust
//! use llm::domain::attention::paged_attention::{PagedKVCache, PagedKVCacheConfig};
//!
//! let config = PagedKVCacheConfig::default();
//! let mut cache = PagedKVCache::new(config);
//! let (block_id, block_offset) = cache.append(&key, &value);
//! let cached_kv = cache.get(block_id, block_offset);
//! ```
//!

use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::fmt::Debug;

/// Page size in tokens (must be power of 2 for efficiency)
pub const PAGE_SIZE: usize = 16;

/// Number of pages in the block pool
pub const BLOCK_POOL_SIZE: usize = 1024;

/// Configuration for PagedAttention
#[derive(Debug, Clone, Copy)]
pub struct PagedKVCacheConfig {
    /// Page size in tokens (power of 2)
    pub page_size: usize,

    /// Number of pages in the physical pool
    pub max_num_pages: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Number of key/value heads
    pub num_heads: usize,

    /// Enable memory sharing between sequences
    pub enable_sharing: bool,

    /// Enable copy-on-write for shared pages
    pub copy_on_write: bool,

    /// Enable compression for old pages
    pub compress_old_pages: bool,
}

impl Default for PagedKVCacheConfig {
    fn default() -> Self {
        Self {
            page_size: PAGE_SIZE,
            max_num_pages: BLOCK_POOL_SIZE,
            head_dim: 64,
            num_heads: 8,
            enable_sharing: true,
            copy_on_write: true,
            compress_old_pages: false,
        }
    }
}

impl PagedKVCacheConfig {
    /// Create config optimized for inference (memory constrained)
    pub fn inference_optimized() -> Self {
        Self {
            page_size: 32,
            max_num_pages: 2048,
            head_dim: 64,
            num_heads: 8,
            enable_sharing: true,
            copy_on_write: true,
            compress_old_pages: true,
        }
    }

    /// Create config optimized for training (throughput focused)
    pub fn training_optimized() -> Self {
        Self {
            page_size: 16,
            max_num_pages: 4096,
            head_dim: 64,
            num_heads: 8,
            enable_sharing: false,
            copy_on_write: false,
            compress_old_pages: false,
        }
    }

    /// Calculate total capacity in tokens
    pub fn total_capacity(&self) -> usize {
        self.page_size * self.max_num_pages
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.page_size == 0 || (self.page_size & (self.page_size - 1)) != 0 {
            return Err("page_size must be power of 2".to_string());
        }
        if self.max_num_pages == 0 {
            return Err("max_num_pages must be > 0".to_string());
        }
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".to_string());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be > 0".to_string());
        }
        Ok(())
    }
}

/// A physical page containing KV data
#[derive(Debug, Clone)]
pub struct KVPage {
    /// Page ID
    pub page_id: usize,

    /// Keys: (page_size, head_dim)
    pub keys: Array2<f32>,

    /// Values: (page_size, head_dim)
    pub values: Array2<f32>,

    /// Reference count (for sharing)
    pub ref_count: usize,

    /// Whether this page is shared
    pub is_shared: bool,

    /// Whether this page has been modified (for copy-on-write)
    pub is_dirty: bool,
}

impl KVPage {
    /// Create a new empty page
    pub fn new(page_id: usize, _num_heads: usize, page_size: usize, head_dim: usize) -> Self {
        // Store as (page_size, head_dim) - one token per row
        Self {
            page_id,
            keys: Array2::zeros((page_size, head_dim)),
            values: Array2::zeros((page_size, head_dim)),
            ref_count: 1,
            is_shared: false,
            is_dirty: false,
        }
    }

    /// Reset page for reuse
    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.ref_count = 1;
        self.is_shared = false;
        self.is_dirty = false;
    }

    /// Increment reference count
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
        if self.ref_count > 1 {
            self.is_shared = true;
        }
    }

    /// Decrement reference count, return true if still referenced
    pub fn dec_ref(&mut self) -> bool {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
        self.ref_count > 0
    }

    /// Get reference count
    pub fn ref_count(&self) -> usize {
        self.ref_count
    }

    /// Check if page is shared
    pub fn is_shared(&self) -> bool {
        self.is_shared
    }
}

/// Logical block mapping for a sequence
#[derive(Debug, Clone)]
pub struct LogicalBlock {
    /// Physical page ID
    pub page_id: usize,

    /// Block ID in the logical sequence
    pub logical_block_id: usize,

    /// Offset within the page (0 to page_size-1)
    pub offset: usize,

    /// Number of tokens in this block
    pub num_tokens: usize,
}

/// A logical KV sequence (logical view of KV cache)
#[derive(Debug, Clone)]
pub struct LogicalKVSequence {
    /// Sequence ID
    pub seq_id: u64,

    /// Block ID for this sequence (for mapping)
    pub block_id: usize,

    /// Logical blocks in order
    pub blocks: Vec<LogicalBlock>,

    /// Total number of tokens
    pub num_tokens: usize,

    /// Maximum sequence length
    pub max_len: usize,

    /// Reference to parent sequence (for sharing)
    pub parent: Option<Box<LogicalKVSequence>>,
}

impl LogicalKVSequence {
    /// Create a new logical sequence
    pub fn new(seq_id: u64, block_id: usize) -> Self {
        Self {
            seq_id,
            block_id,
            blocks: Vec::new(),
            num_tokens: 0,
            max_len: 0,
            parent: None,
        }
    }

    /// Append a token to the sequence
    pub fn append(&mut self, _page_id: usize, num_tokens: usize) {
        self.num_tokens += num_tokens;
        self.max_len = self.max_len.max(self.num_tokens);
    }

    /// Get the number of blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
}

/// Physical block pool manager
#[derive(Debug)]
pub struct BlockPool {
    /// Available pages (free list)
    free_pages: Vec<usize>,

    /// All allocated pages
    pages: Vec<KVPage>,

    /// Config
    config: PagedKVCacheConfig,

    /// Total pages allocated
    pages_allocated: usize,
}

impl BlockPool {
    /// Create a new block pool
    pub fn new(config: PagedKVCacheConfig) -> Self {
        config.validate().expect("Invalid config");

        let mut pages = Vec::with_capacity(config.max_num_pages);
        let mut free_pages = Vec::with_capacity(config.max_num_pages);

        // Pre-allocate all pages
        for i in 0..config.max_num_pages {
            pages.push(KVPage::new(
                i,
                config.num_heads,
                config.page_size,
                config.head_dim,
            ));
            free_pages.push(i);
        }

        Self {
            free_pages,
            pages,
            config,
            pages_allocated: config.max_num_pages,
        }
    }

    /// Allocate a page, returns page_id
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_pages.pop()
    }

    /// Free a page
    pub fn free(&mut self, page_id: usize) {
        if page_id < self.pages.len() {
            self.pages[page_id].reset();
            self.free_pages.push(page_id);
        }
    }

    /// Get a page (immutable)
    pub fn get(&self, page_id: usize) -> Option<&KVPage> {
        self.pages.get(page_id)
    }

    /// Get a page (mutable)
    pub fn get_mut(&mut self, page_id: usize) -> Option<&mut KVPage> {
        self.pages.get_mut(page_id)
    }

    /// Get number of free pages
    pub fn free_count(&self) -> usize {
        self.free_pages.len()
    }

    /// Get total pages
    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }

    /// Get utilization
    pub fn utilization(&self) -> f32 {
        1.0 - (self.free_pages.len() as f32 / self.pages.len() as f32)
    }

    /// Copy a page (for copy-on-write)
    pub fn copy_page(&mut self, src_page_id: usize) -> Option<usize> {
        let src = self.pages.get(src_page_id).cloned()?;
        let dst_id = self.allocate()?;

        // Copy the page data
        if let Some(dst) = self.pages.get_mut(dst_id) {
            dst.keys = src.keys.clone();
            dst.values = src.values.clone();
            dst.ref_count = 1;
            dst.is_shared = false;
            dst.is_dirty = false;
        }

        Some(dst_id)
    }
}

/// PagedAttention KV Cache
///
/// Provides efficient paged memory for KV cache storage.
#[derive(Debug)]
pub struct PagedKVCache {
    /// Block pool
    block_pool: BlockPool,

    /// Logical sequences
    sequences: HashMap<u64, LogicalKVSequence>,

    /// Config
    config: PagedKVCacheConfig,

    /// Next sequence ID
    next_seq_id: u64,
}

impl PagedKVCache {
    /// Create a new paged KV cache
    pub fn new(config: PagedKVCacheConfig) -> Self {
        Self {
            block_pool: BlockPool::new(config),
            sequences: HashMap::new(),
            config,
            next_seq_id: 0,
        }
    }

    /// Create a new sequence
    pub fn create_sequence(&mut self) -> u64 {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;

        let block_id = self.block_pool.allocate().expect("No free pages");

        let sequence = LogicalKVSequence::new(seq_id, block_id);
        self.sequences.insert(seq_id, sequence);

        seq_id
    }

    /// Append tokens to a sequence
    pub fn append(
        &mut self,
        seq_id: u64,
        keys: &ArrayView2<f32>,
        values: &ArrayView2<f32>,
    ) -> Result<Vec<(usize, usize)>, String> {
        let sequence = self
            .sequences
            .get_mut(&seq_id)
            .ok_or("Sequence not found")?;

        let num_tokens = keys.nrows();
        let head_dim = keys.ncols();
        let _num_heads = self.config.num_heads;
        let page_size = self.config.page_size;

        // Calculate how many pages we need
        let mut remaining = num_tokens;
        let mut offset = sequence.num_tokens % page_size;
        let mut positions = Vec::new();

        let mut current_page_id = if offset > 0 {
            // Continue on current page
            sequence.blocks.last().map(|b| b.page_id)
        } else {
            None
        };

        while remaining > 0 {
            // Get or allocate page
            let page_id = match current_page_id {
                Some(pid) if offset > 0 && remaining < page_size - offset => pid,
                _ => {
                    // Need new page
                    let new_page_id = self.block_pool.allocate().ok_or("No free pages")?;
                    current_page_id = Some(new_page_id);
                    let logical_block = LogicalBlock {
                        page_id: new_page_id,
                        logical_block_id: sequence.blocks.len(),
                        offset: 0,
                        num_tokens: 0,
                    };
                    sequence.blocks.push(logical_block);
                    new_page_id
                }
            };

            // Copy data to page
            let (to_copy, new_offset) = if offset == 0 && remaining >= page_size {
                // Full page
                (page_size.min(remaining), 0)
            } else if offset > 0 {
                // Partial page continuation
                let available = page_size - offset;
                let to_copy = available.min(remaining);
                (to_copy, offset + to_copy)
            } else {
                // Start of new page
                (page_size.min(remaining), page_size)
            };

            // Copy keys and values
            // keys/values: (num_tokens, head_dim)
            // page.keys/values: (page_size, head_dim)
            if let Some(page) = self.block_pool.get_mut(page_id) {
                for i in 0..to_copy {
                    let src_row = i; // Token index within the appended batch
                    let dst_row = offset + i; // Position in page
                    // Copy entire row from keys to page
                    for d in 0..head_dim {
                        page.keys[[dst_row, d]] = keys[[src_row, d]];
                        page.values[[dst_row, d]] = values[[src_row, d]];
                    }
                }
            }

            positions.push((page_id, offset));
            sequence.append(page_id, to_copy);
            remaining -= to_copy;
            offset = new_offset;

            if offset >= page_size {
                offset = 0;
                current_page_id = None;
            }
        }

        Ok(positions)
    }

    /// Get KV data for a position
    pub fn get(
        &self,
        seq_id: u64,
        positions: &[usize],
    ) -> Result<(Array2<f32>, Array2<f32>), String> {
        let sequence = self.sequences.get(&seq_id).ok_or("Sequence not found")?;
        let num_tokens = positions.len();
        let head_dim = self.config.head_dim;

        let mut keys = Array2::zeros((num_tokens, head_dim));
        let mut values = Array2::zeros((num_tokens, head_dim));

        for (i, &pos) in positions.iter().enumerate() {
            // Find which block this position is in
            let page_size = self.config.page_size;
            let block_idx = pos / page_size;
            let offset = pos % page_size;

            if block_idx >= sequence.blocks.len() {
                continue; // Position not yet filled
            }

            let page_id = sequence.blocks[block_idx].page_id;

            if let Some(page) = self.block_pool.get(page_id) {
                // page.keys is (page_size, head_dim)
                // output is (num_tokens, head_dim)
                for d in 0..head_dim {
                    keys[[i, d]] = page.keys[[offset, d]];
                    values[[i, d]] = page.values[[offset, d]];
                }
            }
        }

        Ok((keys, values))
    }

    /// Get the number of sequences
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get the number of tokens in a sequence
    pub fn sequence_num_tokens(&self, seq_id: u64) -> Option<usize> {
        self.sequences.get(&seq_id).map(|s| s.num_tokens)
    }

    /// Get total tokens cached
    pub fn total_tokens(&self) -> usize {
        self.sequences.values().map(|s| s.num_tokens).sum()
    }

    /// Get memory utilization
    pub fn memory_utilization(&self) -> f32 {
        self.block_pool.utilization()
    }

    /// Free a sequence
    pub fn free_sequence(&mut self, seq_id: u64) {
        if let Some(sequence) = self.sequences.remove(&seq_id) {
            // Free all pages
            for block in &sequence.blocks {
                if !self.block_pool.pages[block.page_id].dec_ref() {
                    // Last reference, actually free
                    self.block_pool.free(block.page_id);
                }
            }
        }
    }

    /// Share a sequence from a parent (copy-on-write)
    pub fn share_sequence(&mut self, parent_id: u64, child_seq_id: u64) -> Result<(), String> {
        // Clone parent sequence before any mutable borrows
        let (parent_blocks, parent_clone) = {
            let parent = self
                .sequences
                .get(&parent_id)
                .ok_or("Parent sequence not found")?;
            (parent.blocks.clone(), parent.clone())
        };

        let child = self
            .sequences
            .get_mut(&child_seq_id)
            .ok_or("Child sequence not found")?;

        // Make parent's pages shared
        for block in &parent_blocks {
            if let Some(page) = self.block_pool.get_mut(block.page_id) {
                page.inc_ref();
            }
        }

        // Child references parent's blocks
        child.parent = Some(Box::new(parent_clone));

        Ok(())
    }

    /// Get statistics
    pub fn stats(&self) -> PagedKVCacheStats {
        PagedKVCacheStats {
            num_sequences: self.sequences.len(),
            total_tokens: self.total_tokens(),
            free_pages: self.block_pool.free_count(),
            total_pages: self.block_pool.total_pages(),
            utilization: self.block_pool.utilization(),
        }
    }
}

/// Statistics for PagedKVCache
#[derive(Debug, Clone)]
pub struct PagedKVCacheStats {
    /// Number of active sequences
    pub num_sequences: usize,

    /// Total tokens cached
    pub total_tokens: usize,

    /// Number of free pages
    pub free_pages: usize,

    /// Total pages
    pub total_pages: usize,

    /// Memory utilization (0.0 to 1.0)
    pub utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_operations() {
        let page = KVPage::new(0, 8, 16, 64);

        assert_eq!(page.ref_count(), 1);
        assert!(!page.is_shared());
    }

    #[test]
    fn test_block_pool() {
        let config = PagedKVCacheConfig::default();
        let mut pool = BlockPool::new(config);

        assert_eq!(pool.free_count(), config.max_num_pages);

        let _page_id = pool.allocate().unwrap();
        assert!(pool.free_count() < config.max_num_pages);

        // Note: pages are pre-allocated, so free just resets
        pool.free(0);
        assert_eq!(pool.free_count(), config.max_num_pages);
    }

    #[test]
    fn test_sequence_creation() {
        let config = PagedKVCacheConfig::default();
        let mut cache = PagedKVCache::new(config);

        let _seq_id = cache.create_sequence();
        assert_eq!(cache.num_sequences(), 1);
    }

    #[test]
    fn test_append_and_get() {
        let config = PagedKVCacheConfig::default();
        let mut cache = PagedKVCache::new(config);

        let seq_id = cache.create_sequence();

        // Create test KV data - (num_tokens, head_dim)
        let keys = Array2::zeros((8, 64));
        let values = Array2::zeros((8, 64));

        let positions = cache.append(seq_id, &keys.view(), &values.view()).unwrap();
        assert_eq!(positions.len(), 1);

        let (retrieved_k, retrieved_v) = cache.get(seq_id, &[0]).unwrap();
        assert_eq!(retrieved_k.shape(), &[1, 64]);
    }

    #[test]
    fn test_memory_utilization() {
        let config = PagedKVCacheConfig::default();
        let mut cache = PagedKVCache::new(config);

        let seq_id = cache.create_sequence();
        let keys = Array2::zeros((32, 64));
        let values = Array2::zeros((32, 64));
        cache.append(seq_id, &keys.view(), &values.view()).unwrap();

        let stats = cache.stats();
        assert!(stats.total_tokens > 0);
        assert!(stats.utilization > 0.0);
    }

    #[test]
    fn test_free_sequence() {
        let config = PagedKVCacheConfig::default();
        let mut cache = PagedKVCache::new(config);

        let seq_id = cache.create_sequence();
        let keys = Array2::zeros((16, 64));
        let values = Array2::zeros((16, 64));
        cache.append(seq_id, &keys.view(), &values.view()).unwrap();

        assert_eq!(cache.num_sequences(), 1);

        cache.free_sequence(seq_id);
        assert_eq!(cache.num_sequences(), 0);
    }
}
