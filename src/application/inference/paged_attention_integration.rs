//! PagedAttention Integration Layer

use ndarray::Array2;
use std::sync::{Arc, Mutex};

use crate::domain::attention::paged_attention::{PagedKVCache, PagedKVCacheConfig};

pub type KVArray = Array2<f32>;

#[derive(Debug, Clone, Copy)]
pub struct CacheSequenceHandle(u64);

impl CacheSequenceHandle {
    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub num_sequences: usize,
    pub total_tokens: usize,
    pub free_pages: usize,
    pub total_pages: usize,
    pub utilization: f32,
}

#[derive(Clone)]
pub struct PagedAttentionIntegration {
    cache: Arc<Mutex<PagedKVCache>>,
    #[allow(dead_code)]
    config: PagedKVCacheConfig,
    next_handle: Arc<Mutex<u64>>,
}

impl PagedAttentionIntegration {
    pub fn new(config: PagedKVCacheConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(PagedKVCache::new(config))),
            config,
            next_handle: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(PagedKVCacheConfig::default())
    }

    pub fn inference_optimized() -> Self {
        Self::new(PagedKVCacheConfig::inference_optimized())
    }

    pub fn create_sequence(&self) -> CacheSequenceHandle {
        let mut cache = self.cache.lock().unwrap();
        let mut next = self.next_handle.lock().unwrap();

        let _seq_id = cache.create_sequence();
        let handle = CacheSequenceHandle(*next);
        *next += 1;

        handle
    }

    pub fn append(
        &self,
        handle: CacheSequenceHandle,
        keys: &Array2<f32>,
        values: &Array2<f32>,
    ) -> Result<(), String> {
        let mut cache = self.cache.lock().unwrap();
        cache.append(handle.id(), &keys.view(), &values.view())?;
        Ok(())
    }

    pub fn get(
        &self,
        handle: CacheSequenceHandle,
        positions: &[usize],
    ) -> Result<(KVArray, KVArray), String> {
        let cache = self.cache.lock().unwrap();
        cache.get(handle.id(), positions)
    }

    pub fn free(&self, handle: CacheSequenceHandle) {
        let mut cache = self.cache.lock().unwrap();
        cache.free_sequence(handle.id());
    }

    pub fn share_sequence(
        &self,
        parent: CacheSequenceHandle,
        child: CacheSequenceHandle,
    ) -> Result<(), String> {
        let mut cache = self.cache.lock().unwrap();
        cache.share_sequence(parent.id(), child.id())
    }

    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        let s = cache.stats();
        CacheStats {
            num_sequences: s.num_sequences,
            total_tokens: s.total_tokens,
            free_pages: s.free_pages,
            total_pages: s.total_pages,
            utilization: s.utilization,
        }
    }

    pub fn memory_utilization(&self) -> f32 {
        let cache = self.cache.lock().unwrap();
        cache.memory_utilization()
    }

    pub fn num_sequences(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.num_sequences()
    }

    pub fn total_tokens(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.total_tokens()
    }

    pub fn sequence_num_tokens(&self, handle: CacheSequenceHandle) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.sequence_num_tokens(handle.id()).unwrap_or(0)
    }
}

pub type SharedPagedAttention = Arc<PagedAttentionIntegration>;

impl Default for PagedAttentionIntegration {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sequence() {
        let integration = PagedAttentionIntegration::with_defaults();
        let _handle = integration.create_sequence();
        assert_eq!(integration.num_sequences(), 1);
    }

    #[test]
    fn test_append_and_get() {
        let integration = PagedAttentionIntegration::with_defaults();
        let handle = integration.create_sequence();

        let keys = Array2::zeros((16, 64));
        let values = Array2::zeros((16, 64));

        integration.append(handle, &keys, &values).unwrap();

        let (retrieved_k, _retrieved_v) = integration.get(handle, &[0, 1, 2]).unwrap();
        assert_eq!(retrieved_k.shape()[0], 3);
    }

    #[test]
    fn test_free_sequence() {
        let integration = PagedAttentionIntegration::with_defaults();
        let handle = integration.create_sequence();

        let keys = Array2::zeros((16, 64));
        let values = Array2::zeros((16, 64));
        integration.append(handle, &keys, &values).unwrap();

        assert_eq!(integration.num_sequences(), 1);

        integration.free(handle);

        assert_eq!(integration.num_sequences(), 0);
    }

    #[test]
    fn test_stats() {
        let integration = PagedAttentionIntegration::with_defaults();
        let _handle = integration.create_sequence();

        let stats = integration.stats();
        assert_eq!(stats.num_sequences, 1);
        assert!(stats.utilization >= 0.0);
    }

    #[test]
    fn test_inference_optimized() {
        let integration = PagedAttentionIntegration::inference_optimized();
        let stats = integration.stats();

        assert!(stats.total_pages > 1000);
    }
}
