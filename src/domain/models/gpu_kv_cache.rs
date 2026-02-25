//! GPU KV-Cache for Inference
//!
//! This module provides GPU-resident key-value cache for efficient autoregressive
//! inference. All cache operations remain on GPU without CPU transfer.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                         GpuKVCache                                   │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  key_cache   [batch, num_heads, max_seq_len, head_dim]              │
//! │  value_cache [batch, num_heads, max_seq_len, head_dim]              │
//! │  current_pos: usize                                                  │
//! │  max_seq_len: usize                                                  │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Efficiency
//!
//! - Pre-allocated buffers for maximum sequence length
//! - In-place updates during generation
//! - No CPU-GPU transfer during inference

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::{ModelError, Result};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

/// Configuration for GPU KV-cache
#[derive(Debug, Clone)]
pub struct GpuKVCacheConfig {
    /// Batch size
    pub batch_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Number of layers (for multi-layer cache)
    pub num_layers: usize,
}

impl GpuKVCacheConfig {
    /// Create a new cache configuration
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            num_layers,
        }
    }

    /// Size of a single layer's key or value cache in elements
    pub fn layer_cache_size(&self) -> usize {
        self.batch_size * self.num_heads * self.max_seq_len * self.head_dim
    }

    /// Total cache size in bytes for one layer (key + value)
    pub fn layer_cache_bytes(&self) -> usize {
        self.layer_cache_size() * 2 * std::mem::size_of::<f32>()
    }
}

/// GPU-resident KV-cache for a single layer
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuKVCacheLayer {
    /// Key cache [batch, num_heads, max_seq_len, head_dim]
    pub key_cache: GpuBuffer,
    /// Value cache [batch, num_heads, max_seq_len, head_dim]
    pub value_cache: GpuBuffer,
    /// Current sequence position
    pub current_pos: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Configuration
    config: GpuKVCacheConfig,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuKVCacheLayer {
    /// Create a new KV-cache layer
    pub fn new(device: &mut GpuDevice, config: GpuKVCacheConfig) -> Result<Self> {
        let cache_size = config.layer_cache_size();
        let key_cache = device.allocate_f32(cache_size)?;
        let value_cache = device.allocate_f32(cache_size)?;

        // Initialize cache to zeros
        device.fill_f32(&mut key_cache.clone(), 0.0)?;
        device.fill_f32(&mut value_cache.clone(), 0.0)?;

        Ok(Self {
            key_cache,
            value_cache,
            current_pos: 0,
            max_seq_len: config.max_seq_len,
            config,
        })
    }

    /// Append new key-value pairs to the cache
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `new_keys` - New keys [batch, num_heads, new_seq_len, head_dim]
    /// * `new_values` - New values [batch, num_heads, new_seq_len, head_dim]
    /// * `new_seq_len` - Number of new tokens
    pub fn append(
        &mut self,
        device: &mut GpuDevice,
        new_keys: &GpuBuffer,
        new_values: &GpuBuffer,
        new_seq_len: usize,
    ) -> Result<()> {
        if self.current_pos + new_seq_len > self.max_seq_len {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "KV-cache overflow: current_pos={}, new_seq_len={}, max_seq_len={}",
                    self.current_pos, new_seq_len, self.max_seq_len
                ),
            });
        }

        // Copy new keys and values to the appropriate position in cache
        // This requires a device copy operation
        let offset_elements = self.current_pos * self.config.head_dim;
        let copy_size = new_seq_len * self.config.head_dim * self.config.batch_size
            * self.config.num_heads;

        // Copy new keys to cache position
        device.copy_within_device_range(new_keys, 0, &mut self.key_cache, offset_elements, copy_size)?;

        // Copy new values to cache position
        device.copy_within_device_range(new_values, 0, &mut self.value_cache, offset_elements, copy_size)?;

        self.current_pos += new_seq_len;
        Ok(())
    }

    /// Get the current cached key-value pairs
    ///
    /// Returns references to the full cache buffers. The caller should slice
    /// to current_pos for the actual cached content.
    pub fn get_cached_kv(&self) -> (&GpuBuffer, &GpuBuffer) {
        (&self.key_cache, &self.value_cache)
    }

    /// Get current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.current_pos
    }

    /// Reset the cache for a new sequence
    pub fn reset(&mut self, device: &mut GpuDevice) -> Result<()> {
        self.current_pos = 0;
        device.fill_f32(&mut self.key_cache.clone(), 0.0)?;
        device.fill_f32(&mut self.value_cache.clone(), 0.0)?;
        Ok(())
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.current_pos == 0
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }
}

/// Multi-layer GPU KV-cache for transformer models
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuKVCache {
    /// Per-layer KV caches
    layers: Vec<GpuKVCacheLayer>,
    /// Configuration
    config: GpuKVCacheConfig,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuKVCache {
    /// Create a new multi-layer KV-cache
    pub fn new(device: &mut GpuDevice, config: GpuKVCacheConfig) -> Result<Self> {
        let layers = (0..config.num_layers)
            .map(|_| GpuKVCacheLayer::new(device, config.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers, config })
    }

    /// Get KV-cache for a specific layer
    pub fn layer(&mut self, layer_idx: usize) -> &mut GpuKVCacheLayer {
        &mut self.layers[layer_idx]
    }

    /// Get KV-cache for a specific layer (immutable)
    pub fn layer_ref(&self, layer_idx: usize) -> &GpuKVCacheLayer {
        &self.layers[layer_idx]
    }

    /// Append to all layers
    pub fn append_all(
        &mut self,
        device: &mut GpuDevice,
        new_keys: &[&GpuBuffer],
        new_values: &[&GpuBuffer],
        new_seq_len: usize,
    ) -> Result<()> {
        if new_keys.len() != self.layers.len() || new_values.len() != self.layers.len() {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "KV-cache layer mismatch: expected {}, got {} keys and {} values",
                    self.layers.len(),
                    new_keys.len(),
                    new_values.len()
                ),
            });
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.append(device, new_keys[i], new_values[i], new_seq_len)?;
        }

        Ok(())
    }

    /// Reset all layers
    pub fn reset_all(&mut self, device: &mut GpuDevice) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset(device)?;
        }
        Ok(())
    }

    /// Get current sequence length (same for all layers)
    pub fn current_seq_len(&self) -> usize {
        self.layers.first().map(|l| l.current_pos).unwrap_or(0)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get configuration
    pub fn config(&self) -> &GpuKVCacheConfig {
        &self.config
    }

    /// Total memory used by the cache in bytes
    pub fn memory_bytes(&self) -> usize {
        self.config.layer_cache_bytes() * self.layers.len()
    }
}

/// GPU KV-cache manager for efficient memory reuse
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct GpuKVCacheManager {
    /// GPU device
    device: Arc<Mutex<GpuDevice>>,
    /// Active cache (if any)
    active_cache: Option<GpuKVCache>,
    /// Default configuration
    default_config: GpuKVCacheConfig,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuKVCacheManager {
    /// Create a new cache manager
    pub fn new(device: Arc<Mutex<GpuDevice>>, config: GpuKVCacheConfig) -> Self {
        Self {
            device,
            active_cache: None,
            default_config: config,
        }
    }

    /// Get or create the active cache
    pub fn get_or_create(&mut self) -> Result<&mut GpuKVCache> {
        if self.active_cache.is_none() {
            let mut device = self.device.lock().map_err(|_| ModelError::Lock {
                message: "Failed to lock GPU device".to_string(),
            })?;
            self.active_cache = Some(GpuKVCache::new(&mut device, self.default_config.clone())?);
        }
        Ok(self.active_cache.as_mut().unwrap())
    }

    /// Reset the active cache
    pub fn reset(&mut self) -> Result<()> {
        if let Some(cache) = &mut self.active_cache {
            let mut device = self.device.lock().map_err(|_| ModelError::Lock {
                message: "Failed to lock GPU device".to_string(),
            })?;
            cache.reset_all(&mut device)?;
        }
        Ok(())
    }

    /// Clear the active cache (free memory)
    pub fn clear(&mut self) {
        self.active_cache = None;
    }

    /// Check if cache is active
    pub fn is_active(&self) -> bool {
        self.active_cache.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_config() {
        let config = GpuKVCacheConfig::new(1, 8, 512, 64, 6);
        assert_eq!(config.layer_cache_size(), 1 * 8 * 512 * 64);
        assert_eq!(config.num_layers, 6);
    }

    #[test]
    fn test_kv_cache_config_bytes() {
        let config = GpuKVCacheConfig::new(1, 8, 512, 64, 6);
        // 2 buffers (key + value) * size * 4 bytes per f32
        let expected_bytes = 2 * config.layer_cache_size() * 4;
        assert_eq!(config.layer_cache_bytes(), expected_bytes);
    }
}
