//! Shared GPU Memory Pool for Cross-Architecture Memory Sharing
//!
//! Provides a centralized memory pool that can be shared across Diffusion, SSM,
//! and Transformer GPU backends, eliminating duplicate allocations and improving
//! memory efficiency.
//!
//! ## Architecture
//!
//! ```text
//!                    SharedGpuMemoryPool
//!                           |
//!           +---------------+---------------+
//!           |               |               |
//!     DiffusionGpu      SsmGpu       TransformerGpu
//!       Backend           Backend          Backend
//!           |               |               |
//!           +---------------+---------------+
//!                           |
//!                    UnifiedGpuKernels
//! ```
//!
//! ## Memory Efficiency Benefits
//!
//! 1. **Single allocation per buffer type**: Instead of each backend allocating
//!    its own Q, K, V buffers, they share a single allocation
//! 2. **Power-of-2 sizing**: Buffers use power-of-2 sizing to minimize reallocations
//! 3. **Lazy allocation**: Buffers are only allocated when first needed
//! 4. **Thread-safe access**: Uses RwLock for concurrent read access

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::collections::HashMap;
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, RwLock};

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, GpuDevice};

/// Buffer slot identifiers for the shared memory pool.
///
/// Each slot represents a logical buffer that can be shared across architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SharedBufferSlot {
    // === Attention/Temporal Buffers ===
    /// Query projection buffer (batch * heads * seq * head_dim)
    Query,
    /// Key projection buffer
    Key,
    /// Value projection buffer
    Value,
    /// Attention scores buffer (batch * heads * seq * seq)
    AttentionScores,
    /// Attention output buffer
    AttentionOutput,

    // === SSM/Recurrent Buffers ===
    /// SSM state buffer (batch * state_dim)
    SsmState,
    /// SSM intermediate buffer
    SsmIntermediate,
    /// SSM output buffer
    SsmOutput,

    // === Diffusion Buffers ===
    /// Diffusion noise buffer
    DiffusionNoise,
    /// Diffusion latent buffer
    DiffusionLatent,
    /// Diffusion output buffer
    DiffusionOutput,

    // === Shared Buffers ===
    /// Input activation buffer
    Input,
    /// Output activation buffer
    Output,
    /// Intermediate buffer for FFN
    FfnIntermediate,
    /// Scratch buffer 1 (general purpose)
    Scratch1,
    /// Scratch buffer 2 (general purpose)
    Scratch2,
    /// Weight matrix buffer
    Weight,
    /// Gradient buffer (for backward pass)
    Gradient,
}

/// Configuration for a shared buffer slot.
#[derive(Debug, Clone)]
pub struct SharedBufferConfig {
    /// Minimum size in f32 elements
    pub min_size: usize,
    /// Whether to use power-of-2 sizing
    pub power_of_two: bool,
    /// Growth factor when resizing
    pub growth_factor: f32,
    /// Whether this buffer is architecture-specific
    pub is_shared: bool,
}

impl Default for SharedBufferConfig {
    fn default() -> Self {
        Self {
            min_size: 64,
            power_of_two: true,
            growth_factor: 2.0,
            is_shared: true,
        }
    }
}

/// Statistics for the shared memory pool.
#[derive(Debug, Clone, Default)]
pub struct SharedPoolStats {
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Number of buffer slots in use
    pub slots_in_use: usize,
    /// Number of architectures sharing the pool
    pub architecture_count: usize,
    /// Reuse count (successful buffer reuse without reallocation)
    pub reuse_count: usize,
    /// Resize count (buffer reallocations)
    pub resize_count: usize,
    /// Peak memory usage
    pub peak_bytes: usize,
}

/// Shared GPU memory pool for cross-architecture memory sharing.
///
/// This is the central memory management component that allows Diffusion, SSM,
/// and Transformer backends to share GPU memory allocations.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct SharedGpuMemoryPool {
    /// GPU device for allocations
    device: Arc<RwLock<GpuDevice>>,
    /// Allocated buffers by slot
    buffers: RwLock<HashMap<SharedBufferSlot, GpuBuffer>>,
    /// Buffer capacities (in f32 elements)
    capacities: RwLock<HashMap<SharedBufferSlot, usize>>,
    /// Buffer configurations
    configs: HashMap<SharedBufferSlot, SharedBufferConfig>,
    /// Statistics
    stats: RwLock<SharedPoolStats>,
    /// Architecture reference count
    architecture_refs: RwLock<usize>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl SharedGpuMemoryPool {
    /// Create a new shared memory pool with automatic GPU detection.
    pub fn auto_detect() -> Result<Self> {
        let device = GpuDevice::auto_detect()?;
        Self::with_device(Arc::new(RwLock::new(device)))
    }

    /// Create a shared memory pool with a specific backend.
    pub fn with_backend(backend: crate::domain::compute_backend::ComputeBackend) -> Result<Self> {
        let device = GpuDevice::new(backend)?;
        Self::with_device(Arc::new(RwLock::new(device)))
    }

    /// Create a shared memory pool with an existing device.
    pub fn with_device(device: Arc<RwLock<GpuDevice>>) -> Result<Self> {
        let mut configs = HashMap::new();

        // Default configurations for all buffer slots
        for slot in [
            SharedBufferSlot::Query,
            SharedBufferSlot::Key,
            SharedBufferSlot::Value,
            SharedBufferSlot::AttentionScores,
            SharedBufferSlot::AttentionOutput,
            SharedBufferSlot::SsmState,
            SharedBufferSlot::SsmIntermediate,
            SharedBufferSlot::SsmOutput,
            SharedBufferSlot::DiffusionNoise,
            SharedBufferSlot::DiffusionLatent,
            SharedBufferSlot::DiffusionOutput,
            SharedBufferSlot::Input,
            SharedBufferSlot::Output,
            SharedBufferSlot::FfnIntermediate,
            SharedBufferSlot::Scratch1,
            SharedBufferSlot::Scratch2,
            SharedBufferSlot::Weight,
            SharedBufferSlot::Gradient,
        ] {
            configs.insert(slot, SharedBufferConfig::default());
        }

        Ok(Self {
            device,
            buffers: RwLock::new(HashMap::new()),
            capacities: RwLock::new(HashMap::new()),
            configs,
            stats: RwLock::new(SharedPoolStats::default()),
            architecture_refs: RwLock::new(0),
        })
    }

    /// Register an architecture as using this pool.
    ///
    /// Increments the reference count. The pool should not be dropped
    /// until all architectures have unregistered.
    pub fn register_architecture(&self) {
        let mut refs = self.architecture_refs.write().unwrap();
        *refs += 1;
        let mut stats = self.stats.write().unwrap();
        stats.architecture_count = *refs;
    }

    /// Unregister an architecture from this pool.
    ///
    /// Decrements the reference count.
    pub fn unregister_architecture(&self) {
        let mut refs = self.architecture_refs.write().unwrap();
        *refs = refs.saturating_sub(1);
        let mut stats = self.stats.write().unwrap();
        stats.architecture_count = *refs;
    }

    /// Get the GPU device.
    pub fn device(&self) -> Arc<RwLock<GpuDevice>> {
        self.device.clone()
    }

    /// Ensure a buffer slot has sufficient capacity.
    ///
    /// If the buffer doesn't exist or is too small, it will be (re)allocated.
    /// Uses power-of-2 sizing to minimize reallocations.
    ///
    /// # Arguments
    ///
    /// * `slot` - Buffer slot identifier
    /// * `required_size` - Required size in f32 elements
    ///
    /// # Returns
    ///
    /// Ok(()) if the buffer is ready for use
    pub fn ensure_capacity(&self, slot: SharedBufferSlot, required_size: usize) -> Result<()> {
        let current_capacity = self
            .capacities
            .read()
            .unwrap()
            .get(&slot)
            .copied()
            .unwrap_or(0);

        if current_capacity >= required_size {
            // Buffer already has sufficient capacity - record reuse
            self.stats.write().unwrap().reuse_count += 1;
            return Ok(());
        }

        // Calculate new capacity
        let config = self.configs.get(&slot).cloned().unwrap_or_default();
        let new_capacity = if config.power_of_two {
            required_size.next_power_of_two().max(config.min_size)
        } else {
            (required_size as f32 * config.growth_factor).ceil() as usize
        };

        // Deallocate old buffer if it exists
        {
            let mut buffers = self.buffers.write().unwrap();
            if let Some(old_buffer) = buffers.remove(&slot) {
                let mut stats = self.stats.write().unwrap();
                stats.total_bytes -= old_buffer.size_bytes;
            }
        }

        // Allocate new buffer
        let mut device = self.device.write().map_err(|_| ModelError::Backend {
            message: "Failed to acquire device lock for allocation".to_string(),
        })?;

        let size_bytes = new_capacity * std::mem::size_of::<f32>();
        let new_buffer = device.allocate(size_bytes)?;

        // Update state
        {
            let mut buffers = self.buffers.write().unwrap();
            let mut capacities = self.capacities.write().unwrap();
            let mut stats = self.stats.write().unwrap();

            stats.total_bytes += new_buffer.size_bytes;
            stats.peak_bytes = stats.peak_bytes.max(stats.total_bytes);
            stats.resize_count += 1;
            stats.slots_in_use = buffers.len() + 1;

            buffers.insert(slot, new_buffer);
            capacities.insert(slot, new_capacity);
        }

        Ok(())
    }

    /// Get a buffer by slot.
    ///
    /// Returns a clone of the GpuBuffer for the given slot.
    /// The buffer must have been allocated via `ensure_capacity` first.
    pub fn get_buffer(&self, slot: SharedBufferSlot) -> Option<GpuBuffer> {
        self.buffers.read().unwrap().get(&slot).copied()
    }

    /// Upload data to a buffer slot.
    ///
    /// Ensures capacity before uploading.
    pub fn upload(&self, slot: SharedBufferSlot, data: &[f32]) -> Result<()> {
        self.ensure_capacity(slot, data.len())?;

        let buffer = self.buffers.read().unwrap().get(&slot).copied();
        if let Some(mut buffer) = buffer {
            let mut device = self.device.write().map_err(|_| ModelError::Backend {
                message: "Failed to acquire device lock for upload".to_string(),
            })?;
            device.upload(data, &mut buffer)?;

            // Update buffer in map
            self.buffers.write().unwrap().insert(slot, buffer);
        }

        Ok(())
    }

    /// Download data from a buffer slot.
    pub fn download(&self, slot: SharedBufferSlot, output: &mut [f32]) -> Result<()> {
        let buffer = self.buffers.read().unwrap().get(&slot).copied();
        if let Some(buffer) = buffer {
            let mut device = self.device.write().map_err(|_| ModelError::Backend {
                message: "Failed to acquire device lock for download".to_string(),
            })?;
            device.download(&buffer, output)?;
        }

        Ok(())
    }

    /// Get the capacity of a buffer slot.
    pub fn capacity(&self, slot: SharedBufferSlot) -> usize {
        self.capacities
            .read()
            .unwrap()
            .get(&slot)
            .copied()
            .unwrap_or(0)
    }

    /// Get pool statistics.
    pub fn stats(&self) -> SharedPoolStats {
        self.stats.read().unwrap().clone()
    }

    /// Get total bytes allocated.
    pub fn total_bytes(&self) -> usize {
        self.stats.read().unwrap().total_bytes
    }

    /// Clear all buffers.
    ///
    /// This deallocates all GPU memory. Should only be called when
    /// no architectures are actively using the pool.
    pub fn clear(&self) -> Result<()> {
        let mut buffers = self.buffers.write().unwrap();
        let mut capacities = self.capacities.write().unwrap();
        let mut stats = self.stats.write().unwrap();

        // Deallocate all buffers
        let mut device = self.device.write().map_err(|_| ModelError::Backend {
            message: "Failed to acquire device lock for clear".to_string(),
        })?;

        for (_, buffer) in buffers.drain() {
            device.deallocate(buffer);
        }

        capacities.clear();
        stats.total_bytes = 0;
        stats.slots_in_use = 0;

        Ok(())
    }

    // ========================================================================
    // Convenience Methods for Common Operations
    // ========================================================================

    /// Ensure buffers for attention operation.
    ///
    /// Allocates Q, K, V, scores, and output buffers.
    pub fn ensure_attention_buffers(
        &self,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let qkv_size = batch_size * num_heads * seq_len * head_dim;
        let scores_size = batch_size * num_heads * seq_len * seq_len;

        self.ensure_capacity(SharedBufferSlot::Query, qkv_size)?;
        self.ensure_capacity(SharedBufferSlot::Key, qkv_size)?;
        self.ensure_capacity(SharedBufferSlot::Value, qkv_size)?;
        self.ensure_capacity(SharedBufferSlot::AttentionScores, scores_size)?;
        self.ensure_capacity(SharedBufferSlot::AttentionOutput, qkv_size)?;

        Ok(())
    }

    /// Ensure buffers for SSM operation.
    ///
    /// Allocates state, intermediate, and output buffers.
    pub fn ensure_ssm_buffers(
        &self,
        batch_size: usize,
        state_dim: usize,
        embed_dim: usize,
    ) -> Result<()> {
        self.ensure_capacity(SharedBufferSlot::SsmState, batch_size * state_dim)?;
        self.ensure_capacity(SharedBufferSlot::SsmIntermediate, batch_size * embed_dim)?;
        self.ensure_capacity(SharedBufferSlot::SsmOutput, batch_size * embed_dim)?;

        Ok(())
    }

    /// Ensure buffers for diffusion operation.
    ///
    /// Allocates noise, latent, and output buffers.
    pub fn ensure_diffusion_buffers(&self, batch_size: usize, latent_dim: usize) -> Result<()> {
        let size = batch_size * latent_dim;

        self.ensure_capacity(SharedBufferSlot::DiffusionNoise, size)?;
        self.ensure_capacity(SharedBufferSlot::DiffusionLatent, size)?;
        self.ensure_capacity(SharedBufferSlot::DiffusionOutput, size)?;

        Ok(())
    }

    /// Ensure shared input/output buffers.
    pub fn ensure_io_buffers(&self, batch_size: usize, embed_dim: usize) -> Result<()> {
        let size = batch_size * embed_dim;
        self.ensure_capacity(SharedBufferSlot::Input, size)?;
        self.ensure_capacity(SharedBufferSlot::Output, size)?;
        Ok(())
    }
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct SharedGpuMemoryPool {
    _private: (),
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl SharedGpuMemoryPool {
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Shared GPU memory pool requires GPU features".to_string(),
        })
    }
}

/// Global shared memory pool singleton.
///
/// This provides a default shared pool that can be used across all architectures
/// without explicit management. Uses lazy initialization.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub fn global_shared_pool() -> Result<&'static SharedGpuMemoryPool> {
    use std::sync::OnceLock;

    static POOL: OnceLock<Result<SharedGpuMemoryPool>> = OnceLock::new();

    POOL.get_or_init(|| SharedGpuMemoryPool::auto_detect())
        .as_ref()
        .map_err(|e| ModelError::Backend {
            message: format!("Failed to initialize global shared pool: {}", e),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_shared_pool_creation() {
        // Test that we can create a shared pool
        let pool = SharedGpuMemoryPool::auto_detect();
        assert!(
            pool.is_ok(),
            "Pool creation should succeed with GPU features"
        );
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_architecture_registration() {
        let pool = SharedGpuMemoryPool::auto_detect().unwrap();

        pool.register_architecture();
        let stats = pool.stats();
        assert_eq!(stats.architecture_count, 1);

        pool.register_architecture();
        let stats = pool.stats();
        assert_eq!(stats.architecture_count, 2);

        pool.unregister_architecture();
        let stats = pool.stats();
        assert_eq!(stats.architecture_count, 1);
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_ensure_capacity() {
        let pool = SharedGpuMemoryPool::auto_detect().unwrap();

        // Request capacity for input buffer
        let result = pool.ensure_capacity(SharedBufferSlot::Input, 1024);
        assert!(result.is_ok(), "ensure_capacity should succeed");

        // Check that capacity is power-of-2
        let capacity = pool.capacity(SharedBufferSlot::Input);
        assert!(
            capacity >= 1024,
            "Capacity should be at least requested size"
        );
        assert!(capacity.is_power_of_two(), "Capacity should be power-of-2");
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_attention_buffers() {
        let pool = SharedGpuMemoryPool::auto_detect().unwrap();

        let result = pool.ensure_attention_buffers(2, 4, 16, 32);
        assert!(result.is_ok(), "ensure_attention_buffers should succeed");

        // Check that all buffers are allocated
        assert!(pool.get_buffer(SharedBufferSlot::Query).is_some());
        assert!(pool.get_buffer(SharedBufferSlot::Key).is_some());
        assert!(pool.get_buffer(SharedBufferSlot::Value).is_some());
        assert!(pool.get_buffer(SharedBufferSlot::AttentionScores).is_some());
        assert!(pool.get_buffer(SharedBufferSlot::AttentionOutput).is_some());
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_reuse_count() {
        let pool = SharedGpuMemoryPool::auto_detect().unwrap();

        // First allocation
        pool.ensure_capacity(SharedBufferSlot::Scratch1, 256)
            .unwrap();
        let stats = pool.stats();
        assert_eq!(stats.resize_count, 1);

        // Same capacity - should reuse
        pool.ensure_capacity(SharedBufferSlot::Scratch1, 256)
            .unwrap();
        let stats = pool.stats();
        assert_eq!(stats.reuse_count, 1);
        assert_eq!(stats.resize_count, 1); // No new resize
    }
}
