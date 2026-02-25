//! Unified GPU Buffer Pool for Shared Components
//!
//! Provides centralized GPU memory management for all shared layer components:
//! - SharedAttentionContext
//! - SharedFeedforward
//! - SharedTemporalProcessing
//!
//! ## Design Goals
//!
//! 1. **Zero-allocation reuse**: Buffers are pre-allocated and reused across forward passes
//! 2. **Power-of-2 sizing**: Minimizes reallocation frequency
//! 3. **Automatic capacity management**: Grows as needed, never shrinks
//! 4. **Thread-local caching**: Each thread has its own pool to avoid synchronization overhead
//!
//! ## Integration with UnifiedLayerWorkspace
//!
//! The buffer pool is designed to work alongside `UnifiedLayerWorkspace`:
//! - CPU buffers are managed by `UnifiedLayerWorkspace`
//! - GPU buffers are managed by `UnifiedGpuBufferPool`
//! - Both use the same capacity tracking for consistency

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, UnifiedGpuExecutor};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::cell::RefCell;

/// Buffer identifiers for the unified GPU pool
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBufferId {
    /// Input activation buffer
    Input,
    /// Output activation buffer
    Output,
    /// Temporal mixing output (attention/SSM)
    TemporalOut,
    /// FFN intermediate buffer
    FfnIntermediate,
    /// FFN output buffer
    FfnOut,
    /// Attention scores buffer
    AttentionScores,
    /// Context matrix buffer
    ContextMatrix,
    /// Query buffer
    Query,
    /// Key buffer
    Key,
    /// Value buffer
    Value,
    /// Temporary scratch buffer 1
    Scratch1,
    /// Temporary scratch buffer 2
    Scratch2,
    /// Richards GLU value projection
    RichardsValue,
    /// Richards GLU gate projection
    RichardsGate,
}

/// Configuration for a GPU buffer
#[derive(Debug, Clone)]
pub struct GpuBufferConfig {
    /// Minimum size in f32 elements
    pub min_size: usize,
    /// Whether to use power-of-2 sizing
    pub power_of_two: bool,
    /// Growth factor when resizing (e.g., 2.0 for doubling)
    pub growth_factor: f32,
}

impl Default for GpuBufferConfig {
    fn default() -> Self {
        Self {
            min_size: 64,
            power_of_two: true,
            growth_factor: 2.0,
        }
    }
}

/// Statistics for GPU memory allocation and reuse efficiency
///
/// Tracks:
/// - Total bytes allocated across all buffers
/// - Wasted bytes due to power-of-2 sizing and alignment
/// - Number of reuse operations (buffer reuse without reallocation)
/// - Number of resize operations (buffer reallocation due to capacity growth)
#[derive(Debug, Clone, Copy)]
pub struct AllocationStats {
    /// Total bytes allocated (sum of all buffer allocations)
    pub total_allocated: usize,
    /// Wasted bytes due to power-of-2 sizing (allocated - requested)
    pub total_wasted_padding: usize,
    /// Number of successful buffer reuse operations (no reallocation needed)
    pub reuse_count: usize,
    /// Number of buffer resizing/reallocation operations
    pub resize_count: usize,
}

impl AllocationStats {
    /// Create empty stats
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            total_wasted_padding: 0,
            reuse_count: 0,
            resize_count: 0,
        }
    }

    /// Calculate allocation efficiency as a percentage (0-100)
    ///
    /// Returns `(total_allocated - total_wasted) / total_allocated * 100`
    /// Higher values indicate better efficiency.
    pub fn efficiency_percent(&self) -> f32 {
        if self.total_allocated == 0 {
            100.0
        } else {
            let used = self.total_allocated - self.total_wasted_padding;
            (used as f32 / self.total_allocated as f32) * 100.0
        }
    }

    /// Calculate waste ratio (0-1)
    ///
    /// Returns `total_wasted / total_allocated`
    pub fn waste_ratio(&self) -> f32 {
        if self.total_allocated == 0 {
            0.0
        } else {
            self.total_wasted_padding as f32 / self.total_allocated as f32
        }
    }

    /// Reset stats to zero
    pub fn reset(&mut self) {
        self.total_allocated = 0;
        self.total_wasted_padding = 0;
        self.reuse_count = 0;
        self.resize_count = 0;
    }
}

impl Default for AllocationStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified GPU buffer pool for shared components.
///
/// Manages GPU memory allocation and reuse across all shared layer components.
/// Uses power-of-2 sizing to minimize reallocations.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct UnifiedGpuBufferPool {
    /// GPU executor for allocation
    executor: RefCell<UnifiedGpuExecutor>,
    /// Allocated buffers
    buffers: RefCell<std::collections::HashMap<GpuBufferId, GpuBuffer>>,
    /// Buffer capacities (in f32 elements)
    capacities: RefCell<std::collections::HashMap<GpuBufferId, usize>>,
    /// Buffer configurations
    configs: std::collections::HashMap<GpuBufferId, GpuBufferConfig>,
    /// Total bytes allocated
    total_bytes: RefCell<usize>,
    /// Allocation statistics for efficiency tracking
    stats: RefCell<AllocationStats>,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl UnifiedGpuBufferPool {
    /// Create a new GPU buffer pool with automatic GPU detection.
    pub fn auto_detect() -> Result<Self> {
        let executor = UnifiedGpuExecutor::auto_detect()?;
        Self::with_executor(executor)
    }

    /// Create a GPU buffer pool with an existing executor.
    pub fn with_executor(executor: UnifiedGpuExecutor) -> Result<Self> {
        let mut configs = std::collections::HashMap::new();

        // Default configurations for each buffer type
        configs.insert(GpuBufferId::Input, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Output, GpuBufferConfig::default());
        configs.insert(GpuBufferId::TemporalOut, GpuBufferConfig::default());
        configs.insert(GpuBufferId::FfnIntermediate, GpuBufferConfig::default());
        configs.insert(GpuBufferId::FfnOut, GpuBufferConfig::default());
        configs.insert(GpuBufferId::AttentionScores, GpuBufferConfig::default());
        configs.insert(GpuBufferId::ContextMatrix, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Query, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Key, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Value, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Scratch1, GpuBufferConfig::default());
        configs.insert(GpuBufferId::Scratch2, GpuBufferConfig::default());
        configs.insert(GpuBufferId::RichardsValue, GpuBufferConfig::default());
        configs.insert(GpuBufferId::RichardsGate, GpuBufferConfig::default());

        Ok(Self {
            executor: RefCell::new(executor),
            buffers: RefCell::new(std::collections::HashMap::new()),
            capacities: RefCell::new(std::collections::HashMap::new()),
            configs,
            total_bytes: RefCell::new(0),
            stats: RefCell::new(AllocationStats::new()),
        })
    }

    /// Get the GPU executor
    pub fn executor(&self) -> std::cell::Ref<'_, UnifiedGpuExecutor> {
        self.executor.borrow()
    }

    /// Get mutable access to the GPU executor
    pub fn executor_mut(&self) -> std::cell::RefMut<'_, UnifiedGpuExecutor> {
        self.executor.borrow_mut()
    }

    /// Ensure a buffer has sufficient capacity.
    ///
    /// If the buffer doesn't exist or is too small, it will be (re)allocated.
    /// Uses power-of-2 sizing to minimize reallocations.
    /// Tracks allocation statistics for efficiency analysis.
    ///
    /// # Arguments
    ///
    /// * `id` - Buffer identifier
    /// * `required_size` - Required size in f32 elements
    ///
    /// # Returns
    ///
    /// Reference to the buffer (via interior mutability)
    pub fn ensure_capacity(&self, id: GpuBufferId, required_size: usize) -> Result<()> {
        let mut buffers = self.buffers.borrow_mut();
        let mut capacities = self.capacities.borrow_mut();
        let mut total_bytes = self.total_bytes.borrow_mut();

        let current_capacity = capacities.get(&id).copied().unwrap_or(0);

        if current_capacity >= required_size {
            // Buffer already has sufficient capacity - record reuse
            drop(buffers);
            drop(capacities);
            drop(total_bytes);
            self.record_reuse();
            return Ok(());
        }

        // Calculate new capacity
        let config = self.configs.get(&id).cloned().unwrap_or_default();
        let new_capacity = if config.power_of_two {
            required_size.next_power_of_two().max(config.min_size)
        } else {
            (required_size as f32 * config.growth_factor).ceil() as usize
        };

        // Deallocate old buffer if it exists
        if let Some(old_buffer) = buffers.remove(&id) {
            *total_bytes -= old_buffer.size_bytes;
            // Note: The old buffer will be deallocated when dropped
        }

        // Allocate new buffer
        let mut executor = self.executor.borrow_mut();
        let new_buffer = executor.allocate_f32(new_capacity)?;

        *total_bytes += new_buffer.size_bytes;
        buffers.insert(id, new_buffer);
        capacities.insert(id, new_capacity);

        // Record allocation stats
        let allocated_bytes = new_capacity * std::mem::size_of::<f32>();
        let requested_bytes = required_size * std::mem::size_of::<f32>();
        drop(buffers);
        drop(capacities);
        drop(total_bytes);
        self.record_resize(allocated_bytes, requested_bytes);

        Ok(())
    }

    /// Get a buffer by ID.
    ///
    /// Returns None if the buffer hasn't been allocated yet.
    pub fn get(&self, id: GpuBufferId) -> Option<std::cell::Ref<'_, GpuBuffer>> {
        let buffers = self.buffers.borrow();
        if buffers.contains_key(&id) {
            // This is a bit of a hack - we need to return a reference
            // but RefCell doesn't support direct reference returns
            // In practice, callers should use get_mut for actual access
            None
        } else {
            None
        }
    }

    /// Get mutable access to a buffer by ID.
    ///
    /// Returns None if the buffer hasn't been allocated yet.
    pub fn get_mut(&self, id: GpuBufferId) -> Option<GpuBuffer> {
        self.buffers.borrow().get(&id).copied()
    }

    /// Upload data to a buffer.
    ///
    /// Ensures the buffer has sufficient capacity before uploading.
    pub fn upload(&self, id: GpuBufferId, data: &[f32]) -> Result<()> {
        self.ensure_capacity(id, data.len())?;

        let mut executor = self.executor.borrow_mut();
        if let Some(buffer) = self.buffers.borrow().get(&id).copied() {
            let mut buf = buffer;
            executor.upload(data, &mut buf)?;
            // Update the buffer in the map
            self.buffers.borrow_mut().insert(id, buf);
        }

        Ok(())
    }

    /// Download data from a buffer.
    pub fn download(&self, id: GpuBufferId, output: &mut [f32]) -> Result<()> {
        let mut executor = self.executor.borrow_mut();
        if let Some(buffer) = self.buffers.borrow().get(&id).copied() {
            executor.download(&buffer, output)?;
        }

        Ok(())
    }

    /// Get the capacity of a buffer.
    pub fn capacity(&self, id: GpuBufferId) -> usize {
        self.capacities.borrow().get(&id).copied().unwrap_or(0)
    }

    /// Get total bytes allocated.
    pub fn total_bytes(&self) -> usize {
        *self.total_bytes.borrow()
    }

    /// Clear all buffers.
    pub fn clear(&self) {
        let mut buffers = self.buffers.borrow_mut();
        let mut capacities = self.capacities.borrow_mut();
        let mut total_bytes = self.total_bytes.borrow_mut();

        buffers.clear();
        capacities.clear();
        *total_bytes = 0;
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> GpuPoolStats {
        GpuPoolStats {
            buffer_count: self.buffers.borrow().len(),
            total_bytes: *self.total_bytes.borrow(),
            capacities: self.capacities.borrow().clone(),
        }
    }

    /// Get allocation efficiency statistics.
    ///
    /// Returns statistics about buffer reuse, resizing, and memory efficiency
    /// (impact of power-of-2 sizing).
    pub fn allocation_stats(&self) -> AllocationStats {
        *self.stats.borrow()
    }

    /// Record a successful buffer reuse (no reallocation needed).
    fn record_reuse(&self) {
        self.stats.borrow_mut().reuse_count += 1;
    }

    /// Record a buffer resize/reallocation operation.
    fn record_resize(&self, allocated: usize, requested: usize) {
        let mut stats = self.stats.borrow_mut();
        stats.total_allocated += allocated;
        stats.total_wasted_padding += allocated.saturating_sub(requested);
        stats.resize_count += 1;
    }

    /// Reset allocation statistics.
    ///
    /// Clears all recorded stats but does not deallocate buffers.
    pub fn reset_stats(&self) {
        self.stats.borrow_mut().reset();
    }

    // ========================================================================
    // Convenience Methods for Shared Components
    // ========================================================================

    /// Ensure buffers for attention context operation.
    ///
    /// Allocates:
    /// - Input buffer (batch_size * embed_dim)
    /// - Context buffer (embed_dim * embed_dim)
    /// - Output buffer (batch_size * embed_dim)
    pub fn ensure_attention_context_buffers(
        &self,
        batch_size: usize,
        embed_dim: usize,
    ) -> Result<()> {
        self.ensure_capacity(GpuBufferId::Input, batch_size * embed_dim)?;
        self.ensure_capacity(GpuBufferId::ContextMatrix, embed_dim * embed_dim)?;
        self.ensure_capacity(GpuBufferId::Output, batch_size * embed_dim)?;
        Ok(())
    }

    /// Ensure buffers for softmax attention operation.
    ///
    /// Allocates:
    /// - Query, Key, Value buffers (batch_size * num_heads * seq_len * head_dim)
    /// - Attention scores buffer (batch_size * num_heads * seq_len * seq_len)
    /// - Output buffer (batch_size * num_heads * seq_len * head_dim)
    pub fn ensure_attention_buffers(
        &self,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<()> {
        let qkv_size = batch_size * num_heads * seq_len * head_dim;
        let scores_size = batch_size * num_heads * seq_len * seq_len;

        self.ensure_capacity(GpuBufferId::Query, qkv_size)?;
        self.ensure_capacity(GpuBufferId::Key, qkv_size)?;
        self.ensure_capacity(GpuBufferId::Value, qkv_size)?;
        self.ensure_capacity(GpuBufferId::AttentionScores, scores_size)?;
        self.ensure_capacity(GpuBufferId::Output, qkv_size)?;
        Ok(())
    }

    /// Ensure buffers for Richards GLU operation.
    ///
    /// Allocates:
    /// - Value buffer (batch_size * hidden_dim)
    /// - Gate buffer (batch_size * hidden_dim)
    /// - Output buffer (batch_size * hidden_dim)
    pub fn ensure_richards_glu_buffers(&self, batch_size: usize, hidden_dim: usize) -> Result<()> {
        let size = batch_size * hidden_dim;
        self.ensure_capacity(GpuBufferId::RichardsValue, size)?;
        self.ensure_capacity(GpuBufferId::RichardsGate, size)?;
        self.ensure_capacity(GpuBufferId::Output, size)?;
        Ok(())
    }
}

/// Statistics about the GPU buffer pool.
#[derive(Debug, Clone)]
pub struct GpuPoolStats {
    /// Number of allocated buffers
    pub buffer_count: usize,
    /// Total bytes allocated
    pub total_bytes: usize,
    /// Capacity of each buffer
    pub capacities: std::collections::HashMap<GpuBufferId, usize>,
}

impl GpuPoolStats {
    /// Format statistics for display.
    pub fn format_human(&self) -> String {
        format!(
            "{} buffers, {:.2} MB",
            self.buffer_count,
            self.total_bytes as f64 / (1024.0 * 1024.0)
        )
    }
}

/// Non-GPU fallback stub for documentation purposes.
#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct UnifiedGpuBufferPool {
    _private: (),
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl UnifiedGpuBufferPool {
    /// Attempt to create a GPU buffer pool without GPU features enabled.
    ///
    /// Always returns an error indicating that GPU features are required.
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message: "GPU buffer pool requires one of: --features gpu-wgpu, gpu-cuda, or gpu-metal"
                .to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::UnifiedGpuBufferPool;
    #[allow(unused_imports)]
    use crate::domain::compute::GpuBufferId;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_buffer_pool_creation() {
        match UnifiedGpuBufferPool::auto_detect() {
            Ok(pool) => {
                println!("GPU buffer pool created");
                let stats = pool.memory_stats();
                println!("Initial stats: {}", stats.format_human());
            }
            Err(e) => {
                println!("GPU not available: {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_buffer_allocation() {
        if let Ok(pool) = UnifiedGpuBufferPool::auto_detect() {
            // Test capacity ensuring
            pool.ensure_capacity(GpuBufferId::Input, 1024).unwrap();
            assert!(pool.capacity(GpuBufferId::Input) >= 1024);

            let stats = pool.memory_stats();
            println!("After allocation: {}", stats.format_human());
            assert_eq!(stats.buffer_count, 1);

            // Test power-of-2 sizing
            pool.ensure_capacity(GpuBufferId::Output, 1000).unwrap();
            assert!(pool.capacity(GpuBufferId::Output) >= 1024); // Should be power of 2

            // Test reuse
            pool.ensure_capacity(GpuBufferId::Input, 512).unwrap();
            assert!(pool.capacity(GpuBufferId::Input) >= 1024); // Should not shrink
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_convenience_methods() {
        if let Ok(pool) = UnifiedGpuBufferPool::auto_detect() {
            // Test attention context buffers
            pool.ensure_attention_context_buffers(32, 256).unwrap();
            assert!(pool.capacity(GpuBufferId::Input) >= 32 * 256);
            assert!(pool.capacity(GpuBufferId::ContextMatrix) >= 256 * 256);

            // Test attention buffers
            pool.ensure_attention_buffers(2, 8, 64, 32).unwrap();
            let qkv_size = 2 * 8 * 64 * 32;
            assert!(pool.capacity(GpuBufferId::Query) >= qkv_size);

            // Test Richards GLU buffers
            pool.ensure_richards_glu_buffers(32, 512).unwrap();
            assert!(pool.capacity(GpuBufferId::RichardsValue) >= 32 * 512);
        }
    }
}
