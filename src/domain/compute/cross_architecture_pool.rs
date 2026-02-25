//! Cross-Architecture Buffer Pooling Integration
//!
//! Provides unified buffer pooling across Diffusion, SSM, and Transformer architectures.
//! This module integrates `SharedGpuMemoryPool` with the existing GPU backend variants
//! for maximum memory efficiency and zero-allocation buffer reuse.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    CrossArchitectureBufferPool                          │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐   │
//! │  │ DiffusionGpu   │  │   SsmGpu       │  │    TransformerGpu      │   │
//! │  │   Backend      │  │   Backend      │  │       Backend          │   │
//! │  └───────┬────────┘  └───────┬────────┘  └───────────┬────────────┘   │
//! │          │                   │                       │                 │
//! │          └───────────────────┼───────────────────────┘                 │
//! │                              │                                          │
//! │                              ▼                                          │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │                    SharedGpuMemoryPool                            │  │
//! │  │   (18 buffer slots: Query, Key, Value, AttentionScores, etc.)    │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! │                              │                                          │
//! │                              ▼                                          │
//! │  ┌──────────────────────────────────────────────────────────────────┐  │
//! │  │                        GpuDevice                                  │  │
//! │  │   (CUDA > Metal > Vulkan auto-detection)                         │  │
//! │  └──────────────────────────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Memory Efficiency
//!
//! | Scenario | Before | After | Savings |
//! |----------|--------|-------|---------|
//! | Transformer + SSM | 2x buffers | 1x shared | 50% |
//! | All 3 architectures | 3x buffers | 1x shared | 67% |
//! | Batch size doubling | Full realloc | Power-of-2 reuse | ~40% |

use crate::common::errors::{ModelError, Result};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, RwLock};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::{GpuBuffer, SharedBufferSlot, SharedGpuMemoryPool, SharedPoolStats};

// ============================================================================
// Cross-Architecture Buffer Pool
// ============================================================================

/// Cross-architecture buffer pool for unified memory management.
///
/// Provides a single pool that can be shared across Diffusion, SSM, and Transformer
/// GPU backends, eliminating redundant buffer allocations.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub struct CrossArchitectureBufferPool {
    /// Shared memory pool
    pool: Arc<RwLock<SharedGpuMemoryPool>>,
    /// Current capacity (batch_size, embed_dim, seq_len)
    capacity: (usize, usize, usize),
    /// Architecture flags
    architectures: ArchitectureFlags,
    /// Statistics
    stats: CrossPoolStats,
}

/// Flags indicating which architectures are using the pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct ArchitectureFlags {
    /// Transformer architecture is active
    pub transformer: bool,
    /// SSM architecture is active
    pub ssm: bool,
    /// Diffusion architecture is active
    pub diffusion: bool,
}

/// Statistics for cross-architecture buffer pool.
#[derive(Debug, Clone, Default)]
pub struct CrossPoolStats {
    /// Total buffer allocations
    pub total_allocations: usize,
    /// Buffer reuse count (across architectures)
    pub buffer_reuses: usize,
    /// Architecture switches
    pub architecture_switches: usize,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl CrossArchitectureBufferPool {
    /// Create a new cross-architecture buffer pool with automatic GPU detection.
    ///
    /// # Errors
    ///
    /// Returns an error if no GPU is detected (strict no-fallback).
    pub fn auto_detect() -> Result<Self> {
        let pool = SharedGpuMemoryPool::auto_detect()?;
        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            capacity: (0, 0, 0),
            architectures: ArchitectureFlags::default(),
            stats: CrossPoolStats::default(),
        })
    }

    /// Create a new cross-architecture buffer pool with strict Intel NPU detection.
    pub fn auto_detect_npu() -> Result<Self> {
        let pool = SharedGpuMemoryPool::auto_detect_npu()?;
        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            capacity: (0, 0, 0),
            architectures: ArchitectureFlags::default(),
            stats: CrossPoolStats::default(),
        })
    }

    /// Create with specific backend.
    pub fn with_backend(backend: crate::domain::compute_backend::ComputeBackend) -> Result<Self> {
        let pool = SharedGpuMemoryPool::with_backend(backend)?;
        Ok(Self {
            pool: Arc::new(RwLock::new(pool)),
            capacity: (0, 0, 0),
            architectures: ArchitectureFlags::default(),
            stats: CrossPoolStats::default(),
        })
    }

    /// Get a clone of the shared pool reference.
    ///
    /// This allows multiple architectures to share the same pool.
    pub fn shared_pool(&self) -> Arc<RwLock<SharedGpuMemoryPool>> {
        Arc::clone(&self.pool)
    }

    /// Ensure capacity for Transformer operations.
    ///
    /// Pre-allocates buffers for attention, feedforward, and layer norm.
    pub fn ensure_transformer_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Result<()> {
        let needs_resize = batch_size > self.capacity.0
            || embed_dim > self.capacity.1
            || seq_len > self.capacity.2;

        if needs_resize {
            let mut pool = self.pool.write().map_err(|e| ModelError::Backend {
                message: format!("Failed to lock pool: {}", e),
            })?;

            pool.ensure_attention_buffers(batch_size, embed_dim, seq_len, num_heads)?;

            // Update capacity with power-of-2 sizing
            self.capacity = (
                batch_size.next_power_of_two(),
                embed_dim.next_power_of_two(),
                seq_len.next_power_of_two(),
            );

            self.stats.total_allocations += 1;
        }

        // Track architecture usage
        if !self.architectures.transformer {
            self.architectures.transformer = true;
            self.stats.architecture_switches += 1;
        } else {
            self.stats.buffer_reuses += 1;
        }

        Ok(())
    }

    /// Ensure capacity for SSM operations.
    ///
    /// Pre-allocates buffers for selective scan and state updates.
    pub fn ensure_ssm_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
        state_dim: usize,
    ) -> Result<()> {
        let needs_resize = batch_size > self.capacity.0
            || embed_dim > self.capacity.1
            || seq_len > self.capacity.2;

        if needs_resize {
            let mut pool = self.pool.write().map_err(|e| ModelError::Backend {
                message: format!("Failed to lock pool: {}", e),
            })?;

            pool.ensure_ssm_buffers(batch_size, state_dim, embed_dim)?;

            self.capacity = (
                batch_size.next_power_of_two(),
                embed_dim.next_power_of_two(),
                seq_len.next_power_of_two(),
            );

            self.stats.total_allocations += 1;
        }

        if !self.architectures.ssm {
            self.architectures.ssm = true;
            self.stats.architecture_switches += 1;
        } else {
            self.stats.buffer_reuses += 1;
        }

        Ok(())
    }

    /// Ensure capacity for Diffusion operations.
    ///
    /// Pre-allocates buffers for noise prediction and denoising.
    pub fn ensure_diffusion_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
        latent_dim: usize,
    ) -> Result<()> {
        let needs_resize = batch_size > self.capacity.0
            || embed_dim > self.capacity.1
            || seq_len > self.capacity.2;

        if needs_resize {
            let mut pool = self.pool.write().map_err(|e| ModelError::Backend {
                message: format!("Failed to lock pool: {}", e),
            })?;

            pool.ensure_diffusion_buffers(batch_size, latent_dim)?;

            self.capacity = (
                batch_size.next_power_of_two(),
                embed_dim.next_power_of_two(),
                seq_len.next_power_of_two(),
            );

            self.stats.total_allocations += 1;
        }

        if !self.architectures.diffusion {
            self.architectures.diffusion = true;
            self.stats.architecture_switches += 1;
        } else {
            self.stats.buffer_reuses += 1;
        }

        Ok(())
    }

    /// Get buffer for a specific slot.
    pub fn get_buffer(&self, slot: SharedBufferSlot) -> Result<Option<GpuBuffer>> {
        let pool = self.pool.read().map_err(|e| ModelError::Backend {
            message: format!("Failed to lock pool: {}", e),
        })?;

        Ok(pool.get_buffer(slot))
    }

    /// Get current capacity.
    pub fn capacity(&self) -> (usize, usize, usize) {
        self.capacity
    }

    /// Get statistics.
    pub fn stats(&self) -> &CrossPoolStats {
        &self.stats
    }

    /// Get pool statistics.
    pub fn pool_stats(&self) -> Result<SharedPoolStats> {
        let pool = self.pool.read().map_err(|e| ModelError::Backend {
            message: format!("Failed to lock pool: {}", e),
        })?;
        Ok(pool.stats())
    }

    /// Reset architecture flags (call when switching primary architecture).
    pub fn reset_architecture_flags(&mut self) {
        self.architectures = ArchitectureFlags::default();
    }

    /// Check if capacity is sufficient for given dimensions.
    pub fn has_capacity(&self, batch_size: usize, embed_dim: usize, seq_len: usize) -> bool {
        batch_size <= self.capacity.0 && embed_dim <= self.capacity.1 && seq_len <= self.capacity.2
    }
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
pub struct CrossArchitectureBufferPool {
    _private: (),
}

#[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
impl CrossArchitectureBufferPool {
    pub fn auto_detect() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Cross-architecture buffer pool requires GPU features".to_string(),
        })
    }

    pub fn auto_detect_npu() -> Result<Self> {
        Err(ModelError::Backend {
            message: "Cross-architecture NPU pool requires --features gpu-wgpu".to_string(),
        })
    }
}

// ============================================================================
// Buffer Pool Integration Trait
// ============================================================================

/// Trait for integrating buffer pool with GPU backends.
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub trait BufferPoolIntegration: Sized {
    /// Attach a cross-architecture buffer pool.
    fn attach_buffer_pool(&mut self, pool: Arc<RwLock<SharedGpuMemoryPool>>);

    /// Get the attached buffer pool.
    fn buffer_pool(&self) -> Option<&Arc<RwLock<SharedGpuMemoryPool>>>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_cross_pool_creation() {
        let result = CrossArchitectureBufferPool::auto_detect();
        assert!(result.is_ok(), "Should create cross-architecture pool");
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_cross_pool_capacity() {
        let mut pool = CrossArchitectureBufferPool::auto_detect().unwrap();

        // Initially no capacity
        assert!(!pool.has_capacity(2, 64, 16));

        // Ensure transformer capacity
        pool.ensure_transformer_capacity(2, 64, 16, 4).unwrap();

        // Now has capacity
        assert!(pool.has_capacity(2, 64, 16));

        // Power-of-2 sizing means larger capacity available
        assert!(pool.has_capacity(1, 32, 8));
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_cross_pool_architecture_switching() {
        let mut pool = CrossArchitectureBufferPool::auto_detect().unwrap();

        // Use transformer
        pool.ensure_transformer_capacity(2, 64, 16, 4).unwrap();
        assert!(pool.stats().architecture_switches >= 1);

        // Switch to SSM (should trigger architecture switch, not buffer reuse)
        pool.ensure_ssm_capacity(2, 64, 16, 32).unwrap();
        assert!(pool.stats().architecture_switches >= 2);

        // Switch to diffusion
        pool.ensure_diffusion_capacity(2, 64, 16, 32).unwrap();
        assert!(pool.stats().architecture_switches >= 3);

        // Use transformer again - this should trigger buffer reuse
        pool.ensure_transformer_capacity(2, 64, 16, 4).unwrap();
        assert!(pool.stats().buffer_reuses >= 1);
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_shared_pool_reference() {
        let pool = CrossArchitectureBufferPool::auto_detect().unwrap();
        let shared = pool.shared_pool();

        // Can create multiple references
        let shared2 = pool.shared_pool();

        // Both point to same pool
        assert!(Arc::ptr_eq(&shared, &shared2));
    }
}
