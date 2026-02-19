//! Unified GPU Buffer Management for Shared Components (DEPRECATED)
//!
//! **DEPRECATION NOTICE** (Phase 5.4): This module is being consolidated into
//! `src/domain/compute/unified_gpu_buffer_pool.rs`. Please migrate to use:
//! - `GpuComponent` trait from `domain::compute`
//! - `UnifiedGpuBufferPool` for buffer management
//! - `require_gpu_device` for GPU validation
//!
//! This module will be removed in Phase 6. For new code, use the consolidated
//! GPU backend in `domain::compute` instead.
//!
//! This module provides consolidated GPU buffer management for shared components
//! (SharedAttentionContext, SharedFeedforward, SharedTemporalProcessing).
//!
//! ## Design Goals
//!
//! 1. **Zero-allocation reuse**: Buffers are pre-allocated and reused across forward passes
//! 2. **Automatic GPU detection**: Strict no-fallback mode for troubleshooting
//! 3. **Memory efficiency**: Power-of-2 sizing to minimize reallocations
//! 4. **Backend abstraction**: Works with any GPU backend (CUDA, Metal, Vulkan)
//!
//! ## Usage
//!
//! ```ignore
//! let mut manager = SharedComponentGpuManager::new();
//! manager.enable_gpu_auto_detect()?;  // Strict: errors if no GPU
//! manager.ensure_capacity(batch_size, embed_dim, seq_len)?;
//! let output = manager.forward_attention(&input, &context)?;
//! ```

use std::sync::{Arc, Mutex};

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::GpuDevice;

/// GPU buffer manager for shared components.
///
/// Manages GPU memory allocation and reuse across:
/// - Attention context operations
/// - Feedforward operations  
/// - Temporal processing operations
#[derive(Debug)]
pub struct SharedComponentGpuManager {
    /// GPU device for computation
    device: Option<Arc<Mutex<GpuDevice>>>,
    /// Last known batch size for capacity tracking
    last_batch_size: usize,
    /// Last known embedding dimension for capacity tracking
    last_embed_dim: usize,
    /// Last known sequence length for capacity tracking
    last_seq_len: usize,
    /// Whether buffers are allocated and ready
    buffers_ready: bool,
}

impl Default for SharedComponentGpuManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedComponentGpuManager {
    /// Create a new GPU buffer manager without GPU attached.
    pub fn new() -> Self {
        Self {
            device: None,
            last_batch_size: 0,
            last_embed_dim: 0,
            last_seq_len: 0,
            buffers_ready: false,
        }
    }

    /// Enable GPU with automatic detection (strict no-fallback).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU is detected
    /// - GPU feature flags are not enabled
    /// - GPU initialization fails
    pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.device = Some(Arc::new(Mutex::new(device)));
        self.buffers_ready = false; // Force reallocation on next ensure_capacity
        Ok(())
    }

    /// Attach a specific GPU device.
    pub fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.device = Some(device);
        self.buffers_ready = false;
    }

    /// Check if GPU is attached and ready.
    pub fn is_gpu_ready(&self) -> bool {
        self.device.is_some()
    }

    /// Get the GPU backend name if attached.
    pub fn backend_name(&self) -> Option<&'static str> {
        self.device
            .as_ref()
            .and_then(|d| d.lock().ok().map(|guard| guard.backend().as_str()))
    }

    /// Ensure GPU buffers have sufficient capacity.
    ///
    /// Buffers are resized only when dimensions change, using power-of-2 sizing
    /// to minimize reallocations.
    pub fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()> {
        let device = self.device.as_ref().ok_or_else(|| ModelError::Backend {
            message: "GPU device not attached. Call enable_gpu_auto_detect() first.".to_string(),
        })?;

        // Check if resize is needed
        let needs_resize = !self.buffers_ready
            || batch_size > self.last_batch_size
            || embed_dim > self.last_embed_dim
            || seq_len > self.last_seq_len;

        if needs_resize {
            let mut gpu = device.lock().map_err(|_| ModelError::Backend {
                message: "Failed to acquire GPU device lock".to_string(),
            })?;

            // Power-of-2 sizing for efficient reuse
            let capacity_batch = batch_size.next_power_of_two();
            let capacity_dim = embed_dim.next_power_of_two();
            let capacity_seq = seq_len.next_power_of_two();

            // Allocate buffers for attention context
            let context_size = capacity_dim * capacity_dim * std::mem::size_of::<f32>();
            gpu.allocate(context_size)?;

            // Allocate buffers for temporal processing
            let temporal_size = capacity_batch * capacity_dim * std::mem::size_of::<f32>();
            gpu.allocate(temporal_size)?;

            // Allocate buffers for feedforward
            let ffn_size = capacity_batch * capacity_dim * 4 * std::mem::size_of::<f32>(); // 4x for intermediate
            gpu.allocate(ffn_size)?;

            self.last_batch_size = capacity_batch;
            self.last_embed_dim = capacity_dim;
            self.last_seq_len = capacity_seq;
            self.buffers_ready = true;
        }

        Ok(())
    }

    /// Clear all GPU buffers.
    pub fn clear_buffers(&mut self) {
        // Note: GpuDevice handles buffer deallocation internally when buffers are dropped
        // We just need to mark buffers as not ready
        self.buffers_ready = false;
    }

    /// Get the attached GPU device for direct operations.
    pub fn device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.device.clone()
    }

    /// Get capacity statistics.
    pub fn capacity_info(&self) -> (usize, usize, usize, bool) {
        (
            self.last_batch_size,
            self.last_embed_dim,
            self.last_seq_len,
            self.buffers_ready,
        )
    }
}

/// Re-export from canonical location
pub use crate::domain::compute::require_gpu_or_error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_component_gpu_manager_creation() {
        let manager = SharedComponentGpuManager::new();
        assert!(!manager.is_gpu_ready());
        assert!(manager.backend_name().is_none());
        assert_eq!(manager.capacity_info(), (0, 0, 0, false));
    }

    #[test]
    fn test_shared_component_gpu_manager_strict_detection() {
        let mut manager = SharedComponentGpuManager::new();

        // This should either succeed (GPU available) or fail with clear error
        match manager.enable_gpu_auto_detect() {
            Ok(()) => {
                assert!(manager.is_gpu_ready());
                assert!(manager.backend_name().is_some());
            }
            Err(e) => {
                // Expected on machines without GPU
                assert!(!manager.is_gpu_ready());
                assert!(e.to_string().contains("GPU") || e.to_string().contains("backend"));
            }
        }
    }

    #[test]
    fn test_require_gpu_or_error() {
        let device: Option<Arc<Mutex<GpuDevice>>> = None;
        let result = require_gpu_or_error(&device, "test_op");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("test_op"));
    }
}
