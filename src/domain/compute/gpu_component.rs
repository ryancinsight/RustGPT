//! Unified GPU Component Trait for Shared Components
//!
//! Provides a single trait that all GPU-capable shared components implement,
//! enabling consistent GPU management across Transformer, Diffusion, and SSM architectures.
//!
//! ## Design Goals
//!
//! 1. **Unified Interface**: Single trait for all GPU-capable components
//! 2. **Strict No-Fallback**: GPU operations error clearly when GPU is unavailable
//! 3. **Automatic Detection**: Auto-detect GPU with clear error messages
//! 4. **Memory Efficiency**: Shared buffer pool management
//!
//! ## Usage
//!
//! ```ignore
//! // Enable GPU for any component implementing GpuComponent
//! let mut component = SharedFeedforward::new(...);
//! component.enable_gpu_auto_detect()?;  // Errors if no GPU
//! let output = component.forward_gpu(&input)?;  // GPU-only execution
//! ```

use std::sync::{Arc, Mutex};

use super::GpuDevice;
use crate::common::errors::Result;

/// Unified trait for GPU-capable shared components.
///
/// All shared components (SharedAttentionContext, SharedFeedforward, SharedTemporalProcessing)
/// implement this trait to provide consistent GPU management.
///
/// # Strict No-Fallback Design
///
/// Unlike CPU fallback patterns, this trait requires explicit GPU availability:
/// - `enable_gpu_auto_detect()` returns an error if no GPU is detected
/// - `forward_gpu()` returns an error if GPU is not attached
/// - No silent fallback to CPU execution
///
/// This design ensures predictable performance characteristics and clear error messages
/// for troubleshooting GPU issues.
pub trait GpuComponent: Sized {
    /// Attach a pre-configured GPU device.
    ///
    /// Use this when you want to share a GPU device across multiple components.
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>);

    /// Enable GPU with automatic detection (strict no-fallback).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No GPU is detected on the system
    /// - GPU feature flags are not enabled (compile with `--features gpu-wgpu` or similar)
    /// - GPU initialization fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut component = SharedFeedforward::new(...);
    /// match component.enable_gpu_auto_detect() {
    ///     Ok(()) => println!("GPU enabled: {}", component.gpu_backend_name().unwrap()),
    ///     Err(e) => println!("GPU not available: {}", e),
    /// }
    /// ```
    fn enable_gpu_auto_detect(&mut self) -> Result<()>;

    /// Check if GPU is ready for execution.
    ///
    /// Returns true only if a GPU device is attached and initialized.
    fn is_gpu_ready(&self) -> bool;

    /// Get the GPU backend name if attached.
    ///
    /// Returns the backend name (e.g., "CUDA", "Metal", "Vulkan") or None if no GPU.
    fn gpu_backend_name(&self) -> Option<&'static str>;

    /// Get the attached GPU device for direct operations.
    ///
    /// Returns None if no GPU is attached.
    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>>;

    /// Ensure buffers have sufficient capacity for the given dimensions.
    ///
    /// This is called before GPU operations to pre-allocate or resize buffers
    /// to avoid allocation during forward passes.
    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        seq_len: usize,
    ) -> Result<()>;
}

/// Helper function for strict GPU requirement.
///
/// Returns an error if the device is not available, with a clear message
/// indicating the operation that failed.
///
/// # Example
///
/// ```ignore
/// let device = require_gpu_or_error(&self.device, "forward_gpu")?;
/// // device is now guaranteed to be valid
/// ```
pub fn require_gpu_or_error(
    device: &Option<Arc<Mutex<GpuDevice>>>,
    operation: &str,
) -> Result<Arc<Mutex<GpuDevice>>> {
    device
        .clone()
        .ok_or_else(|| crate::common::errors::ModelError::Backend {
            message: format!(
                "GPU operation '{}' requested without GPU device attached. \
             Call enable_gpu_auto_detect() first.",
                operation
            ),
        })
}

/// GPU execution statistics for performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct GpuExecutionStats {
    /// Number of GPU kernel launches
    pub kernel_launches: usize,
    /// Total bytes transferred to GPU
    pub bytes_uploaded: usize,
    /// Total bytes transferred from GPU
    pub bytes_downloaded: usize,
    /// Total GPU execution time in microseconds
    pub total_time_us: u64,
}

/// Trait for components that can report GPU execution statistics.
pub trait GpuStatsReporting {
    /// Get GPU execution statistics.
    fn gpu_stats(&self) -> GpuExecutionStats;

    /// Reset GPU execution statistics.
    fn reset_gpu_stats(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_gpu_or_error_returns_error() {
        let device: Option<Arc<Mutex<GpuDevice>>> = None;
        let result = require_gpu_or_error(&device, "test_op");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("test_op"));
        assert!(err.to_string().contains("enable_gpu_auto_detect"));
    }

    #[test]
    fn test_gpu_execution_stats_default() {
        let stats = GpuExecutionStats::default();
        assert_eq!(stats.kernel_launches, 0);
        assert_eq!(stats.bytes_uploaded, 0);
        assert_eq!(stats.bytes_downloaded, 0);
        assert_eq!(stats.total_time_us, 0);
    }
}
