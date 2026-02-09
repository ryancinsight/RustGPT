//! Unified Streaming Workspace Abstraction
//!
//! This module provides a common trait and utilities for all streaming workspaces
//! in the codebase, enabling consistent memory management and zero-allocation hot paths.
//!
//! # Design Principles
//!
//! - **SSOT**: Single source of truth for streaming workspace behavior
//! - **SRP**: Each workspace handles one layer type
//! - **SOC**: Separation between allocation, computation, and state management
//!
//! # Research Alignment
//!
//! - Flash Attention v2: Pre-allocated scratch buffers
//! - vLLM PagedAttention: Efficient memory reuse patterns
//! - Mamba: Stateful SSM with streaming inference

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for streaming workspace dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Embedding dimension
    pub embed_dim: usize,
    /// Maximum sequence length for sliding windows
    pub max_seq_len: usize,
    /// Number of attention heads (if applicable)
    pub num_heads: usize,
    /// State dimension for SSM layers
    pub state_dim: usize,
    /// Convolution kernel size (if applicable)
    pub conv_kernel: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            embed_dim: 256,
            max_seq_len: 2048,
            num_heads: 8,
            state_dim: 16,
            conv_kernel: 4,
        }
    }
}

/// Trait for streaming workspaces that support zero-allocation inference.
///
/// All streaming workspaces must implement this trait to ensure consistent
/// behavior across different layer types.
pub trait StreamingWorkspace: Clone + Send {
    /// Reset all buffers to initial state.
    fn reset(&mut self);

    /// Get the configuration for this workspace.
    fn config(&self) -> &StreamingConfig;

    /// Estimate memory usage in bytes.
    fn memory_usage(&self) -> usize;

    /// Check if workspace is initialized for the given dimensions.
    fn is_compatible(&self, config: &StreamingConfig) -> bool;
}

/// A generic workspace for layers that don't need specialized buffers.
#[derive(Debug, Clone)]
pub struct GenericStreamingWorkspace {
    /// Main input/output buffer
    pub io_buffer: Array1<f32>,
    /// Secondary buffer for intermediate computations
    pub temp_buffer: Array1<f32>,
    /// 2D buffer for matrix operations
    pub matrix_buffer: Array2<f32>,
    /// Configuration
    config: StreamingConfig,
}

impl GenericStreamingWorkspace {
    /// Create a new generic workspace.
    #[inline]
    pub fn new(config: StreamingConfig) -> Self {
        let d = config.embed_dim;
        Self {
            io_buffer: Array1::zeros(d),
            temp_buffer: Array1::zeros(d),
            matrix_buffer: Array2::zeros((d, d)),
            config,
        }
    }

    /// Get mutable access to the IO buffer.
    #[inline]
    pub fn io(&mut self) -> &mut Array1<f32> {
        &mut self.io_buffer
    }

    /// Get mutable access to the temp buffer.
    #[inline]
    pub fn temp(&mut self) -> &mut Array1<f32> {
        &mut self.temp_buffer
    }

    /// Get mutable access to the matrix buffer.
    #[inline]
    pub fn matrix(&mut self) -> &mut Array2<f32> {
        &mut self.matrix_buffer
    }
}

impl StreamingWorkspace for GenericStreamingWorkspace {
    #[inline]
    fn reset(&mut self) {
        self.io_buffer.fill(0.0);
        self.temp_buffer.fill(0.0);
        self.matrix_buffer.fill(0.0);
    }

    #[inline]
    fn config(&self) -> &StreamingConfig {
        &self.config
    }

    #[inline]
    fn memory_usage(&self) -> usize {
        let d = self.config.embed_dim;
        // io_buffer: d * 4 bytes
        // temp_buffer: d * 4 bytes
        // matrix_buffer: d * d * 4 bytes
        (d * 2 + d * d) * 4
    }

    #[inline]
    fn is_compatible(&self, config: &StreamingConfig) -> bool {
        self.config.embed_dim >= config.embed_dim && self.config.max_seq_len >= config.max_seq_len
    }
}

/// Workspace manager for coordinating multiple streaming workspaces.
///
/// This manager provides a central point for workspace allocation and reuse,
/// reducing memory fragmentation and improving cache locality.
#[derive(Debug, Clone)]
pub struct WorkspaceManager {
    /// Generic workspaces of various sizes
    generic_small: Vec<GenericStreamingWorkspace>,
    generic_medium: Vec<GenericStreamingWorkspace>,
    generic_large: Vec<GenericStreamingWorkspace>,
    /// Statistics
    allocations_avoided: usize,
    allocations_performed: usize,
}

impl Default for WorkspaceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkspaceManager {
    /// Create a new workspace manager with pre-allocated workspaces.
    pub fn new() -> Self {
        let mut generic_small = Vec::with_capacity(4);
        let mut generic_medium = Vec::with_capacity(2);
        let mut generic_large = Vec::with_capacity(1);

        // Pre-allocate workspaces for common sizes
        for _ in 0..4 {
            generic_small.push(GenericStreamingWorkspace::new(StreamingConfig {
                embed_dim: 128,
                ..Default::default()
            }));
        }

        for _ in 0..2 {
            generic_medium.push(GenericStreamingWorkspace::new(StreamingConfig {
                embed_dim: 512,
                ..Default::default()
            }));
        }

        generic_large.push(GenericStreamingWorkspace::new(StreamingConfig {
            embed_dim: 2048,
            ..Default::default()
        }));

        Self {
            generic_small,
            generic_medium,
            generic_large,
            allocations_avoided: 0,
            allocations_performed: 0,
        }
    }

    /// Acquire a workspace suitable for the given embed dimension.
    #[inline]
    pub fn acquire(&mut self, embed_dim: usize) -> &mut GenericStreamingWorkspace {
        self.allocations_avoided += 1;

        if embed_dim <= 128 {
            let idx = self.allocations_avoided % self.generic_small.len();
            &mut self.generic_small[idx]
        } else if embed_dim <= 512 {
            let idx = self.allocations_avoided % self.generic_medium.len();
            &mut self.generic_medium[idx]
        } else {
            let idx = self.allocations_avoided % self.generic_large.len();
            &mut self.generic_large[idx]
        }
    }

    /// Get statistics about workspace usage.
    pub fn stats(&self) -> WorkspaceManagerStats {
        WorkspaceManagerStats {
            small_available: self.generic_small.len(),
            medium_available: self.generic_medium.len(),
            large_available: self.generic_large.len(),
            allocations_avoided: self.allocations_avoided,
            allocations_performed: self.allocations_performed,
        }
    }

    /// Reset all workspaces.
    pub fn reset_all(&mut self) {
        for ws in &mut self.generic_small {
            ws.reset();
        }
        for ws in &mut self.generic_medium {
            ws.reset();
        }
        for ws in &mut self.generic_large {
            ws.reset();
        }
    }
}

/// Statistics for workspace manager.
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceManagerStats {
    pub small_available: usize,
    pub medium_available: usize,
    pub large_available: usize,
    pub allocations_avoided: usize,
    pub allocations_performed: usize,
}

/// Thread-local workspace manager accessor.
#[inline]
pub fn with_workspace_manager<R>(f: impl FnOnce(&mut WorkspaceManager) -> R) -> R {
    use std::cell::RefCell;

    thread_local! {
        static MANAGER: RefCell<WorkspaceManager> = RefCell::new(WorkspaceManager::new());
    }

    MANAGER.with(|m| {
        let mut m = m.borrow_mut();
        f(&mut m)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_workspace_creation() {
        let config = StreamingConfig {
            embed_dim: 256,
            ..Default::default()
        };
        let ws = GenericStreamingWorkspace::new(config);

        assert_eq!(ws.io_buffer.len(), 256);
        assert_eq!(ws.temp_buffer.len(), 256);
        assert_eq!(ws.matrix_buffer.dim(), (256, 256));
    }

    #[test]
    fn test_generic_workspace_reset() {
        let config = StreamingConfig {
            embed_dim: 64,
            ..Default::default()
        };
        let mut ws = GenericStreamingWorkspace::new(config);

        ws.io_buffer.fill(1.0);
        ws.temp_buffer.fill(2.0);
        ws.matrix_buffer.fill(3.0);

        ws.reset();

        assert!(ws.io_buffer.iter().all(|&x| x == 0.0));
        assert!(ws.temp_buffer.iter().all(|&x| x == 0.0));
        assert!(ws.matrix_buffer.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_workspace_manager_acquire() {
        let mut manager = WorkspaceManager::new();

        let ws = manager.acquire(64);
        assert!(ws.io_buffer.len() >= 64);

        let ws = manager.acquire(256);
        assert!(ws.io_buffer.len() >= 256);

        let ws = manager.acquire(1024);
        assert!(ws.io_buffer.len() >= 1024);
    }

    #[test]
    fn test_workspace_manager_stats() {
        let mut manager = WorkspaceManager::new();

        let _ = manager.acquire(64);
        let _ = manager.acquire(128);
        let _ = manager.acquire(512);

        let stats = manager.stats();
        assert_eq!(stats.small_available, 4);
        assert_eq!(stats.medium_available, 2);
        assert_eq!(stats.large_available, 1);
        assert_eq!(stats.allocations_avoided, 3);
    }

    #[test]
    fn test_streaming_workspace_trait() {
        let config = StreamingConfig {
            embed_dim: 128,
            max_seq_len: 512,
            ..Default::default()
        };
        let ws = GenericStreamingWorkspace::new(config);

        assert_eq!(ws.config().embed_dim, 128);
        assert!(ws.memory_usage() > 0);

        let compatible_config = StreamingConfig {
            embed_dim: 64,
            max_seq_len: 256,
            ..Default::default()
        };
        assert!(ws.is_compatible(&compatible_config));

        let incompatible_config = StreamingConfig {
            embed_dim: 256,
            max_seq_len: 1024,
            ..Default::default()
        };
        assert!(!ws.is_compatible(&incompatible_config));
    }

    #[test]
    fn test_tls_workspace_manager() {
        let result = with_workspace_manager(|m| {
            let ws = m.acquire(128);
            ws.io_buffer[0] = 42.0;
            ws.io_buffer[0]
        });
        assert_eq!(result, 42.0);
    }
}
