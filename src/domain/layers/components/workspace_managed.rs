//! Unified workspace management trait for all block types.
//!
//! This trait consolidates the workspace patterns used by TransformerBlock,
//! DiffusionBlock, and SSM blocks into a single abstraction, eliminating
//! code duplication and enabling consistent memory optimization across
//! all architectures.

use ndarray::Array2;

/// Memory usage statistics for a managed workspace
#[derive(Debug, Clone, Copy)]
pub struct WorkspaceStats {
    /// Total allocated memory in bytes
    pub total_bytes: usize,
    /// Number of active buffers
    pub buffer_count: usize,
    /// Expected shape (rows, cols)
    pub expected_shape: Option<(usize, usize)>,
}

impl WorkspaceStats {
    /// Compute memory stats for an array of given shape
    #[inline]
    pub fn for_shape(rows: usize, cols: usize, buffer_count: usize) -> Self {
        Self {
            total_bytes: rows * cols * std::mem::size_of::<f32>() * buffer_count,
            buffer_count,
            expected_shape: Some((rows, cols)),
        }
    }

    /// Combined memory usage of multiple buffers
    #[inline]
    pub fn combined(stats: &[WorkspaceStats]) -> Self {
        let total_bytes: usize = stats.iter().map(|s| s.total_bytes).sum();
        let buffer_count: usize = stats.iter().map(|s| s.buffer_count).sum();
        Self {
            total_bytes,
            buffer_count,
            expected_shape: None,
        }
    }
}

/// Trait for blocks that manage internal workspace buffers.
///
/// This trait provides a unified interface for capacity management,
/// clearing, and memory statistics across different block types.
/// Implementations should reuse allocated buffers when possible to
/// minimize allocation overhead.
pub trait WorkspaceManaged {
    /// Ensure all workspace buffers have capacity for the given dimensions.
    ///
    /// If buffers are not yet allocated or need resizing, they are allocated
    /// with extra capacity to minimize future reallocations. The exact allocation
    /// strategy (e.g., power-of-2 sizing) is left to implementations.
    ///
    /// # Arguments
    /// * `batch_size` - Batch dimension (rows)
    /// * `seq_len` - Sequence length (cols)
    /// * `embed_dim` - Embedding dimension (used for context matrices)
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize);

    /// Clear all workspace buffers, freeing memory.
    ///
    /// After calling this, the next `ensure_capacity` call will reallocate.
    fn clear_workspace(&mut self);

    /// Return memory statistics for all managed buffers.
    ///
    /// This is useful for profiling and debugging memory usage.
    fn workspace_stats(&self) -> WorkspaceStats;

    /// Check if workspace is currently allocated.
    fn is_workspace_allocated(&self) -> bool {
        self.workspace_stats().buffer_count > 0
    }

    /// Get the currently allocated shape if available.
    fn current_workspace_shape(&self) -> Option<(usize, usize)> {
        self.workspace_stats().expected_shape
    }

    /// Reset workspace to empty state (optional to implement).
    ///
    /// Default implementation calls `clear_workspace()`.
    fn reset_workspace(&mut self) {
        self.clear_workspace();
    }
}

/// Helper trait for blocks that use streaming/stateful computation.
///
/// Extends `WorkspaceManaged` with additional state management
/// for recurrent and SSM-based layers.
pub trait StreamingWorkspaceManaged: WorkspaceManaged {
    /// Initialize streaming state for the given dimensions.
    ///
    /// Called once before streaming begins.
    fn init_streaming(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
    ) -> crate::common::errors::Result<()>;

    /// Reset streaming state between sequences.
    fn reset_streaming_state(&mut self);

    /// Check if streaming state is active
    fn is_streaming(&self) -> bool;

    /// Finalize streaming and return accumulated state if needed
    fn finalize_streaming(&mut self) -> Option<Array2<f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_stats_for_shape() {
        let stats = WorkspaceStats::for_shape(32, 64, 3);
        let expected_bytes = 32 * 64 * 3 * std::mem::size_of::<f32>();
        assert_eq!(stats.total_bytes, expected_bytes);
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.expected_shape, Some((32, 64)));
    }

    #[test]
    fn test_workspace_stats_combined() {
        let stats1 = WorkspaceStats::for_shape(32, 64, 2);
        let stats2 = WorkspaceStats::for_shape(32, 128, 1);
        let combined = WorkspaceStats::combined(&[stats1, stats2]);

        assert_eq!(combined.buffer_count, 3);
        assert_eq!(
            combined.total_bytes,
            stats1.total_bytes + stats2.total_bytes
        );
    }

    /// Mock implementation for testing
    struct MockWorkspace {
        buffers: Vec<Option<Array2<f32>>>,
        shape: Option<(usize, usize)>,
        buffer_capacity: usize,
    }

    impl MockWorkspace {
        fn new(buffer_capacity: usize) -> Self {
            Self {
                buffers: vec![None; buffer_capacity],
                shape: None,
                buffer_capacity,
            }
        }
    }

    impl WorkspaceManaged for MockWorkspace {
        fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, _embed_dim: usize) {
            if self.shape.is_none() || self.shape != Some((batch_size, seq_len)) {
                for i in 0..self.buffer_capacity {
                    self.buffers[i] = Some(Array2::zeros((batch_size, seq_len)));
                }
                self.shape = Some((batch_size, seq_len));
            }
        }

        fn clear_workspace(&mut self) {
            self.buffers.iter_mut().for_each(|b| *b = None);
            self.shape = None;
        }

        fn workspace_stats(&self) -> WorkspaceStats {
            let allocated_count = self.buffers.iter().filter(|b| b.is_some()).count();
            let (rows, cols) = self.shape.unwrap_or((0, 0));
            WorkspaceStats::for_shape(rows, cols, allocated_count)
        }
    }

    #[test]
    fn test_mock_workspace_ensure_capacity() {
        let mut ws = MockWorkspace::new(4);

        // Initially not allocated
        assert!(!ws.is_workspace_allocated());

        // Allocate
        ws.ensure_capacity(32, 64, 128);
        assert!(ws.is_workspace_allocated());
        assert_eq!(ws.current_workspace_shape(), Some((32, 64)));
        assert_eq!(ws.workspace_stats().buffer_count, 4);

        // Same size: reuse
        let old_stats = ws.workspace_stats();
        ws.ensure_capacity(32, 64, 128);
        assert_eq!(ws.workspace_stats().buffer_count, old_stats.buffer_count);

        // Different size: reallocate
        ws.ensure_capacity(64, 128, 256);
        assert_eq!(ws.current_workspace_shape(), Some((64, 128)));

        // Clear
        ws.clear_workspace();
        assert!(!ws.is_workspace_allocated());
    }
}
