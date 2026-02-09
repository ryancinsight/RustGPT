//! Workspace Pool for Zero-Allocation Hot Paths
//!
//! Provides pre-allocated, reusable workspaces for streaming inference operations.
//! This module extends the memory pool concept with type-safe workspace management
//! for specific operations like attention, normalization, and feedforward processing.
//!
//! # Design Principles
//!
//! - **Zero-Allocation Hot Paths**: All critical paths use pre-allocated buffers
//! - **Thread-Local Storage**: Eliminates contention in multi-threaded contexts
//! - **Dimension-Aware**: Workspaces track their dimensions to avoid reallocation
//! - **Type Safety**: Generic workspaces with compile-time type checking
//!
//! # Research Alignment
//!
//! - Flash Attention v2: Pre-allocated scratch buffers
//! - vLLM PagedAttention: Efficient memory reuse patterns
//! - ONNX Runtime: Workspace caching for operators

use ndarray::{Array1, Array2};
use std::cell::RefCell;

/// A reusable 1D buffer with dimension tracking.
#[derive(Debug, Clone)]
pub struct ReusableBuffer1D {
    buffer: Array1<f32>,
    capacity: usize,
}

impl ReusableBuffer1D {
    /// Create a new buffer with given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Array1::zeros(capacity),
            capacity,
        }
    }

    /// Get buffer with at least the requested capacity.
    /// Returns a view that may be larger than requested but never smaller.
    #[inline]
    pub fn get(&mut self, min_capacity: usize) -> &mut Array1<f32> {
        if min_capacity > self.capacity {
            // Grow buffer (2x strategy to reduce reallocations)
            let new_capacity = min_capacity.next_power_of_two();
            self.buffer = Array1::zeros(new_capacity);
            self.capacity = new_capacity;
        }
        // Return slice of actual requested size
        if min_capacity < self.capacity {
            // Return view of the first min_capacity elements
            // Note: We return full buffer and caller uses slice
            &mut self.buffer
        } else {
            &mut self.buffer
        }
    }

    /// Get the raw buffer without size checks.
    #[inline]
    pub fn buffer(&mut self) -> &mut Array1<f32> {
        &mut self.buffer
    }

    /// Get current capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A reusable 2D buffer with dimension tracking.
#[derive(Debug, Clone)]
pub struct ReusableBuffer2D {
    buffer: Array2<f32>,
    rows: usize,
    cols: usize,
}

impl ReusableBuffer2D {
    /// Create a new buffer with given dimensions.
    #[inline]
    pub fn with_capacity(rows: usize, cols: usize) -> Self {
        Self {
            buffer: Array2::zeros((rows, cols)),
            rows,
            cols,
        }
    }

    /// Get buffer with at least the requested dimensions.
    #[inline]
    pub fn get(&mut self, min_rows: usize, min_cols: usize) -> &mut Array2<f32> {
        if min_rows > self.rows || min_cols > self.cols {
            let new_rows = min_rows.next_power_of_two();
            let new_cols = min_cols.next_power_of_two();
            self.buffer = Array2::zeros((new_rows, new_cols));
            self.rows = new_rows;
            self.cols = new_cols;
        }
        &mut self.buffer
    }

    /// Get the raw buffer.
    #[inline]
    pub fn buffer(&mut self) -> &mut Array2<f32> {
        &mut self.buffer
    }

    /// Get current dimensions.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Thread-local workspace pool for streaming operations.
///
/// This pool maintains pre-allocated buffers of common sizes to eliminate
/// allocation overhead in the hot path of token-by-token generation.
#[derive(Debug, Clone)]
pub struct StreamingWorkspacePool {
    /// Small buffers for scalar/vector operations (up to 1024 elements)
    small_1d: Vec<ReusableBuffer1D>,
    /// Medium buffers for head-dimension operations (up to 4096 elements)
    medium_1d: Vec<ReusableBuffer1D>,
    /// Large buffers for embed-dimension operations (up to 16384 elements)
    large_1d: Vec<ReusableBuffer1D>,
    /// 2D buffers for matrix operations
    small_2d: Vec<ReusableBuffer2D>,
    /// Track pool statistics
    allocations_avoided: usize,
    allocations_performed: usize,
}

impl Default for StreamingWorkspacePool {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingWorkspacePool {
    /// Create a new workspace pool with pre-allocated buffers.
    pub fn new() -> Self {
        let mut small_1d = Vec::with_capacity(8);
        let mut medium_1d = Vec::with_capacity(4);
        let mut large_1d = Vec::with_capacity(2);
        let mut small_2d = Vec::with_capacity(4);

        // Pre-allocate common sizes
        for _ in 0..8 {
            small_1d.push(ReusableBuffer1D::with_capacity(1024));
        }
        for _ in 0..4 {
            medium_1d.push(ReusableBuffer1D::with_capacity(4096));
        }
        for _ in 0..2 {
            large_1d.push(ReusableBuffer1D::with_capacity(16384));
        }
        for _ in 0..4 {
            small_2d.push(ReusableBuffer2D::with_capacity(256, 1024));
        }

        Self {
            small_1d,
            medium_1d,
            large_1d,
            small_2d,
            allocations_avoided: 0,
            allocations_performed: 0,
        }
    }

    /// Acquire a small 1D buffer (up to 1024 elements).
    ///
    /// # Panics
    /// Panics if all buffers are exhausted (indicates pool size misconfiguration).
    #[inline]
    pub fn acquire_small_1d(&mut self) -> &mut ReusableBuffer1D {
        // Simple round-robin for now
        let idx = self.allocations_avoided % self.small_1d.len();
        self.allocations_avoided += 1;
        &mut self.small_1d[idx]
    }

    /// Acquire a medium 1D buffer (up to 4096 elements).
    #[inline]
    pub fn acquire_medium_1d(&mut self) -> &mut ReusableBuffer1D {
        let idx = self.allocations_avoided % self.medium_1d.len();
        self.allocations_avoided += 1;
        &mut self.medium_1d[idx]
    }

    /// Acquire a large 1D buffer (up to 16384 elements).
    #[inline]
    pub fn acquire_large_1d(&mut self) -> &mut ReusableBuffer1D {
        let idx = self.allocations_avoided % self.large_1d.len();
        self.allocations_avoided += 1;
        &mut self.large_1d[idx]
    }

    /// Acquire a small 2D buffer (up to 256x1024).
    #[inline]
    pub fn acquire_small_2d(&mut self) -> &mut ReusableBuffer2D {
        let idx = self.allocations_avoided % self.small_2d.len();
        self.allocations_avoided += 1;
        &mut self.small_2d[idx]
    }

    /// Get pool statistics.
    pub fn stats(&self) -> WorkspacePoolStats {
        WorkspacePoolStats {
            small_1d_available: self.small_1d.len(),
            medium_1d_available: self.medium_1d.len(),
            large_1d_available: self.large_1d.len(),
            small_2d_available: self.small_2d.len(),
            operations_served: self.allocations_avoided,
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.allocations_avoided = 0;
        self.allocations_performed = 0;
    }
}

/// Statistics for workspace pool usage.
#[derive(Debug, Clone, Copy)]
pub struct WorkspacePoolStats {
    pub small_1d_available: usize,
    pub medium_1d_available: usize,
    pub large_1d_available: usize,
    pub small_2d_available: usize,
    pub operations_served: usize,
}

/// Thread-local workspace pool accessor.
///
/// Usage:
/// ```rust,ignore
/// with_tls_workspace_pool(|pool| {
///     let buf = pool.acquire_small_1d().get(256);
///     // use buf...
/// });
/// ```
#[inline]
pub fn with_tls_workspace_pool<R>(f: impl FnOnce(&mut StreamingWorkspacePool) -> R) -> R {
    thread_local! {
        static POOL: RefCell<StreamingWorkspacePool> = RefCell::new(StreamingWorkspacePool::new());
    }

    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        f(&mut pool)
    })
}

/// Specialized workspace for polynomial attention streaming operations.
///
/// This workspace pre-allocates all buffers needed for `forward_step_into`,
/// eliminating allocation checks in the hot path.
#[derive(Debug, Clone)]
pub struct PolyAttentionWorkspace {
    /// Query projection buffer (embed_dim)
    pub q: Array1<f32>,
    /// Key projection buffer (embed_dim)
    pub k: Array1<f32>,
    /// Value projection buffer (embed_dim)
    pub v: Array1<f32>,
    /// Gating input buffer (num_heads)
    pub xw: Array1<f32>,
    /// Gating output buffer (num_heads)
    pub gate_values: Array1<f32>,
    /// Attention scores buffer (max_window_size)
    pub scores: Array1<f32>,
    /// Head output buffer (head_dim)
    pub head_out: Array1<f32>,
    /// Final output accumulator (embed_dim)
    pub output: Array1<f32>,
    /// Cached dimensions
    embed_dim: usize,
    num_heads: usize,
    #[allow(dead_code)]
    head_dim: usize,
    max_window: usize,
}

impl PolyAttentionWorkspace {
    /// Create a new workspace with exact dimensions.
    #[inline]
    pub fn new(embed_dim: usize, num_heads: usize, max_window: usize) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            q: Array1::zeros(embed_dim),
            k: Array1::zeros(embed_dim),
            v: Array1::zeros(embed_dim),
            xw: Array1::zeros(num_heads),
            gate_values: Array1::zeros(num_heads),
            scores: Array1::zeros(max_window),
            head_out: Array1::zeros(head_dim),
            output: Array1::zeros(embed_dim),
            embed_dim,
            num_heads,
            head_dim,
            max_window,
        }
    }

    /// Verify dimensions match expected (debug builds only).
    #[inline]
    pub fn verify_dimensions(&self, embed_dim: usize, num_heads: usize, max_window: usize) {
        debug_assert_eq!(self.embed_dim, embed_dim, "Workspace embed_dim mismatch");
        debug_assert_eq!(self.num_heads, num_heads, "Workspace num_heads mismatch");
        debug_assert_eq!(self.max_window, max_window, "Workspace max_window mismatch");
    }

    /// Reset all buffers to zero.
    #[inline]
    pub fn clear(&mut self) {
        self.q.fill(0.0);
        self.k.fill(0.0);
        self.v.fill(0.0);
        self.xw.fill(0.0);
        self.gate_values.fill(0.0);
        self.scores.fill(0.0);
        self.head_out.fill(0.0);
        self.output.fill(0.0);
    }
}

/// Thread-local storage for PolyAttention workspaces.
pub fn with_tls_poly_workspace<R>(
    embed_dim: usize,
    num_heads: usize,
    max_window: usize,
    f: impl FnOnce(&mut PolyAttentionWorkspace) -> R,
) -> R {
    thread_local! {
        static WORKSPACE: RefCell<Option<PolyAttentionWorkspace>> = RefCell::new(None);
    }

    WORKSPACE.with(|ws| {
        let mut ws = ws.borrow_mut();
        if ws.is_none() {
            *ws = Some(PolyAttentionWorkspace::new(embed_dim, num_heads, max_window));
        }
        let workspace = ws.as_mut().unwrap();
        workspace.verify_dimensions(embed_dim, num_heads, max_window);
        f(workspace)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reusable_buffer_1d() {
        let mut buf = ReusableBuffer1D::with_capacity(128);
        assert_eq!(buf.capacity(), 128);

        // Get buffer for smaller size
        let slice = buf.get(64);
        assert!(slice.len() >= 64);

        // Get buffer for larger size (triggers reallocation)
        let slice = buf.get(256);
        assert!(slice.len() >= 256);
        assert_eq!(buf.capacity(), 256);
    }

    #[test]
    fn test_reusable_buffer_2d() {
        let mut buf = ReusableBuffer2D::with_capacity(10, 20);
        assert_eq!(buf.dims(), (10, 20));

        let slice = buf.get(5, 10);
        assert!(slice.nrows() >= 5);
        assert!(slice.ncols() >= 10);

        // Larger request triggers reallocation
        let slice = buf.get(20, 40);
        assert_eq!(buf.dims(), (32, 64)); // next power of 2
    }

    #[test]
    fn test_workspace_pool_stats() {
        with_tls_workspace_pool(|pool| {
            pool.reset_stats();
            let _ = pool.acquire_small_1d();
            let _ = pool.acquire_medium_1d();
            let stats = pool.stats();
            assert_eq!(stats.small_1d_available, 8);
            assert_eq!(stats.operations_served, 2);
        });
    }

    #[test]
    fn test_poly_attention_workspace() {
        let mut ws = PolyAttentionWorkspace::new(128, 8, 256);
        assert_eq!(ws.q.len(), 128);
        assert_eq!(ws.xw.len(), 8);
        assert_eq!(ws.scores.len(), 256);
        assert_eq!(ws.head_out.len(), 16); // 128 / 8

        ws.clear();
        assert!(ws.q.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tls_poly_workspace() {
        let result = with_tls_poly_workspace(64, 4, 128, |ws| {
            ws.q.fill(1.0);
            ws.q[0]
        });
        assert_eq!(result, 1.0);

        // Second call should reuse the same workspace
        with_tls_poly_workspace(64, 4, 128, |ws| {
            // q should still have 1.0 from before (no auto-clear)
            assert_eq!(ws.q[0], 1.0);
        });
    }
}
