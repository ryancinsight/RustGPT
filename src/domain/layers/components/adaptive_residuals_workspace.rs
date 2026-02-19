//! Shared Workspace for AdaptiveResiduals
//!
//! Provides reusable scratch buffers that can be shared across multiple layers
//! to avoid repeated allocations. Uses power-of-2 sizing for efficient pooling.

/// Workspace for adaptive residuals computation
///
/// Contains all temporary buffers needed during forward and backward passes.
/// Can be shared across multiple AdaptiveResiduals instances to reduce allocations.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveResidualsWorkspace {
    /// Per-dimension squared norms: shape (embed_dim,)
    pub nx: Vec<f64>,

    /// Per-dimension squared norms (secondary): shape (embed_dim,)
    pub ny: Vec<f64>,

    /// Per-dimension means: shape (embed_dim,)
    pub mean_x: Vec<f64>,

    /// Per-dimension means (secondary): shape (embed_dim,)
    pub mean_y: Vec<f64>,

    /// Combined means for correlation: shape (embed_dim,)
    pub mean_z: Vec<f64>,

    /// Performance metrics: shape (embed_dim,)
    pub perf_values: Vec<f64>,

    /// Per-channel scaling factors: shape (embed_dim,)
    pub channel_scales: Vec<f32>,

    /// Covariance/correlation matrix (flattened): shape (embed_dim * embed_dim,)
    pub dot: Vec<f64>,

    /// Temporary centered values: shape (embed_dim,)
    pub z: Vec<f64>,

    /// Cached capacity to avoid repeated resizing
    #[doc(hidden)]
    pub capacity: usize,
}

impl AdaptiveResidualsWorkspace {
    /// Create a new empty workspace
    pub fn new() -> Self {
        Self {
            nx: Vec::new(),
            ny: Vec::new(),
            mean_x: Vec::new(),
            mean_y: Vec::new(),
            mean_z: Vec::new(),
            perf_values: Vec::new(),
            channel_scales: Vec::new(),
            dot: Vec::new(),
            z: Vec::new(),
            capacity: 0,
        }
    }

    /// Ensure workspace is sized for the given embed_dim
    ///
    /// Uses power-of-2 rounding to minimize reallocations when dimension changes slightly.
    /// This is crucial for efficient reuse across multiple layers with similar but not identical
    /// embed dimensions.
    pub fn resize_for_dim(&mut self, embed_dim: usize) {
        // Round up to next power of 2 for efficient pooling and alignment
        let new_capacity = (embed_dim).next_power_of_two().max(32);

        if new_capacity != self.capacity {
            self.nx.resize(new_capacity, 0.0);
            self.ny.resize(new_capacity, 0.0);
            self.mean_x.resize(new_capacity, 0.0);
            self.mean_y.resize(new_capacity, 0.0);
            self.mean_z.resize(new_capacity, 0.0);
            self.perf_values.resize(new_capacity, 0.0);
            self.channel_scales.resize(new_capacity, 1.0);
            self.z.resize(new_capacity, 0.0);

            // Covariance matrix is embed_dim × embed_dim
            let matrix_size = new_capacity * new_capacity;
            self.dot.resize(matrix_size, 0.0);

            self.capacity = new_capacity;
        }

        // Clear buffers (don't deallocate, just reset values)
        self.nx.fill(0.0);
        self.ny.fill(0.0);
        self.mean_x.fill(0.0);
        self.mean_y.fill(0.0);
        self.mean_z.fill(0.0);
        self.perf_values.fill(0.0);
        self.channel_scales.fill(1.0);
        self.z.fill(0.0);
        self.dot.fill(0.0);
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let _num_vecs = 8; // nx, ny, mean_x, mean_y, mean_z, perf_values, channel_scales, z
        let scalar_size = std::mem::size_of::<f64>(); // Most are f64
        let f32_size = std::mem::size_of::<f32>(); // channel_scales are f32

        let scalar_buffers = 7 * self.capacity * scalar_size; // 7 f64 buffers
        let f32_buffers = 1 * self.capacity * f32_size; // 1 f32 buffer
        let matrix = self.capacity * self.capacity * scalar_size; // dot product matrix

        scalar_buffers + f32_buffers + matrix + std::mem::size_of::<Self>()
    }

    /// Clear all buffers without deallocating
    pub fn clear(&mut self) {
        self.nx.fill(0.0);
        self.ny.fill(0.0);
        self.mean_x.fill(0.0);
        self.mean_y.fill(0.0);
        self.mean_z.fill(0.0);
        self.perf_values.fill(0.0);
        self.channel_scales.fill(1.0);
        self.z.fill(0.0);
        self.dot.fill(0.0);
    }

    /// Check if workspace is properly initialized for given dimension
    pub fn is_ready_for(&self, embed_dim: usize) -> bool {
        self.capacity >= embed_dim && !self.nx.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_resizing_power_of_two() {
        let mut ws = AdaptiveResidualsWorkspace::new();

        // Request 100 dims -> should allocate 128 (next power of 2)
        ws.resize_for_dim(100);
        assert_eq!(ws.capacity, 128);
        assert_eq!(ws.nx.len(), 128);

        // Request 200 dims -> should reallocate to 256
        ws.resize_for_dim(200);
        assert_eq!(ws.capacity, 256);
        assert_eq!(ws.nx.len(), 256);
    }

    #[test]
    fn test_workspace_reuse_same_capacity() {
        let mut ws = AdaptiveResidualsWorkspace::new();

        ws.resize_for_dim(100);
        let cap1 = ws.capacity;
        let ptr1 = ws.nx.as_ptr();

        // Request within same power-of-2 bracket -> no reallocation
        ws.resize_for_dim(110);
        assert_eq!(ws.capacity, cap1);
        assert_eq!(ws.nx.as_ptr(), ptr1); // Same allocation
    }

    #[test]
    fn test_workspace_clear_resets_values() {
        let mut ws = AdaptiveResidualsWorkspace::new();
        ws.resize_for_dim(64);

        // Set some values
        ws.nx[0] = 42.0;
        ws.mean_x[0] = 3.14;
        ws.channel_scales[0] = 2.5;

        // Clear
        ws.clear();

        // Verify reset
        assert_eq!(ws.nx[0], 0.0);
        assert_eq!(ws.mean_x[0], 0.0);
        assert_eq!(ws.channel_scales[0], 1.0);
    }

    #[test]
    fn test_workspace_memory_accounting() {
        let mut ws = AdaptiveResidualsWorkspace::new();
        ws.resize_for_dim(128);

        let usage = ws.memory_usage_bytes();

        // Should account for 8 buffers: 7×f64 + 1×f32 + matrix + struct
        let expected_min = 7 * 128 * 8 + 128 * 4 + 128 * 128 * 8;
        assert!(usage >= expected_min);
    }

    #[test]
    fn test_workspace_is_ready() {
        let mut ws = AdaptiveResidualsWorkspace::new();

        assert!(!ws.is_ready_for(64)); // Not initialized

        ws.resize_for_dim(64);
        assert!(ws.is_ready_for(64)); // Ready
        assert!(ws.is_ready_for(50)); // Ready (capacity is 64)
        assert!(!ws.is_ready_for(100)); // Not ready (capacity too small)
    }
}
