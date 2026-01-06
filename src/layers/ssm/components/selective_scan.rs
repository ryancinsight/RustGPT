//! Selective Scan Component for SSMs
//!
//! Provides optimized selective scanning operations for state space models
//! with support for different scanning strategies and parallelization.

use ndarray::Array2;
use rayon::prelude::*;

/// Selective scan configuration
#[derive(Debug, Clone, Copy)]
pub struct SelectiveScanConfig {
    /// Use parallel processing for scanning
    pub parallel: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Numerical stability threshold
    pub stability_threshold: f32,
}

impl Default for SelectiveScanConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            chunk_size: 1024,
            stability_threshold: 1e-6,
        }
    }
}

/// Selective scan implementation
pub struct SelectiveScanner {
    config: SelectiveScanConfig,
}

impl SelectiveScanner {
    /// Create a new selective scanner with default configuration
    pub fn new() -> Self {
        Self::with_config(SelectiveScanConfig::default())
    }

    /// Create a new selective scanner with custom configuration
    pub fn with_config(config: SelectiveScanConfig) -> Self {
        Self { config }
    }

    /// Perform selective scan: y = A * x + B * u
    /// Where A is state matrix, B is input projection, x is state, u is input
    pub fn scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let _seq_len = u.nrows();
        let _state_dim = a.ncols();

        // Use adaptive scan by default for better performance
        self.adaptive_scan(a, b, u)
    }

    /// Sequential selective scan implementation
    fn sequential_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let seq_len = u.nrows();
        let state_dim = a.ncols();

        let mut y = Array2::zeros((seq_len, state_dim));
        let mut x_prev = Array2::zeros((1, state_dim));

        for t in 0..seq_len {
            // y_t = A * x_{t-1} + B * u_t
            // A is [state_dim, state_dim], x_prev is [1, state_dim]
            let a_x = x_prev.dot(a); // Result: [1, state_dim]

            // B is [state_dim, state_dim], u_t is [state_dim] (row)
            let u_row = u.row(t); // [state_dim]
            let b_u = u_row.dot(b); // Result: [state_dim]

            // Ensure both terms have same shape for addition
            let y_t = &a_x + &b_u.insert_axis(ndarray::Axis(0));
            y.row_mut(t).assign(&y_t);

            // Update state: x_t = y_t (for simple recurrence)
            x_prev.assign(&y_t);
        }

        y
    }

    /// Enhanced parallel selective scan with better load balancing and memory efficiency
    fn parallel_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let seq_len = u.nrows();
        let state_dim = a.ncols();
        let chunk_size = self.config.chunk_size;

        let mut y = Array2::zeros((seq_len, state_dim));

        // Adaptive chunking based on sequence length for better load balancing
        let adaptive_chunk_size = if seq_len > 4096 {
            chunk_size.max(2048) // Larger chunks for very long sequences
        } else if seq_len > 2048 {
            chunk_size.max(1024)
        } else {
            chunk_size.max(512) // Smaller chunks for shorter sequences
        };

        // Process chunks in parallel with better memory locality
        let indices: Vec<usize> = (0..seq_len).collect();
        let chunks: Vec<Vec<usize>> = indices
            .chunks(adaptive_chunk_size)
            .map(|c| c.to_vec())
            .collect();

        // Pre-allocate results to avoid dynamic resizing
        let mut results = vec![Array2::zeros((0, state_dim)); chunks.len()];

        // Parallel processing with optimized memory access patterns
        results.par_iter_mut().enumerate().for_each(|(i, result)| {
            let chunk = &chunks[i];
            let start = chunk[0];
            let end = chunk[chunk.len() - 1] + 1;
            let chunk_size = end - start;

            *result = Array2::zeros((chunk_size, state_dim));
            let mut x_prev = if start == 0 {
                Array2::zeros((1, state_dim))
            } else {
                y.row(start - 1).to_owned().insert_axis(ndarray::Axis(0))
            };

            // Process chunk with optimized memory access
            for (local_idx, &global_idx) in chunk.iter().enumerate() {
                // Vectorized operations for better cache utilization
                let a_x = x_prev.dot(a);
                let u_row = u.row(global_idx);
                let b_u = u_row.dot(b);

                let y_t = &a_x + &b_u.insert_axis(ndarray::Axis(0));
                result.row_mut(local_idx).assign(&y_t);
                x_prev.assign(&y_t);
            }
        });

        // Combine results with minimal copying
        let mut current_row = 0;
        for chunk_result in results {
            let chunk_rows = chunk_result.nrows();
            if chunk_rows > 0 {
                // Direct copy to avoid intermediate allocations
                for i in 0..chunk_rows {
                    y.row_mut(current_row + i).assign(&chunk_result.row(i));
                }
                current_row += chunk_rows;
            }
        }

        y
    }

    /// Optimized selective scan with numerical stability checks
    pub fn stable_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let result = self.scan(a, b, u);

        // Apply numerical stability checks
        let mut stable_result = result.clone();
        for mut row in stable_result.rows_mut() {
            for val in row.iter_mut() {
                if !val.is_finite() {
                    *val = 0.0;
                } else if val.abs() > 1.0 / self.config.stability_threshold {
                    *val = val.signum() * (1.0 / self.config.stability_threshold);
                }
            }
        }

        stable_result
    }

    /// Memory-efficient selective scan with adaptive chunking
    /// This implementation minimizes memory usage by processing smaller chunks
    /// and is particularly useful for very long sequences or memory-constrained environments
    pub fn memory_efficient_scan(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
        u: &Array2<f32>,
    ) -> Array2<f32> {
        let seq_len = u.nrows();
        let state_dim = a.ncols();

        // Adaptive chunk size based on sequence length and memory constraints
        let base_chunk_size = 256.min(seq_len / 4); // Start with conservative chunk size
        let mut chunk_size = base_chunk_size.max(64); // Minimum chunk size

        // Adjust chunk size based on sequence length for optimal memory usage
        if seq_len > 8192 {
            chunk_size = 512; // Larger chunks for very long sequences
        } else if seq_len > 4096 {
            chunk_size = 256;
        } else if seq_len > 2048 {
            chunk_size = 128;
        }

        let mut y = Array2::zeros((seq_len, state_dim));
        let mut x_prev = Array2::zeros((1, state_dim));

        // Process in chunks to minimize memory footprint
        for chunk_start in (0..seq_len).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            let _current_chunk_size = chunk_end - chunk_start;

            // Process current chunk
            for t in chunk_start..chunk_end {
                let a_x = x_prev.dot(a);
                let b_u = u.row(t).dot(b);

                let y_t = &a_x + &b_u.insert_axis(ndarray::Axis(0));
                y.row_mut(t).assign(&y_t);
                x_prev.assign(&y_t);
            }
        }

        y
    }

    /// Adaptive scan that automatically selects the best strategy based on input characteristics
    pub fn adaptive_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let seq_len = u.nrows();
        let _state_dim = a.ncols();

        // Choose scan strategy based on sequence length and configuration
        if self.config.parallel && seq_len > 1024 {
            // Use parallel scan for longer sequences
            self.parallel_scan(a, b, u)
        } else if seq_len > 4096 {
            // Use memory-efficient scan for very long sequences
            self.memory_efficient_scan(a, b, u)
        } else {
            // Use sequential scan for shorter sequences (better for small sequences)
            self.sequential_scan(a, b, u)
        }
    }

    /// Get configuration
    pub fn config(&self) -> SelectiveScanConfig {
        self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SelectiveScanConfig) {
        self.config = config;
    }
}
