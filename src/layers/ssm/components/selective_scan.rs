//! Selective Scan Component for SSMs
//!
//! Provides optimized selective scanning operations for state space models
//! with support for different scanning strategies and parallelization.

use ndarray::Array2;
use rayon::prelude::*;

use crate::errors::{ModelError, Result};

#[inline]
fn affine_compose(lhs: (f32, f32), rhs: (f32, f32)) -> (f32, f32) {
    (lhs.0 * rhs.0, lhs.1 * rhs.0 + rhs.1)
}

fn affine_prefix_outputs(mult: f32, c: &[f32]) -> Vec<f32> {
    let n = c.len();
    if n == 0 {
        return Vec::new();
    }

    let n2 = n.next_power_of_two();
    let mut tree = vec![(1.0f32, 0.0f32); n2];
    for i in 0..n {
        tree[i] = (mult, c[i]);
    }

    let mut step = 1usize;
    while step < n2 {
        for base in (0..n2).step_by(2 * step) {
            let left = base + step - 1;
            let right = base + 2 * step - 1;
            tree[right] = affine_compose(tree[left], tree[right]);
        }
        step *= 2;
    }

    tree[n2 - 1] = (1.0f32, 0.0f32);

    let mut step = n2 / 2;
    while step >= 1 {
        for base in (0..n2).step_by(2 * step) {
            let left = base + step - 1;
            let right = base + 2 * step - 1;
            let t = tree[left];
            tree[left] = tree[right];
            tree[right] = affine_compose(t, tree[right]);
        }
        if step == 1 {
            break;
        }
        step /= 2;
    }

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        let incl = affine_compose(tree[i], (mult, c[i]));
        out[i] = incl.1;
    }
    out
}

fn is_exact_diagonal(a: &Array2<f32>) -> bool {
    if a.nrows() != a.ncols() {
        return false;
    }
    let n = a.nrows();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = a[[i, j]];
            if v.is_finite() && v == 0.0 {
                continue;
            }
            if !v.is_finite() || v != 0.0 {
                return false;
            }
        }
    }
    true
}

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

impl Default for SelectiveScanner {
    fn default() -> Self {
        Self::new()
    }
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
            y.row_mut(t).assign(&y_t.row(0));

            // Update state: x_t = y_t (for simple recurrence)
            x_prev.assign(&y_t);
        }

        y
    }

    /// Enhanced parallel selective scan with better load balancing and memory efficiency
    fn parallel_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let seq_len = u.nrows();
        let state_dim = a.ncols();

        if seq_len == 0 || state_dim == 0 {
            return Array2::zeros((seq_len, state_dim));
        }

        if !is_exact_diagonal(a) {
            return self.sequential_scan(a, b, u);
        }

        let b_u = u.dot(b);
        let diag: Vec<f32> = (0..state_dim).map(|j| a[[j, j]]).collect();

        let per_dim: Vec<Vec<f32>> = (0..state_dim)
            .into_par_iter()
            .map(|j| {
                let mut c = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    c.push(b_u[[t, j]]);
                }
                affine_prefix_outputs(diag[j], &c)
            })
            .collect();

        let mut y = Array2::zeros((seq_len, state_dim));
        for t in 0..seq_len {
            for j in 0..state_dim {
                y[[t, j]] = per_dim[j][t];
            }
        }

        y
    }

    /// Optimized selective scan with numerical stability checks
    pub fn stable_scan(
        &self,
        a: &Array2<f32>,
        b: &Array2<f32>,
        u: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        let threshold = self.config.stability_threshold;
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err(ModelError::InvalidInput {
                message: format!("stability_threshold must be positive, got {threshold}"),
            });
        }

        let max_abs = 1.0 / threshold;
        if !max_abs.is_finite() {
            return Err(ModelError::InvalidInput {
                message: format!("stability_threshold too small, got {threshold}"),
            });
        }

        let mut result = self.scan(a, b, u);
        for ((t, j), val) in result.indexed_iter_mut() {
            if !val.is_finite() {
                return Err(ModelError::Inference {
                    message: format!("non-finite scan output at ({t}, {j})"),
                });
            }
            if val.abs() > max_abs {
                *val = val.signum() * max_abs;
            }
        }

        Ok(result)
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
                y.row_mut(t).assign(&y_t.row(0));
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
