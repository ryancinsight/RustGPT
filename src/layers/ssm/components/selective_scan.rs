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
        let seq_len = u.nrows();
        let _state_dim = a.ncols();
        
        if self.config.parallel && seq_len > self.config.chunk_size {
            self.parallel_scan(a, b, u)
        } else {
            self.sequential_scan(a, b, u)
        }
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

    /// Parallel selective scan implementation using chunking
    fn parallel_scan(&self, a: &Array2<f32>, b: &Array2<f32>, u: &Array2<f32>) -> Array2<f32> {
        let seq_len = u.nrows();
        let state_dim = a.ncols();
        let chunk_size = self.config.chunk_size;
        
        let mut y = Array2::zeros((seq_len, state_dim));
        
        // Process chunks in parallel
        let indices: Vec<_> = (0..seq_len).collect();
        let chunks: Vec<_> = indices
            .chunks(chunk_size)
            .collect();
        
        let results: Vec<Array2<f32>> = chunks
            .into_par_iter()
            .map(|chunk| {
                let start = chunk[0];
                let end = chunk[chunk.len() - 1] + 1;
                let chunk_size = end - start;
                
                let mut chunk_y = Array2::zeros((chunk_size, state_dim));
                let mut x_prev = if start == 0 {
                    Array2::zeros((1, state_dim))
                } else {
                    y.row(start - 1).to_owned().insert_axis(ndarray::Axis(0))
                };
                
                for (i, &t) in chunk.iter().enumerate() {
                    let a_x = x_prev.dot(a);
                    let b_u = u.row(t).dot(b);
                    
                    let y_t = &a_x + &b_u;
                    chunk_y.row_mut(i).assign(&y_t);
                    x_prev.assign(&y_t);
                }
                
                chunk_y
            })
            .collect();
        
        // Combine results
        let mut current_row = 0;
        for chunk_result in results {
            let chunk_rows = chunk_result.nrows();
            for i in 0..chunk_rows {
                y.row_mut(current_row + i).assign(&chunk_result.row(i));
            }
            current_row += chunk_rows;
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

    /// Get configuration
    pub fn config(&self) -> SelectiveScanConfig {
        self.config
    }

    /// Set configuration
    pub fn set_config(&mut self, config: SelectiveScanConfig) {
        self.config = config;
    }
}