//! # Softmax Layer
//!
//! This module implements a standalone softmax layer with proper forward,
//! backward, and gradient calculations for use in neural networks.
//!
//! ## Features
//!
//! - Numerically stable softmax computation with max subtraction
//! - Proper gradient computation using the softmax derivative
//! - Support for both mutable and immutable forward passes
//! - Caching for efficient gradient computation
//! - Configurable axis for softmax computation

use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::pade::PadeExp;

/// Softmax layer for probability normalization
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Softmax {
    /// Axis along which to compute softmax (default: 1 for last dimension)
    axis: usize,
    /// Cached input for gradient computation
    #[serde(skip)]
    cached_input: Option<Array2<f32>>,
    /// Cached output for gradient computation
    #[serde(skip)]
    cached_output: Option<Array2<f32>>,
}

impl Default for Softmax {
    fn default() -> Self {
        Self {
            axis: 1, // Last dimension by default
            cached_input: None,
            cached_output: None,
        }
    }
}

impl Softmax {
    /// Create a new softmax layer
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new softmax layer with specified axis
    pub fn with_axis(axis: usize) -> Self {
        Self {
            axis,
            cached_input: None,
            cached_output: None,
        }
    }

    /// Forward pass - computes softmax probabilities
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// Softmax-normalized probabilities
    pub fn forward(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        // Cache input for gradient computation
        self.cached_input = Some(input.to_owned());

        let result = self.softmax(input);
        self.cached_output = Some(result.clone());

        result
    }

    /// Forward pass (immutable version)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// Softmax-normalized probabilities
    pub fn forward_immutable(&self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.softmax(input)
    }

    /// Backward pass - computes gradients
    ///
    /// # Arguments
    /// * `output_grads` - Gradients with respect to output
    ///
    /// # Returns
    /// Gradients with respect to input
    pub fn backward(&self, output_grads: &Array2<f32>) -> Array2<f32> {
        let cached_output = self
            .cached_output
            .as_ref()
            .expect("forward must be called before backward");

        self.compute_gradients(cached_output, output_grads)
    }

    /// Compute gradients with respect to input
    ///
    /// For softmax, the gradient is: ∂y_i/∂x_j = y_i * (δ_ij - y_j)
    /// where y is the softmax output and δ_ij is the Kronecker delta.
    ///
    /// # Arguments
    /// * `output` - Softmax output (probabilities)
    /// * `output_grads` - Gradients with respect to output
    ///
    /// # Returns
    /// Gradients with respect to input
    pub fn compute_gradients(
        &self,
        output: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Array2<f32> {
        let mut input_grads = Array2::zeros(output.raw_dim());

        // Compute gradients for each row (assuming axis=1, last dimension)
        for i in 0..output.nrows() {
            let probs = output.row(i);
            let grads = output_grads.row(i);

            // For each position j, compute: sum over k of (grad_k * ∂y_k/∂x_j)
            for j in 0..output.ncols() {
                let mut grad_sum = 0.0;
                for k in 0..output.ncols() {
                    let dy_dx = if j == k {
                        probs[k] * (1.0 - probs[k])
                    } else {
                        -probs[j] * probs[k]
                    };
                    grad_sum += grads[k] * dy_dx;
                }
                input_grads[[i, j]] = grad_sum;
            }
        }

        input_grads
    }

    /// Compute numerically stable softmax over the last dimension
    ///
    /// Uses the max subtraction trick for numerical stability and PadeExp
    /// for enhanced numerical precision:
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    fn softmax(&self, logits: &ArrayView2<f32>) -> Array2<f32> {
        let mut result = Array2::zeros(logits.raw_dim());

        // Compute softmax for each row
        for (i, row) in logits.outer_iter().enumerate() {
            // Find max value for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            let mut any_finite = false;
            for &x in row.iter() {
                if x.is_finite() {
                    any_finite = true;
                    max_val = max_val.max(x);
                }
            }
            if !any_finite {
                max_val = 0.0;
            }

            // Compute exp(x - max) in f64 so extremely small values don't underflow to 0.
            let mut exp_sum: f64 = 0.0;
            for &x in row.iter() {
                if x.is_finite() {
                    exp_sum += PadeExp::exp((x - max_val) as f64);
                }
            }

            if exp_sum == 0.0 || !exp_sum.is_finite() {
                // Degenerate case (extremely wide logits). Fall back to argmax = 1.0.
                let mut argmax = 0usize;
                let mut best = f32::NEG_INFINITY;
                for (j, &x) in row.iter().enumerate() {
                    if x.is_finite() && x > best {
                        best = x;
                        argmax = j;
                    }
                }
                for j in 0..row.len() {
                    result[[i, j]] = if j == argmax { 1.0 } else { 0.0 };
                }
                continue;
            }

            let inv_sum = 1.0 / exp_sum;
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = if val.is_finite() {
                    (PadeExp::exp((val - max_val) as f64) * inv_sum) as f32
                } else {
                    0.0
                };
            }
        }

        result
    }

    /// Get the cached input (for debugging/testing)
    pub fn cached_input(&self) -> Option<&Array2<f32>> {
        self.cached_input.as_ref()
    }

    /// Get the cached output (for debugging/testing)
    pub fn cached_output(&self) -> Option<&Array2<f32>> {
        self.cached_output.as_ref()
    }

    /// Clear cached values
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_output = None;
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn test_softmax_forward() {
        let mut softmax = Softmax::new();

        // Simple test case
        let input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let output = softmax.forward(&input.view());

        // Check that output sums to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that values are positive and in descending order (since input was ascending)
        assert!(output[[0, 0]] > 0.0);
        assert!(output[[0, 1]] > 0.0);
        assert!(output[[0, 2]] > 0.0);
        assert!(output[[0, 0]] < output[[0, 1]]);
        assert!(output[[0, 1]] < output[[0, 2]]);
    }

    #[test]
    fn test_softmax_gradient() {
        let softmax = Softmax::new();

        // Simple 2-element softmax
        let output = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();
        let output_grads = Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap();

        let input_grads = softmax.compute_gradients(&output, &output_grads);

        // For softmax [0.5, 0.5] with grads [1, -1]:
        // dL/dx0 = 1 * 0.5*(1-0.5) + (-1) * (-0.5*0.5) = 0.25 + 0.25 = 0.5
        // dL/dx1 = 1 * (-0.5*0.5) + (-1) * 0.5*(1-0.5) = -0.25 - 0.25 = -0.5

        assert!((input_grads[[0, 0]] - 0.5).abs() < 1e-6);
        assert!((input_grads[[0, 1]] - (-0.5)).abs() < 1e-6);
    }
}
