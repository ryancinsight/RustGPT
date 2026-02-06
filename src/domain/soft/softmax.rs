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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::domain::pade::PadeExp;

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
        // We intentionally do not cache the input here.
        // Softmax backward only needs the softmax output (probabilities), and caching the input
        // would force an unnecessary clone of the entire tensor.
        self.cached_input = None;

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

    /// Forward pass for a single logits row (immutable).
    ///
    /// This avoids the common pattern of `row.to_owned().insert_axis(Axis(0))`.
    pub fn forward_immutable_row(&self, row: &ArrayView1<f32>) -> Array1<f32> {
        self.softmax_row(row)
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

        match self.axis {
            // Axis 1: row-wise softmax (default)
            1 => {
                for (mut input_row, (prob_row, grad_row)) in input_grads
                    .outer_iter_mut()
                    .zip(output.outer_iter().zip(output_grads.outer_iter()))
                {
                    let sum_grad_prob: f32 = prob_row
                        .iter()
                        .zip(grad_row.iter())
                        .map(|(&p, &g)| p * g)
                        .sum();

                    for (j, (&p, &g)) in prob_row.iter().zip(grad_row.iter()).enumerate() {
                        input_row[j] = p * (g - sum_grad_prob);
                    }
                }
            }

            // Axis 0: column-wise softmax
            0 => {
                let nrows = output.nrows();
                let ncols = output.ncols();

                for j in 0..ncols {
                    let mut sum_grad_prob: f32 = 0.0;
                    for i in 0..nrows {
                        sum_grad_prob += output[[i, j]] * output_grads[[i, j]];
                    }

                    for i in 0..nrows {
                        let p = output[[i, j]];
                        let g = output_grads[[i, j]];
                        input_grads[[i, j]] = p * (g - sum_grad_prob);
                    }
                }
            }

            _ => {
                // Unsupported axis for 2D: behave like axis=1.
                let s = Softmax::with_axis(1);
                return s.compute_gradients(output, output_grads);
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

        match self.axis {
            // Axis 1: row-wise (default)
            1 => {
                for (i, row) in logits.outer_iter().enumerate() {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut any_finite = false;
                    let mut argmax = 0usize;

                    for (j, &x) in row.iter().enumerate() {
                        if x.is_finite() {
                            any_finite = true;
                            if x > max_val {
                                max_val = x;
                                argmax = j;
                            }
                        }
                    }

                    if !any_finite {
                        // Match historical behavior: if everything is non-finite, fall back to a
                        // deterministic one-hot at index 0.
                        if !row.is_empty() {
                            result[[i, 0]] = 1.0;
                        }
                        continue;
                    }

                    // Small vectors (e.g., routing/gating) are sensitive to rounding.
                    // Use the classic two-pass f64-normalized computation to preserve
                    // historical behavior and reduce threshold-crossing jitter.
                    let use_two_pass = row.len() <= 64;

                    if use_two_pass {
                        let mut exp_sum: f64 = 0.0;
                        let mut exps = [0.0f64; 64];
                        for (j, &x) in row.iter().enumerate() {
                            if x.is_finite() {
                                let e = PadeExp::exp((x - max_val) as f64);
                                exps[j] = e;
                                exp_sum += e;
                            } else {
                                exps[j] = 0.0;
                            }
                        }

                        if exp_sum <= 0.0 || !exp_sum.is_finite() {
                            for j in 0..row.len() {
                                result[[i, j]] = if j == argmax { 1.0 } else { 0.0 };
                            }
                            continue;
                        }

                        let inv_sum = 1.0 / exp_sum;
                        for j in 0..row.len() {
                            result[[i, j]] = (exps[j] * inv_sum) as f32;
                        }
                    } else {
                        // Fast path: one exp() per element.
                        let mut exp_sum: f64 = 0.0;
                        for (j, &x) in row.iter().enumerate() {
                            if x.is_finite() {
                                let e = PadeExp::exp((x - max_val) as f64);
                                exp_sum += e;
                                result[[i, j]] = e as f32;
                            } else {
                                result[[i, j]] = 0.0;
                            }
                        }

                        if exp_sum <= 0.0 || !exp_sum.is_finite() {
                            for j in 0..row.len() {
                                result[[i, j]] = if j == argmax { 1.0 } else { 0.0 };
                            }
                            continue;
                        }

                        let inv_sum = (1.0 / exp_sum) as f32;
                        for j in 0..row.len() {
                            result[[i, j]] *= inv_sum;
                        }
                    }
                }
            }

            // Axis 0: column-wise
            0 => {
                let nrows = logits.nrows();
                let ncols = logits.ncols();
                for j in 0..ncols {
                    let mut max_val = f32::NEG_INFINITY;
                    let mut any_finite = false;
                    let mut argmax = 0usize;

                    for i in 0..nrows {
                        let x = logits[[i, j]];
                        if x.is_finite() {
                            any_finite = true;
                            if x > max_val {
                                max_val = x;
                                argmax = i;
                            }
                        }
                    }

                    if !any_finite {
                        if nrows > 0 {
                            result[[0, j]] = 1.0;
                        }
                        continue;
                    }

                    let use_two_pass = nrows <= 64;

                    if use_two_pass {
                        let mut exp_sum: f64 = 0.0;
                        let mut exps = [0.0f64; 64];
                        for i in 0..nrows {
                            let x = logits[[i, j]];
                            if x.is_finite() {
                                let e = PadeExp::exp((x - max_val) as f64);
                                exps[i] = e;
                                exp_sum += e;
                            } else {
                                exps[i] = 0.0;
                            }
                        }

                        if exp_sum <= 0.0 || !exp_sum.is_finite() {
                            for i in 0..nrows {
                                result[[i, j]] = if i == argmax { 1.0 } else { 0.0 };
                            }
                            continue;
                        }

                        let inv_sum = 1.0 / exp_sum;
                        for i in 0..nrows {
                            result[[i, j]] = (exps[i] * inv_sum) as f32;
                        }
                    } else {
                        let mut exp_sum: f64 = 0.0;
                        for i in 0..nrows {
                            let x = logits[[i, j]];
                            if x.is_finite() {
                                let e = PadeExp::exp((x - max_val) as f64);
                                exp_sum += e;
                                result[[i, j]] = e as f32;
                            } else {
                                result[[i, j]] = 0.0;
                            }
                        }

                        if exp_sum <= 0.0 || !exp_sum.is_finite() {
                            for i in 0..nrows {
                                result[[i, j]] = if i == argmax { 1.0 } else { 0.0 };
                            }
                            continue;
                        }

                        let inv_sum = (1.0 / exp_sum) as f32;
                        for i in 0..nrows {
                            result[[i, j]] *= inv_sum;
                        }
                    }
                }
            }

            _ => {
                // For 2D tensors we only support axis 0 or 1.
                // Default to row-wise behavior for safety.
                let s = Softmax::with_axis(1);
                return s.softmax(logits);
            }
        }

        result
    }

    fn softmax_row(&self, row: &ArrayView1<f32>) -> Array1<f32> {
        let mut result = Array1::zeros(row.raw_dim());

        // Find max value for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        let mut any_finite = false;
        let mut argmax = 0usize;
        for (j, &x) in row.iter().enumerate() {
            if x.is_finite() {
                any_finite = true;
                if x > max_val {
                    max_val = x;
                    argmax = j;
                }
            }
        }
        if !any_finite {
            if !row.is_empty() {
                result[0] = 1.0;
            }
            return result;
        }

        // Compute exp(x - max) once into output, accumulate in f64, then normalize.
        let mut exp_sum: f64 = 0.0;
        for (j, &x) in row.iter().enumerate() {
            if x.is_finite() {
                let e = PadeExp::exp((x - max_val) as f64);
                exp_sum += e;
                result[j] = e as f32;
            } else {
                result[j] = 0.0;
            }
        }

        if exp_sum <= 0.0 || !exp_sum.is_finite() {
            // Degenerate case (extremely wide logits). Fall back to argmax = 1.0.
            for j in 0..row.len() {
                result[j] = if j == argmax { 1.0 } else { 0.0 };
            }
            return result;
        }

        let inv_sum = (1.0 / exp_sum) as f32;
        for j in 0..row.len() {
            result[j] *= inv_sum;
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
    use ndarray::{Array1, Array2, Axis};

    use super::*;

    fn assert_allclose(a: &Array1<f32>, b: &Array1<f32>, tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            assert!(diff <= tol, "mismatch at {i}: {x} vs {y} (diff={diff})");
        }
    }

    #[test]
    fn test_softmax_row_matches_2d_for_finite_logits() {
        let s = Softmax::new();
        let row = Array1::from_vec(vec![1.0, 2.0, 3.0, -4.0]);
        let two_d = Array2::from_shape_vec((1, row.len()), row.to_vec()).unwrap();

        let out_row = s.forward_immutable_row(&row.view());
        let out_2d = s.forward_immutable(&two_d.view());
        assert_allclose(&out_row, &out_2d.index_axis(Axis(0), 0).to_owned(), 1e-6);
    }

    #[test]
    fn test_softmax_row_matches_2d_with_non_finite_values() {
        let s = Softmax::new();
        let row = Array1::from_vec(vec![f32::NAN, 0.5, f32::INFINITY, -1.0]);
        let two_d = Array2::from_shape_vec((1, row.len()), row.to_vec()).unwrap();

        let out_row = s.forward_immutable_row(&row.view());
        let out_2d = s.forward_immutable(&two_d.view());
        assert_allclose(&out_row, &out_2d.index_axis(Axis(0), 0).to_owned(), 1e-6);
    }

    #[test]
    fn test_softmax_row_degenerate_all_non_finite_falls_back_to_one_hot() {
        let s = Softmax::new();
        let row = Array1::from_vec(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);

        let out_row = s.forward_immutable_row(&row.view());
        assert_eq!(out_row.len(), 3);
        assert!(out_row.iter().all(|x| x.is_finite()));
        let ones = out_row.iter().filter(|&&x| x == 1.0).count();
        assert_eq!(ones, 1);
    }

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

    #[test]
    fn test_softmax_axis0_columnwise_sums_to_one() {
        let s = Softmax::with_axis(0);
        let input = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0]).unwrap();
        let out = s.forward_immutable(&input.view());

        // Column 0 should sum to 1, column 1 should sum to 1.
        let col0_sum: f32 = out.column(0).iter().sum();
        let col1_sum: f32 = out.column(1).iter().sum();
        assert!((col0_sum - 1.0).abs() < 1e-6);
        assert!((col1_sum - 1.0).abs() < 1e-6);
    }
}
