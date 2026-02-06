//! Projection Layers Component for SSMs
//!
//! Provides reusable projection layers and linear transformations
//! for state space models with optimized memory management.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::{
    infrastructure::optimizer::adam::Adam,
    domain::eprop::{EPropError, context::EpropContext, utils::outer_product_into},
};

/// Projection layer configuration
#[derive(Debug, Clone, Copy)]
pub struct ProjectionConfig {
    /// Use bias terms in projections
    pub use_bias: bool,
    /// Initialize with small weights for stability
    pub small_init: bool,
    /// Weight initialization scale
    pub init_scale: f32,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            use_bias: true,
            small_init: true,
            init_scale: 0.02,
        }
    }
}

/// Linear projection layer
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LinearProjection {
    pub weight: Array2<f32>,
    pub bias: Option<Array2<f32>>,

    #[serde(skip_serializing)]
    opt_weight: Adam,
    #[serde(skip_serializing)]
    opt_bias: Option<Adam>,
}

impl LinearProjection {
    /// Create a new linear projection
    pub fn new(input_dim: usize, output_dim: usize, config: ProjectionConfig) -> Self {
        let scale = if config.small_init {
            config.init_scale
        } else {
            (2.0 / (input_dim as f32)).sqrt()
        };

        let weight = if config.small_init {
            Array2::zeros((input_dim, output_dim))
        } else {
            Array2::from_shape_fn((input_dim, output_dim), |_| {
                rand::random::<f32>() * scale * 2.0 - scale
            })
        };

        let bias = if config.use_bias {
            Some(Array2::zeros((1, output_dim)))
        } else {
            None
        };

        let opt_weight = Adam::new((input_dim, output_dim));
        let opt_bias = if config.use_bias {
            Some(Adam::new((1, output_dim)))
        } else {
            None
        };

        Self {
            weight,
            bias,
            opt_weight,
            opt_bias,
        }
    }

    /// Forward pass: y = x * weight + bias
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let result = x.dot(&self.weight);

        if let Some(bias) = &self.bias {
            result + bias
        } else {
            result
        }
    }

    /// Apply gradients to projection parameters
    pub fn apply_gradients(
        &mut self,
        input_grad: &Array2<f32>,
        output_grad: &Array2<f32>,
        lr: f32,
    ) {
        // Gradient for weight: dL/dW = x^T * dL/dy
        let weight_grad = input_grad.t().dot(output_grad);
        self.opt_weight.step(&mut self.weight, &weight_grad, lr);

        // Gradient for bias: dL/db = sum(dL/dy)
        if let (Some(bias), Some(opt_bias)) = (&mut self.bias, &mut self.opt_bias) {
            let bias_grad = output_grad
                .sum_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(0));
            opt_bias.step(bias, &bias_grad, lr);
        }
    }

    pub fn apply_eprop_gradients(
        &mut self,
        layer_idx: usize,
        learning_signal: &Array1<f32>,
        lr: f32,
    ) -> crate::domain::eprop::Result<()> {
        let (modulated_eps_f, eps_x) =
            EpropContext::compute_layer_gradients(layer_idx, learning_signal)?;

        let input_dim = self.weight.nrows();
        let output_dim = self.weight.ncols();

        if eps_x.len() != input_dim || modulated_eps_f.len() != output_dim {
            return Err(EPropError::ShapeMismatch {
                expected: format!("({}, {})", input_dim, output_dim),
                got: format!("({}, {})", eps_x.len(), modulated_eps_f.len()),
            });
        }

        let mut weight_grad = Array2::zeros(self.weight.raw_dim());
        outer_product_into(&mut weight_grad, &eps_x, &modulated_eps_f);
        self.opt_weight.step(&mut self.weight, &weight_grad, lr);

        if let (Some(bias), Some(opt_bias)) = (&mut self.bias, &mut self.opt_bias) {
            let bias_grad = modulated_eps_f.insert_axis(Axis(0));
            opt_bias.step(bias, &bias_grad, lr);
        }

        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let mut count = self.weight.len();
        if let Some(bias) = &self.bias {
            count += bias.len();
        }
        count
    }

    /// Reset parameters (useful for testing)
    pub fn reset_parameters(&mut self, config: ProjectionConfig) {
        let input_dim = self.weight.nrows();
        let _output_dim = self.weight.ncols();

        let scale = if config.small_init {
            config.init_scale
        } else {
            (2.0 / (input_dim as f32)).sqrt()
        };

        if config.small_init {
            self.weight.fill(0.0);
        } else {
            for val in self.weight.iter_mut() {
                *val = rand::random::<f32>() * scale * 2.0 - scale;
            }
        }

        if let Some(bias) = &mut self.bias {
            bias.fill(0.0);
        }
    }
}

/// Depthwise convolution layer for 1D sequences
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DepthwiseConv1D {
    pub kernel: Array2<f32>,       // [kernel_size, input_dim]
    pub bias: Option<Array2<f32>>, // [1, input_dim]
    pub kernel_size: usize,

    #[serde(skip_serializing)]
    opt_kernel: Adam,
    #[serde(skip_serializing)]
    opt_bias: Option<Adam>,
}

impl DepthwiseConv1D {
    /// Create a new depthwise convolution layer
    pub fn new(input_dim: usize, kernel_size: usize, config: ProjectionConfig) -> Self {
        let kernel = if config.small_init {
            Array2::zeros((kernel_size, input_dim))
        } else {
            let scale = (1.0 / (kernel_size as f32)).sqrt();
            Array2::from_shape_fn((kernel_size, input_dim), |_| {
                rand::random::<f32>() * scale * 2.0 - scale
            })
        };

        let bias = if config.use_bias {
            Some(Array2::zeros((1, input_dim)))
        } else {
            None
        };

        let opt_kernel = Adam::new((kernel_size, input_dim));
        let opt_bias = if config.use_bias {
            Some(Adam::new((1, input_dim)))
        } else {
            None
        };

        Self {
            kernel,
            bias,
            kernel_size,
            opt_kernel,
            opt_bias,
        }
    }

    /// Forward pass with causal convolution
    pub fn forward_causal(&self, x: &Array2<f32>) -> Array2<f32> {
        if self.kernel_size == 0 {
            return x.clone();
        }
        let seq_len = x.nrows();
        let input_dim = x.ncols();
        let mut output = Array2::zeros((seq_len, input_dim));

        for t in 0..seq_len {
            // Extract window [t+1-window_size..t], where window_size=min(t+1, kernel_size)
            // This avoids usize underflow for small t.
            let window_size = (t + 1).min(self.kernel_size);
            let start = (t + 1) - window_size;

            // Apply depthwise convolution
            for d in 0..input_dim {
                let mut sum = 0.0;
                for k in 0..window_size {
                    let input_idx = start + k;
                    let kernel_idx = self.kernel_size - window_size + k;
                    sum += x[[input_idx, d]] * self.kernel[[kernel_idx, d]];
                }

                let bias_val = if let Some(bias) = &self.bias {
                    bias[[0, d]]
                } else {
                    0.0
                };

                output[[t, d]] = sum + bias_val;
            }
        }

        output
    }

    /// Apply gradients to convolution parameters
    pub fn apply_gradients(&mut self, input: &Array2<f32>, output_grad: &Array2<f32>, lr: f32) {
        if self.kernel_size == 0 {
            return;
        }
        let seq_len = input.nrows();
        let input_dim = input.ncols();

        // Gradient for kernel
        let mut kernel_grad = Array2::zeros(self.kernel.raw_dim());

        for t in 0..seq_len {
            for d in 0..input_dim {
                // Compute gradient for each kernel position
                let window_size = (t + 1).min(self.kernel_size);
                let start = (t + 1) - window_size;

                for k in 0..window_size {
                    let input_idx = start + k;
                    let kernel_idx = self.kernel_size - window_size + k;
                    kernel_grad[[kernel_idx, d]] += input[[input_idx, d]] * output_grad[[t, d]];
                }
            }
        }

        self.opt_kernel.step(&mut self.kernel, &kernel_grad, lr);

        // Gradient for bias
        if let (Some(bias), Some(opt_bias)) = (&mut self.bias, &mut self.opt_bias) {
            let bias_grad = output_grad
                .sum_axis(ndarray::Axis(0))
                .insert_axis(ndarray::Axis(0));
            opt_bias.step(bias, &bias_grad, lr);
        }
    }
}
