use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    common::{errors::Result, rng::get_rng},
    domain::{
        network::Layer,
        richards::{RichardsActivation, RichardsGate, Variant},
    },
    infrastructure::optimizer::adam::Adam,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsGlu {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w_out: Array2<f32>,
    pub optimizer_w1: Adam,
    pub optimizer_w2: Adam,
    pub optimizer_w_out: Adam,
    pub cached_input: Option<Array2<f32>>,
    pub cached_x1: Option<Array2<f32>>,
    pub cached_x2: Option<Array2<f32>>,
    pub cached_swish: Option<Array2<f32>>,
    pub cached_gated: Option<Array2<f32>>,
    // [MOD] Learnable RichardsActivation for value function
    pub richards_activation: RichardsActivation,
    // [MOD] Learned RichardsGate for gating
    pub gate: RichardsGate,
    /// Workspace for streaming inference
    #[serde(skip)]
    pub streaming_workspace: Option<RichardsGluStreamingWorkspace>,
}

impl RichardsGlu {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Xavier/Glorot initialization via Normal(0, sqrt(2/fan_in))
        let mut rng = get_rng();
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();
        let std_w2 = (2.0 / embedding_dim as f32).sqrt();
        let std_w3 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();
        let normal_w3 = Normal::new(0.0, std_w3).unwrap();
        Self {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            w2: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w2.sample(&mut rng)),
            w_out: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| {
                normal_w3.sample(&mut rng)
            }),
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w2: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w_out: Adam::new((hidden_dim, embedding_dim)),
            cached_input: None,
            cached_x1: None,
            cached_x2: None,
            cached_swish: None,
            cached_gated: None,
            richards_activation: RichardsActivation::new_learnable(Variant::None),
            gate: RichardsGate::new(),
            streaming_workspace: None,
        }
    }

    /// Streaming forward step with pre-allocated output buffer (zero-allocation)
    pub fn forward_step_into(
        &mut self, 
        input: &ndarray::ArrayView1<f32>, 
        output: &mut ndarray::Array1<f32>,
    ) {
        // Initialize workspace if needed
        if self.streaming_workspace.is_none() {
             let d_hidden = self.w1.ncols();
             self.streaming_workspace = Some(RichardsGluStreamingWorkspace {
                 x1: ndarray::Array1::zeros(d_hidden),
                 x2: ndarray::Array1::zeros(d_hidden),
                 value: ndarray::Array1::zeros(d_hidden),
                 gate_sigma: ndarray::Array1::zeros(d_hidden),
                 gated: ndarray::Array1::zeros(d_hidden),
             });
        }
        let ws = self.streaming_workspace.as_mut().unwrap();

        // Ensure workspace dimensions
        if ws.x1.len() != self.w1.ncols() {
            let d_hidden = self.w1.ncols();
            ws.x1 = ndarray::Array1::zeros(d_hidden);
            ws.x2 = ndarray::Array1::zeros(d_hidden);
            ws.value = ndarray::Array1::zeros(d_hidden);
            ws.gate_sigma = ndarray::Array1::zeros(d_hidden);
            ws.gated = ndarray::Array1::zeros(d_hidden);
        }

        // x1 = input * W1
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w1.t(), input, 0.0, &mut ws.x1);
        // x2 = input * W2
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w2.t(), input, 0.0, &mut ws.x2);

        // Apply Richards activation
        if let (Some(x1_slice), Some(value_slice)) = (ws.x1.as_slice(), ws.value.as_slice_mut()) {
            self.richards_activation.forward_into_f32(x1_slice, value_slice);
        } else {
             // Fallback
             ws.value.assign(&self.richards_activation.forward_matrix_f32(&ws.x1.view().insert_axis(ndarray::Axis(0)).to_owned()).row(0));
        }

        // Apply Richards gate
        if let (Some(x2_slice), Some(gate_slice)) = (ws.x2.as_slice(), ws.gate_sigma.as_slice_mut()) {
            self.gate.forward_into_f32(x2_slice, gate_slice);
        } else {
             // Fallback
             ws.gate_sigma.assign(&self.gate.forward_const(&ws.x2.view().insert_axis(ndarray::Axis(0)).to_owned()).row(0));
        }

        // Gating: value * gate
        ndarray::Zip::from(&mut ws.gated)
            .and(&ws.value)
            .and(&ws.gate_sigma)
            .for_each(|g, &v, &s| *g = v * s);

        // Output = gated * W_out + input
        // gated: (H,), W_out: (H, D)
        // output = input (residual)
        output.assign(input);
        // output += gated * W_out
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_out.t(), &ws.gated, 1.0, output);
    }

    /// Streaming forward step for token-by-token inference.
    ///
    /// This method processes a single vector input (Array1) and returns a single vector output.
    /// It uses zero-copy views to reuse the optimized matrix implementations of the
    /// underlying components.
    pub fn forward_step(&mut self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
        let mut output = ndarray::Array1::zeros(input.raw_dim());
        self.forward_step_into(&input.view(), &mut output);
        output
    }
}

#[derive(Debug, Clone)]
pub struct RichardsGluStreamingWorkspace {
    pub x1: ndarray::Array1<f32>,
    pub x2: ndarray::Array1<f32>,
    pub value: ndarray::Array1<f32>,
    pub gate_sigma: ndarray::Array1<f32>,
    pub gated: ndarray::Array1<f32>,
}

impl Layer for RichardsGlu {
    fn layer_type(&self) -> &str {
        "RichardsGlu"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);

        // Apply Richards activation directly on f32 without materializing f64 matrices.
        let value = self.richards_activation.forward_matrix_f32(&x1);

        // Compute gate values using RichardsGate
        let gate_sigma = self.gate.forward(&x2);

        let gated = &value * &gate_sigma;
        let output = gated.dot(&self.w_out) + input;

        // Cache values for backward pass
        self.cached_input = Some(input.clone());
        self.cached_x1 = Some(x1);
        self.cached_x2 = Some(x2);
        self.cached_swish = Some(value);
        self.cached_gated = Some(gated);
        output
    }


    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (grad_input, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        grad_input
    }

    fn parameters(&self) -> usize {
        let base = self.w1.len() + self.w2.len() + self.w_out.len();
        base + self.richards_activation.weights().len() + self.gate.parameters()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let x1 = self
            .cached_x1
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w1));
        let x2 = self
            .cached_x2
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w2));
        let value = self
            .cached_swish
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.richards_activation.forward_matrix_f32(&x1));
        // Compute gate values
        let gate_sigma = self.gate.forward_const(&x2);

        let gated = self
            .cached_gated
            .as_ref()
            .cloned()
            .unwrap_or_else(|| &value * &gate_sigma);

        // Gradients wrt parameters
        let grad_w_out = gated.t().dot(output_grads);
        let grad_gated = output_grads.dot(&self.w_out.t());

        let grad_value = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &value;

        // Compute gradients through RichardsActivation / RichardsGate (parallelized)
        let mut grad_x1 = Array2::<f32>::zeros(x1.raw_dim());
        let mut grad_x2 = Array2::<f32>::zeros(x2.raw_dim());
        let gate_temp_reciprocal = 1.0 / self.gate.temperature;

        // Ensure arrays are contiguous for slice-based parallel iteration.
        // In most cases (from dot/arithmetic), they are already contiguous.
        let x1_contig = x1.as_standard_layout();
        let x2_contig = x2.as_standard_layout();
        let grad_val_contig = grad_value.as_standard_layout();
        let grad_gate_contig = grad_gate_sigma.as_standard_layout();

        let hidden_dim = x1.shape()[1];
        debug_assert_eq!(hidden_dim, x2.shape()[1]);

        // Get raw slices for parallel processing
        let x1_slice = x1_contig.as_slice().expect("x1 must be contiguous");
        let x2_slice = x2_contig.as_slice().expect("x2 must be contiguous");
        let gv_slice = grad_val_contig.as_slice().expect("grad_value must be contiguous");
        let gg_slice = grad_gate_contig.as_slice().expect("grad_gate must be contiguous");
        
        let gx1_slice = grad_x1.as_slice_mut().expect("grad_x1 must be contiguous");
        let gx2_slice = grad_x2.as_slice_mut().expect("grad_x2 must be contiguous");

        gx1_slice
            .par_chunks_mut(hidden_dim)
            .zip(gx2_slice.par_chunks_mut(hidden_dim))
            .zip(x1_slice.par_chunks(hidden_dim))
            .zip(x2_slice.par_chunks(hidden_dim))
            .zip(gv_slice.par_chunks(hidden_dim))
            .zip(gg_slice.par_chunks(hidden_dim))
            .for_each(|(((((gx1_row, gx2_row), x1_row), x2_row), gv_row), gg_row)| {
                // Thread-local scratch buffers
                let mut value_deriv_row = vec![0.0; x1_row.len()];
                let mut value_deriv_tmp = vec![0.0; x1_row.len()];
                let mut gate_scaled_row = vec![0.0; x2_row.len()];
                let mut gate_curve_deriv_row = vec![0.0; x2_row.len()];

                // value_deriv_row = d/dx[x * Richards(x)]
                self.richards_activation.derivative_into_f32_with_scratch(
                    x1_row,
                    &mut value_deriv_row,
                    &mut value_deriv_tmp,
                );

                // Gate derivative with temperature scaling:
                // g(x) = curve(x/T) => dg/dx = curve'(x/T) * (1/T)
                for j in 0..x2_row.len() {
                    gate_scaled_row[j] = x2_row[j] * gate_temp_reciprocal;
                }
                self.gate
                    .curve
                    .derivative_into_f32(&gate_scaled_row, &mut gate_curve_deriv_row);

                for j in 0..x1_row.len() {
                    gx1_row[j] = value_deriv_row[j] * gv_row[j];
                }
                for j in 0..x2_row.len() {
                    let gate_deriv = gate_curve_deriv_row[j] * gate_temp_reciprocal;
                    gx2_row[j] = gate_deriv * gg_row[j];
                }
            });

        // Use input directly for weight gradients (fallback to cached input if available)
        let weight_input = self.cached_input.as_ref().unwrap_or(input);
        let grad_w1 = weight_input.t().dot(&grad_x1);
        let grad_w2 = weight_input.t().dot(&grad_x2);

        // Input gradient (include residual branch)
        let grad_input_glu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());
        let grad_input = grad_input_glu + output_grads;

        // Parameter gradients vector
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Compute RichardsActivation gradients (value function) in one shot.
        // value(x) = x * curve(x) => dL/d(curve(x)) = x * dL/d(value).
        let curve_output_grads = &x1 * &grad_value;
        let value_grads = self
            .richards_activation
            .richards_curve
            .grad_weights_matrix_f32(&x1, &curve_output_grads);
        let mut value_grads_sum = Array2::<f32>::zeros((1, value_grads.len()));
        for (k, &g) in value_grads.iter().enumerate() {
            value_grads_sum[[0, k]] = g as f32;
        }

        // Compute RichardsGate gradients using the gate's own gradient computation
        let (_, gate_param_grads) = self.gate.compute_gradients(&x2, &grad_gate_sigma);

        param_grads.push(value_grads_sum);
        param_grads.extend(gate_param_grads);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients in order: W1, W2, W_out, richards_activation, gate_parameters...
        if param_grads.len() < 4 {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "RichardsGlu expects at least 4 gradient blocks, got {}",
                    param_grads.len()
                ),
            });
        }

        // Update w1, w2, w_out
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out
            .step(&mut self.w_out, &param_grads[2], lr);

        // Update RichardsActivation weights
        let grad_value_vec: Vec<f64> = param_grads[3].iter().map(|&x| x as f64).collect();
        self.richards_activation.step(&grad_value_vec, lr as f64);

        // Update RichardsGate parameters (parameters 4 onwards)
        if param_grads.len() > 4 {
            let gate_grads = &param_grads[4..];
            self.gate.apply_gradients(gate_grads, lr)?;
        }

        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        sumsq += self.w1.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w2.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w_out.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self
            .richards_activation
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq += self.gate.weight_norm();
        sumsq.sqrt()
    }

    fn zero_gradients(&mut self) {
        // RichardsGlu doesn't maintain internal gradient state
        // Gradients are computed on-demand
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_richards_glu_forward_backward() {
        let batch_size = 2;
        let embedding_dim = 4;
        let hidden_dim = 8;
        let mut glu = RichardsGlu::new(embedding_dim, hidden_dim);

        let input = Array2::from_shape_vec(
            (batch_size, embedding_dim),
            vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.0, -0.5],
        )
        .unwrap();

        // Forward
        let output = glu.forward(&input);
        assert_eq!(output.dim(), (batch_size, embedding_dim));

        // Backward
        let grad_output = Array2::from_elem(output.dim(), 0.1);
        let grad_input = glu.backward(&grad_output, 0.01);
        assert_eq!(grad_input.dim(), (batch_size, embedding_dim));
    }

    #[test]
    fn test_richards_glu_shapes() {
        let mut glu = RichardsGlu::new(10, 20);
        let input = Array2::zeros((5, 10));
        let output = glu.forward(&input);
        assert_eq!(output.dim(), (5, 10));
        
        let grad_out = Array2::ones((5, 10));
        let grad_in = glu.backward(&grad_out, 0.001);
        assert_eq!(grad_in.dim(), (5, 10));
    }
}
