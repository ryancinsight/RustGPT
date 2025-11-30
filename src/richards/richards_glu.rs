use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, errors::Result, network::Layer, richards::{RichardsActivation, RichardsGate, Variant}, rng::get_rng};

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
    // [MOD] Learned RichardsGate for gating (ensures [0,1] output range)
    pub gate: RichardsGate,
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
        }
    }

}

impl Layer for RichardsGlu {
    fn layer_type(&self) -> &str {
        "RichardsGlu"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);

        // Vectorized processing: convert entire matrices to f64, apply Richards functions, convert back
        let x1_f64 = x1.mapv(|x| x as f64);
        let _x2_f64 = x2.mapv(|x| x as f64);

        // Apply Richards functions to entire matrices at once
        let value_f64 = self.richards_activation.forward_matrix(&x1_f64);
        // Convert back to f32
        let value = value_f64.mapv(|x| x as f32);

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
            .unwrap_or_else(|| {
                // Recompute value if not cached
                let mut result = Array2::<f32>::zeros(x1.raw_dim());
                for (i, x1_row) in x1.outer_iter().enumerate() {
                    let x1_f64: Array1<f64> = x1_row.mapv(|x| x as f64);
                    let value_f64 = self.richards_activation.forward(&x1_f64);
                    for (j, &v) in value_f64.iter().enumerate() {
                        result[[i, j]] = v as f32;
                    }
                }
                result
            });
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

        // Compute gate_sigma for gradient computation
        // Compute gate values
        let gate_sigma = self.gate.forward_const(&x2);
        
        let grad_value = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &value;

        // Compute gradients through RichardsActivation (row by row)
        let mut grad_x1 = Array2::<f32>::zeros(x1.raw_dim());
        let mut grad_x2 = Array2::<f32>::zeros(x2.raw_dim());
        
        for (i, (x1_row, x2_row)) in x1.outer_iter().zip(x2.outer_iter()).enumerate() {
            let x1_f64: Array1<f64> = x1_row.mapv(|x| x as f64);
            let x2_f64: Array1<f64> = x2_row.mapv(|x| x as f64);

            let value_deriv = self.richards_activation.derivative(&x1_f64);

            // For RichardsGate, compute derivative through temperature scaling
            // RichardsGate applies temperature scaling: x_temp = x / T
            // Derivative: d/dx[g(x/T)] = g'(x/T) * (1/T)
            let scaled_x = x2_f64.mapv(|x| x / self.gate.temperature as f64);
            let curve_deriv = self.gate.curve.derivative(&scaled_x);
            let gate_deriv = curve_deriv.mapv(|d| d / self.gate.temperature as f64);

            for j in 0..x1_row.len() {
                grad_x1[[i, j]] = (value_deriv[j] * grad_value[[i, j]] as f64) as f32;
            }
            for j in 0..x2_row.len() {
                grad_x2[[i, j]] = (gate_deriv[j] * grad_gate_sigma[[i, j]] as f64) as f32;
            }
        }

        // Use input directly for weight gradients (fallback to cached input if available)
        let weight_input = self.cached_input.as_ref().unwrap_or(input);
        let grad_w1 = weight_input.t().dot(&grad_x1);
        let grad_w2 = weight_input.t().dot(&grad_x2);

        // Input gradient (include residual branch)
        let grad_input_glu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());
        let grad_input = grad_input_glu + output_grads;

        // Parameter gradients vector
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Compute RichardsActivation gradients (value function)
        let mut value_grads_sum = Array2::<f32>::zeros((1, self.richards_activation.weights().len()));

        // Accumulate gradients for richards_activation (value function)
        for (i, x1_row) in x1.outer_iter().enumerate() {
            for (j, &x1_val) in x1_row.iter().enumerate() {
                let grad_output = grad_value[[i, j]] as f64;
                if grad_output != 0.0 {
                    let value_grads = self.richards_activation.richards_curve.grad_weights_scalar(x1_val as f64, grad_output);
                    for (k, &grad) in value_grads.iter().enumerate() {
                        value_grads_sum[[0, k]] += grad as f32;
                    }
                }
            }
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
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "RichardsGlu expects at least 4 gradient blocks, got {}",
                    param_grads.len()
                ),
            });
        }
        
        // Update w1, w2, w_out
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out.step(&mut self.w_out, &param_grads[2], lr);
        
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
