use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{adam::Adam, errors::Result, llm::Layer, richards::{RichardsActivation, RichardsCurve, Variant}};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SwiGLU {
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
    // [MOD] Learnable RichardsActivation for swish activation
    pub swish_activation: RichardsActivation,
    // [MOD] Learnable RichardsCurve for gate (sigmoid)
    pub gate_curve: RichardsCurve,
}

impl SwiGLU {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Xavier/Glorot initialization via Normal(0, sqrt(2/fan_in))
        let mut rng = rand::rng();
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
            swish_activation: RichardsActivation::new_learnable(Variant::None),
            gate_curve: RichardsCurve::new_learnable(Variant::Sigmoid),
        }
    }

}

impl Layer for SwiGLU {
    fn layer_type(&self) -> &str {
        "SwiGLU"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);
        
        // Convert to f64 for RichardsActivation and process row by row
        let mut swish = Array2::<f32>::zeros(x1.raw_dim());
        let mut gate_sigma = Array2::<f32>::zeros(x2.raw_dim());
        
        for (i, (x1_row, x2_row)) in x1.outer_iter().zip(x2.outer_iter()).enumerate() {
            // Convert to f64 Array1 for RichardsActivation
            let x1_f64: Array1<f64> = x1_row.mapv(|x| x as f64);
            let x2_f64: Array1<f64> = x2_row.mapv(|x| x as f64);
            
            // Apply RichardsActivation for swish (x * Richards(x))
            let swish_f64 = self.swish_activation.forward(&x1_f64);
            // Apply RichardsCurve for gate (just sigmoid)
            let gate_f64 = self.gate_curve.forward(&x2_f64);
            
            // Convert back to f32 and store
            for (j, (&s, &g)) in swish_f64.iter().zip(gate_f64.iter()).enumerate() {
                swish[[i, j]] = s as f32;
                gate_sigma[[i, j]] = g as f32;
            }
        }
        
        let gated = &swish * &gate_sigma;
        let output = gated.dot(&self.w_out) + input;

        // Cache values for backward pass
        self.cached_input = Some(input.clone());
        self.cached_x1 = Some(x1);
        self.cached_x2 = Some(x2);
        self.cached_swish = Some(swish);
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
        base + self.swish_activation.weights().len() + self.gate_curve.weights().len()
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
        let swish = self
            .cached_swish
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                // Recompute swish if not cached
                let mut result = Array2::<f32>::zeros(x1.raw_dim());
                for (i, x1_row) in x1.outer_iter().enumerate() {
                    let x1_f64: Array1<f64> = x1_row.mapv(|x| x as f64);
                    let swish_f64 = self.swish_activation.forward(&x1_f64);
                    for (j, &s) in swish_f64.iter().enumerate() {
                        result[[i, j]] = s as f32;
                    }
                }
                result
            });
        let gated = self
            .cached_gated
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                // Recompute gate if not cached
                let mut gate_sigma = Array2::<f32>::zeros(x2.raw_dim());
                for (i, x2_row) in x2.outer_iter().enumerate() {
                    let x2_f64: Array1<f64> = x2_row.mapv(|x| x as f64);
                    let gate_f64 = self.gate_curve.forward(&x2_f64);
                    for (j, &g) in gate_f64.iter().enumerate() {
                        gate_sigma[[i, j]] = g as f32;
                    }
                }
                &swish * &gate_sigma
            });

        // Gradients wrt parameters
        let grad_w_out = gated.t().dot(output_grads);
        let grad_gated = output_grads.dot(&self.w_out.t());

        // Compute gate_sigma for gradient computation
        let mut gate_sigma = Array2::<f32>::zeros(x2.raw_dim());
        for (i, x2_row) in x2.outer_iter().enumerate() {
            let x2_f64: Array1<f64> = x2_row.mapv(|x| x as f64);
            let gate_f64 = self.gate_curve.forward(&x2_f64);
            for (j, &g) in gate_f64.iter().enumerate() {
                gate_sigma[[i, j]] = g as f32;
            }
        }
        
        let grad_swish = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &swish;

        // Compute gradients through RichardsActivation (row by row)
        let mut grad_x1 = Array2::<f32>::zeros(x1.raw_dim());
        let mut grad_x2 = Array2::<f32>::zeros(x2.raw_dim());
        
        for (i, (x1_row, x2_row)) in x1.outer_iter().zip(x2.outer_iter()).enumerate() {
            let x1_f64: Array1<f64> = x1_row.mapv(|x| x as f64);
            let x2_f64: Array1<f64> = x2_row.mapv(|x| x as f64);
            
            let swish_deriv = self.swish_activation.derivative(&x1_f64);
            let gate_deriv = self.gate_curve.derivative(&x2_f64);
            
            for j in 0..x1_row.len() {
                grad_x1[[i, j]] = (swish_deriv[j] * grad_swish[[i, j]] as f64) as f32;
            }
            for j in 0..x2_row.len() {
                grad_x2[[i, j]] = (gate_deriv[j] * grad_gate_sigma[[i, j]] as f64) as f32;
            }
        }

        // Use input directly for weight gradients
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("forward must cache input before compute_gradients");
        let grad_w1 = cached_input.t().dot(&grad_x1);
        let grad_w2 = cached_input.t().dot(&grad_x2);

        // Input gradient (include residual branch)
        let grad_input_swiglu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());
        let grad_input = grad_input_swiglu + output_grads;

        // Parameter gradients vector
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Compute RichardsActivation/RichardsCurve gradients (scalar version for now)
        let mut swish_grads_sum = Array2::<f32>::zeros((1, self.swish_activation.weights().len()));
        let mut gate_grads_sum = Array2::<f32>::zeros((1, self.gate_curve.weights().len()));
        
        for (i, (x1_row, x2_row)) in x1.outer_iter().zip(x2.outer_iter()).enumerate() {
            for (j, (&x1_val, &x2_val)) in x1_row.iter().zip(x2_row.iter()).enumerate() {
                let swish_grads = self.swish_activation.grad_weights_scalar(x1_val as f64, grad_swish[[i, j]] as f64);
                let gate_grads = self.gate_curve.grad_weights_scalar(x2_val as f64, grad_gate_sigma[[i, j]] as f64);
                
                for (k, &grad) in swish_grads.iter().enumerate() {
                    swish_grads_sum[[0, k]] += grad as f32;
                }
                for (k, &grad) in gate_grads.iter().enumerate() {
                    gate_grads_sum[[0, k]] += grad as f32;
                }
            }
        }
        
        param_grads.push(swish_grads_sum);
        param_grads.push(gate_grads_sum);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients in order: W1, W2, W_out, swish_activation, gate_curve
        if param_grads.len() != 5 {
            return Err(crate::errors::ModelError::GradientError {
                message: format!(
                    "SwiGLU expects 5 gradient blocks, got {}",
                    param_grads.len()
                ),
            });
        }
        
        // Update w1, w2, w_out
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out.step(&mut self.w_out, &param_grads[2], lr);
        
        // Update RichardsActivation weights
        let grad_swish_vec: Vec<f64> = param_grads[3].iter().map(|&x| x as f64).collect();
        self.swish_activation.step(&grad_swish_vec, lr as f64);
        
        // Update RichardsCurve weights
        let grad_gate_vec: Vec<f64> = param_grads[4].iter().map(|&x| x as f64).collect();
        self.gate_curve.step(&grad_gate_vec, lr as f64);
        
        Ok(())
    }
}
