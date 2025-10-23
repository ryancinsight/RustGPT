use ndarray::Array2;
use serde::{Deserialize, Serialize};

use rand_distr::{Distribution, Normal};
use crate::adam::Adam;
use crate::errors::Result;
use crate::llm::Layer;

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
    pub alpha_swish: Array2<f32>,
    pub optimizer_alpha_swish: Adam,
    // Learned sigmoid gate parameters
    pub alpha_gate: Array2<f32>,
    pub beta_gate: Array2<f32>,
    pub optimizer_alpha_gate: Adam,
    pub optimizer_beta_gate: Adam,
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
            w_out: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_w3.sample(&mut rng)),
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w2: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w_out: Adam::new((hidden_dim, embedding_dim)),
            cached_input: None,
            cached_x1: None,
            cached_x2: None,
            cached_swish: None,
            cached_gated: None,
            alpha_swish: Array2::ones((1, hidden_dim)),
            optimizer_alpha_swish: Adam::new((1, hidden_dim)),
            // Initialize learned gate parameters: alpha_gate (steepness) and beta_gate (transition)
            alpha_gate: Array2::ones((1, hidden_dim)),
            beta_gate: Array2::zeros((1, hidden_dim)),
            optimizer_alpha_gate: Adam::new((1, hidden_dim)),
            optimizer_beta_gate: Adam::new((1, hidden_dim)),
        }
    }





    // Removed fixed swish path; always use learned swish
    fn swish_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::zeros(x.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(x)
            .and_broadcast(&self.alpha_swish)
            .for_each(|o, &xv, &a| {
                let z = a * xv;
                let zc = z.max(-20.0).min(20.0);
                let s = 1.0 / (1.0 + (-zc).exp());
                *o = xv * s;
            });
        out
    }

    fn swish_derivative_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::zeros(x.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(x)
            .and_broadcast(&self.alpha_swish)
            .for_each(|o, &xv, &a| {
                let z = a * xv;
                let zc = z.max(-20.0).min(20.0);
                let s = 1.0 / (1.0 + (-zc).exp());
                *o = s + xv * a * s * (1.0 - s);
            });
        out
    }
}

impl Layer for SwiGLU {
    fn layer_type(&self) -> &str { "SwiGLU" }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);
        let swish = self.swish_learned(&x1);

        // Fused dynamic sigmoid gate computation and gating
        let mut gate_sigma = Array2::zeros(x2.raw_dim());
        let mut gated = Array2::zeros(x2.raw_dim());
        ndarray::Zip::from(&mut gate_sigma)
            .and(&mut gated)
            .and(&x2)
            .and_broadcast(&self.alpha_gate)
            .and_broadcast(&self.beta_gate)
            .and(&swish)
            .for_each(|gs, gd, &x2v, &ag, &bg, &sv| {
                let z = ag * x2v + bg;
                let zc = z.max(-20.0).min(20.0);
                let s = 1.0 / (1.0 + (-zc).exp());
                *gs = s;
                *gd = sv * s;
            });

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
        let input = self.cached_input.as_ref().expect("forward must be called before backward");
        let (grad_input, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        grad_input
    }

    fn parameters(&self) -> usize {
        let base = self.w1.len() + self.w2.len() + self.w_out.len();
        base + self.alpha_swish.len() + self.alpha_gate.len() + self.beta_gate.len()
    }

    fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let x1 = self.cached_x1.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w1));
        let x2 = self.cached_x2.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w2));
        let swish = self.cached_swish.as_ref().cloned().unwrap_or_else(|| self.swish_learned(&x1));
        let gated = self.cached_gated.as_ref().cloned().unwrap_or_else(|| &swish * &x2);

        // Gradients wrt parameters
        let grad_w_out = gated.t().dot(output_grads);
        let grad_gated = output_grads.dot(&self.w_out.t());

        // Fused gate_sigma and dsig_dz computation
        let (gate_sigma, dsig_dz) = {
            let mut gs = Array2::zeros(x2.raw_dim());
            let mut ds = Array2::zeros(x2.raw_dim());
            ndarray::Zip::from(&mut gs)
                .and(&mut ds)
                .and(&x2)
                .and_broadcast(&self.alpha_gate)
                .and_broadcast(&self.beta_gate)
                .for_each(|gs, ds, &x2v, &ag, &bg| {
                    let z = ag * x2v + bg;
                    let zc = z.max(-20.0).min(20.0);
                    let s = 1.0 / (1.0 + (-zc).exp());
                    *gs = s;
                    *ds = s * (1.0 - s);
                });
            (gs, ds)
        };

        let grad_swish = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &swish;
        let grad_z = &grad_gate_sigma * &dsig_dz;

        // grad_x2 = grad_z ⊙ alpha_gate (broadcast)
        let mut grad_x2 = grad_z.clone();
        ndarray::Zip::from(&mut grad_x2)
            .and_broadcast(&self.alpha_gate)
            .for_each(|elem, &a| { *elem *= a; });
        let grad_x1 = self.swish_derivative_learned(&x1) * &grad_swish;

        // Use input directly for weight gradients
        let cached_input = self.cached_input.as_ref().expect("forward must cache input before compute_gradients");
        let grad_w1 = cached_input.t().dot(&grad_x1);
        let grad_w2 = cached_input.t().dot(&grad_x2);

        // Input gradient (include residual branch)
        let grad_input_swiglu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());
        let grad_input = grad_input_swiglu + output_grads;

        // Parameter gradients vector
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Efficient alpha_swish gradient accumulation without intermediate arrays
        let mut grad_alpha_swish = Array2::zeros((1, x1.ncols()));
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                let xv = x1[[i, j]];
                let a = self.alpha_swish[[0, j]];
                let z = a * xv;
                let zc = z.max(-20.0).min(20.0);
                let s = 1.0 / (1.0 + (-zc).exp());
                let d = xv * xv * s * (1.0 - s);
                grad_alpha_swish[[0, j]] += grad_swish[[i, j]] * d;
            }
        }
        param_grads.push(grad_alpha_swish);

        // Efficient gate parameter gradients accumulation
        let mut grad_alpha_gate = Array2::zeros((1, x2.ncols()));
        let mut grad_beta_gate = Array2::zeros((1, x2.ncols()));
        for i in 0..x2.nrows() {
            for j in 0..x2.ncols() {
                grad_alpha_gate[[0, j]] += grad_z[[i, j]] * x2[[i, j]];
                grad_beta_gate[[0, j]] += grad_z[[i, j]];
            }
        }
        param_grads.push(grad_alpha_gate);
        param_grads.push(grad_beta_gate);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients: W1, W2, W_out, alpha_swish, alpha_gate, beta_gate
        if param_grads.len() != 6 {
            return Err(crate::errors::ModelError::GradientError { message: format!("SwiGLU expected 6 parameter gradients (W1, W2, W_out, alpha_swish, alpha_gate, beta_gate), got {}", param_grads.len()) });
        }

        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out.step(&mut self.w_out, &param_grads[2], lr);
        self.optimizer_alpha_swish.step(&mut self.alpha_swish, &param_grads[3], lr);
        self.optimizer_alpha_gate.step(&mut self.alpha_gate, &param_grads[4], lr);
        self.optimizer_beta_gate.step(&mut self.beta_gate, &param_grads[5], lr);
        Ok(())
    }
}
