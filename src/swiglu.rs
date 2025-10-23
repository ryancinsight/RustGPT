use ndarray::Array2;
use serde::{Deserialize, Serialize};

use rand_distr::{Distribution, Normal};
use crate::adam::Adam;
use crate::errors::Result;
use crate::llm::Layer;
use crate::sigmoid_poly::SigmoidPoly; // [MOD] Import learnable polynomial activation

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
    // [MOD] Learnable polynomial approximations replacing sigmoid
    pub swish_poly: SigmoidPoly,
    pub gate_poly: SigmoidPoly,
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
            // [MOD] Initialize learnable polynomial activations (cubic defaults)
            swish_poly: SigmoidPoly::new_cubic_default(),
            gate_poly: SigmoidPoly::new_cubic_default(),
        }
    }

    // Removed fixed swish path; always use learned swish
    fn swish_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::zeros(x.raw_dim());
        // [MOD] Update polynomial scaling from batch and use phi instead of sigmoid
        let max_abs = x.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        let mut poly = self.swish_poly.clone();
        poly.update_scaling_from_max_abs(max_abs);
        ndarray::Zip::from(&mut out)
            .and(x)
            .and_broadcast(&self.alpha_swish)
            .for_each(|o, &xv, &a| {
                let z = a * xv;
                let s = poly.forward_scalar(z as f64) as f32;
                *o = xv * s;
            });
        out
    }

    fn swish_derivative_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::zeros(x.raw_dim());
        // [MOD] Derivative using polynomial: d/dx [x * phi(a x)] = phi(a x) + x * a * phi'(a x)
        let max_abs = x.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        let mut poly = self.swish_poly.clone();
        poly.update_scaling_from_max_abs(max_abs);
        ndarray::Zip::from(&mut out)
            .and(x)
            .and_broadcast(&self.alpha_swish)
            .for_each(|o, &xv, &a| {
                let z = a * xv;
                let s = poly.forward_scalar(z as f64) as f32;
                let dphi_dz = poly.backward_scalar(z as f64) as f32;
                *o = s + xv * a * dphi_dz;
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
        // [MOD] Polynomial gate: g = phi(alpha * x2 + beta)
        let max_abs_gate = x2.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        let mut gate_poly = self.gate_poly.clone();
        gate_poly.update_scaling_from_max_abs(max_abs_gate);
        ndarray::Zip::from(&mut gate_sigma)
            .and(&mut gated)
            .and(&x2)
            .and_broadcast(&self.alpha_gate)
            .and_broadcast(&self.beta_gate)
            .and(&swish)
            .for_each(|gs, gd, &x2v, &ag, &bg, &sv| {
                let z = ag * x2v + bg;
                let s = gate_poly.forward_scalar(z as f64) as f32;
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
        base + self.alpha_swish.len() + self.alpha_gate.len() + self.beta_gate.len() + self.swish_poly.weights.len() + self.gate_poly.weights.len()
    }

    fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let x1 = self.cached_x1.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w1));
        let x2 = self.cached_x2.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w2));
        let swish = self.cached_swish.as_ref().cloned().unwrap_or_else(|| self.swish_learned(&x1));
        let gated = self.cached_gated.as_ref().cloned().unwrap_or_else(|| &swish * &x2);

        // Gradients wrt parameters
        let grad_w_out = gated.t().dot(output_grads);
        let grad_gated = output_grads.dot(&self.w_out.t());

        // [MOD] Compute gate_sigma and dphi/dz using polynomial gate
        let (gate_sigma, dphi_dz) = {
            let mut gs = Array2::zeros(x2.raw_dim());
            let mut ds = Array2::zeros(x2.raw_dim());
            let max_abs_gate = x2.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
            let mut gate_poly = self.gate_poly.clone();
            gate_poly.update_scaling_from_max_abs(max_abs_gate);
            ndarray::Zip::from(&mut gs)
                .and(&mut ds)
                .and(&x2)
                .and_broadcast(&self.alpha_gate)
                .and_broadcast(&self.beta_gate)
                .for_each(|gs, ds, &x2v, &ag, &bg| {
                    let z = ag * x2v + bg;
                    let s = gate_poly.forward_scalar(z as f64) as f32;
                    let d = gate_poly.backward_scalar(z as f64) as f32;
                    *gs = s;
                    *ds = d;
                });
            (gs, ds)
        };

        let grad_swish = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &swish;
        // [MOD] grad_z uses polynomial derivative
        let grad_z = &grad_gate_sigma * &dphi_dz;

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

        // [MOD] Efficient alpha_swish gradient accumulation using polynomial derivative
        let mut grad_alpha_swish = Array2::zeros((1, x1.ncols()));
        let max_abs_swish = x1.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        let mut poly = self.swish_poly.clone();
        poly.update_scaling_from_max_abs(max_abs_swish);
        for i in 0..x1.nrows() {
            for j in 0..x1.ncols() {
                let xv = x1[[i, j]];
                let a = self.alpha_swish[[0, j]];
                let z = a * xv;
                let dphi_dz = poly.backward_scalar(z as f64) as f32;
                let d = xv * xv * dphi_dz; // d/d alpha of swish(x) = x * phi(a x) => x^2 * phi'(a x)
                grad_alpha_swish[[0, j]] += grad_swish[[i, j]] * d;
            }
        }
        param_grads.push(grad_alpha_swish);

        // [LEARN] swish polynomial coefficient gradients: dL/dw_k = Σ_i,j grad_swish[i,j] * x1[i,j] * (c * (a_j * x1[i,j]))^k
        let n_swish_w = poly.weights.len();
        let mut grad_swish_poly_vec = vec![0.0_f64; n_swish_w];
        for j in 0..x1.ncols() {
            let a = self.alpha_swish[[0, j]] as f64;
            for i in 0..x1.nrows() {
                let xv = x1[[i, j]] as f64;
                let g = grad_swish[[i, j]] as f64;
                let cx = poly.scaling * (a * xv);
                let mut power = 1.0_f64; // (c*z)^k
                for k in 0..n_swish_w {
                    grad_swish_poly_vec[k] += g * xv * power;
                    power *= cx;
                }
            }
        }
        let grad_swish_poly = Array2::<f32>::from_shape_vec((1, n_swish_w), grad_swish_poly_vec.into_iter().map(|v| v as f32).collect()).unwrap();
        param_grads.push(grad_swish_poly);

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

        // [LEARN] gate polynomial coefficient gradients: dL/dw_k = Σ_i,j grad_gate_sigma[i,j] * (c * (a_j * x2[i,j] + b_j))^k
        let max_abs_gate = x2.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        let mut gate_poly = self.gate_poly.clone();
        gate_poly.update_scaling_from_max_abs(max_abs_gate);
        let n_gate_w = gate_poly.weights.len();
        let mut grad_gate_poly_vec = vec![0.0_f64; n_gate_w];
        for j in 0..x2.ncols() {
            let a = self.alpha_gate[[0, j]] as f64;
            let b = self.beta_gate[[0, j]] as f64;
            for i in 0..x2.nrows() {
                let xv = x2[[i, j]] as f64;
                let g = grad_gate_sigma[[i, j]] as f64; // dL/dg
                let z = a * xv + b;
                let cx = gate_poly.scaling * z;
                let mut power = 1.0_f64;
                for k in 0..n_gate_w {
                    grad_gate_poly_vec[k] += g * power;
                    power *= cx;
                }
            }
        }
        let grad_gate_poly = Array2::<f32>::from_shape_vec((1, n_gate_w), grad_gate_poly_vec.into_iter().map(|v| v as f32).collect()).unwrap();
        param_grads.push(grad_gate_poly);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients in order: W1, W2, W_out, alpha_swish, swish_poly_w, alpha_gate, beta_gate, gate_poly_w
        if param_grads.len() != 8 {
            return Err(crate::errors::ModelError::GradientError{ message: format!("SwiGLU expects 8 gradient blocks, got {}", param_grads.len()) });
        }
        // update w1, w2, w_out
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out.step(&mut self.w_out, &param_grads[2], lr);
        // update alpha_swish
        self.optimizer_alpha_swish.step(&mut self.alpha_swish, &param_grads[3], lr);
        // swish polynomial weights via SGD
        let grad_swish_poly = &param_grads[4];
        for i in 0..self.swish_poly.weights.len() {
            self.swish_poly.weights[i] -= (lr as f64) * (grad_swish_poly[[0, i]] as f64);
        }
        // update gate alphas and betas
        self.optimizer_alpha_gate.step(&mut self.alpha_gate, &param_grads[5], lr);
        self.optimizer_beta_gate.step(&mut self.beta_gate, &param_grads[6], lr);
        // gate polynomial weights via SGD
        let grad_gate_poly = &param_grads[7];
        for i in 0..self.gate_poly.weights.len() {
            self.gate_poly.weights[i] -= (lr as f64) * (grad_gate_poly[[0, i]] as f64);
        }
        Ok(())
    }
}
