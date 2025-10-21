use ndarray::{Array2, Axis};
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
    pub use_dynamic_swish: bool,
    pub alpha_swish: Array2<f32>,
    pub optimizer_alpha_swish: Adam,
}

impl SwiGLU {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Xavier/Glorot initialization via Normal(0, sqrt(2/fan_in))
        let mut rng = rand::thread_rng();
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
            use_dynamic_swish: false,
            alpha_swish: Array2::ones((1, hidden_dim)),
            optimizer_alpha_swish: Adam::new((1, hidden_dim)),
        }
    }

    pub fn enable_dynamic_swish(&mut self) { self.use_dynamic_swish = true; }

    fn swish(x: &Array2<f32>) -> Array2<f32> {
        let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        x * &sigmoid
    }

    fn swish_derivative(x: &Array2<f32>) -> Array2<f32> {
        let sigmoid = x.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let one_minus_sigmoid = sigmoid.mapv(|v| 1.0 - v);
        &sigmoid + (x * &sigmoid * &one_minus_sigmoid)
    }

    fn swish_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut ax = x.clone();
        ndarray::Zip::from(&mut ax)
            .and_broadcast(&self.alpha_swish)
            .for_each(|elem, &a| { *elem *= a; });
        let sigmoid = ax.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        x * &sigmoid
    }

    fn swish_derivative_learned(&self, x: &Array2<f32>) -> Array2<f32> {
        let ax = {
            let mut tmp = x.clone();
            ndarray::Zip::from(&mut tmp)
                .and_broadcast(&self.alpha_swish)
                .for_each(|elem, &a| { *elem *= a; });
            tmp
        };
        let sigmoid = ax.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        let one_minus = sigmoid.mapv(|v| 1.0 - v);
        &sigmoid + &(ax * &sigmoid * &one_minus)
    }
}

impl Layer for SwiGLU {
    fn layer_type(&self) -> &str { "SwiGLU" }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);
        let swish = if self.use_dynamic_swish { self.swish_learned(&x1) } else { Self::swish(&x1) };
        let gated = &swish * &x2;
        let output = gated.dot(&self.w_out) + input;
        // cache
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
        if self.use_dynamic_swish { base + self.alpha_swish.len() } else { base }
    }

    fn compute_gradients(&self, input: &Array2<f32>, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        let x1 = self.cached_x1.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w1));
        let x2 = self.cached_x2.as_ref().cloned().unwrap_or_else(|| input.dot(&self.w2));
        let swish = self.cached_swish.as_ref().cloned().unwrap_or_else(|| if self.use_dynamic_swish { self.swish_learned(&x1) } else { Self::swish(&x1) });
        let gated = self.cached_gated.as_ref().cloned().unwrap_or_else(|| &swish * &x2);

        // Gradients wrt parameters
        let grad_w_out = gated.t().dot(output_grads);
        let grad_gated = output_grads.dot(&self.w_out.t());
        let grad_swish = &grad_gated * &x2;
        let grad_x2 = &grad_gated * &swish;
        let grad_x1 = if self.use_dynamic_swish { self.swish_derivative_learned(&x1) * &grad_swish } else { Self::swish_derivative(&x1) * &grad_swish };
        let input_cached = self.cached_input.as_ref().expect("forward must be called before compute_gradients");
        let grad_w1 = input_cached.t().dot(&grad_x1);
        let grad_w2 = input_cached.t().dot(&grad_x2);

        // Input gradient (include residual branch)
        let grad_input_swiglu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());
        let grad_input = grad_input_swiglu + output_grads;

        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];
        if self.use_dynamic_swish {
            let ax = {
                let mut tmp = x1.clone();
                ndarray::Zip::from(&mut tmp)
                    .and_broadcast(&self.alpha_swish)
                    .for_each(|elem, &a| { *elem *= a; });
                tmp
            };
            let sigmoid = ax.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            let one_minus = sigmoid.mapv(|v| 1.0 - v);
            let x_sq = x1.mapv(|v| v * v);
            let dydalpha = x_sq * &sigmoid * &one_minus;
            let grad_alpha_mat = grad_swish * &dydalpha;
            let grad_alpha = grad_alpha_mat.sum_axis(Axis(0)).insert_axis(Axis(0));
            param_grads.push(grad_alpha);
        }

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        if self.use_dynamic_swish {
            if param_grads.len() != 4 {
                return Err(crate::errors::ModelError::GradientError { message: format!("SwiGLU expected 4 parameter gradients (W1, W2, W_out, alpha), got {}", param_grads.len()) });
            }
        } else if param_grads.len() != 3 {
            return Err(crate::errors::ModelError::GradientError { message: format!("SwiGLU expected 3 parameter gradients (W1, W2, W_out), got {}", param_grads.len()) });
        }

        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out.step(&mut self.w_out, &param_grads[2], lr);
        if self.use_dynamic_swish { self.optimizer_alpha_swish.step(&mut self.alpha_swish, &param_grads[3], lr); }
        Ok(())
    }
}
