use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Learnable polynomial approximation to sigmoid:
/// phi(x) = sum_{i=0}^n w_i * (c * x)^i
/// - n is implied by the length of `weights`
/// - c (scaling) is typically 1/sqrt(max |x|) for numerical stability
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SigmoidPoly {
    /// Polynomial weights w_i (double precision for numerical robustness)
    pub weights: Vec<f64>,
    /// Scaling factor c applied to input before polynomial evaluation
    pub scaling: f64,
}

impl SigmoidPoly {
    /// Initialize cubic polynomial with coefficients fitted to sigmoid over [-10, 10]
    /// Defaults mimic sigmoid shape without saturation: w0=0.5, w1=0.25, w2=0.0, w3=-1/24
    pub fn new_cubic_default() -> Self {
        Self {
            weights: vec![0.5, 0.25, 0.0, -1.0 / 24.0],
            scaling: 1.0,
        }
    }

    /// Update scaling to c = 1/sqrt(max|x|) with safe fallback
    pub fn update_scaling_from_max_abs(&mut self, max_abs_x: f64) {
        self.scaling = if max_abs_x > 0.0 { 1.0 / max_abs_x.sqrt() } else { 1.0 };
    }

    /// Forward for a single scalar (f64)
    #[inline]
    pub fn forward_scalar(&self, x: f64) -> f64 {
        let mut sum = 0.0;
        let mut power = 1.0; // (c*x)^0 initially
        let cx = self.scaling * x;
        for &w in &self.weights {
            sum += w * power;
            power *= cx;
        }
        sum
    }

    /// Backward derivative dphi/dx for a single scalar (f64)
    /// dphi/dx = sum_{i=1} i * w_i * c * (c*x)^{i-1}
    #[inline]
    pub fn backward_scalar(&self, x: f64) -> f64 {
        let mut grad = 0.0;
        let mut power = 1.0; // (c*x)^(i-1), starts at i=1 -> (c*x)^0
        let cx = self.scaling * x;
        for (i, &w) in self.weights.iter().enumerate().skip(1) {
            grad += (i as f64) * w * self.scaling * power;
            power *= cx;
        }
        grad
    }

    /// Vectorized forward over Array2<f32>. Updates scaling based on batch max |x|
    pub fn forward_array_f32(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let max_abs = x.iter().fold(0.0_f64, |m, &v| m.max((v as f64).abs()));
        self.update_scaling_from_max_abs(max_abs);
        let c = self.scaling as f32;
        let mut out = Array2::<f32>::zeros(x.raw_dim());
        ndarray::Zip::from(&mut out).and(x).for_each(|o, &xi| {
            let mut sum = 0.0_f64;
            let mut power = 1.0_f64;
            let cx = (c as f64) * (xi as f64);
            for &w in &self.weights {
                sum += w * power;
                power *= cx;
            }
            *o = sum as f32;
        });
        out
    }

    /// Vectorized backward dphi/dx over Array2<f32> (does not update scaling)
    pub fn backward_array_f32(&self, x: &Array2<f32>) -> Array2<f32> {
        let c = self.scaling as f32;
        let mut out = Array2::<f32>::zeros(x.raw_dim());
        ndarray::Zip::from(&mut out).and(x).for_each(|o, &xi| {
            let mut grad = 0.0_f64;
            let mut power = 1.0_f64; // (c*x)^(i-1)
            let cx = (c as f64) * (xi as f64);
            for (i, &w) in self.weights.iter().enumerate().skip(1) {
                grad += (i as f64) * w * (self.scaling) * power;
                power *= cx;
            }
            *o = grad as f32;
        });
        out
    }

    /// Gradient w.r.t. weights for a single scalar input: dphi/dw_i = (c*x)^i
    pub fn grad_weights_scalar(&self, x: f64) -> Vec<f64> {
        let mut grads = Vec::with_capacity(self.weights.len());
        let mut power = 1.0; // (c*x)^0 initially
        let cx = self.scaling * x;
        for _ in 0..self.weights.len() {
            grads.push(power);
            power *= cx;
        }
        grads
    }

    /// Mutable access for optimizers
    #[inline]
    pub fn weights_mut(&mut self) -> &mut [f64] { &mut self.weights }
}