use ndarray::{Array1, Array2};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{
    adam::Adam,
    errors::{ModelError, Result},
    llm::Layer,
};
use super::{RichardsActivation, RichardsCurve, Variant};

/// RichardsGLU (RiGLU): Dynamic/adaptive GLU using parameterized Richards curves
///
/// Formulation:
/// - Value branch: v = X · W_v
/// - Adaptive swish-like activation: s = v ⊙ φ_v(v), where φ_v is a learnable RichardsActivation
/// - Gate branch: g_z = X · W_g, gate = φ_g(g_z), where φ_g is a learnable RichardsCurve
/// - Elementwise gating: h = s ⊙ gate
/// - Output projection + residual: Y = h · W_out + X
///
/// Notes:
/// - φ_g is initialized fully learnable (Variant::Adaptive) to allow converging to Sigmoid,
///   Gompertz, or Tanh gates; φ_v is similarly fully learnable to subsume Swish and related forms.
/// - Shapes follow the project’s conventions: W_v ∈ ℝ^{D×H}, W_g ∈ ℝ^{D×H}, W_out ∈ ℝ^{H×D}.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsGlu {
    pub w_v: Array2<f32>,
    pub w_g: Array2<f32>,
    pub w_out: Array2<f32>,

    pub opt_w_v: Adam,
    pub opt_w_g: Adam,
    pub opt_w_out: Adam,

    pub cached_input: Option<Array2<f32>>,   // [N×D]
    pub cached_v: Option<Array2<f32>>,       // [N×H]
    pub cached_gz: Option<Array2<f32>>,      // [N×H]
    pub cached_s: Option<Array2<f32>>,       // [N×H]
    pub cached_h: Option<Array2<f32>>,       // [N×H]

    pub value_activation: RichardsActivation, // φ_v
    pub gate_curve: RichardsCurve,            // φ_g
}

impl RichardsGlu {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Xavier/Glorot initialization via Normal(0, sqrt(2/fan_in))
        let mut rng = rand::rng();
        let std_v = (2.0 / embedding_dim as f32).sqrt();
        let std_g = (2.0 / embedding_dim as f32).sqrt();
        let std_o = (2.0 / hidden_dim as f32).sqrt();
        let normal_v = Normal::new(0.0, std_v).unwrap();
        let normal_g = Normal::new(0.0, std_g).unwrap();
        let normal_o = Normal::new(0.0, std_o).unwrap();

        // Fully learnable Richards for dynamic adaptation (no output_gain/bias constraints)
        let mut gate = RichardsCurve::new_learnable(Variant::Adaptive);
        // Temperature often destabilizes gates; disable to match training ergonomics
        gate.temperature_learnable = false;

        let mut act = RichardsActivation::new_fully_learnable();
        // Also disable temperature in value activation curve for stability
        act.richards_curve.temperature_learnable = false;

        Self {
            w_v: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_v.sample(&mut rng)),
            w_g: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_g.sample(&mut rng)),
            w_out: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_o.sample(&mut rng)),
            opt_w_v: Adam::new((embedding_dim, hidden_dim)),
            opt_w_g: Adam::new((embedding_dim, hidden_dim)),
            opt_w_out: Adam::new((hidden_dim, embedding_dim)),
            cached_input: None,
            cached_v: None,
            cached_gz: None,
            cached_s: None,
            cached_h: None,
            value_activation: act,
            gate_curve: gate,
        }
    }
}

impl Layer for RichardsGlu {
    fn layer_type(&self) -> &str {
        // Identify this layer explicitly as RichardsGlu (RiGLU)
        "RichardsGlu"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Projections
        let v = input.dot(&self.w_v); // [N×H]
        let gz = input.dot(&self.w_g); // [N×H]

        // Vectorized Richards pipeline on f64 then back to f32
        let v_f64 = v.mapv(|x| x as f64);
        let gz_f64 = gz.mapv(|x| x as f64);

        let s_f64 = self.value_activation.forward_matrix(&v_f64); // v ⊙ φ_v(v)
        let gate_f64 = self.gate_curve.forward_matrix(&gz_f64);    // φ_g(gz)

        let s = s_f64.mapv(|x| x as f32);
        let gate = gate_f64.mapv(|x| x as f32);

        let h = &s * &gate;                   // [N×H]
        let y = h.dot(&self.w_out) + input;   // Residual

        // Cache for backward
        self.cached_input = Some(input.clone());
        self.cached_v = Some(v);
        self.cached_gz = Some(gz);
        self.cached_s = Some(s);
        self.cached_h = Some(h);
        y
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
        let base = self.w_v.len() + self.w_g.len() + self.w_out.len();
        base + self.value_activation.weights().len() + self.gate_curve.weights().len()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Reuse caches when available
        let v = self
            .cached_v
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w_v));
        let gz = self
            .cached_gz
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w_g));

        let s = self
            .cached_s
            .as_ref()
            .cloned()
            .unwrap_or_else(|| {
                // Compute s = v ⊙ φ_v(v)
                let mut out = Array2::<f32>::zeros(v.raw_dim());
                for (i, v_row) in v.outer_iter().enumerate() {
                    let v_f64: Array1<f64> = v_row.mapv(|x| x as f64);
                    let s_row = self.value_activation.forward(&v_f64);
                    for (j, &sv) in s_row.iter().enumerate() {
                        out[[i, j]] = sv as f32;
                    }
                }
                out
            });

        // h = s ⊙ gate, so need gate values too
        let mut gate = Array2::<f32>::zeros(gz.raw_dim());
        for (i, gz_row) in gz.outer_iter().enumerate() {
            let gz_f64: Array1<f64> = gz_row.mapv(|x| x as f64);
            let g_row = self.gate_curve.forward(&gz_f64);
            for (j, &gv) in g_row.iter().enumerate() {
                gate[[i, j]] = gv as f32;
            }
        }

        // Gradients wrt parameters
        // More direct: grad_w_out = h^T · dY
        let h = self
            .cached_h
            .as_ref()
            .cloned()
            .unwrap_or_else(|| &s * &gate);
        let grad_w_out = h.t().dot(output_grads);

        // Backprop into h
        let grad_h = output_grads.dot(&self.w_out.t());
        let grad_s = &grad_h * &gate;
        let grad_gate = &grad_h * &s;

        // Compute grad wrt v and gz via Richards derivatives
        let mut grad_v = Array2::<f32>::zeros(v.raw_dim());
        let mut grad_gz = Array2::<f32>::zeros(gz.raw_dim());
        for (i, (v_row, gz_row)) in v.outer_iter().zip(gz.outer_iter()).enumerate() {
            let v_f64: Array1<f64> = v_row.mapv(|x| x as f64);
            let gz_f64: Array1<f64> = gz_row.mapv(|x| x as f64);

            let d_val = self.value_activation.derivative(&v_f64); // d/dv [v·φ_v(v)]
            let d_gate = self.gate_curve.derivative(&gz_f64);      // d/dz φ_g(z)

            for j in 0..v_row.len() {
                grad_v[[i, j]] = (d_val[j] * grad_s[[i, j]] as f64) as f32;
            }
            for j in 0..gz_row.len() {
                grad_gz[[i, j]] = (d_gate[j] * grad_gate[[i, j]] as f64) as f32;
            }
        }

        // Parameter grads for W_v, W_g
        let cached_input = self
            .cached_input
            .as_ref()
            .expect("forward must cache input before compute_gradients");
        let grad_w_v = cached_input.t().dot(&grad_v);
        let grad_w_g = cached_input.t().dot(&grad_gz);

        // Input gradient including residual
        let grad_input_glu = grad_v.dot(&self.w_v.t()) + grad_gz.dot(&self.w_g.t());
        let grad_input = grad_input_glu + output_grads;

        // Accumulate Richards parameter grads (flattened to 1×P arrays)
        let mut val_grads_sum = Array2::<f32>::zeros((1, self.value_activation.weights().len()));
        let mut gate_grads_sum = Array2::<f32>::zeros((1, self.gate_curve.weights().len()));

        for (i, v_row) in v.outer_iter().enumerate() {
            for (j, &v_ij) in v_row.iter().enumerate() {
                let go = grad_s[[i, j]] as f64;
                if go != 0.0 {
                    let grads = self.value_activation.grad_weights_scalar(v_ij as f64, go);
                    for (k, &g) in grads.iter().enumerate() {
                        val_grads_sum[[0, k]] += g as f32;
                    }
                }
            }
        }

        for (i, gz_row) in gz.outer_iter().enumerate() {
            for (j, &gz_ij) in gz_row.iter().enumerate() {
                let go = grad_gate[[i, j]] as f64;
                if go != 0.0 {
                    let grads = self.gate_curve.grad_weights_scalar(gz_ij as f64, go);
                    for (k, &g) in grads.iter().enumerate() {
                        gate_grads_sum[[0, k]] += g as f32;
                    }
                }
            }
        }

        let mut param_grads = vec![grad_w_v, grad_w_g, grad_w_out];
        param_grads.push(val_grads_sum);
        param_grads.push(gate_grads_sum);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients in order: W_v, W_g, W_out, value_activation, gate_curve
        if param_grads.len() != 5 {
            return Err(ModelError::GradientError {
                message: format!(
                    "RichardsGlu expects 5 gradient blocks, got {}",
                    param_grads.len()
                ),
            });
        }

        self.opt_w_v.step(&mut self.w_v, &param_grads[0], lr);
        self.opt_w_g.step(&mut self.w_g, &param_grads[1], lr);
        self.opt_w_out.step(&mut self.w_out, &param_grads[2], lr);

        // Update RichardsActivation weights
        let grad_val_vec: Vec<f64> = param_grads[3].iter().map(|&x| x as f64).collect();
        self.value_activation.step(&grad_val_vec, lr as f64);

        // Update RichardsCurve weights
        let grad_gate_vec: Vec<f64> = param_grads[4].iter().map(|&x| x as f64).collect();
        self.gate_curve.step(&grad_gate_vec, lr as f64);

        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        sumsq += self.w_v.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w_g.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w_out.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self
            .value_activation
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq += self
            .gate_curve
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq.sqrt()
    }
}
#[cfg(test)]
mod tests {
    use super::RichardsGlu;
    use ndarray::{Array2};
    use crate::llm::Layer;

    #[test]
    fn riglu_forward_and_backward_are_finite_on_extremes() {
        let mut layer = RichardsGlu::new(8, 16);
        let input: Array2<f32> = Array2::from_shape_vec((4, 8), vec![
            100.0, -100.0, 50.0, -50.0, 0.0, 10.0, -10.0, 0.5,
            -100.0, 100.0, -50.0, 50.0, 0.0, -10.0, 10.0, -0.5,
            80.0, -60.0, 40.0, -20.0, 10.0, -5.0, 2.5, -1.25,
            -80.0, 60.0, -40.0, 20.0, -10.0, 5.0, -2.5, 1.25,
        ]).unwrap();

        let out = layer.forward(&input);
        assert!(out.iter().all(|&v| v.is_finite()));

        let grads = out.mapv(|v| v * 0.1);
        let in_grads = layer.backward(&grads, 1e-3);
        assert!(in_grads.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn riglu_parameter_update_keeps_weights_finite() {
        let mut layer = RichardsGlu::new(4, 8);
        // Small random input
        let input = Array2::from_shape_vec((2, 4), vec![
            0.05, -0.02, 0.01, 0.03,
            -0.04, 0.02, -0.01, 0.00,
        ]).unwrap();
        let out = layer.forward(&input);
        let grads = out.mapv(|v| v);
        let _in_grads = layer.backward(&grads, 1e-2);

        // Check model parameters remain finite post-update
        assert!(layer.w_v.iter().all(|&w| w.is_finite()));
        assert!(layer.w_g.iter().all(|&w| w.is_finite()));
        assert!(layer.w_out.iter().all(|&w| w.is_finite()));
    }
}
