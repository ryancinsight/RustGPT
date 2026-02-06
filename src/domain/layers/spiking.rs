use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::domain::network::Layer;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifLayer {
    dim: usize,
    config: crate::domain::eprop::NeuronConfig,

    #[serde(skip, default)]
    voltage: Array1<f32>,

    #[serde(skip, default)]
    cached_spikes: Option<Array2<f32>>,

    #[serde(skip, default)]
    cached_surrogate: Option<Array2<f32>>,

    #[serde(skip, default)]
    cached_threshold: Option<Array2<f32>>,
}

impl LifLayer {
    pub fn new(dim: usize) -> Self {
        let mut config = crate::domain::eprop::NeuronConfig::lif();
        config.use_adaptive_surrogate = false;
        Self {
            dim,
            config,
            voltage: Array1::zeros(dim),
            cached_spikes: None,
            cached_surrogate: None,
            cached_threshold: None,
        }
    }
}

impl Layer for LifLayer {
    fn layer_type(&self) -> &str {
        "LIFLayer"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            input.ncols(),
            self.dim,
            "LIFLayer input dim mismatch: expected {}, got {}",
            self.dim,
            input.ncols()
        );

        self.voltage.fill(0.0);

        let t = input.nrows();
        let mut spikes_out = Array2::<f32>::zeros((t, self.dim));
        let mut surrogate_out = Array2::<f32>::zeros((t, self.dim));
        let mut threshold_out = Array2::<f32>::zeros((t, self.dim));

        let v_th = self.config.v_threshold;
        let gamma_pd = self.config.gamma_pd;
        let alpha = self.config.alpha;

        for step in 0..t {
            let input_row = input.row(step);

            let threshold = Array1::from_elem(self.dim, v_th);
            threshold_out.row_mut(step).assign(&threshold);

            let u = &self.voltage * alpha + input_row;
            let delta = &u - &threshold;

            let spikes = delta.mapv(|d| if d >= 0.0 { 1.0 } else { 0.0 });
            let surrogate = delta.mapv(|d| {
                let abs_delta = (d.abs() / v_th).min(f32::INFINITY);
                if abs_delta < 1.0 {
                    (1.0 - abs_delta) / (gamma_pd * v_th)
                } else {
                    0.0
                }
            });

            spikes_out.row_mut(step).assign(&spikes);
            surrogate_out.row_mut(step).assign(&surrogate);

            self.voltage = &u - &(&spikes * v_th);
        }

        self.cached_spikes = Some(spikes_out.clone());
        self.cached_surrogate = Some(surrogate_out);
        self.cached_threshold = Some(threshold_out);

        spikes_out
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        let (input_grads, _) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        input_grads
    }

    fn parameters(&self) -> usize {
        0
    }

    fn weight_norm(&self) -> f32 {
        0.0
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let Some(surrogate) = self.cached_surrogate.as_ref() else {
            panic!("LIFLayer gradients requested before forward");
        };
        let Some(threshold) = self.cached_threshold.as_ref() else {
            panic!("LIFLayer gradients requested before forward");
        };

        assert_eq!(output_grads.raw_dim(), surrogate.raw_dim());

        let t = output_grads.nrows();
        let mut grad_input = Array2::<f32>::zeros((t, self.dim));

        let alpha = self.config.alpha;
        let mut g_v_next = Array1::<f32>::zeros(self.dim);

        for step in (0..t).rev() {
            let g_z = output_grads.row(step).to_owned();
            let psi = surrogate.row(step).to_owned();
            let a_t = threshold.row(step).to_owned();

            let one_minus_a_psi = Array1::from_elem(self.dim, 1.0) - &(&a_t * &psi);
            let grad_i = &g_v_next * &one_minus_a_psi + &g_z * &psi;
            grad_input.row_mut(step).assign(&grad_i);

            let g_v = (&g_v_next * &one_minus_a_psi + &g_z * &psi) * alpha;
            g_v_next = g_v;
        }

        (grad_input, Vec::new())
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        _learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        if gradients.is_empty() {
            Ok(())
        } else {
            Err(crate::common::errors::ModelError::GradientError {
                message: "LIFLayer has no parameters, but received gradients".to_string(),
            })
        }
    }

    fn zero_gradients(&mut self) {
        self.cached_spikes = None;
        self.cached_surrogate = None;
        self.cached_threshold = None;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlifLayer {
    dim: usize,
    config: crate::domain::eprop::NeuronConfig,

    #[serde(skip, default)]
    voltage: Array1<f32>,

    #[serde(skip, default)]
    adaptation: Array1<f32>,

    #[serde(skip, default)]
    cached_spikes: Option<Array2<f32>>,

    #[serde(skip, default)]
    cached_surrogate: Option<Array2<f32>>,

    #[serde(skip, default)]
    cached_threshold: Option<Array2<f32>>,
}

impl AlifLayer {
    pub fn new(dim: usize) -> Self {
        let mut config = crate::domain::eprop::NeuronConfig::alif();
        config.use_adaptive_surrogate = false;
        Self {
            dim,
            config,
            voltage: Array1::zeros(dim),
            adaptation: Array1::zeros(dim),
            cached_spikes: None,
            cached_surrogate: None,
            cached_threshold: None,
        }
    }
}

impl Layer for AlifLayer {
    fn layer_type(&self) -> &str {
        "ALIFLayer"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        assert_eq!(
            input.ncols(),
            self.dim,
            "ALIFLayer input dim mismatch: expected {}, got {}",
            self.dim,
            input.ncols()
        );

        self.voltage.fill(0.0);
        self.adaptation.fill(0.0);

        let t = input.nrows();
        let mut spikes_out = Array2::<f32>::zeros((t, self.dim));
        let mut surrogate_out = Array2::<f32>::zeros((t, self.dim));
        let mut threshold_out = Array2::<f32>::zeros((t, self.dim));

        let v_th = self.config.v_threshold;
        let gamma_pd = self.config.gamma_pd;
        let alpha = self.config.alpha;
        let rho = self.config.rho;
        let beta = self.config.beta;

        for step in 0..t {
            let input_row = input.row(step);

            let threshold = Array1::from_elem(self.dim, v_th) + &(&self.adaptation * beta);
            threshold_out.row_mut(step).assign(&threshold);

            let u = &self.voltage * alpha + input_row;
            let delta = &u - &threshold;

            let spikes = delta.mapv(|d| if d >= 0.0 { 1.0 } else { 0.0 });
            let surrogate = delta.mapv(|d| {
                let abs_delta = (d.abs() / v_th).min(f32::INFINITY);
                if abs_delta < 1.0 {
                    (1.0 - abs_delta) / (gamma_pd * v_th)
                } else {
                    0.0
                }
            });

            spikes_out.row_mut(step).assign(&spikes);
            surrogate_out.row_mut(step).assign(&surrogate);

            self.voltage = &u - &(&spikes * &threshold);
            self.adaptation = &self.adaptation * rho + &spikes;
        }

        self.cached_spikes = Some(spikes_out.clone());
        self.cached_surrogate = Some(surrogate_out);
        self.cached_threshold = Some(threshold_out);

        spikes_out
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        let (input_grads, _) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        input_grads
    }

    fn parameters(&self) -> usize {
        0
    }

    fn weight_norm(&self) -> f32 {
        0.0
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let Some(spikes) = self.cached_spikes.as_ref() else {
            panic!("ALIFLayer gradients requested before forward");
        };
        let Some(surrogate) = self.cached_surrogate.as_ref() else {
            panic!("ALIFLayer gradients requested before forward");
        };
        let Some(threshold) = self.cached_threshold.as_ref() else {
            panic!("ALIFLayer gradients requested before forward");
        };

        assert_eq!(output_grads.raw_dim(), surrogate.raw_dim());

        let t = output_grads.nrows();
        let mut grad_input = Array2::<f32>::zeros((t, self.dim));

        let alpha = self.config.alpha;
        let rho = self.config.rho;
        let beta = self.config.beta;

        let mut g_v_next = Array1::<f32>::zeros(self.dim);
        let mut g_a_next = Array1::<f32>::zeros(self.dim);

        for step in (0..t).rev() {
            let g_z = output_grads.row(step).to_owned();
            let z = spikes.row(step).to_owned();
            let psi = surrogate.row(step).to_owned();
            let a_t = threshold.row(step).to_owned();

            let one_minus_a_psi = Array1::from_elem(self.dim, 1.0) - &(&a_t * &psi);
            let grad_i = &g_v_next * &one_minus_a_psi + &g_a_next * &psi + &g_z * &psi;
            grad_input.row_mut(step).assign(&grad_i);

            let g_v = (&g_v_next * &one_minus_a_psi + (&g_a_next + &g_z) * &psi) * alpha;

            let psi_a = &psi * &a_t;
            let mut psi_a_minus_z = &psi_a - &z;
            psi_a_minus_z.mapv_inplace(|x| x * beta);
            let gv_beta = &g_v_next * &psi_a_minus_z;

            let ga_coeff = Array1::from_elem(self.dim, rho) - &psi.mapv(|p| beta * p);
            let ga_term = &g_a_next * &ga_coeff;

            let gz_term = &g_z * &psi.mapv(|p| -beta * p);

            let g_a = gv_beta + ga_term + gz_term;

            g_v_next = g_v;
            g_a_next = g_a;
        }

        (grad_input, Vec::new())
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        _learning_rate: f32,
    ) -> crate::common::errors::Result<()> {
        if gradients.is_empty() {
            Ok(())
        } else {
            Err(crate::common::errors::ModelError::GradientError {
                message: "ALIFLayer has no parameters, but received gradients".to_string(),
            })
        }
    }

    fn zero_gradients(&mut self) {
        self.cached_spikes = None;
        self.cached_surrogate = None;
        self.cached_threshold = None;
    }
}
