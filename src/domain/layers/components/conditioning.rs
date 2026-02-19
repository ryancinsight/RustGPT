//! Shared Conditioning Components
//!
//! This module contains components for time embeddings, FiLM conditioning,
//! and other modulation mechanisms used in diffusion models and potentially others.

use std::borrow::Cow;

use ndarray::{Array1, Array2, ArrayView1, linalg::general_mat_mul};
use serde::{Deserialize, Serialize};

use crate::{domain::richards::RichardsCurve, infrastructure::optimizer::adam::Adam};

/// Minimum number of elements before FiLM switches to parallel row processing.
const FILM_PARALLEL_MIN_ELEMENTS: usize = 4_096;

/// Standard transformer-style sinusoidal time embedding
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TimeEmbedding {
    pub b: Array1<f32>,
}

impl TimeEmbedding {
    pub fn new(embed_dim: usize) -> Self {
        let b = Array1::zeros(embed_dim);
        Self { b }
    }

    pub fn forward(&self, t: usize, max_t: usize) -> Array1<f32> {
        // Standard transformer-style sinusoidal embedding with log-spaced frequencies.
        // Uses a normalized timestep in [0,1] to make embeddings stable across different T.
        let dim = self.b.len();
        let mut emb = Array1::zeros(dim);
        let half_dim = dim / 2;
        if half_dim == 0 {
            return emb;
        }
        let t_norm = if max_t > 1 {
            t as f32 / (max_t - 1) as f32
        } else {
            0.0
        };
        let base: f32 = 10_000.0;
        for i in 0..half_dim {
            let exponent = (i as f32) / (half_dim as f32);
            let inv_freq = base.powf(-exponent);
            let arg = t_norm * inv_freq;
            emb[2 * i] = arg.sin();
            if 2 * i + 1 < dim {
                emb[2 * i + 1] = arg.cos();
            }
        }
        emb
    }
}

/// MLP for processing time embeddings into FiLM modulation parameters
#[derive(Serialize, Deserialize, Debug)]
pub struct TimeConditioner {
    pub w1: Array2<f32>,
    pub b1: Array2<f32>,
    pub w2: Array2<f32>,
    pub b2: Array2<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_w1: Option<Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_b1: Option<Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_w2: Option<Adam>,
    #[serde(skip_serializing, skip_deserializing)]
    pub opt_b2: Option<Adam>,
    pub ema_w1: Array2<f32>,
    pub ema_b1: Array2<f32>,
    pub ema_w2: Array2<f32>,
    pub ema_b2: Array2<f32>,
}

impl TimeConditioner {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Initialize with small random weights
        let scale = (2.0 / input_dim as f32).sqrt();
        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            (rand::random::<f32>() - 0.5) * scale
        });
        let b1 = Array2::zeros((hidden_dim, 1));

        let scale2 = (2.0 / hidden_dim as f32).sqrt();
        let w2 = Array2::from_shape_fn((output_dim, hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * scale2
        });
        let b2 = Array2::zeros((output_dim, 1));

        Self {
            ema_w1: w1.clone(),
            ema_b1: b1.clone(),
            ema_w2: w2.clone(),
            ema_b2: b2.clone(),
            w1,
            b1,
            w2,
            b2,
            opt_w1: Some(Adam::new((1, 1))),
            opt_b1: Some(Adam::new((1, 1))),
            opt_w2: Some(Adam::new((1, 1))),
            opt_b2: Some(Adam::new((1, 1))),
        }
    }

    /// Forward pass returning (gamma_beta, hidden_state)
    pub fn forward(&self, input: &Array1<f32>, use_ema: bool) -> (Array1<f32>, Array2<f32>) {
        let (w1, b1, w2, b2) = if use_ema {
            (&self.ema_w1, &self.ema_b1, &self.ema_w2, &self.ema_b2)
        } else {
            (&self.w1, &self.b1, &self.w2, &self.b2)
        };

        // Layer 1: Linear -> Swish/SiLU
        // input shape: [dim]
        // w1 shape: [hidden, dim]
        // Compute w1 * input using general_mat_mul to avoid intermediate allocation
        let mut h_pre = Array1::zeros(w1.nrows());
        {
            let h_pre_len = h_pre.len();
            let input_len = input.len();
            let mut h_pre_2d = h_pre
                .view_mut()
                .into_shape_with_order((h_pre_len, 1))
                .unwrap();
            let input_2d = input.view().into_shape_with_order((input_len, 1)).unwrap();
            general_mat_mul(1.0, &w1, &input_2d, 0.0, &mut h_pre_2d);
        }

        // Add bias (broadcast)
        h_pre += &b1.column(0);

        // Swish activation: x * sigmoid(x)
        let h = h_pre.mapv(|x| x / (1.0 + (-x).exp()));

        // Layer 2: Linear
        // Compute w2 * h using general_mat_mul to avoid intermediate allocation
        let mut out = Array1::zeros(w2.nrows());
        {
            let out_len = out.len();
            let h_len = h.len();
            let mut out_2d = out.view_mut().into_shape_with_order((out_len, 1)).unwrap();
            let h_2d = h.view().into_shape_with_order((h_len, 1)).unwrap();
            general_mat_mul(1.0, &w2, &h_2d, 0.0, &mut out_2d);
        }

        out += &b2.column(0);

        // Return output and hidden state (for caching/skip connections if needed)
        // Hidden state returned as 2D [1, hidden] for consistency with previous API
        let h_2d = h.insert_axis(ndarray::Axis(0));

        (out, h_2d)
    }

    pub fn compute_gradients(
        &self,
        input: &Array1<f32>,
        grad_output: &Array1<f32>,
    ) -> (
        Array1<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
        Array2<f32>,
    ) {
        // Recompute forward pass for gradients
        // We assume non-EMA weights for training
        let mut h_pre = Array1::zeros(self.w1.nrows());
        {
            let h_pre_len = h_pre.len();
            let input_len = input.len();
            let mut h_pre_2d = h_pre
                .view_mut()
                .into_shape_with_order((h_pre_len, 1))
                .unwrap();
            let input_2d = input.view().into_shape_with_order((input_len, 1)).unwrap();
            general_mat_mul(1.0, &self.w1, &input_2d, 0.0, &mut h_pre_2d);
        }
        h_pre += &self.b1.column(0);
        let h = h_pre.mapv(|x| x / (1.0 + (-x).exp())); // Swish

        // Layer 2 gradients
        // out = w2 * h + b2
        // dL/dw2 = dL/dout * h^T
        // dL/db2 = dL/dout
        // dL/dh = w2^T * dL/dout

        let grad_output_col = grad_output.clone().insert_axis(ndarray::Axis(1));
        let h_row = h.clone().insert_axis(ndarray::Axis(0));
        let mut grad_w2 = Array2::zeros((grad_output.len(), h.len()));
        general_mat_mul(1.0, &grad_output_col, &h_row, 0.0, &mut grad_w2);

        let grad_b2 = grad_output.clone().insert_axis(ndarray::Axis(1));

        let mut grad_h = Array1::zeros(self.w2.ncols());
        let grad_h_len = grad_h.len();
        let grad_output_len = grad_output.len();
        {
            let mut grad_h_2d = grad_h
                .view_mut()
                .into_shape_with_order((grad_h_len, 1))
                .unwrap();
            let grad_output_2d = grad_output
                .view()
                .into_shape_with_order((grad_output_len, 1))
                .unwrap();
            general_mat_mul(1.0, &self.w2.t(), &grad_output_2d, 0.0, &mut grad_h_2d);
        }

        // Swish gradient: f'(x) = f(x) + sigmoid(x)(1 - f(x))
        let sigmoid_h_pre = h_pre.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let grad_h_pre = &grad_h * (&h + &sigmoid_h_pre * (1.0 - &h));

        // Layer 1 gradients
        let grad_h_pre_col = grad_h_pre.clone().insert_axis(ndarray::Axis(1));
        let input_row = input.clone().insert_axis(ndarray::Axis(0));
        let mut grad_w1 = Array2::zeros((grad_h_pre.len(), input.len()));
        general_mat_mul(1.0, &grad_h_pre_col, &input_row, 0.0, &mut grad_w1);

        let grad_b1 = grad_h_pre.clone().insert_axis(ndarray::Axis(1));

        let mut grad_input = Array1::zeros(self.w1.ncols());
        let grad_input_len = grad_input.len();
        let grad_h_pre_len = grad_h_pre.len();
        {
            let mut grad_input_2d = grad_input
                .view_mut()
                .into_shape_with_order((grad_input_len, 1))
                .unwrap();
            let grad_h_pre_2d = grad_h_pre
                .view()
                .into_shape_with_order((grad_h_pre_len, 1))
                .unwrap();
            general_mat_mul(1.0, &self.w1.t(), &grad_h_pre_2d, 0.0, &mut grad_input_2d);
        }

        (grad_input, grad_w1, grad_b1, grad_w2, grad_b2)
    }

    pub fn apply_gradients<T: AsRef<Array2<f32>>>(&mut self, grads: &[T], lr: f32, ema_decay: f32) {
        if grads.len() != 4 {
            tracing::error!(
                "TimeConditioner::apply_gradients expected 4 gradients, got {}",
                grads.len()
            );
            return;
        }
        let g_w1 = grads[0].as_ref();
        let g_b1 = grads[1].as_ref();
        let g_w2 = grads[2].as_ref();
        let g_b2 = grads[3].as_ref();

        if let Some(opt) = &mut self.opt_w2 {
            opt.step(&mut self.w2, g_w2, lr);
        }
        if let Some(opt) = &mut self.opt_b2 {
            opt.step(&mut self.b2, g_b2, lr);
        }
        if let Some(opt) = &mut self.opt_w1 {
            opt.step(&mut self.w1, g_w1, lr);
        }
        if let Some(opt) = &mut self.opt_b1 {
            opt.step(&mut self.b1, g_b1, lr);
        }

        // Update EMA
        let d = ema_decay;
        self.ema_w2
            .zip_mut_with(&self.w2, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_b2
            .zip_mut_with(&self.b2, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_w1
            .zip_mut_with(&self.w1, |e, &w| *e = d * *e + (1.0 - d) * w);
        self.ema_b1
            .zip_mut_with(&self.b1, |e, &w| *e = d * *e + (1.0 - d) * w);
    }

    pub fn weight_norm(&self) -> f32 {
        (self.w1.iter().map(|&w| w * w).sum::<f32>() + self.w2.iter().map(|&w| w * w).sum::<f32>())
            .sqrt()
    }

    pub fn backward(
        &self,
        grad_output: &Array1<f32>,
        _hidden_state: &Array2<f32>,
        input: &Array1<f32>,
    ) -> (Array1<f32>, Vec<Array2<f32>>) {
        let (grad_input, grad_w1, grad_b1, grad_w2, grad_b2) =
            self.compute_gradients(input, grad_output);
        (grad_input, vec![grad_w1, grad_b1, grad_w2, grad_b2])
    }
}

/// Shared FiLM modulation buffers and logic
#[derive(Serialize, Deserialize, Debug)]
pub struct SharedFilmModulation {
    pub gamma_attn: Array2<f32>,
    pub beta_attn: Array2<f32>,
    pub gamma_ffn: Array2<f32>,
    pub beta_ffn: Array2<f32>,
    pub scale_gamma: f32,
    pub scale_beta: f32,
    #[serde(skip)]
    scratch: Vec<f32>,
    /// Cached capacity (power-of-2) to minimize reallocations
    #[serde(skip)]
    scratch_capacity: usize,
}

#[inline]
fn apply_film_inplace(
    output: &mut Array2<f32>,
    gamma: ArrayView1<'_, f32>,
    beta: ArrayView1<'_, f32>,
) {
    if output.len() >= FILM_PARALLEL_MIN_ELEMENTS && output.nrows() > 1 {
        ndarray::Zip::from(output.outer_iter_mut()).par_for_each(|mut row| {
            row.zip_mut_with(&gamma, |x, &g| *x *= g);
            row.zip_mut_with(&beta, |x, &b| *x += b);
        });
    } else {
        for mut row in output.outer_iter_mut() {
            row.zip_mut_with(&gamma, |x, &g| *x *= g);
            row.zip_mut_with(&beta, |x, &b| *x += b);
        }
    }
}

#[inline]
fn apply_delta_film_inplace(
    output: &mut Array2<f32>,
    gamma_delta: ArrayView1<'_, f32>,
    beta: ArrayView1<'_, f32>,
) {
    if output.len() >= FILM_PARALLEL_MIN_ELEMENTS && output.nrows() > 1 {
        ndarray::Zip::from(output.outer_iter_mut()).par_for_each(|mut row| {
            row.zip_mut_with(&gamma_delta, |x, &g| *x *= 1.0 + g);
            row.zip_mut_with(&beta, |x, &b| *x += b);
        });
    } else {
        for mut row in output.outer_iter_mut() {
            row.zip_mut_with(&gamma_delta, |x, &g| *x *= 1.0 + g);
            row.zip_mut_with(&beta, |x, &b| *x += b);
        }
    }
}

/// Apply FiLM modulation with delta-gamma (`x *= 1 + gamma`) when both inputs are present.
///
/// Returns a borrowed input when conditioning is disabled, and an owned
/// conditioned tensor when enabled.
pub fn apply_optional_delta_film<'a>(
    input: &'a Array2<f32>,
    gamma: Option<ArrayView1<'_, f32>>,
    beta: Option<ArrayView1<'_, f32>>,
) -> Cow<'a, Array2<f32>> {
    match (gamma, beta) {
        (Some(gamma), Some(beta)) => {
            let mut out = input.clone();
            apply_delta_film_inplace(&mut out, gamma, beta);
            Cow::Owned(out)
        }
        _ => Cow::Borrowed(input),
    }
}

impl Default for SharedFilmModulation {
    fn default() -> Self {
        Self {
            gamma_attn: Array2::zeros((1, 0)),
            beta_attn: Array2::zeros((1, 0)),
            gamma_ffn: Array2::zeros((1, 0)),
            beta_ffn: Array2::zeros((1, 0)),
            scale_gamma: 0.1,
            scale_beta: 0.1,
            scratch: Vec::new(),
            scratch_capacity: 0,
        }
    }
}

impl SharedFilmModulation {
    pub fn new(embed_dim: usize) -> Self {
        Self::with_scales(embed_dim, 0.1, 0.1)
    }

    pub fn with_scales(embed_dim: usize, scale_gamma: f32, scale_beta: f32) -> Self {
        Self {
            gamma_attn: Array2::zeros((1, embed_dim)),
            beta_attn: Array2::zeros((1, embed_dim)),
            gamma_ffn: Array2::zeros((1, embed_dim)),
            beta_ffn: Array2::zeros((1, embed_dim)),
            scale_gamma,
            scale_beta,
            scratch: Vec::new(),
            scratch_capacity: 0,
        }
    }

    /// Update modulation parameters from a flat gamma_beta vector
    pub fn update(&mut self, gamma_beta: &[f32], embed_dim: usize) {
        let tanh = RichardsCurve::tanh(false);

        // Ensure buffers are sized correctly (in case embed_dim changed or init was empty)
        if self.gamma_attn.len() != embed_dim {
            self.gamma_attn = Array2::zeros((1, embed_dim));
            self.beta_attn = Array2::zeros((1, embed_dim));
            self.gamma_ffn = Array2::zeros((1, embed_dim));
            self.beta_ffn = Array2::zeros((1, embed_dim));
        }

        // Use power-of-2 sizing for scratch buffer to minimize reallocations
        let required_len = gamma_beta.len();
        let new_capacity = required_len.next_power_of_two().max(64);
        if new_capacity > self.scratch_capacity {
            self.scratch.resize(new_capacity, 0.0);
            self.scratch_capacity = new_capacity;
        }
        // Clear only the portion we'll use
        self.scratch[..required_len].fill(0.0);
        tanh.forward_into_f32(gamma_beta, &mut self.scratch[..required_len]);

        let ga = self.gamma_attn.as_slice_mut().unwrap();
        let ba = self.beta_attn.as_slice_mut().unwrap();
        let gf = self.gamma_ffn.as_slice_mut().unwrap();
        let bf = self.beta_ffn.as_slice_mut().unwrap();

        for j in 0..embed_dim {
            // Mapping based on DiffusionBlock logic:
            // 0..embed: gamma_attn
            // embed..2*embed: beta_attn
            // 2*embed..3*embed: gamma_ffn
            // 3*embed..4*embed: beta_ffn

            ga[j] = 1.0 + self.scale_gamma * self.scratch[j];
            ba[j] = self.scale_beta * self.scratch[embed_dim + j];
            gf[j] = 1.0 + self.scale_gamma * self.scratch[2 * embed_dim + j];
            bf[j] = self.scale_beta * self.scratch[3 * embed_dim + j];
        }
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.gamma_attn.len() * std::mem::size_of::<f32>();
        size += self.beta_attn.len() * std::mem::size_of::<f32>();
        size += self.gamma_ffn.len() * std::mem::size_of::<f32>();
        size += self.beta_ffn.len() * std::mem::size_of::<f32>();
        size += self.scratch.capacity() * std::mem::size_of::<f32>();
        size
    }

    pub fn gamma_attn(&self) -> ArrayView1<'_, f32> {
        self.gamma_attn.row(0)
    }

    pub fn beta_attn(&self) -> ArrayView1<'_, f32> {
        self.beta_attn.row(0)
    }

    pub fn gamma_ffn(&self) -> ArrayView1<'_, f32> {
        self.gamma_ffn.row(0)
    }

    pub fn beta_ffn(&self) -> ArrayView1<'_, f32> {
        self.beta_ffn.row(0)
    }

    fn apply_conditioning(
        input: &Array2<f32>,
        gamma: ArrayView1<f32>,
        beta: ArrayView1<f32>,
    ) -> Array2<f32> {
        let mut out = input.clone();
        apply_film_inplace(&mut out, gamma, beta);
        out
    }

    fn apply_conditioning_into(
        input: &Array2<f32>,
        gamma: ArrayView1<f32>,
        beta: ArrayView1<f32>,
        output: &mut Array2<f32>,
    ) -> crate::common::errors::Result<()> {
        if output.dim() != input.dim() {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "FiLM output dimension mismatch: expected {:?}, got {:?}",
                    input.dim(),
                    output.dim()
                ),
            });
        }
        output.assign(input);
        apply_film_inplace(output, gamma, beta);
        Ok(())
    }

    pub fn apply_attn_conditioning(&self, input: &Array2<f32>) -> Array2<f32> {
        Self::apply_conditioning(input, self.gamma_attn(), self.beta_attn())
    }

    pub fn apply_ffn_conditioning(&self, input: &Array2<f32>) -> Array2<f32> {
        Self::apply_conditioning(input, self.gamma_ffn(), self.beta_ffn())
    }

    pub fn apply_attn_conditioning_into(
        &self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> crate::common::errors::Result<()> {
        Self::apply_conditioning_into(input, self.gamma_attn(), self.beta_attn(), output)
    }

    pub fn apply_ffn_conditioning_into(
        &self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> crate::common::errors::Result<()> {
        Self::apply_conditioning_into(input, self.gamma_ffn(), self.beta_ffn(), output)
    }

    pub fn film_backward(
        &self,
        output_grads: &Array2<f32>,
        input: &Array2<f32>,
        gamma: &Array2<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array1<f32>) {
        let embed_dim = input.ncols();
        let gamma_row = gamma.row(0);

        let mut input_grads = output_grads.clone();
        for mut row in input_grads.outer_iter_mut() {
            row.zip_mut_with(&gamma_row, |g, &ga| *g *= ga);
        }

        // Use vectorized operations instead of nested loops
        let mut grad_gamma = Array1::<f32>::zeros(embed_dim);
        let mut grad_beta = Array1::<f32>::zeros(embed_dim);

        // Compute grad_gamma = output_grads^T · input and grad_beta = sum(output_grads)
        for j in 0..embed_dim {
            let mut sum_gamma = 0.0f32;
            let mut sum_beta = 0.0f32;
            for i in 0..output_grads.nrows() {
                let go = output_grads[[i, j]];
                sum_gamma += go * input[[i, j]];
                sum_beta += go;
            }
            grad_gamma[j] = sum_gamma;
            grad_beta[j] = sum_beta;
        }

        (input_grads, grad_gamma, grad_beta)
    }

    pub fn compute_mlp_gradients(
        &self,
        grad_gamma_attn: &Array1<f32>,
        grad_beta_attn: &Array1<f32>,
        grad_gamma_ffn: &Array1<f32>,
        grad_beta_ffn: &Array1<f32>,
        gamma_attn: &Array2<f32>,
        beta_attn: &Array2<f32>,
        gamma_ffn: &Array2<f32>,
        beta_ffn: &Array2<f32>,
        embed_dim: usize,
    ) -> Array1<f32> {
        let mut grad = Array1::<f32>::zeros(embed_dim * 4);
        let sg = self.scale_gamma.max(1e-6);
        let sb = self.scale_beta.max(1e-6);

        for j in 0..embed_dim {
            let ga_tanh = ((gamma_attn[[0, j]] - 1.0) / sg).clamp(-1.0, 1.0);
            let ba_tanh = (beta_attn[[0, j]] / sb).clamp(-1.0, 1.0);
            let gf_tanh = ((gamma_ffn[[0, j]] - 1.0) / sg).clamp(-1.0, 1.0);
            let bf_tanh = (beta_ffn[[0, j]] / sb).clamp(-1.0, 1.0);

            let d_ga = sg * (1.0 - ga_tanh * ga_tanh);
            let d_ba = sb * (1.0 - ba_tanh * ba_tanh);
            let d_gf = sg * (1.0 - gf_tanh * gf_tanh);
            let d_bf = sb * (1.0 - bf_tanh * bf_tanh);

            grad[j] = grad_gamma_attn[j] * d_ga;
            grad[embed_dim + j] = grad_beta_attn[j] * d_ba;
            grad[2 * embed_dim + j] = grad_gamma_ffn[j] * d_gf;
            grad[3 * embed_dim + j] = grad_beta_ffn[j] * d_bf;
        }

        grad
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array2, array};

    use super::{SharedFilmModulation, apply_optional_delta_film};

    #[test]
    fn apply_optional_delta_film_borrows_when_disabled() {
        let input = Array2::<f32>::ones((2, 3));
        let conditioned = apply_optional_delta_film(&input, None, None);
        assert!(matches!(conditioned, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn apply_optional_delta_film_applies_delta_gamma_and_beta() {
        let input = array![[1.0f32, 2.0f32]];
        let gamma = array![0.5f32, -0.5f32];
        let beta = array![0.25f32, 1.0f32];

        let conditioned = apply_optional_delta_film(&input, Some(gamma.view()), Some(beta.view()));
        let output = conditioned.into_owned();

        assert!((output[[0, 0]] - 1.75).abs() < 1e-6);
        assert!((output[[0, 1]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn film_modulation_scratch_buffer_power_of_two_sizing() {
        let mut film = SharedFilmModulation::new(64);

        // First update with 4*64 = 256 elements
        let gamma_beta: Vec<f32> = vec![0.5; 256];
        film.update(&gamma_beta, 64);

        // Scratch capacity should be power-of-2 and >= 256
        assert!(film.scratch_capacity >= 256);
        assert!(film.scratch_capacity.is_power_of_two());

        // Second update with same size should not reallocate
        let old_capacity = film.scratch_capacity;
        let gamma_beta2: Vec<f32> = vec![0.3; 256];
        film.update(&gamma_beta2, 64);
        assert_eq!(film.scratch_capacity, old_capacity);

        // Larger update should reallocate to next power-of-2
        let gamma_beta3: Vec<f32> = vec![0.1; 512];
        film.update(&gamma_beta3, 128);
        assert!(film.scratch_capacity >= 512);
        assert!(film.scratch_capacity.is_power_of_two());
    }

    #[test]
    fn film_modulation_memory_usage() {
        let film = SharedFilmModulation::new(64);
        let usage = film.memory_usage_bytes();

        // Should account for 4 arrays of 64 elements each + scratch buffer
        // gamma_attn, beta_attn, gamma_ffn, beta_ffn = 4 * 64 * 4 bytes = 1024 bytes
        assert!(usage >= 1024);
    }
}
