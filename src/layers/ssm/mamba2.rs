use ndarray::{Array2, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use super::mamba::Mamba;
use crate::{
    mixtures::{HeadSelectionStrategy, MoHGating},
    network::Layer,
};

/// A pragmatic "Mamba-2 style" temporal mixer.
///
/// Implemented as a thin wrapper around the full `Mamba` reference
/// implementation to avoid duplicating scan/gradient logic.
///
/// Differences vs `Mamba`:
/// - larger default convolution kernel
#[derive(Serialize, Debug, Clone)]
pub struct Mamba2 {
    #[serde(flatten)]
    pub inner: Mamba,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoHMamba2 {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    #[serde(flatten)]
    pub moh: MoHGating,

    pub heads: Vec<Mamba2>,

    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_eff: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_head_out: Option<Vec<Array2<f32>>>,

    #[serde(skip_serializing, skip_deserializing)]
    pub last_avg_active_heads: Option<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub last_head_activity_vec: Option<Vec<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub last_token_head_activity_vec: Option<Vec<f32>>,
}

impl<'de> Deserialize<'de> for Mamba2 {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let inner = Mamba::deserialize(deserializer)?;
        Ok(Self { inner })
    }
}

impl Mamba2 {
    pub fn new(embed_dim: usize) -> Self {
        Self::new_with_kernel(embed_dim, 8)
    }

    pub fn new_with_kernel(embed_dim: usize, conv_kernel: usize) -> Self {
        Self {
            inner: Mamba::new_with_kernel(embed_dim, conv_kernel),
        }
    }

    #[inline]
    fn forward_view(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.inner.forward_mamba2_view(input)
    }

    #[inline]
    fn compute_gradients_view(
        &self,
        input: &ArrayView2<f32>,
        output_grads: &ArrayView2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.inner
            .compute_gradients_mamba2_view(input, output_grads)
    }
}

impl MoHMamba2 {
    pub fn new(embed_dim: usize, num_heads: usize, head_selection: &HeadSelectionStrategy) -> Self {
        let mut nh = num_heads.max(1);
        if embed_dim == 0 || !embed_dim.is_multiple_of(nh) {
            nh = 1;
        }
        let head_dim = if nh > 0 { embed_dim / nh } else { embed_dim };

        let budget = 1000usize;
        let gate_params = crate::richards::RichardsGate::new().parameters();
        let overhead = 2usize.saturating_mul(nh).saturating_add(gate_params);
        let max_wg = budget.saturating_sub(overhead);
        let gating_embed_dim = if nh > 0 {
            (max_wg / nh).max(1).min(embed_dim.max(1))
        } else {
            embed_dim.max(1)
        };

        let mut moh = MoHGating::new(gating_embed_dim, nh);
        moh.set_head_selection_config(head_selection);
        moh.head_selection_config.gating.use_learned_predictor = false;
        moh.threshold_predictor = None;
        moh.opt_w_tau = None;
        moh.opt_b_tau = None;
        moh.opt_w2_tau = None;
        moh.opt_b2_tau = None;
        moh.opt_cond_w_tau = None;

        let mut heads = Vec::with_capacity(nh);
        for _ in 0..nh {
            heads.push(Mamba2::new(head_dim));
        }

        Self {
            embed_dim,
            num_heads: nh,
            head_dim,
            moh,
            heads,
            cached_input: None,
            cached_eff: None,
            cached_head_out: None,
            last_avg_active_heads: None,
            last_head_activity_vec: None,
            last_token_head_activity_vec: None,
        }
    }

    #[inline]
    fn clear_caches(&mut self) {
        self.cached_input = None;
        self.cached_eff = None;
        self.cached_head_out = None;
        self.last_avg_active_heads = None;
        self.last_head_activity_vec = None;
        self.last_token_head_activity_vec = None;
    }

    pub fn take_tau_metrics(&mut self) -> Option<(f32, f32)> {
        self.moh.take_tau_metrics()
    }

    pub fn take_pred_norm(&mut self) -> Option<f32> {
        self.moh.take_pred_norm()
    }

    pub fn get_head_metrics_and_reset(&mut self) -> Vec<(f32, usize)> {
        self.moh.get_head_metrics_and_reset()
    }
}

impl Layer for Mamba2 {
    fn layer_type(&self) -> &str {
        "Mamba2"
    }

    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        self.inner.forward_mamba2(input)
    }

    fn backward(&mut self, grads: &ndarray::Array2<f32>, lr: f32) -> ndarray::Array2<f32> {
        self.inner.backward(grads, lr)
    }

    fn parameters(&self) -> usize {
        self.inner.parameters()
    }

    fn weight_norm(&self) -> f32 {
        self.inner.weight_norm()
    }

    fn compute_gradients(
        &self,
        input: &ndarray::Array2<f32>,
        output_grads: &ndarray::Array2<f32>,
    ) -> (ndarray::Array2<f32>, Vec<ndarray::Array2<f32>>) {
        self.inner.compute_gradients(input, output_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[ndarray::Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        self.inner.apply_gradients(gradients, learning_rate)
    }

    fn zero_gradients(&mut self) {
        self.inner.zero_gradients();
    }
}

impl Layer for MoHMamba2 {
    fn layer_type(&self) -> &str {
        "MoHMamba2"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 || self.num_heads == 0 || self.head_dim == 0 {
            self.clear_caches();
            self.cached_input = Some(input.clone());
            return Array2::<f32>::zeros((t, d));
        }

        self.cached_input = Some(input.clone());

        let gd = self.moh.w_g.nrows().min(d);
        let gate_input = input.slice(s![.., 0..gd]);
        let eff = self.moh.forward_weights_view(&gate_input, None, None);
        self.cached_eff = Some(eff.clone());

        let mut out = Array2::<f32>::zeros((t, d));
        let head_outs: Vec<Array2<f32>> = self
            .heads
            .par_iter_mut()
            .enumerate()
            .map(|(h, head)| {
                let c0 = h * self.head_dim;
                let c1 = c0 + self.head_dim;
                let x_view = input.slice(s![.., c0..c1]);
                head.forward_view(&x_view)
            })
            .collect();

        for (h, y_h) in head_outs.iter().enumerate().take(self.num_heads) {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let eff_col = eff.column(h);
            let eff_col = eff_col.insert_axis(Axis(1));
            let eff_col = eff_col
                .broadcast((t, self.head_dim))
                .expect("broadcast must succeed for (t, head_dim)");
            let mut out_block = out.slice_mut(s![.., c0..c1]);
            Zip::from(&mut out_block)
                .and(y_h)
                .and(eff_col)
                .for_each(|o, &y, &w| {
                    *o = y * w;
                });
        }

        self.cached_head_out = Some(head_outs);

        let avg = self
            .moh
            .head_selection_config
            .gating
            .get_avg_active_components();
        self.last_avg_active_heads = Some(avg);

        let mut hv = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let mean = eff.column(h).iter().map(|&x| x.max(0.0)).sum::<f32>() / (t.max(1) as f32);
            hv.push(mean);
        }
        self.last_head_activity_vec = Some(hv);
        let mut tv = Vec::with_capacity(t);
        for i in 0..t {
            let mut sum = 0.0f32;
            for h in 0..self.num_heads {
                let w = eff[[i, h]];
                sum += w.max(0.0);
            }
            let denom = self.num_heads.max(1) as f32;
            let v = if denom > 0.0 { sum / denom } else { 0.0 };
            tv.push(v.clamp(0.0, 1.0));
        }
        self.last_token_head_activity_vec = Some(tv);

        out
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (grad_input, param_grads) = self.compute_gradients(input, grads);
        let _ = self.apply_gradients(&param_grads, lr);
        grad_input
    }

    fn parameters(&self) -> usize {
        let heads_params: usize = self.heads.iter().map(|h| h.parameters()).sum();
        let mut moh_params = self.moh.w_g.len()
            + self.moh.alpha_g.len()
            + self.moh.beta_g.len()
            + self.moh.gate.parameters();
        if let Some(pred) = &self.moh.threshold_predictor {
            moh_params +=
                pred.weights1.len() + pred.bias1.len() + pred.weights2.len() + pred.bias2.len();
            moh_params += pred.cond_w.len();
            moh_params += pred.activation.scalar_weights_len();
        }
        heads_params + moh_params
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        for h in &self.heads {
            let wn = h.weight_norm();
            sumsq += wn * wn;
        }
        sumsq += self.moh.w_g.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.moh.alpha_g.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.moh.beta_g.iter().map(|&x| x * x).sum::<f32>();
        for w in self.moh.gate.curve.weights() {
            let wf = w as f32;
            sumsq += wf * wf;
        }
        if let Some(pred) = &self.moh.threshold_predictor {
            sumsq += pred.weights1.iter().map(|&x| x * x).sum::<f32>();
            sumsq += pred.bias1.iter().map(|&x| x * x).sum::<f32>();
            sumsq += pred.weights2.iter().map(|&x| x * x).sum::<f32>();
            sumsq += pred.bias2.iter().map(|&x| x * x).sum::<f32>();
            sumsq += pred.cond_w.iter().map(|&x| x * x).sum::<f32>();
            for w in pred.activation.weights() {
                let wf = w as f32;
                sumsq += wf * wf;
            }
        }
        sumsq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 || self.num_heads == 0 || self.head_dim == 0 {
            return (Array2::<f32>::zeros(input.raw_dim()), vec![]);
        }

        let can_use_cache = self
            .cached_input
            .as_ref()
            .is_some_and(|x| x.dim() == input.dim())
            && self.cached_input.as_ref().is_some_and(|x| {
                if std::ptr::eq(x, input) {
                    true
                } else {
                    x.iter()
                        .zip(input.iter())
                        .all(|(&a, &b)| a.to_bits() == b.to_bits())
                }
            });

        let eff_local: Array2<f32>;
        let eff: &Array2<f32> = if can_use_cache
            && let Some(e) = self
                .cached_eff
                .as_ref()
                .filter(|e| e.dim() == (t, self.num_heads))
        {
            e
        } else {
            let mut moh_tmp = self.moh.clone();
            let gd = moh_tmp.w_g.nrows().min(d);
            let gate_input = input.slice(s![.., 0..gd]);
            eff_local = moh_tmp.forward_weights_view(&gate_input, None, None);
            &eff_local
        };

        let head_outputs_local: Vec<Array2<f32>>;
        let head_outputs: &Vec<Array2<f32>> =
            if can_use_cache && let Some(v) = self.cached_head_out.as_ref() {
                let ok_len = v.len() == self.num_heads;
                let ok_dims = ok_len && v.iter().all(|y| y.dim() == (t, self.head_dim));
                if ok_dims {
                    v
                } else {
                    head_outputs_local = (0..self.num_heads)
                        .map(|h| {
                            let c0 = h * self.head_dim;
                            let c1 = c0 + self.head_dim;
                            let x_view = input.slice(s![.., c0..c1]);
                            let mut head = self.heads[h].clone();
                            head.forward_view(&x_view)
                        })
                        .collect();
                    &head_outputs_local
                }
            } else {
                head_outputs_local = (0..self.num_heads)
                    .map(|h| {
                        let c0 = h * self.head_dim;
                        let c1 = c0 + self.head_dim;
                        let x_view = input.slice(s![.., c0..c1]);
                        let mut head = self.heads[h].clone();
                        head.forward_view(&x_view)
                    })
                    .collect();
                &head_outputs_local
            };

        let mut eff_grads = Array2::<f32>::zeros((t, self.num_heads));
        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            for i in 0..t {
                let mut acc = 0.0f32;
                for j in 0..self.head_dim {
                    acc += output_grads[[i, c0 + j]] * head_outputs[h][[i, j]];
                }
                eff_grads[[i, h]] = acc;
            }
        }

        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
        let mut grads: Vec<Array2<f32>> = Vec::new();

        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let x_view = input.slice(s![.., c0..c1]);

            let mut scaled_grads = Array2::<f32>::zeros((t, self.head_dim));
            let eff_col = eff.column(h);
            let eff_col = eff_col.insert_axis(Axis(1));
            let eff_col = eff_col
                .broadcast((t, self.head_dim))
                .expect("broadcast must succeed for (t, head_dim)");
            let og_block = output_grads.slice(s![.., c0..c1]);
            Zip::from(&mut scaled_grads)
                .and(og_block)
                .and(eff_col)
                .for_each(|sg, &og, &w| {
                    *sg = og * w;
                });

            let scaled_grads_view = scaled_grads.view();
            let (dx_h, pgrads_h) = if can_use_cache {
                self.heads[h].compute_gradients_view(&x_view, &scaled_grads_view)
            } else {
                let mut head = self.heads[h].clone();
                head.forward_view(&x_view);
                head.compute_gradients_view(&x_view, &scaled_grads_view)
            };
            let mut gi_block = grad_input.slice_mut(s![.., c0..c1]);
            gi_block += &dx_h;
            grads.extend(pgrads_h);
        }

        let (dx_moh, moh_grads) = {
            let mut moh_local = self.moh.clone();
            let gd = moh_local.w_g.nrows().min(d);
            let gate_input = input.slice(s![.., 0..gd]);
            moh_local.compute_gradients_from_eff_view(&gate_input, &eff_grads)
        };
        {
            let gd = self.moh.w_g.nrows().min(d);
            let mut gi = grad_input.slice_mut(s![.., 0..gd]);
            gi += &dx_moh;
        }
        grads.extend(moh_grads);

        (grad_input, grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        let per_head = 14usize;
        let needed_heads = self.num_heads * per_head;
        if gradients.len() < needed_heads + 4 {
            return Ok(());
        }

        let mut idx = 0usize;
        for h in 0..self.num_heads {
            let slice = &gradients[idx..idx + per_head];
            self.heads[h].apply_gradients(slice, learning_rate)?;
            idx += per_head;
        }

        let moh_slice = &gradients[idx..];
        self.moh.apply_gradients(moh_slice, learning_rate)?;
        Ok(())
    }

    fn zero_gradients(&mut self) {
        for h in &mut self.heads {
            h.zero_gradients();
        }
        self.moh.cached_soft_top_p_mask = None;
        self.clear_caches();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn mamba2_forward_backward_shapes() {
        let mut layer = Mamba2::new_with_kernel(16, 5);
        let x = Array2::<f32>::zeros((8, 16));
        let y = layer.forward(&x);
        assert_eq!(y.shape(), [8, 16]);

        let grads = Array2::<f32>::ones((8, 16));
        let dx = layer.backward(&grads, 1e-3);
        assert_eq!(dx.shape(), [8, 16]);
        assert!(dx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn moh_mamba2_forward_shape() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba2::new(16, 4, &cfg);
        let x = Array2::<f32>::from_elem((7, 16), 0.1);
        let y = layer.forward(&x);
        assert_eq!(y.dim(), (7, 16));
        assert!(layer.last_avg_active_heads.is_some());
        assert!(
            layer
                .last_head_activity_vec
                .as_ref()
                .is_some_and(|v| v.len() == 4)
        );
    }

    #[test]
    fn moh_mamba2_grad_shapes() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba2::new(12, 3, &cfg);
        let x = Array2::<f32>::from_elem((5, 12), 0.2);
        let y = layer.forward(&x);
        let grads = Array2::<f32>::from_elem(y.dim(), 0.1);

        let (dx, pgrads) = layer.compute_gradients(&x, &grads);
        assert_eq!(dx.dim(), x.dim());
        assert!(pgrads.len() >= 3 * 14 + 4);
    }

    #[test]
    fn moh_mamba2_compute_gradients_without_forward_is_finite() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let layer = MoHMamba2::new(12, 3, &cfg);
        let x = Array2::from_shape_fn((7, 12), |(i, j)| ((i * 12 + j) as f32 * 0.017).sin());
        let grads = Array2::<f32>::from_elem((7, 12), 0.1);

        let expected_len = 3 * 14 + layer.moh.grad_arrays_len();
        let (dx, pgrads) = layer.compute_gradients(&x, &grads);

        assert_eq!(dx.dim(), x.dim());
        assert!(dx.iter().all(|v| v.is_finite()));
        assert_eq!(pgrads.len(), expected_len);
        assert!(pgrads.iter().all(|g| g.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn moh_mamba2_backward_updates_output() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba2::new(12, 3, &cfg);
        let x = Array2::from_shape_fn((9, 12), |(i, j)| ((i * 12 + j) as f32 * 0.019).sin());
        let y0 = layer.forward(&x);

        let grads = Array2::<f32>::from_elem(y0.dim(), 0.1);
        let dx = layer.backward(&grads, 1e-2);
        assert_eq!(dx.dim(), x.dim());
        assert!(dx.iter().all(|v| v.is_finite()));

        let y1 = layer.forward(&x);
        let delta: f32 = (&y1 - &y0).mapv(|v| v.abs()).sum();
        assert!(delta.is_finite());
        assert!(delta > 0.0);
    }

    #[test]
    fn moh_mamba2_parameter_delta_within_1000() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let layer = MoHMamba2::new(64, 16, &cfg);
        let baseline: usize = layer.heads.iter().map(|h| h.parameters()).sum();
        let moh_total = layer.parameters();
        assert!(moh_total >= baseline);
        assert!(moh_total - baseline <= 1000);
    }
}
