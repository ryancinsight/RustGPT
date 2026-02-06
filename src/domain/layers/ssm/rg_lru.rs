use std::borrow::Cow;

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    infrastructure::optimizer::adam::Adam,
    common::{errors::Result, rng::get_rng},
    domain::{
        mixtures::{HeadSelectionStrategy, MoHGating},
        network::Layer,
        richards::RichardsCurve,
    },
};

type GatesAndState<'a> = (
    Cow<'a, Array2<f32>>,
    Cow<'a, Array2<f32>>,
    Cow<'a, Array2<f32>>,
    Cow<'a, Array2<f32>>,
);

#[inline]
fn array2_bitwise_eq_f32(a: &Array2<f32>, b: &Array2<f32>) -> bool {
    if a.dim() != b.dim() {
        return false;
    }
    if std::ptr::eq(a, b) {
        return true;
    }
    match (a.as_slice_memory_order(), b.as_slice_memory_order()) {
        (Some(sa), Some(sb)) => sa
            .iter()
            .zip(sb.iter())
            .all(|(&x, &y)| x.to_bits() == y.to_bits()),
        _ => a
            .iter()
            .zip(b.iter())
            .all(|(&x, &y)| x.to_bits() == y.to_bits()),
    }
}

#[inline]
fn array2_bitwise_eq_base_f32<D: Data<Elem = f32>>(a: &Array2<f32>, b: &ArrayBase<D, Ix2>) -> bool {
    if a.dim() != b.dim() {
        return false;
    }
    match (a.as_slice_memory_order(), b.as_slice_memory_order()) {
        (Some(sa), Some(sb)) => sa
            .iter()
            .zip(sb.iter())
            .all(|(&x, &y)| x.to_bits() == y.to_bits()),
        _ => a
            .iter()
            .zip(b.iter())
            .all(|(&x, &y)| x.to_bits() == y.to_bits()),
    }
}

#[derive(Copy, Clone)]
struct GatesParams<'a> {
    w_a: &'a Array2<f32>,
    b_a: &'a Array2<f32>,
    w_x: &'a Array2<f32>,
    b_x: &'a Array2<f32>,
    lambda: &'a Array2<f32>,
}

#[inline]
fn softplus(x: f32) -> f32 {
    crate::domain::soft::softplus(x)
}

/// Real-Gated Linear Recurrent Unit (RG-LRU) layer.
///
/// This is a trainable temporal-mixing layer that maps (T × D) → (T × D)
/// using a diagonal, stable recurrence. This implementation currently computes
/// gradients with full backpropagation through time (BPTT) across the recurrent state.
#[derive(Serialize, Debug, Clone)]
pub struct RgLru {
    pub embed_dim: usize,

    // Gates: r_t = σ(x W_a + b_a), i_t = σ(x W_x + b_x)
    pub w_a: Array2<f32>,
    pub b_a: Array2<f32>, // [1, D]
    pub w_x: Array2<f32>,
    pub b_x: Array2<f32>, // [1, D]

    // Diagonal recurrence parameterization: a = σ(lambda)
    pub lambda: Array2<f32>, // [1, D]

    #[serde(skip_serializing)]
    opt_w_a: Adam,
    #[serde(skip_serializing)]
    opt_b_a: Adam,
    #[serde(skip_serializing)]
    opt_w_x: Adam,
    #[serde(skip_serializing)]
    opt_b_x: Adam,
    #[serde(skip_serializing)]
    opt_lambda: Adam,

    // Forward caches (optional; used to avoid recompute in backward)
    #[serde(skip_serializing)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_r: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_i: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_a: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_hprev: Option<Array2<f32>>, // h_{t-1} per t (hprev[0]=0)

    #[serde(skip)]
    streaming_state: Option<Array1<f32>>,
}

/// Multi-head RG-LRU with shared Mixture-of-Heads (MoH) gating.
///
/// Splits the embedding dimension into `num_heads` chunks, runs an independent
/// RG-LRU per head, then scales each head output by MoH effective weights.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoHRgLru {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    #[serde(flatten)]
    pub moh: MoHGating,

    pub heads: Vec<RgLru>,

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

impl MoHRgLru {
    pub fn new(embed_dim: usize, num_heads: usize, head_selection: &HeadSelectionStrategy) -> Self {
        let mut nh = num_heads.max(1);
        if embed_dim == 0 || embed_dim % nh != 0 {
            nh = 1;
        }
        let head_dim = if nh > 0 { embed_dim / nh } else { embed_dim };

        let budget = 1000usize;
        let gate_params = crate::domain::richards::RichardsGate::new().parameters();
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
            heads.push(RgLru::new(head_dim));
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

    /// Set verification overrides for max_abs_z (for parity testing)
    pub fn set_verification_overrides(&mut self, overrides: Option<Vec<f64>>) {
        self.moh.set_verification_overrides(overrides);
    }

    /// Get last max_abs_z (for parity testing)
    pub fn get_last_max_abs_z(&self) -> Option<Vec<f64>> {
        self.moh.last_max_abs_z.clone()
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // 1. Compute MoH gating weights
        let d = input.len();
        let input_2d = input.to_shape((1, d)).unwrap().to_owned();
        let eff_weights_2d = self.moh.forward_weights(&input_2d, None, None);
        let eff_weights = eff_weights_2d.row(0);

        // 2. Split input into heads
        let head_dim = self.head_dim;
        let mut output = Array1::<f32>::zeros(d);

        // 3. Process each head
        for (h, head) in self.heads.iter_mut().enumerate() {
            let start = h * head_dim;
            let end = start + head_dim;
            if start >= d { break; }
            
            let head_input = input.slice(s![start..end]).to_owned();
            let head_out = head.forward_step(&head_input);
            
            let w = eff_weights[h];
            if w.abs() > 1e-9 {
                 let mut out_slice = output.slice_mut(s![start..end]);
                 ndarray::Zip::from(&mut out_slice).and(&head_out).for_each(|o, &v| *o += w * v);
            }
        }
        
        output
    }
}

impl<'de> Deserialize<'de> for RgLru {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RgLruSerde {
            embed_dim: usize,
            w_a: Array2<f32>,
            b_a: Array2<f32>,
            w_x: Array2<f32>,
            b_x: Array2<f32>,
            lambda: Array2<f32>,
        }

        let data = RgLruSerde::deserialize(deserializer)?;
        let embed_dim = data.embed_dim;

        Ok(Self {
            embed_dim,
            w_a: data.w_a,
            b_a: data.b_a,
            w_x: data.w_x,
            b_x: data.b_x,
            lambda: data.lambda,
            opt_w_a: Adam::new((embed_dim, embed_dim)),
            opt_b_a: Adam::new((1, embed_dim)),
            opt_w_x: Adam::new((embed_dim, embed_dim)),
            opt_b_x: Adam::new((1, embed_dim)),
            opt_lambda: Adam::new((1, embed_dim)),
            cached_input: None,
            cached_r: None,
            cached_i: None,
            cached_a: None,
            cached_hprev: None,
            streaming_state: None,
        })
    }
}

impl RgLru {
    pub fn new(embed_dim: usize) -> Self {
        let mut rng = get_rng();
        // LeCun-ish init (Normal(0, sqrt(1/fan_in))) to keep gates sane.
        let std = (1.0 / embed_dim.max(1) as f32).sqrt();
        let normal = Normal::new(0.0, std as f64).unwrap();

        let w_a = Array2::from_shape_fn((embed_dim, embed_dim), |_| normal.sample(&mut rng) as f32);
        let w_x = Array2::from_shape_fn((embed_dim, embed_dim), |_| normal.sample(&mut rng) as f32);
        let b_a = Array2::zeros((1, embed_dim));
        let b_x = Array2::zeros((1, embed_dim));

        // Initialize lambda so sigmoid(lambda) is moderately close to 1.
        // This biases a towards retention at init, similar to Hawk/Griffin.
        let lambda = Array2::from_shape_fn((1, embed_dim), |_| 2.0);

        Self {
            embed_dim,
            w_a,
            b_a,
            w_x,
            b_x,
            lambda,
            opt_w_a: Adam::new((embed_dim, embed_dim)),
            opt_b_a: Adam::new((1, embed_dim)),
            opt_w_x: Adam::new((embed_dim, embed_dim)),
            opt_b_x: Adam::new((1, embed_dim)),
            opt_lambda: Adam::new((1, embed_dim)),
            cached_input: None,
            cached_r: None,
            cached_i: None,
            cached_a: None,
            cached_hprev: None,
            streaming_state: None,
        }
    }

    #[cfg(test)]
    #[inline]
    fn compute_gates(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let t = input.nrows();
        let d = input.ncols();

        let mut r = Array2::<f32>::zeros((t, d));
        let mut i = Array2::<f32>::zeros((t, d));
        let mut a = Array2::<f32>::zeros((t, d));
        Self::compute_gates_into_parts(
            input,
            GatesParams {
                w_a: &self.w_a,
                b_a: &self.b_a,
                w_x: &self.w_x,
                b_x: &self.b_x,
                lambda: &self.lambda,
            },
            &mut r,
            &mut i,
            &mut a,
        );
        (r, i, a)
    }

    #[inline]
    fn compute_gates_into_parts(
        input: &ArrayBase<impl Data<Elem = f32>, Ix2>,
        p: GatesParams<'_>,
        r: &mut Array2<f32>,
        i: &mut Array2<f32>,
        a: &mut Array2<f32>,
    ) {
        let t = input.nrows();
        let d = input.ncols();

        if r.dim() != (t, d) {
            *r = Array2::<f32>::zeros((t, d));
        }
        if i.dim() != (t, d) {
            *i = Array2::<f32>::zeros((t, d));
        }
        if a.dim() != (t, d) {
            *a = Array2::<f32>::zeros((t, d));
        }

        ndarray::linalg::general_mat_mul(1.0, input, p.w_a, 0.0, r);
        if p.b_a.ncols() == d {
            for ti in 0..t {
                for j in 0..d {
                    r[[ti, j]] += p.b_a[[0, j]];
                }
            }
        }
        ndarray::linalg::general_mat_mul(1.0, input, p.w_x, 0.0, i);
        if p.b_x.ncols() == d {
            for ti in 0..t {
                for j in 0..d {
                    i[[ti, j]] += p.b_x[[0, j]];
                }
            }
        }

        let sigmoid = RichardsCurve::sigmoid(false);
        for ti in 0..t {
            for j in 0..d {
                r[[ti, j]] = sigmoid.forward_scalar_f32(r[[ti, j]]);
                i[[ti, j]] = sigmoid.forward_scalar_f32(i[[ti, j]]);
            }
        }

        let c: f32 = 8.0;
        let log_base_a: Array1<f32> = p.lambda.row(0).to_owned().mapv(|x| -softplus(-x));
        for ti in 0..t {
            for j in 0..d {
                let lt = (c * r[[ti, j]] * log_base_a[j]).clamp(-80.0, 0.0);
                a[[ti, j]] = crate::domain::pade::exp(lt);
            }
        }
    }

    #[cfg(test)]
    #[inline]
    fn compute_state(
        &self,
        input: &Array2<f32>,
        i: &Array2<f32>,
        a: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let t = input.nrows();
        let d = input.ncols();
        let mut hprev = Array2::<f32>::zeros((t, d));
        let mut h = Array2::<f32>::zeros((t, d));
        Self::compute_state_into(input, i, a, &mut hprev, &mut h);
        (hprev, h)
    }

    #[inline]
    fn compute_state_into(
        input: &ArrayBase<impl Data<Elem = f32>, Ix2>,
        i: &Array2<f32>,
        a: &Array2<f32>,
        hprev: &mut Array2<f32>,
        h: &mut Array2<f32>,
    ) {
        let t = input.nrows();
        let d = input.ncols();

        if hprev.dim() != (t, d) {
            *hprev = Array2::<f32>::zeros((t, d));
        }
        if h.dim() != (t, d) {
            *h = Array2::<f32>::zeros((t, d));
        }

        for ti in 0..t {
            for j in 0..d {
                let prev = if ti == 0 { 0.0 } else { h[[ti - 1, j]] };
                hprev[[ti, j]] = prev;

                let at = a[[ti, j]];
                let u = i[[ti, j]] * input[[ti, j]];
                let one_minus_a = 1.0 - at;
                let val = at * prev + one_minus_a * u;
                h[[ti, j]] = val;
            }
        }
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        // Ensure state is initialized
        if self.streaming_state.is_none() {
            self.streaming_state = Some(Array1::zeros(input.len()));
        }
        let h_prev = self.streaming_state.as_mut().unwrap();

        // Compute gates
        let r_pre = input.dot(&self.w_a) + &self.b_a.row(0);
        let i_pre = input.dot(&self.w_x) + &self.b_x.row(0);

        let sigmoid = RichardsCurve::sigmoid(false);
        let r = r_pre.mapv(|x| sigmoid.forward_scalar_f32(x));
        let i = i_pre.mapv(|x| sigmoid.forward_scalar_f32(x));

        let c: f32 = 8.0;
        let log_base_a: Array1<f32> = self.lambda.row(0).to_owned().mapv(|x| -softplus(-x));
        let a = (&r * &log_base_a * c).mapv(|x| crate::domain::pade::exp(x.clamp(-80.0, 0.0)));

        // Update state: h_t = a * h_{t-1} + (1 - a) * (i * x)
        let one_minus_a = 1.0 - &a;
        let u = &i * input;
        let h_new = &a * &*h_prev + &one_minus_a * &u;

        h_prev.assign(&h_new);
        h_new
    }

    #[inline]
    fn compute_forward_cached(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            self.cached_r = Some(Array2::<f32>::zeros((t, d)));
            self.cached_i = Some(Array2::<f32>::zeros((t, d)));
            self.cached_a = Some(Array2::<f32>::zeros((t, d)));
            self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
            return Array2::<f32>::zeros((t, d));
        }

        if self.cached_r.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_r = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_i.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_i = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_a.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_a = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_hprev.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
        }

        let r = self.cached_r.as_mut().expect("cached_r must exist");
        let i = self.cached_i.as_mut().expect("cached_i must exist");
        let a = self.cached_a.as_mut().expect("cached_a must exist");
        let hprev = self.cached_hprev.as_mut().expect("cached_hprev must exist");

        let p = GatesParams {
            w_a: &self.w_a,
            b_a: &self.b_a,
            w_x: &self.w_x,
            b_x: &self.b_x,
            lambda: &self.lambda,
        };
        Self::compute_gates_into_parts(input, p, r, i, a);
        let mut h = Array2::<f32>::zeros((t, d));
        Self::compute_state_into(input, i, a, hprev, &mut h);

        self.cached_input = Some(input.clone());
        h
    }

    #[inline]
    fn compute_forward_cached_view(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.to_owned());
            self.cached_r = Some(Array2::<f32>::zeros((t, d)));
            self.cached_i = Some(Array2::<f32>::zeros((t, d)));
            self.cached_a = Some(Array2::<f32>::zeros((t, d)));
            self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
            return Array2::<f32>::zeros((t, d));
        }

        if self.cached_r.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_r = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_i.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_i = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_a.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_a = Some(Array2::<f32>::zeros((t, d)));
        }
        if self.cached_hprev.as_ref().is_none_or(|x| x.dim() != (t, d)) {
            self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
        }

        let r = self.cached_r.as_mut().expect("cached_r must exist");
        let i = self.cached_i.as_mut().expect("cached_i must exist");
        let a = self.cached_a.as_mut().expect("cached_a must exist");
        let hprev = self.cached_hprev.as_mut().expect("cached_hprev must exist");

        let p = GatesParams {
            w_a: &self.w_a,
            b_a: &self.b_a,
            w_x: &self.w_x,
            b_x: &self.b_x,
            lambda: &self.lambda,
        };
        Self::compute_gates_into_parts(input, p, r, i, a);
        let mut h = Array2::<f32>::zeros((t, d));
        Self::compute_state_into(input, i, a, hprev, &mut h);

        self.cached_input = Some(input.to_owned());
        h
    }

    #[inline]
    fn forward_view(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.compute_forward_cached_view(input)
    }

    #[inline]
    fn compute_gates_and_state_from_cache_or_recompute<'a>(
        &'a self,
        input: &ArrayBase<impl Data<Elem = f32>, Ix2>,
    ) -> GatesAndState<'a> {
        let can_use = self
            .cached_input
            .as_ref()
            .is_some_and(|x| x.dim() == input.dim());
        let same_input = can_use
            && self
                .cached_input
                .as_ref()
                .is_some_and(|x| array2_bitwise_eq_base_f32(x, input));
        if same_input
            && let (Some(r), Some(i), Some(a), Some(hp)) = (
                self.cached_r.as_ref(),
                self.cached_i.as_ref(),
                self.cached_a.as_ref(),
                self.cached_hprev.as_ref(),
            )
        {
            return (
                Cow::Borrowed(r),
                Cow::Borrowed(i),
                Cow::Borrowed(a),
                Cow::Borrowed(hp),
            );
        }

        let t = input.nrows();
        let d = input.ncols();

        let mut r = Array2::<f32>::zeros((t, d));
        let mut i = Array2::<f32>::zeros((t, d));
        let mut a = Array2::<f32>::zeros((t, d));
        Self::compute_gates_into_parts(
            input,
            GatesParams {
                w_a: &self.w_a,
                b_a: &self.b_a,
                w_x: &self.w_x,
                b_x: &self.b_x,
                lambda: &self.lambda,
            },
            &mut r,
            &mut i,
            &mut a,
        );
        let mut hprev = Array2::<f32>::zeros((t, d));
        let mut h = Array2::<f32>::zeros((t, d));
        Self::compute_state_into(input, &i, &a, &mut hprev, &mut h);
        let _ = h;

        (
            Cow::Owned(r),
            Cow::Owned(i),
            Cow::Owned(a),
            Cow::Owned(hprev),
        )
    }

    fn compute_gradients_impl<Din: Data<Elem = f32>, Dout: Data<Elem = f32>>(
        &self,
        input: &ArrayBase<Din, Ix2>,
        output_grads: &ArrayBase<Dout, Ix2>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let (r, i, a, hprev) = self.compute_gates_and_state_from_cache_or_recompute(input);
        let r = r.as_ref();
        let i = i.as_ref();
        let a = a.as_ref();
        let hprev = hprev.as_ref();

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            return (Array2::zeros(input.raw_dim()), vec![]);
        }

        let c: f32 = 8.0;
        let log_base_a: Array1<f32> = self.lambda.row(0).to_owned().mapv(|x| -softplus(-x));
        let dlogsig_dlambda: Array1<f32> = {
            let sigmoid = RichardsCurve::sigmoid(false);
            self.lambda
                .row(0)
                .to_owned()
                .mapv(|x| sigmoid.forward_scalar_f32(-x))
        };

        let mut dh_next = Array1::<f32>::zeros(d);

        let mut dlogits_r = Array2::<f32>::zeros((t, d));
        let mut dlogits_i = Array2::<f32>::zeros((t, d));

        let mut dlog_base_a = Array1::<f32>::zeros(d);

        let mut d_x_from_u = Array2::<f32>::zeros((t, d));

        for ti in (0..t).rev() {
            for j in 0..d {
                let g = output_grads[[ti, j]];

                let dh = g + dh_next[j];

                let at = a[[ti, j]];
                let it = i[[ti, j]];
                let rt = r[[ti, j]];
                let xt = input[[ti, j]];
                let prev = hprev[[ti, j]];

                let u = it * xt;
                let one_minus_a = 1.0 - at;

                let du = dh * one_minus_a;
                d_x_from_u[[ti, j]] = du * it;
                let di = du * xt;

                let da = dh * (prev - u);

                dh_next[j] = dh * at;

                let k = c * rt * log_base_a[j];
                let active = (-80.0..=0.0).contains(&k);
                let dk = if active { da * at } else { 0.0 };

                let dr = dk * c * log_base_a[j];
                dlog_base_a[j] += dk * c * rt;

                let zr_grad = dr * rt * (1.0 - rt);
                dlogits_r[[ti, j]] = zr_grad;

                let zi_grad = di * it * (1.0 - it);
                dlogits_i[[ti, j]] = zi_grad;
            }
        }

        let mut d_lambda = Array2::<f32>::zeros((1, d));
        for j in 0..d {
            let dl = dlog_base_a[j] * dlogsig_dlambda[j];
            d_lambda[[0, j]] = dl;
        }

        let grad_w_a = input.t().dot(&dlogits_r);
        let grad_b_a = dlogits_r.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_w_x = input.t().dot(&dlogits_i);
        let grad_b_x = dlogits_i.sum_axis(Axis(0)).insert_axis(Axis(0));

        let dx_gate = dlogits_r.dot(&self.w_a.t()) + dlogits_i.dot(&self.w_x.t());
        let grad_input = dx_gate + d_x_from_u;

        (
            grad_input,
            vec![grad_w_a, grad_b_a, grad_w_x, grad_b_x, d_lambda],
        )
    }

    #[inline]
    fn compute_gradients_view(
        &self,
        input: &ArrayView2<f32>,
        output_grads: &ArrayView2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients_impl(input, output_grads)
    }

    fn opt_init_if_needed(&mut self) {
        let d = self.embed_dim.max(1);
        if self.opt_w_a.m().dim() != (d, d) {
            self.opt_w_a = Adam::new((d, d));
        }
        if self.opt_w_x.m().dim() != (d, d) {
            self.opt_w_x = Adam::new((d, d));
        }
        if self.opt_b_a.m().dim() != (1, d) {
            self.opt_b_a = Adam::new((1, d));
        }
        if self.opt_b_x.m().dim() != (1, d) {
            self.opt_b_x = Adam::new((1, d));
        }
        if self.opt_lambda.m().dim() != (1, d) {
            self.opt_lambda = Adam::new((1, d));
        }
    }
}

impl Layer for RgLru {
    fn layer_type(&self) -> &str {
        "RgLru"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.compute_forward_cached(input)
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
        self.w_a.len() + self.b_a.len() + self.w_x.len() + self.b_x.len() + self.lambda.len()
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        sumsq += self.w_a.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.b_a.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.w_x.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.b_x.iter().map(|&x| x * x).sum::<f32>();
        sumsq += self.lambda.iter().map(|&x| x * x).sum::<f32>();
        sumsq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients_impl(input, output_grads)
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        // Expected order: w_a, b_a, w_x, b_x, lambda
        if gradients.len() < 5 {
            return Ok(());
        }

        self.opt_init_if_needed();

        self.opt_w_a
            .step(&mut self.w_a, &gradients[0], learning_rate);
        self.opt_b_a
            .step(&mut self.b_a, &gradients[1], learning_rate);
        self.opt_w_x
            .step(&mut self.w_x, &gradients[2], learning_rate);
        self.opt_b_x
            .step(&mut self.b_x, &gradients[3], learning_rate);
        self.opt_lambda
            .step(&mut self.lambda, &gradients[4], learning_rate);

        Ok(())
    }

    fn zero_gradients(&mut self) {
        // No persistent gradient buffers; clear caches to reduce memory.
        self.cached_input = None;
        self.cached_r = None;
        self.cached_i = None;
        self.cached_a = None;
        self.cached_hprev = None;
    }
}

impl Layer for MoHRgLru {
    fn layer_type(&self) -> &str {
        "MoHRgLru"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 || self.num_heads == 0 || self.head_dim == 0 {
            self.clear_caches();
            self.cached_input = Some(input.clone());
            return Array2::<f32>::zeros((t, d));
        }

        // Cache input for backward.
        self.cached_input = Some(input.clone());

        let gd = self.moh.w_g.nrows().min(d);
        let gate_input = input.slice(s![.., 0..gd]);
        let eff = self.moh.forward_weights_view(&gate_input, None, None);
        self.cached_eff = Some(eff.clone());

        let mut out = Array2::<f32>::zeros((t, d));

        use rayon::prelude::*;
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

        // Compute per-head outputs and apply per-token scaling.
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

        // Cache head outputs for dEff computation in backward.
        self.cached_head_out = Some(head_outs);

        // MoH head-usage metrics.
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
            && self
                .cached_input
                .as_ref()
                .is_some_and(|x| array2_bitwise_eq_f32(x, input));

        // Prefer cached forward intermediates when available; fall back to recompute.
        let eff_local: Array2<f32>;
        let eff: &Array2<f32> = if can_use_cache
            && let Some(e) = self
                .cached_eff
                .as_ref()
                .filter(|e| e.dim() == (t, self.num_heads))
        {
            e
        } else {
            // Recompute eff weights without mutating gating caches.
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

        // dEff: per token/head scalar gradient from y = eff * y_h.
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

        // Per-head RG-LRU gradients.
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
            let (dx_h, pgrads_h) = if can_use_cache
                && let Some(x) = self.heads[h]
                    .cached_input
                    .as_ref()
                    .filter(|x| x.dim() == (t, self.head_dim))
            {
                self.heads[h].compute_gradients(x, &scaled_grads)
            } else {
                self.heads[h].compute_gradients_view(&x_view, &scaled_grads_view)
            };
            let mut gi_block = grad_input.slice_mut(s![.., c0..c1]);
            gi_block += &dx_h;
            grads.extend(pgrads_h);
        }

        // MoH gating gradients from dEff.
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

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        let per_head = 5usize;
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
    use super::*;

    #[test]
    fn test_rg_lru_forward_shape() {
        let mut layer = RgLru::new(16);
        let x = Array2::<f32>::from_elem((7, 16), 0.1);
        let y = layer.forward(&x);
        assert_eq!(y.dim(), (7, 16));
    }

    #[test]
    fn test_rg_lru_grad_shapes() {
        let mut layer = RgLru::new(8);
        let x = Array2::<f32>::from_elem((5, 8), 0.2);
        let y = layer.forward(&x);
        let grads = Array2::<f32>::from_elem(y.dim(), 0.1);

        let (dx, pgrads) = layer.compute_gradients(&x, &grads);
        assert_eq!(dx.dim(), x.dim());
        assert_eq!(pgrads.len(), 5);
        assert_eq!(pgrads[0].dim(), (8, 8));
        assert_eq!(pgrads[1].dim(), (1, 8));
        assert_eq!(pgrads[2].dim(), (8, 8));
        assert_eq!(pgrads[3].dim(), (1, 8));
        assert_eq!(pgrads[4].dim(), (1, 8));
    }

    #[test]
    fn test_rg_lru_gate_ranges() {
        let layer = RgLru::new(16);
        let x = Array2::<f32>::from_shape_fn((11, 16), |(t, d)| {
            ((t as f32) - 0.5) * (d as f32 + 1.0) * 0.01
        });
        let (r, i, a) = layer.compute_gates(&x);

        for v in r.iter() {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
        for v in i.iter() {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
        for v in a.iter() {
            assert!(*v > 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_rg_lru_recurrence_matches_state_computation() {
        let layer = RgLru::new(8);
        let x =
            Array2::<f32>::from_shape_fn((9, 8), |(t, d)| (t as f32 * 0.03) - (d as f32 * 0.01));
        let (r, i, a) = layer.compute_gates(&x);
        let (_hprev, h) = layer.compute_state(&x, &i, &a);
        let _ = r;

        for ti in 0..x.nrows() {
            for j in 0..x.ncols() {
                let prev = if ti == 0 { 0.0 } else { h[[ti - 1, j]] };
                let u = i[[ti, j]] * x[[ti, j]];
                let at = a[[ti, j]];
                let expected = at * prev + (1.0 - at) * u;
                assert!((h[[ti, j]] - expected).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn test_rg_lru_cached_vs_clone_gradients_match() {
        let mut layer = RgLru::new(8);
        let x = Array2::<f32>::from_shape_fn((6, 8), |(t, d)| {
            (t as f32 + 1.0) * (d as f32 + 2.0) * 0.001
        });
        let y = layer.forward(&x);
        let grads = Array2::<f32>::from_elem(y.dim(), 0.1);

        let cached = layer
            .cached_input
            .as_ref()
            .expect("cached_input must exist");
        let (dx_cached, pg_cached) = layer.compute_gradients(cached, &grads);

        let x_clone = x.clone();
        let (dx_clone, pg_clone) = layer.compute_gradients(&x_clone, &grads);

        assert_eq!(dx_cached, dx_clone);
        assert_eq!(pg_cached.len(), pg_clone.len());
        for (a, b) in pg_cached.iter().zip(pg_clone.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_moh_rg_lru_forward_shape() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHRgLru::new(16, 4, &cfg);
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
    fn test_moh_rg_lru_grad_shapes() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHRgLru::new(12, 3, &cfg);
        let x = Array2::<f32>::from_elem((5, 12), 0.2);
        let y = layer.forward(&x);
        let grads = Array2::<f32>::from_elem(y.dim(), 0.1);

        let (dx, pgrads) = layer.compute_gradients(&x, &grads);
        assert_eq!(dx.dim(), x.dim());
        // 3 heads * 5 grads + MoH grads (>=4)
        assert!(pgrads.len() >= 3 * 5 + 4);
    }

    #[test]
    fn test_moh_rg_lru_cache_not_reused_for_different_input() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHRgLru::new(12, 3, &cfg);
        let x1 = Array2::<f32>::from_shape_fn((5, 12), |(t, d)| {
            (t as f32 + 1.0) * (d as f32 + 1.0) * 0.01
        });
        let _ = layer.forward(&x1);

        let x2 = Array2::<f32>::from_shape_fn((5, 12), |(t, d)| {
            (t as f32 + 2.0) * (d as f32 + 3.0) * 0.02
        });
        let grads = Array2::<f32>::from_elem((5, 12), 0.1);

        let (dx_cached, pg_cached) = layer.compute_gradients(&x2, &grads);

        let mut layer_nocache = layer.clone();
        layer_nocache.clear_caches();
        let (dx_fresh, pg_fresh) = layer_nocache.compute_gradients(&x2, &grads);

        assert_eq!(dx_cached, dx_fresh);
        assert_eq!(pg_cached.len(), pg_fresh.len());
        for (a, b) in pg_cached.iter().zip(pg_fresh.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn moh_rg_lru_parameter_delta_within_1000() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let layer = MoHRgLru::new(64, 16, &cfg);
        let baseline: usize = layer.heads.iter().map(|h| h.parameters()).sum();
        let moh_total = layer.parameters();
        assert!(moh_total >= baseline);
        assert!(moh_total - baseline <= 1000);
    }
}
