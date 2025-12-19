use ndarray::{Array1, Array2, Axis, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize, Deserializer};

use crate::{
    adam::Adam,
    errors::Result,
    mixtures::{HeadSelectionStrategy, MoHGating},
    network::Layer,
    rng::get_rng,
};

#[inline]
fn sigmoid(x: f32) -> f32 {
    crate::richards::math::sigmoid_f32(x)
}

#[inline]
fn softplus(x: f32) -> f32 {
    crate::richards::math::softplus_f32(x)
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

    // Cheap cache key to avoid reusing stale caches on different inputs with same shape.
    #[serde(skip_serializing)]
    cached_input_sum: Option<f32>,
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
}

impl MoHRgLru {
    pub fn new(embed_dim: usize, num_heads: usize, head_selection: &HeadSelectionStrategy) -> Self {
        let mut nh = num_heads.max(1);
        if embed_dim == 0 || embed_dim % nh != 0 {
            nh = 1;
        }
        let head_dim = if nh > 0 { embed_dim / nh } else { embed_dim };

        let mut moh = MoHGating::new(embed_dim, nh);
        moh.set_head_selection_config(head_selection);

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
        }
    }

    #[inline]
    fn clear_caches(&mut self) {
        self.cached_input = None;
        self.cached_eff = None;
        self.cached_head_out = None;
        self.last_avg_active_heads = None;
        self.last_head_activity_vec = None;
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
            cached_input_sum: None,
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
            cached_input_sum: None,
        }
    }

    #[inline]
    fn input_cache_key(input: &Array2<f32>) -> f32 {
        // Deterministic (iteration order fixed) and cheap vs. matmul; enough to prevent
        // obvious wrong-cache reuse when shapes match but content differs.
        input.iter().copied().sum::<f32>()
    }

    #[inline]
    fn compute_gates(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // Returns (r, i, a)
        let t = input.nrows();
        let d = input.ncols();

        // z_r = X W_a + b_a; z_i = X W_x + b_x
        let mut z_r = input.dot(&self.w_a);
        let mut z_i = input.dot(&self.w_x);
        if self.b_a.ncols() == d {
            z_r += &self.b_a.broadcast((t, d)).unwrap();
        }
        if self.b_x.ncols() == d {
            z_i += &self.b_x.broadcast((t, d)).unwrap();
        }

        let r = z_r.mapv(sigmoid);
        let i = z_i.mapv(sigmoid);

        // log_base_a = log(sigmoid(lambda)) = -softplus(-lambda)
        // a_t = exp(c * r_t * log_base_a)
        let c: f32 = 8.0;
        let log_base_a: Array1<f32> = self
            .lambda
            .row(0)
            .to_owned()
            .mapv(|x| -softplus(-x));

        let mut a = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                let lt = (c * r[[ti, j]] * log_base_a[j]).max(-80.0).min(0.0);
                a[[ti, j]] = crate::richards::math::exp_f32(lt);
            }
        }

        (r, i, a)
    }

    #[inline]
    fn compute_state(&self, input: &Array2<f32>, i: &Array2<f32>, a: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        // Returns (hprev, h)
        let t = input.nrows();
        let d = input.ncols();
        let mut hprev = Array2::<f32>::zeros((t, d));
        let mut h = Array2::<f32>::zeros((t, d));

        for ti in 0..t {
            for j in 0..d {
                let prev = if ti == 0 { 0.0 } else { h[[ti - 1, j]] };
                hprev[[ti, j]] = prev;

                let at = a[[ti, j]];
                let u = i[[ti, j]] * input[[ti, j]];
                // Convex-combination form (stable): h_t = a_t * h_{t-1} + (1-a_t) * u_t
                let one_minus_a = 1.0 - at;
                let val = at * prev + one_minus_a * u;
                h[[ti, j]] = if val.is_finite() { val } else { 0.0 };
            }
        }

        (hprev, h)
    }

    #[inline]
    fn compute_forward_cached(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            self.cached_r = Some(Array2::zeros((t, d)));
            self.cached_i = Some(Array2::zeros((t, d)));
            self.cached_a = Some(Array2::zeros((t, d)));
            self.cached_hprev = Some(Array2::zeros((t, d)));
            self.cached_input_sum = Some(Self::input_cache_key(input));
            return Array2::zeros((t, d));
        }

        let (r, i, a) = self.compute_gates(input);
        let (hprev, h) = self.compute_state(input, &i, &a);

        self.cached_input = Some(input.clone());
        self.cached_r = Some(r);
        self.cached_i = Some(i);
        self.cached_a = Some(a);
        self.cached_hprev = Some(hprev);
        self.cached_input_sum = Some(Self::input_cache_key(input));

        h
    }

    #[inline]
    fn compute_gates_and_state_from_cache_or_recompute(
        &self,
        input: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        // Returns (r, i, a, hprev)
        let can_use = self.cached_input.as_ref().is_some_and(|x| x.dim() == input.dim());
        if can_use {
            let key = Self::input_cache_key(input);
            let key_ok = self
                .cached_input_sum
                .is_some_and(|k| (k - key).abs() <= (1e-5 * k.abs().max(1.0) + 1e-6));
            if let (Some(r), Some(i), Some(a), Some(hp)) = (
                self.cached_r.as_ref(),
                self.cached_i.as_ref(),
                self.cached_a.as_ref(),
                self.cached_hprev.as_ref(),
            ) {
                if key_ok {
                    return (r.clone(), i.clone(), a.clone(), hp.clone());
                }
            }
        }

        // Recompute forward pieces (without mutating self).
        let (r, i, a) = self.compute_gates(input);
        let (hprev, _h) = self.compute_state(input, &i, &a);
        (r, i, a, hprev)
    }

    fn opt_init_if_needed(&mut self) {
        let d = self.embed_dim.max(1);
        if self.opt_w_a.m.dim() != (d, d) {
            self.opt_w_a = Adam::new((d, d));
        }
        if self.opt_w_x.m.dim() != (d, d) {
            self.opt_w_x = Adam::new((d, d));
        }
        if self.opt_b_a.m.dim() != (1, d) {
            self.opt_b_a = Adam::new((1, d));
        }
        if self.opt_b_x.m.dim() != (1, d) {
            self.opt_b_x = Adam::new((1, d));
        }
        if self.opt_lambda.m.dim() != (1, d) {
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
        let (r, i, a, hprev) = self.compute_gates_and_state_from_cache_or_recompute(input);

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            return (Array2::zeros(input.raw_dim()), vec![]);
        }

        let c: f32 = 8.0;
        // log(sigmoid(lambda)) = -softplus(-lambda)
        let log_base_a: Array1<f32> = self.lambda.row(0).to_owned().mapv(|x| -softplus(-x));
        // d/dlambda log(sigmoid(lambda)) = sigmoid(-lambda) = 1 - sigmoid(lambda)
        let dlogsig_dlambda: Array1<f32> = self.lambda.row(0).to_owned().mapv(|x| sigmoid(-x));

        // Full BPTT through diagonal recurrence: dh_next carries gradients to h_{t-1}.
        let mut dh_next = Array1::<f32>::zeros(d);

        // Accumulate gate preactivation gradients per time step.
        let mut dlogits_r = Array2::<f32>::zeros((t, d));
        let mut dlogits_i = Array2::<f32>::zeros((t, d));

        // Accumulate dL/d(log_base_a) per channel across time.
        let mut dlog_base_a = Array1::<f32>::zeros(d);

        // Direct input gradient path via u = i ⊙ x.
        let mut d_x_from_u = Array2::<f32>::zeros((t, d));

        for ti in (0..t).rev() {
            for j in 0..d {
                let mut g = output_grads[[ti, j]];
                g = if g.is_finite() { g } else { 0.0 };

                // Add gradient arriving through time from h_{t} -> h_{t-1}.
                let mut dh = g + dh_next[j];
                dh = if dh.is_finite() { dh } else { 0.0 };

                let at = a[[ti, j]];
                let it = i[[ti, j]];
                let rt = r[[ti, j]];
                let xt = input[[ti, j]];
                let prev = hprev[[ti, j]];

                let u = it * xt;
                let one_minus_a = 1.0 - at;

                // dh/du
                let du = dh * one_minus_a;
                // direct dL/dx via u = i ⊙ x
                d_x_from_u[[ti, j]] = if (du * it).is_finite() { du * it } else { 0.0 };
                // dL/di
                let di = du * xt;

                // h_t = a_t * prev + (1-a_t) * u  => dL/da = dh * (prev - u)
                let da = dh * (prev - u);

                // Propagate to previous hidden state: h_t = a_t ⊙ h_{t-1} + ...
                dh_next[j] = if (dh * at).is_finite() { dh * at } else { 0.0 };

                // a_t = exp(clamp(k_t, -80, 0)), k_t = c * r_t * log_base_a
                let k = c * rt * log_base_a[j];
                let active = (k >= -80.0) && (k <= 0.0);
                let dk = if active { da * at } else { 0.0 };

                // k depends on r_t and log_base_a
                let dr = dk * c * log_base_a[j];
                dlog_base_a[j] += dk * c * rt;

                // r = sigmoid(z_r)
                let zr_grad = dr * rt * (1.0 - rt);
                dlogits_r[[ti, j]] = if zr_grad.is_finite() { zr_grad } else { 0.0 };

                // i = sigmoid(z_i)
                let zi_grad = di * it * (1.0 - it);
                dlogits_i[[ti, j]] = if zi_grad.is_finite() { zi_grad } else { 0.0 };
            }
        }

        // dL/dlambda = dL/dlog_base_a * d/dlambda log(sigmoid(lambda))
        let mut d_lambda = Array2::<f32>::zeros((1, d));
        for j in 0..d {
            let dl = dlog_base_a[j] * dlogsig_dlambda[j];
            d_lambda[[0, j]] = if dl.is_finite() { dl } else { 0.0 };
        }

        // Weight/bias grads: X^T dot dlogits
        let grad_w_a = input.t().dot(&dlogits_r);
        let grad_b_a = dlogits_r.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_w_x = input.t().dot(&dlogits_i);
        let grad_b_x = dlogits_i.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Input grads through gate preactivations
        let dx_gate = dlogits_r.dot(&self.w_a.t()) + dlogits_i.dot(&self.w_x.t());
        let grad_input = dx_gate + d_x_from_u;

        (
            grad_input,
            vec![grad_w_a, grad_b_a, grad_w_x, grad_b_x, d_lambda],
        )
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        // Expected order: w_a, b_a, w_x, b_x, lambda
        if gradients.len() < 5 {
            return Ok(());
        }

        self.opt_init_if_needed();

        self.opt_w_a.step(&mut self.w_a, &gradients[0], learning_rate);
        self.opt_b_a.step(&mut self.b_a, &gradients[1], learning_rate);
        self.opt_w_x.step(&mut self.w_x, &gradients[2], learning_rate);
        self.opt_b_x.step(&mut self.b_x, &gradients[3], learning_rate);
        self.opt_lambda.step(&mut self.lambda, &gradients[4], learning_rate);

        Ok(())
    }

    fn zero_gradients(&mut self) {
        // No persistent gradient buffers; clear caches to reduce memory.
        self.cached_input = None;
        self.cached_r = None;
        self.cached_i = None;
        self.cached_a = None;
        self.cached_hprev = None;
        self.cached_input_sum = None;
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

        // Compute MoH effective weights on the full embedding.
        let eff = self.moh.forward_weights(input, None, None);
        self.cached_eff = Some(eff.clone());

        let mut out = Array2::<f32>::zeros((t, d));
        let mut head_outs: Vec<Array2<f32>> = Vec::with_capacity(self.num_heads);

        // Compute per-head outputs and apply per-token scaling.
        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let x_h = input.slice(s![.., c0..c1]).to_owned();
            let y_h = self.heads[h].forward(&x_h);
            for i in 0..t {
                let w = eff[[i, h]];
                for j in 0..self.head_dim {
                    out[[i, c0 + j]] = y_h[[i, j]] * w;
                }
            }
            head_outs.push(y_h);
        }

        // Cache head outputs for dEff computation in backward.
        self.cached_head_out = Some(head_outs);

        // MoH head-usage metrics.
        let avg = self.moh.head_selection_config.gating.get_avg_active_components();
        self.last_avg_active_heads = if avg.is_finite() { Some(avg) } else { Some(0.0) };

        // Provide a per-head activity vector for downstream MoE conditioning.
        let mut hv = Vec::with_capacity(self.num_heads);
        for h in 0..self.num_heads {
            let mean = eff.column(h).iter().map(|&x| x.max(0.0)).sum::<f32>() / (t.max(1) as f32);
            hv.push(if mean.is_finite() { mean } else { 0.0 });
        }
        self.last_head_activity_vec = Some(hv);

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
            moh_params += pred.weights1.len() + pred.bias1.len() + pred.weights2.len() + pred.bias2.len();
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

        // Prefer cached forward intermediates when available; fall back to recompute.
        let eff_local: Array2<f32>;
        let eff: &Array2<f32> = if let Some(e) = self
            .cached_eff
            .as_ref()
            .filter(|e| e.dim() == (t, self.num_heads))
        {
            e
        } else {
            // Recompute eff weights without mutating gating caches.
            let mut moh_tmp = self.moh.clone();
            eff_local = moh_tmp.forward_weights(input, None, None);
            &eff_local
        };

        let head_outputs_local: Vec<Array2<f32>>;
        let head_outputs: &Vec<Array2<f32>> = if let Some(v) = self.cached_head_out.as_ref() {
            let ok_len = v.len() == self.num_heads;
            let ok_dims = ok_len
                && v.iter()
                    .all(|y| y.dim() == (t, self.head_dim));
            if ok_dims { v } else { 
                head_outputs_local = (0..self.num_heads)
                    .map(|h| {
                        let c0 = h * self.head_dim;
                        let c1 = c0 + self.head_dim;
                        let x_h = input.slice(s![.., c0..c1]).to_owned();
                        let mut head = self.heads[h].clone();
                        head.forward(&x_h)
                    })
                    .collect();
                &head_outputs_local
            }
        } else {
            head_outputs_local = (0..self.num_heads)
                .map(|h| {
                    let c0 = h * self.head_dim;
                    let c1 = c0 + self.head_dim;
                    let x_h = input.slice(s![.., c0..c1]).to_owned();
                    let mut head = self.heads[h].clone();
                    head.forward(&x_h)
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
                eff_grads[[i, h]] = if acc.is_finite() { acc } else { 0.0 };
            }
        }

        // Per-head RG-LRU gradients.
        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
        let mut grads: Vec<Array2<f32>> = Vec::new();

        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let x_h = input.slice(s![.., c0..c1]).to_owned();
            let mut scaled_grads = Array2::<f32>::zeros((t, self.head_dim));
            for i in 0..t {
                let w = eff[[i, h]];
                for j in 0..self.head_dim {
                    scaled_grads[[i, j]] = output_grads[[i, c0 + j]] * w;
                }
            }

            let (dx_h, pgrads_h) = self.heads[h].compute_gradients(&x_h, &scaled_grads);
            for i in 0..t {
                for j in 0..self.head_dim {
                    grad_input[[i, c0 + j]] += dx_h[[i, j]];
                }
            }
            grads.extend(pgrads_h);
        }

        // MoH gating gradients from dEff.
        let (dx_moh, moh_grads) = {
            let mut moh_local = self.moh.clone();
            moh_local.compute_gradients_from_eff(input, &eff_grads)
        };
        grad_input += &dx_moh;
        grads.extend(moh_grads);

        (grad_input, grads)
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        let per_head = 5usize;
        let needed_heads = self.num_heads * per_head;
        if gradients.len() < needed_heads {
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
    fn test_moh_rg_lru_forward_shape() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHRgLru::new(16, 4, &cfg);
        let x = Array2::<f32>::from_elem((7, 16), 0.1);
        let y = layer.forward(&x);
        assert_eq!(y.dim(), (7, 16));
        assert!(layer.last_avg_active_heads.is_some());
        assert!(layer.last_head_activity_vec.as_ref().is_some_and(|v| v.len() == 4));
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
}
