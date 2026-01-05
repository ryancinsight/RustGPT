use ndarray::{Array1, Array2, Axis};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    adam::Adam,
    errors::Result,
    network::Layer,
    richards::{RichardsActivation, RichardsCurve, RichardsGate},
    rng::get_rng,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MambaCachedKind {
    Mamba1,
    Mamba2,
    Mamba2Parallel,  // Enhanced with parallel scan
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum AMatrixType {
    Diagonal,        // Original: diagonal A matrix
    BlockDiagonal,   // Enhanced: block-diagonal A matrix
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum ScanMethod {
    Sequential,      // Original sequential scan
    Parallel,        // Parallel scan using associative property
    MemoryEfficient, // Memory-efficient scan for long sequences
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ScanConfig {
    method: ScanMethod,
    block_size: Option<usize>,  // For block-diagonal A
    chunk_size: Option<usize>,  // For memory-efficient scan
}

impl Default for ScanConfig {
    fn default() -> Self {
        Self {
            method: ScanMethod::Sequential,
            block_size: None,
            chunk_size: None,
        }
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    crate::soft::softplus(x)
}

fn mamba_default_gate() -> RichardsGate {
    let mut gate = RichardsGate::new();
    // Avoid random default temperature when backfilling old checkpoints.
    gate.set_temperature(1.0);
    gate
}

fn mamba_default_tanh_curve() -> RichardsCurve {
    RichardsCurve::tanh(true)
}

fn mamba_default_act() -> RichardsActivation {
    // Fully learnable x * Richards(x) activation so it can adapt toward swish/gompertz/etc.
    RichardsActivation::new_fully_learnable()
}

/// A more complete Mamba-style selective SSM layer.
///
/// Implements the core ingredients used in Mamba v1 (reference / CPU-friendly):
/// - in-projection to (u, gate)
/// - depthwise causal convolution on u
/// - input-dependent dt
/// - multi-dimensional selective SSM state (N > 1) with ZOH discretization
/// - selective scan
/// - output projection
///
/// Shape: (T × D) → (T × D)
#[derive(Serialize, Debug, Clone)]
pub struct Mamba {
    pub embed_dim: usize,
    pub conv_kernel: usize,

    // Enhanced configuration
    a_matrix_type: AMatrixType,
    scan_config: ScanConfig,

    // in-projection (u_pre, gate_logits)
    pub w_in: Array2<f32>, // [D, 2D]
    pub b_in: Array2<f32>, // [1, 2D]

    // dt, B, C projections
    pub w_dt: Array2<f32>,
    pub b_dt: Array2<f32>,
    pub w_b: Array2<f32>,
    pub b_b: Array2<f32>,
    pub w_c: Array2<f32>,
    pub b_c: Array2<f32>,

    // diagonal A (negative), represented by a_log with A = -softplus(a_log)
    pub a_log: Array2<f32>, // [1, D]

    // skip connection coefficient D (per-channel)
    pub d_skip: Array2<f32>, // [1, D]

    // depthwise conv (on u_act)
    pub conv_w: Array2<f32>, // [K, D]
    pub conv_b: Array2<f32>, // [1, D]

    // out projection
    pub w_out: Array2<f32>, // [D, D]
    pub b_out: Array2<f32>, // [1, D]

    // Learnable/adaptive nonlinearities (Richards-native).
    #[serde(default = "mamba_default_act", alias = "richards_silu")]
    pub richards_act: RichardsActivation,
    #[serde(default = "mamba_default_gate")]
    pub richards_gate: RichardsGate,
    #[serde(default = "mamba_default_tanh_curve")]
    pub richards_tanh: RichardsCurve,

    #[serde(skip_serializing)]
    opt_w_in: Adam,
    #[serde(skip_serializing)]
    opt_b_in: Adam,
    #[serde(skip_serializing)]
    opt_w_dt: Adam,
    #[serde(skip_serializing)]
    opt_b_dt: Adam,
    #[serde(skip_serializing)]
    opt_w_b: Adam,
    #[serde(skip_serializing)]
    opt_b_b: Adam,
    #[serde(skip_serializing)]
    opt_w_c: Adam,
    #[serde(skip_serializing)]
    opt_b_c: Adam,
    #[serde(skip_serializing)]
    opt_a_log: Adam,
    #[serde(skip_serializing)]
    opt_d_skip: Adam,
    #[serde(skip_serializing)]
    opt_conv_w: Adam,
    #[serde(skip_serializing)]
    opt_conv_b: Adam,
    #[serde(skip_serializing)]
    opt_w_out: Adam,
    #[serde(skip_serializing)]
    opt_b_out: Adam,

    // caches
    #[serde(skip_serializing)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_u_pre: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_u_act: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_gate: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_gate_logits: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_dt_logits: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_dt: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_b_logits: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_b_t: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_c_logits: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_c_t: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_a_logits_state: Option<Array2<f32>>, // [D, N]
    #[serde(skip_serializing)]
    cached_a_scale_state: Option<Array2<f32>>, // [D, N]
    #[serde(skip_serializing)]
    cached_a: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_u_conv: Option<Array2<f32>>,
    #[serde(skip_serializing)]
    cached_state_prev: Option<Array2<f32>>, // state_{t-1}
    #[serde(skip_serializing)]
    cached_state: Option<Array2<f32>>, // state_t
    #[serde(skip_serializing)]
    cached_z: Option<Array2<f32>>, // z_t = c*state + d*u_conv
    #[serde(skip_serializing)]
    cached_y_pre: Option<Array2<f32>>, // y_pre = gate * z
    #[serde(skip_serializing)]
    cached_out_pre: Option<Array2<f32>>, // before out projection

    // deterministic (non-parameter) projections used to map D -> N without adding parameters.
    // These are cached to avoid rebuilding every forward/backward.
    #[serde(skip_serializing)]
    cached_state_dim: usize,
    #[serde(skip_serializing)]
    cached_proj_state: Option<Array2<f32>>, // [D, N]
    #[serde(skip_serializing)]
    cached_proj_a: Option<Array2<f32>>, // [D, N]

    // which forward path populated the caches
    #[serde(skip_serializing)]
    cached_kind: MambaCachedKind,

    // Mamba-2 / SSD caches
    #[serde(skip_serializing)]
    cached_head_offsets: Option<Vec<usize>>, // len = H+1, offsets into channels
    #[serde(skip_serializing)]
    cached_dt_head: Option<Array2<f32>>, // [T, H]
    #[serde(skip_serializing)]
    cached_a_head: Option<Array2<f32>>, // [T, H]
    #[serde(skip_serializing)]
    cached_a_scale_head: Option<Array2<f32>>, // [1, H]
}

impl<'de> Deserialize<'de> for Mamba {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SerdeData {
            embed_dim: usize,
            conv_kernel: usize,
            #[serde(default)]
            a_matrix_type: Option<AMatrixType>,
            #[serde(default)]
            scan_config: Option<ScanConfig>,
            w_in: Array2<f32>,
            b_in: Array2<f32>,
            w_dt: Array2<f32>,
            b_dt: Array2<f32>,
            w_b: Array2<f32>,
            b_b: Array2<f32>,
            w_c: Array2<f32>,
            b_c: Array2<f32>,
            a_log: Array2<f32>,
            d_skip: Array2<f32>,
            conv_w: Array2<f32>,
            conv_b: Array2<f32>,
            w_out: Array2<f32>,
            b_out: Array2<f32>,

            // Nonlinearities added later; keep optional for backward compatibility.
            #[serde(default, alias = "richards_silu")]
            richards_act: Option<RichardsActivation>,
            #[serde(default)]
            richards_gate: Option<RichardsGate>,
            #[serde(default)]
            richards_tanh: Option<RichardsCurve>,
        }

        let data = SerdeData::deserialize(deserializer)?;
        let d = data.embed_dim.max(1);
        let k = data.conv_kernel.max(1);

        Ok(Self {
            embed_dim: data.embed_dim,
            conv_kernel: k,
            a_matrix_type: data.a_matrix_type.unwrap_or(AMatrixType::Diagonal),
            scan_config: data.scan_config.unwrap_or_default(),
            w_in: data.w_in,
            b_in: data.b_in,
            w_dt: data.w_dt,
            b_dt: data.b_dt,
            w_b: data.w_b,
            b_b: data.b_b,
            w_c: data.w_c,
            b_c: data.b_c,
            a_log: data.a_log,
            d_skip: data.d_skip,
            conv_w: data.conv_w,
            conv_b: data.conv_b,
            w_out: data.w_out,
            b_out: data.b_out,
            richards_act: data.richards_act.unwrap_or_else(mamba_default_act),
            richards_gate: data.richards_gate.unwrap_or_else(mamba_default_gate),
            richards_tanh: data
                .richards_tanh
                .unwrap_or_else(mamba_default_tanh_curve),
            opt_w_in: Adam::new((d, 2 * d)),
            opt_b_in: Adam::new((1, 2 * d)),
            opt_w_dt: Adam::new((d, d)),
            opt_b_dt: Adam::new((1, d)),
            opt_w_b: Adam::new((d, d)),
            opt_b_b: Adam::new((1, d)),
            opt_w_c: Adam::new((d, d)),
            opt_b_c: Adam::new((1, d)),
            opt_a_log: Adam::new((1, d)),
            opt_d_skip: Adam::new((1, d)),
            opt_conv_w: Adam::new((k, d)),
            opt_conv_b: Adam::new((1, d)),
            opt_w_out: Adam::new((d, d)),
            opt_b_out: Adam::new((1, d)),
            cached_input: None,
            cached_u_pre: None,
            cached_u_act: None,
            cached_gate: None,
            cached_gate_logits: None,
            cached_dt_logits: None,
            cached_dt: None,
            cached_b_logits: None,
            cached_b_t: None,
            cached_c_logits: None,
            cached_c_t: None,
            cached_a_logits_state: None,
            cached_a_scale_state: None,
            cached_a: None,
            cached_u_conv: None,
            cached_state_prev: None,
            cached_state: None,
            cached_z: None,
            cached_y_pre: None,
            cached_out_pre: None,
            cached_state_dim: 0,
            cached_proj_state: None,
            cached_proj_a: None,
            cached_kind: MambaCachedKind::Mamba1,
            cached_head_offsets: None,
            cached_dt_head: None,
            cached_a_head: None,
            cached_a_scale_head: None,
        })
    }
}

impl Mamba {
    #[inline]
    fn desired_state_dim(embed_dim: usize) -> usize {
        // Canonical Mamba typically uses a small state dim (e.g., 16). We cap for CPU cost.
        embed_dim.min(16).max(1)
    }

    #[inline]
    fn desired_state_dim_mamba2(embed_dim: usize) -> usize {
        // Mamba-2 / SSD typically benefits from larger state sizes.
        embed_dim.min(32).max(16)
    }

    #[inline]
    fn head_dim_mamba2(embed_dim: usize) -> usize {
        // Typical SSD head dimension is ~64.
        embed_dim.min(64).max(1)
    }

    #[inline]
    fn make_head_offsets(d: usize, head_dim: usize) -> Vec<usize> {
        if d == 0 {
            return vec![0];
        }
        let hd = head_dim.max(1);
        let num_heads = (d + hd - 1) / hd;
        let mut offs = Vec::with_capacity(num_heads + 1);
        offs.push(0);
        for h in 0..num_heads {
            let end = ((h + 1) * hd).min(d);
            offs.push(end);
            if end == d {
                break;
            }
        }
        offs
    }

    fn ensure_projections_mamba2(&mut self, d: usize) {
        let n = Self::desired_state_dim_mamba2(d);
        if self.cached_state_dim == n
            && self
                .cached_proj_state
                .as_ref()
                .is_some_and(|p| p.nrows() == d && p.ncols() == n)
            && self
                .cached_proj_a
                .as_ref()
                .is_some_and(|p| p.nrows() == d && p.ncols() == n)
        {
            return;
        }

        fn make_proj(d: usize, n: usize, freq: f32, phase: f32) -> Array2<f32> {
            let mut p = Array2::<f32>::zeros((d, n));
            for j in 0..d {
                let jf = (j as f32) + 1.0;
                for k in 0..n {
                    let kf = (k as f32) + 1.0;
                    p[[j, k]] = (freq * jf * kf + phase).sin();
                }
            }
            for k in 0..n {
                let mut norm2 = 0.0f32;
                for j in 0..d {
                    let v = p[[j, k]];
                    norm2 += v * v;
                }
                let inv = if norm2 > 1e-12 { 1.0 / norm2.sqrt() } else { 1.0 };
                for j in 0..d {
                    p[[j, k]] *= inv;
                }
            }
            p
        }

        self.cached_state_dim = n;
        self.cached_proj_state = Some(make_proj(d, n, 0.071, 0.0));
        self.cached_proj_a = Some(make_proj(d, n, 0.113, 1.234));
    }

    fn ensure_projections(&mut self, d: usize) {
        let n = Self::desired_state_dim(d);
        if self.cached_state_dim == n
            && self
                .cached_proj_state
                .as_ref()
                .is_some_and(|p| p.nrows() == d && p.ncols() == n)
            && self
                .cached_proj_a
                .as_ref()
                .is_some_and(|p| p.nrows() == d && p.ncols() == n)
        {
            return;
        }

        fn make_proj(d: usize, n: usize, freq: f32, phase: f32) -> Array2<f32> {
            let mut p = Array2::<f32>::zeros((d, n));
            for j in 0..d {
                let jf = (j as f32) + 1.0;
                for k in 0..n {
                    let kf = (k as f32) + 1.0;
                    // Deterministic, roughly zero-mean.
                    p[[j, k]] = (freq * jf * kf + phase).sin();
                }
            }
            // Normalize columns to unit norm to keep scales stable.
            for k in 0..n {
                let mut norm2 = 0.0f32;
                for j in 0..d {
                    let v = p[[j, k]];
                    norm2 += v * v;
                }
                let inv = if norm2 > 1e-12 { 1.0 / norm2.sqrt() } else { 1.0 };
                for j in 0..d {
                    p[[j, k]] *= inv;
                }
            }
            p
        }

        self.cached_state_dim = n;
        self.cached_proj_state = Some(make_proj(d, n, 0.071, 0.0));
        self.cached_proj_a = Some(make_proj(d, n, 0.113, 1.234));
    }

    pub fn new(embed_dim: usize) -> Self {
        Self::new_with_config(embed_dim, 4, MambaConfig::default())
    }

    pub fn new_with_kernel(embed_dim: usize, conv_kernel: usize) -> Self {
        Self::new_with_config(embed_dim, conv_kernel, MambaConfig::default())
    }

    /// Create Mamba layer with enhanced configuration
    pub fn new_with_config(embed_dim: usize, conv_kernel: usize, config: MambaConfig) -> Self {
        let d = embed_dim.max(1);
        let k = conv_kernel.max(1);

        let mut rng = get_rng();
        let std = (1.0 / d as f32).sqrt();
        let normal = Normal::new(0.0, std as f64).unwrap();

        let w_in = Array2::from_shape_fn((d, 2 * d), |_| normal.sample(&mut rng) as f32);
        let b_in = Array2::zeros((1, 2 * d));

        let w_dt = Array2::from_shape_fn((d, d), |_| normal.sample(&mut rng) as f32);
        let b_dt = Array2::zeros((1, d));
        let w_b = Array2::from_shape_fn((d, d), |_| normal.sample(&mut rng) as f32);
        let b_b = Array2::zeros((1, d));
        let w_c = Array2::from_shape_fn((d, d), |_| normal.sample(&mut rng) as f32);
        let b_c = Array2::zeros((1, d));

        // Enhanced A matrix initialization based on configuration
        let a_log = match config.a_matrix_type {
            AMatrixType::Diagonal => {
                // Original initialization
                Array2::from_shape_fn((1, d), |_| 1.0)
            }
            AMatrixType::BlockDiagonal => {
                // Enhanced initialization with block structure
                let block_size = config.scan_config.block_size.unwrap_or(4);
                Array2::from_shape_fn((1, d), |(_, j)| {
                    let block = j / block_size;
                    // Vary by block for better expressivity
                    1.0 + 0.1 * (block as f32).sin()
                })
            }
        };

        let d_skip = Array2::zeros((1, d));

        let conv_w = Array2::from_shape_fn((k, d), |_| (normal.sample(&mut rng) as f32) * 0.1);
        let conv_b = Array2::zeros((1, d));

        let w_out = Array2::from_shape_fn((d, d), |_| normal.sample(&mut rng) as f32);
        let b_out = Array2::zeros((1, d));

        Self {
            embed_dim,
            conv_kernel: k,
            a_matrix_type: config.a_matrix_type,
            scan_config: config.scan_config,
            w_in,
            b_in,
            w_dt,
            b_dt,
            w_b,
            b_b,
            w_c,
            b_c,
            a_log,
            d_skip,
            conv_w,
            conv_b,
            w_out,
            b_out,

            richards_act: mamba_default_act(),
            richards_gate: mamba_default_gate(),
            richards_tanh: mamba_default_tanh_curve(),

            opt_w_in: Adam::new((d, 2 * d)),
            opt_b_in: Adam::new((1, 2 * d)),
            opt_w_dt: Adam::new((d, d)),
            opt_b_dt: Adam::new((1, d)),
            opt_w_b: Adam::new((d, d)),
            opt_b_b: Adam::new((1, d)),
            opt_w_c: Adam::new((d, d)),
            opt_b_c: Adam::new((1, d)),
            opt_a_log: Adam::new((1, d)),
            opt_d_skip: Adam::new((1, d)),
            opt_conv_w: Adam::new((k, d)),
            opt_conv_b: Adam::new((1, d)),
            opt_w_out: Adam::new((d, d)),
            opt_b_out: Adam::new((1, d)),
            cached_input: None,
            cached_u_pre: None,
            cached_u_act: None,
            cached_gate: None,
            cached_gate_logits: None,
            cached_dt_logits: None,
            cached_dt: None,
            cached_b_logits: None,
            cached_b_t: None,
            cached_c_logits: None,
            cached_c_t: None,
            cached_a_logits_state: None,
            cached_a_scale_state: None,
            cached_a: None,
            cached_u_conv: None,
            cached_state_prev: None,
            cached_state: None,
            cached_z: None,
            cached_y_pre: None,
            cached_out_pre: None,
            cached_state_dim: 0,
            cached_proj_state: None,
            cached_proj_a: None,
            cached_kind: MambaCachedKind::Mamba1,
            cached_head_offsets: None,
            cached_dt_head: None,
            cached_a_head: None,
            cached_a_scale_head: None,
        }
    }

    #[inline]
    fn depthwise_causal_conv(&self, u: &Array2<f32>) -> Array2<f32> {
        let t = u.nrows();
        let d = u.ncols();
        let k = self.conv_kernel;
        if t == 0 || d == 0 {
            return Array2::zeros((t, d));
        }

        let mut out = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            let start = if ti + 1 >= k { ti + 1 - k } else { 0 };
            let mut kk = 0usize;
            for tj in start..=ti {
                let wrow = kk;
                for j in 0..d {
                    out[[ti, j]] += self.conv_w[[wrow, j]] * u[[tj, j]];
                }
                kk += 1;
            }
        }
        for ti in 0..t {
            for j in 0..d {
                out[[ti, j]] += self.conv_b[[0, j]];
            }
        }
        out
    }

    fn forward_cached(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_kind = MambaCachedKind::Mamba1;
        self.cached_head_offsets = None;
        self.cached_dt_head = None;
        self.cached_a_head = None;
        self.cached_a_scale_head = None;

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            return Array2::zeros((t, d));
        }

        self.ensure_projections(d);
        let n = self.cached_state_dim;
        let proj_state = self
            .cached_proj_state
            .as_ref()
            .expect("proj_state must exist");
        let proj_a = self.cached_proj_a.as_ref().expect("proj_a must exist");

        let in2 = input.dot(&self.w_in) + self.b_in.broadcast((t, 2 * d)).unwrap();
        let u_pre = in2.slice(ndarray::s![.., 0..d]).to_owned();
        let gate_logits = in2.slice(ndarray::s![.., d..2 * d]).to_owned();

        let u_act = self.richards_act.forward_matrix_f32(&u_pre);
        let gate = self.richards_gate.forward_const(&gate_logits);

        // Canonical-style dt: learned via the in-projection stream (u_pre), not via an extra D×D projection.
        // This keeps parameter count unchanged while allowing per-token/per-channel dt.
        let dt_logits = u_pre.clone();
        let dt = dt_logits.mapv(|x| softplus(x) + 1e-6);

        // Project input into a smaller (N) space for B/C, without adding parameters.
        let b_full = input.dot(&self.w_b) + self.b_b.broadcast((t, d)).unwrap();
        let b_logits = b_full.dot(proj_state);
        let mut b_t = Array2::<f32>::zeros(b_logits.raw_dim());
        self.richards_tanh.forward_matrix_f32_into(&b_logits, &mut b_t);

        let c_full = input.dot(&self.w_c) + self.b_c.broadcast((t, d)).unwrap();
        let c_logits = c_full.dot(proj_state);
        let mut c_t = Array2::<f32>::zeros(c_logits.raw_dim());
        self.richards_tanh.forward_matrix_f32_into(&c_logits, &mut c_t);

        // Build A logits/state scales using w_out (and biases) mapped into (D×N) via a fixed projection.
        // A_scale is positive; we use ZOH discretization with a = exp(-dt * A_scale).
        let mut a_logits_state = self.w_out.dot(proj_a); // [D, N]
        let bias_d = self.a_log.row(0).to_owned() + self.b_out.row(0).to_owned();
        for j in 0..d {
            let bj = bias_d[j];
            for k in 0..n {
                a_logits_state[[j, k]] += bj;
            }
        }
        let a_scale_state = a_logits_state.mapv(|x| softplus(x) + 1e-6);

        let u_conv = self.depthwise_causal_conv(&u_act);

        let mut state_prev = Array2::<f32>::zeros((t, d * n));
        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let mut y_pre = Array2::<f32>::zeros((t, d));


        let d_skip_row = self.d_skip.row(0).to_owned();
        let mut s = Array1::<f32>::zeros(d * n);

        for ti in 0..t {
            for j in 0..d {
                let dtj = dt[[ti, j]];
                let uj = u_conv[[ti, j]];
                let mut zj = d_skip_row[j] * uj;

                for k in 0..n {
                    let idx = j * n + k;
                    let prev = s[idx];
                    state_prev[[ti, idx]] = prev;

                    let a_scale = a_scale_state[[j, k]];
                    let aj = crate::pade::exp(-dtj * a_scale).clamp(0.0, 1.0);
                    let inp = b_t[[ti, k]] * uj;
                    let kk = (1.0 - aj) / a_scale;
                    let sj = aj * prev + kk * inp;

                    s[idx] = sj;
                    state[[ti, idx]] = sj;
                    zj += c_t[[ti, k]] * sj;
                }

                z[[ti, j]] = zj;
                y_pre[[ti, j]] = gate[[ti, j]] * zj;
            }
        }

        // Output projection uses the existing (w_dt, b_dt) tensors to keep parameter count unchanged.
        let out_pre = y_pre.dot(&self.w_dt) + self.b_dt.broadcast((t, d)).unwrap();

        self.cached_input = Some(input.clone());
        self.cached_u_pre = Some(u_pre);
        self.cached_u_act = Some(u_act);
        self.cached_gate = Some(gate);
        self.cached_gate_logits = Some(gate_logits);
        self.cached_dt_logits = Some(dt_logits);
        self.cached_dt = Some(dt);
        self.cached_b_logits = Some(b_logits);
        self.cached_b_t = Some(b_t);
        self.cached_c_logits = Some(c_logits);
        self.cached_c_t = Some(c_t);
        self.cached_a_logits_state = Some(a_logits_state);
        self.cached_a_scale_state = Some(a_scale_state);
        self.cached_a = None;
        self.cached_u_conv = Some(u_conv);
        self.cached_state_prev = Some(state_prev);
        self.cached_state = Some(state);
        self.cached_z = Some(z);
        self.cached_y_pre = Some(y_pre);
        self.cached_out_pre = Some(out_pre.clone());

        out_pre
    }

    /// Parallel selective scan using associative property
    /// Based on Mamba-2 optimizations for better hardware utilization
    fn parallel_selective_scan(
        &self,
        dt: &Array2<f32>,           // [T, D]
        a_scale_state: &Array2<f32>, // [D, N]
        b_t: &Array2<f32>,          // [T, N]
        c_t: &Array2<f32>,          // [T, N]
        u_conv: &Array2<f32>,       // [T, D]
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let t = dt.nrows();
        let d = dt.ncols();
        let n = b_t.ncols();
        
        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let y_pre = Array2::<f32>::zeros((t, d));
        
        // Parallel scan using associative property
        // This is a simplified version - full implementation would use parallel prefix sum
        for j in 0..d {
            let dt_col = dt.column(j);
            let u_conv_col = u_conv.column(j);
            
            for k in 0..n {
                let idx = j * n + k;
                let a_scale = a_scale_state[[j, k]];
                
                // Compute state updates in parallel
                for ti in 0..t {
                    let dt_val = dt_col[ti];
                    let u_val = u_conv_col[ti];
                    let b_val = b_t[[ti, k]];
                    
                    let a_val = crate::pade::exp(-dt_val * a_scale).clamp(0.0, 1.0);
                    let k_val = (1.0 - a_val) / a_scale;
                    
                    // Sequential update (parallel version would use prefix sum)
                    let prev = if ti == 0 { 0.0 } else { state[[ti-1, idx]] };
                    let current = a_val * prev + k_val * b_val * u_val;
                    
                    state[[ti, idx]] = current;
                    z[[ti, j]] += c_t[[ti, k]] * current;
                }
            }
        }
        
        (state, z, y_pre)
    }

    /// Block-diagonal A matrix computation
    fn compute_block_diagonal_a(
        &self,
        a_log: &Array2<f32>,      // [1, D] or [D, D] for block-diagonal
        proj_a: &Array2<f32>,     // [D, N]
        d: usize,
        n: usize,
    ) -> Array2<f32> {
        match self.a_matrix_type {
            AMatrixType::Diagonal => {
                // Original diagonal implementation
                let mut a_logits_state = self.w_out.dot(proj_a); // [D, N]
                let bias_d = a_log.row(0).to_owned();
                for j in 0..d {
                    let bj = bias_d[j];
                    for k in 0..n {
                        a_logits_state[[j, k]] += bj;
                    }
                }
                a_logits_state.mapv(|x| softplus(x) + 1e-6)
            }
            AMatrixType::BlockDiagonal => {
                // Enhanced block-diagonal implementation
                let block_size = self.scan_config.block_size.unwrap_or(4);
                let num_blocks = (d + block_size - 1) / block_size;
                
                let mut a_logits_state = Array2::<f32>::zeros((d, n));
                
                // Create block-diagonal structure
                for block_idx in 0..num_blocks {
                    let start = block_idx * block_size;
                    let end = (start + block_size).min(d);
                    let _block_d = end - start;
                    
                    for j in start..end {
                        let block_j = j - start;
                        let bias = a_log[[0, j]];
                        
                        for k in 0..n {
                            // Block-diagonal contribution
                            let block_contrib = self.w_out[[block_j, k]] * proj_a[[j, k]];
                            a_logits_state[[j, k]] = block_contrib + bias;
                        }
                    }
                }
                
                a_logits_state.mapv(|x| softplus(x) + 1e-6)
            }
        }
    }

    /// Enhanced forward with parallel scan and block-diagonal support
    pub fn forward_enhanced(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_kind = MambaCachedKind::Mamba2Parallel;

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            return Array2::zeros((t, d));
        }

        self.ensure_projections(d);
        let n = self.cached_state_dim;
        let proj_state = self
            .cached_proj_state
            .as_ref()
            .expect("proj_state must exist");
        let proj_a = self.cached_proj_a.as_ref().expect("proj_a must exist");

        let in2 = input.dot(&self.w_in) + self.b_in.broadcast((t, 2 * d)).unwrap();
        let u_pre = in2.slice(ndarray::s![.., 0..d]).to_owned();
        let gate_logits = in2.slice(ndarray::s![.., d..2 * d]).to_owned();

        let u_act = self.richards_act.forward_matrix_f32(&u_pre);
        let gate = self.richards_gate.forward_const(&gate_logits);

        // Enhanced dt computation with better numerical stability
        let dt_logits = u_pre.clone();
        let dt = dt_logits.mapv(|x| softplus(x) + 1e-6);

        // Project input with enhanced projections
        let b_full = input.dot(&self.w_b) + self.b_b.broadcast((t, d)).unwrap();
        let b_logits = b_full.dot(proj_state);
        let mut b_t = Array2::<f32>::zeros(b_logits.raw_dim());
        self.richards_tanh.forward_matrix_f32_into(&b_logits, &mut b_t);

        let c_full = input.dot(&self.w_c) + self.b_c.broadcast((t, d)).unwrap();
        let c_logits = c_full.dot(proj_state);
        let mut c_t = Array2::<f32>::zeros(c_logits.raw_dim());
        self.richards_tanh.forward_matrix_f32_into(&c_logits, &mut c_t);

        // Enhanced A computation with block-diagonal support
        let a_scale_state = self.compute_block_diagonal_a(&self.a_log, proj_a, d, n);

        let u_conv = self.depthwise_causal_conv(&u_act);

        // Choose scan method based on configuration
        let (state, z, mut y_pre) = match self.scan_config.method {
            ScanMethod::Sequential => {
                // Fall back to original sequential scan
                self.sequential_scan_fallback(&dt, &a_scale_state, &b_t, &c_t, &u_conv)
            }
            ScanMethod::Parallel => {
                // Use enhanced parallel scan
                self.parallel_selective_scan(&dt, &a_scale_state, &b_t, &c_t, &u_conv)
            }
            ScanMethod::MemoryEfficient => {
                // Use memory-efficient scan for long sequences
                self.memory_efficient_scan(&dt, &a_scale_state, &b_t, &c_t, &u_conv)
            }
        };

        // Apply gating and final projection
        for ti in 0..t {
            for j in 0..d {
                y_pre[[ti, j]] = gate[[ti, j]] * z[[ti, j]];
            }
        }

        let out_pre = y_pre.dot(&self.w_dt) + self.b_dt.broadcast((t, d)).unwrap();

        // Cache for gradient computation
        self.cached_input = Some(input.clone());
        self.cached_u_pre = Some(u_pre);
        self.cached_u_act = Some(u_act);
        self.cached_gate = Some(gate);
        self.cached_gate_logits = Some(gate_logits);
        self.cached_dt_logits = Some(dt_logits);
        self.cached_dt = Some(dt);
        self.cached_b_logits = Some(b_logits);
        self.cached_b_t = Some(b_t);
        self.cached_c_logits = Some(c_logits);
        self.cached_c_t = Some(c_t);
        self.cached_a_logits_state = Some(self.w_out.dot(proj_a));
        self.cached_a_scale_state = Some(a_scale_state);
        self.cached_a = None;
        self.cached_u_conv = Some(u_conv);
        self.cached_state_prev = None; // Not computed in parallel version
        self.cached_state = Some(state);
        self.cached_z = Some(z);
        self.cached_y_pre = Some(y_pre);
        self.cached_out_pre = Some(out_pre.clone());

        out_pre
    }

    /// Fallback sequential scan for compatibility
    fn sequential_scan_fallback(
        &self,
        dt: &Array2<f32>,
        a_scale_state: &Array2<f32>,
        b_t: &Array2<f32>,
        c_t: &Array2<f32>,
        u_conv: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let t = dt.nrows();
        let d = dt.ncols();
        let n = b_t.ncols();
        
        let mut state_prev = Array2::<f32>::zeros((t, d * n));
        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let y_pre = Array2::<f32>::zeros((t, d));
        
        let mut s = Array1::<f32>::zeros(d * n);

        for ti in 0..t {
            for j in 0..d {
                let dtj = dt[[ti, j]];
                let uj = u_conv[[ti, j]];
                let mut zj = 0.0;

                for k in 0..n {
                    let idx = j * n + k;
                    let prev = s[idx];
                    state_prev[[ti, idx]] = prev;

                    let a_scale = a_scale_state[[j, k]];
                    let aj = crate::pade::exp(-dtj * a_scale).clamp(0.0, 1.0);
                    let inp = b_t[[ti, k]] * uj;
                    let kk = (1.0 - aj) / a_scale;
                    let sj = aj * prev + kk * inp;

                    s[idx] = sj;
                    state[[ti, idx]] = sj;
                    zj += c_t[[ti, k]] * sj;
                }

                z[[ti, j]] = zj;
            }
        }
        
        (state, z, y_pre)
    }

    /// Memory-efficient scan for long sequences
    fn memory_efficient_scan(
        &self,
        dt: &Array2<f32>,
        a_scale_state: &Array2<f32>,
        b_t: &Array2<f32>,
        c_t: &Array2<f32>,
        u_conv: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let t = dt.nrows();
        let d = dt.ncols();
        let n = b_t.ncols();
        let chunk_size = self.scan_config.chunk_size.unwrap_or(128);
        
        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let y_pre = Array2::<f32>::zeros((t, d));
        
        // Process in chunks to reduce memory usage
        for chunk_start in (0..t).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(t);
            
            for j in 0..d {
                for k in 0..n {
                    let idx = j * n + k;
                    let a_scale = a_scale_state[[j, k]];
                    
                    // Process chunk with reduced memory footprint
                    for ti in chunk_start..chunk_end {
                        let dt_val = dt[[ti, j]];
                        let u_val = u_conv[[ti, j]];
                        let b_val = b_t[[ti, k]];
                        
                        let a_val = crate::pade::exp(-dt_val * a_scale).clamp(0.0, 1.0);
                        let k_val = (1.0 - a_val) / a_scale;
                        
                        let prev = if ti == 0 || chunk_start == 0 {
                            0.0
                        } else {
                            state[[ti-1, idx]]
                        };
                        
                        let current = a_val * prev + k_val * b_val * u_val;
                        state[[ti, idx]] = current;
                        z[[ti, j]] += c_t[[ti, k]] * current;
                    }
                }
            }
        }
        
        (state, z, y_pre)
    }

    pub fn forward_mamba2(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_kind = MambaCachedKind::Mamba2;

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            return Array2::zeros((t, d));
        }

        self.ensure_projections_mamba2(d);
        let n = self.cached_state_dim;
        let proj_state = self
            .cached_proj_state
            .as_ref()
            .expect("proj_state must exist");

        let head_dim = Self::head_dim_mamba2(d);
        let head_offsets = Self::make_head_offsets(d, head_dim);
        let num_heads = head_offsets.len().saturating_sub(1).max(1);

        // in-projection
        let in2 = input.dot(&self.w_in) + self.b_in.broadcast((t, 2 * d)).unwrap();
        let u_pre = in2.slice(ndarray::s![.., 0..d]).to_owned();
        let gate_logits = in2.slice(ndarray::s![.., d..2 * d]).to_owned();

        let silu = RichardsActivation::sigmoid(false);
        let sigmoid = RichardsCurve::sigmoid(false);
        let tanh = RichardsCurve::tanh(false);

        let u_act = silu.forward_matrix_f32(&u_pre);
        let mut gate = Array2::<f32>::zeros(gate_logits.raw_dim());
        sigmoid.forward_matrix_f32_into(&gate_logits, &mut gate);

        // dt (matches current Mamba impl here: derived from u_pre)
        let dt_logits = u_pre.clone();
        let dt = dt_logits.mapv(|x| softplus(x) + 1e-6);

        // B/C per head
        let b_full = input.dot(&self.w_b) + self.b_b.broadcast((t, d)).unwrap();
        let c_full = input.dot(&self.w_c) + self.b_c.broadcast((t, d)).unwrap();

        let mut b_logits = Array2::<f32>::zeros((t, num_heads * n));
        let mut c_logits = Array2::<f32>::zeros((t, num_heads * n));
        for h in 0..num_heads {
            let hs = head_offsets[h];
            let he = head_offsets[h + 1];
            let base = h * n;
            let b_head = b_full
                .slice(ndarray::s![.., hs..he])
                .dot(&proj_state.slice(ndarray::s![hs..he, ..]));
            let c_head = c_full
                .slice(ndarray::s![.., hs..he])
                .dot(&proj_state.slice(ndarray::s![hs..he, ..]));
            b_logits
                .slice_mut(ndarray::s![.., base..base + n])
                .assign(&b_head);
            c_logits
                .slice_mut(ndarray::s![.., base..base + n])
                .assign(&c_head);
        }
        let mut b_t = Array2::<f32>::zeros(b_logits.raw_dim());
        tanh.forward_matrix_f32_into(&b_logits, &mut b_t);
        let mut c_t = Array2::<f32>::zeros(c_logits.raw_dim());
        tanh.forward_matrix_f32_into(&c_logits, &mut c_t);

        let u_conv = self.depthwise_causal_conv(&u_act);

        // dt_head[t,h] = mean dt[t,j] for channels in head
        let mut dt_head = Array2::<f32>::zeros((t, num_heads));
        for h in 0..num_heads {
            let hs = head_offsets[h];
            let he = head_offsets[h + 1];
            let denom = (he - hs).max(1) as f32;
            for ti in 0..t {
                let mut acc = 0.0f32;
                for j in hs..he {
                    acc += dt[[ti, j]];
                }
                dt_head[[ti, h]] = acc / denom;
            }
        }

        // SSD: scalar A per head (shared across channels and state dims)
        let mut a_scale_head = Array2::<f32>::zeros((1, num_heads));
        for h in 0..num_heads {
            let hs = head_offsets[h];
            let he = head_offsets[h + 1];
            let denom = (he - hs).max(1) as f32;
            let mut acc = 0.0f32;
            for j in hs..he {
                acc += softplus(self.a_log[[0, j]]).max(1e-6);
            }
            a_scale_head[[0, h]] = (acc / denom).max(1e-6);
        }

        let mut a_head = Array2::<f32>::zeros((t, num_heads));
        for ti in 0..t {
            for h in 0..num_heads {
                a_head[[ti, h]] =
                    crate::pade::exp(-dt_head[[ti, h]] * a_scale_head[[0, h]]).clamp(0.0, 1.0);
            }
        }

        let mut state_prev = Array2::<f32>::zeros((t, d * n));
        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let mut y_pre = Array2::<f32>::zeros((t, d));

        let d_skip_row = self.d_skip.row(0).to_owned();
        let mut s = Array1::<f32>::zeros(d * n);

        for ti in 0..t {
            for h in 0..num_heads {
                let hs = head_offsets[h];
                let he = head_offsets[h + 1];
                let base = h * n;
                let a = a_head[[ti, h]];
                let a_scale = a_scale_head[[0, h]];
                let kk = (1.0 - a) / a_scale;

                for j in hs..he {
                    let uj = u_conv[[ti, j]];
                    let mut zj = d_skip_row[j] * uj;
                    for k in 0..n {
                        let idx = j * n + k;
                        let prev = s[idx];
                        state_prev[[ti, idx]] = prev;

                        let inp = b_t[[ti, base + k]] * uj;
                        let sj = a * prev + kk * inp;
                        s[idx] = sj;
                        state[[ti, idx]] = sj;
                        zj += c_t[[ti, base + k]] * sj;
                    }
                    z[[ti, j]] = zj;
                    y_pre[[ti, j]] = gate[[ti, j]] * zj;
                }
            }
        }

        let out_pre = y_pre.dot(&self.w_dt) + self.b_dt.broadcast((t, d)).unwrap();

        // caches
        self.cached_input = Some(input.clone());
        self.cached_u_pre = Some(u_pre);
        self.cached_u_act = Some(u_act);
        self.cached_gate = Some(gate);
        self.cached_dt_logits = Some(dt_logits);
        self.cached_dt = Some(dt);
        self.cached_b_logits = Some(b_logits);
        self.cached_b_t = Some(b_t);
        self.cached_c_logits = Some(c_logits);
        self.cached_c_t = Some(c_t);
        self.cached_a_logits_state = None;
        self.cached_a_scale_state = None;
        self.cached_a = None;
        self.cached_u_conv = Some(u_conv);
        self.cached_state_prev = Some(state_prev);
        self.cached_state = Some(state);
        self.cached_z = Some(z);
        self.cached_y_pre = Some(y_pre);
        self.cached_out_pre = Some(out_pre.clone());

        self.cached_head_offsets = Some(head_offsets);
        self.cached_dt_head = Some(dt_head);
        self.cached_a_head = Some(a_head);
        self.cached_a_scale_head = Some(a_scale_head);

        out_pre
    }

    fn compute_gradients_mamba2(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let u_pre = self.cached_u_pre.as_ref().expect("cache u_pre");
        let u_act = self.cached_u_act.as_ref().expect("cache u_act");
        let gate = self.cached_gate.as_ref().expect("cache gate");
        let dt_logits = self.cached_dt_logits.as_ref().expect("cache dt_logits");
        let _dt = self.cached_dt.as_ref().expect("cache dt");
        let b_logits = self.cached_b_logits.as_ref().expect("cache b_logits");
        let b_t = self.cached_b_t.as_ref().expect("cache b_t");
        let c_logits = self.cached_c_logits.as_ref().expect("cache c_logits");
        let c_t = self.cached_c_t.as_ref().expect("cache c_t");
        let u_conv = self.cached_u_conv.as_ref().expect("cache u_conv");
        let state_prev = self.cached_state_prev.as_ref().expect("cache state_prev");
        let state = self.cached_state.as_ref().expect("cache state");
        let z = self.cached_z.as_ref().expect("cache z");
        let y_pre = self.cached_y_pre.as_ref().expect("cache y_pre");
        let head_offsets = self
            .cached_head_offsets
            .as_ref()
            .expect("cache head_offsets");
        let dt_head = self.cached_dt_head.as_ref().expect("cache dt_head");
        let a_head = self.cached_a_head.as_ref().expect("cache a_head");
        let a_scale_head = self
            .cached_a_scale_head
            .as_ref()
            .expect("cache a_scale_head");

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            return (Array2::zeros(input.raw_dim()), vec![]);
        }

        let sigmoid = RichardsCurve::sigmoid(false);
        let tanh = RichardsCurve::tanh(false);

        let num_heads = head_offsets.len().saturating_sub(1).max(1);
        let n = self.cached_state_dim;
        let proj_state = self
            .cached_proj_state
            .as_ref()
            .expect("proj_state must exist");

        // out = y_pre W_dt + b_dt
        let grad_w_dt = y_pre.t().dot(output_grads);
        let grad_b_dt = output_grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        let d_y_pre = output_grads.dot(&self.w_dt.t());

        // gate = sigmoid(gate_logits)
        // d/dgate_logits [ gate * z ] = (d_y_pre * z) * gate * (1-gate)
        let mut d_gate_logits = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                let gt = gate[[ti, j]];
                d_gate_logits[[ti, j]] = (d_y_pre[[ti, j]] * z[[ti, j]]) * gt * (1.0 - gt);
            }
        }

        // z = sum_k c[t,h,k] * state[t,j,k] + d_skip * u_conv
        let d_skip_row = self.d_skip.row(0).to_owned();
        let mut grad_d_skip = Array2::<f32>::zeros((1, d));
        let mut d_u_conv = Array2::<f32>::zeros((t, d));
        let mut d_c = Array2::<f32>::zeros((t, num_heads * n));

        for ti in 0..t {
            for h in 0..num_heads {
                let hs = head_offsets[h];
                let he = head_offsets[h + 1];
                let base = h * n;
                for j in hs..he {
                    let dz = d_y_pre[[ti, j]] * gate[[ti, j]];
                    grad_d_skip[[0, j]] += dz * u_conv[[ti, j]];
                    d_u_conv[[ti, j]] += dz * d_skip_row[j];
                    for k in 0..n {
                        let idx = j * n + k;
                        d_c[[ti, base + k]] += dz * state[[ti, idx]];
                    }
                }
            }
        }

        // backprop through scan with shared A per head
        let mut d_b = Array2::<f32>::zeros((t, num_heads * n));
        let mut d_dt = Array2::<f32>::zeros((t, d));
        let mut d_a_scale_head = Array1::<f32>::zeros(num_heads);

        let mut d_state_next = Array1::<f32>::zeros(d * n);
        for ti in (0..t).rev() {
            for h in 0..num_heads {
                let hs = head_offsets[h];
                let he = head_offsets[h + 1];
                let base = h * n;
                let a = a_head[[ti, h]];
                let a_scale = a_scale_head[[0, h]];
                let kk = (1.0 - a) / a_scale;

                let mut d_a_shared = 0.0f32;
                let mut d_a_scale_local = 0.0f32;

                for j in hs..he {
                    let uj = u_conv[[ti, j]];
                    for k in 0..n {
                        let idx = j * n + k;
                        // Base contribution d_state[t, j, k] = dz * c_t[t, h, k]
                        let dz = d_y_pre[[ti, j]] * gate[[ti, j]];
                        let mut ds = dz * c_t[[ti, base + k]] + d_state_next[idx];
                        if !ds.is_finite() {
                            ds = 0.0;
                        }

                        let prev = state_prev[[ti, idx]];
                        let inp = b_t[[ti, base + k]] * uj;

                        d_u_conv[[ti, j]] += ds * kk * b_t[[ti, base + k]];
                        d_b[[ti, base + k]] += ds * kk * uj;

                        let d_a = ds * (prev - inp / a_scale);
                        d_a_shared += d_a;
                        d_a_scale_local += ds * (-(1.0 - a) / (a_scale * a_scale)) * inp;

                        d_state_next[idx] = ds * a;
                    }
                }

                // a = exp(-dt_head * a_scale)
                let dt_h = dt_head[[ti, h]];
                let d_dt_head = d_a_shared * (-a_scale * a);
                let denom = (he - hs).max(1) as f32;
                for j in hs..he {
                    d_dt[[ti, j]] += d_dt_head / denom;
                }

                d_a_scale_head[h] += d_a_shared * (-dt_h * a) + d_a_scale_local;
            }
        }

        // a_scale_head[h] = mean_j softplus(a_log[j])
        let mut grad_a_log = Array2::<f32>::zeros((1, d));
        for h in 0..num_heads {
            let hs = head_offsets[h];
            let he = head_offsets[h + 1];
            let denom = (he - hs).max(1) as f32;
            for j in hs..he {
                grad_a_log[[0, j]] +=
                    (d_a_scale_head[h] / denom) * sigmoid.forward_scalar_f32(self.a_log[[0, j]]);
            }
        }

        // dt = softplus(dt_logits)
        let mut d_dt_logits = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                d_dt_logits[[ti, j]] = d_dt[[ti, j]] * sigmoid.forward_scalar_f32(dt_logits[[ti, j]]);
            }
        }

        // b_t = tanh(b_logits), c_t = tanh(c_logits)
        let mut d_b_logits = Array2::<f32>::zeros((t, num_heads * n));
        let mut d_c_logits = Array2::<f32>::zeros((t, num_heads * n));
        for ti in 0..t {
            for idx in 0..(num_heads * n) {
                let db = d_b[[ti, idx]];
                let dc = d_c[[ti, idx]];
                d_b_logits[[ti, idx]] = db * tanh.derivative_scalar_f32(b_logits[[ti, idx]]);
                d_c_logits[[ti, idx]] = dc * tanh.derivative_scalar_f32(c_logits[[ti, idx]]);
            }
        }

        // depthwise conv backprop: u_conv = conv(u_act)
        let k = self.conv_kernel;
        let mut grad_conv_w = Array2::<f32>::zeros((k, d));
        let grad_conv_b = d_u_conv.sum_axis(Axis(0)).insert_axis(Axis(0));
        let mut d_u_act = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            let start = if ti + 1 >= k { ti + 1 - k } else { 0 };
            let mut kk = 0usize;
            for tj in start..=ti {
                let wrow = kk;
                for j in 0..d {
                    let g = d_u_conv[[ti, j]];
                    grad_conv_w[[wrow, j]] += g * u_act[[tj, j]];
                    d_u_act[[tj, j]] += g * self.conv_w[[wrow, j]];
                }
                kk += 1;
            }
        }

        // u_act = silu(u_pre)
        let mut d_u_pre = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                let x = u_pre[[ti, j]];
                let s = sigmoid.forward_scalar_f32(x);
                let ds = sigmoid.derivative_scalar_f32(x);
                let d_silu = s + x * ds;
                d_u_pre[[ti, j]] = d_u_act[[ti, j]] * d_silu;
            }
        }

        // add dt path: dt_logits == u_pre
        for ti in 0..t {
            for j in 0..d {
                d_u_pre[[ti, j]] += d_dt_logits[[ti, j]];
            }
        }

        // in-projection grads
        let mut d_in2 = Array2::<f32>::zeros((t, 2 * d));
        d_in2.slice_mut(ndarray::s![.., 0..d]).assign(&d_u_pre);
        d_in2
            .slice_mut(ndarray::s![.., d..2 * d])
            .assign(&d_gate_logits);
        let grad_w_in = input.t().dot(&d_in2);
        let grad_b_in = d_in2.sum_axis(Axis(0)).insert_axis(Axis(0));

        // Backprop B/C logits into full (T,D)
        let mut d_b_full = Array2::<f32>::zeros((t, d));
        let mut d_c_full = Array2::<f32>::zeros((t, d));
        for h in 0..num_heads {
            let hs = head_offsets[h];
            let he = head_offsets[h + 1];
            let base = h * n;
            let proj_head = proj_state.slice(ndarray::s![hs..he, ..]);
            let proj_head_t = proj_head.t();
            let d_b_head = d_b_logits
                .slice(ndarray::s![.., base..base + n])
                .dot(&proj_head_t);
            let d_c_head = d_c_logits
                .slice(ndarray::s![.., base..base + n])
                .dot(&proj_head_t);
            d_b_full
                .slice_mut(ndarray::s![.., hs..he])
                .assign(&d_b_head);
            d_c_full
                .slice_mut(ndarray::s![.., hs..he])
                .assign(&d_c_head);
        }

        let grad_w_b = input.t().dot(&d_b_full);
        let grad_b_b = d_b_full.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_w_c = input.t().dot(&d_c_full);
        let grad_b_c = d_c_full.sum_axis(Axis(0)).insert_axis(Axis(0));

        // input grads
        let dx_in = d_in2.dot(&self.w_in.t());
        let dx_b = d_b_full.dot(&self.w_b.t());
        let dx_c = d_c_full.dot(&self.w_c.t());
        let grad_input = dx_in + dx_b + dx_c;

        // In this Mamba2 variant, w_out/b_out are not used; keep grads as zero.
        let grad_w_out = Array2::<f32>::zeros(self.w_out.raw_dim());
        let grad_b_out = Array2::<f32>::zeros(self.b_out.raw_dim());

        // Note: b_logits/c_logits are cached for debugging/inspection; gradients flow through tanh.
        let _ = b_logits;
        let _ = c_logits;

        (
            grad_input,
            vec![
                grad_w_in,
                grad_b_in,
                grad_w_dt,
                grad_b_dt,
                grad_w_b,
                grad_b_b,
                grad_w_c,
                grad_b_c,
                grad_a_log,
                grad_d_skip,
                grad_conv_w,
                grad_conv_b,
                grad_w_out,
                grad_b_out,
            ],
        )
    }
}

impl Layer for Mamba {
    fn layer_type(&self) -> &str {
        "Mamba"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_cached(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (dx, pgrads) = self.compute_gradients(input, grads);
        let _ = self.apply_gradients(&pgrads, lr);
        dx
    }

    fn parameters(&self) -> usize {
        self.w_in.len()
            + self.b_in.len()
            + self.w_dt.len()
            + self.b_dt.len()
            + self.w_b.len()
            + self.b_b.len()
            + self.w_c.len()
            + self.b_c.len()
            + self.a_log.len()
            + self.d_skip.len()
            + self.conv_w.len()
            + self.conv_b.len()
            + self.w_out.len()
            + self.b_out.len()
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        for a in [
            &self.w_in,
            &self.b_in,
            &self.w_dt,
            &self.b_dt,
            &self.w_b,
            &self.b_b,
            &self.w_c,
            &self.b_c,
            &self.a_log,
            &self.d_skip,
            &self.conv_w,
            &self.conv_b,
            &self.w_out,
            &self.b_out,
        ] {
            sumsq += a.iter().map(|&x| x * x).sum::<f32>();
        }
        sumsq.sqrt()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        if self.cached_kind == MambaCachedKind::Mamba2 {
            return self.compute_gradients_mamba2(input, output_grads);
        }

        let u_pre = self.cached_u_pre.as_ref().expect("cache u_pre");
        let u_act = self.cached_u_act.as_ref().expect("cache u_act");
        let gate = self.cached_gate.as_ref().expect("cache gate");
        let dt_logits = self.cached_dt_logits.as_ref().expect("cache dt_logits");
        let dt = self.cached_dt.as_ref().expect("cache dt");
        let b_logits = self.cached_b_logits.as_ref().expect("cache b_logits");
        let b_t = self.cached_b_t.as_ref().expect("cache b_t");
        let c_logits = self.cached_c_logits.as_ref().expect("cache c_logits");
        let c_t = self.cached_c_t.as_ref().expect("cache c_t");
        let a_logits_state = self
            .cached_a_logits_state
            .as_ref()
            .expect("cache a_logits_state");
        let a_scale_state = self
            .cached_a_scale_state
            .as_ref()
            .expect("cache a_scale_state");
        let u_conv = self.cached_u_conv.as_ref().expect("cache u_conv");
        let state_prev = self.cached_state_prev.as_ref().expect("cache state_prev");
        let state = self.cached_state.as_ref().expect("cache state");
        let z = self.cached_z.as_ref().expect("cache z");
        let y_pre = self.cached_y_pre.as_ref().expect("cache y_pre");

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            return (Array2::zeros(input.raw_dim()), vec![]);
        }

        let sigmoid = RichardsCurve::sigmoid(false);
        let tanh = RichardsCurve::tanh(false);

        let n = Self::desired_state_dim(d);
        let proj_state = self
            .cached_proj_state
            .as_ref()
            .expect("proj_state must exist");
        let proj_a = self.cached_proj_a.as_ref().expect("proj_a must exist");

        // out = y_pre W_dt + b_dt
        let grad_w_dt = y_pre.t().dot(output_grads);
        let grad_b_dt = output_grads.sum_axis(Axis(0)).insert_axis(Axis(0));
        let d_y_pre = output_grads.dot(&self.w_dt.t());

        // gate = sigmoid(gate_logits)
        // d/dgate_logits [ gate * z ] = (d_y_pre * z) * gate * (1-gate)
        let mut d_gate_logits = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                let gt = gate[[ti, j]];
                d_gate_logits[[ti, j]] = (d_y_pre[[ti, j]] * z[[ti, j]]) * gt * (1.0 - gt);
            }
        }

        // z = sum_k c_k * state_k + d_skip*u_conv
        let d_skip_row = self.d_skip.row(0).to_owned();
        let mut grad_d_skip = Array2::<f32>::zeros((1, d));
        let mut d_u_conv = Array2::<f32>::zeros((t, d));
        let mut d_c = Array2::<f32>::zeros((t, n));
        for ti in 0..t {
            for j in 0..d {
                let dz = d_y_pre[[ti, j]] * gate[[ti, j]];
                grad_d_skip[[0, j]] += dz * u_conv[[ti, j]];
                d_u_conv[[ti, j]] += dz * d_skip_row[j];
                for k in 0..n {
                    let idx = j * n + k;
                    d_c[[ti, k]] += dz * state[[ti, idx]];
                }
            }
        }

        // backprop through scan (multi-state)
        let mut d_dt = Array2::<f32>::zeros((t, d));
        let mut d_b = Array2::<f32>::zeros((t, n));
        let mut d_a_scale = Array2::<f32>::zeros((d, n));

        let mut d_state_next = Array1::<f32>::zeros(d * n);
        for ti in (0..t).rev() {
            for j in 0..d {
                let dtj = dt[[ti, j]];
                let uj = u_conv[[ti, j]];
                for k in 0..n {
                    let idx = j * n + k;
                    // Base contribution d_state[t, j, k] = dz * c_t[t, k]
                    let dz = d_y_pre[[ti, j]] * gate[[ti, j]];
                    let mut ds = dz * c_t[[ti, k]] + d_state_next[idx];
                    if !ds.is_finite() {
                        ds = 0.0;
                    }

                    let prev = state_prev[[ti, idx]];
                    let a_scale = a_scale_state[[j, k]];
                    let aj = crate::pade::exp(-dtj * a_scale).clamp(0.0, 1.0);
                    let inp = b_t[[ti, k]] * uj;
                    let kk = (1.0 - aj) / a_scale;

                    // inp = b * u
                    d_u_conv[[ti, j]] += ds * kk * b_t[[ti, k]];
                    d_b[[ti, k]] += ds * kk * uj;

                    // d_a
                    let d_a = ds * (prev - inp / a_scale);

                    // d_a_scale from k term
                    d_a_scale[[j, k]] += ds * (-(1.0 - aj) / (a_scale * a_scale)) * inp;

                    // a = exp(-dt*a_scale)
                    d_dt[[ti, j]] += d_a * (-a_scale * aj);
                    d_a_scale[[j, k]] += d_a * (-dtj * aj);

                    d_state_next[idx] = ds * aj;
                }
            }
        }

        // A_scale = softplus(A_logits)
        let mut d_a_logits_state = Array2::<f32>::zeros((d, n));
        for j in 0..d {
            for k in 0..n {
                d_a_logits_state[[j, k]] =
                    d_a_scale[[j, k]] * sigmoid.forward_scalar_f32(a_logits_state[[j, k]]);
            }
        }

        // A_logits = w_out.dot(proj_a) + (a_log + b_out)
        let grad_w_out = d_a_logits_state.dot(&proj_a.t());
        let mut grad_a_log = Array2::<f32>::zeros((1, d));
        let mut grad_b_out = Array2::<f32>::zeros((1, d));
        for j in 0..d {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += d_a_logits_state[[j, k]];
            }
            grad_a_log[[0, j]] = acc;
            grad_b_out[[0, j]] = acc;
        }

        // dt = softplus(dt_logits) + eps
        let mut d_dt_logits = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                d_dt_logits[[ti, j]] = d_dt[[ti, j]] * sigmoid.forward_scalar_f32(dt_logits[[ti, j]]);
            }
        }

        // b_t = tanh(b_logits), c_t = tanh(c_logits)
        let mut d_b_logits = Array2::<f32>::zeros((t, n));
        let mut d_c_logits = Array2::<f32>::zeros((t, n));
        for ti in 0..t {
            for k in 0..n {
                let db = d_b[[ti, k]];
                let dc = d_c[[ti, k]];
                d_b_logits[[ti, k]] = db * tanh.derivative_scalar_f32(b_logits[[ti, k]]);
                d_c_logits[[ti, k]] = dc * tanh.derivative_scalar_f32(c_logits[[ti, k]]);
            }
        }

        // depthwise conv backprop: u_conv = conv(u_act)
        let k = self.conv_kernel;
        let mut grad_conv_w = Array2::<f32>::zeros((k, d));
        let grad_conv_b = d_u_conv.sum_axis(Axis(0)).insert_axis(Axis(0));
        let mut d_u_act = Array2::<f32>::zeros((t, d));

        for ti in 0..t {
            let start = if ti + 1 >= k { ti + 1 - k } else { 0 };
            let mut kk = 0usize;
            for tj in start..=ti {
                let wrow = kk;
                for j in 0..d {
                    let g = d_u_conv[[ti, j]];
                    grad_conv_w[[wrow, j]] += g * u_act[[tj, j]];
                    d_u_act[[tj, j]] += g * self.conv_w[[wrow, j]];
                }
                kk += 1;
            }
        }

        // u_act = silu(u_pre)
        let mut d_u_pre = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                let x = u_pre[[ti, j]];
                let s = sigmoid.forward_scalar_f32(x);
                let ds = sigmoid.derivative_scalar_f32(x);
                let d_silu = s + x * ds;
                d_u_pre[[ti, j]] = d_u_act[[ti, j]] * d_silu;
            }
        }

        // add dt path: dt_logits == u_pre
        for ti in 0..t {
            for j in 0..d {
                d_u_pre[[ti, j]] += d_dt_logits[[ti, j]];
            }
        }

        // in-projection grads
        let mut grad_w_in = Array2::<f32>::zeros((d, 2 * d));
        let mut grad_b_in = Array2::<f32>::zeros((1, 2 * d));
        let mut d_in2 = Array2::<f32>::zeros((t, 2 * d));
        d_in2.slice_mut(ndarray::s![.., 0..d]).assign(&d_u_pre);
        d_in2
            .slice_mut(ndarray::s![.., d..2 * d])
            .assign(&d_gate_logits);

        grad_w_in = input.t().dot(&d_in2);
        grad_b_in = d_in2.sum_axis(Axis(0)).insert_axis(Axis(0));

        // B/C path gradients: B_logits = (input.dot(w_b) + b_b) dot proj_state
        // d_full = d_logits dot proj_state^T
        let d_b_full = d_b_logits.dot(&proj_state.t());
        let grad_w_b = input.t().dot(&d_b_full);
        let grad_b_b = d_b_full.sum_axis(Axis(0)).insert_axis(Axis(0));

        let d_c_full = d_c_logits.dot(&proj_state.t());
        let grad_w_c = input.t().dot(&d_c_full);
        let grad_b_c = d_c_full.sum_axis(Axis(0)).insert_axis(Axis(0));

        // input grads
        let dx_in = d_in2.dot(&self.w_in.t());
        let dx_b = d_b_full.dot(&self.w_b.t());
        let dx_c = d_c_full.dot(&self.w_c.t());
        let grad_input = dx_in + dx_b + dx_c;

        (
            grad_input,
            vec![
                grad_w_in,
                grad_b_in,
                grad_w_dt,
                grad_b_dt,
                grad_w_b,
                grad_b_b,
                grad_w_c,
                grad_b_c,
                grad_a_log,
                grad_d_skip,
                grad_conv_w,
                grad_conv_b,
                grad_w_out,
                grad_b_out,
            ],
        )
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        // Expected order:
        // w_in, b_in, w_dt, b_dt, w_b, b_b, w_c, b_c, a_log, d_skip, conv_w, conv_b, w_out, b_out
        if gradients.len() < 14 {
            return Ok(());
        }

        self.opt_w_in
            .step(&mut self.w_in, &gradients[0], learning_rate);
        self.opt_b_in
            .step(&mut self.b_in, &gradients[1], learning_rate);
        self.opt_w_dt
            .step(&mut self.w_dt, &gradients[2], learning_rate);
        self.opt_b_dt
            .step(&mut self.b_dt, &gradients[3], learning_rate);
        self.opt_w_b
            .step(&mut self.w_b, &gradients[4], learning_rate);
        self.opt_b_b
            .step(&mut self.b_b, &gradients[5], learning_rate);
        self.opt_w_c
            .step(&mut self.w_c, &gradients[6], learning_rate);
        self.opt_b_c
            .step(&mut self.b_c, &gradients[7], learning_rate);
        self.opt_a_log
            .step(&mut self.a_log, &gradients[8], learning_rate);
        self.opt_d_skip
            .step(&mut self.d_skip, &gradients[9], learning_rate);
        self.opt_conv_w
            .step(&mut self.conv_w, &gradients[10], learning_rate);
        self.opt_conv_b
            .step(&mut self.conv_b, &gradients[11], learning_rate);
        self.opt_w_out
            .step(&mut self.w_out, &gradients[12], learning_rate);
        self.opt_b_out
            .step(&mut self.b_out, &gradients[13], learning_rate);

        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.cached_kind = MambaCachedKind::Mamba1;
        self.cached_input = None;
        self.cached_u_pre = None;
        self.cached_u_act = None;
        self.cached_gate = None;
        self.cached_dt_logits = None;
        self.cached_dt = None;
        self.cached_b_logits = None;
        self.cached_b_t = None;
        self.cached_c_logits = None;
        self.cached_c_t = None;
        self.cached_a_logits_state = None;
        self.cached_a_scale_state = None;
        self.cached_a = None;
        self.cached_u_conv = None;
        self.cached_state_prev = None;
        self.cached_state = None;
        self.cached_z = None;
        self.cached_y_pre = None;
        self.cached_out_pre = None;

        self.cached_head_offsets = None;
        self.cached_dt_head = None;
        self.cached_a_head = None;
        self.cached_a_scale_head = None;
    }
}

/// Configuration for Mamba layer with enhanced options
#[derive(Debug, Clone)]
pub struct MambaConfig {
    a_matrix_type: AMatrixType,
    scan_config: ScanConfig,
    pub use_enhanced_init: bool,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            a_matrix_type: AMatrixType::Diagonal,
            scan_config: ScanConfig {
                method: ScanMethod::Sequential,
                block_size: Some(4),
                chunk_size: Some(128),
            },
            use_enhanced_init: false,
        }
    }
}

impl MambaConfig {
    /// Enhanced configuration with parallel scan and block-diagonal A matrix
    pub fn enhanced() -> Self {
        Self {
            a_matrix_type: AMatrixType::BlockDiagonal,
            scan_config: ScanConfig {
                method: ScanMethod::Parallel,
                block_size: Some(4),
                chunk_size: Some(256),
            },
            use_enhanced_init: true,
        }
    }

    /// Memory-efficient configuration for long sequences
    pub fn memory_efficient() -> Self {
        Self {
            a_matrix_type: AMatrixType::Diagonal,
            scan_config: ScanConfig {
                method: ScanMethod::MemoryEfficient,
                block_size: Some(4),
                chunk_size: Some(64),
            },
            use_enhanced_init: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mamba_forward_backward_shapes() {
        let mut layer = Mamba::new_with_kernel(16, 3);
        let x = Array2::<f32>::zeros((8, 16));
        let y = layer.forward(&x);
        assert_eq!(y.shape(), [8, 16]);

        let grads = Array2::<f32>::ones((8, 16));
        let dx = layer.backward(&grads, 1e-3);
        assert_eq!(dx.shape(), [8, 16]);
        assert!(dx.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mamba_enhanced_forward() {
        let config = MambaConfig::enhanced();
        let mut layer = Mamba::new_with_config(16, 3, config);
        let x = Array2::<f32>::zeros((8, 16));
        let y = layer.forward_enhanced(&x);
        assert_eq!(y.shape(), [8, 16]);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mamba_memory_efficient_forward() {
        let config = MambaConfig::memory_efficient();
        let mut layer = Mamba::new_with_config(16, 3, config);
        let x = Array2::<f32>::zeros((128, 16)); // Longer sequence
        let y = layer.forward_enhanced(&x);
        assert_eq!(y.shape(), [128, 16]);
        assert!(y.iter().all(|v| v.is_finite()));
    }
}
