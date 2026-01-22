use std::cell::RefCell;

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip, s};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use crate::{
    adam::Adam,
    errors::Result,
    mixtures::{HeadSelectionStrategy, MoHGating},
    network::Layer,
    richards::{RichardsActivation, RichardsCurve, RichardsGate},
    rng::get_rng,
};

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_SCAN_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    #[allow(clippy::missing_const_for_thread_local)]
    static TLS_SCAN_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn with_tls_scan_a<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TLS_SCAN_A.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() != len {
            buf.resize(len, 0.0);
        }
        f(buf.as_mut_slice())
    })
}

#[inline]
fn with_tls_scan_b<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    TLS_SCAN_B.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() != len {
            buf.resize(len, 0.0);
        }
        f(buf.as_mut_slice())
    })
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MambaCachedKind {
    Mamba1,
    Mamba2,
    Mamba2Parallel, // Enhanced with parallel scan
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum AMatrixType {
    Diagonal,      // Original: diagonal A matrix
    BlockDiagonal, // Enhanced: block-diagonal A matrix
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
    block_size: Option<usize>, // For block-diagonal A
    chunk_size: Option<usize>, // For memory-efficient scan
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
            richards_tanh: data.richards_tanh.unwrap_or_else(mamba_default_tanh_curve),
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
        embed_dim.clamp(1, 16)
    }

    #[inline]
    fn desired_state_dim_mamba2(embed_dim: usize) -> usize {
        // Mamba-2 / SSD typically benefits from larger state sizes.
        embed_dim.clamp(16, 32)
    }

    #[inline]
    fn head_dim_mamba2(embed_dim: usize) -> usize {
        // Typical SSD head dimension is ~64.
        embed_dim.clamp(1, 64)
    }

    #[inline]
    fn make_head_offsets(d: usize, head_dim: usize) -> Vec<usize> {
        if d == 0 {
            return vec![0];
        }
        let hd = head_dim.max(1);
        let num_heads = d.div_ceil(hd);
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
                let inv = if norm2 > 1e-12 {
                    1.0 / norm2.sqrt()
                } else {
                    1.0
                };
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
                let inv = if norm2 > 1e-12 {
                    1.0 / norm2.sqrt()
                } else {
                    1.0
                };
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
            let start = (ti + 1).saturating_sub(k);
            for (wrow, tj) in (start..=ti).enumerate() {
                for j in 0..d {
                    out[[ti, j]] += self.conv_w[[wrow, j]] * u[[tj, j]];
                }
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

        // Canonical-style dt: learned via the in-projection stream (u_pre), not via an extra D×D
        // projection. This keeps parameter count unchanged while allowing
        // per-token/per-channel dt.
        let dt_logits = u_pre.clone();
        let dt = dt_logits.mapv(|x| softplus(x) + 1e-6);

        // Project input into a smaller (N) space for B/C, without adding parameters.
        let b_full = input.dot(&self.w_b) + self.b_b.broadcast((t, d)).unwrap();
        let b_logits = b_full.dot(proj_state);
        let mut b_t = Array2::<f32>::zeros(b_logits.raw_dim());
        self.richards_tanh
            .forward_matrix_f32_into(&b_logits, &mut b_t);

        let c_full = input.dot(&self.w_c) + self.b_c.broadcast((t, d)).unwrap();
        let c_logits = c_full.dot(proj_state);
        let mut c_t = Array2::<f32>::zeros(c_logits.raw_dim());
        self.richards_tanh
            .forward_matrix_f32_into(&c_logits, &mut c_t);

        // Build A logits/state scales using w_out (and biases) mapped into (D×N) via a fixed
        // projection. A_scale is positive; we use ZOH discretization with a = exp(-dt *
        // A_scale).
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

        // Output projection uses the existing (w_dt, b_dt) tensors to keep parameter count
        // unchanged.
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
        dt: &Array2<f32>,            // [T, D]
        a_scale_state: &Array2<f32>, // [D, N]
        b_t: &Array2<f32>,           // [T, N]
        c_t: &Array2<f32>,           // [T, N]
        u_conv: &Array2<f32>,        // [T, D]
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let t = dt.nrows();
        let d = dt.ncols();
        let n = b_t.ncols();

        let mut state = Array2::<f32>::zeros((t, d * n));
        let mut z = Array2::<f32>::zeros((t, d));
        let y_pre = Array2::<f32>::zeros((t, d));

        if t == 0 || d == 0 || n == 0 {
            return (state, z, y_pre);
        }

        let d_skip_row = self.d_skip.row(0).to_owned();
        let chunk_size = self.scan_config.chunk_size.unwrap_or(256).max(1).min(t);
        let num_chunks = t.div_ceil(chunk_size);

        // For each feature dimension j, run a chunk-parallel associative scan over time.
        // Each state component follows an affine recurrence:
        //   s_t = A_t * s_{t-1} + B_t
        // and compositions are associative:
        //   (A2,B2) ⊕ (A1,B1) = (A2*A1, A2*B1 + B2)
        let mut chunk_a = vec![0.0f32; num_chunks * n];
        let mut chunk_b = vec![0.0f32; num_chunks * n];
        let mut prefix_b = vec![0.0f32; num_chunks * n];
        let mut b_prefix = vec![0.0f32; n];
        for j in 0..d {
            chunk_a.fill(0.0);
            chunk_b.fill(0.0);
            prefix_b.fill(0.0);
            b_prefix.fill(0.0);

            // 1) Compute per-chunk totals in parallel over time chunks.
            // Stored as flat arrays to avoid Vec<Vec<..>> cloning.
            chunk_a
                .par_chunks_mut(n)
                .zip(chunk_b.par_chunks_mut(n))
                .enumerate()
                .for_each(|(chunk_idx, (a_out, b_out))| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(t);
                    with_tls_scan_a(n, |a_tot| {
                        a_tot.fill(1.0);
                        with_tls_scan_b(n, |b_tot| {
                            b_tot.fill(0.0);

                            for ti in start..end {
                                let dt_val = dt[[ti, j]];
                                let u_val = u_conv[[ti, j]];
                                for k in 0..n {
                                    let a_scale = a_scale_state[[j, k]].max(1e-6);
                                    let a_val = crate::pade::exp(-dt_val * a_scale).clamp(0.0, 1.0);
                                    let b_step = ((1.0 - a_val) / a_scale) * b_t[[ti, k]] * u_val;

                                    b_tot[k] = a_val * b_tot[k] + b_step;
                                    a_tot[k] *= a_val;
                                }
                            }

                            a_out.copy_from_slice(a_tot);
                            b_out.copy_from_slice(b_tot);
                        })
                    });
                });

            // 2) Prefix over chunk totals (sequential; num_chunks is small), producing
            // initial state for each chunk.
            for chunk_idx in 0..num_chunks {
                let base = chunk_idx * n;
                prefix_b[base..(base + n)].copy_from_slice(&b_prefix[..n]);

                for k in 0..n {
                    let a_chunk = chunk_a[base + k];
                    let b_chunk = chunk_b[base + k];
                    b_prefix[k] = a_chunk * b_prefix[k] + b_chunk;
                }
            }

            // 3) Compute per-time states within each chunk in parallel, then write out.
            let chunk_outputs: Vec<(Vec<f32>, Vec<f32>)> = (0..num_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(t);
                    let len = end - start;

                    let pre_b = {
                        let base = chunk_idx * n;
                        &prefix_b[base..(base + n)]
                    };

                    let mut state_flat = vec![0.0f32; len * n];
                    let mut z_col = vec![0.0f32; len];

                    with_tls_scan_a(n, |a_loc| {
                        a_loc.fill(1.0);
                        with_tls_scan_b(n, |b_loc| {
                            b_loc.fill(0.0);

                            for (off, ti) in (start..end).enumerate() {
                                let dt_val = dt[[ti, j]];
                                let u_val = u_conv[[ti, j]];

                                let mut z_sum = d_skip_row[j] * u_val;
                                for k in 0..n {
                                    let a_scale = a_scale_state[[j, k]].max(1e-6);
                                    let a_val = crate::pade::exp(-dt_val * a_scale).clamp(0.0, 1.0);
                                    let b_step = ((1.0 - a_val) / a_scale) * b_t[[ti, k]] * u_val;

                                    b_loc[k] = a_val * b_loc[k] + b_step;
                                    a_loc[k] *= a_val;

                                    let s = a_loc[k] * pre_b[k] + b_loc[k];
                                    state_flat[off * n + k] = s;
                                    z_sum += c_t[[ti, k]] * s;
                                }
                                z_col[off] = z_sum;
                            }
                        })
                    });

                    (state_flat, z_col)
                })
                .collect();

            // Combine chunk outputs into the final state and z for this j.
            for (chunk_idx, (state_flat, z_col)) in chunk_outputs.iter().enumerate() {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(t);
                let len = end - start;
                let idx0 = j * n;

                for off in 0..len {
                    let ti = start + off;
                    for k in 0..n {
                        state[[ti, idx0 + k]] = state_flat[off * n + k];
                    }
                    z[[ti, j]] = z_col[off];
                }
            }
        }

        (state, z, y_pre)
    }

    /// Block-diagonal A matrix computation
    fn compute_block_diagonal_a(
        &self,
        a_log: &Array2<f32>,  // [1, D] or [D, D] for block-diagonal
        proj_a: &Array2<f32>, // [D, N]
        d: usize,
        n: usize,
    ) -> (Array2<f32>, Array2<f32>) {
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
                let a_scale_state = a_logits_state.mapv(|x| softplus(x) + 1e-6);
                (a_logits_state, a_scale_state)
            }
            AMatrixType::BlockDiagonal => {
                // Enhanced block-diagonal implementation
                let block_size = self.scan_config.block_size.unwrap_or(4).max(1);
                let num_blocks = d.div_ceil(block_size);

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

                let a_scale_state = a_logits_state.mapv(|x| softplus(x) + 1e-6);
                (a_logits_state, a_scale_state)
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
        self.richards_tanh
            .forward_matrix_f32_into(&b_logits, &mut b_t);

        let c_full = input.dot(&self.w_c) + self.b_c.broadcast((t, d)).unwrap();
        let c_logits = c_full.dot(proj_state);
        let mut c_t = Array2::<f32>::zeros(c_logits.raw_dim());
        self.richards_tanh
            .forward_matrix_f32_into(&c_logits, &mut c_t);

        // Enhanced A computation with block-diagonal support
        let (a_logits_state, a_scale_state) =
            self.compute_block_diagonal_a(&self.a_log, proj_a, d, n);

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
        self.cached_a_logits_state = Some(a_logits_state);
        self.cached_a_scale_state = Some(a_scale_state);
        self.cached_a = None;
        self.cached_u_conv = Some(u_conv);
        let mut state_prev = Array2::<f32>::zeros(state.raw_dim());
        for ti in 1..t {
            state_prev.row_mut(ti).assign(&state.row(ti - 1));
        }
        self.cached_state_prev = Some(state_prev);
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

        let d_skip_row = self.d_skip.row(0).to_owned();
        for ti in 0..t {
            for j in 0..d {
                z[[ti, j]] = d_skip_row[j] * u_conv[[ti, j]];
            }
        }

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

                        let prev = if ti == 0 { 0.0 } else { state[[ti - 1, idx]] };

                        let current = a_val * prev + k_val * b_val * u_val;
                        state[[ti, idx]] = current;
                        z[[ti, j]] += c_t[[ti, k]] * current;
                    }
                }
            }
        }

        (state, z, y_pre)
    }

    fn forward_mamba2_impl<D: Data<Elem = f32>>(
        &mut self,
        input: &ArrayBase<D, Ix2>,
    ) -> Array2<f32> {
        self.cached_kind = MambaCachedKind::Mamba2;

        let t = input.nrows();
        let d = input.ncols();
        if t == 0 || d == 0 {
            self.cached_input = Some(input.to_owned());
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
        self.cached_input = Some(input.to_owned());
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

    pub fn forward_mamba2(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.forward_mamba2_impl(input)
    }

    pub(crate) fn forward_mamba2_view(&mut self, input: &ArrayView2<f32>) -> Array2<f32> {
        self.forward_mamba2_impl(input)
    }

    fn compute_gradients_mamba2_impl<Din: Data<Elem = f32>, Dout: Data<Elem = f32>>(
        &self,
        input: &ArrayBase<Din, Ix2>,
        output_grads: &ArrayBase<Dout, Ix2>,
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
                d_dt_logits[[ti, j]] =
                    d_dt[[ti, j]] * sigmoid.forward_scalar_f32(dt_logits[[ti, j]]);
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
            let start = (ti + 1).saturating_sub(k);
            for (wrow, tj) in (start..=ti).enumerate() {
                for j in 0..d {
                    let g = d_u_conv[[ti, j]];
                    grad_conv_w[[wrow, j]] += g * u_act[[tj, j]];
                    d_u_act[[tj, j]] += g * self.conv_w[[wrow, j]];
                }
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

    fn compute_gradients_mamba2(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients_mamba2_impl(input, output_grads)
    }

    pub(crate) fn compute_gradients_mamba2_view(
        &self,
        input: &ArrayView2<f32>,
        output_grads: &ArrayView2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        self.compute_gradients_mamba2_impl(input, output_grads)
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
            + self.richards_act.weights().len()
            + self.richards_tanh.weights().len()
            + self.richards_gate.parameters()
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
        sumsq += self
            .richards_act
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq += self
            .richards_tanh
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq += self.richards_gate.weight_norm().powi(2);
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
        let gate_logits = self.cached_gate_logits.as_ref().expect("cache gate_logits");
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

        let mut d_gate = Array2::<f32>::zeros((t, d));
        for ti in 0..t {
            for j in 0..d {
                d_gate[[ti, j]] = d_y_pre[[ti, j]] * z[[ti, j]];
            }
        }
        let (d_gate_logits, gate_param_grads) =
            self.richards_gate.compute_gradients(gate_logits, &d_gate);

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
                d_dt_logits[[ti, j]] =
                    d_dt[[ti, j]] * sigmoid.forward_scalar_f32(dt_logits[[ti, j]]);
            }
        }

        // b_t = tanh(b_logits), c_t = tanh(c_logits)
        let mut d_b_logits = Array2::<f32>::zeros((t, n));
        let mut d_c_logits = Array2::<f32>::zeros((t, n));
        for ti in 0..t {
            for k in 0..n {
                let db = d_b[[ti, k]];
                let dc = d_c[[ti, k]];
                d_b_logits[[ti, k]] =
                    db * self.richards_tanh.derivative_scalar_f32(b_logits[[ti, k]]);
                d_c_logits[[ti, k]] =
                    dc * self.richards_tanh.derivative_scalar_f32(c_logits[[ti, k]]);
            }
        }

        // depthwise conv backprop: u_conv = conv(u_act)
        let k = self.conv_kernel;
        let mut grad_conv_w = Array2::<f32>::zeros((k, d));
        let grad_conv_b = d_u_conv.sum_axis(Axis(0)).insert_axis(Axis(0));
        let mut d_u_act = Array2::<f32>::zeros((t, d));

        for ti in 0..t {
            let start = (ti + 1).saturating_sub(k);
            for (kk, tj) in (start..=ti).enumerate() {
                let wrow = kk;
                for j in 0..d {
                    let g = d_u_conv[[ti, j]];
                    grad_conv_w[[wrow, j]] += g * u_act[[tj, j]];
                    d_u_act[[tj, j]] += g * self.conv_w[[wrow, j]];
                }
            }
        }

        let curve_output_grads = u_pre * &d_u_act;
        let u_act_param_grads = self
            .richards_act
            .richards_curve
            .grad_weights_matrix_f32(u_pre, &curve_output_grads);
        let mut u_act_param_grads_sum = Array2::<f32>::zeros((1, u_act_param_grads.len()));
        for (k, &g) in u_act_param_grads.iter().enumerate() {
            u_act_param_grads_sum[[0, k]] = g as f32;
        }

        let b_param_grads = self.richards_tanh.grad_weights_matrix_f32(b_logits, &d_b);
        let c_param_grads = self.richards_tanh.grad_weights_matrix_f32(c_logits, &d_c);
        let mut tanh_param_grads_sum = Array2::<f32>::zeros((1, b_param_grads.len()));
        for k in 0..b_param_grads.len() {
            tanh_param_grads_sum[[0, k]] = (b_param_grads[k] + c_param_grads[k]) as f32;
        }

        // u_act = richards_act(u_pre)
        let mut d_u_pre = Array2::<f32>::zeros((t, d));
        let mut act_deriv_row: Vec<f32> = Vec::new();
        let mut act_deriv_tmp: Vec<f32> = Vec::new();
        for (ti, row) in u_pre.outer_iter().enumerate() {
            let x_row = row.as_slice().unwrap();
            if act_deriv_row.len() != x_row.len() {
                act_deriv_row.resize(x_row.len(), 0.0);
                act_deriv_tmp.resize(x_row.len(), 0.0);
            }
            self.richards_act.derivative_into_f32_with_scratch(
                x_row,
                &mut act_deriv_row,
                &mut act_deriv_tmp,
            );
            for j in 0..d {
                d_u_pre[[ti, j]] = d_u_act[[ti, j]] * act_deriv_row[j];
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

        let mut param_grads = vec![
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
            u_act_param_grads_sum,
            tanh_param_grads_sum,
        ];
        param_grads.extend(gate_param_grads);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        if self.cached_kind == MambaCachedKind::Mamba2 {
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

            return Ok(());
        }

        // Expected order:
        // w_in, b_in, w_dt, b_dt, w_b, b_b, w_c, b_c, a_log, d_skip, conv_w, conv_b, w_out, b_out,
        // richards_act, richards_tanh, richards_gate...
        if gradients.len() < 16 {
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

        let mut idx = 14usize;
        let grad_act_vec: Vec<f64> = gradients[idx].iter().map(|&x| x as f64).collect();
        self.richards_act.step(&grad_act_vec, learning_rate as f64);
        idx += 1;

        let grad_tanh_vec: Vec<f64> = gradients[idx].iter().map(|&x| x as f64).collect();
        self.richards_tanh
            .step(&grad_tanh_vec, learning_rate as f64);
        idx += 1;

        if gradients.len() > idx {
            self.richards_gate
                .apply_gradients(&gradients[idx..], learning_rate)?;
        }

        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.cached_kind = MambaCachedKind::Mamba1;
        self.cached_input = None;
        self.cached_u_pre = None;
        self.cached_u_act = None;
        self.cached_gate = None;
        self.cached_gate_logits = None;
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoHMamba {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub gating_embed_dim: usize,

    #[serde(flatten)]
    pub moh: MoHGating,

    pub inner: Mamba,

    #[serde(skip_serializing, skip_deserializing)]
    cached_input: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_eff: Option<Array2<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    cached_inner_out: Option<Array2<f32>>,

    #[serde(skip_serializing, skip_deserializing)]
    pub last_avg_active_heads: Option<f32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub last_head_activity_vec: Option<Vec<f32>>,
    #[serde(skip_serializing, skip_deserializing)]
    pub last_token_head_activity_vec: Option<Vec<f32>>,
}

impl MoHMamba {
    pub fn new(embed_dim: usize, num_heads: usize, head_selection: &HeadSelectionStrategy) -> Self {
        let mut nh = num_heads.max(1);
        if embed_dim == 0 || embed_dim % nh != 0 {
            nh = 1;
        }
        let head_dim = if nh > 0 { embed_dim / nh } else { embed_dim };

        let budget = 1000usize;
        let gate_params = crate::richards::RichardsGate::new().parameters();
        let overhead = 2usize.saturating_mul(nh).saturating_add(gate_params);
        let max_wg = budget.saturating_sub(overhead);
        let gating_embed_dim = (max_wg / nh).max(1).min(embed_dim.max(1));

        let mut moh = MoHGating::new(gating_embed_dim, nh);
        moh.set_head_selection_config(head_selection);
        moh.head_selection_config.gating.use_learned_predictor = false;
        moh.threshold_predictor = None;
        moh.opt_w_tau = None;
        moh.opt_b_tau = None;
        moh.opt_w2_tau = None;
        moh.opt_b2_tau = None;
        moh.opt_cond_w_tau = None;

        let inner = Mamba::new(embed_dim);

        Self {
            embed_dim,
            num_heads: nh,
            head_dim,
            gating_embed_dim,
            moh,
            inner,
            cached_input: None,
            cached_eff: None,
            cached_inner_out: None,
            last_avg_active_heads: None,
            last_head_activity_vec: None,
            last_token_head_activity_vec: None,
        }
    }

    #[inline]
    fn clear_caches(&mut self) {
        self.cached_input = None;
        self.cached_eff = None;
        self.cached_inner_out = None;
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

impl Layer for MoHMamba {
    fn layer_type(&self) -> &str {
        "MoHMamba"
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

        let gd = self.gating_embed_dim.min(d);
        let gate_input = input.slice(s![.., 0..gd]);
        let eff = self.moh.forward_weights_view(&gate_input, None, None);
        self.cached_eff = Some(eff.clone());

        let y_inner = self.inner.forward(input);
        self.cached_inner_out = Some(y_inner.clone());

        let mut out = y_inner;
        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let eff_col = eff.column(h);
            let eff_col = eff_col.insert_axis(Axis(1));
            let eff_col = eff_col
                .broadcast((t, self.head_dim))
                .expect("broadcast must succeed for (t, head_dim)");
            let mut out_block = out.slice_mut(s![.., c0..c1]);
            Zip::from(&mut out_block).and(eff_col).for_each(|o, &w| {
                *o *= w;
            });
        }

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
        let heads_params: usize = self.inner.parameters();
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
        let wn = self.inner.weight_norm();
        sumsq += wn * wn;
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
            let gd = self.gating_embed_dim.min(d);
            let gate_input = input.slice(s![.., 0..gd]);
            eff_local = moh_tmp.forward_weights_view(&gate_input, None, None);
            &eff_local
        };

        let inner_out_local: Array2<f32>;
        let inner_out: &Array2<f32> = if can_use_cache
            && let Some(y) = self.cached_inner_out.as_ref().filter(|y| y.dim() == (t, d))
        {
            y
        } else {
            let mut inner = self.inner.clone();
            inner_out_local = inner.forward(input);
            &inner_out_local
        };

        let mut eff_grads = Array2::<f32>::zeros((t, self.num_heads));
        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            for i in 0..t {
                let mut acc = 0.0f32;
                for j in 0..self.head_dim {
                    acc += output_grads[[i, c0 + j]] * inner_out[[i, c0 + j]];
                }
                eff_grads[[i, h]] = acc;
            }
        }

        let mut scaled_grads = Array2::<f32>::zeros((t, d));
        for h in 0..self.num_heads {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let eff_col = eff.column(h);
            let eff_col = eff_col.insert_axis(Axis(1));
            let eff_col = eff_col
                .broadcast((t, self.head_dim))
                .expect("broadcast must succeed for (t, head_dim)");
            let og_block = output_grads.slice(s![.., c0..c1]);
            let mut sg_block = scaled_grads.slice_mut(s![.., c0..c1]);
            Zip::from(&mut sg_block)
                .and(og_block)
                .and(eff_col)
                .for_each(|sg, &og, &w| {
                    *sg = og * w;
                });
        }

        let (mut grad_input, mut grads) = if can_use_cache {
            self.inner.compute_gradients(input, &scaled_grads)
        } else {
            let mut inner = self.inner.clone();
            inner.forward(input);
            inner.compute_gradients(input, &scaled_grads)
        };

        let (dx_moh, moh_grads) = {
            let mut moh_local = self.moh.clone();
            let gd = self.gating_embed_dim.min(d);
            let gate_input = input.slice(s![.., 0..gd]);
            moh_local.compute_gradients_from_eff_view(&gate_input, &eff_grads)
        };
        {
            let gd = self.gating_embed_dim.min(d);
            let mut gi = grad_input.slice_mut(s![.., 0..gd]);
            gi += &dx_moh;
        }
        grads.extend(moh_grads);

        (grad_input, grads)
    }

    fn apply_gradients(&mut self, gradients: &[Array2<f32>], learning_rate: f32) -> Result<()> {
        let inner_n = if self.inner.cached_kind == MambaCachedKind::Mamba2 {
            14usize
        } else {
            16usize + self.inner.richards_gate.parameters()
        };
        let moh_n = self.moh.grad_arrays_len();
        if gradients.len() < inner_n + moh_n {
            return Ok(());
        }

        self.inner
            .apply_gradients(&gradients[..inner_n], learning_rate)?;
        self.moh
            .apply_gradients(&gradients[inner_n..], learning_rate)?;
        Ok(())
    }

    fn zero_gradients(&mut self) {
        self.inner.zero_gradients();
        self.moh.cached_soft_top_p_mask = None;
        self.clear_caches();
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

    #[test]
    fn parallel_scan_matches_sequential_fallback() {
        let config = MambaConfig::enhanced();
        let layer = Mamba::new_with_config(8, 3, config);

        let t = 64usize;
        let d = 8usize;
        let n = 4usize;

        let dt = Array2::from_shape_fn((t, d), |(ti, j)| 0.01 + (ti + j) as f32 * 1e-4);
        let a_scale_state = Array2::from_shape_fn((d, n), |(j, k)| 0.25 + (j + k) as f32 * 1e-2);
        let b_t = Array2::from_shape_fn((t, n), |(ti, k)| {
            ((ti as f32 * 0.03 + k as f32 * 0.7).sin() * 0.2).tanh()
        });
        let c_t = Array2::from_shape_fn((t, n), |(ti, k)| {
            ((ti as f32 * 0.02 + k as f32 * 0.5).cos() * 0.2).tanh()
        });
        let u_conv = Array2::from_shape_fn((t, d), |(ti, j)| {
            ((ti as f32 * 0.01 + j as f32 * 0.2).sin() * 0.5).tanh()
        });

        let (state_seq, z_seq, _) =
            layer.sequential_scan_fallback(&dt, &a_scale_state, &b_t, &c_t, &u_conv);
        let (state_par, z_par, _) =
            layer.parallel_selective_scan(&dt, &a_scale_state, &b_t, &c_t, &u_conv);
        let (state_mem, z_mem, _) =
            layer.memory_efficient_scan(&dt, &a_scale_state, &b_t, &c_t, &u_conv);

        let state_diff = (&state_seq - &state_par)
            .mapv(|v| v.abs())
            .mean()
            .unwrap_or(0.0);
        let z_diff = (&z_seq - &z_par).mapv(|v| v.abs()).mean().unwrap_or(0.0);

        assert!(
            state_diff < 1e-4,
            "state mismatch too large (mean abs diff={state_diff})"
        );
        assert!(
            z_diff < 1e-4,
            "z mismatch too large (mean abs diff={z_diff})"
        );

        let mem_state_diff = (&state_seq - &state_mem)
            .mapv(|v| v.abs())
            .mean()
            .unwrap_or(0.0);
        let mem_z_diff = (&z_seq - &z_mem).mapv(|v| v.abs()).mean().unwrap_or(0.0);
        assert!(
            mem_state_diff < 1e-4,
            "memory state mismatch too large (mean abs diff={mem_state_diff})"
        );
        assert!(
            mem_z_diff < 1e-4,
            "memory z mismatch too large (mean abs diff={mem_z_diff})"
        );

        let d_skip_row = layer.d_skip.row(0).to_owned();
        for (state, z) in [
            (&state_seq, &z_seq),
            (&state_par, &z_par),
            (&state_mem, &z_mem),
        ] {
            let mut mean_abs = 0.0f32;
            let mut count = 0.0f32;
            for ti in 0..t {
                for j in 0..d {
                    let mut expected = d_skip_row[j] * u_conv[[ti, j]];
                    for kk in 0..n {
                        expected += c_t[[ti, kk]] * state[[ti, j * n + kk]];
                    }
                    mean_abs += (z[[ti, j]] - expected).abs();
                    count += 1.0;
                }
            }
            mean_abs /= count.max(1.0);
            assert!(
                mean_abs < 1e-4,
                "z missing skip or state contribution (mean abs err={mean_abs})"
            );
        }
    }

    #[test]
    fn moh_mamba_forward_shape() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba::new(16, 4, &cfg);
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
    fn moh_mamba_grad_shapes() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba::new(12, 3, &cfg);
        let x = Array2::<f32>::from_elem((5, 12), 0.2);
        let y = layer.forward(&x);
        let grads = Array2::<f32>::from_elem(y.dim(), 0.1);

        let (dx, pgrads) = layer.compute_gradients(&x, &grads);
        assert_eq!(dx.dim(), x.dim());
        assert!(pgrads.len() >= 16 + layer.inner.richards_gate.parameters() + 4);
    }

    #[test]
    fn moh_mamba_compute_gradients_without_forward_is_finite() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let layer = MoHMamba::new(12, 3, &cfg);
        let x = Array2::from_shape_fn((7, 12), |(i, j)| ((i * 12 + j) as f32 * 0.013).sin());
        let grads = Array2::<f32>::from_elem((7, 12), 0.1);

        let expected_len =
            16 + layer.inner.richards_gate.parameters() + layer.moh.grad_arrays_len();
        let (dx, pgrads) = layer.compute_gradients(&x, &grads);

        assert_eq!(dx.dim(), x.dim());
        assert!(dx.iter().all(|v| v.is_finite()));
        assert_eq!(pgrads.len(), expected_len);
        assert!(pgrads.iter().all(|g| g.iter().all(|v| v.is_finite())));
    }

    #[test]
    fn moh_mamba_parameter_delta_within_1000() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let baseline = Mamba::new(64).parameters();
        let moh = MoHMamba::new(64, 16, &cfg).parameters();
        assert!(moh >= baseline);
        assert!(moh - baseline <= 1000);
    }

    #[test]
    fn moh_mamba_backward_updates_output() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut layer = MoHMamba::new(12, 3, &cfg);
        let x = Array2::from_shape_fn((9, 12), |(i, j)| ((i * 12 + j) as f32 * 0.011).sin());
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
}
