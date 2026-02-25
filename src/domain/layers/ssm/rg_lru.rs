use std::borrow::Cow;

use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Axis, Data, Ix2, Zip, s};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use std::sync::{Arc, Mutex};

use crate::{
    common::{
        errors::{ModelError, Result},
        rng::get_rng,
    },
    domain::{
        compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_gpu},
        layers::components::{
            StreamingWorkspaceManaged, UnifiedLayerWorkspace, WorkspaceManaged, WorkspaceStats,
        },
        mixtures::{HeadSelectionStrategy, MoHGating, moh_gating::MoHStreamingWorkspace},
        network::Layer,
        richards::RichardsCurve,
    },
    infrastructure::optimizer::adam::Adam,
};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::{
    gpu_backend_variants::SsmGpuBackend, unified_gpu_kernels::SsmParams,
};

#[derive(Debug, Clone, Default)]
pub struct RgLruStreamingWorkspace {
    pub h_prev: Array1<f32>,
    pub r_pre: Array1<f32>,
    pub i_pre: Array1<f32>,
    pub r: Array1<f32>,
    pub i: Array1<f32>,
    pub a: Array1<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct MoHRgLruStreamingWorkspace {
    pub moh_workspace: MoHStreamingWorkspace,
    pub output_buffer: Array1<f32>,
    pub head_output_buffer: Array1<f32>,
}

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
    pub streaming_workspace: Option<RgLruStreamingWorkspace>,

    /// Unified workspace for batch forward passes (consolidates buffer management).
    /// Replaces separate workspace pools with a single, coherent design.
    #[serde(skip_serializing, skip_deserializing)]
    unified_workspace: UnifiedLayerWorkspace,

    #[serde(skip, default)]
    compute_backend: ComputeBackend,

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip, default)]
    ssm_gpu_backend: Option<Arc<Mutex<SsmGpuBackend>>>,
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

    #[serde(skip)]
    pub streaming_workspace: Option<MoHRgLruStreamingWorkspace>,
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
            streaming_workspace: None,
        }
    }

    /// Set runtime compute backend for all per-head RG-LRU layers.
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set MoHRgLru backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    pub fn set_compute_backend_checked(&mut self, compute_backend: ComputeBackend) -> Result<()> {
        if self
            .heads
            .iter()
            .all(|head| head.compute_backend() == compute_backend)
        {
            return Ok(());
        }
        for head in &mut self.heads {
            head.set_compute_backend_checked(compute_backend)?;
        }
        Ok(())
    }

    /// Resolve and apply strict auto-GPU backend preference.
    pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        self.set_compute_backend_checked(backend)
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

    pub fn forward_step_into(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut Array1<f32>,
    ) {
        // Streaming step mode uses CPU path
        let d = input.len();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // Initialize workspace if needed
        if self.streaming_workspace.is_none() {
            self.streaming_workspace = Some(MoHRgLruStreamingWorkspace::default());
        }
        let ws = self.streaming_workspace.as_mut().unwrap();

        // Resize buffers if needed
        if ws.output_buffer.len() != d {
            ws.output_buffer = Array1::zeros(d);
        } else {
            ws.output_buffer.fill(0.0);
        }

        if ws.head_output_buffer.len() != head_dim {
            ws.head_output_buffer = Array1::zeros(head_dim);
        }

        if ws.moh_workspace.xw.len() != num_heads {
            ws.moh_workspace.xw = Array1::zeros(num_heads);
            ws.moh_workspace.g = Array1::zeros(num_heads);
            ws.moh_workspace.m = Array1::zeros(num_heads);
        }

        // 1. Compute MoH gating weights
        let gate_input = self.moh.gate_input_view(input);
        self.moh
            .forward_weights_into(&gate_input, &mut ws.moh_workspace);
        let eff_weights = &ws.moh_workspace.m;

        // 2. Process heads
        for (h, head) in self.heads.iter_mut().enumerate() {
            let start = h * head_dim;
            let end = start + head_dim;
            if start >= d {
                break;
            }

            let head_input = input.slice(s![start..end]);

            // Forward step into head_output_buffer
            head.forward_step_into(&head_input, &mut ws.head_output_buffer);

            let w = eff_weights[h];
            if w.abs() > 1e-9 {
                let mut out_slice = ws.output_buffer.slice_mut(s![start..end]);
                // Accumulate: out += w * head_out
                ndarray::Zip::from(&mut out_slice)
                    .and(&ws.head_output_buffer)
                    .for_each(|o, &v| *o += w * v);
            }
        }

        let (active_heads, head_vec, token_vec) =
            MoHGating::summarize_streaming_weights(&ws.moh_workspace.m);
        self.last_avg_active_heads = Some(active_heads);
        self.last_head_activity_vec = Some(head_vec);
        self.last_token_head_activity_vec = Some(token_vec);

        output.assign(&ws.output_buffer);
    }

    /// Forward pass with in-place output (Zero Allocation Pattern).
    ///
    /// Computes MoH routing and head outputs, writing results directly to the provided buffer.
    /// Eliminates the intermediate allocation of the output array.
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len, embed_dim)
    /// * `output` - Pre-allocated output buffer (seq_len, embed_dim)
    ///
    /// # Returns
    /// `Ok(())` if successful, error if output buffer has incorrect dimensions
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        let t = input.nrows();
        let d = input.ncols();
        let use_gpu = self
            .heads
            .first()
            .is_some_and(|first| first.compute_backend().is_gpu());

        // Validate output buffer dimensions
        if output.dim() != (t, d) {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Output dimension mismatch: expected ({}, {}), got {:?}",
                    t,
                    d,
                    output.dim()
                ),
            });
        }

        // Handle empty input case
        if t == 0 || d == 0 || self.num_heads == 0 || self.head_dim == 0 {
            self.clear_caches();
            self.cached_input = Some(input.clone());
            output.fill(0.0);
            return Ok(());
        }

        // Cache input for backward.
        self.cached_input = Some(input.clone());

        let input_view = input.view();
        let gate_input = self.moh.gate_input_view2(&input_view);
        let eff = self.moh.forward_weights_view(&gate_input, None, None);
        self.cached_eff = Some(eff.clone());

        // Zero-initialize output buffer
        output.fill(0.0);

        let head_outs: Vec<Array2<f32>> = if use_gpu {
            let mut outs = Vec::with_capacity(self.num_heads);
            for (h, head) in self.heads.iter_mut().enumerate() {
                let c0 = h * self.head_dim;
                let c1 = c0 + self.head_dim;
                let x_view = input.slice(s![.., c0..c1]);
                let x_owned = x_view.to_owned();
                let y_h = head.forward_gpu(&x_owned).map_err(|err| {
                    crate::common::errors::ModelError::Backend {
                        message: format!(
                            "MoHRgLru head {} GPU forward failed on backend '{}': {}",
                            h,
                            head.compute_backend().as_str(),
                            err
                        ),
                    }
                })?;
                outs.push(y_h);
            }
            outs
        } else {
            use rayon::prelude::*;
            self.heads
                .par_iter_mut()
                .enumerate()
                .map(|(h, head)| {
                    let c0 = h * self.head_dim;
                    let c1 = c0 + self.head_dim;
                    let x_view = input.slice(s![.., c0..c1]);
                    head.forward_view(&x_view)
                })
                .collect()
        };

        // Compute per-head outputs and apply per-token scaling.
        for (h, y_h) in head_outs.iter().enumerate().take(self.num_heads) {
            let c0 = h * self.head_dim;
            let c1 = c0 + self.head_dim;
            let eff_col = eff.column(h);
            let eff_col = eff_col.insert_axis(Axis(1));
            let eff_col = eff_col
                .broadcast((t, self.head_dim))
                .expect("broadcast must succeed for (t, head_dim)");
            let mut out_block = output.slice_mut(s![.., c0..c1]);
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

        Ok(())
    }

    /// GPU-accelerated MoH RG-LRU forward pass.
    ///
    /// Uses GPU execution for each RG-LRU head and applies MoH routing weights on CPU.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let backend = self
            .heads
            .first()
            .map(|h| h.compute_backend())
            .unwrap_or(ComputeBackend::Cpu);
        if !backend.is_gpu() {
            return Err(ModelError::Backend {
                message: "MoHRgLru::forward_gpu called without a GPU backend selected. \
                          Call set_compute_backend_checked(...) with a GPU backend first."
                    .to_string(),
            });
        }

        let mut output = Array2::zeros(input.raw_dim());
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// GPU-aware backward gradients for MoH RG-LRU.
    ///
    /// This path is strict: it errors when GPU backend is not selected.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        let backend = self
            .heads
            .first()
            .map(|h| h.compute_backend())
            .unwrap_or(ComputeBackend::Cpu);
        if !backend.is_gpu() {
            return Err(ModelError::Backend {
                message: "MoHRgLru::compute_gradients_gpu called without a GPU backend selected."
                    .to_string(),
            });
        }
        Ok(self.compute_gradients(input, output_grads))
    }

    /// GPU forward on non-GPU builds (strict no-fallback error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_gpu(&mut self, _input: &Array2<f32>) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message: "MoHRgLru GPU forward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
        })
    }

    /// GPU-aware backward gradients on non-GPU builds (strict error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn compute_gradients_gpu(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        Err(ModelError::Backend {
            message: "MoHRgLru GPU backward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
        })
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(input.len());
        self.forward_step_into(&input.view(), &mut output);
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
            streaming_workspace: None,
            unified_workspace: UnifiedLayerWorkspace::new(),
            compute_backend: ComputeBackend::Cpu,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            ssm_gpu_backend: None,
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
            streaming_workspace: None,
            unified_workspace: UnifiedLayerWorkspace::new(),
            compute_backend: ComputeBackend::Cpu,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            ssm_gpu_backend: None,
        }
    }

    /// Set runtime compute backend.
    #[inline]
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set RgLru backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    #[inline]
    pub fn set_compute_backend_checked(&mut self, compute_backend: ComputeBackend) -> Result<()> {
        if self.compute_backend == compute_backend {
            return Ok(());
        }

        if compute_backend.is_gpu() {
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                // Eagerly validate backend availability in strict mode.
                let _ = crate::domain::compute::GpuDevice::new(compute_backend)?;
            }

            #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
            {
                return Err(ModelError::Backend {
                    message: format!(
                        "RgLru requested GPU backend '{}' but this binary was built without GPU features.",
                        compute_backend.as_str()
                    ),
                });
            }
        }

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.ssm_gpu_backend = None;
        }

        self.compute_backend = compute_backend;
        self.unified_workspace.set_compute_backend(compute_backend);
        Ok(())
    }

    /// Resolve and apply strict auto-GPU backend preference.
    #[inline]
    pub fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        self.set_compute_backend_checked(backend)
    }

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn kernel_matrices_for_ssm_backend(
        &self,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
        let d = self.embed_dim.max(1);
        let w_f = self.w_a.clone();
        let w_r = self.w_x.clone();
        let mut w_o = Array2::<f32>::zeros((d, d));
        for i in 0..d {
            w_o[[i, i]] = 1.0;
        }

        let mut h_init = Array2::<f32>::zeros((1, d));
        if let Some(hprev) = self.cached_hprev.as_ref()
            && hprev.ncols() == d
            && hprev.nrows() > 0
        {
            h_init.row_mut(0).assign(&hprev.row(hprev.nrows() - 1));
        }

        (w_f, w_r, w_o, h_init)
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn ensure_ssm_gpu_backend(&mut self, seq_len: usize) -> Result<Arc<Mutex<SsmGpuBackend>>> {
        if self.ssm_gpu_backend.is_none() {
            let backend = SsmGpuBackend::rg_lru_with_backend(
                self.embed_dim.max(1),
                self.embed_dim,
                seq_len,
                1,
                self.compute_backend,
            )?;
            self.ssm_gpu_backend = Some(Arc::new(Mutex::new(backend)));
        }

        let backend_arc = self
            .ssm_gpu_backend
            .as_ref()
            .expect("SSM backend must exist after initialization")
            .clone();

        {
            let mut backend = backend_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to acquire RgLru cached GPU backend lock".to_string(),
            })?;
            backend.set_params(SsmParams::new(
                self.embed_dim.max(1),
                self.embed_dim,
                seq_len,
                1,
            ));
            let (w_f, w_r, w_o, h_init) = self.kernel_matrices_for_ssm_backend();
            backend.set_rg_lru_kernel_matrices(w_f, w_r, w_o, h_init);
        }

        Ok(backend_arc)
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

        Self::compute_decay_from_r_lambda(r, p.lambda, a);
    }

    #[inline]
    fn compute_decay_from_r_lambda(r: &Array2<f32>, lambda: &Array2<f32>, a: &mut Array2<f32>) {
        let (t, d) = r.dim();
        if a.dim() != (t, d) {
            *a = Array2::<f32>::zeros((t, d));
        }
        let c: f32 = 8.0;
        let log_base_a: Array1<f32> = lambda.row(0).to_owned().mapv(|x| -softplus(-x));
        for ti in 0..t {
            for j in 0..d {
                let lt = (c * r[[ti, j]] * log_base_a[j]).clamp(-80.0, 0.0);
                a[[ti, j]] = crate::domain::pade::exp(lt);
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn gpu_gemm_to_host(
        device: &mut crate::domain::compute::GpuDevice,
        lhs: &Array2<f32>,
        rhs: &Array2<f32>,
        m: usize,
        n: usize,
        k: usize,
        trans_lhs: bool,
        trans_rhs: bool,
    ) -> Result<Array2<f32>> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(Array2::zeros((m, n)));
        }

        let lhs_slice = lhs.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru gpu_gemm_to_host lhs must be contiguous".to_string(),
        })?;
        let rhs_slice = rhs.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru gpu_gemm_to_host rhs must be contiguous".to_string(),
        })?;

        let mut lhs_buf = device.allocate_f32(lhs.len())?;
        let mut rhs_buf = device.allocate_f32(rhs.len())?;
        let mut out_buf = device.allocate_f32(m * n)?;
        device.upload(lhs_slice, &mut lhs_buf)?;
        device.upload(rhs_slice, &mut rhs_buf)?;
        device.gemm_f32(
            1.0,
            &lhs_buf,
            &rhs_buf,
            0.0,
            &mut out_buf,
            m,
            n,
            k,
            trans_lhs,
            trans_rhs,
        )?;

        let mut host = vec![0.0f32; m * n];
        device.download(&out_buf, &mut host)?;
        device.deallocate(lhs_buf);
        device.deallocate(rhs_buf);
        device.deallocate(out_buf);

        Array2::from_shape_vec((m, n), host).map_err(|err| ModelError::InvalidInput {
            message: format!("RgLru gpu_gemm_to_host reshape failed: {err}"),
        })
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn compute_gates_into_parts_gpu(
        device: &mut crate::domain::compute::GpuDevice,
        input: &Array2<f32>,
        p: GatesParams<'_>,
        r: &mut Array2<f32>,
        i: &mut Array2<f32>,
        a: &mut Array2<f32>,
    ) -> Result<()> {
        let (t, d) = input.dim();
        if p.w_a.dim() != (d, d) || p.w_x.dim() != (d, d) {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "RgLru GPU gate weights must be ({d},{d}), got w_a={:?}, w_x={:?}",
                    p.w_a.dim(),
                    p.w_x.dim()
                ),
            });
        }
        if p.b_a.dim() != (1, d) || p.b_x.dim() != (1, d) || p.lambda.dim() != (1, d) {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "RgLru GPU gate biases/lambda must be (1,{d}), got b_a={:?}, b_x={:?}, lambda={:?}",
                    p.b_a.dim(),
                    p.b_x.dim(),
                    p.lambda.dim()
                ),
            });
        }
        if r.dim() != (t, d) {
            *r = Array2::<f32>::zeros((t, d));
        }
        if i.dim() != (t, d) {
            *i = Array2::<f32>::zeros((t, d));
        }
        if a.dim() != (t, d) {
            *a = Array2::<f32>::zeros((t, d));
        }
        if t == 0 || d == 0 {
            return Ok(());
        }

        let input_slice = input.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates input must be contiguous".to_string(),
        })?;
        let wa_slice = p.w_a.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates w_a must be contiguous".to_string(),
        })?;
        let wx_slice = p.w_x.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates w_x must be contiguous".to_string(),
        })?;

        let mut input_buf = device.allocate_f32(t * d)?;
        let mut wa_buf = device.allocate_f32(d * d)?;
        let mut wx_buf = device.allocate_f32(d * d)?;
        let mut r_logits_buf = device.allocate_f32(t * d)?;
        let mut i_logits_buf = device.allocate_f32(t * d)?;
        let mut r_sig_buf = device.allocate_f32(t * d)?;
        let mut i_sig_buf = device.allocate_f32(t * d)?;

        device.upload(input_slice, &mut input_buf)?;
        device.upload(wa_slice, &mut wa_buf)?;
        device.upload(wx_slice, &mut wx_buf)?;
        device.gemm_f32(
            1.0,
            &input_buf,
            &wa_buf,
            0.0,
            &mut r_logits_buf,
            t,
            d,
            d,
            false,
            false,
        )?;
        device.gemm_f32(
            1.0,
            &input_buf,
            &wx_buf,
            0.0,
            &mut i_logits_buf,
            t,
            d,
            d,
            false,
            false,
        )?;

        let b_a = p.b_a.row(0).to_owned();
        let b_x = p.b_x.row(0).to_owned();
        let mut b_a_expanded = vec![0.0f32; t * d];
        let mut b_x_expanded = vec![0.0f32; t * d];
        let b_a_slice = b_a.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates b_a must be contiguous".to_string(),
        })?;
        let b_x_slice = b_x.as_slice().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates b_x must be contiguous".to_string(),
        })?;
        for row in b_a_expanded.chunks_exact_mut(d) {
            row.copy_from_slice(b_a_slice);
        }
        for row in b_x_expanded.chunks_exact_mut(d) {
            row.copy_from_slice(b_x_slice);
        }
        let mut ba_buf = device.allocate_f32(t * d)?;
        let mut bx_buf = device.allocate_f32(t * d)?;
        device.upload(&b_a_expanded, &mut ba_buf)?;
        device.upload(&b_x_expanded, &mut bx_buf)?;
        device.add_scaled(1.0, &ba_buf, &mut r_logits_buf, t * d)?;
        device.add_scaled(1.0, &bx_buf, &mut i_logits_buf, t * d)?;

        device.sigmoid(&r_logits_buf, &mut r_sig_buf, t * d)?;
        device.sigmoid(&i_logits_buf, &mut i_sig_buf, t * d)?;

        let r_slice = r.as_slice_mut().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates output r must be contiguous".to_string(),
        })?;
        let i_slice = i.as_slice_mut().ok_or_else(|| ModelError::InvalidInput {
            message: "RgLru GPU gates output i must be contiguous".to_string(),
        })?;
        device.download(&r_sig_buf, r_slice)?;
        device.download(&i_sig_buf, i_slice)?;

        device.deallocate(input_buf);
        device.deallocate(wa_buf);
        device.deallocate(wx_buf);
        device.deallocate(r_logits_buf);
        device.deallocate(i_logits_buf);
        device.deallocate(r_sig_buf);
        device.deallocate(i_sig_buf);
        device.deallocate(ba_buf);
        device.deallocate(bx_buf);

        Self::compute_decay_from_r_lambda(r, p.lambda, a);
        Ok(())
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

    /// GPU-accelerated forward pass for RG-LRU (Phase 5.6.4)
    ///
    /// Computes recurrent gating with diagonal state updates on GPU.
    /// Target: 15x speedup on diagonal recurrence operations.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let (seq_len, embed_dim) = input.dim();
        if embed_dim != self.embed_dim {
            return Err(ModelError::InvalidInput {
                message: format!(
                    "RG-LRU GPU forward embed_dim mismatch: input={}, layer={}",
                    embed_dim, self.embed_dim
                ),
            });
        }
        if seq_len == 0 || embed_dim == 0 {
            return Ok(Array2::zeros((seq_len, embed_dim)));
        }

        if !self.compute_backend.is_gpu() {
            return Err(ModelError::Backend {
                message: "RgLru::forward_gpu called without a GPU backend selected. \
                          Call set_compute_backend_checked(...) with a GPU backend first."
                    .to_string(),
            });
        }

        if self
            .cached_r
            .as_ref()
            .is_none_or(|x| x.dim() != (seq_len, embed_dim))
        {
            self.cached_r = Some(Array2::<f32>::zeros((seq_len, embed_dim)));
        }
        if self
            .cached_i
            .as_ref()
            .is_none_or(|x| x.dim() != (seq_len, embed_dim))
        {
            self.cached_i = Some(Array2::<f32>::zeros((seq_len, embed_dim)));
        }
        if self
            .cached_a
            .as_ref()
            .is_none_or(|x| x.dim() != (seq_len, embed_dim))
        {
            self.cached_a = Some(Array2::<f32>::zeros((seq_len, embed_dim)));
        }
        if self
            .cached_hprev
            .as_ref()
            .is_none_or(|x| x.dim() != (seq_len, embed_dim))
        {
            self.cached_hprev = Some(Array2::<f32>::zeros((seq_len, embed_dim)));
        }

        let backend_arc = self.ensure_ssm_gpu_backend(seq_len)?;
        let device_arc = {
            let backend = backend_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to acquire RgLru cached GPU backend lock for device".to_string(),
            })?;
            backend.kernels().device()
        };
        let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock RgLru GPU device".to_string(),
        })?;

        let r = self
            .cached_r
            .as_mut()
            .expect("cached_r must be initialized");
        let i = self
            .cached_i
            .as_mut()
            .expect("cached_i must be initialized");
        let a = self
            .cached_a
            .as_mut()
            .expect("cached_a must be initialized");
        let hprev = self
            .cached_hprev
            .as_mut()
            .expect("cached_hprev must be initialized");

        Self::compute_gates_into_parts_gpu(
            &mut device,
            input,
            GatesParams {
                w_a: &self.w_a,
                b_a: &self.b_a,
                w_x: &self.w_x,
                b_x: &self.b_x,
                lambda: &self.lambda,
            },
            r,
            i,
            a,
        )?;

        let mut output = Array2::<f32>::zeros((seq_len, embed_dim));
        Self::compute_state_into(input, i, a, hprev, &mut output);
        self.cached_input = Some(input.clone());
        Ok(output)
    }

    /// GPU-aware backward gradients for RG-LRU.
    ///
    /// This path is strict: it errors when GPU backend is not selected.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        if !self.compute_backend.is_gpu() {
            return Err(ModelError::Backend {
                message: "RgLru::compute_gradients_gpu called without a GPU backend selected."
                    .to_string(),
            });
        }
        self.compute_gradients_impl_gpu(input, output_grads)
    }

    /// GPU-accelerated forward pass on non-GPU builds (strict no-fallback error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_gpu(&mut self, _input: &Array2<f32>) -> Result<Array2<f32>> {
        Err(ModelError::Backend {
            message: "RG-LRU GPU forward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
        })
    }

    /// GPU-aware backward gradients on non-GPU builds (strict error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn compute_gradients_gpu(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        Err(ModelError::Backend {
            message:
                "RG-LRU GPU backward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal."
                    .to_string(),
        })
    }

    pub fn forward_step_into(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut Array1<f32>,
    ) {
        // Streaming step mode uses CPU path
        let d = input.len();
        if self.streaming_workspace.is_none() {
            self.streaming_workspace = Some(RgLruStreamingWorkspace {
                h_prev: Array1::zeros(d),
                r_pre: Array1::zeros(d),
                i_pre: Array1::zeros(d),
                r: Array1::zeros(d),
                i: Array1::zeros(d),
                a: Array1::zeros(d),
            });
        }
        let ws = self.streaming_workspace.as_mut().unwrap();

        // 1. Compute pre-activations
        // r_pre = input * w_a^T + b_a
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_a.t(), input, 0.0, &mut ws.r_pre);
        ws.r_pre += &self.b_a.row(0);

        // i_pre = input * w_x^T + b_x
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_x.t(), input, 0.0, &mut ws.i_pre);
        ws.i_pre += &self.b_x.row(0);

        // 2. Activations
        let sigmoid = RichardsCurve::sigmoid(false);
        Zip::from(&mut ws.r)
            .and(&ws.r_pre)
            .for_each(|y, &x| *y = sigmoid.forward_scalar_f32(x));
        Zip::from(&mut ws.i)
            .and(&ws.i_pre)
            .for_each(|y, &x| *y = sigmoid.forward_scalar_f32(x));

        // 3. Compute 'a' (decay)
        let c: f32 = 8.0;
        let lambda = self.lambda.row(0);
        Zip::from(&mut ws.a)
            .and(&ws.r)
            .and(&lambda)
            .for_each(|y, &r, &l| {
                let log_base_a = -crate::domain::soft::softplus(-l);
                let lt = (c * r * log_base_a).clamp(-80.0, 0.0);
                *y = crate::domain::pade::exp(lt);
            });

        // 4. Update state and output
        // h_t = a * h_{t-1} + (1 - a) * (i * x)
        Zip::from(output.view_mut())
            .and(&mut ws.h_prev)
            .and(&ws.a)
            .and(&ws.i)
            .and(input)
            .for_each(|out, h, &a, &i, &x| {
                let u = i * x;
                let val = a * *h + (1.0 - a) * u;
                *h = val;
                *out = val;
            });
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(input.len());
        self.forward_step_into(&input.view(), &mut output);
        output
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

    /// In-place forward pass for RG-LRU.
    ///
    /// Computes the RG-LRU forward pass and writes the result directly into the output buffer,
    /// eliminating the allocation of intermediate state `h`. This reduces memory usage by ~4-8 KB/step
    /// for typical layer configurations.
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len × embed_dim)
    /// * `output` - Pre-allocated output buffer (seq_len × embed_dim)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if output dimensions don't match input
    ///
    /// # Panics
    /// Does not panic; returns error on dimension mismatch.
    ///
    /// # Example
    /// ```ignore
    /// let mut rg = RgLru::new(embed_dim);
    /// let input = Array2::zeros((seq_len, embed_dim));
    /// let mut output = Array2::zeros((seq_len, embed_dim));
    ///
    /// rg.forward_into(&input, &mut output)?;
    /// // output now contains the RG-LRU forward pass result
    /// ```
    pub fn forward_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) -> Result<()> {
        if self.compute_backend.is_gpu() {
            let gpu_out = self.forward_gpu(input)?;
            if output.raw_dim() == gpu_out.raw_dim() {
                output.assign(&gpu_out);
            } else {
                *output = gpu_out;
            }
            return Ok(());
        }
        let (t, d) = input.dim();

        // Validate output buffer dimensions
        if output.dim() != (t, d) {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Output dimension mismatch: expected ({}, {}), got {:?}",
                    t,
                    d,
                    output.dim()
                ),
            });
        }

        // Handle empty input case
        if t == 0 || d == 0 {
            self.cached_input = Some(input.clone());
            self.cached_r = Some(Array2::<f32>::zeros((t, d)));
            self.cached_i = Some(Array2::<f32>::zeros((t, d)));
            self.cached_a = Some(Array2::<f32>::zeros((t, d)));
            self.cached_hprev = Some(Array2::<f32>::zeros((t, d)));
            return Ok(());
        }

        // Allocate or reuse cached gate buffers
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

        let r = self
            .cached_r
            .as_mut()
            .expect("cached_r must be initialized");
        let i = self
            .cached_i
            .as_mut()
            .expect("cached_i must be initialized");
        let a = self
            .cached_a
            .as_mut()
            .expect("cached_a must be initialized");
        let hprev = self
            .cached_hprev
            .as_mut()
            .expect("cached_hprev must be initialized");

        // Compute gates in-place into cached buffers
        Self::compute_gates_into_parts(
            input,
            GatesParams {
                w_a: &self.w_a,
                b_a: &self.b_a,
                w_x: &self.w_x,
                b_x: &self.b_x,
                lambda: &self.lambda,
            },
            r,
            i,
            a,
        );

        // Compute state directly into output buffer (eliminates intermediate allocation)
        Self::compute_state_into(input, i, a, hprev, output);

        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        Ok(())
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

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn compute_gradients_impl_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        let (r, i, a, hprev) = self.compute_gates_and_state_from_cache_or_recompute(input);
        let r = r.as_ref();
        let i = i.as_ref();
        let a = a.as_ref();
        let hprev = hprev.as_ref();

        let (t, d) = input.dim();
        if output_grads.dim() != (t, d) {
            return Err(ModelError::DimensionMismatchDetailed {
                expected: format!("output_grads shape ({t}, {d})"),
                got: format!("{:?}", output_grads.dim()),
            });
        }
        if t == 0 || d == 0 {
            return Ok((Array2::zeros(input.raw_dim()), vec![]));
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

                dlogits_r[[ti, j]] = dr * rt * (1.0 - rt);
                dlogits_i[[ti, j]] = di * it * (1.0 - it);
            }
        }

        let mut d_lambda = Array2::<f32>::zeros((1, d));
        for j in 0..d {
            d_lambda[[0, j]] = dlog_base_a[j] * dlogsig_dlambda[j];
        }

        let backend_arc = self
            .ssm_gpu_backend
            .as_ref()
            .ok_or_else(|| ModelError::Backend {
                message: format!(
                    "RgLru GPU backward requires initialized cached GPU backend for '{}'. \
                     Call forward_gpu before compute_gradients.",
                    self.compute_backend.as_str()
                ),
            })?
            .clone();
        let device_arc = {
            let backend = backend_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock RgLru cached GPU backend during backward".to_string(),
            })?;
            backend.kernels().device()
        };
        let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
            message: "Failed to lock RgLru GPU device during backward".to_string(),
        })?;

        let grad_w_a =
            Self::gpu_gemm_to_host(&mut device, input, &dlogits_r, d, d, t, true, false)?;
        let grad_w_x =
            Self::gpu_gemm_to_host(&mut device, input, &dlogits_i, d, d, t, true, false)?;
        let dx_gate_r =
            Self::gpu_gemm_to_host(&mut device, &dlogits_r, &self.w_a, t, d, d, false, true)?;
        let dx_gate_i =
            Self::gpu_gemm_to_host(&mut device, &dlogits_i, &self.w_x, t, d, d, false, true)?;

        let grad_b_a = dlogits_r.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_b_x = dlogits_i.sum_axis(Axis(0)).insert_axis(Axis(0));
        let grad_input = &(&dx_gate_r + &dx_gate_i) + &d_x_from_u;

        Ok((
            grad_input,
            vec![grad_w_a, grad_b_a, grad_w_x, grad_b_x, d_lambda],
        ))
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
        // GPU execution uses forward_gpu when available
        if self.compute_backend.is_gpu() {
            return self
                .forward_gpu(input)
                .unwrap_or_else(|err| panic!("RgLru GPU forward failed: {err}"));
        }
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
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.compute_backend.is_gpu() {
            return self
                .compute_gradients_impl_gpu(input, output_grads)
                .unwrap_or_else(|err| panic!("RgLru GPU backward failed: {err}"));
        }
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

impl WorkspaceManaged for RgLru {
    /// Ensure workspace buffers are allocated with correct capacity.
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, embed_dim: usize) {
        self.unified_workspace
            .ensure_capacity(batch_size, seq_len, embed_dim);
    }

    /// Clear all workspace buffers to free memory.
    fn clear_workspace(&mut self) {
        self.unified_workspace.clear_workspace();
        // Also clear streaming state caches
        self.cached_input = None;
        self.cached_r = None;
        self.cached_i = None;
        self.cached_a = None;
        self.cached_hprev = None;
    }

    /// Get memory statistics for all managed buffers.
    fn workspace_stats(&self) -> WorkspaceStats {
        self.unified_workspace.workspace_stats()
    }
}

impl StreamingWorkspaceManaged for RgLru {
    /// Initialize streaming state for the given dimensions.
    fn init_streaming(&mut self, batch_size: usize, _embed_dim: usize) -> Result<()> {
        // Ensure unified workspace has capacity for streaming
        self.unified_workspace
            .ensure_capacity(batch_size, 1, self.embed_dim);

        // Enable streaming state buffer in unified workspace
        self.unified_workspace.set_streaming_state_enabled(true);

        // Initialize RG-LRU streaming workspace
        let h_prev = Array1::zeros(self.embed_dim);
        let r_pre = Array1::zeros(self.embed_dim);
        let i_pre = Array1::zeros(self.embed_dim);
        let r = Array1::zeros(self.embed_dim);
        let i = Array1::zeros(self.embed_dim);
        let a = Array1::zeros(self.embed_dim);

        self.streaming_workspace = Some(RgLruStreamingWorkspace {
            h_prev,
            r_pre,
            i_pre,
            r,
            i,
            a,
        });

        Ok(())
    }

    /// Reset streaming state between sequences.
    fn reset_streaming_state(&mut self) {
        if let Some(ref mut ws) = self.streaming_workspace {
            ws.h_prev.fill(0.0);
            ws.r_pre.fill(0.0);
            ws.i_pre.fill(0.0);
            ws.r.fill(0.0);
            ws.i.fill(0.0);
            ws.a.fill(0.0);
        }
    }

    /// Check if streaming state is active
    fn is_streaming(&self) -> bool {
        self.streaming_workspace.is_some()
    }
}

impl Layer for MoHRgLru {
    fn layer_type(&self) -> &str {
        "MoHRgLru"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(input.raw_dim());
        self.forward_into(input, &mut out)
            .unwrap_or_else(|err| panic!("MoHRgLru forward failed: {err}"));
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

impl WorkspaceManaged for MoHRgLru {
    /// Ensure workspace buffers are allocated with correct capacity.
    fn ensure_capacity(&mut self, batch_size: usize, seq_len: usize, _embed_dim: usize) {
        // Ensure capacity for all heads
        for head in &mut self.heads {
            head.ensure_capacity(batch_size, seq_len, self.head_dim);
        }
    }

    /// Clear all workspace buffers to free memory.
    fn clear_workspace(&mut self) {
        for head in &mut self.heads {
            head.clear_workspace();
        }
        self.clear_caches();
    }

    /// Get memory statistics for all managed buffers.
    fn workspace_stats(&self) -> WorkspaceStats {
        if self.heads.is_empty() {
            return WorkspaceStats {
                total_bytes: 0,
                buffer_count: 0,
                expected_shape: None,
            };
        }
        let head_stats: Vec<_> = self.heads.iter().map(|h| h.workspace_stats()).collect();
        WorkspaceStats::combined(&head_stats)
    }
}

impl StreamingWorkspaceManaged for MoHRgLru {
    /// Initialize streaming state for the given dimensions.
    fn init_streaming(&mut self, batch_size: usize, _embed_dim: usize) -> Result<()> {
        // Initialize streaming for each head
        for head in &mut self.heads {
            head.init_streaming(batch_size, self.head_dim)?;
        }

        // Initialize MoH streaming workspace
        let moh_ws = MoHRgLruStreamingWorkspace {
            moh_workspace: MoHStreamingWorkspace {
                xw: Array1::zeros(self.moh.w_g.nrows()),
                g: Array1::zeros(self.num_heads),
                m: Array1::zeros(self.num_heads),
            },
            output_buffer: Array1::zeros(self.embed_dim),
            head_output_buffer: Array1::zeros(self.head_dim),
        };

        self.streaming_workspace = Some(moh_ws);
        Ok(())
    }

    /// Reset streaming state between sequences.
    fn reset_streaming_state(&mut self) {
        for head in &mut self.heads {
            head.reset_streaming_state();
        }
        if let Some(ref mut ws) = self.streaming_workspace {
            ws.output_buffer.fill(0.0);
            ws.head_output_buffer.fill(0.0);
        }
    }

    /// Check if streaming state is active
    fn is_streaming(&self) -> bool {
        self.streaming_workspace.is_some() && self.heads.iter().any(|h| h.is_streaming())
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

    #[test]
    fn test_rg_lru_forward_into_equivalence() {
        use approx::assert_abs_diff_eq;

        let mut rg = RgLru::new(16);
        let input = Array2::<f32>::from_shape_fn((7, 16), |(i, j)| {
            (i as f32 * 0.1 + j as f32 * 0.01).sin()
        });

        // Standard forward
        let output_standard = rg.forward(&input);

        // In-place forward
        let mut output_into = Array2::zeros(input.dim());
        rg.forward_into(&input, &mut output_into)
            .expect("forward_into should succeed");

        // Verify outputs are bitwise equal (or very close due to floating-point operations)
        assert_abs_diff_eq!(output_standard.view(), output_into.view(), epsilon = 1e-5);
    }

    #[test]
    fn test_rg_lru_forward_into_dimension_validation() {
        let mut rg = RgLru::new(8);
        let input = Array2::<f32>::from_elem((5, 8), 0.1);

        // Wrong output dimensions should error
        let mut output_wrong = Array2::zeros((5, 16));
        let result = rg.forward_into(&input, &mut output_wrong);
        assert!(
            result.is_err(),
            "Should reject mismatched output dimensions"
        );

        // Correct dimensions should succeed
        let mut output_correct = Array2::zeros((5, 8));
        let result = rg.forward_into(&input, &mut output_correct);
        assert!(result.is_ok(), "Should accept matching dimensions");
    }

    #[test]
    fn test_rg_lru_forward_into_empty_input() {
        let mut rg = RgLru::new(8);
        let input = Array2::<f32>::zeros((0, 8));
        let mut output = Array2::zeros((0, 8));

        let result = rg.forward_into(&input, &mut output);
        assert!(result.is_ok(), "Should handle empty input gracefully");
    }

    #[test]
    fn test_rg_lru_forward_into_large_batch() {
        use approx::assert_abs_diff_eq;

        let mut rg = RgLru::new(32);
        let input =
            Array2::<f32>::from_shape_fn((256, 32), |(i, j)| (i as f32 * j as f32 * 0.001).cos());

        // Standard forward
        let output_standard = rg.forward(&input);

        // In-place forward
        let mut output_into = Array2::zeros(input.dim());
        rg.forward_into(&input, &mut output_into)
            .expect("forward_into should handle large batches");

        // Verify equivalence
        assert_abs_diff_eq!(output_standard.view(), output_into.view(), epsilon = 1e-5);
    }

    #[test]
    fn test_rg_lru_forward_into_backward_compatibility() {
        let mut rg1 = RgLru::new(12);
        let mut rg2 = RgLru::new(12);

        // Copy weights to ensure identical forward pass
        rg2.w_a = rg1.w_a.clone();
        rg2.w_x = rg1.w_x.clone();
        rg2.b_a = rg1.b_a.clone();
        rg2.b_x = rg1.b_x.clone();
        rg2.lambda = rg1.lambda.clone();

        let input = Array2::<f32>::from_elem((6, 12), 0.2);

        // Forward then backward with standard path
        let output_standard = rg1.forward(&input);
        let grad_output = Array2::<f32>::from_elem(output_standard.dim(), 0.1);
        let (grad_input_std, param_grads_std) = rg1.compute_gradients(&input, &grad_output);

        // Forward then backward with in-place path
        let mut output_into = Array2::zeros((6, 12));
        rg2.forward_into(&input, &mut output_into).unwrap();
        let (grad_input_into, param_grads_into) = rg2.compute_gradients(&input, &grad_output);

        // Gradients should match
        assert_eq!(grad_input_std.dim(), grad_input_into.dim());
        assert_eq!(param_grads_std.len(), param_grads_into.len());
    }

    #[test]
    fn test_moh_rg_lru_forward_into_equivalence() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh1 = MoHRgLru::new(12, 3, &cfg);
        let mut moh2 = MoHRgLru::new(12, 3, &cfg);

        // Copy MoH gating parameters to ensure identical routing
        moh2.moh = moh1.moh.clone();
        // Copy head parameters
        for (h1, h2) in moh1.heads.iter_mut().zip(moh2.heads.iter_mut()) {
            h2.w_a = h1.w_a.clone();
            h2.w_x = h1.w_x.clone();
            h2.b_a = h1.b_a.clone();
            h2.b_x = h1.b_x.clone();
            h2.lambda = h1.lambda.clone();
        }

        let input = Array2::<f32>::from_elem((8, 12), 0.15);

        // Standard forward pass
        let output_forward = moh1.forward(&input);

        // In-place forward pass
        let mut output_into = Array2::zeros((8, 12));
        let result = moh2.forward_into(&input, &mut output_into);

        assert!(result.is_ok());
        assert_eq!(output_forward.dim(), output_into.dim());

        // Check numerical equivalence (allow small numerical differences)
        for (a, b) in output_forward.iter().zip(output_into.iter()) {
            assert!((a - b).abs() < 1e-4, "Outputs differ: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_moh_rg_lru_forward_into_dimension_validation() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHRgLru::new(12, 3, &cfg);
        let input = Array2::<f32>::from_elem((6, 12), 0.1);

        // Wrong dimensions should fail
        let mut output_wrong = Array2::zeros((5, 12));
        let result = moh.forward_into(&input, &mut output_wrong);
        assert!(result.is_err());

        // Correct dimensions should succeed
        let mut output_correct = Array2::zeros((6, 12));
        let result = moh.forward_into(&input, &mut output_correct);
        assert!(result.is_ok());
    }

    #[test]
    fn test_moh_rg_lru_forward_into_empty_input() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        let cfg = HeadSelectionStrategy::Fixed { num_active: 1 };
        let mut moh = MoHRgLru::new(12, 3, &cfg);
        let input = Array2::<f32>::zeros((0, 12));
        let mut output = Array2::zeros((0, 12));

        let result = moh.forward_into(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(output.nrows(), 0);
    }

    #[test]
    fn test_moh_rg_lru_forward_into_large_batch() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        let cfg = HeadSelectionStrategy::Fixed { num_active: 3 };
        let mut moh = MoHRgLru::new(32, 4, &cfg);

        let mut rng = get_rng();
        let dist = Normal::new(0.0, 0.1).unwrap();
        let input_data: Vec<f32> = (0..256 * 32).map(|_| dist.sample(&mut rng)).collect();
        let input = Array2::from_shape_vec((256, 32), input_data).unwrap();

        let mut output = Array2::zeros((256, 32));
        let result = moh.forward_into(&input, &mut output);
        assert!(result.is_ok());
        assert_eq!(output.dim(), (256, 32));

        // Check that output is not all zeros
        let has_nonzero = output.iter().any(|&x| x.abs() > 1e-6);
        assert!(has_nonzero, "Output should contain non-zero values");
    }

    #[test]
    fn test_rg_lru_set_compute_backend_checked_cpu() {
        let mut rg = RgLru::new(8);
        rg.set_compute_backend_checked(ComputeBackend::Cpu)
            .expect("CPU backend should always be accepted");
        assert_eq!(rg.compute_backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_rg_lru_set_compute_backend_checked_gpu_is_strict_validation() {
        let mut rg = RgLru::new(8);
        match rg.set_compute_backend_checked(ComputeBackend::Vulkan) {
            Ok(()) => assert!(rg.compute_backend().is_gpu()),
            Err(err) => {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("gpu")
                        || msg.contains("vulkan")
                        || msg.contains("feature")
                        || msg.contains("backend")
                        || msg.contains("unavailable"),
                    "expected strict GPU validation error, got: {}",
                    msg
                );
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_rg_lru_reapply_gpu_backend_preserves_cached_ssm_backend() {
        use std::sync::Arc;

        let backend =
            match crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu() {
                Ok(backend) => backend,
                Err(_) => return,
            };

        let mut rg = RgLru::new(8);
        rg.set_compute_backend_checked(backend)
            .expect("resolved GPU backend should be accepted");
        let first_backend = rg
            .ensure_ssm_gpu_backend(4)
            .expect("should initialize cached SSM GPU backend");

        rg.set_compute_backend_checked(backend)
            .expect("re-applying same backend should be idempotent");
        let second_backend = rg
            .ensure_ssm_gpu_backend(4)
            .expect("cached SSM GPU backend should still be available");

        assert!(Arc::ptr_eq(&first_backend, &second_backend));
    }

    #[test]
    fn test_moh_rg_lru_set_compute_backend_checked_gpu_is_strict_validation() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHRgLru::new(12, 3, &cfg);
        match moh.set_compute_backend_checked(ComputeBackend::Vulkan) {
            Ok(()) => {
                assert!(moh.heads.iter().all(|h| h.compute_backend().is_gpu()));
            }
            Err(err) => {
                let msg = err.to_string().to_ascii_lowercase();
                assert!(
                    msg.contains("gpu")
                        || msg.contains("vulkan")
                        || msg.contains("feature")
                        || msg.contains("backend")
                        || msg.contains("unavailable"),
                    "expected strict GPU validation error, got: {}",
                    msg
                );
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_moh_rg_lru_reapply_gpu_backend_preserves_head_cached_ssm_backend() {
        use crate::domain::mixtures::HeadSelectionStrategy;
        use std::sync::Arc;

        let backend =
            match crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu() {
                Ok(backend) => backend,
                Err(_) => return,
            };

        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHRgLru::new(12, 3, &cfg);
        moh.set_compute_backend_checked(backend)
            .expect("resolved GPU backend should be accepted");
        let first_backend = moh.heads[0]
            .ensure_ssm_gpu_backend(4)
            .expect("head should initialize cached SSM GPU backend");

        moh.set_compute_backend_checked(backend)
            .expect("re-applying same backend should be idempotent");
        let second_backend = moh.heads[0]
            .ensure_ssm_gpu_backend(4)
            .expect("head cached SSM GPU backend should still be available");

        assert!(Arc::ptr_eq(&first_backend, &second_backend));
    }

    #[test]
    fn test_rg_lru_forward_gpu_requires_gpu_backend_selection() {
        let mut rg = RgLru::new(8);
        let input = Array2::<f32>::zeros((2, 8));
        let result = rg.forward_gpu(&input);
        assert!(
            result.is_err(),
            "forward_gpu without GPU backend selection must error"
        );
        let msg = result
            .err()
            .expect("error expected")
            .to_string()
            .to_ascii_lowercase();
        assert!(
            msg.contains("gpu backend selected")
                || msg.contains("requires gpu features")
                || msg.contains("gpu"),
            "expected strict GPU-forward validation error, got: {}",
            msg
        );
    }
}
