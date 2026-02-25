use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

use super::mamba::{AMatrixType, Mamba};
use crate::domain::{
    compute_backend::{ComputeBackend, resolve_compute_backend_strict_auto_gpu},
    mixtures::{HeadSelectionStrategy, MoHGating, moh_gating::MoHStreamingWorkspace},
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

#[derive(Debug, Clone)]
pub struct MoHMamba2StreamingWorkspace {
    pub moh: MoHStreamingWorkspace,
    pub head_out_buffer: Array1<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoHMamba2 {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    #[serde(flatten)]
    pub moh: MoHGating,

    pub heads: Vec<Mamba2>,

    #[serde(skip, default)]
    pub streaming_workspace: Option<Box<MoHMamba2StreamingWorkspace>>,

    #[serde(default)]
    compute_backend: ComputeBackend,

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[serde(skip_serializing, skip_deserializing, default)]
    head_gpu_backends: Option<
        Vec<
            std::sync::Arc<
                std::sync::Mutex<
                    crate::domain::layers::components::gpu_backend_variants::SsmGpuBackend,
                >,
            >,
        >,
    >,

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
        let mut inner = Mamba::deserialize(deserializer)?;
        inner.set_a_matrix_type(AMatrixType::BlockDiagonal);
        Ok(Self { inner })
    }
}

impl Mamba2 {
    pub fn new(embed_dim: usize) -> Self {
        Self::new_with_kernel(embed_dim, 8)
    }

    pub fn new_with_kernel(embed_dim: usize, conv_kernel: usize) -> Self {
        let mut inner = Mamba::new_with_kernel(embed_dim, conv_kernel);
        inner.set_a_matrix_type(AMatrixType::BlockDiagonal);
        Self { inner }
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        self.inner.forward_step(input)
    }

    pub fn forward_step_into(&mut self, input: &ArrayView1<f32>, output: &mut Array1<f32>) {
        self.inner.forward_step_into(input, output);
    }

    /// GPU-accelerated forward pass for Mamba2 (Phase 5.6.4)
    ///
    /// Delegates to inner Mamba GPU implementation.
    /// Target: 20x speedup on multi-head selective scan.
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
    ) -> crate::common::errors::Result<Array2<f32>> {
        self.inner.forward_gpu(input)
    }

    /// GPU-aware backward gradients for Mamba2.
    ///
    /// This path is strict: it errors when GPU backend is not selected.
    #[inline]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> crate::common::errors::Result<(Array2<f32>, Vec<Array2<f32>>)> {
        self.inner.compute_gradients_gpu(input, output_grads)
    }

    /// Set runtime compute backend.
    #[inline]
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.inner.set_compute_backend(compute_backend);
    }

    /// Set runtime compute backend with strict validation.
    #[inline]
    pub fn set_compute_backend_checked(
        &mut self,
        compute_backend: ComputeBackend,
    ) -> crate::common::errors::Result<()> {
        self.inner.set_compute_backend_checked(compute_backend)
    }

    /// Resolve and apply strict auto-GPU backend preference.
    #[inline]
    pub fn enable_gpu_auto_detect(&mut self) -> crate::common::errors::Result<()> {
        self.inner.enable_gpu_auto_detect()
    }

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.inner.compute_backend()
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
            heads.push(Mamba2::new(head_dim));
        }

        Self {
            embed_dim,
            num_heads: nh,
            head_dim,
            moh,
            heads,
            streaming_workspace: None,
            compute_backend: ComputeBackend::Cpu,
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            head_gpu_backends: None,
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

    /// Set runtime compute backend.
    #[inline]
    pub fn set_compute_backend(&mut self, compute_backend: ComputeBackend) {
        self.set_compute_backend_checked(compute_backend)
            .unwrap_or_else(|err| {
                panic!(
                    "Failed to set MoHMamba2 backend '{}': {}",
                    compute_backend.as_str(),
                    err
                )
            });
    }

    /// Set runtime compute backend with strict validation.
    #[inline]
    pub fn set_compute_backend_checked(
        &mut self,
        compute_backend: ComputeBackend,
    ) -> crate::common::errors::Result<()> {
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
                return Err(crate::common::errors::ModelError::Backend {
                    message: format!(
                        "MoHMamba2 requested GPU backend '{}' but this binary was built without GPU features.",
                        compute_backend.as_str()
                    ),
                });
            }
        }

        self.compute_backend = compute_backend;
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.head_gpu_backends = None;
        }
        Ok(())
    }

    /// Resolve and apply strict auto-GPU backend preference.
    #[inline]
    pub fn enable_gpu_auto_detect(&mut self) -> crate::common::errors::Result<()> {
        let backend = resolve_compute_backend_strict_auto_gpu()?;
        self.set_compute_backend_checked(backend)
    }

    /// Get runtime compute backend.
    #[inline]
    pub fn compute_backend(&self) -> ComputeBackend {
        self.compute_backend
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[inline]
    fn desired_state_dim_mamba2(head_dim: usize) -> usize {
        head_dim.clamp(16, 32)
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn ensure_head_gpu_backends(
        &mut self,
        seq_len: usize,
    ) -> crate::common::errors::Result<
        Vec<
            std::sync::Arc<
                std::sync::Mutex<
                    crate::domain::layers::components::gpu_backend_variants::SsmGpuBackend,
                >,
            >,
        >,
    > {
        use crate::domain::layers::components::{
            gpu_backend_variants::SsmGpuBackend, unified_gpu_kernels::SsmParams,
        };
        use std::sync::{Arc, Mutex};

        let state_dim = Self::desired_state_dim_mamba2(self.head_dim);
        let params = SsmParams::new(state_dim, self.head_dim, seq_len, 1);

        let needs_rebuild = self
            .head_gpu_backends
            .as_ref()
            .is_none_or(|backends| backends.len() != self.num_heads);
        if needs_rebuild {
            let mut backends = Vec::with_capacity(self.num_heads);
            for _ in 0..self.num_heads {
                let backend = SsmGpuBackend::mamba_with_backend(
                    state_dim,
                    self.head_dim,
                    seq_len,
                    1,
                    self.compute_backend,
                )?;
                backends.push(Arc::new(Mutex::new(backend)));
            }
            self.head_gpu_backends = Some(backends);
        }

        let backends = self.head_gpu_backends.as_ref().cloned().unwrap_or_default();
        for (h, backend) in backends.iter().enumerate().take(self.num_heads) {
            let mut locked =
                backend
                    .lock()
                    .map_err(|_| crate::common::errors::ModelError::Backend {
                        message: format!(
                            "Failed to acquire MoHMamba2 cached head GPU backend lock (head {})",
                            h
                        ),
                    })?;
            locked.set_params(params.clone());
            let (a, b, c, d, h_init) = self.heads[h]
                .inner
                .kernel_matrices_for_ssm_backend(state_dim)?;
            locked.set_mamba_kernel_matrices(a, b, c, d, h_init);
        }

        Ok(backends)
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

    pub fn ensure_streaming_workspace(&mut self) {
        if self.streaming_workspace.is_some() {
            return;
        }
        self.streaming_workspace = Some(Box::new(MoHMamba2StreamingWorkspace {
            moh: MoHStreamingWorkspace::default(),
            head_out_buffer: Array1::zeros(self.head_dim),
        }));
    }

    pub fn forward_step_into(&mut self, input: &ArrayView1<f32>, output: &mut Array1<f32>) {
        self.ensure_streaming_workspace();
        let ws = self.streaming_workspace.as_mut().unwrap();

        if ws.head_out_buffer.len() != self.head_dim {
            ws.head_out_buffer = Array1::zeros(self.head_dim);
        }

        let gate_input = self.moh.gate_input_view(input);
        self.moh.forward_weights_into(&gate_input, &mut ws.moh);

        output.fill(0.0);

        for (h, head) in self.heads.iter_mut().enumerate() {
            let weight = ws.moh.m[h];
            if weight > 0.0 {
                let s = h * self.head_dim;
                let e = s + self.head_dim;

                let input_view = input.slice(s![s..e]);

                // Run head into temp buffer
                head.forward_step_into(&input_view, &mut ws.head_out_buffer);

                let mut out_slice = output.slice_mut(s![s..e]);
                Zip::from(&mut out_slice)
                    .and(&ws.head_out_buffer)
                    .for_each(|o, &v| *o = v * weight);
            }
        }

        // Update activity stats from shared MoH streaming summarizer.
        let (active_heads, head_vec, token_vec) = MoHGating::summarize_streaming_weights(&ws.moh.m);
        self.last_avg_active_heads = Some(active_heads);
        self.last_head_activity_vec = Some(head_vec);
        self.last_token_head_activity_vec = Some(token_vec);
    }

    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        let mut output = Array1::zeros(input.raw_dim());
        self.forward_step_into(&input.view(), &mut output);
        output
    }

    /// GPU-accelerated forward pass for MoHMamba2.
    ///
    /// Uses per-head SSM GPU backends with strict no-fallback behavior.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn forward_gpu(
        &mut self,
        input: &Array2<f32>,
    ) -> crate::common::errors::Result<Array2<f32>> {
        if !self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: "MoHMamba2::forward_gpu called without a GPU backend selected. \
                          Call set_compute_backend_checked(...) with a GPU backend first."
                    .to_string(),
            });
        }

        let mut output = Array2::zeros(input.raw_dim());
        self.forward_into(input, &mut output)?;
        Ok(output)
    }

    /// GPU-aware backward gradients for MoH Mamba2.
    ///
    /// This path is strict: it errors when GPU backend is not selected.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> crate::common::errors::Result<(Array2<f32>, Vec<Array2<f32>>)> {
        if !self.compute_backend.is_gpu() {
            return Err(crate::common::errors::ModelError::Backend {
                message: "MoHMamba2::compute_gradients_gpu called without a GPU backend selected."
                    .to_string(),
            });
        }
        Ok(self.compute_gradients(input, output_grads))
    }

    /// GPU forward on non-GPU builds (strict no-fallback error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn forward_gpu(
        &mut self,
        _input: &Array2<f32>,
    ) -> crate::common::errors::Result<Array2<f32>> {
        Err(crate::common::errors::ModelError::Backend {
            message: "MoHMamba2 GPU forward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
        })
    }

    /// GPU-aware backward gradients on non-GPU builds (strict error).
    #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
    pub fn compute_gradients_gpu(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> crate::common::errors::Result<(Array2<f32>, Vec<Array2<f32>>)> {
        Err(crate::common::errors::ModelError::Backend {
            message: "MoHMamba2 GPU backward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
        })
    }

    pub fn set_verification_overrides(&mut self, overrides: Option<Vec<f64>>) {
        self.moh.set_verification_overrides(overrides);
    }

    /// Forward pass with in-place output (Zero Allocation Pattern).
    ///
    /// Computes MoH routing and per-head selective scan for Mamba2, writing results directly to output buffer.
    /// Eliminates intermediate allocations during head computation and aggregation.
    ///
    /// # Arguments
    /// * `input` - Input tensor (seq_len, embed_dim)
    /// * `output` - Pre-allocated output buffer (seq_len, embed_dim)
    ///
    /// # Returns
    /// `Ok(())` if successful, error if output buffer has incorrect dimensions
    pub fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> crate::common::errors::Result<()> {
        let (t, d) = input.dim();
        let use_gpu = self.compute_backend.is_gpu();

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

        if t == 0 || d == 0 || self.num_heads == 0 || self.head_dim == 0 {
            self.clear_caches();
            self.cached_input = Some(input.clone());
            output.fill(0.0);
            return Ok(());
        }

        self.cached_input = Some(input.clone());

        let input_view = input.view();
        let gate_input = self.moh.gate_input_view2(&input_view);
        let eff = self.moh.forward_weights_view(&gate_input, None, None);
        self.cached_eff = Some(eff.clone());

        output.fill(0.0);

        let head_outs: Vec<Array2<f32>> = if use_gpu {
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                let backend_name = self.compute_backend.as_str().to_string();
                let backends = self.ensure_head_gpu_backends(t)?;
                let mut outs = Vec::with_capacity(self.num_heads);

                for (h, backend) in backends.into_iter().enumerate() {
                    let c0 = h * self.head_dim;
                    let c1 = c0 + self.head_dim;
                    let x_view = input.slice(s![.., c0..c1]);
                    let x_owned = x_view.to_owned();

                    let mut locked =
                        backend
                            .lock()
                            .map_err(|_| crate::common::errors::ModelError::Backend {
                                message: "Failed to acquire MoHMamba2 cached head GPU backend lock"
                                    .to_string(),
                            })?;
                    let y_h = locked.forward(&x_owned).map_err(|err| {
                        crate::common::errors::ModelError::Backend {
                            message: format!(
                                "MoHMamba2 head {} GPU forward failed on backend '{}': {}",
                                h, backend_name, err
                            ),
                        }
                    })?;
                    outs.push(y_h);
                }

                outs
            }

            #[cfg(not(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal")))]
            {
                return Err(crate::common::errors::ModelError::Backend {
                    message: "MoHMamba2 GPU forward requires GPU features. Compile with --features gpu-wgpu, gpu-cuda, or gpu-metal.".to_string(),
                });
            }
        } else {
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

        Ok(())
    }
}

impl Layer for Mamba2 {
    fn layer_type(&self) -> &str {
        "Mamba2"
    }

    fn forward(&mut self, input: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
        if self.inner.compute_backend().is_gpu() {
            return self
                .forward_gpu(input)
                .unwrap_or_else(|err| panic!("Mamba2 GPU forward failed: {err}"));
        }
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
    ) -> crate::common::errors::Result<()> {
        self.inner.apply_gradients(gradients, learning_rate)
    }

    fn set_training_progress(&mut self, progress: f64) {
        self.inner.set_training_progress(progress);
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
        let mut output = Array2::zeros(input.raw_dim());
        self.forward_into(input, &mut output)
            .unwrap_or_else(|err| panic!("MoHMamba2 forward failed: {err}"));
        output
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
        let use_gpu = self.compute_backend.is_gpu();
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
                            if use_gpu {
                                let x_owned = x_view.to_owned();
                                head.forward_gpu(&x_owned).unwrap_or_else(|err| {
                                    panic!("MoHMamba2 head {h} GPU forward failed: {err}")
                                })
                            } else {
                                head.forward_view(&x_view)
                            }
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
                        if use_gpu {
                            let x_owned = x_view.to_owned();
                            head.forward_gpu(&x_owned).unwrap_or_else(|err| {
                                panic!("MoHMamba2 head {h} GPU forward failed: {err}")
                            })
                        } else {
                            head.forward_view(&x_view)
                        }
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

            let (dx_h, pgrads_h) = if use_gpu {
                let x_owned = x_view.to_owned();
                let mut head = self.heads[h].clone();
                head.forward_gpu(&x_owned).unwrap_or_else(|err| {
                    panic!("MoHMamba2 head {h} GPU forward failed during backward prep: {err}")
                });
                head.compute_gradients_gpu(&x_owned, &scaled_grads)
                    .unwrap_or_else(|err| panic!("MoHMamba2 head {h} GPU backward failed: {err}"))
            } else if can_use_cache {
                let scaled_grads_view = scaled_grads.view();
                self.heads[h].compute_gradients_view(&x_view, &scaled_grads_view)
            } else {
                let mut head = self.heads[h].clone();
                head.forward_view(&x_view);
                let scaled_grads_view = scaled_grads.view();
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
    ) -> crate::common::errors::Result<()> {
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

    fn set_training_progress(&mut self, progress: f64) {
        // Mamba2 itself doesn't use training progress yet, but maybe heads do?
        for head in &mut self.heads {
            head.set_training_progress(progress);
        }
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
        let grads = Array2::<f32>::ones((5, 12));
        layer.forward(&x);
        let dx = layer.backward(&grads, 0.01);
        assert_eq!(dx.dim(), (5, 12));
    }

    #[test]
    fn test_mamba2_forward_into_dimension_validation() {
        let mut mamba = Mamba2::new(16);
        let input = Array2::<f32>::from_elem((8, 16), 0.1);

        // Wrong dimensions should fail
        let mut output_wrong = Array2::zeros((7, 16));
        let result = mamba.inner.forward_into(&input, &mut output_wrong);
        assert!(result.is_err());

        // Correct dimensions should succeed
        let mut output_correct = Array2::zeros((8, 16));
        let result = mamba.inner.forward_into(&input, &mut output_correct);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mamba2_forward_into_basic() {
        let mut mamba = Mamba2::new(16);
        let input = Array2::<f32>::from_elem((8, 16), 0.1);

        // Should successfully compute and write to output
        let mut output = Array2::zeros((8, 16));
        let result = mamba.inner.forward_into(&input, &mut output);

        assert!(result.is_ok());
        assert_eq!(output.dim(), (8, 16));
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_moh_mamba2_forward_into_basic() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHMamba2::new(12, 3, &cfg);
        let input = Array2::<f32>::from_elem((6, 12), 0.1);

        // Should successfully compute and write to output
        let mut output = Array2::zeros((6, 12));
        let result = moh.forward_into(&input, &mut output);

        assert!(result.is_ok());
        assert_eq!(output.dim(), (6, 12));
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_moh_mamba2_forward_into_dimension_validation() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHMamba2::new(12, 3, &cfg);
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
    fn test_moh_mamba2_set_compute_backend_checked_cpu() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHMamba2::new(12, 3, &cfg);
        let result = moh.set_compute_backend_checked(ComputeBackend::Cpu);
        assert!(result.is_ok());
        assert_eq!(moh.compute_backend(), ComputeBackend::Cpu);
    }

    #[test]
    fn test_moh_mamba2_set_compute_backend_checked_gpu_is_strict_validation() {
        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHMamba2::new(12, 3, &cfg);
        let result = moh.set_compute_backend_checked(ComputeBackend::Vulkan);

        match result {
            Ok(()) => assert!(moh.compute_backend().is_gpu()),
            Err(err) => {
                let msg = format!("{}", err).to_lowercase();
                assert!(
                    msg.contains("without gpu features")
                        || msg.contains("unavailable")
                        || msg.contains("gpu")
                        || msg.contains("backend"),
                    "expected strict GPU validation error, got: {}",
                    err
                );
            }
        }
    }

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    #[test]
    fn test_moh_mamba2_reapply_gpu_backend_preserves_cached_head_backends() {
        use std::sync::Arc;

        let backend =
            match crate::domain::compute_backend::resolve_compute_backend_strict_auto_gpu() {
                Ok(backend) => backend,
                Err(_) => return,
            };

        let cfg = HeadSelectionStrategy::Fixed { num_active: 2 };
        let mut moh = MoHMamba2::new(12, 3, &cfg);
        moh.set_compute_backend_checked(backend)
            .expect("resolved GPU backend should be accepted");
        let first_backends = moh
            .ensure_head_gpu_backends(6)
            .expect("should initialize cached head GPU backends");

        moh.set_compute_backend_checked(backend)
            .expect("re-applying same backend should be idempotent");
        let second_backends = moh
            .ensure_head_gpu_backends(6)
            .expect("cached head GPU backends should still be available");

        assert_eq!(first_backends.len(), second_backends.len());
        for (first, second) in first_backends.iter().zip(second_backends.iter()) {
            assert!(Arc::ptr_eq(first, second));
        }
    }
}
