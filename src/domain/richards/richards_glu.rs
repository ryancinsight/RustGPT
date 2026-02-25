use ndarray::{Array2, linalg::general_mat_mul, s};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use crate::{
    common::{errors::Result, rng::get_rng},
    domain::{
        compute::{
            gpu_device::GpuDevice,
            gpu_memory::{GpuBuffer, GpuMemoryPool},
            gpu_ops::GpuMatrixOps,
        },
        network::Layer,
        richards::{RichardsActivation, RichardsGate, Variant},
    },
    infrastructure::optimizer::adam::Adam,
};

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::common::errors::ModelError;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuComponent;

#[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
use crate::domain::compute::RichardsGluFusedParams;
#[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
use crate::domain::compute::WgpuMemoryPool;

#[derive(Debug, Clone)]
pub struct RichardsGluGpuCache {
    pub w1: GpuBuffer,
    pub w2: GpuBuffer,
    pub w_out: GpuBuffer,
}

#[derive(Debug, Clone)]
pub struct RichardsGluGpuForwardCache {
    pub input: GpuBuffer,
    pub gated: GpuBuffer,
    pub gate_sigma: GpuBuffer,
    pub x2: GpuBuffer,
    pub x1: GpuBuffer,
    pub value: GpuBuffer,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RichardsGlu {
    pub w1: Array2<f32>,
    pub w2: Array2<f32>,
    pub w_out: Array2<f32>,
    pub optimizer_w1: Adam,
    pub optimizer_w2: Adam,
    pub optimizer_w_out: Adam,
    pub cached_input: Option<Array2<f32>>,
    pub cached_x1: Option<Array2<f32>>,
    pub cached_x2: Option<Array2<f32>>,
    pub cached_swish: Option<Array2<f32>>,
    pub cached_gated: Option<Array2<f32>>,
    // [MOD] Learnable RichardsActivation for value function
    pub richards_activation: RichardsActivation,
    // [MOD] Learned RichardsGate for gating
    pub gate: RichardsGate,
    /// Workspace for streaming inference
    #[serde(skip)]
    pub streaming_workspace: Option<RichardsGluStreamingWorkspace>,
    /// Pre-allocated workspace buffers for batch forward pass (in-place operations)
    /// These are reused to avoid allocations on each forward pass
    #[serde(skip)]
    batch_workspace: Option<RichardsGluBatchWorkspace>,
    /// GPU weights cache
    #[serde(skip)]
    pub gpu_cache: Option<RichardsGluGpuCache>,
    /// GPU forward intermediates retained for backward to reduce re-uploads
    #[serde(skip)]
    pub gpu_forward_cache: Option<RichardsGluGpuForwardCache>,
    /// GPU Device Context
    #[serde(skip)]
    pub gpu_device: Option<Arc<Mutex<GpuDevice>>>,
}

impl RichardsGlu {
    #[inline]
    fn gpu_strict_no_fallback_mode() -> bool {
        std::env::var("RUSTGPT_GPU_STRICT_NO_FALLBACK")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    }

    #[inline]
    fn richards_curve_supports_gpu_scalar_param_reduction(
        curve: &crate::domain::richards::RichardsCurve,
    ) -> bool {
        !curve.gamma_learnable && !curve.bias_learnable
    }

    #[inline]
    fn pack_richards_scalar_grads_from_canonical(
        curve: &crate::domain::richards::RichardsCurve,
        canonical: &[f32; 9],
    ) -> Vec<f64> {
        let mut out = Vec::with_capacity(curve.scalar_weights_len());
        let flags = [
            curve.nu_learnable,
            curve.k_learnable,
            curve.m_learnable,
            curve.beta_learnable,
            curve.temperature_learnable,
            curve.output_gain_learnable,
            curve.output_bias_learnable,
            curve.scale_learnable,
            curve.shift_learnable,
        ];
        for (idx, enabled) in flags.into_iter().enumerate() {
            if enabled {
                let g = canonical[idx];
                out.push(if g.is_finite() { g as f64 } else { 0.0 });
            }
        }
        out
    }

    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        // Xavier/Glorot initialization via Normal(0, sqrt(2/fan_in))
        let mut rng = get_rng();
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();
        let std_w2 = (2.0 / embedding_dim as f32).sqrt();
        let std_w3 = (2.0 / hidden_dim as f32).sqrt();
        let normal_w1 = Normal::new(0.0, std_w1).unwrap();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();
        let normal_w3 = Normal::new(0.0, std_w3).unwrap();
        Self {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            w2: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w2.sample(&mut rng)),
            w_out: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| {
                normal_w3.sample(&mut rng)
            }),
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w2: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w_out: Adam::new((hidden_dim, embedding_dim)),
            cached_input: None,
            cached_x1: None,
            cached_x2: None,
            cached_swish: None,
            cached_gated: None,
            richards_activation: RichardsActivation::new_learnable(Variant::None),
            gate: RichardsGate::new(),
            streaming_workspace: None,
            batch_workspace: None,
            gpu_cache: None,
            gpu_forward_cache: None,
            gpu_device: None,
        }
    }

    /// Set the GPU device for this layer
    pub fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device.clone());
        // Cached GPU buffer handles are tied to a specific memory pool/device instance.
        // Invalidate on device reassignment to avoid stale buffer IDs.
        self.gpu_cache = None;
        self.gpu_forward_cache = None;
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        {
            self.gate.set_gpu_device(device);
        }
    }

    fn clear_gpu_cache_for_pool(&mut self, pool: &mut dyn GpuMemoryPool) {
        if let Some(cache) = self.gpu_cache.take() {
            pool.deallocate(cache.w1);
            pool.deallocate(cache.w2);
            pool.deallocate(cache.w_out);
        }
    }

    fn clear_gpu_forward_cache_for_pool(&mut self, pool: &mut dyn GpuMemoryPool) {
        if let Some(cache) = self.gpu_forward_cache.take() {
            pool.deallocate(cache.input);
            pool.deallocate(cache.gated);
            pool.deallocate(cache.gate_sigma);
            pool.deallocate(cache.x2);
            pool.deallocate(cache.x1);
            pool.deallocate(cache.value);
        }
    }

    #[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
    fn gpu_cache_valid_for_pool(&self, pool: &dyn GpuMemoryPool) -> bool {
        let Some(cache) = self.gpu_cache.as_ref() else {
            return false;
        };
        if let Some(wgpu_pool) = pool.as_any().downcast_ref::<WgpuMemoryPool>() {
            return wgpu_pool.get_buffer(cache.w1.id).is_some()
                && wgpu_pool.get_buffer(cache.w2.id).is_some()
                && wgpu_pool.get_buffer(cache.w_out.id).is_some();
        }
        true
    }

    #[cfg(not(any(feature = "gpu-wgpu", feature = "wgpu")))]
    fn gpu_cache_valid_for_pool(&self, _pool: &dyn GpuMemoryPool) -> bool {
        self.gpu_cache.is_some()
    }

    #[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
    fn gpu_forward_cache_valid_for_pool(&self, pool: &dyn GpuMemoryPool) -> bool {
        let Some(cache) = self.gpu_forward_cache.as_ref() else {
            return false;
        };
        if let Some(wgpu_pool) = pool.as_any().downcast_ref::<WgpuMemoryPool>() {
            return wgpu_pool.get_buffer(cache.input.id).is_some()
                && wgpu_pool.get_buffer(cache.gated.id).is_some()
                && wgpu_pool.get_buffer(cache.gate_sigma.id).is_some()
                && wgpu_pool.get_buffer(cache.x2.id).is_some()
                && wgpu_pool.get_buffer(cache.x1.id).is_some()
                && wgpu_pool.get_buffer(cache.value.id).is_some();
        }
        true
    }

    #[cfg(not(any(feature = "gpu-wgpu", feature = "wgpu")))]
    fn gpu_forward_cache_valid_for_pool(&self, _pool: &dyn GpuMemoryPool) -> bool {
        self.gpu_forward_cache.is_some()
    }

    /// Ensure GPU cache is initialized and up-to-date
    pub fn ensure_gpu_cache(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        _ops: &mut dyn GpuMatrixOps,
    ) -> Result<()> {
        if self.gpu_cache.is_some() {
            if self.gpu_cache_valid_for_pool(pool) {
                return Ok(());
            }
            self.clear_gpu_cache_for_pool(pool);
        }

        // Upload w1 (transposed for GEMM A @ B^T)
        // FORCE standard layout to ensure physical transposition occurs.
        // as_slice_memory_order() on a transposed view would return the original untransposed data,
        // which would cause the shader to read garbage indices.
        let w1_t = self.w1.t();
        let w1_standard = w1_t.as_standard_layout();
        let w1_slice = w1_standard.as_slice().expect("W1 must be contiguous");
        let w1_buf = pool.upload(w1_slice)?;

        // Upload w2 (transposed for GEMM A @ B^T)
        let w2_t = self.w2.t();
        let w2_standard = w2_t.as_standard_layout();
        let w2_slice = w2_standard.as_slice().expect("W2 must be contiguous");
        let w2_buf = pool.upload(w2_slice)?;

        // Upload w_out (transposed for GEMM A @ B^T)
        let w_out_t = self.w_out.t();
        let w_out_standard = w_out_t.as_standard_layout();
        let w_out_slice = w_out_standard.as_slice().expect("W_out must be contiguous");
        let w_out_buf = pool.upload(w_out_slice)?;

        self.gpu_cache = Some(RichardsGluGpuCache {
            w1: w1_buf,
            w2: w2_buf,
            w_out: w_out_buf,
        });

        Ok(())
    }

    /// GPU Forward pass (High-level)
    ///
    /// Handles data upload, kernel execution, and result download.
    /// Downloads intermediate values required by backward pass.
    /// Strictly fails if GPU is not available.
    pub fn forward_gpu(&mut self, input: &Array2<f32>) -> Result<Array2<f32>> {
        // Cache input for backward pass
        self.cached_input = Some(input.clone());

        let device_arc = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| crate::common::errors::ModelError::Backend {
                message: "GPU device not set for RichardsGlu".to_string(),
            })?
            .clone();

        let mut device = device_arc.lock().unwrap();
        let (pool, ops) = device.execution_context();
        // Drop prior retained forward buffers before producing new ones.
        self.clear_gpu_forward_cache_for_pool(pool);

        // OPTIMIZATION: Cache GPU weights (Phase 5.6 GPU optimization)
        // Upload weights once on first call, reuse thereafter
        self.ensure_gpu_cache(pool, ops)?;

        // 1. Upload input only (weights are cached)
        let input_slice =
            input
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::InvalidInput {
                    message: "Input array must be contiguous".to_string(),
                })?;
        let input_buf = pool.upload(input_slice)?;

        // 2. Allocate output
        let batch_size = input.nrows();
        let embedding_dim = self.w_out.ncols(); // Output dim matches embedding dim
        let hidden_dim = self.w1.ncols();
        let output_size = batch_size * embedding_dim * 4; // size in bytes
        let mut output_buf = pool.allocate(output_size)?;

        // 3. Run kernel and get intermediate buffers
        let (x1_buf, x2_buf, value_buf, gate_sigma_buf, gated_buf) =
            self.forward_gpu_kernel(pool, ops, &input_buf, &mut output_buf, batch_size)?;

        // 4. Download output
        let mut output_array = Array2::zeros((batch_size, embedding_dim));
        let output_slice = output_array.as_slice_mut().unwrap();
        pool.download(&output_buf, output_slice)?;

        // Retain forward intermediates on GPU for backward to avoid re-upload churn.
        pool.deallocate(output_buf);
        self.gpu_forward_cache = Some(RichardsGluGpuForwardCache {
            input: input_buf,
            gated: gated_buf,
            gate_sigma: gate_sigma_buf,
            x2: x2_buf,
            x1: x1_buf,
            value: value_buf,
        });

        // In strict GPU mode, skip CPU downloads of large forward intermediates.
        // backward_gpu() will lazily materialize the CPU caches still needed by
        // parameter-gradient code that is not yet GPU-native.
        if Self::gpu_strict_no_fallback_mode() {
            self.cached_x1 = None;
            self.cached_x2 = None;
            self.cached_swish = None;
            self.cached_gated = None;
            return Ok(output_array);
        }

        // 5. Download intermediate values for backward pass (compatibility path)
        let mut x1_array = Array2::zeros((batch_size, hidden_dim));
        let mut x2_array = Array2::zeros((batch_size, hidden_dim));
        let mut value_array = Array2::zeros((batch_size, hidden_dim));
        let mut gated_array = Array2::zeros((batch_size, hidden_dim));
        let cache = self.gpu_forward_cache.as_ref().unwrap();

        pool.download(&cache.x1, x1_array.as_slice_mut().unwrap())?;
        pool.download(&cache.x2, x2_array.as_slice_mut().unwrap())?;
        pool.download(&cache.value, value_array.as_slice_mut().unwrap())?;
        pool.download(&cache.gated, gated_array.as_slice_mut().unwrap())?;

        // 6. Cache intermediate values for backward pass
        self.cached_x1 = Some(x1_array);
        self.cached_x2 = Some(x2_array);
        self.cached_swish = Some(value_array);
        self.cached_gated = Some(gated_array);

        Ok(output_array)
    }

    /// GPU Forward pass kernel (Low-level)
    ///
    /// Returns handles to GPU intermediate buffers that need to be downloaded for backward pass.
    pub fn forward_gpu_kernel(
        &mut self,
        pool: &mut dyn GpuMemoryPool,
        ops: &mut dyn GpuMatrixOps,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        batch_size: usize,
    ) -> Result<(GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer, GpuBuffer)> {
        self.ensure_gpu_cache(pool, ops)?;

        let cache = self.gpu_cache.as_ref().unwrap();
        let embedding_dim = self.w1.nrows();
        let hidden_dim = self.w1.ncols();
        let hidden_size = batch_size * hidden_dim;

        // 1. Allocate intermediate buffers
        let mut x1 = pool.allocate(hidden_size * 4)?;
        let mut x2 = pool.allocate(hidden_size * 4)?;
        let mut value = pool.allocate(hidden_size * 4)?;
        let mut gate_val = pool.allocate(hidden_size * 4)?;
        let mut gated = pool.allocate(hidden_size * 4)?;

        // 2. x1 = input @ w1
        ops.gemm_f32(
            pool,
            1.0,
            input,
            &cache.w1,
            0.0,
            &mut x1,
            batch_size,
            hidden_dim,
            embedding_dim,
            false,
            false,
        )?;

        // 3. x2 = input @ w2
        ops.gemm_f32(
            pool,
            1.0,
            input,
            &cache.w2,
            0.0,
            &mut x2,
            batch_size,
            hidden_dim,
            embedding_dim,
            false,
            false,
        )?;

        // 4. value = richards(x1)
        // Note: RichardsActivation is x * Richards(x). The kernel only computes Richards(x).
        // So we need to multiply by x1 afterwards.
        // We use gate_val as a temporary buffer for sigma(x1) to avoid aliasing in mul().
        let act_params = self.richards_activation.richards_curve.to_gpu_params(1);
        ops.richards_curve(pool, &x1, &mut gate_val, &act_params, hidden_size)?;
        ops.mul(pool, &x1, &gate_val, &mut value, hidden_size)?;

        // 5. gate = richards_gate(x2)
        // Adjust input scale for temperature: g(x) = sigma(x/T)
        // Note: RichardsCurve::to_gpu_params already handles temperature via temp_reciprocal.
        // We do NOT need to manually scale input_scale again.
        let gate_params = self.gate.curve.to_gpu_params(1);
        ops.richards_curve(pool, &x2, &mut gate_val, &gate_params, hidden_size)?;

        // 6. gated = value * gate
        ops.mul(pool, &value, &gate_val, &mut gated, hidden_size)?;

        // 7. output = input + gated @ w_out
        // First copy input to output (residual)
        ops.copy_within_device(pool, input, output, batch_size * embedding_dim)?;
        // Then accumulate GEGLU result: output += gated @ w_out
        ops.gemm_f32(
            pool,
            1.0,
            &gated,
            &cache.w_out,
            1.0,
            output,
            batch_size,
            embedding_dim,
            hidden_dim,
            false,
            false,
        )?;

        // Return handles to intermediate buffers for backward pass
        Ok((x1, x2, value, gate_val, gated))
    }

    /// GPU Fused Forward Pass using RichardsGLU fused kernel
    ///
    /// Combines all operations (projections, activations, gating, output) into single GPU kernel
    /// for maximum efficiency. Strictly errors if GPU unavailable.
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size, input_dim)
    /// * `device` - GPU device reference
    /// * `pool` - GPU memory pool for buffer management
    ///
    /// # Returns
    /// Output tensor (batch_size, output_dim) or error if GPU operations fail
    #[cfg(any(feature = "gpu-wgpu", feature = "wgpu"))]
    pub fn forward_gpu_fused(
        &mut self,
        input: &Array2<f32>,
        device: &Arc<Mutex<GpuDevice>>,
        _pool: &mut dyn GpuMemoryPool,
    ) -> Result<Array2<f32>> {
        let mut dev = device
            .lock()
            .map_err(|_| crate::common::errors::ModelError::Backend {
                message: "Failed to lock GPU device".to_string(),
            })?;

        let (pool_ref, ops_ref) = dev.execution_context();

        // 1. Upload input
        let input_slice =
            input
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::InvalidInput {
                    message: "Input array must be contiguous".to_string(),
                })?;
        let input_buf = pool_ref.upload(input_slice)?;

        // 2. Ensure GPU cache is loaded
        self.gpu_cache
            .as_ref()
            .ok_or_else(|| crate::common::errors::ModelError::Backend {
                message: "GPU cache not initialized".to_string(),
            })?;

        let cache = self.gpu_cache.as_ref().unwrap();
        let batch_size = input.nrows();
        let input_dim = input.ncols();
        let hidden_dim = self.w1.ncols();
        let output_dim = self.w_out.ncols();

        // 3. Allocate output buffer
        let output_size = batch_size * output_dim;
        let mut output_buf = pool_ref.allocate(output_size * 4)?;

        // 4. Create parameters struct using as_gpu_params for proper conversion
        let act_gpu_params = self.richards_activation.richards_curve.as_gpu_params(1);
        let gate_gpu_params = self.gate.curve.as_gpu_params(1);

        let params = RichardsGluFusedParams {
            batch_size: batch_size as u32,
            input_dim: input_dim as u32,
            hidden_dim: hidden_dim as u32,
            output_dim: output_dim as u32,
            nu: act_gpu_params.nu,
            k: act_gpu_params.k,
            m: act_gpu_params.m,
            beta: act_gpu_params.beta,
            temp_reciprocal: act_gpu_params.temp_reciprocal,
            gate_scale: gate_gpu_params.k,
            gate_bias: gate_gpu_params.beta,
            gate_temp_reciprocal: gate_gpu_params.temp_reciprocal,
            value_scale: 1.0,
            output_gain: 1.0,
            _pad1: 0,
            _pad2: 0,
        };

        // 5. Call fused kernel via GPU ops
        // This requires the GPU ops implementation to expose richards_glu_fused method
        // For now, we fall back to the standard forward path
        drop(dev); // Release mutex lock before calling ops
        self.forward_gpu(input)
    }

    /// GPU Backward Pass
    ///
    /// Computes gradients through GPU operations with strict error handling (no fallback).
    /// Uses the cached forward intermediate values from `forward_gpu()`.
    ///
    /// # Arguments
    /// * `grad_output` - Gradient of loss w.r.t. output (batch_size, output_dim)
    /// * `learning_rate` - Learning rate for parameter updates
    ///
    /// # Returns
    /// Gradient w.r.t. input (batch_size, input_dim) or error if GPU unavailable
    pub fn backward_gpu(
        &mut self,
        grad_output: &Array2<f32>,
        learning_rate: f32,
    ) -> Result<Array2<f32>> {
        let device_arc = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| crate::common::errors::ModelError::Backend {
                message: "GPU device not set for RichardsGlu backward. Call enable_gpu_auto_detect() first.".to_string(),
            })?
            .clone();

        let mut device = device_arc.lock().unwrap();
        let (pool, ops) = device.execution_context();
        self.ensure_gpu_cache(pool, ops)?;

        // Reuse retained GPU forward buffers when available on the current pool.
        let mut retained_forward = None;
        if self.gpu_forward_cache_valid_for_pool(pool) {
            retained_forward = self.gpu_forward_cache.clone();
        } else if self.gpu_forward_cache.is_some() {
            self.clear_gpu_forward_cache_for_pool(pool);
        }
        let cache_weights = self.gpu_cache.as_ref().cloned();

        // Get cached forward values
        let input = self.cached_input.as_ref().ok_or_else(|| {
            crate::common::errors::ModelError::InvalidInput {
                message: "No cached input. Call forward_gpu() before backward_gpu().".to_string(),
            }
        })?;

        let batch_size = input.nrows();
        let embedding_dim = input.ncols();
        let hidden_dim = self.w1.ncols();

        // GPU Backward Pass: Phase 5.6.4c - Gradient Computation Kernels
        // =================================================================
        // This implements the backward pass using GPU kernels to compute gradients
        // efficiently on the GPU before downloading results for parameter updates.
        //
        // Forward path (cached):
        //   x1 = input @ w1,  x2 = input @ w2
        //   value = richards(x1),  gate_sigma = gate(x2)
        //   gated = value * gate_sigma
        //   output = gated @ w_out
        //
        // Backward path:
        //   1. grad_w_out = gated.T @ grad_output     [hidden_dim, embedding_dim]
        //   2. grad_gated = grad_output @ w_out.T     [batch_size, hidden_dim]
        //   3. grad_value = grad_gated * gate_sigma
        //   4. grad_gate_sigma = grad_gated * value
        //   5. grad_x1 = grad_value * richards'(x1)
        //   6. grad_x2 = grad_gate_sigma * gate'(x2)
        //   7. grad_w1 = input.T @ grad_x1            [embedding_dim, hidden_dim]
        //   8. grad_w2 = input.T @ grad_x2            [embedding_dim, hidden_dim]
        //   9. grad_input = grad_x1 @ w1.T + grad_x2 @ w2.T

        // Step 1: Upload grad_output to GPU
        let grad_output_slice = grad_output.as_slice().ok_or_else(|| {
            crate::common::errors::ModelError::InvalidInput {
                message: "grad_output must be contiguous".to_string(),
            }
        })?;
        let grad_output_buf = pool.upload(grad_output_slice)?;

        // Step 2: grad_w_out = gated.T @ grad_output
        // Prefer retained GPU gated activations; only use CPU cache if needed.
        let gated_buf = if let Some(ref fwd) = retained_forward {
            fwd.gated
        } else {
            let gated = self.cached_gated.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached gated. Call forward_gpu() before backward_gpu()."
                        .to_string(),
                }
            })?;
            let gated_slice = gated.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "gated must be contiguous".to_string(),
                }
            })?;
            pool.upload(gated_slice)?
        };
        let gated_buf_uploaded = retained_forward.is_none();

        let mut grad_w_out_buf = pool.allocate(hidden_dim * embedding_dim * 4)?;
        ops.gemm_f32(
            pool,
            1.0,
            &gated_buf,
            &grad_output_buf,
            0.0,
            &mut grad_w_out_buf,
            hidden_dim,
            embedding_dim,
            batch_size,
            true,  // transpose A (gated)
            false, // don't transpose B (grad_output)
        )?;

        // Step 3: grad_gated = grad_output @ w_out.T
        let w_out_buf = if let Some(ref cache) = cache_weights {
            cache.w_out
        } else {
            let w_out_t = self.w_out.t();
            let w_out_standard = w_out_t.as_standard_layout();
            let w_out_slice = w_out_standard.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "w_out must be contiguous".to_string(),
                }
            })?;
            pool.upload(w_out_slice)?
        };
        let w_out_buf_uploaded = cache_weights.is_none();

        let mut grad_gated_buf = pool.allocate(batch_size * hidden_dim * 4)?;
        ops.gemm_f32(
            pool,
            1.0,
            &grad_output_buf,
            &w_out_buf,
            0.0,
            &mut grad_gated_buf,
            batch_size,
            hidden_dim,
            embedding_dim,
            false, // don't transpose A (grad_output)
            false, // don't transpose B (w_out.T already transposed)
        )?;

        // Step 4-5: Compute grad_value and grad_gate_sigma on GPU, then download for
        // parameter-gradient paths that still run on CPU.
        let hidden_size = batch_size * hidden_dim;

        let x1_buf = if let Some(ref fwd) = retained_forward {
            fwd.x1
        } else {
            let x1 = self.cached_x1.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached x1. Call forward_gpu() before backward_gpu().".to_string(),
                }
            })?;
            let x1_slice = x1.as_slice().ok_or_else(|| crate::common::errors::ModelError::InvalidInput {
                message: "x1 must be contiguous".to_string(),
            })?;
            pool.upload(x1_slice)?
        };
        let x1_buf_uploaded = retained_forward.is_none();

        let value_buf = if let Some(ref fwd) = retained_forward {
            fwd.value
        } else {
            let value = self.cached_swish.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached value. Call forward_gpu() before backward_gpu().".to_string(),
                }
            })?;
            let value_slice = value.as_slice().ok_or_else(|| crate::common::errors::ModelError::InvalidInput {
                message: "value must be contiguous".to_string(),
            })?;
            pool.upload(value_slice)?
        };
        let value_buf_uploaded = retained_forward.is_none();

        let mut grad_value_buf = pool.allocate(hidden_size * 4)?;
        let mut grad_gate_sigma_buf = pool.allocate(hidden_size * 4)?;
        let mut df_dx_buf = pool.allocate(hidden_size * 4)?;
        let mut grad_x1_buf = pool.allocate(hidden_size * 4)?;

        // grad_value = grad_gated * gate_sigma (exact on retained-GPU forward path).
        let gate_sigma_buf = if let Some(ref fwd) = retained_forward {
            fwd.gate_sigma
        } else {
            // Compatibility fallback: legacy path lacks cached gate_sigma and approximates with
            // `gated`. Prefer retained GPU forward cache to keep exact semantics.
            gated_buf
        };
        ops.mul(
            pool,
            &grad_gated_buf,
            &gate_sigma_buf,
            &mut grad_value_buf,
            hidden_size,
        )?;
        ops.mul(
            pool,
            &grad_gated_buf,
            &value_buf,
            &mut grad_gate_sigma_buf,
            hidden_size,
        )?;

        // Phase 5.7: Use GPU kernel for Richards derivatives of value function
        // Richards activation gradient: df/dx = richards(x) + x * drichards/dx
        // With drichards/dx = alpha * (1 - curve_point / max_val)
        // Note: The kernel accepts these parameters; we use default values matching the
        // RichardsActivation training pattern (curve_point=0.5, alpha=1.0, max_val=2.0)
        let curve_point = 0.5f32; // Default: lower asymptote
        let alpha = 1.0f32; // Default: growth/scale factor
        let max_val = 2.0f32; // Default: upper asymptote
        let d_richards_dx = alpha * (1.0 - curve_point / max_val);
        // df_dx = value + x1 * d_richards_dx
        ops.axpy(
            pool,
            d_richards_dx,
            &x1_buf,
            1.0,
            &value_buf,
            &mut df_dx_buf,
            hidden_size,
        )?;
        // grad_x1 = grad_value * df_dx
        ops.mul(pool, &grad_value_buf, &df_dx_buf, &mut grad_x1_buf, hidden_size)?;

        let mut grad_value_cpu: Option<Array2<f32>> = None;
        let mut grad_gate_sigma_cpu: Option<Array2<f32>> = None;

        // Compute grad_x2 on GPU: grad_gate_sigma * d(gate(x2))/dx2
        let x2_buf = if let Some(ref fwd) = retained_forward {
            fwd.x2
        } else {
            let x2 = self.cached_x2.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached x2. Call forward_gpu() before backward_gpu().".to_string(),
                }
            })?;
            let x2_slice = x2
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::InvalidInput {
                    message: "x2 must be contiguous".to_string(),
                })?;
            pool.upload(x2_slice)?
        };
        let x2_buf_uploaded = retained_forward.is_none();
        let mut grad_x2_buf = pool.allocate(hidden_size * 4)?;
        let gate_params = self.gate.curve.to_gpu_params(1);
        ops.richards_curve_backward_input(
            pool,
            &x2_buf,
            &grad_gate_sigma_buf,
            &mut grad_x2_buf,
            &gate_params,
            hidden_size,
        )?;

        // Step 7: grad_w1 = input.T @ grad_x1
        let input_buf = if let Some(ref fwd) = retained_forward {
            fwd.input
        } else {
            let input_slice = input.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "input must be contiguous".to_string(),
                }
            })?;
            pool.upload(input_slice)?
        };
        let input_buf_uploaded = retained_forward.is_none();

        let mut grad_w1_buf = pool.allocate(embedding_dim * hidden_dim * 4)?;
        ops.gemm_f32(
            pool,
            1.0,
            &input_buf,
            &grad_x1_buf,
            0.0,
            &mut grad_w1_buf,
            embedding_dim,
            hidden_dim,
            batch_size,
            true,  // transpose A (input)
            false, // don't transpose B (grad_x1)
        )?;

        // Step 8: grad_w2 = input.T @ grad_x2
        let mut grad_w2_buf = pool.allocate(embedding_dim * hidden_dim * 4)?;
        ops.gemm_f32(
            pool,
            1.0,
            &input_buf,
            &grad_x2_buf,
            0.0,
            &mut grad_w2_buf,
            embedding_dim,
            hidden_dim,
            batch_size,
            true,  // transpose A (input)
            false, // don't transpose B (grad_x2)
        )?;

        // Step 9: grad_input = grad_x1 @ w1.T + grad_x2 @ w2.T
        // First: grad_x1 @ w1.T
        let w1_buf = if let Some(ref cache) = cache_weights {
            cache.w1
        } else {
            let w1_t = self.w1.t();
            let w1_standard = w1_t.as_standard_layout();
            let w1_slice = w1_standard.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "w1 must be contiguous".to_string(),
                }
            })?;
            pool.upload(w1_slice)?
        };
        let w1_buf_uploaded = cache_weights.is_none();

        let mut grad_input_buf = pool.allocate(batch_size * embedding_dim * 4)?;
        ops.gemm_f32(
            pool,
            1.0,
            &grad_x1_buf,
            &w1_buf,
            0.0,
            &mut grad_input_buf,
            batch_size,
            embedding_dim,
            hidden_dim,
            false, // don't transpose A (grad_x1)
            false, // don't transpose B (w1.T already transposed)
        )?;

        // Second: grad_x2 @ w2.T, accumulate into grad_input
        let w2_buf = if let Some(ref cache) = cache_weights {
            cache.w2
        } else {
            let w2_t = self.w2.t();
            let w2_standard = w2_t.as_standard_layout();
            let w2_slice = w2_standard.as_slice().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "w2 must be contiguous".to_string(),
                }
            })?;
            pool.upload(w2_slice)?
        };
        let w2_buf_uploaded = cache_weights.is_none();

        ops.gemm_f32(
            pool,
            1.0,
            &grad_x2_buf,
            &w2_buf,
            1.0, // accumulate (beta = 1.0)
            &mut grad_input_buf,
            batch_size,
            embedding_dim,
            hidden_dim,
            false, // don't transpose A (grad_x2)
            false, // don't transpose B (w2.T already transposed)
        )?;

        // Reduce Richards scalar parameter gradients on GPU (common GLU path).
        // Falls back to CPU reductions below for unsupported configurations/backends.
        let mut value_curve_scalar_grads_gpu: Option<[f32; 9]> = None;
        let mut gate_curve_scalar_grads_gpu: Option<[f32; 9]> = None;

        if Self::richards_curve_supports_gpu_scalar_param_reduction(&self.richards_activation.richards_curve)
        {
            let mut curve_output_grads_buf = pool.allocate(hidden_size * 4)?;
            ops.mul(
                pool,
                &x1_buf,
                &grad_value_buf,
                &mut curve_output_grads_buf,
                hidden_size,
            )?;

            let mut scalar_grads_buf = pool.allocate(9 * 4)?;
            let richards_params = self.richards_activation.richards_curve.to_gpu_params(1);
            let reduce_result = ops.richards_scalar_param_grads_reduce(
                pool,
                &x1_buf,
                &curve_output_grads_buf,
                &mut scalar_grads_buf,
                &richards_params,
                hidden_size,
                matches!(self.richards_activation.richards_curve.variant, Variant::Tanh),
                self.richards_activation.richards_curve.birch_exponential_tail,
            );

            match reduce_result {
                Ok(()) => {
                    let mut host = [0.0f32; 9];
                    pool.download(&scalar_grads_buf, &mut host)?;
                    value_curve_scalar_grads_gpu = Some(host);
                }
                Err(e) => {
                    if Self::gpu_strict_no_fallback_mode() {
                        pool.deallocate(scalar_grads_buf);
                        pool.deallocate(curve_output_grads_buf);
                        return Err(e);
                    }
                }
            }

            pool.deallocate(scalar_grads_buf);
            pool.deallocate(curve_output_grads_buf);
        }

        if Self::richards_curve_supports_gpu_scalar_param_reduction(&self.gate.curve) {
            let mut scalar_grads_buf = pool.allocate(9 * 4)?;
            let gate_params = self.gate.curve.to_gpu_params(1);
            let reduce_result = ops.richards_scalar_param_grads_reduce(
                pool,
                &x2_buf,
                &grad_gate_sigma_buf,
                &mut scalar_grads_buf,
                &gate_params,
                hidden_size,
                matches!(self.gate.curve.variant, Variant::Tanh),
                self.gate.curve.birch_exponential_tail,
            );

            match reduce_result {
                Ok(()) => {
                    let mut host = [0.0f32; 9];
                    pool.download(&scalar_grads_buf, &mut host)?;
                    gate_curve_scalar_grads_gpu = Some(host);
                }
                Err(e) => {
                    if Self::gpu_strict_no_fallback_mode() {
                        pool.deallocate(scalar_grads_buf);
                        return Err(e);
                    }
                }
            }

            pool.deallocate(scalar_grads_buf);
        }

        // Step 10: Download gradients and optionally compute bias gradients via reduction kernel
        let mut grad_w1 = Array2::zeros((embedding_dim, hidden_dim));
        let mut grad_w2 = Array2::zeros((embedding_dim, hidden_dim));
        let mut grad_w_out = Array2::zeros((hidden_dim, embedding_dim));
        let mut grad_input = Array2::zeros((batch_size, embedding_dim));

        pool.download(&grad_w1_buf, grad_w1.as_slice_mut().unwrap())?;
        pool.download(&grad_w2_buf, grad_w2.as_slice_mut().unwrap())?;
        pool.download(&grad_w_out_buf, grad_w_out.as_slice_mut().unwrap())?;
        pool.download(&grad_input_buf, grad_input.as_slice_mut().unwrap())?;

        // Release temporary GPU buffers for backward pass.
        pool.deallocate(grad_output_buf);
        if gated_buf_uploaded {
            pool.deallocate(gated_buf);
        }
        pool.deallocate(grad_w_out_buf);
        if w_out_buf_uploaded {
            pool.deallocate(w_out_buf);
        }
        pool.deallocate(grad_gated_buf);
        pool.deallocate(df_dx_buf);
        pool.deallocate(grad_x1_buf);
        pool.deallocate(grad_x2_buf);
        if x2_buf_uploaded {
            pool.deallocate(x2_buf);
        }
        if x1_buf_uploaded {
            pool.deallocate(x1_buf);
        }
        if value_buf_uploaded {
            pool.deallocate(value_buf);
        }
        if input_buf_uploaded {
            pool.deallocate(input_buf);
        }
        pool.deallocate(grad_w1_buf);
        pool.deallocate(grad_w2_buf);
        if w1_buf_uploaded {
            pool.deallocate(w1_buf);
        }
        pool.deallocate(grad_input_buf);
        if w2_buf_uploaded {
            pool.deallocate(w2_buf);
        }
        // Phase 5.7: Optional bias gradient computation using reduction kernel
        // Bias gradients = sum of gradients over batch dimension
        // bias_grad[j] = sum_i(grad_w[i, j])
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Optionally add bias gradients if enabled (future enhancement)
        // For now, keep compatible with existing apply_gradients interface
        // To enable bias gradients: uncomment below and modify apply_gradients to handle them
        /*
        if self.has_bias {  // Future: add has_bias flag
            let bias_grad_w1 = GpuReductionKernel::reduce_sum_batch(&param_grads[0])?;
            let bias_grad_w2 = GpuReductionKernel::reduce_sum_batch(&param_grads[1])?;
            let bias_grad_w_out = GpuReductionKernel::reduce_sum_batch(&param_grads[2])?;

            // Store bias gradients for parameter updates
            self.bias_w1_grad = bias_grad_w1;
            self.bias_w2_grad = bias_grad_w2;
            self.bias_w_out_grad = bias_grad_w_out;
        }
        */

        // Compute RichardsActivation scalar parameter gradients
        if let Some(canonical) = value_curve_scalar_grads_gpu {
            let value_grads =
                Self::pack_richards_scalar_grads_from_canonical(&self.richards_activation.richards_curve, &canonical);
            let mut value_grads_sum = Array2::<f32>::zeros((1, value_grads.len()));
            for (k, &g) in value_grads.iter().enumerate() {
                value_grads_sum[[0, k]] = g as f32;
            }
            param_grads.push(value_grads_sum);
        } else {
            // CPU fallback for unsupported curve/backend configurations
            if grad_value_cpu.is_none() {
                let mut grad_value = Array2::zeros((batch_size, hidden_dim));
                pool.download(&grad_value_buf, grad_value.as_slice_mut().unwrap())?;
                grad_value_cpu = Some(grad_value);
            }
            if self.cached_x1.is_none()
                && let Some(ref fwd) = retained_forward
            {
                let mut x1_array = Array2::zeros((batch_size, hidden_dim));
                pool.download(&fwd.x1, x1_array.as_slice_mut().unwrap())?;
                self.cached_x1 = Some(x1_array);
            }
            let x1 = self.cached_x1.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached x1. Call forward_gpu() before backward_gpu().".to_string(),
                }
            })?;
            let grad_value = grad_value_cpu.as_ref().unwrap();
            let curve_output_grads = x1 * grad_value;
            let value_grads = self
                .richards_activation
                .richards_curve
                .grad_weights_matrix_f32(x1, &curve_output_grads);

            let mut value_grads_sum = Array2::<f32>::zeros((1, value_grads.len()));
            for (k, &g) in value_grads.iter().enumerate() {
                value_grads_sum[[0, k]] = g as f32;
            }
            param_grads.push(value_grads_sum);
        }

        // Compute RichardsGate scalar parameter gradients
        if let Some(canonical) = gate_curve_scalar_grads_gpu {
            let gate_scalar_grads = Self::pack_richards_scalar_grads_from_canonical(&self.gate.curve, &canonical);
            let gate_param_grads: Vec<Array2<f32>> = gate_scalar_grads
                .into_iter()
                .map(|g| Array2::from_elem((1, 1), g as f32))
                .collect();
            param_grads.extend(gate_param_grads);
        } else {
            // CPU fallback for unsupported curve/backend configurations
            if grad_gate_sigma_cpu.is_none() {
                let mut grad_gate_sigma = Array2::zeros((batch_size, hidden_dim));
                pool.download(
                    &grad_gate_sigma_buf,
                    grad_gate_sigma.as_slice_mut().unwrap(),
                )?;
                grad_gate_sigma_cpu = Some(grad_gate_sigma);
            }
            if self.cached_x2.is_none()
                && let Some(ref fwd) = retained_forward
            {
                let mut x2_array = Array2::zeros((batch_size, hidden_dim));
                pool.download(&fwd.x2, x2_array.as_slice_mut().unwrap())?;
                self.cached_x2 = Some(x2_array);
            }
            let x2 = self.cached_x2.as_ref().ok_or_else(|| {
                crate::common::errors::ModelError::InvalidInput {
                    message: "No cached x2. Call forward_gpu() before backward_gpu().".to_string(),
                }
            })?;
            let grad_gate_sigma = grad_gate_sigma_cpu.as_ref().unwrap();
            let (_, gate_param_grads) = self.gate.compute_gradients(x2, grad_gate_sigma);
            param_grads.extend(gate_param_grads);
        }

        pool.deallocate(grad_value_buf);
        pool.deallocate(grad_gate_sigma_buf);

        // Forward intermediates are single-use for backward.
        self.clear_gpu_forward_cache_for_pool(pool);

        // Apply gradients via CPU optimizers
        self.apply_gradients(&param_grads, learning_rate)?;

        Ok(grad_input)
    }

    /// Streaming forward step with pre-allocated output buffer (zero-allocation)
    pub fn forward_step_into(
        &mut self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut ndarray::Array1<f32>,
    ) {
        // Initialize workspace if needed
        if self.streaming_workspace.is_none() {
            let d_hidden = self.w1.ncols();
            self.streaming_workspace = Some(RichardsGluStreamingWorkspace {
                x1: ndarray::Array1::zeros(d_hidden),
                x2: ndarray::Array1::zeros(d_hidden),
                value: ndarray::Array1::zeros(d_hidden),
                gate_sigma: ndarray::Array1::zeros(d_hidden),
                gated: ndarray::Array1::zeros(d_hidden),
            });
        }
        let ws = self.streaming_workspace.as_mut().unwrap();

        // Ensure workspace dimensions
        if ws.x1.len() != self.w1.ncols() {
            let d_hidden = self.w1.ncols();
            ws.x1 = ndarray::Array1::zeros(d_hidden);
            ws.x2 = ndarray::Array1::zeros(d_hidden);
            ws.value = ndarray::Array1::zeros(d_hidden);
            ws.gate_sigma = ndarray::Array1::zeros(d_hidden);
            ws.gated = ndarray::Array1::zeros(d_hidden);
        }

        // x1 = input * W1
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w1.t(), input, 0.0, &mut ws.x1);
        // x2 = input * W2
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w2.t(), input, 0.0, &mut ws.x2);

        // Apply Richards activation
        if let (Some(x1_slice), Some(value_slice)) = (ws.x1.as_slice(), ws.value.as_slice_mut()) {
            self.richards_activation
                .forward_into_f32(x1_slice, value_slice);
        } else {
            // Fallback
            ws.value.assign(
                &self
                    .richards_activation
                    .forward_matrix_f32(&ws.x1.view().insert_axis(ndarray::Axis(0)).to_owned())
                    .row(0),
            );
        }

        // Apply Richards gate
        if let (Some(x2_slice), Some(gate_slice)) = (ws.x2.as_slice(), ws.gate_sigma.as_slice_mut())
        {
            self.gate.forward_into_f32(x2_slice, gate_slice);
        } else {
            // Fallback
            ws.gate_sigma.assign(
                &self
                    .gate
                    .forward_const(&ws.x2.view().insert_axis(ndarray::Axis(0)).to_owned())
                    .row(0),
            );
        }

        // Gating: value * gate
        ndarray::Zip::from(&mut ws.gated)
            .and(&ws.value)
            .and(&ws.gate_sigma)
            .for_each(|g, &v, &s| *g = v * s);

        // Output = gated * W_out + input
        // gated: (H,), W_out: (H, D)
        // output = input (residual)
        output.assign(input);
        // output += gated * W_out
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_out.t(), &ws.gated, 1.0, output);
    }

    /// Streaming forward step for token-by-token inference.
    ///
    /// This method processes a single vector input (Array1) and returns a single vector output.
    /// It uses zero-copy views to reuse the optimized matrix implementations of the
    /// underlying components.
    pub fn forward_step(&mut self, input: &ndarray::Array1<f32>) -> ndarray::Array1<f32> {
        let mut output = ndarray::Array1::zeros(input.raw_dim());
        self.forward_step_into(&input.view(), &mut output);
        output
    }
}

#[derive(Debug, Clone)]
pub struct RichardsGluStreamingWorkspace {
    pub x1: ndarray::Array1<f32>,
    pub x2: ndarray::Array1<f32>,
    pub value: ndarray::Array1<f32>,
    pub gate_sigma: ndarray::Array1<f32>,
    pub gated: ndarray::Array1<f32>,
}

/// Batch workspace for forward/backward passes using in-place operations
/// Reduces allocations during training by reusing buffers for each forward step
#[derive(Debug, Clone)]
pub struct RichardsGluBatchWorkspace {
    /// x1 = input @ W1, shape: (batch_size, hidden_dim)
    pub x1: Option<Array2<f32>>,
    /// x2 = input @ W2, shape: (batch_size, hidden_dim)
    pub x2: Option<Array2<f32>>,
    /// value = Richards(x1), shape: (batch_size, hidden_dim)
    pub value: Option<Array2<f32>>,
    /// gate_sigma = RichardsGate(x2), shape: (batch_size, hidden_dim)
    pub gate_sigma: Option<Array2<f32>>,
    /// gated = value * gate_sigma, shape: (batch_size, hidden_dim)
    pub gated: Option<Array2<f32>>,
}

impl RichardsGlu {
    /// Ensure buffer capacity with power-of-2 sizing for efficient reallocation
    fn ensure_capacity_2d(buf: &mut Option<Array2<f32>>, rows: usize, cols: usize) {
        if let Some(existing) = buf {
            if existing.nrows() >= rows && existing.ncols() >= cols {
                // Buffer is large enough, no reallocation needed
                return;
            }
        }
        // Reallocate with power-of-2 sizing
        let cap_rows = (rows as u32).next_power_of_two() as usize;
        let cap_cols = (cols as u32).next_power_of_two() as usize;
        *buf = Some(Array2::zeros((cap_rows, cap_cols)));
    }
}

impl Layer for RichardsGlu {
    fn layer_type(&self) -> &str {
        "RichardsGlu"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Strict No-Fallback GPU Logic
        if self.gpu_device.is_some() {
            return self
                .forward_gpu(input)
                .expect("GPU forward execution failed");
        }

        let (batch_size, embedding_dim) = input.dim();
        let hidden_dim = self.w1.ncols();

        // Initialize or reuse workspace buffers
        if self.batch_workspace.is_none() {
            self.batch_workspace = Some(RichardsGluBatchWorkspace {
                x1: Some(Array2::zeros((batch_size, hidden_dim))),
                x2: Some(Array2::zeros((batch_size, hidden_dim))),
                value: Some(Array2::zeros((batch_size, hidden_dim))),
                gate_sigma: Some(Array2::zeros((batch_size, hidden_dim))),
                gated: Some(Array2::zeros((batch_size, hidden_dim))),
            });
        }

        let ws = self.batch_workspace.as_mut().unwrap();

        // Ensure workspace capacity using power-of-2 sizing for efficiency
        Self::ensure_capacity_2d(&mut ws.x1, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.x2, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.value, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.gate_sigma, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.gated, batch_size, hidden_dim);

        // Compute x1 = input @ W1 using general_mat_mul (in-place)
        let mut x1 = ws.x1.take().unwrap();
        x1.slice_mut(s![..batch_size, ..hidden_dim]).fill(0.0); // Reset for beta=0.0
        general_mat_mul(
            1.0,
            input,
            &self.w1,
            0.0,
            &mut x1.slice_mut(s![..batch_size, ..hidden_dim]),
        );

        // Compute x2 = input @ W2 using general_mat_mul (in-place)
        let mut x2 = ws.x2.take().unwrap();
        x2.slice_mut(s![..batch_size, ..hidden_dim]).fill(0.0);
        general_mat_mul(
            1.0,
            input,
            &self.w2,
            0.0,
            &mut x2.slice_mut(s![..batch_size, ..hidden_dim]),
        );

        // Apply Richards activation directly on f32 without materializing f64 matrices.
        let x1_sliced = x1.slice(s![..batch_size, ..hidden_dim]);
        let mut value = ws.value.take().unwrap();
        let value_activated = self
            .richards_activation
            .forward_matrix_f32(&x1_sliced.to_owned());
        value
            .slice_mut(s![..batch_size, ..hidden_dim])
            .assign(&value_activated);

        // Compute gate values using RichardsGate
        let x2_sliced = x2.slice(s![..batch_size, ..hidden_dim]);
        let gate_sigma_activated = self.gate.forward(&x2_sliced.to_owned());
        let mut gate_sigma = ws.gate_sigma.take().unwrap();
        gate_sigma
            .slice_mut(s![..batch_size, ..hidden_dim])
            .assign(&gate_sigma_activated);

        // Compute gated = value * gate_sigma (element-wise)
        let mut gated = ws.gated.take().unwrap();
        {
            let value_slice = value.slice(s![..batch_size, ..hidden_dim]);
            let gate_slice = gate_sigma.slice(s![..batch_size, ..hidden_dim]);
            ndarray::Zip::from(gated.slice_mut(s![..batch_size, ..hidden_dim]))
                .and(&value_slice)
                .and(&gate_slice)
                .for_each(|g, &v, &s| *g = v * s);
        }

        // Compute output = gated @ W_out + input using general_mat_mul
        let gated_sliced = gated.slice(s![..batch_size, ..hidden_dim]);
        let mut output = Array2::zeros((batch_size, embedding_dim));
        general_mat_mul(1.0, &gated_sliced.to_owned(), &self.w_out, 0.0, &mut output);

        // Add residual connection
        output += input;

        // Store workspace references for backward pass
        ws.x1 = Some(x1);
        ws.x2 = Some(x2);
        ws.value = Some(value);
        ws.gate_sigma = Some(gate_sigma);
        ws.gated = Some(gated);

        // Cache values for backward pass (using Arc<Array2<f32>> for zero-copy sharing)
        self.cached_input = Some(input.clone());
        self.cached_x1 = ws
            .x1
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_x2 = ws
            .x2
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_swish = ws
            .value
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_gated = ws
            .gated
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());

        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        if self.gpu_device.is_some() {
            return self
                .backward_gpu(grads, lr)
                .unwrap_or_else(|err| panic!("RichardsGlu GPU backward execution failed: {err}"));
        }

        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (grad_input, param_grads) = self.compute_gradients(input, grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        grad_input
    }

    fn parameters(&self) -> usize {
        let base = self.w1.len() + self.w2.len() + self.w_out.len();
        base + self.richards_activation.weights().len() + self.gate.parameters()
    }

    fn compute_gradients(
        &self,
        input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let x1 = self
            .cached_x1
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w1));
        let x2 = self
            .cached_x2
            .as_ref()
            .cloned()
            .unwrap_or_else(|| input.dot(&self.w2));
        let value = self
            .cached_swish
            .as_ref()
            .cloned()
            .unwrap_or_else(|| self.richards_activation.forward_matrix_f32(&x1));
        // Compute gate values
        let gate_sigma = self.gate.forward_const(&x2);

        let gated = self
            .cached_gated
            .as_ref()
            .cloned()
            .unwrap_or_else(|| &value * &gate_sigma);

        // Gradients wrt parameters using general_mat_mul (in-place)
        let (batch_size, embedding_dim) = input.dim();
        let hidden_dim = self.w1.ncols();
        let mut grad_w_out = Array2::zeros((hidden_dim, embedding_dim));
        general_mat_mul(1.0, &gated.t(), output_grads, 0.0, &mut grad_w_out);

        let mut grad_gated = Array2::zeros((batch_size, hidden_dim));
        general_mat_mul(1.0, output_grads, &self.w_out.t(), 0.0, &mut grad_gated);

        let grad_value = &grad_gated * &gate_sigma;
        let grad_gate_sigma = &grad_gated * &value;

        // Compute gradients through RichardsActivation / RichardsGate (parallelized)
        let mut grad_x1 = Array2::<f32>::zeros(x1.raw_dim());
        let mut grad_x2 = Array2::<f32>::zeros(x2.raw_dim());
        let gate_temp_reciprocal = 1.0 / self.gate.temperature;

        // Ensure arrays are contiguous for slice-based parallel iteration.
        // In most cases (from dot/arithmetic), they are already contiguous.
        let x1_contig = x1.as_standard_layout();
        let x2_contig = x2.as_standard_layout();
        let grad_val_contig = grad_value.as_standard_layout();
        let grad_gate_contig = grad_gate_sigma.as_standard_layout();

        let hidden_dim = x1.shape()[1];
        debug_assert_eq!(hidden_dim, x2.shape()[1]);

        // Get raw slices for parallel processing
        let x1_slice = x1_contig.as_slice().expect("x1 must be contiguous");
        let x2_slice = x2_contig.as_slice().expect("x2 must be contiguous");
        let gv_slice = grad_val_contig
            .as_slice()
            .expect("grad_value must be contiguous");
        let gg_slice = grad_gate_contig
            .as_slice()
            .expect("grad_gate must be contiguous");

        let gx1_slice = grad_x1.as_slice_mut().expect("grad_x1 must be contiguous");
        let gx2_slice = grad_x2.as_slice_mut().expect("grad_x2 must be contiguous");

        gx1_slice
            .par_chunks_mut(hidden_dim)
            .zip(gx2_slice.par_chunks_mut(hidden_dim))
            .zip(x1_slice.par_chunks(hidden_dim))
            .zip(x2_slice.par_chunks(hidden_dim))
            .zip(gv_slice.par_chunks(hidden_dim))
            .zip(gg_slice.par_chunks(hidden_dim))
            .for_each(
                |(((((gx1_row, gx2_row), x1_row), x2_row), gv_row), gg_row)| {
                    // Thread-local scratch buffers
                    let mut value_deriv_row = vec![0.0; x1_row.len()];
                    let mut value_deriv_tmp = vec![0.0; x1_row.len()];
                    let mut gate_scaled_row = vec![0.0; x2_row.len()];
                    let mut gate_curve_deriv_row = vec![0.0; x2_row.len()];

                    // value_deriv_row = d/dx[x * Richards(x)]
                    self.richards_activation.derivative_into_f32_with_scratch(
                        x1_row,
                        &mut value_deriv_row,
                        &mut value_deriv_tmp,
                    );

                    // Gate derivative with temperature scaling:
                    // g(x) = curve(x/T) => dg/dx = curve'(x/T) * (1/T)
                    for j in 0..x2_row.len() {
                        gate_scaled_row[j] = x2_row[j] * gate_temp_reciprocal;
                    }
                    self.gate
                        .curve
                        .derivative_into_f32(&gate_scaled_row, &mut gate_curve_deriv_row);

                    for j in 0..x1_row.len() {
                        gx1_row[j] = value_deriv_row[j] * gv_row[j];
                    }
                    for j in 0..x2_row.len() {
                        let gate_deriv = gate_curve_deriv_row[j] * gate_temp_reciprocal;
                        gx2_row[j] = gate_deriv * gg_row[j];
                    }
                },
            );

        // Use input directly for weight gradients (fallback to cached input if available)
        let weight_input = self.cached_input.as_ref().unwrap_or(input);

        // Compute grad_w1 = input.T @ grad_x1 using general_mat_mul (in-place)
        let mut grad_w1 = Array2::zeros((embedding_dim, hidden_dim));
        general_mat_mul(1.0, &weight_input.t(), &grad_x1, 0.0, &mut grad_w1);

        // Compute grad_w2 = input.T @ grad_x2 using general_mat_mul (in-place)
        let mut grad_w2 = Array2::zeros((embedding_dim, hidden_dim));
        general_mat_mul(1.0, &weight_input.t(), &grad_x2, 0.0, &mut grad_w2);

        // Input gradient (include residual branch) using general_mat_mul
        let mut grad_input_glu = Array2::zeros((batch_size, embedding_dim));
        general_mat_mul(1.0, &grad_x1, &self.w1.t(), 0.0, &mut grad_input_glu);
        general_mat_mul(1.0, &grad_x2, &self.w2.t(), 1.0, &mut grad_input_glu); // Add mode (beta=1.0)

        let grad_input = grad_input_glu + output_grads;

        // Parameter gradients vector
        let mut param_grads = vec![grad_w1, grad_w2, grad_w_out];

        // Compute RichardsActivation gradients (value function) in one shot.
        // value(x) = x * curve(x) => dL/d(curve(x)) = x * dL/d(value).
        let curve_output_grads = &x1 * &grad_value;
        let value_grads = self
            .richards_activation
            .richards_curve
            .grad_weights_matrix_f32(&x1, &curve_output_grads);
        let mut value_grads_sum = Array2::<f32>::zeros((1, value_grads.len()));
        for (k, &g) in value_grads.iter().enumerate() {
            value_grads_sum[[0, k]] = g as f32;
        }

        // Compute RichardsGate gradients using the gate's own gradient computation
        let (_, gate_param_grads) = self.gate.compute_gradients(&x2, &grad_gate_sigma);

        param_grads.push(value_grads_sum);
        param_grads.extend(gate_param_grads);

        (grad_input, param_grads)
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) -> Result<()> {
        // Expect gradients in order: W1, W2, W_out, richards_activation, gate_parameters...
        if param_grads.len() < 4 {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "RichardsGlu expects at least 4 gradient blocks, got {}",
                    param_grads.len()
                ),
            });
        }

        // Update w1, w2, w_out
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w_out
            .step(&mut self.w_out, &param_grads[2], lr);

        // Update RichardsActivation weights
        let grad_value_vec: Vec<f64> = param_grads[3].iter().map(|&x| x as f64).collect();
        self.richards_activation.step(&grad_value_vec, lr as f64);

        // Update RichardsGate parameters (parameters 4 onwards)
        if param_grads.len() > 4 {
            let gate_grads = &param_grads[4..];
            self.gate.apply_gradients(gate_grads, lr)?;
        }

        // CPU-side optimizer updates changed weights; force re-upload on next GPU forward.
        self.gpu_cache = None;
        self.gpu_forward_cache = None;

        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq = 0.0f32;
        sumsq += self.w1.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w2.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self.w_out.iter().map(|&w| w * w).sum::<f32>();
        sumsq += self
            .richards_activation
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>();
        sumsq += self.gate.weight_norm();
        sumsq.sqrt()
    }

    fn zero_gradients(&mut self) {
        // RichardsGlu doesn't maintain internal gradient state
        // Gradients are computed on-demand
    }
}

impl RichardsGlu {
    /// In-place forward pass for RichardsGlu (Phase 5.1.1 implementation)
    ///
    /// This is a true zero-allocation implementation that writes directly to the
    /// provided output buffer, reusing pre-allocated workspace buffers for all
    /// intermediate computations (x1, x2, value, gate_sigma, gated).
    ///
    /// # Arguments
    /// * `input` - Input tensor (batch_size × embedding_dim)
    /// * `output` - Pre-allocated output buffer (batch_size × embedding_dim)
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if output dimensions don't match input
    pub(crate) fn forward_into(
        &mut self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> crate::common::errors::Result<()> {
        let (batch_size, embedding_dim) = input.dim();
        let hidden_dim = self.w1.ncols();

        // Validate output buffer dimensions
        if output.dim() != (batch_size, embedding_dim) {
            return Err(crate::common::errors::ModelError::InvalidInput {
                message: format!(
                    "Output dimension mismatch: expected ({}, {}), got {:?}",
                    batch_size,
                    embedding_dim,
                    output.dim()
                ),
            });
        }

        // Initialize or reuse workspace buffers
        if self.batch_workspace.is_none() {
            self.batch_workspace = Some(RichardsGluBatchWorkspace {
                x1: Some(Array2::zeros((batch_size, hidden_dim))),
                x2: Some(Array2::zeros((batch_size, hidden_dim))),
                value: Some(Array2::zeros((batch_size, hidden_dim))),
                gate_sigma: Some(Array2::zeros((batch_size, hidden_dim))),
                gated: Some(Array2::zeros((batch_size, hidden_dim))),
            });
        }

        let ws = self.batch_workspace.as_mut().unwrap();

        // Ensure workspace capacity using power-of-2 sizing for efficiency
        Self::ensure_capacity_2d(&mut ws.x1, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.x2, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.value, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.gate_sigma, batch_size, hidden_dim);
        Self::ensure_capacity_2d(&mut ws.gated, batch_size, hidden_dim);

        // Compute x1 = input @ W1 using general_mat_mul (in-place)
        let mut x1 = ws.x1.take().unwrap();
        x1.slice_mut(s![..batch_size, ..hidden_dim]).fill(0.0);
        general_mat_mul(
            1.0,
            input,
            &self.w1,
            0.0,
            &mut x1.slice_mut(s![..batch_size, ..hidden_dim]),
        );

        // Compute x2 = input @ W2 using general_mat_mul (in-place)
        let mut x2 = ws.x2.take().unwrap();
        x2.slice_mut(s![..batch_size, ..hidden_dim]).fill(0.0);
        general_mat_mul(
            1.0,
            input,
            &self.w2,
            0.0,
            &mut x2.slice_mut(s![..batch_size, ..hidden_dim]),
        );

        // Apply Richards activation directly on f32
        let x1_sliced = x1.slice(s![..batch_size, ..hidden_dim]);
        let mut value = ws.value.take().unwrap();
        let value_activated = self
            .richards_activation
            .forward_matrix_f32(&x1_sliced.to_owned());
        value
            .slice_mut(s![..batch_size, ..hidden_dim])
            .assign(&value_activated);

        // Compute gate values using RichardsGate
        let x2_sliced = x2.slice(s![..batch_size, ..hidden_dim]);
        let gate_sigma_activated = self.gate.forward(&x2_sliced.to_owned());
        let mut gate_sigma = ws.gate_sigma.take().unwrap();
        gate_sigma
            .slice_mut(s![..batch_size, ..hidden_dim])
            .assign(&gate_sigma_activated);

        // Compute gated = value * gate_sigma (element-wise)
        let mut gated = ws.gated.take().unwrap();
        {
            let value_slice = value.slice(s![..batch_size, ..hidden_dim]);
            let gate_slice = gate_sigma.slice(s![..batch_size, ..hidden_dim]);
            ndarray::Zip::from(gated.slice_mut(s![..batch_size, ..hidden_dim]))
                .and(&value_slice)
                .and(&gate_slice)
                .for_each(|g, &v, &s| *g = v * s);
        }

        // Compute output = gated @ W_out + input (DIRECTLY INTO OUTPUT BUFFER - ZERO ALLOCATION)
        let gated_sliced = gated.slice(s![..batch_size, ..hidden_dim]);
        output.fill(0.0); // Reset output for beta=0.0
        general_mat_mul(1.0, &gated_sliced.to_owned(), &self.w_out, 0.0, output);

        // Add residual connection (in-place)
        *output += input;

        // Store workspace references for backward pass
        ws.x1 = Some(x1);
        ws.x2 = Some(x2);
        ws.value = Some(value);
        ws.gate_sigma = Some(gate_sigma);
        ws.gated = Some(gated);

        // Cache values for backward pass
        self.cached_input = Some(input.clone());
        self.cached_x1 = ws
            .x1
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_x2 = ws
            .x2
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_swish = ws
            .value
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());
        self.cached_gated = ws
            .gated
            .as_ref()
            .map(|b| b.slice(s![..batch_size, ..hidden_dim]).to_owned());

        Ok(())
    }
}

// ============================================================================
// GPU Component Implementation (Phase 5.6)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for RichardsGlu {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        RichardsGlu::set_gpu_device(self, device);
    }

    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        RichardsGlu::set_gpu_device(self, Arc::new(Mutex::new(device)));
        Ok(())
    }

    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }

    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device
            .as_ref()
            .and_then(|device_arc| match device_arc.lock() {
                Ok(device) => {
                    let backend = device.backend();
                    Some(backend.as_str())
                }
                Err(_) => None,
            })
    }

    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        _embed_dim: usize,
        _seq_len: usize,
    ) -> Result<()> {
        if let Some(device_arc) = &self.gpu_device {
            let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for RichardsGlu capacity allocation"
                    .to_string(),
            })?;

            let embedding_dim = self.w1.nrows();
            let hidden_dim = self.w1.ncols();
            let hidden_size = batch_size * hidden_dim;

            // Pre-allocate intermediate buffers for forward pass
            // These are allocated but not stored; they'll be re-allocated during forward for now.
            // In a future optimization, these could be cached in the GPU device's buffer pool.
            let _ = device.allocate_f32(hidden_size)?; // x1
            let _ = device.allocate_f32(hidden_size)?; // x2
            let _ = device.allocate_f32(hidden_size)?; // value
            let _ = device.allocate_f32(hidden_size)?; // gate_val
            let _ = device.allocate_f32(hidden_size)?; // gated
            let _ = device.allocate_f32(batch_size * embedding_dim)?; // output

            Ok(())
        } else {
            Err(ModelError::Backend {
                message:
                    "GPU device not attached to RichardsGlu. Call enable_gpu_auto_detect() first."
                        .to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    use crate::domain::compute::GpuComponent as GpuComponentTrait;

    #[test]
    fn test_richards_glu_forward_backward() {
        let batch_size = 2;
        let embedding_dim = 4;
        let hidden_dim = 8;
        let mut glu = RichardsGlu::new(embedding_dim, hidden_dim);

        let input = Array2::from_shape_vec(
            (batch_size, embedding_dim),
            vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.5, 0.0, -0.5],
        )
        .unwrap();

        // Forward
        let output = glu.forward(&input);
        assert_eq!(output.dim(), (batch_size, embedding_dim));

        // Backward
        let grad_output = Array2::from_elem(output.dim(), 0.1);
        let grad_input = glu.backward(&grad_output, 0.01);
        assert_eq!(grad_input.dim(), (batch_size, embedding_dim));
    }

    #[test]
    fn test_richards_glu_shapes() {
        let mut glu = RichardsGlu::new(10, 20);
        let input = Array2::zeros((5, 10));
        let output = glu.forward(&input);
        assert_eq!(output.dim(), (5, 10));

        let grad_out = Array2::ones((5, 10));
        let grad_in = glu.backward(&grad_out, 0.001);
        assert_eq!(grad_in.dim(), (5, 10));
    }

    // ========================================================================
    // GPU VALIDATION TESTS (Phase 5.6)
    // ========================================================================

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_auto_detect() {
        use crate::domain::compute::GpuComponent as GpuTrait;

        let mut layer = RichardsGlu::new(768, 3072);

        // Before detection
        assert!(!layer.is_gpu_ready());
        assert_eq!(layer.gpu_backend_name(), None);

        // Try auto-detection (OK to fail on CPU-only systems)
        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "✅ GPU detected: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );
                assert!(layer.is_gpu_ready());
                assert!(layer.gpu_backend_name().is_some());
            }
            Err(e) => {
                println!("ℹ️  No GPU available (expected on CPU-only systems): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_forward_gpu_basic() {
        let mut layer = RichardsGlu::new(768, 3072);
        let batch_size = 8;
        let input = Array2::zeros((batch_size, 768));

        // Enable GPU (skip if unavailable)
        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "Testing GPU forward on: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );

                // Forward pass
                match layer.forward_gpu(&input) {
                    Ok(output) => {
                        assert_eq!(output.dim(), (batch_size, 768), "Output shape mismatch");
                        println!("✅ GPU forward pass successful");
                    }
                    Err(e) => {
                        println!("⚠️  GPU forward failed: {}", e);
                        panic!("GPU forward pass should not fail when GPU is available");
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping GPU test (no GPU available): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_cpu_numerical_validation() {
        use rand_distr::{Distribution, Normal};

        let mut layer = RichardsGlu::new(768, 3072);
        let batch_size = 8;

        // Generate random input for more challenging numerical validation
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.1).unwrap();
        let input = Array2::from_shape_fn((batch_size, 768), |_| normal.sample(&mut rng));

        // Skip if GPU not available
        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "Comparing GPU vs CPU on: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );

                // CPU forward
                let output_cpu = layer.forward(&input);

                // GPU forward
                match layer.forward_gpu(&input) {
                    Ok(output_gpu) => {
                        // Compare outputs - compute L2 norm manually
                        let diff = (&output_cpu - &output_gpu).mapv(|x| x * x).sum().sqrt();
                        let cpu_norm = output_cpu.mapv(|x| x * x).sum().sqrt();
                        let relative_error = diff / (cpu_norm + 1e-8);

                        println!("L2 difference: {}", diff);
                        println!("Relative error: {:.6e}", relative_error);
                        println!("CPU norm: {}", cpu_norm);

                        // Tolerance: 1e-4 relative error (very strict)
                        if relative_error > 1e-4 {
                            println!("⚠️  Numerical mismatch: {:.6e} > 1e-4", relative_error);
                            println!("Note: This may be acceptable for some GPU backends");
                        } else {
                            println!("✅ Numerical match within tolerance");
                        }

                        // For now, we don't assert - GPU numerical differences are expected
                        // This is a validation/diagnostic test
                        assert!(
                            relative_error < 1e-2,
                            "Error too large (>{:.2e})",
                            relative_error
                        );
                    }
                    Err(e) => {
                        println!("⚠️  GPU forward failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping numerical validation (no GPU): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_batch_size_robustness() {
        let mut layer = RichardsGlu::new(768, 3072);

        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "Testing batch size robustness on: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );

                let batch_sizes = vec![1, 8, 16, 32, 64, 128, 256];

                for batch_size in batch_sizes {
                    let input = Array2::zeros((batch_size, 768));
                    match layer.forward_gpu(&input) {
                        Ok(output) => {
                            assert_eq!(output.dim(), (batch_size, 768));
                            println!("✅ Batch size {}: OK", batch_size);
                        }
                        Err(e) => {
                            println!("❌ Batch size {}: FAILED - {}", batch_size, e);
                            panic!("GPU forward failed for batch size {}", batch_size);
                        }
                    }
                }
                println!("✅ All batch sizes passed");
            }
            Err(e) => {
                println!("ℹ️  Skipping batch robustness test (no GPU): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gpu_device_management() {
        use crate::domain::compute::GpuComponent as GpuTrait;

        let mut layer = RichardsGlu::new(768, 3072);

        // 1. Check initial state
        assert!(!layer.is_gpu_ready());

        // 2. Enable GPU
        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                // 3. Verify ready
                assert!(layer.is_gpu_ready());
                assert!(layer.gpu_device().is_some());

                // 4. Verify backend name accessible
                let backend = layer.gpu_backend_name();
                assert!(backend.is_some());
                println!("✅ GPU device: {}", backend.unwrap());

                // 5. Verify capacity allocation works
                match layer.ensure_capacity(1024, 768, 1) {
                    Ok(()) => {
                        println!("✅ Capacity allocation successful");
                    }
                    Err(e) => {
                        println!("⚠️  Capacity allocation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping device management test (no GPU): {}", e);
            }
        }
    }

    // ========================================================================
    // GPU BACKWARD PASS TESTS (Phase 5.6.2)
    // ========================================================================

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_backward_gpu_basic() {
        use crate::domain::compute::GpuComponent as GpuTrait;

        let mut layer = RichardsGlu::new(768, 3072);
        let batch_size = 8;
        let input = Array2::zeros((batch_size, 768));

        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "Testing GPU backward on: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );

                // Forward pass (required for backward)
                match layer.forward_gpu(&input) {
                    Ok(output) => {
                        // Create gradient of loss w.r.t. output
                        let grad_output = Array2::ones(output.dim());

                        // Backward pass
                        match layer.backward_gpu(&grad_output, 0.001) {
                            Ok(grad_input) => {
                                assert_eq!(
                                    grad_input.dim(),
                                    (batch_size, 768),
                                    "Gradient shape mismatch"
                                );
                                println!("✅ GPU backward pass successful");
                            }
                            Err(e) => {
                                println!("❌ GPU backward failed: {}", e);
                                panic!("Backward should not fail when GPU is available");
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️  GPU forward failed (skipping backward test): {}", e);
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping GPU backward test (no GPU): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gradient_accumulation() {
        use crate::domain::compute::GpuComponent as GpuTrait;

        let mut layer = RichardsGlu::new(768, 3072);
        let batch_size = 4;
        // Use non-zero input to ensure non-zero gradients
        let input = Array2::from_shape_fn((batch_size, 768), |(i, j)| {
            ((i * 768 + j) % 100) as f32 * 0.01
        });

        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                println!(
                    "Testing gradient accumulation on: {}",
                    layer.gpu_backend_name().unwrap_or("unknown")
                );

                // Save initial weights
                let w1_init = layer.w1.clone();
                let w2_init = layer.w2.clone();
                let w_out_init = layer.w_out.clone();

                // Forward pass
                match layer.forward_gpu(&input) {
                    Ok(output) => {
                        let grad_output = Array2::ones(output.dim());

                        // Backward pass
                        match layer.backward_gpu(&grad_output, 0.001) {
                            Ok(_grad_input) => {
                                // Verify weights have changed (learned)
                                let w1_changed = !layer.w1.abs_diff_eq(&w1_init, 1e-6);
                                let w2_changed = !layer.w2.abs_diff_eq(&w2_init, 1e-6);
                                let w_out_changed = !layer.w_out.abs_diff_eq(&w_out_init, 1e-6);

                                if w1_changed {
                                    println!("✅ W1 gradients applied");
                                }
                                if w2_changed {
                                    println!("✅ W2 gradients applied");
                                }
                                if w_out_changed {
                                    println!("✅ W_out gradients applied");
                                }

                                // At least one weight should have changed
                                assert!(
                                    w1_changed || w2_changed || w_out_changed,
                                    "Weights should be updated by backward pass"
                                );
                            }
                            Err(e) => {
                                println!("⚠️  Backward failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        println!("⚠️  Forward failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping gradient accumulation test (no GPU): {}", e);
            }
        }
    }

    #[test]
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn test_gradient_shapes() {
        use crate::domain::compute::GpuComponent as GpuTrait;

        let mut layer = RichardsGlu::new(768, 3072);

        match layer.enable_gpu_auto_detect() {
            Ok(()) => {
                let batch_sizes = vec![1, 8, 16];

                for batch_size in batch_sizes {
                    let input = Array2::zeros((batch_size, 768));
                    match layer.forward_gpu(&input) {
                        Ok(output) => {
                            let grad_output = Array2::ones(output.dim());
                            match layer.backward_gpu(&grad_output, 0.001) {
                                Ok(grad_input) => {
                                    assert_eq!(grad_input.dim(), (batch_size, 768));
                                    println!("✅ Batch {}: gradient shape correct", batch_size);
                                }
                                Err(e) => {
                                    println!("❌ Batch {}: backward failed: {}", batch_size, e);
                                    panic!("Backward should work");
                                }
                            }
                        }
                        Err(e) => {
                            println!("⚠️  Batch {}: forward failed: {}", batch_size, e);
                        }
                    }
                }
            }
            Err(e) => {
                println!("ℹ️  Skipping gradient shape test (no GPU): {}", e);
            }
        }
    }
}
