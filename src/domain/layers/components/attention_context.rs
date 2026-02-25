//! Shared Attention Context Component
//!
//! This component provides attention context management that can be used
//! by multiple architectures (Transformer, Diffusion).
//! It encapsulates the logic for applying similarity-based context modulation.

use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, Axis, Zip};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use crate::common::errors::{ModelError, Result};
use crate::domain::compute::GpuDevice;

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::compute::GpuComponent;
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
use crate::domain::layers::components::gpu_device_utils::gpu_gemm_with_attached_device;

/// Shared attention context component
///
/// ## GPU Support (Phase 5.6)
///
/// Implements `GpuComponent` trait for unified GPU device management.
/// Supports automatic GPU detection with strict no-fallback semantics.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedAttentionContext {
    /// Incoming similarity context from previous layer
    #[serde(skip)]
    pub incoming_context: Option<Array2<f32>>,
    /// Current similarity context strength
    pub similarity_context_strength: Array2<f32>,
    /// Outgoing similarity context (activation similarity matrix)
    /// Lazily allocated: only created when update_outgoing_context is called
    #[serde(skip)]
    pub outgoing_context: Option<Array2<f32>>,
    /// Update rate for outgoing context
    #[serde(skip)]
    pub similarity_update_rate: f32,
    /// Scratch buffer for sampled input data (reused across calls)
    #[serde(skip)]
    scratch_sub_x: Option<Array2<f32>>,
    /// Scratch buffer for sampled output data (reused across calls)
    #[serde(skip)]
    scratch_sub_y: Option<Array2<f32>>,
    /// Scratch buffer for covariance matrix (reused across calls)
    #[serde(skip)]
    scratch_cov: Option<Array2<f32>>,
    /// Scratch buffer for denominator matrix (reused across calls)
    #[serde(skip)]
    scratch_denom: Option<Array2<f32>>,
    /// Scratch buffer for indices (reused across calls)
    #[serde(skip)]
    scratch_indices: Vec<usize>,
    /// Tracked capacity for power-of-2 sizing
    #[serde(skip)]
    scratch_capacity: usize,

    /// GPU device for this component (Phase 5.6)
    /// If attached, enables GPU execution with strict no-fallback semantics
    #[serde(skip)]
    #[allow(dead_code)]
    gpu_device: Option<Arc<Mutex<GpuDevice>>>,
}

impl Default for SharedAttentionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedAttentionContext {
    /// Clear attached GPU device so context operations run CPU-only.
    #[inline]
    pub fn clear_gpu_device(&mut self) {
        self.gpu_device = None;
    }

    /// Get outgoing similarity context if allocated
    pub fn get_outgoing_context(&self) -> Option<&Array2<f32>> {
        self.outgoing_context.as_ref()
    }

    /// Set update rate for outgoing context
    pub fn set_update_rate(&mut self, rate: f32) {
        self.similarity_update_rate = rate;
    }

    /// Update outgoing similarity context (activation similarity matrix)
    ///
    /// Uses reusable scratch buffers to minimize allocations during training.
    /// Buffers are lazily allocated with power-of-2 sizing for efficiency.
    pub fn update_outgoing_context(
        &mut self,
        input: &ndarray::ArrayView2<f32>,
        output: &ndarray::ArrayView2<f32>,
        embed_dim_config: usize,
    ) {
        let rate = self.similarity_update_rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return;
        }

        let seq_len = input.nrows().min(output.nrows());
        let embed_dim = input.ncols().min(output.ncols()).min(embed_dim_config);

        if seq_len == 0 || embed_dim == 0 {
            return;
        }

        // Lazy allocation: allocate only on first update or if shape changes
        if self.outgoing_context.is_none()
            || self.outgoing_context.as_ref().unwrap().shape() != [embed_dim, embed_dim]
        {
            self.outgoing_context = Some(Array2::zeros((embed_dim, embed_dim)));
        }

        let sample_size = seq_len.min(32);
        let step = (seq_len / sample_size).max(1);

        // Reuse indices buffer with power-of-2 capacity
        self.scratch_indices.clear();
        self.scratch_indices
            .extend((0..seq_len).step_by(step).take(sample_size));
        let actual_sample_size = self.scratch_indices.len();

        if actual_sample_size == 0 {
            return;
        }

        // Ensure scratch buffers have sufficient capacity with power-of-2 sizing
        let required_capacity = actual_sample_size * embed_dim;
        if self.scratch_capacity < required_capacity {
            let new_capacity = required_capacity.next_power_of_two().max(64);
            self.scratch_sub_x = Some(Array2::zeros((actual_sample_size, embed_dim)));
            self.scratch_sub_y = Some(Array2::zeros((actual_sample_size, embed_dim)));
            self.scratch_cov = Some(Array2::zeros((embed_dim, embed_dim)));
            self.scratch_denom = Some(Array2::zeros((embed_dim, embed_dim)));
            self.scratch_capacity = new_capacity;
        }

        // Take ownership of scratch buffers to work with them directly
        let mut sub_x = self.scratch_sub_x.take().unwrap();
        let mut sub_y = self.scratch_sub_y.take().unwrap();
        let mut cov = self.scratch_cov.take().unwrap();
        let mut denom = self.scratch_denom.take().unwrap();

        // Ensure shapes are correct (in case embed_dim changed)
        if sub_x.dim() != (actual_sample_size, embed_dim) {
            sub_x = Array2::zeros((actual_sample_size, embed_dim));
            sub_y = Array2::zeros((actual_sample_size, embed_dim));
        }
        if cov.dim() != (embed_dim, embed_dim) {
            cov = Array2::zeros((embed_dim, embed_dim));
            denom = Array2::zeros((embed_dim, embed_dim));
        }

        // 1. Gather sampled data and handle non-finite values
        for (i, &idx) in self.scratch_indices.iter().enumerate() {
            let row_x = input.row(idx);
            let row_y = output.row(idx);

            for j in 0..embed_dim {
                let val_x = row_x[j];
                sub_x[[i, j]] = if val_x.is_finite() { val_x } else { 0.0 };

                let val_y = row_y[j];
                sub_y[[i, j]] = if val_y.is_finite() { val_y } else { 0.0 };
            }
        }

        // 2. Compute means
        let mean_x = sub_x.mean_axis(Axis(0)).unwrap();
        let mean_y = sub_y.mean_axis(Axis(0)).unwrap();

        // 3. Center data (broadcasting works: (S, D) - (D,))
        sub_x -= &mean_x;
        sub_y -= &mean_y;

        // 4. Compute Norms (Sqrt of sum of squares)
        let norm_x_sq = sub_x.mapv(|v| v * v).sum_axis(Axis(0));
        let norm_y_sq = sub_y.mapv(|v| v * v).sum_axis(Axis(0));

        let norm_x = norm_x_sq.mapv(|v| v.sqrt());
        let norm_y = norm_y_sq.mapv(|v| v.sqrt());

        // 5. Covariance Matrix: X^T * Y -> (D, D)
        // Using general_mat_mul for in-place computation (avoids intermediate allocation)
        cov.fill(0.0);
        general_mat_mul(1.0, &sub_x.t(), &sub_y, 0.0, &mut cov);

        // 6. Denominator Matrix: Outer product of norms
        let norm_x_col = norm_x.insert_axis(Axis(1));
        let norm_y_row = norm_y.insert_axis(Axis(0));
        denom.fill(0.0);
        general_mat_mul(1.0, &norm_x_col, &norm_y_row, 0.0, &mut denom);

        let tanh = crate::domain::richards::RichardsCurve::tanh(false);

        // Get reference to outgoing context
        let outgoing_context = self.outgoing_context.as_mut().unwrap();

        // 7. Update with EMA (Parallelized)
        Zip::from(outgoing_context)
            .and(&cov)
            .and(&denom)
            .par_for_each(|prev, &c, &d| {
                let sim_raw = if d > 1e-12 { c / d } else { 0.0 };
                let sim = if sim_raw.is_finite() {
                    tanh.forward_scalar_f32(sim_raw)
                } else {
                    0.0
                };
                *prev = (1.0 - rate) * *prev + rate * sim;
            });

        // Return scratch buffers to struct
        self.scratch_sub_x = Some(sub_x);
        self.scratch_sub_y = Some(sub_y);
        self.scratch_cov = Some(cov);
        self.scratch_denom = Some(denom);
    }
}

impl SharedAttentionContext {
    /// Create a new shared attention context component
    pub fn new() -> Self {
        Self {
            incoming_context: None,
            similarity_context_strength: Array2::zeros((1, 1)),
            outgoing_context: None, // Lazy allocation
            similarity_update_rate: 0.01,
            scratch_sub_x: None,
            scratch_sub_y: None,
            scratch_cov: None,
            scratch_denom: None,
            scratch_indices: Vec::new(),
            scratch_capacity: 0,
            gpu_device: None,
        }
    }

    /// Set incoming similarity context.
    ///
    /// Reuses previously allocated storage when shape is unchanged.
    pub fn set_incoming_context(&mut self, context: Option<&Array2<f32>>) {
        self.set_incoming_context_reuse(context);
    }

    /// Set incoming similarity context while minimizing allocations.
    ///
    /// If an existing context buffer has the same shape, this assigns into it
    /// instead of replacing the buffer.
    pub fn set_incoming_context_reuse(&mut self, context: Option<&Array2<f32>>) {
        match context {
            Some(ctx) => {
                if let Some(existing) = self.incoming_context.as_mut() {
                    if existing.dim() == ctx.dim() {
                        existing.assign(ctx);
                        return;
                    }
                }
                self.incoming_context = Some(ctx.clone());
            }
            None => {
                self.incoming_context = None;
            }
        }
    }

    /// Set incoming similarity context with an embed-dimension guard.
    ///
    /// If the provided context shape does not match `embed_dim x embed_dim`,
    /// the incoming context is cleared and no assignment is performed.
    #[inline]
    pub fn set_incoming_context_checked_reuse(
        &mut self,
        context: Option<&Array2<f32>>,
        embed_dim: usize,
    ) {
        if let Some(ctx) = context
            && (ctx.nrows() != embed_dim || ctx.ncols() != embed_dim)
        {
            self.clear_context();
            return;
        }
        self.set_incoming_context_reuse(context);
    }

    /// Get incoming similarity context
    pub fn get_incoming_context(&self) -> Option<&Array2<f32>> {
        self.incoming_context.as_ref()
    }

    /// Set similarity context strength
    pub fn set_strength(&mut self, strength: f32) {
        if self.similarity_context_strength.len() != 1 {
            self.similarity_context_strength = Array2::zeros((1, 1));
        }
        self.similarity_context_strength[[0, 0]] = strength;
    }

    /// Get similarity context strength
    pub fn get_strength(&self) -> f32 {
        self.similarity_context_strength
            .get((0, 0))
            .copied()
            .unwrap_or(0.0)
    }

    /// Check if context is available
    pub fn has_context(&self) -> bool {
        self.incoming_context.is_some()
    }

    /// Clear the incoming context
    pub fn clear_context(&mut self) {
        self.incoming_context = None;
    }

    /// Set outgoing context for the next layer without allocating new buffers.
    /// Used internally for zero-allocation context passing between layers.
    #[inline]
    pub fn set_outgoing_context_reuse_silent(&mut self, context: Option<&Array2<f32>>) {
        if let Some(ctx) = context {
            if let Some(existing) = self.outgoing_context.as_mut() {
                if existing.dim() == ctx.dim() {
                    existing.assign(ctx);
                    return;
                }
            }
            self.outgoing_context = Some(ctx.clone());
        } else {
            self.outgoing_context = None;
        }
    }

    /// Get parameter count (1 scalar for strength)
    pub fn parameters(&self) -> usize {
        1
    }

    /// Get L2 norm of parameters
    pub fn weight_norm(&self) -> f32 {
        self.get_strength().abs()
    }

    /// Get approximate memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();

        // Incoming context
        if let Some(ctx) = &self.incoming_context {
            size += ctx.len() * std::mem::size_of::<f32>();
        }

        // Outgoing context (lazy)
        if let Some(ctx) = &self.outgoing_context {
            size += ctx.len() * std::mem::size_of::<f32>();
        }

        size
    }

    /// Apply similarity context to input (Batch/Sequence Mode)
    ///
    /// Computes: Output = Input + (Strength / EmbedDim) * (Input · Context)
    /// Returns Cow::Borrowed if no context is applied, or Cow::Owned if transformed.
    ///
    /// GPU acceleration: Uses GPU backend if available and no fallback policy.
    pub fn apply_context<'a>(&self, input: &'a Array2<f32>) -> Cow<'a, Array2<f32>> {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            if strength == 0.0 || embed_dim == 0 {
                return Cow::Borrowed(input);
            }

            // Expect embed_dim × embed_dim context.
            if input.ncols() != context.nrows() || context.nrows() != context.ncols() {
                return Cow::Borrowed(input);
            }

            // Attempt GPU dispatch if GPU is ready
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                if self.is_gpu_ready() {
                    let result = self.apply_context_gpu_strict(input).unwrap_or_else(|err| {
                        panic!("SharedAttentionContext GPU apply_context failed: {err}")
                    });
                    return Cow::Owned(result);
                }
            }

            // CPU path
            self.apply_context_cpu(input)
        } else {
            Cow::Borrowed(input)
        }
    }

    /// CPU-only path for applying similarity context
    #[inline]
    fn apply_context_cpu<'a>(&self, input: &'a Array2<f32>) -> Cow<'a, Array2<f32>> {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            let scale = strength / (embed_dim as f32).max(1.0);

            // Optimized matrix multiplication: Out = Input · Context (using general_mat_mul)
            let mut out = Array2::<f32>::zeros((input.nrows(), embed_dim));
            general_mat_mul(1.0, input, context, 0.0, &mut out);

            // Mix: Out = Input + Scale * Out
            // Using Zip for efficient element-wise operation
            Zip::from(&mut out).and(input).for_each(|o, &i| {
                let ms = if o.is_finite() { *o } else { 0.0 };
                let xs = if i.is_finite() { i } else { 0.0 };
                *o = xs + scale * ms;
            });
            Cow::Owned(out)
        } else {
            Cow::Borrowed(input)
        }
    }

    /// Apply similarity context to input with pre-allocated output buffer (in-place mode)
    ///
    /// Computes into provided output: Output = Input + (Strength / EmbedDim) * (Input · Context)
    /// This variant avoids allocating intermediate arrays for hot-path optimization.
    /// Returns Ok(true) if transformation was applied, Ok(false) if output equals input.
    ///
    /// GPU acceleration: Uses attached GPU backend in strict mode when available.
    #[inline]
    pub fn apply_context_into(
        &self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<bool> {
        if output.dim() != input.dim() {
            return Err(ModelError::InvalidInput {
                message: "Output buffer dimensions mismatch".to_string(),
            });
        }

        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            if strength == 0.0 || embed_dim == 0 {
                output.assign(input);
                return Ok(false);
            }

            // Expect embed_dim × embed_dim context.
            if input.ncols() != context.nrows() || context.nrows() != context.ncols() {
                output.assign(input);
                return Ok(false);
            }

            // Attempt GPU dispatch if GPU is ready
            #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
            {
                if self.is_gpu_ready() {
                    let result = self.apply_context_gpu_strict(input)?;
                    output.assign(&result);
                    return Ok(true);
                }
            }

            // CPU path
            self.apply_context_into_cpu(input, output)
        } else {
            output.assign(input);
            Ok(false)
        }
    }

    /// CPU-only path for applying similarity context into buffer
    #[inline]
    fn apply_context_into_cpu(
        &self,
        input: &Array2<f32>,
        output: &mut Array2<f32>,
    ) -> Result<bool> {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            let scale = strength / (embed_dim as f32).max(1.0);

            // Use general_mat_mul for optimized in-place mixing:
            // temp = Input · Context, then output = Input + Scale * temp
            // This avoids intermediate allocation by using output as temp buffer
            ndarray::linalg::general_mat_mul(1.0, input, context, 0.0, output);

            // Now output contains Input · Context
            // Mix: output = Input + Scale * output
            Zip::from(output).and(input).for_each(|o, &i| {
                let ms = if o.is_finite() { *o } else { 0.0 };
                let xs = if i.is_finite() { i } else { 0.0 };
                *o = xs + scale * ms;
            });

            Ok(true)
        } else {
            output.assign(input);
            Ok(false)
        }
    }

    /// Strict GPU dispatch for context application using the attached GPU device.
    ///
    /// This path never auto-detects a new backend and never falls back to CPU once
    /// a GPU device is attached to this component.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    fn apply_context_gpu_strict(&self, input: &Array2<f32>) -> Result<Array2<f32>> {
        let device_arc = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ModelError::Backend {
                message: "SharedAttentionContext GPU path requested without attached device"
                    .to_string(),
            })?;

        let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
            message: "Failed to acquire SharedAttentionContext GPU device lock".to_string(),
        })?;

        let mut workspace = crate::domain::layers::components::unified_layer_workspace::UnifiedLayerWorkspace::new_with_backend(
            device.backend(),
        );
        workspace.set_context_buffer_enabled(true);

        let result = self
            .apply_context_gpu_with_workspace(input, &mut workspace, &mut device)
            .map_err(|e| ModelError::Backend {
                message: format!(
                    "SharedAttentionContext GPU workspace dispatch failed: {}",
                    e
                ),
            })?;

        workspace.clear_gpu_buffers_with_device(&mut device);
        Ok(result)
    }

    /// Compute gradients for similarity context
    ///
    /// Returns (final_input_grads, similarity_strength_grad)
    pub fn compute_gradients(
        &self,
        input_original: &Array2<f32>,
        final_input_used_grads: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.is_gpu_ready() {
            return self
                .compute_gradients_gpu(input_original, final_input_used_grads)
                .unwrap_or_else(|err| {
                    panic!("SharedAttentionContext GPU compute_gradients failed: {err}")
                });
        }

        let mut similarity_strength_grad = Array2::zeros((1, 1));
        let mut final_input_grads = final_input_used_grads.clone();

        if let Some(ctx) = &self.incoming_context {
            let embed_dim = input_original.ncols();
            if ctx.nrows() == embed_dim && ctx.ncols() == embed_dim {
                let d = (embed_dim.max(1)) as f32;

                // 1. Gradient for learnable similarity_context_strength.
                // dL/ds = (1/d) * sum(dX' ⊙ (X·S))
                // Using general_mat_mul to avoid intermediate allocation
                let mut mixed = Array2::<f32>::zeros((input_original.nrows(), embed_dim));
                general_mat_mul(1.0, input_original, ctx, 0.0, &mut mixed);

                let mut acc = 0.0f64;
                Zip::from(final_input_used_grads)
                    .and(&mixed)
                    .for_each(|&g, &m| {
                        let gs = if g.is_finite() { g as f64 } else { 0.0 };
                        let ms = if m.is_finite() { m as f64 } else { 0.0 };
                        acc += gs * ms;
                    });
                similarity_strength_grad[[0, 0]] = (acc as f32) / d;

                // 2. Backprop through similarity-context mixing for upstream gradient.
                // dX = dX' + k * dX'·S^T
                let s = self.get_strength();
                let s = if s.is_finite() { s } else { 0.0 };
                let k = s / d;

                if k != 0.0 {
                    // Using general_mat_mul to avoid intermediate allocation
                    let mut corr = Array2::<f32>::zeros((final_input_grads.nrows(), embed_dim));
                    general_mat_mul(1.0, &final_input_grads, &ctx.t(), 0.0, &mut corr);

                    Zip::from(&mut final_input_grads)
                        .and(&corr)
                        .for_each(|g, &c| {
                            let cs = if c.is_finite() { c } else { 0.0 };
                            *g += k * cs;
                        });
                }
            }
        }

        (final_input_grads, similarity_strength_grad)
    }

    /// GPU-aware gradient computation for similarity-context mixing.
    ///
    /// Uses attached GPU device for the two dense matrix multiplications in this
    /// path while preserving the existing analytical gradient behavior.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn compute_gradients_gpu(
        &self,
        input_original: &Array2<f32>,
        final_input_used_grads: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let mut similarity_strength_grad = Array2::zeros((1, 1));
        let mut final_input_grads = final_input_used_grads.clone();

        let Some(ctx) = &self.incoming_context else {
            return Ok((final_input_grads, similarity_strength_grad));
        };

        let embed_dim = input_original.ncols();
        if embed_dim == 0 || ctx.nrows() != embed_dim || ctx.ncols() != embed_dim {
            return Ok((final_input_grads, similarity_strength_grad));
        }

        let device_arc = self
            .gpu_device
            .as_ref()
            .ok_or_else(|| ModelError::Backend {
                message:
                    "SharedAttentionContext::compute_gradients_gpu requires an attached GPU device"
                        .to_string(),
            })?;

        let d = (embed_dim.max(1)) as f32;

        let mixed = gpu_gemm_with_attached_device(
            device_arc,
            input_original,
            ctx,
            input_original.nrows(),
            embed_dim,
            embed_dim,
            false,
            false,
            "SharedAttentionContext::compute_gradients mixed",
        )?;

        let mut acc = 0.0f64;
        Zip::from(final_input_used_grads)
            .and(&mixed)
            .for_each(|&g, &m| {
                let gs = if g.is_finite() { g as f64 } else { 0.0 };
                let ms = if m.is_finite() { m as f64 } else { 0.0 };
                acc += gs * ms;
            });
        similarity_strength_grad[[0, 0]] = (acc as f32) / d;

        let s = self.get_strength();
        let s = if s.is_finite() { s } else { 0.0 };
        let k = s / d;
        if k != 0.0 {
            let corr = gpu_gemm_with_attached_device(
                device_arc,
                &final_input_grads,
                ctx,
                final_input_grads.nrows(),
                embed_dim,
                embed_dim,
                false,
                true,
                "SharedAttentionContext::compute_gradients corr",
            )?;

            Zip::from(&mut final_input_grads)
                .and(&corr)
                .for_each(|g, &c| {
                    let cs = if c.is_finite() { c } else { 0.0 };
                    *g += k * cs;
                });
        }

        Ok((final_input_grads, similarity_strength_grad))
    }

    /// Update outgoing context in step mode (optimized for 1D inference)
    ///
    /// This is a specialized version for single-vector updates during inference.
    /// It mirrors the batch method but optimized for 1D vectors: avoids 2D view creation
    /// and computes covariance/denominator directly.
    ///
    /// Parameters:
    /// - `input_step`: Single input vector at current timestep
    /// - `output_step`: Single output vector at current timestep
    pub fn update_outgoing_context_step(
        &mut self,
        input_step: &ndarray::ArrayView1<f32>,
        output_step: &ndarray::ArrayView1<f32>,
        embed_dim_config: usize,
    ) {
        let rate = self.similarity_update_rate.clamp(0.0, 1.0);
        if rate <= 0.0 {
            return;
        }

        let embed_dim = input_step
            .len()
            .min(output_step.len())
            .min(embed_dim_config);
        if embed_dim == 0 {
            return;
        }

        // Lazy allocation for outgoing context
        if self.outgoing_context.is_none()
            || self.outgoing_context.as_ref().unwrap().shape() != [embed_dim, embed_dim]
        {
            self.outgoing_context = Some(Array2::zeros((embed_dim, embed_dim)));
        }

        // Handle non-finite values (matching batch method)
        let mut x = ndarray::Array1::zeros(embed_dim);
        let mut y = ndarray::Array1::zeros(embed_dim);
        for i in 0..embed_dim {
            x[i] = if input_step[i].is_finite() {
                input_step[i]
            } else {
                0.0
            };
            y[i] = if output_step[i].is_finite() {
                output_step[i]
            } else {
                0.0
            };
        }

        // Compute means (matching batch method)
        let mean_x = x.sum() / (embed_dim as f32).max(1.0);
        let mean_y = y.sum() / (embed_dim as f32).max(1.0);

        // Center data (matching batch method)
        x.iter_mut().for_each(|v| *v -= mean_x);
        y.iter_mut().for_each(|v| *v -= mean_y);

        // Compute norms (matching batch method)
        let norm_x_sq: f32 = x.iter().map(|&v| v * v).sum();
        let norm_y_sq: f32 = y.iter().map(|&v| v * v).sum();

        let norm_x = norm_x_sq.sqrt();
        let norm_y = norm_y_sq.sqrt();

        let tanh = crate::domain::richards::RichardsCurve::tanh(false);
        let outgoing_context = self.outgoing_context.as_mut().unwrap();

        // Update using covariance/denominator approach (matching batch method)
        // Direct computation: sim_ij = tanh((x[i] * y[j]) / (norm_x * norm_y))
        let denom = norm_x * norm_y;

        for i in 0..embed_dim {
            for j in 0..embed_dim {
                let cov = x[i] * y[j];
                let sim_raw = if denom > 1e-12 { cov / denom } else { 0.0 };
                let sim = if sim_raw.is_finite() {
                    tanh.forward_scalar_f32(sim_raw)
                } else {
                    0.0
                };
                outgoing_context[[i, j]] = (1.0 - rate) * outgoing_context[[i, j]] + rate * sim;
            }
        }
    }

    /// Apply similarity context to input (Step Mode)
    ///
    /// Computes in-place: Output = Input + (Strength / EmbedDim) * (Context^T · Input)
    /// Note: For vector-matrix product, we use Context^T · Input equivalent to Input · Context
    pub fn apply_step_into(
        &self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut ndarray::Array1<f32>,
    ) {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();

            if strength == 0.0 {
                output.assign(input);
                return;
            }

            let embed_dim = input.len();
            if embed_dim != context.nrows() || context.nrows() != context.ncols() {
                output.assign(input);
                return;
            }

            let scale = strength / (embed_dim as f32).max(1.0);

            // Step 1: y = scale * context^T * input
            // We use general_mat_vec_mul which computes y = alpha * A * x + beta * y
            // Here: output = scale * context^T * input
            ndarray::linalg::general_mat_vec_mul(scale, &context.t(), input, 0.0, output);

            // Step 2: output += input
            Zip::from(output).and(input).for_each(|o, &i| *o += i);
        } else {
            output.assign(input);
        }
    }

    // --- GPU Support (Phase 5.3) ---

    /// GPU-accelerated attention context application with workspace-managed buffers.
    ///
    /// Computes on GPU:
    /// `output = input + (strength / embed_dim) * (input @ context)`
    ///
    /// This variant reuses `UnifiedLayerWorkspace` GPU buffers to avoid per-call
    /// allocate/deallocate churn.
    ///
    /// # Arguments
    /// * `activation` - Input activation matrix (batch_size, embed_dim)
    /// * `workspace` - Shared workspace providing reusable GPU buffers
    /// * `gpu_device` - GPU device context
    ///
    /// # Returns
    /// GPU result downloaded to CPU, or error if GPU operation fails.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn apply_context_gpu_with_workspace(
        &self,
        activation: &Array2<f32>,
        workspace: &mut crate::domain::layers::components::unified_layer_workspace::UnifiedLayerWorkspace,
        gpu_device: &mut crate::domain::compute::GpuDevice,
    ) -> std::result::Result<Array2<f32>, Box<dyn std::error::Error>> {
        let batch_size = activation.nrows();
        let embed_dim = activation.ncols();

        if batch_size == 0 || embed_dim == 0 {
            return Err("Empty activation matrix".into());
        }

        let Some(context) = self.incoming_context.as_ref() else {
            return Ok(activation.clone());
        };
        let strength = self.get_strength();
        if strength == 0.0 {
            return Ok(activation.clone());
        }
        if context.nrows() != embed_dim || context.ncols() != embed_dim {
            return Ok(activation.clone());
        }

        workspace.set_context_buffer_enabled(true);
        workspace
            .ensure_gpu_capacity(gpu_device, batch_size, embed_dim, embed_dim)
            .map_err(|e| format!("Workspace GPU capacity ensure failed: {}", e))?;

        let mut gpu_input = workspace
            .gpu_norm1_out()
            .ok_or("Workspace GPU input buffer is not allocated")?;
        let mut gpu_output = workspace
            .gpu_temporal_out()
            .ok_or("Workspace GPU output buffer is not allocated")?;
        let mut gpu_context = workspace
            .gpu_context_buffer()
            .ok_or("Workspace GPU context buffer is not allocated")?;

        let activation_flat: Cow<'_, [f32]> = match activation.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(activation.iter().copied().collect()),
        };
        let context_flat: Cow<'_, [f32]> = match context.as_slice() {
            Some(slice) => Cow::Borrowed(slice),
            None => Cow::Owned(context.iter().copied().collect()),
        };

        gpu_device
            .upload(activation_flat.as_ref(), &mut gpu_input)
            .map_err(|e| format!("GPU upload (activation) failed: {}", e))?;
        gpu_device
            .upload(context_flat.as_ref(), &mut gpu_context)
            .map_err(|e| format!("GPU upload (context) failed: {}", e))?;

        gpu_device
            .apply_attention_context(
                &gpu_input,
                &gpu_context,
                &mut gpu_output,
                strength,
                batch_size,
                embed_dim,
            )
            .map_err(|e| format!("GPU attention context apply failed: {}", e))?;

        let mut result_flat = vec![0.0f32; batch_size * embed_dim];
        gpu_device
            .download(&gpu_output, &mut result_flat)
            .map_err(|e| format!("GPU download failed: {}", e))?;

        let result = Array2::from_shape_vec((batch_size, embed_dim), result_flat)
            .map_err(|e| format!("Failed to reshape result: {}", e))?;

        Ok(result)
    }

    /// GPU-accelerated attention context application.
    ///
    /// Convenience wrapper for one-off calls. This allocates a temporary
    /// workspace internally, so repeated calls should prefer
    /// `apply_context_gpu_with_workspace` for better memory efficiency.
    #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
    pub fn apply_context_gpu(
        &mut self,
        activation: &Array2<f32>,
        gpu_device: &mut crate::domain::compute::GpuDevice,
    ) -> std::result::Result<Array2<f32>, Box<dyn std::error::Error>> {
        let mut workspace =
            crate::domain::layers::components::unified_layer_workspace::UnifiedLayerWorkspace::new_with_backend(
                gpu_device.backend(),
            );
        workspace.set_context_buffer_enabled(true);
        let result = self.apply_context_gpu_with_workspace(activation, &mut workspace, gpu_device);
        workspace.clear_gpu_buffers_with_device(gpu_device);
        result
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::SharedAttentionContext;

    #[test]
    fn test_outgoing_context_lazy_allocation() {
        let mut ctx = SharedAttentionContext::new();

        // Initially no allocation
        assert!(ctx.outgoing_context.is_none());
        assert!(ctx.get_outgoing_context().is_none());

        // Allocate on first update
        let input = Array2::from_elem((10, 128), 0.5f32);
        let output = Array2::from_elem((10, 128), 0.3f32);
        ctx.update_outgoing_context(&input.view(), &output.view(), 128);

        // Now it's allocated
        assert!(ctx.outgoing_context.is_some());
        assert_eq!(ctx.outgoing_context.as_ref().unwrap().shape(), [128, 128]);

        // Second update with same dims should reuse
        let old_ptr = ctx.outgoing_context.as_ref().unwrap().as_ptr();
        ctx.update_outgoing_context(&input.view(), &output.view(), 128);
        let new_ptr = ctx.outgoing_context.as_ref().unwrap().as_ptr();
        assert_eq!(old_ptr, new_ptr); // Same allocation
    }

    #[test]
    fn test_apply_context_into_vs_apply_context() {
        let mut ctx = SharedAttentionContext::new();
        ctx.set_strength(0.5);
        ctx.set_incoming_context(Some(&Array2::from_elem((64, 64), 0.1f32)));

        let input = Array2::from_elem((10, 64), 1.0f32);

        // Using apply_context (Cow variant)
        let cow_result = ctx.apply_context(&input);
        let cow_output = cow_result.as_ref();

        // Using apply_context_into (in-place variant)
        let mut into_output = Array2::zeros(input.dim());
        let applied = ctx.apply_context_into(&input, &mut into_output).unwrap();

        assert!(applied); // Should have applied transformation
        assert_eq!(cow_output.shape(), into_output.shape());

        // Results should be identical (within floating point tolerance)
        for (a, b) in cow_output.iter().zip(into_output.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn set_incoming_context_reuse_keeps_allocation_when_shape_matches() {
        let mut ctx = SharedAttentionContext::new();
        let first = Array2::from_elem((2, 2), 1.0f32);
        let second = Array2::from_elem((2, 2), 2.0f32);

        ctx.set_incoming_context_reuse(Some(&first));
        let ptr_before = ctx
            .incoming_context
            .as_ref()
            .expect("incoming context should exist")
            .as_ptr();

        ctx.set_incoming_context_reuse(Some(&second));
        let stored = ctx
            .incoming_context
            .as_ref()
            .expect("incoming context should exist");
        let ptr_after = stored.as_ptr();

        assert_eq!(ptr_before, ptr_after);
        assert_eq!(stored[[0, 0]], 2.0);
    }

    #[test]
    fn set_incoming_context_reuse_reallocates_when_shape_changes() {
        let mut ctx = SharedAttentionContext::new();
        let first = Array2::from_elem((2, 2), 1.0f32);
        let second = Array2::from_elem((3, 3), 2.0f32);

        ctx.set_incoming_context_reuse(Some(&first));
        let ptr_before = ctx
            .incoming_context
            .as_ref()
            .expect("incoming context should exist")
            .as_ptr();

        ctx.set_incoming_context_reuse(Some(&second));
        let stored = ctx
            .incoming_context
            .as_ref()
            .expect("incoming context should exist");
        let ptr_after = stored.as_ptr();

        assert_ne!(ptr_before, ptr_after);
        assert_eq!(stored.dim(), (3, 3));
    }

    #[test]
    fn test_general_mat_mul_optimization_numerical_equivalence() {
        use ndarray::Array2;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_strength(0.5);

        // Create a test context and input
        let context = Array2::from_shape_fn((8, 8), |(i, j)| {
            ((i as f32 + 1.0) * (j as f32 + 1.0) * 0.1).sin()
        });
        ctx.set_incoming_context(Some(&context));

        let input = Array2::from_shape_fn((16, 8), |(i, j)| {
            ((i as f32) * 0.2 + (j as f32) * 0.1).cos()
        });

        // Test apply_context (uses optimized general_mat_mul)
        let result = ctx.apply_context(&input);

        // Verify numerical properties
        assert_eq!(result.shape(), input.shape());

        // All values should be finite
        for &val in result.iter() {
            assert!(val.is_finite(), "Non-finite value in result");
        }

        // Test apply_context_into for in-place variant
        let mut into_result = Array2::zeros(input.dim());
        let _ = ctx.apply_context_into(&input, &mut into_result).unwrap();

        // Both methods should produce identical results
        for (a, b) in result.iter().zip(into_result.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Mismatch between apply_context and apply_context_into: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_gradient_computation_general_mat_mul() {
        use ndarray::Array2;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_strength(0.3);

        let context = Array2::from_elem((16, 16), 0.1f32);
        ctx.set_incoming_context(Some(&context));

        let input = Array2::from_elem((32, 16), 0.5f32);
        let grads = Array2::from_elem((32, 16), 0.2f32);

        let (input_grads, strength_grad) = ctx.compute_gradients(&input, &grads);

        // Check dimensions
        assert_eq!(input_grads.shape(), input.shape());
        assert_eq!(strength_grad.shape(), [1, 1]);

        // Check all values are finite
        for &val in input_grads.iter() {
            assert!(val.is_finite(), "Non-finite gradient value");
        }
        for &val in strength_grad.iter() {
            assert!(val.is_finite(), "Non-finite strength gradient");
        }
    }

    #[test]
    fn test_update_outgoing_context_step_basic() {
        use ndarray::Array1;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_update_rate(0.5);

        let input = Array1::from_elem(8, 0.5f32);
        let output = Array1::from_elem(8, 0.3f32);

        // Initially no allocation
        assert!(ctx.outgoing_context.is_none());

        // Update should allocate
        ctx.update_outgoing_context_step(&input.view(), &output.view(), 8);

        // Now should be allocated
        assert!(ctx.outgoing_context.is_some());
        assert_eq!(ctx.outgoing_context.as_ref().unwrap().shape(), [8, 8]);

        // All values should be finite
        for &val in ctx.outgoing_context.as_ref().unwrap().iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_update_outgoing_context_step_reuse_allocation() {
        use ndarray::Array1;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_update_rate(0.5);

        let input = Array1::from_elem(16, 0.5f32);
        let output = Array1::from_elem(16, 0.3f32);

        // First update
        ctx.update_outgoing_context_step(&input.view(), &output.view(), 16);
        let ptr_before = ctx.outgoing_context.as_ref().unwrap().as_ptr();

        // Second update with same dims should reuse
        ctx.update_outgoing_context_step(&input.view(), &output.view(), 16);
        let ptr_after = ctx.outgoing_context.as_ref().unwrap().as_ptr();

        assert_eq!(ptr_before, ptr_after); // Same allocation
    }

    #[test]
    fn test_update_outgoing_context_step_handles_nonfinite() {
        use ndarray::Array1;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_update_rate(0.5);

        let mut input = Array1::from_elem(8, 0.5f32);
        let mut output = Array1::from_elem(8, 0.3f32);

        // Insert some NaN and Inf values
        input[2] = f32::NAN;
        output[5] = f32::INFINITY;

        // Should not panic and produce finite results
        ctx.update_outgoing_context_step(&input.view(), &output.view(), 8);

        // All results should be finite
        for &val in ctx.outgoing_context.as_ref().unwrap().iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_update_outgoing_context_step_zero_vectors() {
        use ndarray::Array1;

        let mut ctx = SharedAttentionContext::new();
        ctx.set_update_rate(0.5);

        let input = Array1::from_elem(8, 0.0f32);
        let output = Array1::from_elem(8, 0.0f32);

        // Should handle zero vectors gracefully
        ctx.update_outgoing_context_step(&input.view(), &output.view(), 8);

        // Should not panic, result may be None or all zeros
        if let Some(out_ctx) = ctx.outgoing_context.as_ref() {
            for &val in out_ctx.iter() {
                assert!(val.is_finite());
            }
        }
    }
}

// ============================================================================
// GPU Component Implementation (Phase 5.6)
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl GpuComponent for SharedAttentionContext {
    fn set_gpu_device(&mut self, device: Arc<Mutex<GpuDevice>>) {
        self.gpu_device = Some(device);
    }

    fn enable_gpu_auto_detect(&mut self) -> Result<()> {
        let device = GpuDevice::auto_detect()?;
        self.gpu_device = Some(Arc::new(Mutex::new(device)));
        Ok(())
    }

    fn is_gpu_ready(&self) -> bool {
        self.gpu_device.is_some()
    }

    fn gpu_backend_name(&self) -> Option<&'static str> {
        self.gpu_device
            .as_ref()
            .and_then(|device_arc| match device_arc.lock() {
                Ok(device) => Some(device.backend().as_str()),
                Err(_) => None,
            })
    }

    fn gpu_device(&self) -> Option<Arc<Mutex<GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        _seq_len: usize,
    ) -> Result<()> {
        if let Some(device_arc) = &self.gpu_device {
            let mut device = device_arc.lock().map_err(|_| ModelError::Backend {
                message: "Failed to lock GPU device for SharedAttentionContext capacity allocation"
                    .to_string(),
            })?;

            // Pre-allocate buffers for attention context operations
            let _ = device.allocate_f32(batch_size * embed_dim)?; // input
            let _ = device.allocate_f32(batch_size * embed_dim)?; // activation
            let _ = device.allocate_f32(embed_dim * embed_dim)?; // context matrix
            let _ = device.allocate_f32(batch_size * batch_size)?; // similarity matrix
            let _ = device.allocate_f32(batch_size * batch_size)?; // softmax output

            Ok(())
        } else {
            Err(ModelError::Backend {
                message: "GPU device not attached to SharedAttentionContext. Call enable_gpu_auto_detect() first.".to_string(),
            })
        }
    }
}
