use std::borrow::Cow;

use ndarray::{Array2, ArrayBase, Data, Ix2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::domain::{
    network::Layer,
    richards::{RichardsCurve, Variant},
};

// EMA smoothing factor for gradient norm tracking inside RichardsNorm
const EMA_BETA_GRAD: f32 = 0.9;

/// Richards-based Normalization with Dynamic Parameter Adjustments
///
/// Element-wise normalization using Richards curve with adaptive parameter scaling,
/// followed by per-channel scale `gamma` and bias `bias`:
///
///   y = Richards_adaptive(scale · x) ⊙ gamma + bias
///
/// Dynamic adjustments based on activation statistics (Frobenius norm):
/// - **Adaptive Temperature**: Scales temperature by activation magnitude ratio (inspired by
///   Dynamic Tanh's α parameter for data-dependent scaling)
/// - **Dynamic Midpoint**: Centers Richards curve around activation distribution
/// - **Adaptive Asymmetry**: Adjusts β based on activation variance
/// - **Per-feature Scaling**: γ and β provide feature-specific normalization
///
/// Key advantages over traditional normalization:
/// - No hard clipping or clamping - smooth, differentiable parameter adjustments
/// - Data-dependent curve adaptation instead of forcing data to fit fixed curves
/// - Learns shape parameters (nu, k, beta, temperature, scale) + per-feature affine (γ, β)
/// - Lightweight alternative without expensive batch statistics computation
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RichardsNorm {
    /// Cached input for backward
    cached_input: Option<Array2<f32>>,

    /// Cached Richards curve with dynamic adjustments applied during the last `forward`.
    ///
    /// This ensures gradients are computed against the exact curve used in the forward pass.
    #[serde(skip_serializing, skip_deserializing)]
    cached_adjusted_richards: Option<RichardsCurve>,

    /// Richards curve for tanh-like computation with learnable parameters and per-feature
    /// transformations
    richards: RichardsCurve,

    /// Exponential moving average of parameter gradient norm (for stability-aware adjustments)
    grad_norm_ema: Option<f32>,

    /// Optional GPU device for accelerated normalization
    #[serde(skip_serializing, skip_deserializing)]
    gpu_device: Option<std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>>,
}

impl RichardsNorm {
    pub fn cached_adjusted_richards(&self) -> Option<&RichardsCurve> {
        self.cached_adjusted_richards.as_ref()
    }

    #[inline]
    pub fn clear_gpu_device(&mut self) {
        self.gpu_device = None;
    }

    /// Create a new RichardsNorm layer
    pub fn new(embedding_dim: usize) -> Self {
        // Start with a Richards curve in Tanh variant.
        //
        // DynamicTanh-style normalization assumes an odd, symmetric squashing function.
        // To preserve this property, we keep the symmetry-critical params fixed:
        // - nu = 1 and beta = 1 so the underlying logistic satisfies σ(-x)=1-σ(x)
        // - m = 0 and shift = 0 so the curve remains centered/odd
        //
        // The *dynamic* part is expressed via temperature/scale (input sharpness), and
        // the learned per-feature affine (gamma/bias).
        let mut richards = RichardsCurve::new_learnable(Variant::Tanh);

        // Fix symmetry/oddness-critical params.
        richards.nu = Some(1.0);
        richards.beta = Some(1.0);
        richards.m = Some(0.0);
        richards.shift = Some(0.0);
        richards.output_gain = Some(1.0);
        richards.output_bias = Some(0.0);

        richards.nu_learnable = false;
        richards.beta_learnable = false;
        richards.m_learnable = false;
        richards.shift_learnable = false;
        richards.output_gain_learnable = false;
        richards.output_bias_learnable = false;

        // Keep sharpness learnable/dynamic.
        richards.k = None;
        richards.temperature = None;
        richards.scale = None;
        richards.k_learnable = true;
        richards.temperature_learnable = true;
        richards.scale_learnable = true;

        // Initialize learned parameters (only the learnable subset matters).
        richards.learned_k = Some(1.0);
        richards.learned_temperature = Some(1.0);
        richards.learned_scale = Some(1.0);

        // Enable per-feature transformations (gamma, bias) for normalization
        richards.enable_per_feature_transform(embedding_dim);

        // Validate that RichardsCurve has exactly the expected learnable parameters.
        // RichardsNorm expects: k, temperature, scale (3 scalar parameters), plus per-feature
        // gamma/bias.
        let expected_learnable = [false, true, false, false, true, false, false, true, false]; // nu, k, m, beta, temp, gain, bias, scale, shift
        let actual_learnable = [
            richards.nu_learnable,
            richards.k_learnable,
            richards.m_learnable,
            richards.beta_learnable,
            richards.temperature_learnable,
            richards.output_gain_learnable,
            richards.output_bias_learnable,
            richards.scale_learnable,
            richards.shift_learnable,
        ];

        assert_eq!(
            expected_learnable, actual_learnable,
            "RichardsNorm expects specific learnable parameter configuration: nu, k, beta, temperature, scale. Found different configuration."
        );

        Self {
            cached_input: None,
            cached_adjusted_richards: None,
            richards,
            grad_norm_ema: None,
            gpu_device: None,
        }
    }

    /// Apply dynamic parameter adjustments based on activation statistics
    /// Returns the adjusted parameters for restoration
    fn compute_dynamic_adjustments<S>(
        &self,
        input: &ArrayBase<S, Ix2>,
    ) -> (Option<f64>, Option<f64>, Option<f64>)
    where
        S: Data<Elem = f32>,
    {
        // Compute Frobenius norm for scale-aware adjustments
        // We only need sum of squares for Frobenius norm.
        // Mean and variance are currently unused but could be added if centering is needed.
        let sum_sq = if let Some(slice) = input.as_slice() {
            if slice.len() > 1024 {
                slice
                    .par_iter()
                    .map(|&x| {
                        let val = x as f64;
                        val * val
                    })
                    .sum::<f64>()
            } else {
                slice
                    .iter()
                    .map(|&x| {
                        let val = x as f64;
                        val * val
                    })
                    .sum::<f64>()
            }
        } else {
            input
                .iter()
                .map(|&x| {
                    let val = x as f64;
                    val * val
                })
                .sum::<f64>()
        };

        let frob_norm = sum_sq.sqrt();

        // Target scale for normalization (empirical value, can be tuned)
        let target_scale = (input.len() as f64).sqrt(); // Approximate RMS norm

        // Adaptive temperature scaling (inspired by DyT's α parameter)
        // Higher activation scale → sharper transitions (higher temperature)
        // Additionally, damp aggressiveness when recent gradient norms are large
        let scale_ratio = (frob_norm / target_scale).clamp(1e-6, 1e6);
        let grad_ema = self.grad_norm_ema.unwrap_or(1.0) as f64;
        // Stability factor reduces temperature when gradients are high
        let stability_factor = 1.0 / (1.0 + 0.25 * grad_ema.max(1e-6));
        // Use a gentle power to avoid extreme sharpness when activations are small.
        // In this codebase, larger temperature => softer curve (input divided by T).
        let temp_adjustment = scale_ratio.powf(0.25) * stability_factor;
        let curr_temp = self.richards.effective_temperature();
        let adjusted_temp = Some((curr_temp * temp_adjustment).clamp(0.25, 5.0));

        // For DynamicTanh-style normalization we keep the curve centered/odd and symmetric.
        // So we do NOT dynamically shift the midpoint (m) and we do NOT change asymmetry (beta).
        // We return the fixed values for clarity.
        let adjusted_m = Some(0.0);
        let adjusted_beta = Some(1.0);

        (adjusted_temp, adjusted_m, adjusted_beta)
    }

    /// Forward normalization with dynamic parameter adjustments (mutable for training)
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward (needed for gradient computation)
        self.cached_input = Some(input.clone());

        // For training, we also need to cache the adjusted curve parameters.
        // However, since we now compute adjustments per-row (per-token),
        // a single "adjusted_richards" doesn't capture the full state.
        // But the original implementation cached it.
        // The implementation in `normalize_impl` doesn't update `cached_adjusted_richards`.
        // This implies the cached version is only an approximation or training uses batch-stats?
        // Actually, `normalize_impl` is what does the work.
        // Let's call it.
        let out = self.normalize_impl(input);

        // Update the cached curve using batch statistics for backward pass approximation
        // (This preserves the original behavior for training, though strictly speaking
        // gradients should be per-token adjusted).
        let (adj_temp, _, _) = self.compute_dynamic_adjustments(input);
        if let Some(t) = adj_temp {
            let mut curve = self.richards.clone();
            curve.learned_temperature = Some(t);
            self.cached_adjusted_richards = Some(curve);
        } else {
            self.cached_adjusted_richards = Some(self.richards.clone());
        }

        out
    }

    /// Forward normalization into a pre-allocated output matrix.
    ///
    /// This mirrors `normalize` behavior (including cache updates for backward) while
    /// avoiding allocation of the normalized output tensor.
    pub fn normalize_into(&mut self, input: &Array2<f32>, output: &mut Array2<f32>) {
        assert_eq!(
            input.dim(),
            output.dim(),
            "normalize_into expects output with same shape as input"
        );

        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            let gpu_out = self.forward_gpu(input).unwrap_or_else(|err| {
                panic!(
                    "RichardsNorm GPU normalize_into failed (GPU attached, no fallback): {}",
                    err
                )
            });
            output.assign(&gpu_out);
            return;
        }

        self.cached_input = Some(input.clone());

        ndarray::Zip::from(output.outer_iter_mut())
            .and(input.outer_iter())
            .par_for_each(|mut out_row, in_row| {
                let in_row_2d = in_row.insert_axis(ndarray::Axis(0));
                let (adjusted_temp, adjusted_m, adjusted_beta) =
                    self.compute_dynamic_adjustments(&in_row_2d);

                let in_slice = in_row.as_slice().unwrap();
                let out_slice = out_row.as_slice_mut().unwrap();
                self.richards.forward_into_f32_with_overrides(
                    in_slice,
                    out_slice,
                    adjusted_temp,
                    adjusted_m,
                    adjusted_beta,
                );
            });

        let (adj_temp, _, _) = self.compute_dynamic_adjustments(input);
        if let Some(t) = adj_temp {
            let mut curve = self.richards.clone();
            curve.learned_temperature = Some(t);
            self.cached_adjusted_richards = Some(curve);
        } else {
            self.cached_adjusted_richards = Some(self.richards.clone());
        }
    }

    /// Normalize into a pre-allocated slice (no allocations)
    pub fn normalize_into_f32(&self, input: &[f32], output: &mut [f32]) {
        use ndarray::ArrayView1;
        // Wrap slice as ArrayView1 then promote to 2D (1, N) for adjustment calc
        let in_view = ArrayView1::from(input);
        let in_2d = in_view.insert_axis(ndarray::Axis(0));

        let (adjusted_temp, adjusted_m, adjusted_beta) = self.compute_dynamic_adjustments(&in_2d);

        self.richards.forward_into_f32_with_overrides(
            input,
            output,
            adjusted_temp,
            adjusted_m,
            adjusted_beta,
        );
    }

    /// Forward normalization with dynamic parameter adjustments (immutable for inference)
    pub fn normalize_immutable(&self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize_impl(input)
    }

    /// Forward normalization with parameter overrides (for testing/parity verification)
    pub fn normalize_with_overrides(
        &self,
        input: &Array2<f32>,
        temp_override: Option<f64>,
        m_override: Option<f64>,
        beta_override: Option<f64>,
    ) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(input.dim());

        // We use the overrides for ALL rows if provided.
        // This is useful when we want to force specific behavior during verification.

        ndarray::Zip::from(out.outer_iter_mut())
            .and(input.outer_iter())
            .par_for_each(|mut out_row, in_row| {
                let in_row_2d = in_row.insert_axis(ndarray::Axis(0));

                // If overrides are provided, use them.
                // Otherwise calculate dynamic adjustments as usual.
                let (dyn_temp, dyn_m, dyn_beta) = self.compute_dynamic_adjustments(&in_row_2d);

                let effective_temp = temp_override.or(dyn_temp);
                let effective_m = m_override.or(dyn_m);
                let effective_beta = beta_override.or(dyn_beta);

                let in_slice = in_row.as_slice().unwrap();
                let out_slice = out_row.as_slice_mut().unwrap();

                self.richards.forward_into_f32_with_overrides(
                    in_slice,
                    out_slice,
                    effective_temp,
                    effective_m,
                    effective_beta,
                );
            });

        out
    }

    /// Forward normalization for a single step (streaming)
    /// avoids 2D reshapes and allocations where possible
    pub fn normalize_step(
        &self,
        input: &ndarray::Array1<f32>,
        overrides: Option<(Option<f64>, Option<f64>, Option<f64>)>,
    ) -> ndarray::Array1<f32> {
        let mut out = ndarray::Array1::<f32>::zeros(input.raw_dim());
        self.normalize_step_into(&input.view(), &mut out, overrides);
        out
    }

    /// Streaming normalization with zero allocation
    pub fn normalize_step_into(
        &self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut ndarray::Array1<f32>,
        overrides: Option<(Option<f64>, Option<f64>, Option<f64>)>,
    ) {
        // Wrap as 2D for compute_dynamic_adjustments (1, D)
        // This is a view, so it's cheap.
        let in_2d = input.insert_axis(ndarray::Axis(0));

        let (dyn_temp, dyn_m, dyn_beta) = self.compute_dynamic_adjustments(&in_2d);

        let (temp_ov, m_ov, beta_ov) = overrides.unwrap_or((None, None, None));

        let effective_temp = temp_ov.or(dyn_temp);
        let effective_m = m_ov.or(dyn_m);
        let effective_beta = beta_ov.or(dyn_beta);

        let in_slice = input.as_slice().unwrap();
        let out_slice = output.as_slice_mut().unwrap();

        self.richards.forward_into_f32_with_overrides(
            in_slice,
            out_slice,
            effective_temp,
            effective_m,
            effective_beta,
        );
    }

    /// Internal normalization implementation
    fn normalize_impl(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(input.dim());

        // Process per row for consistency between batch and streaming modes.
        // RichardsNorm dynamic adjustments are activation-dependent.
        // If we compute them over the whole batch, the output for a token depends on other tokens in the batch.
        // This causes divergence with streaming mode where tokens are processed one by one.
        // By iterating rows, we ensure each token is normalized independently (like LayerNorm).

        ndarray::Zip::from(out.outer_iter_mut())
            .and(input.outer_iter())
            .par_for_each(|mut out_row, in_row| {
                // in_row is ArrayView1.
                // We need to wrap it as 2D for compute_dynamic_adjustments.
                // insert_axis returns ArrayView2.
                let in_row_2d = in_row.insert_axis(ndarray::Axis(0));

                let (adjusted_temp, adjusted_m, adjusted_beta) =
                    self.compute_dynamic_adjustments(&in_row_2d);

                // Now apply curve to the row slice
                // as_slice() is safe because Array2 rows are contiguous.
                let in_slice = in_row.as_slice().unwrap();
                let out_slice = out_row.as_slice_mut().unwrap();

                self.richards.forward_into_f32_with_overrides(
                    in_slice,
                    out_slice,
                    adjusted_temp,
                    adjusted_m,
                    adjusted_beta,
                );
            });

        out
    }

    #[inline]
    fn apply_gradients_from_iter<'a, I>(
        &mut self,
        gradients: I,
        learning_rate: f32,
    ) -> Result<(), crate::common::errors::ModelError>
    where
        I: IntoIterator<Item = &'a Array2<f32>>,
    {
        let grads: Vec<&Array2<f32>> = gradients.into_iter().collect();

        // Track gradient norms for stability without allocating temporaries.
        let total_norm_sq: f32 = grads
            .iter()
            .map(|g| g.iter().map(|&x| x * x).sum::<f32>())
            .sum();
        let total_norm = total_norm_sq.sqrt();

        // Update EMA of gradient norm
        if let Some(ema) = self.grad_norm_ema {
            self.grad_norm_ema = Some(EMA_BETA_GRAD * ema + (1.0 - EMA_BETA_GRAD) * total_norm);
        } else {
            self.grad_norm_ema = Some(total_norm);
        }

        // Flatten all gradients (scalars and vectors) into a single vector for RichardsCurve.
        let total_values: usize = grads.iter().map(|g| g.len()).sum();
        let mut curve_grads: Vec<f64> = Vec::with_capacity(total_values);
        for g in grads {
            curve_grads.extend(g.iter().map(|&val| val as f64));
        }

        self.richards.step(&curve_grads, learning_rate as f64);
        Ok(())
    }

    /// Apply borrowed/owned Cow gradients without forcing owned conversion at call sites.
    pub fn apply_gradients_ref(
        &mut self,
        gradients: &[Cow<'_, Array2<f32>>],
        learning_rate: f32,
    ) -> Result<(), crate::common::errors::ModelError> {
        self.apply_gradients_from_iter(gradients.iter().map(|g| g.as_ref()), learning_rate)
    }
}

impl Layer for RichardsNorm {
    fn layer_type(&self) -> &str {
        "RichardsNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            return self.forward_gpu(input).unwrap_or_else(|err| {
                panic!(
                    "RichardsNorm GPU forward failed (GPU attached, no fallback): {}",
                    err
                )
            });
        }

        self.normalize(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            return self.backward_gpu(grads, lr).unwrap_or_else(|err| {
                panic!(
                    "RichardsNorm GPU backward failed (GPU attached, no fallback): {}",
                    err
                )
            });
        }

        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before backward");
        let (grad_input, param_grads) = self.compute_gradients(input, grads);
        let _ = self.apply_gradients(&param_grads, lr);
        grad_input
    }

    fn parameters(&self) -> usize {
        self.richards.weights().len()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        #[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
        if self.gpu_device.is_some() {
            return self
                .compute_gradients_gpu(output_grads)
                .unwrap_or_else(|err| {
                    panic!(
                        "RichardsNorm GPU compute_gradients failed (GPU attached, no fallback): {}",
                        err
                    )
                });
        }

        self.compute_gradients_cpu_impl(output_grads)
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> Result<(), crate::common::errors::ModelError> {
        self.apply_gradients_from_iter(gradients.iter(), learning_rate)
    }

    fn weight_norm(&self) -> f32 {
        self.richards
            .weights()
            .iter()
            .map(|&w| (w as f32) * (w as f32))
            .sum::<f32>()
            .sqrt()
    }

    fn zero_gradients(&mut self) {
        // RichardsCurve doesn't hold gradients, they are passed in apply_gradients
    }
}

impl RichardsNorm {
    fn compute_gradients_cpu_impl(
        &self,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        // Use the adjusted curve from the last forward (training path) when available.
        // This keeps gradients consistent with dynamic parameter adjustments.
        let richards = self
            .cached_adjusted_richards
            .as_ref()
            .unwrap_or(&self.richards);

        // Compute parameter gradients without materializing f64 matrices.
        // This significantly reduces peak memory in backward passes.
        let richards_grads = richards.grad_weights_matrix_f32(input, output_grads);

        // Compute input gradients without materializing f64 matrices.
        let mut grad_input = Array2::<f32>::zeros(input.raw_dim());
        richards.backward_matrix_f32_into(input, output_grads, &mut grad_input);

        // Extract gradients by parameter type (nu, k, beta, temperature, scale, gamma, bias)
        let mut grad_vecs = Vec::new();
        let mut pos = 0;

        // Scalar parameters
        if richards.nu_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.k_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.m_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.beta_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.temperature_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.output_gain_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.output_bias_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.scale_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if richards.shift_learnable {
            grad_vecs
                .push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }

        // Vector parameters (gamma, bias)
        // These are always last in the list of gradients from RichardsCurve
        let embedding_dim = input.ncols();

        if richards.gamma_learnable {
            // Gamma
            let gamma_len = embedding_dim;
            let gamma_grad: Vec<f32> = richards_grads[pos..pos + gamma_len]
                .iter()
                .map(|&x| x as f32)
                .collect();
            grad_vecs.push(Array2::from_shape_vec((1, gamma_len), gamma_grad).unwrap());
            pos += gamma_len;
        }

        if richards.bias_learnable {
            // Bias
            let bias_len = embedding_dim;
            let bias_grad: Vec<f32> = richards_grads[pos..pos + bias_len]
                .iter()
                .map(|&x| x as f32)
                .collect();
            grad_vecs.push(Array2::from_shape_vec((1, bias_len), bias_grad).unwrap());
        }

        (grad_input, grad_vecs)
    }
}

// ============================================================================
// GPU Component Implementation
// ============================================================================

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl crate::domain::compute::GpuComponent for RichardsNorm {
    fn set_gpu_device(
        &mut self,
        device: std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>,
    ) {
        self.gpu_device = Some(device);
    }

    fn enable_gpu_auto_detect(&mut self) -> crate::common::errors::Result<()> {
        let device = crate::domain::compute::GpuDevice::auto_detect()?;
        self.gpu_device = Some(std::sync::Arc::new(std::sync::Mutex::new(device)));
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

    fn gpu_device(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<crate::domain::compute::GpuDevice>>> {
        self.gpu_device.clone()
    }

    fn ensure_capacity(
        &mut self,
        batch_size: usize,
        embed_dim: usize,
        _seq_len: usize,
    ) -> crate::common::errors::Result<()> {
        if let Some(device_arc) = &self.gpu_device {
            let mut device =
                device_arc
                    .lock()
                    .map_err(|_| crate::common::errors::ModelError::Backend {
                        message: "Failed to lock GPU device for RichardsNorm capacity allocation"
                            .to_string(),
                    })?;
            // Pre-allocate buffers for normalization operations
            let size = batch_size * embed_dim;
            let _ = device.allocate_f32(size)?; // input buffer
            let _ = device.allocate_f32(size)?; // output buffer
            Ok(())
        } else {
            Err(crate::common::errors::ModelError::Backend {
                message:
                    "GPU device not attached to RichardsNorm. Call enable_gpu_auto_detect() first."
                        .to_string(),
            })
        }
    }
}

#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
impl RichardsNorm {
    /// GPU-accelerated forward normalization pass.
    ///
    /// Uploads input to GPU, applies Richards-based normalization with per-feature
    /// gamma/bias, and downloads the result.
    pub fn forward_gpu(
        &mut self,
        input: &ndarray::Array2<f32>,
    ) -> crate::common::errors::Result<ndarray::Array2<f32>> {
        let device_arc = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "RichardsNorm::forward_gpu",
        )?;
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for RichardsNorm forward".to_string(),
                })?;

        let (batch_size, embed_dim) = input.dim();
        let size = batch_size * embed_dim;

        // Upload input
        let mut gpu_input = device.allocate_f32(size)?;
        let input_slice =
            input
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "RichardsNorm input is not contiguous".to_string(),
                })?;
        device.upload(input_slice, &mut gpu_input)?;

        // Apply Richards curve on GPU using the real kernel
        let params = self.richards.to_gpu_params(1);
        let mut gpu_output = device.allocate_f32(size)?;
        device.richards_curve(&gpu_input, &mut gpu_output, &params, size)?;

        // Apply per-feature gamma/bias if present
        let has_gamma_bias = self.richards.gamma.is_some()
            && self.richards.bias.is_some()
            && self
                .richards
                .gamma
                .as_ref()
                .map_or(false, |g| g.len() == embed_dim);

        if has_gamma_bias {
            let gamma_arc = self.richards.gamma.as_ref().unwrap();
            let bias_arc = self.richards.bias.as_ref().unwrap();
            let gamma_slice = gamma_arc.as_slice().unwrap();
            let bias_slice = bias_arc.as_slice().unwrap();

            // Scale by gamma: element-wise multiply with broadcast
            let mut gpu_gamma = device.allocate_f32(embed_dim)?;
            device.upload(gamma_slice, &mut gpu_gamma)?;

            // Expand gamma to (batch_size, embed_dim) for element-wise multiply
            let mut gamma_expanded = vec![0.0f32; size];
            for row in gamma_expanded.chunks_exact_mut(embed_dim) {
                row.copy_from_slice(gamma_slice);
            }
            let mut gpu_gamma_expanded = device.allocate_f32(size)?;
            device.upload(&gamma_expanded, &mut gpu_gamma_expanded)?;

            let mut gpu_scaled = device.allocate_f32(size)?;
            device.mul(&gpu_output, &gpu_gamma_expanded, &mut gpu_scaled, size)?;

            // Add bias via broadcast
            let mut gpu_bias = device.allocate_f32(embed_dim)?;
            device.upload(bias_slice, &mut gpu_bias)?;
            device.broadcast_add_rows(&mut gpu_scaled, &gpu_bias, batch_size, embed_dim)?;

            // Download result
            let mut result = vec![0.0f32; size];
            device.download(&gpu_scaled, &mut result)?;

            // Cleanup
            device.deallocate(gpu_input);
            device.deallocate(gpu_output);
            device.deallocate(gpu_gamma);
            device.deallocate(gpu_gamma_expanded);
            device.deallocate(gpu_scaled);
            device.deallocate(gpu_bias);

            // Cache input for backward pass
            drop(device);
            self.cached_input = Some(input.clone());

            ndarray::Array2::from_shape_vec((batch_size, embed_dim), result).map_err(|e| {
                crate::common::errors::ModelError::Backend {
                    message: format!("Failed to reshape RichardsNorm GPU output: {}", e),
                }
            })
        } else {
            // No per-feature params, just download the Richards curve output directly
            let mut result = vec![0.0f32; size];
            device.download(&gpu_output, &mut result)?;

            device.deallocate(gpu_input);
            device.deallocate(gpu_output);

            // Cache input for backward pass
            drop(device);
            self.cached_input = Some(input.clone());

            ndarray::Array2::from_shape_vec((batch_size, embed_dim), result).map_err(|e| {
                crate::common::errors::ModelError::Backend {
                    message: format!("Failed to reshape RichardsNorm GPU output: {}", e),
                }
            })
        }
    }

    /// GPU-accelerated backward pass for RichardsNorm.
    ///
    /// Computes gradients on GPU using the cached forward state.
    pub fn compute_gradients_gpu(
        &self,
        output_grads: &ndarray::Array2<f32>,
    ) -> crate::common::errors::Result<(ndarray::Array2<f32>, Vec<ndarray::Array2<f32>>)> {
        let device_arc = crate::domain::compute::require_gpu_or_error(
            &self.gpu_device,
            "RichardsNorm::compute_gradients_gpu",
        )?;
        let mut device =
            device_arc
                .lock()
                .map_err(|_| crate::common::errors::ModelError::Backend {
                    message: "Failed to lock GPU device for RichardsNorm compute_gradients"
                        .to_string(),
                })?;

        let (batch_size, embed_dim) = output_grads.dim();
        let size = batch_size * embed_dim;
        if size == 0 {
            let (grad_input, param_grads) = self.compute_gradients_cpu_impl(output_grads);
            return Ok((grad_input, param_grads));
        }

        let grads_slice =
            output_grads
                .as_slice()
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "RichardsNorm output_grads is not contiguous".to_string(),
                })?;

        let mut gpu_grads = device.allocate_f32(size)?;
        device.upload(grads_slice, &mut gpu_grads)?;

        let mut gpu_input_grads = device.allocate_f32(size)?;
        let has_gamma = self.richards.gamma.is_some()
            && self
                .richards
                .gamma
                .as_ref()
                .is_some_and(|g| g.len() == embed_dim);

        if has_gamma {
            let gamma_slice = self
                .richards
                .gamma
                .as_ref()
                .and_then(|g| g.as_slice())
                .ok_or_else(|| crate::common::errors::ModelError::Backend {
                    message: "RichardsNorm gamma is not contiguous".to_string(),
                })?;

            let mut gamma_expanded = vec![0.0f32; size];
            for row in gamma_expanded.chunks_exact_mut(embed_dim) {
                row.copy_from_slice(gamma_slice);
            }

            let mut gpu_gamma_expanded = device.allocate_f32(size)?;
            device.upload(&gamma_expanded, &mut gpu_gamma_expanded)?;
            device.mul(&gpu_grads, &gpu_gamma_expanded, &mut gpu_input_grads, size)?;
            device.deallocate(gpu_gamma_expanded);
        } else {
            device.copy_within_device(&gpu_grads, &mut gpu_input_grads, size)?;
        }

        let mut input_grads_vec = vec![0.0f32; size];
        device.download(&gpu_input_grads, &mut input_grads_vec)?;
        device.deallocate(gpu_grads);
        device.deallocate(gpu_input_grads);
        drop(device);

        let input_grads = ndarray::Array2::from_shape_vec((batch_size, embed_dim), input_grads_vec)
            .map_err(|e| crate::common::errors::ModelError::Backend {
                message: format!("Failed to reshape RichardsNorm GPU input gradients: {}", e),
            })?;

        // Keep parameter-gradient layout and optimizer routing identical to CPU path.
        let (_cpu_input_grads, param_grads) = self.compute_gradients_cpu_impl(output_grads);
        Ok((input_grads, param_grads))
    }

    /// GPU-accelerated backward pass for RichardsNorm.
    ///
    /// Computes gradients on GPU using the cached forward state.
    pub fn backward_gpu(
        &mut self,
        output_grads: &ndarray::Array2<f32>,
        lr: f32,
    ) -> crate::common::errors::Result<ndarray::Array2<f32>> {
        let (input_grads, param_grads) = self.compute_gradients_gpu(output_grads)?;
        self.apply_gradients(&param_grads, lr)?;
        Ok(input_grads)
    }
}
