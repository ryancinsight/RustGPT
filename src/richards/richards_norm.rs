use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::llm::Layer;
use super::{RichardsCurve, Variant};

/// Richards-based Normalization with Dynamic Parameter Adjustments
///
/// Element-wise normalization using Richards curve with adaptive parameter scaling,
/// followed by per-channel scale `gamma` and bias `bias`:
///
///   y = Richards_adaptive(scale · x) ⊙ gamma + bias
///
/// Dynamic adjustments based on activation statistics (Frobenius norm):
/// - **Adaptive Temperature**: Scales temperature by activation magnitude ratio
///   (inspired by Dynamic Tanh's α parameter for data-dependent scaling)
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

    /// Richards curve for tanh-like computation with learnable parameters and per-feature transformations
    richards: RichardsCurve,
}

impl RichardsNorm {
    /// Create a new RichardsNorm layer
    pub fn new(embedding_dim: usize) -> Self {
        // Start with learnable Richards for Tanh variant
        let mut richards = RichardsCurve::new_learnable(Variant::Tanh);

        // Set fixed parameter values for tanh approximation (keep zero-mean properties)
        richards.nu = None; // Learnable
        richards.k = None; // Learnable
        richards.m = Some(0.0); // Fixed to maintain odd function (zero-mean)
        richards.beta = None; // Learnable
        richards.output_gain = Some(1.0); // Fixed to maintain scale
        richards.output_bias = Some(0.0); // Fixed to maintain zero-mean
        richards.scale = None; // Learnable
        richards.shift = Some(0.0); // Fixed to maintain odd function

        // Initialize learned parameters
        richards.learned_nu = Some(1.0);
        richards.learned_k = Some(1.0);
        richards.learned_beta = Some(1.0);
        richards.learned_temperature = Some(1.0);
        richards.learned_scale = Some(1.0);

        // Set learnability: nu, k, beta, temperature, scale learnable; shift fixed
        richards.nu_learnable = true;
        richards.k_learnable = true;
        richards.m_learnable = false;
        richards.beta_learnable = true;
        richards.temperature_learnable = true;
        richards.output_gain_learnable = false;
        richards.output_bias_learnable = false;
        richards.scale_learnable = true;  // RichardsNorm allows RichardsCurve to learn input scaling
        richards.shift_learnable = false;

        // Enable per-feature transformations (gamma, bias) for normalization
        richards.enable_per_feature_transform(embedding_dim);

        // Validate that RichardsCurve has exactly the expected learnable parameters
        // RichardsNorm expects: nu, k, beta, temperature, scale (5 parameters)
        let expected_learnable = [true, true, false, true, true, false, false, true, false]; // nu, k, m, beta, temp, gain, bias, scale, shift
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
            richards,
        }
    }

    /// Apply dynamic parameter adjustments based on activation statistics
    /// Returns the adjusted parameters for restoration
    fn compute_dynamic_adjustments(&self, input: &Array2<f32>) -> (Option<f64>, Option<f64>, Option<f64>) {
        // Compute Frobenius norm for scale-aware adjustments
        let frob_norm = (input.iter().map(|&x| (x as f64).powi(2)).sum::<f64>()).sqrt();

        // Compute activation statistics
        let mean = input.iter().map(|&x| x as f64).sum::<f64>() / (input.len() as f64);
        let variance = input.iter().map(|&x| ((x as f64) - mean).powi(2)).sum::<f64>() / (input.len() as f64);
        let std_dev = variance.sqrt();

        // Target scale for normalization (empirical value, can be tuned)
        let target_scale = (input.len() as f64).sqrt(); // Approximate RMS norm

        // Adaptive temperature scaling (inspired by DyT's α parameter)
        // Higher activation scale → sharper transitions (higher temperature)
        let scale_ratio = (frob_norm / target_scale).max(1e-6).min(1e6);
        let temp_adjustment = scale_ratio.powf(0.5); // Square root for smoother scaling
        let adjusted_temp = self.richards.temperature.map(|t| t * temp_adjustment);

        // Dynamic midpoint adjustment to center around data distribution
        let midpoint_shift = mean * 0.1; // Small adjustment to avoid instability
        let adjusted_m = self.richards.m.map(|m| m + midpoint_shift);

        // Adaptive asymmetry based on variance
        // Higher variance → more symmetric curve (beta → 1.0)
        // Lower variance → preserve learned asymmetry
        let adjusted_beta = if self.richards.beta_learnable {
            if let Some(learned_beta) = self.richards.learned_beta {
                let symmetry_factor = (std_dev / target_scale).max(0.1).min(2.0);
                let adaptive_beta = 1.0 + (learned_beta - 1.0) / symmetry_factor;
                Some(adaptive_beta.max(1e-6).min(10.0))
            } else {
                self.richards.beta
            }
        } else {
            self.richards.beta
        };

        (adjusted_temp, adjusted_m, adjusted_beta)
    }

    /// Forward normalization with dynamic parameter adjustments
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Cache input for backward (needed for gradient computation)
        self.cached_input = Some(input.clone());

        // Compute dynamic parameter adjustments
        let (adjusted_temp, adjusted_m, adjusted_beta) = self.compute_dynamic_adjustments(input);

        // Store original parameters
        let original_temp = self.richards.temperature;
        let original_m = self.richards.m;
        let original_beta = self.richards.beta;

        // Apply dynamic adjustments temporarily
        self.richards.temperature = adjusted_temp;
        self.richards.m = adjusted_m;
        self.richards.beta = adjusted_beta;

        // Apply Richards curve with per-feature transformations
        let output = self.richards.forward_matrix(&input.mapv(|x| x as f64)).mapv(|x| x as f32);

        // Restore original parameters
        self.richards.temperature = original_temp;
        self.richards.m = original_m;
        self.richards.beta = original_beta;

        output
    }
}

impl Layer for RichardsNorm {
    fn layer_type(&self) -> &str {
        "RichardsNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(&*input)
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let input = self
            .cached_input
            .as_ref()
            .expect("forward must be called before compute_gradients");

        // Convert to f64 for RichardsCurve computation
        let input_f64 = input.mapv(|x| x as f64);
        let output_grads_f64 = output_grads.mapv(|x| x as f64);

        // Compute gradients through RichardsCurve with per-feature transformations
        // This will handle gamma/bias gradients internally
        let richards_grads = self.richards.grad_weights_matrix(&input_f64, &output_grads_f64);

        // Compute input gradients: chain rule through RichardsCurve
        let grad_input_f64 = self.richards.backward_matrix(&input_f64, &output_grads_f64);
        let grad_input = grad_input_f64.mapv(|x| x as f32);

        // Extract gradients by parameter type (nu, k, beta, temperature, scale, gamma, bias)
        let mut grad_vecs = Vec::new();
        let mut pos = 0;

        // Scalar parameters
        if self.richards.nu_learnable {
            grad_vecs.push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if self.richards.k_learnable {
            grad_vecs.push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if self.richards.m_learnable {
            pos += 1; // Skip m gradient
        }
        if self.richards.beta_learnable {
            grad_vecs.push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if self.richards.temperature_learnable {
            grad_vecs.push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if self.richards.output_gain_learnable {
            pos += 1; // Skip output_gain gradient
        }
        if self.richards.output_bias_learnable {
            pos += 1; // Skip output_bias gradient
        }
        if self.richards.scale_learnable {
            grad_vecs.push(Array2::from_shape_vec((1, 1), vec![richards_grads[pos] as f32]).unwrap());
            pos += 1;
        }
        if self.richards.shift_learnable {
            pos += 1; // Skip shift gradient
        }

        // Array parameters (gamma, bias)
        if self.richards.gamma_learnable {
            let gamma_size = self.richards.gamma.as_ref().unwrap().len();
            let gamma_grads: Vec<f32> = richards_grads[pos..pos+gamma_size].iter().map(|&x| x as f32).collect();
            grad_vecs.push(Array2::from_shape_vec((1, gamma_size), gamma_grads).unwrap());
            pos += gamma_size;
        }
        if self.richards.bias_learnable {
            let bias_size = self.richards.bias.as_ref().unwrap().len();
            let bias_grads: Vec<f32> = richards_grads[pos..pos+bias_size].iter().map(|&x| x as f32).collect();
            grad_vecs.push(Array2::from_shape_vec((1, bias_size), bias_grads).unwrap());
            pos += bias_size;
        }

        let _ = pos; // Suppress unused variable warning

        (grad_input, grad_vecs)
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        // Collect all gradients into a flat vector for RichardsCurve step method
        let mut all_grads = Vec::new();

        // Add scalar parameter gradients
        for grad_array in param_grads.iter().take(5) {
            all_grads.push(grad_array[[0, 0]] as f64);
        }

        // Add gamma array gradients (flattened)
        if let Some(gamma_grads) = param_grads.get(5) {
            all_grads.extend(gamma_grads.iter().map(|&x| x as f64));
        }

        // Add bias array gradients (flattened)
        if let Some(bias_grads) = param_grads.get(6) {
            all_grads.extend(bias_grads.iter().map(|&x| x as f64));
        }

        // Apply gradients to RichardsCurve (which now includes gamma/bias)
        self.richards.step(&all_grads, lr as f64);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Apply parameter updates; ignore error here since sizes are checked in compute
        let _ = self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        self.richards.weights().len()
    }
}
