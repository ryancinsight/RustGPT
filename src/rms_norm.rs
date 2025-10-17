use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

/// Root Mean Square Layer Normalization (RMSNorm)
///
/// RMSNorm is a normalization technique that normalizes inputs using only
/// the root mean square (RMS), without centering by the mean. This approach
/// reduces computational cost by ~50% compared to LayerNorm (no mean computation)
/// while maintaining comparable or superior performance in modern LLMs.
///
/// # Mathematical Formulation
///
/// Given input vector **x** ∈ ℝ^d:
///
/// ```text
/// RMS(x) = √(1/d ∑ᵢ xᵢ² + ε)
/// y = (x / RMS(x)) ⊙ γ
/// ```
///
/// Where:
/// - d: feature dimension
/// - ε: small constant for numerical stability (default: 1e-6)
/// - γ: learnable scale parameter (initialized to 1)
/// - ⊙: element-wise multiplication
///
/// # Key Differences from LayerNorm
///
/// 1. **No mean centering**: RMSNorm doesn't subtract the mean
/// 2. **No bias parameter**: Only scale (γ), no shift (β)
/// 3. **Simpler computation**: ~10-15% faster than LayerNorm
/// 4. **Better stability**: Used in LLaMA, GPT-NeoX, PaLM, Mistral
///
/// # Gradient Derivation
///
/// Let:
/// - rms = √(mean(x²) + ε)
/// - norm = x / rms
///
/// Then:
/// ```text
/// ∂L/∂γ = ∑ₜ (∂L/∂yₜ ⊙ normₜ)  [sum over tokens]
///
/// ∂L/∂x = (∂L/∂y ⊙ γ / rms) - (norm ⊙ mean(∂L/∂y ⊙ γ ⊙ norm))
/// ```
///
/// # References
///
/// - Zhang & Sennrich (2019): "Root Mean Square Layer Normalization"
/// - Touvron et al. (2023): "LLaMA: Open and Efficient Foundation Language Models"
/// - HRM Paper (2025): Uses RMSNorm for stability in hierarchical reasoning
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use llm::rms_norm::RMSNorm;
/// use llm::llm::Layer;
///
/// let mut rms_norm = RMSNorm::new(128);
/// let input = Array2::ones((10, 128)); // (seq_len, embedding_dim)
/// let output = rms_norm.forward(&input);
/// assert_eq!(output.shape(), &[10, 128]);
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RMSNorm {
    /// Small constant for numerical stability (default: 1e-6)
    epsilon: f32,

    /// Learnable scale parameter γ (shape: [1, embedding_dim])
    gamma: Array2<f32>,

    /// Cached input for backward pass
    cached_input: Option<Array2<f32>>,

    /// Cached RMS values for backward pass
    cached_rms: Option<Array2<f32>>,

    /// Cached normalized values for backward pass
    cached_normalized: Option<Array2<f32>>,

    /// Adam optimizer for gamma parameter
    optimizer_gamma: Adam,
}

impl RMSNorm {
    /// Create a new RMSNorm layer
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of the input features
    ///
    /// # Returns
    ///
    /// A new `RMSNorm` instance with gamma initialized to 1
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let rms_norm = RMSNorm::new(128);
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        Self::with_epsilon(embedding_dim, 1e-6)
    }

    /// Create a new RMSNorm layer with custom epsilon
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of the input features
    /// * `epsilon` - Small constant for numerical stability
    ///
    /// # Returns
    ///
    /// A new `RMSNorm` instance
    pub fn with_epsilon(embedding_dim: usize, epsilon: f32) -> Self {
        RMSNorm {
            epsilon,
            gamma: Array2::ones((1, embedding_dim)), // Initialize gamma to 1
            cached_input: None,
            cached_rms: None,
            cached_normalized: None,
            optimizer_gamma: Adam::new((1, embedding_dim)),
        }
    }

    /// Get a reference to the gamma parameter (for testing/inspection)
    ///
    /// # Returns
    ///
    /// Reference to the gamma (scale) parameter
    pub fn gamma(&self) -> &Array2<f32> {
        &self.gamma
    }

    /// Normalize input using RMS normalization
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape (seq_len, embedding_dim)
    ///
    /// # Returns
    ///
    /// Normalized tensor of same shape as input
    ///
    /// # Mathematical Operation
    ///
    /// ```text
    /// For each token t:
    ///   rms_t = √(mean(x_t²) + ε)
    ///   y_t = (x_t / rms_t) ⊙ γ
    /// ```
    pub fn normalize(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Compute RMS per token: sqrt(mean(x²) + ε)
        let squared = input.mapv(|x| x * x);
        let mean_squared = squared.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
        let rms = mean_squared.mapv(|x| (x + self.epsilon).sqrt());

        // Normalize: x / rms
        let normalized = input / &rms;

        // Cache values for backward pass
        self.cached_input = Some(input.clone());
        self.cached_rms = Some(rms);
        self.cached_normalized = Some(normalized.clone());

        // Scale by gamma: y = normalized * γ
        &self.gamma * &normalized
    }
}

impl Layer for RMSNorm {
    fn layer_type(&self) -> &str {
        "RMSNorm"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.normalize(input)
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
        let rms = self
            .cached_rms
            .as_ref()
            .expect("forward must be called before compute_gradients");
        let normalized = self
            .cached_normalized
            .as_ref()
            .expect("forward must be called before compute_gradients");

        let n_features = input.shape()[1] as f32;

        // Gradient w.r.t. gamma: ∂L/∂γ = sum(∂L/∂y ⊙ norm)
        let grad_gamma = (normalized * output_grads)
            .sum_axis(Axis(0))
            .insert_axis(Axis(0));

        // Gradient w.r.t. normalized values: ∂L/∂norm = ∂L/∂y ⊙ γ
        let grad_normalized = &self.gamma * output_grads;

        // Gradient w.r.t. input using chain rule:
        //
        // Forward: y = (x / rms(x)) * γ where rms(x) = sqrt(mean(x²) + ε)
        //
        // Let norm = x / rms, then y = norm * γ
        //
        // ∂L/∂x = ∂L/∂norm * ∂norm/∂x
        //
        // For ∂norm/∂x, we have:
        // ∂(xᵢ/rms)/∂xⱼ = δᵢⱼ/rms - xᵢ * ∂rms/∂xⱼ / rms²
        //
        // where ∂rms/∂xⱼ = ∂/∂xⱼ sqrt(mean(x²) + ε)
        //                = (1/2) * (mean(x²) + ε)^(-1/2) * (2xⱼ/d)
        //                = xⱼ / (d * rms)
        //
        // Therefore:
        // ∂(xᵢ/rms)/∂xⱼ = δᵢⱼ/rms - xᵢ * xⱼ / (d * rms³)
        //
        // Applying chain rule:
        // ∂L/∂xⱼ = sum_i [∂L/∂normᵢ * ∂normᵢ/∂xⱼ]
        //        = sum_i [∂L/∂normᵢ * (δᵢⱼ/rms - xᵢ * xⱼ / (d * rms³))]
        //        = ∂L/∂normⱼ/rms - xⱼ/(d * rms³) * sum_i [∂L/∂normᵢ * xᵢ]
        //        = ∂L/∂normⱼ/rms - normⱼ/d * sum_i [∂L/∂normᵢ * normᵢ]

        let grad_input = {
            // Term 1: ∂L/∂norm / rms
            let term1 = &grad_normalized / rms;

            // Term 2: norm * (1/d) * sum(∂L/∂norm * norm)
            let grad_norm_product = &grad_normalized * normalized;
            let sum_grad_norm_product = grad_norm_product.sum_axis(Axis(1)).insert_axis(Axis(1));
            let term2 = normalized * &sum_grad_norm_product / n_features;

            term1 - term2
        };

        (grad_input, vec![grad_gamma])
    }

    fn apply_gradients(
        &mut self,
        param_grads: &[Array2<f32>],
        lr: f32,
    ) -> crate::errors::Result<()> {
        if param_grads.is_empty() {
            return Err(crate::errors::ModelError::GradientError {
                message: "RMSNorm expected 1 parameter gradient (gamma), got 0".to_string(),
            });
        }

        self.optimizer_gamma
            .step(&mut self.gamma, &param_grads[0], lr);
        Ok(())
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        // Unwrap is safe: backward is only called from training loop which validates inputs
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.gamma.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_rms_norm_creation() {
        let rms_norm = RMSNorm::new(128);
        assert_eq!(rms_norm.gamma.shape(), &[1, 128]);
        assert_eq!(rms_norm.epsilon, 1e-6);
        assert_eq!(rms_norm.parameters(), 128);
    }

    #[test]
    fn test_rms_norm_forward() {
        let mut rms_norm = RMSNorm::new(4);
        let input =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0])
                .unwrap();

        let output = rms_norm.forward(&input);
        assert_eq!(output.shape(), &[2, 4]);

        // Check that output is normalized (RMS ≈ 1 after scaling by gamma=1)
        let output_squared = output.mapv(|x| x * x);
        let rms_output = output_squared
            .mean_axis(Axis(1))
            .unwrap()
            .mapv(|x| x.sqrt());

        // RMS should be close to 1 for each token
        for &rms_val in rms_output.iter() {
            assert!(
                (rms_val - 1.0).abs() < 0.1,
                "RMS should be close to 1, got {}",
                rms_val
            );
        }
    }

    #[test]
    fn test_rms_norm_gradient_shape() {
        let mut rms_norm = RMSNorm::new(4);
        let input = Array2::ones((3, 4));

        // Forward pass
        let _output = rms_norm.forward(&input);

        // Backward pass
        let output_grads = Array2::ones((3, 4));
        let (input_grads, param_grads) = rms_norm.compute_gradients(&input, &output_grads);

        assert_eq!(input_grads.shape(), &[3, 4]);
        assert_eq!(param_grads.len(), 1); // Only gamma
        assert_eq!(param_grads[0].shape(), &[1, 4]);
    }

    #[test]
    fn test_rms_norm_vs_layer_norm_no_mean() {
        // RMSNorm should NOT center by mean (key difference from LayerNorm)
        let mut rms_norm = RMSNorm::new(4);

        // Input with non-zero mean
        let input = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = rms_norm.forward(&input);

        // Output should NOT have zero mean (unlike LayerNorm)
        let output_mean = output.mean_axis(Axis(1)).unwrap()[[0]];
        assert!(
            output_mean.abs() > 0.01,
            "RMSNorm should not center by mean"
        );
    }
}
