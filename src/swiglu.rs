use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::adam::Adam;
use crate::llm::Layer;

/// SwiGLU (Swish-Gated Linear Unit) activation function
///
/// SwiGLU is a gating mechanism that combines the Swish activation function
/// with Gated Linear Units (GLU). It was introduced in "GLU Variants Improve
/// Transformer" (Shazeer, 2020) and is used in modern LLMs like LLaMA, PaLM,
/// and Mistral.
///
/// # Mathematical Formulation
///
/// ```text
/// SwiGLU(x) = (Swish(xW₁) ⊙ xW₂)W₃
/// ```
///
/// Where:
/// - `Swish(x) = x * σ(x)` where `σ` is the sigmoid function
/// - `⊙` denotes element-wise multiplication (gating)
/// - `W₁, W₂, W₃` are weight matrices (no bias terms)
///
/// # Architecture
///
/// ```text
/// Input (seq_len, embedding_dim)
///   ├─> xW₁ -> Swish ──┐
///   │                   ├─> ⊙ (gate) -> hidden -> W₃ -> Output
///   └─> xW₂ ───────────┘
/// ```
///
/// # Benefits
///
/// - **Better gradient flow**: No dead neurons like ReLU
/// - **Improved capacity**: Gating mechanism allows selective information flow
/// - **Parameter efficiency**: No bias terms (modern LLM practice)
/// - **Empirically superior**: Outperforms ReLU, GELU, and other activations
///
/// # References
///
/// - Shazeer (2020), "GLU Variants Improve Transformer", arXiv:2002.05202
/// - Touvron et al. (2023), "LLaMA", arXiv:2302.13971
///
/// # Example
///
/// ```rust,ignore
/// use llm::swiglu::SwiGLU;
/// use llm::llm::Layer;
/// use ndarray::Array2;
///
/// let mut swiglu = SwiGLU::new(128, 512);
/// let input = Array2::ones((10, 128));
/// let output = swiglu.forward(&input);
/// assert_eq!(output.shape(), &[10, 128]);
/// ```
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwiGLU {
    /// First projection matrix: (embedding_dim, hidden_dim)
    w1: Array2<f32>,
    /// Second projection matrix for gating: (embedding_dim, hidden_dim)
    w2: Array2<f32>,
    /// Output projection matrix: (hidden_dim, embedding_dim)
    w3: Array2<f32>,

    // Cached values for backward pass
    cached_input: Option<Array2<f32>>,
    cached_x1: Option<Array2<f32>>,      // xW₁
    cached_x2: Option<Array2<f32>>,      // xW₂
    cached_swish: Option<Array2<f32>>,   // Swish(xW₁)
    cached_gated: Option<Array2<f32>>,   // Swish(xW₁) ⊙ xW₂

    // Adam optimizers for each weight matrix
    optimizer_w1: Adam,
    optimizer_w2: Adam,
    optimizer_w3: Adam,
}

impl SwiGLU {
    /// Create a new SwiGLU layer
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Input/output dimension
    /// * `hidden_dim` - Hidden dimension (typically 4× embedding_dim in Transformers)
    ///
    /// # Weight Initialization
    ///
    /// Uses Xavier/Glorot initialization: `std = sqrt(2 / fan_in)`
    ///
    /// # Returns
    ///
    /// A new SwiGLU layer with randomly initialized weights
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let swiglu = SwiGLU::new(128, 512);
    /// assert_eq!(swiglu.parameters(), 128 * 512 + 128 * 512 + 512 * 128);
    /// ```
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = rand::rng();

        // Xavier/Glorot initialization for all weight matrices
        let std_w1 = (2.0 / embedding_dim as f32).sqrt();
        let std_w2 = (2.0 / embedding_dim as f32).sqrt();
        let std_w3 = (2.0 / hidden_dim as f32).sqrt();

        let normal_w1 = Normal::new(0.0, std_w1).unwrap();
        let normal_w2 = Normal::new(0.0, std_w2).unwrap();
        let normal_w3 = Normal::new(0.0, std_w3).unwrap();

        SwiGLU {
            w1: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w1.sample(&mut rng)),
            w2: Array2::from_shape_fn((embedding_dim, hidden_dim), |_| normal_w2.sample(&mut rng)),
            w3: Array2::from_shape_fn((hidden_dim, embedding_dim), |_| normal_w3.sample(&mut rng)),
            cached_input: None,
            cached_x1: None,
            cached_x2: None,
            cached_swish: None,
            cached_gated: None,
            optimizer_w1: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w2: Adam::new((embedding_dim, hidden_dim)),
            optimizer_w3: Adam::new((hidden_dim, embedding_dim)),
        }
    }

    /// Swish activation function (also called SiLU)
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// Swish(x) = x * σ(x) = x * (1 / (1 + e^(-x)))
    /// ```
    ///
    /// # Derivative
    ///
    /// ```text
    /// d/dx Swish(x) = σ(x) + x * σ(x) * (1 - σ(x))
    ///               = σ(x) * (1 + x * (1 - σ(x)))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor
    ///
    /// # Returns
    ///
    /// Swish-activated tensor of same shape as input
    #[inline]
    fn swish(x: &Array2<f32>) -> Array2<f32> {
        let sigmoid = x.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        x * &sigmoid
    }

    /// Derivative of Swish activation function
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// d/dx Swish(x) = σ(x) * (1 + x * (1 - σ(x)))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor (pre-activation)
    ///
    /// # Returns
    ///
    /// Derivative tensor of same shape as input
    #[inline]
    fn swish_derivative(x: &Array2<f32>) -> Array2<f32> {
        let sigmoid = x.mapv(|val| 1.0 / (1.0 + (-val).exp()));
        &sigmoid * &(1.0 + x * &(1.0 - &sigmoid))
    }
}

impl Layer for SwiGLU {
    fn layer_type(&self) -> &str {
        "SwiGLU"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Compute xW₁ and xW₂
        let x1 = input.dot(&self.w1);
        let x2 = input.dot(&self.w2);

        // Apply Swish to x1
        let swish = Self::swish(&x1);

        // Gate: Swish(xW₁) ⊙ xW₂
        let gated = &swish * &x2;

        // Output projection: gated * W₃
        let output = gated.dot(&self.w3);

        // Cache for backward pass
        self.cached_input = Some(input.clone());
        self.cached_x1 = Some(x1);
        self.cached_x2 = Some(x2);
        self.cached_swish = Some(swish);
        self.cached_gated = Some(gated);

        // Add residual connection
        output + input
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // Retrieve cached values
        let input = self.cached_input.as_ref().expect("forward must be called before compute_gradients");
        let x1 = self.cached_x1.as_ref().expect("forward must be called before compute_gradients");
        let x2 = self.cached_x2.as_ref().expect("forward must be called before compute_gradients");
        let swish = self.cached_swish.as_ref().expect("forward must be called before compute_gradients");
        let gated = self.cached_gated.as_ref().expect("forward must be called before compute_gradients");

        // Gradient w.r.t. W₃: ∂L/∂W₃ = gated^T · ∂L/∂output
        let grad_w3 = gated.t().dot(output_grads);

        // Gradient w.r.t. gated: ∂L/∂gated = ∂L/∂output · W₃^T
        let grad_gated = output_grads.dot(&self.w3.t());

        // Gradient through gating: gated = swish ⊙ x2
        // ∂L/∂swish = ∂L/∂gated ⊙ x2
        // ∂L/∂x2 = ∂L/∂gated ⊙ swish
        let grad_swish = &grad_gated * x2;
        let grad_x2 = &grad_gated * swish;

        // Gradient through Swish activation: ∂L/∂x1 = ∂L/∂swish ⊙ swish'(x1)
        let swish_deriv = Self::swish_derivative(x1);
        let grad_x1 = &grad_swish * &swish_deriv;

        // Gradient w.r.t. W₁: ∂L/∂W₁ = input^T · ∂L/∂x1
        let grad_w1 = input.t().dot(&grad_x1);

        // Gradient w.r.t. W₂: ∂L/∂W₂ = input^T · ∂L/∂x2
        let grad_w2 = input.t().dot(&grad_x2);

        // Gradient w.r.t. input (through SwiGLU computation)
        let grad_input_swiglu = grad_x1.dot(&self.w1.t()) + grad_x2.dot(&self.w2.t());

        // Add gradient from residual connection
        // Forward: output = SwiGLU(input) + input
        // Backward: grad_input = grad_swiglu + grad_residual
        let grad_input = grad_input_swiglu + output_grads;

        (grad_input, vec![grad_w1, grad_w2, grad_w3])
    }

    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        self.optimizer_w1.step(&mut self.w1, &param_grads[0], lr);
        self.optimizer_w2.step(&mut self.w2, &param_grads[1], lr);
        self.optimizer_w3.step(&mut self.w3, &param_grads[2], lr);
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr);
        input_grads
    }

    fn parameters(&self) -> usize {
        // No bias terms: only W₁, W₂, W₃
        self.w1.len() + self.w2.len() + self.w3.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_creation() {
        let swiglu = SwiGLU::new(128, 512);
        assert_eq!(swiglu.layer_type(), "SwiGLU");
        // Parameters: 128*512 + 128*512 + 512*128 = 196,608
        assert_eq!(swiglu.parameters(), 128 * 512 * 3);
    }

    #[test]
    fn test_swiglu_forward() {
        let mut swiglu = SwiGLU::new(64, 256);
        let input = Array2::ones((10, 64));
        let output = swiglu.forward(&input);
        
        // Output shape should match input shape (with residual)
        assert_eq!(output.shape(), &[10, 64]);
        
        // Output should not be all zeros
        assert!(output.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_swiglu_gradient_shapes() {
        let mut swiglu = SwiGLU::new(32, 128);
        let input = Array2::ones((5, 32));
        
        // Forward pass
        let _output = swiglu.forward(&input);
        
        // Compute gradients
        let output_grads = Array2::ones((5, 32));
        let (input_grads, param_grads) = swiglu.compute_gradients(&input, &output_grads);
        
        // Check gradient shapes
        assert_eq!(input_grads.shape(), &[5, 32]);
        assert_eq!(param_grads.len(), 3); // W₁, W₂, W₃
        assert_eq!(param_grads[0].shape(), &[32, 128]); // grad_w1
        assert_eq!(param_grads[1].shape(), &[32, 128]); // grad_w2
        assert_eq!(param_grads[2].shape(), &[128, 32]); // grad_w3
    }

    #[test]
    fn test_swish_activation() {
        let x = Array2::from_shape_vec((2, 3), vec![
            -2.0, 0.0, 2.0,
            -1.0, 0.5, 1.0,
        ]).unwrap();
        
        let swish = SwiGLU::swish(&x);
        
        // Swish(0) should be close to 0
        assert!((swish[[0, 1]]).abs() < 0.01);
        
        // Swish should be smooth and non-zero for most inputs
        assert!(swish.iter().any(|&val| val.abs() > 0.1));
    }
}

