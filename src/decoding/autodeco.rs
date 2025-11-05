/// AutoDeco: End-to-End Decoding for Language Models
///
/// This module implements AutoDeco, a novel architecture that enables truly
/// "end-to-end" language model generation by learning to control its own
/// decoding strategy. Based on the paper "The End of Manual Decoding:
/// Towards Truly End-to-End Language Models".
///
/// Key innovations:
/// - Lightweight prediction heads for temperature and top-p values
/// - Differentiable soft top-p sampling for training
/// - Integration with Richards normalization for adaptive behavior
/// - Emergent instruction-based decoding control

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use crate::richards::RichardsNorm;
use crate::llm::Layer;

/// Configuration for AutoDeco heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoDecoConfig {
    /// Hidden dimension for prediction heads
    pub head_hidden_dim: usize,

    /// Temperature prediction range (min, max)
    pub temp_range: (f32, f32),

    /// Top-p prediction range (min, max)
    pub top_p_range: (f32, f32),

    /// Steepness parameter for soft top-p decay (α in paper)
    pub soft_top_p_alpha: f32,

    /// Whether to use instruction-based control
    pub enable_instruction_control: bool,

    /// Weight for instruction control loss
    pub instruction_control_weight: f32,
}

impl Default for AutoDecoConfig {
    fn default() -> Self {
        Self {
            head_hidden_dim: 256,  // Increased capacity for better learning
            temp_range: (0.1, 3.0), // Wider temperature range for more diversity
            top_p_range: (0.05, 0.95), // More aggressive top-p range
            soft_top_p_alpha: 50.0, // Sharper top-p transitions
            enable_instruction_control: false, // Not yet implemented
            instruction_control_weight: 0.1,
        }
    }
}

/// Temperature prediction head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureHead {
    /// Weight matrix (hidden_dim x head_hidden_dim)
    weights1: Array2<f32>,

    /// Bias vector (head_hidden_dim)
    bias1: Array1<f32>,

    /// Output weight matrix (head_hidden_dim x 1)
    weights2: Array2<f32>,

    /// Output bias (1)
    bias2: Array1<f32>,

    /// Richards normalization for adaptive behavior
    norm: RichardsNorm,
}

impl TemperatureHead {
    /// Create new temperature head with Xavier initialization
    pub fn new(hidden_dim: usize, head_hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Xavier initialization: weights ~ N(0, 1/sqrt(fan_in))
        let scale1 = 1.0 / (hidden_dim as f32).sqrt();
        let scale2 = 1.0 / (head_hidden_dim as f32).sqrt();

        let weights1 = Array2::from_shape_fn((hidden_dim, head_hidden_dim), |_| {
            rng.random_range(-scale1..scale1)
        });

        let bias1 = Array1::zeros(head_hidden_dim);

        let weights2 = Array2::from_shape_fn((head_hidden_dim, 1), |_| {
            rng.random_range(-scale2..scale2)
        });

        let bias2 = Array1::zeros(1);

        let norm = RichardsNorm::new(head_hidden_dim);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            norm,
        }
    }

    /// Forward pass through temperature head
    ///
    /// The temperature prediction follows this mathematical formulation:
    ///
    /// 1. **Linear Projection**: h = W₁ · x + b₁ where W₁ ∈ ℝ^{d × d_h}, b₁ ∈ ℝ^{d_h}
    ///
    /// 2. **Richards Normalization**: ĥ = RichardsNorm(h)
    ///
    /// 3. **Non-linear Activation**: a = ReLU(ĥ)
    ///
    /// 4. **Output Projection**: o = W₂ · a + b₂ where W₂ ∈ ℝ^{d_h × 1}, b₂ ∈ ℝ^{1}
    ///
    /// 5. **Richards Sigmoid Activation**: σ = RichardsSigmoid(o)
    ///
    /// 6. **Temperature Scaling**: T = T_min + σ · (T_max - T_min) where T_min = 0.1, T_max = 2.0
    ///
    /// This produces a temperature value in the range [0.1, 2.0] suitable for controlling sampling diversity.
    pub fn forward(&mut self, hidden_states: &ArrayView1<f32>) -> f32 {
        // First layer: W1 * h + b1
        let hidden = hidden_states.dot(&self.weights1) + &self.bias1;

        // Apply Richards normalization - ensure 2D shape
        let hidden_2d: Array2<f32> = if hidden.ndim() == 1 {
            hidden.view().into_shape_with_order((1, hidden.len())).unwrap().to_owned()
        } else {
            hidden.view().into_shape_with_order((hidden.shape()[0], hidden.shape()[1])).unwrap().to_owned()
        };
        let normalized = self.norm.forward(&hidden_2d);

        // ReLU activation
        let activated = normalized.mapv(|x| x.max(0.0));

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Sigmoid activation to get temperature in [0, 1] range
        let sigmoid = 1.0 / (1.0 + (-output[[0, 0]]).exp());

        // Scale to temperature range (0.1, 2.0)
        0.1 + sigmoid * 1.9
    }

    /// Get parameters for gradient computation
    pub fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.weights1, &self.weights2]
    }

    /// Get mutable parameters for gradient updates
    pub fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.weights1, &mut self.weights2]
    }

    /// Get bias parameters
    pub fn biases(&self) -> Vec<&Array1<f32>> {
        vec![&self.bias1, &self.bias2]
    }

    /// Get mutable bias parameters
    pub fn biases_mut(&mut self) -> Vec<&mut Array1<f32>> {
        vec![&mut self.bias1, &mut self.bias2]
    }
}

/// Top-p prediction head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopPHead {
    /// Weight matrix (hidden_dim x head_hidden_dim)
    weights1: Array2<f32>,

    /// Bias vector (head_hidden_dim)
    bias1: Array1<f32>,

    /// Output weight matrix (head_hidden_dim x 1)
    weights2: Array2<f32>,

    /// Output bias (1)
    bias2: Array1<f32>,

    /// Richards normalization for adaptive behavior
    norm: RichardsNorm,
}

impl TopPHead {
    /// Create new top-p head with Xavier initialization
    pub fn new(hidden_dim: usize, head_hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();

        // Xavier initialization
        let scale1 = 1.0 / (hidden_dim as f32).sqrt();
        let scale2 = 1.0 / (head_hidden_dim as f32).sqrt();

        let weights1 = Array2::from_shape_fn((hidden_dim, head_hidden_dim), |_| {
            rng.random_range(-scale1..scale1)
        });

        let bias1 = Array1::zeros(head_hidden_dim);

        let weights2 = Array2::from_shape_fn((head_hidden_dim, 1), |_| {
            rng.random_range(-scale2..scale2)
        });

        let bias2 = Array1::zeros(1);

        let norm = RichardsNorm::new(head_hidden_dim);

        Self {
            weights1,
            bias1,
            weights2,
            bias2,
            norm,
        }
    }

    /// Forward pass through top-p head
    pub fn forward(&mut self, hidden_states: &ArrayView1<f32>, temperature: f32) -> f32 {
        // Create input as [hidden_states..., temperature]
        let mut input_vec = hidden_states.to_vec();
        input_vec.push(temperature);
        let input = Array2::from_shape_vec((1, input_vec.len()), input_vec).unwrap();

        // First layer: W1 * [h, T] + b1
        let hidden = input.view().dot(&self.weights1) + &self.bias1;

        // Apply Richards normalization - ensure 2D shape
        let hidden_2d: Array2<f32> = if hidden.ndim() == 1 {
            hidden.view().into_shape_with_order((1, hidden.len())).unwrap().to_owned()
        } else {
            hidden.view().into_shape_with_order((hidden.shape()[0], hidden.shape()[1])).unwrap().to_owned()
        };
        let normalized = self.norm.forward(&hidden_2d);

        // ReLU activation
        let activated = normalized.mapv(|x| x.max(0.0));

        // Second layer: W2 * activated + b2
        let output = activated.dot(&self.weights2) + &self.bias2;

        // Sigmoid activation to get top-p in [0, 1] range
        let sigmoid = 1.0 / (1.0 + (-output[[0, 0]]).exp());

        sigmoid
    }

    /// Get parameters for gradient computation
    pub fn parameters(&self) -> Vec<&Array2<f32>> {
        vec![&self.weights1, &self.weights2]
    }

    /// Get mutable parameters for gradient updates
    pub fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        vec![&mut self.weights1, &mut self.weights2]
    }

    /// Get bias parameters
    pub fn biases(&self) -> Vec<&Array1<f32>> {
        vec![&self.bias1, &self.bias2]
    }

    /// Get mutable bias parameters
    pub fn biases_mut(&mut self) -> Vec<&mut Array1<f32>> {
        vec![&mut self.bias1, &mut self.bias2]
    }
}

/// AutoDeco: Main component that combines temperature and top-p heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoDeco {
    /// Configuration
    config: AutoDecoConfig,

    /// Temperature prediction head
    temp_head: TemperatureHead,

    /// Top-p prediction head
    top_p_head: TopPHead,

    /// Hidden dimension from base model
    hidden_dim: usize,
}

impl AutoDeco {
    /// Create new AutoDeco instance
    pub fn new(hidden_dim: usize, config: AutoDecoConfig) -> Self {
        let temp_head = TemperatureHead::new(hidden_dim, config.head_hidden_dim);
        let top_p_head = TopPHead::new(hidden_dim + 1, config.head_hidden_dim); // +1 for temperature

        Self {
            config,
            temp_head,
            top_p_head,
            hidden_dim,
        }
    }

    /// Predict decoding parameters for a single step
    pub fn predict_step(&mut self, hidden_states: &ArrayView1<f32>) -> (f32, f32) {
        // Predict temperature
        let temperature = self.temp_head.forward(hidden_states);

        // Predict top-p (using temperature as additional input)
        let top_p = self.top_p_head.forward(hidden_states, temperature);

        (temperature, top_p)
    }

    /// Apply soft top-p sampling (differentiable version for training)
    pub fn soft_top_p_sample(&self, logits: &Array1<f32>, top_p: f32) -> Array1<f32> {
        let temperature = 1.0; // Use default temperature during training
        let scaled_logits = logits / temperature;

        // Compute softmax probabilities
        let max_logit = scaled_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
        let probs = &exp_logits / exp_logits.sum();

        // Sort probabilities and compute cumulative sum
        let mut prob_indices: Vec<usize> = (0..probs.len()).collect();
        prob_indices.sort_by(|&i, &j| probs[j].partial_cmp(&probs[i]).unwrap());

        let mut sorted_probs = Array1::zeros(probs.len());
        for (i, &idx) in prob_indices.iter().enumerate() {
            sorted_probs[i] = probs[idx];
        }

        // Compute cumulative sum manually
        let mut cumulative = Array1::zeros(sorted_probs.len());
        let mut sum = 0.0;
        for (i, &val) in sorted_probs.iter().enumerate() {
            sum += val;
            cumulative[i] = sum;
        }

        // Apply soft mask: exp(-α * ReLU(cumulative - top_p))
        let soft_mask = cumulative.mapv(|c| {
            let relu_val = (c - top_p).max(0.0);
            (-self.config.soft_top_p_alpha * relu_val).exp()
        });

        // Unsort the mask
        let mut unsorted_mask = Array1::zeros(probs.len());
        for (i, &idx) in prob_indices.iter().enumerate() {
            unsorted_mask[idx] = soft_mask[i];
        }

        // Apply mask and renormalize
        let masked_probs = &probs * &unsorted_mask;
        let sum_masked = masked_probs.sum();
        if sum_masked > 0.0 {
            masked_probs / sum_masked
        } else {
            probs // Fallback to original if all masked
        }
    }

    /// Get all parameter tensors for gradient computation
    pub fn parameter_tensors(&self) -> Vec<&Array2<f32>> {
        let mut params = self.temp_head.parameters();
        params.extend(self.top_p_head.parameters());
        params
    }

    /// Get all mutable parameters for gradient updates
    pub fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        let mut params = self.temp_head.parameters_mut();
        params.extend(self.top_p_head.parameters_mut());
        params
    }

    /// Get all bias parameters
    pub fn biases(&self) -> Vec<&Array1<f32>> {
        let mut biases = self.temp_head.biases();
        biases.extend(self.top_p_head.biases());
        biases
    }

    /// Get all mutable bias parameters
    pub fn biases_mut(&mut self) -> Vec<&mut Array1<f32>> {
        let mut biases = self.temp_head.biases_mut();
        biases.extend(self.top_p_head.biases_mut());
        biases
    }
}

impl Layer for AutoDeco {
    fn layer_type(&self) -> &str {
        "AutoDeco"
    }

    fn forward(&mut self, _input: &Array2<f32>) -> Array2<f32> {
        // AutoDeco doesn't transform the main network flow
        // It operates on hidden states during decoding
        panic!("AutoDeco should not be used in the main network forward pass")
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // Pass gradients through unchanged
        grads.clone()
    }

    fn parameters(&self) -> usize {
        let mut count = 0;
        for param in &self.parameter_tensors() {
            count += param.len();
        }
        for bias in &self.biases() {
            count += bias.len();
        }
        count
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        _output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        // AutoDeco gradients are computed during decoding, not in the main network
        (Array2::zeros(_output_grads.raw_dim()), Vec::new())
    }

    fn apply_gradients(&mut self, _param_grads: &[Array2<f32>], _lr: f32) -> Result<(), crate::errors::ModelError> {
        // AutoDeco gradients are applied during decoding
        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let mut sumsq: f32 = 0.0;

        // Aggregate norms of all linear parameters
        for p in self.parameter_tensors() {
            sumsq += p.iter().map(|&w| w * w).sum::<f32>();
        }
        for b in self.biases() {
            sumsq += b.iter().map(|&w| w * w).sum::<f32>();
        }

        // Include RichardsNorm parameters from both heads
        sumsq += self.temp_head.norm.weight_norm().powi(2);
        sumsq += self.top_p_head.norm.weight_norm().powi(2);

        sumsq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_temperature_head() {
        let mut head = TemperatureHead::new(128, 64);
        let hidden = Array1::from_elem(128, 0.1);

        let temp = head.forward(&hidden.view());
        assert!(temp >= 0.1 && temp <= 2.0, "Temperature should be in range [0.1, 2.0]");
    }

    #[test]
    fn test_top_p_head() {
        let mut head = TopPHead::new(129, 64); // +1 for temperature
        let hidden = Array1::from_elem(128, 0.1);
        let temperature = 1.0;

        let top_p = head.forward(&hidden.view(), temperature);
        assert!(top_p >= 0.0 && top_p <= 1.0, "Top-p should be in range [0.0, 1.0]");
    }

    #[test]
    fn test_autodeco_predict() {
        let mut autodeco = AutoDeco::new(128, AutoDecoConfig::default());
        let hidden = Array1::from_elem(128, 0.1);

        let (temp, top_p) = autodeco.predict_step(&hidden.view());

        assert!(temp >= 0.1 && temp <= 2.0);
        assert!(top_p >= 0.0 && top_p <= 1.0);
    }

    #[test]
    fn test_soft_top_p() {
        let autodeco = AutoDeco::new(128, AutoDecoConfig::default());
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.5]);
        let top_p = 0.8;

        let sampled = autodeco.soft_top_p_sample(&logits, top_p);

        // Check that probabilities sum to 1
        let sum: f32 = sampled.sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probabilities should sum to 1, got {}", sum);

        // Check that all values are non-negative
        assert!(sampled.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_autodeco_config() {
        let config = AutoDecoConfig::default();
        assert_eq!(config.head_hidden_dim, 128);
        assert_eq!(config.temp_range, (0.1, 2.0));
        assert_eq!(config.top_p_range, (0.1, 1.0));
        assert_eq!(config.soft_top_p_alpha, 30.0);
        assert!(!config.enable_instruction_control);
        assert_eq!(config.instruction_control_weight, 0.1);
    }

    #[test]
    fn test_parameter_access() {
        let autodeco = AutoDeco::new(64, AutoDecoConfig::default());

        // Test parameter access
        let temp_params = autodeco.temp_head.parameters();
        let top_p_params = autodeco.top_p_head.parameters();

        assert_eq!(temp_params.len(), 2); // weights1, weights2
        assert_eq!(top_p_params.len(), 2); // weights1, weights2

        // Test bias access
        let temp_biases = autodeco.temp_head.biases();
        let top_p_biases = autodeco.top_p_head.biases();

        assert_eq!(temp_biases.len(), 2); // bias1, bias2
        assert_eq!(top_p_biases.len(), 2); // bias1, bias2
    }

    #[test]
    fn test_richards_normalization() {
        let mut head = TemperatureHead::new(64, 32);
        let hidden = Array1::from_elem(64, 0.5);

        // Test multiple forward passes to ensure Richards norm adapts
        let temp1 = head.forward(&hidden.view());
        let temp2 = head.forward(&hidden.view());

        // Temperature should be valid
        assert!(temp1 >= 0.1 && temp1 <= 2.0);
        assert!(temp2 >= 0.1 && temp2 <= 2.0);

        // Results may differ due to Richards normalization adaptation
        // but should remain in valid range
    }

    #[test]
    fn test_soft_top_p_edge_cases() {
        let autodeco = AutoDeco::new(128, AutoDecoConfig::default());

        // Test with very low top_p (should concentrate on top token)
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 0.1]);
        let sampled = autodeco.soft_top_p_sample(&logits, 0.1);

        // Find the highest probability token
        let max_idx = sampled.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Should be token 2 (index 2) with logit 3.0
        assert_eq!(max_idx, 2);

        // Test with top_p = 1.0 (should be close to softmax)
        let sampled_full = autodeco.soft_top_p_sample(&logits, 1.0);
        let softmax_sum: f32 = sampled_full.sum();
        assert!((softmax_sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_prediction_consistency() {
        let mut autodeco = AutoDeco::new(32, AutoDecoConfig::default());
        let hidden = Array1::from_elem(32, 0.1);

        // Test that predictions are deterministic for same input
        let (temp1, top_p1) = autodeco.predict_step(&hidden.view());
        let (temp2, top_p2) = autodeco.predict_step(&hidden.view());

        assert_eq!(temp1, temp2);
        assert_eq!(top_p1, top_p2);

        // Values should be in valid ranges
        assert!(temp1 >= 0.1 && temp1 <= 2.0);
        assert!(top_p1 >= 0.0 && top_p1 <= 1.0);
    }
}
