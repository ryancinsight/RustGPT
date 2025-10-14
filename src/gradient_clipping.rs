//! Adaptive Gradient Clipping Implementation
//!
//! This module provides advanced gradient clipping techniques beyond simple L2 norm clipping,
//! including Adaptive Gradient Clipping (AGC) and Gradient Centralization.

use ndarray::Array2;

/// Trait for gradient clipping strategies
pub trait GradientClipping: Send + Sync {
    /// Apply gradient clipping to the given gradients
    fn clip_gradients(&mut self, grads: &mut Array2<f32>);

    /// Clone the clipping strategy
    fn clone_box(&self) -> Box<dyn GradientClipping>;
}

/// Configuration for adaptive gradient clipping
#[derive(Debug, Clone)]
pub struct AdaptiveClippingConfig {
    /// Clipping threshold for AGC (λ parameter)
    pub agc_threshold: f32,
    /// Whether to apply gradient centralization
    pub use_centralization: bool,
    /// Whether to use AGC (Adaptive Gradient Clipping)
    pub use_agc: bool,
    /// Fallback L2 norm threshold when AGC is disabled
    pub l2_threshold: f32,
}

impl Default for AdaptiveClippingConfig {
    fn default() -> Self {
        Self {
            agc_threshold: 0.01, // Standard AGC threshold
            use_centralization: true,
            use_agc: true,
            l2_threshold: 5.0, // Fallback L2 threshold
        }
    }
}

/// Adaptive gradient clipping implementation
pub struct AdaptiveGradientClipping {
    config: AdaptiveClippingConfig,
}

impl AdaptiveGradientClipping {
    /// Create a new adaptive gradient clipping instance
    pub fn new(config: AdaptiveClippingConfig) -> Self {
        Self { config }
    }

    /// Apply gradient centralization (center gradients around zero mean)
    pub fn centralize_gradients(grads: &mut Array2<f32>) {
        // For each row (feature dimension), subtract the mean
        for mut row in grads.rows_mut() {
            let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
            row.mapv_inplace(|x| x - mean);
        }
    }

    /// Apply Adaptive Gradient Clipping (AGC) based on parameter norms
    /// Note: This is a simplified version since we don't have direct access to parameters
    /// In a full implementation, this would use parameter norms per layer
    fn apply_agc(grads: &mut Array2<f32>, threshold: f32) {
        // Calculate gradient norm
        let grad_norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if grad_norm > 0.0 {
            // AGC scaling factor: min(1, threshold / (grad_norm / sqrt(num_elements)))
            // This approximates the parameter-norm based scaling
            let num_elements = grads.len() as f32;
            let effective_grad_norm = grad_norm / num_elements.sqrt();
            let scale = (threshold / effective_grad_norm).min(1.0);

            if scale < 1.0 {
                grads.mapv_inplace(|x| x * scale);
            }
        }
    }

    /// Fallback L2 norm clipping
    fn apply_l2_clipping(grads: &mut Array2<f32>, threshold: f32) {
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm > threshold {
            let scale = threshold / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }
}

impl GradientClipping for AdaptiveGradientClipping {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        // Apply gradient centralization first if enabled
        if self.config.use_centralization {
            Self::centralize_gradients(grads);
        }

        // Apply AGC or fallback to L2 clipping
        if self.config.use_agc {
            Self::apply_agc(grads, self.config.agc_threshold);
        } else {
            Self::apply_l2_clipping(grads, self.config.l2_threshold);
        }
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            config: self.config.clone(),
        })
    }
}

/// Simple L2 norm gradient clipping (legacy implementation)
pub struct L2GradientClipping {
    threshold: f32,
}

impl L2GradientClipping {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl GradientClipping for L2GradientClipping {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm > self.threshold {
            let scale = self.threshold / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            threshold: self.threshold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_l2_clipping_basic() {
        let mut grads = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        let mut clipper = L2GradientClipping::new(5.0);

        // Norm is sqrt(3^2 + 4^2 + 5^2) = sqrt(9 + 16 + 25) = sqrt(50) ≈ 7.07
        // Should be scaled by 5.0/7.07 ≈ 0.707
        clipper.clip_gradients(&mut grads);

        let expected_norm = 5.0;
        let actual_norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((actual_norm - expected_norm).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_clipping_with_centralization() {
        let mut grads = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let config = AdaptiveClippingConfig {
            agc_threshold: 0.1,
            use_centralization: true,
            use_agc: true,
            l2_threshold: 5.0,
        };
        let mut clipper = AdaptiveGradientClipping::new(config);

        let original_grads = grads.clone();
        clipper.clip_gradients(&mut grads);

        // After centralization, each row should have zero mean
        for row in grads.rows() {
            let mean: f32 = row.iter().sum::<f32>() / row.len() as f32;
            assert!(
                mean.abs() < 1e-6,
                "Row mean should be zero after centralization"
            );
        }

        // Gradients should be modified from original
        assert_ne!(grads, original_grads);
    }

    #[test]
    fn test_adaptive_clipping_without_agc() {
        let mut grads = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        let config = AdaptiveClippingConfig {
            agc_threshold: 0.01,
            use_centralization: false,
            use_agc: false,
            l2_threshold: 5.0,
        };
        let mut clipper = AdaptiveGradientClipping::new(config);

        clipper.clip_gradients(&mut grads);

        // Should behave like L2 clipping
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_centralization() {
        let mut grads = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Row 0: [1, 2, 3], mean = 2.0, centralized: [-1, 0, 1]
        // Row 1: [4, 5, 6], mean = 5.0, centralized: [-1, 0, 1]

        AdaptiveGradientClipping::centralize_gradients(&mut grads);

        let expected =
            Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]).unwrap();
        assert_eq!(grads, expected);
    }
}
