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
    /// Clipping threshold for adaptive clipping
    pub threshold: f32,
    /// Whether to apply gradient centralization
    pub use_centralization: bool,
    /// Adaptive scaling factor (approximates AGC)
    pub adaptive_factor: f32,
    /// Fallback clipping type when adaptive is disabled
    pub fallback_clipping: ClippingType,
}

/// Types of gradient clipping
#[derive(Debug, Clone)]
pub enum ClippingType {
    L2(f32),
    L1(f32),
    ElementWise(f32),
}

impl Default for AdaptiveClippingConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0, // Increased from 0.01 for less aggressive clipping
            use_centralization: true,
            adaptive_factor: 1.0,                      // No additional scaling
            fallback_clipping: ClippingType::L2(10.0), // Increased from 5.0
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

    /// Apply adaptive scaling based on gradient magnitude
    fn apply_adaptive_scaling(grads: &mut Array2<f32>, threshold: f32, factor: f32) {
        // Calculate gradient norm
        let grad_norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if grad_norm > 0.0 {
            // Adaptive scaling: scale based on effective gradient magnitude
            let num_elements = grads.len() as f32;
            let effective_norm = grad_norm / num_elements.sqrt();
            let adaptive_threshold = threshold * factor;
            let scale = (adaptive_threshold / effective_norm).min(1.0);

            if scale < 1.0 {
                grads.mapv_inplace(|x| x * scale);
            }
        }
    }

    /// Apply fallback clipping based on type
    fn apply_fallback_clipping(grads: &mut Array2<f32>, clipping_type: &ClippingType) {
        match clipping_type {
            ClippingType::L2(threshold) => {
                let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > *threshold {
                    let scale = *threshold / norm;
                    grads.mapv_inplace(|x| x * scale);
                }
            }
            ClippingType::L1(threshold) => {
                let l1_norm = grads.iter().map(|&x| x.abs()).sum::<f32>();
                if l1_norm > *threshold {
                    let scale = *threshold / l1_norm;
                    grads.mapv_inplace(|x| x * scale);
                }
            }
            ClippingType::ElementWise(threshold) => {
                grads.mapv_inplace(|x| {
                    if x > *threshold {
                        *threshold
                    } else if x < -*threshold {
                        -*threshold
                    } else {
                        x
                    }
                });
            }
        }
    }
}

impl GradientClipping for AdaptiveGradientClipping {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        // Handle NaN/inf values
        grads.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });

        // Apply gradient centralization first if enabled
        if self.config.use_centralization {
            Self::centralize_gradients(grads);
        }

        // Apply adaptive scaling
        Self::apply_adaptive_scaling(grads, self.config.threshold, self.config.adaptive_factor);

        // Apply fallback clipping if needed (adaptive scaling might not be sufficient)
        Self::apply_fallback_clipping(grads, &self.config.fallback_clipping);
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            config: self.config.clone(),
        })
    }
}

/// Simple L2 norm gradient clipping (legacy implementation)
#[derive(Clone, Debug)]
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
        // Handle NaN/inf values
        grads.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });

        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if norm > self.threshold && norm > 0.0 {
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

/// L1 norm gradient clipping
#[derive(Clone, Debug)]
pub struct L1GradientClipping {
    threshold: f32,
}

impl L1GradientClipping {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl GradientClipping for L1GradientClipping {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        let l1_norm = grads.iter().map(|&x| x.abs()).sum::<f32>();

        if l1_norm > self.threshold {
            let scale = self.threshold / l1_norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            threshold: self.threshold,
        })
    }
}

/// Element-wise gradient clipping (clip each gradient individually)
#[derive(Clone, Debug)]
pub struct ElementWiseClipping {
    threshold: f32,
}

impl ElementWiseClipping {
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

impl GradientClipping for ElementWiseClipping {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        grads.mapv_inplace(|x| {
            if x > self.threshold {
                self.threshold
            } else if x < -self.threshold {
                -self.threshold
            } else {
                x
            }
        });
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            threshold: self.threshold,
        })
    }
}

/// True Adaptive Gradient Clipping (AGC) from Brock et al. 2021
/// Reference: "High-Performance Large-Scale Image Recognition Without Normalization"
///
/// AGC clips gradients based on the ratio of gradient norm to parameter norm:
///   g_i ← g_i * min(1, λ * ||w_i|| / (||g_i|| + ε))
///
/// where:
/// - g_i: gradient for parameter i
/// - w_i: parameter i
/// - λ: clipping threshold (typically 0.01-0.1)
/// - ε: small constant for numerical stability (typically 1e-3)
///
/// This is more adaptive than global norm clipping because it considers
/// the scale of each parameter group separately.
#[derive(Clone, Debug)]
pub struct TrueAGC {
    /// Clipping threshold λ (typically 0.01-0.1)
    pub lambda: f32,
    /// Numerical stability constant ε (typically 1e-3)
    pub epsilon: f32,
}

impl TrueAGC {
    /// Create a new AGC clipper with default parameters
    pub fn new() -> Self {
        Self {
            lambda: 0.01,
            epsilon: 1e-3,
        }
    }

    /// Create a new AGC clipper with custom parameters
    pub fn with_params(lambda: f32, epsilon: f32) -> Self {
        Self { lambda, epsilon }
    }

    /// Apply AGC to a single parameter-gradient pair
    /// This should be called per-parameter, not globally
    pub fn clip_parameter_gradient(
        &self,
        param: &Array2<f32>,
        grad: &mut Array2<f32>,
    ) {
        // Compute parameter norm
        let param_norm = param.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Compute gradient norm
        let grad_norm = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Compute clipping coefficient
        let clip_coef = self.lambda * param_norm / (grad_norm + self.epsilon);

        // Apply clipping if needed
        if clip_coef < 1.0 {
            grad.mapv_inplace(|x| x * clip_coef);
        }
    }
}

impl Default for TrueAGC {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientClipping for TrueAGC {
    fn clip_gradients(&mut self, grads: &mut Array2<f32>) {
        // Note: This implementation assumes grads is a single parameter's gradients
        // For proper AGC, we need access to the parameters themselves
        // This is a simplified version that uses gradient norm only

        // Handle NaN/inf values
        grads.mapv_inplace(|x| if x.is_finite() { x } else { 0.0 });

        // Simplified AGC without parameter access:
        // Use gradient norm as a proxy for parameter norm
        let grad_norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if grad_norm > 0.0 {
            // Assume parameter norm is proportional to gradient norm
            // This is a rough approximation
            let estimated_param_norm = grad_norm * 10.0; // Heuristic scaling
            let clip_coef = self.lambda * estimated_param_norm / (grad_norm + self.epsilon);

            if clip_coef < 1.0 {
                grads.mapv_inplace(|x| x * clip_coef);
            }
        }
    }

    fn clone_box(&self) -> Box<dyn GradientClipping> {
        Box::new(Self {
            lambda: self.lambda,
            epsilon: self.epsilon,
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
            threshold: 0.1,
            use_centralization: true,
            adaptive_factor: 1.0,
            fallback_clipping: ClippingType::L2(5.0),
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
    fn test_adaptive_clipping_without_centralization() {
        let mut grads = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
        let config = AdaptiveClippingConfig {
            threshold: 0.01,
            use_centralization: false,
            adaptive_factor: 1.0,
            fallback_clipping: ClippingType::L2(5.0),
        };
        let mut clipper = AdaptiveGradientClipping::new(config);

        clipper.clip_gradients(&mut grads);

        // Should apply adaptive scaling and fallback L2
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 5.0);
    }

    #[test]
    fn test_l1_clipping() {
        let mut grads =
            Array2::from_shape_vec((2, 3), vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();
        let mut clipper = L1GradientClipping::new(10.0);

        // L1 norm is |1| + |-2| + |3| + |-4| + |5| + |-6| = 21
        // Should scale by 10/21 ≈ 0.476
        clipper.clip_gradients(&mut grads);

        let l1_norm = grads.iter().map(|&x| x.abs()).sum::<f32>();
        assert!((l1_norm - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_element_wise_clipping() {
        let mut grads =
            Array2::from_shape_vec((2, 3), vec![1.0, -3.0, 2.0, -5.0, 0.5, 4.0]).unwrap();
        let mut clipper = ElementWiseClipping::new(2.0);

        clipper.clip_gradients(&mut grads);

        // Values should be clamped to [-2, 2]
        for &val in grads.iter() {
            assert!(val >= -2.0 && val <= 2.0);
        }
        assert_eq!(grads[[0, 0]], 1.0); // unchanged
        assert_eq!(grads[[0, 1]], -2.0); // clamped
        assert_eq!(grads[[0, 2]], 2.0); // clamped
    }
}
