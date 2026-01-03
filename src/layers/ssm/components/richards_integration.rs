//! Richards Activation Integration for SSMs
//!
//! Provides integration between SSM components and the Richards activation system,
//! enabling learnable, adaptive activation functions for state space models.

use ndarray::Array2;
use serde::{Serialize, Deserialize};

use crate::richards::{RichardsActivation, Variant};

/// SSM-specific Richards activation wrapper
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SsmRichardsActivation {
    /// Underlying Richards activation
    pub activation: RichardsActivation,
    /// Whether to use element-wise multiplication (x * Richards(x))
    pub use_elementwise_mult: bool,
}

impl SsmRichardsActivation {
    /// Create a new SSM Richards activation with specified variant
    pub fn new(variant: Variant, use_elementwise_mult: bool) -> Self {
        Self {
            activation: RichardsActivation::new_learnable(variant),
            use_elementwise_mult,
        }
    }

    /// Create a sigmoid-based activation (similar to Swish)
    pub fn sigmoid(learnable: bool, use_elementwise_mult: bool) -> Self {
        Self {
            activation: RichardsActivation::sigmoid(learnable),
            use_elementwise_mult,
        }
    }

    /// Create a tanh-based activation
    pub fn tanh(learnable: bool, use_elementwise_mult: bool) -> Self {
        Self {
            activation: RichardsActivation::tanh(learnable),
            use_elementwise_mult,
        }
    }

    /// Create a Gompertz-based activation
    pub fn gompertz(learnable: bool, use_elementwise_mult: bool) -> Self {
        Self {
            activation: RichardsActivation::gompertz(learnable),
            use_elementwise_mult,
        }
    }

    /// Forward pass for f32 matrix input
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        if self.use_elementwise_mult {
            self.activation.forward_matrix_f32(x)
        } else {
            // Just apply Richards curve without elementwise multiplication
            let mut result = Array2::zeros(x.raw_dim());
            self.activation.richards_curve.forward_matrix_f32_into(x, &mut result);
            result
        }
    }

    /// Forward pass that writes to a provided output buffer
    pub fn forward_into(&self, x: &Array2<f32>, out: &mut Array2<f32>) {
        if self.use_elementwise_mult {
            self.activation.forward_matrix_f32_into(x, out);
        } else {
            self.activation.richards_curve.forward_matrix_f32_into(x, out);
        }
    }

    /// Get the underlying Richards curve
    pub fn richards_curve(&self) -> &RichardsActivation {
        &self.activation
    }

    /// Get mutable access to the underlying Richards curve
    pub fn richards_curve_mut(&mut self) -> &mut RichardsActivation {
        &mut self.activation
    }

    /// Reset the Richards curve parameters
    pub fn reset_parameters(&mut self) {
        // Create a new Richards curve with the same variant
        let variant = self.activation.richards_curve.variant;
        self.activation = RichardsActivation::new_learnable(variant);
    }
}

/// SSM activation configuration
#[derive(Debug, Clone, Copy)]
pub struct SsmActivationConfig {
    /// Activation variant to use
    pub variant: Variant,
    /// Whether to use element-wise multiplication (x * Richards(x))
    pub use_elementwise_mult: bool,
    /// Whether the activation parameters are learnable
    pub learnable: bool,
}

impl Default for SsmActivationConfig {
    fn default() -> Self {
        Self {
            variant: Variant::Sigmoid, // Default to sigmoid-like activation
            use_elementwise_mult: true, // Default to Swish-like behavior
            learnable: true, // Default to learnable parameters
        }
    }
}

impl SsmActivationConfig {
    /// Create a sigmoid-based activation config (Swish-like)
    pub fn sigmoid(learnable: bool) -> Self {
        Self {
            variant: Variant::Sigmoid,
            use_elementwise_mult: true,
            learnable,
        }
    }

    /// Create a tanh-based activation config
    pub fn tanh(learnable: bool) -> Self {
        Self {
            variant: Variant::Tanh,
            use_elementwise_mult: true,
            learnable,
        }
    }

    /// Create a Gompertz-based activation config
    pub fn gompertz(learnable: bool) -> Self {
        Self {
            variant: Variant::Gompertz,
            use_elementwise_mult: true,
            learnable,
        }
    }

    /// Create from the config
    pub fn create_activation(&self) -> SsmRichardsActivation {
        match self.variant {
            Variant::Sigmoid => SsmRichardsActivation::sigmoid(self.learnable, self.use_elementwise_mult),
            Variant::Tanh => SsmRichardsActivation::tanh(self.learnable, self.use_elementwise_mult),
            Variant::Gompertz => SsmRichardsActivation::gompertz(self.learnable, self.use_elementwise_mult),
            _ => SsmRichardsActivation::new(self.variant, self.use_elementwise_mult),
        }
    }
}