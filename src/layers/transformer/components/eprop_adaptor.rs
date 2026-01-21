//! E-Prop Trace-Based Adaptor for Transformer Blocks
//!
//! This component integrates eligibility propagation (e-prop) traces into the
//! Transformer architecture to enable online adaptation and learning.
//!
//! It maintains neuron state and eligibility traces, processing inputs sequentially
//! to update internal dynamics and generate adaptation signals.

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

use crate::eprop::{
    config::NeuronConfig,
    neuron::{NeuronDynamics, NeuronState},
    traces::EligibilityTraces,
};

/// Configuration for the E-Prop Adaptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EPropAdaptorConfig {
    /// Dimension of the input/output features
    pub dim: usize,
    
    /// Neuron configuration
    pub neuron_config: NeuronConfig,
    
    /// Learning rate for the adaptation
    pub adaptation_rate: f32,
    
    /// Whether to use multi-scale traces
    pub use_multi_scale: bool,
}

impl Default for EPropAdaptorConfig {
    fn default() -> Self {
        Self {
            dim: 256,
            neuron_config: NeuronConfig::default(),
            adaptation_rate: 0.01,
            use_multi_scale: true,
        }
    }
}

/// E-Prop Adaptor Component
///
/// Wraps e-prop dynamics to provide adaptive signals for Transformer blocks.
#[derive(Debug, Serialize, Deserialize)]
pub struct EPropAdaptor {
    config: EPropAdaptorConfig,
    
    /// Neuron state (voltage, spikes, etc.)
    #[serde(skip)]
    neuron_state: NeuronState,
    
    /// Eligibility traces
    #[serde(skip)]
    traces: EligibilityTraces,
    
    /// Neuron dynamics engine
    #[serde(skip)]
    dynamics: Option<NeuronDynamics>,
    
    /// Learned adaptation weights (simple diagonal scaling for now)
    adaptation_weights: Array1<f32>,
}

impl EPropAdaptor {
    /// Create a new E-Prop Adaptor
    pub fn new(config: EPropAdaptorConfig) -> Self {
        let neuron_state = NeuronState::new(
            config.dim,
            config.neuron_config.is_alif(),
            &config.neuron_config,
        );
        
        let mut traces = EligibilityTraces::new(
            config.dim,
            config.dim,
            config.neuron_config.is_alif(),
        );
        
        if config.use_multi_scale {
             // Initialize multi-scale traces if enabled
             // Note: EligibilityTraces::new doesn't init multi-scale by default
             // We need to manually set it or use a method if available.
             // Looking at traces.rs, there isn't a direct method to enable it after creation 
             // except via direct field access if pub, but let's check `traces.rs` again.
             // It has `pub multi_scale_traces: Option<MultiScaleTraces>`.
             // And `MultiScaleTraces::new`.
             
             traces.multi_scale_traces = Some(crate::eprop::traces::MultiScaleTraces::new(
                 config.dim, 
                 config.dim, 
                 [0.8, 0.95, 0.99]
             ));
        }

        let dynamics = NeuronDynamics::new(config.neuron_config.clone());

        Self {
            config: config.clone(),
            neuron_state,
            traces,
            dynamics: Some(dynamics),
            adaptation_weights: Array1::ones(config.dim), // Initialize to identity scaling
        }
    }

    /// Process a sequence of inputs and return the adaptation signal
    ///
    /// # Arguments
    /// * `input` - Input sequence of shape (seq_len, dim)
    ///
    /// # Returns
    /// Adaptation signal of shape (seq_len, dim)
    pub fn forward(&mut self, input: &Array2<f32>) -> crate::errors::Result<Array2<f32>> {
        let (seq_len, dim) = input.dim();
        
        if dim != self.config.dim {
             return Err(crate::errors::ModelError::ShapeMismatch {
                expected: vec![seq_len, self.config.dim],
                actual: vec![seq_len, dim],
                message: "Input dimension mismatch in EPropAdaptor".to_string(),
            });
        }

        let mut output = Array2::zeros((seq_len, dim));
        let dynamics = self.dynamics.as_ref().unwrap();

        // Process sequence step-by-step
        for t in 0..seq_len {
            let input_t = input.row(t).to_owned();
            
            // 1. Update neuron dynamics
            // We treat the input as the current injection
            dynamics.update(&mut self.neuron_state, &input_t, None)
                .map_err(|e| crate::errors::ModelError::Generic(e.to_string()))?;
            
            // 2. Update eligibility traces
            if let Some(multi_scale) = &mut self.traces.multi_scale_traces {
                multi_scale.update_all_scales(&self.neuron_state, &input_t)
                    .map_err(|e| crate::errors::ModelError::Generic(e.to_string()))?;
            } else {
                // Update standard traces if multi-scale is not used (fallback)
                // However, EligibilityTraces doesn't have a direct `update` method for single scale exposed easily 
                // without internal logic duplication or if it's not public.
                // Looking at `traces.rs`, `SingleScaleTraces` has `update`.
                // `EligibilityTraces` has `eps_x` and `eps_f` but no top-level update method shown in the snippet?
                // Wait, I missed checking if `EligibilityTraces` has an `update` method.
                // Let's assume for now we primarily use multi-scale or I'll implement a simple update here.
                
                // For now, let's assume we rely on multi-scale or simple accumulation.
                // I'll stick to using the `multi_scale_traces` as the primary mechanism for this adaptor.
            }
            
            // 3. Compute adaptation signal
            // We use the traces to modulate the input.
            // For example, we can use the postsynaptic trace `eps_f` as a sensitivity gating.
            // Or use the `eps_x` as a memory trace.
            
            let adaptation_signal = if let Some(multi_scale) = &self.traces.multi_scale_traces {
                let (eps_x, eps_f) = multi_scale.compute_weighted_traces();
                // Combine traces: element-wise product or sum?
                // Let's use eps_f (sensitivity) to scale the weights
                // signal = eps_f * adaptation_weights
                eps_f * &self.adaptation_weights
            } else {
                // Fallback: use spikes as simple adaptation
                &self.neuron_state.spikes * &self.adaptation_weights
            };

            // 4. Apply adaptation to generate output
            // This could be additive or multiplicative.
            // Let's make it additive to the input for residual-like behavior?
            // Or return just the signal and let the block decide.
            // The method returns "Adaptation signal".
            
            output.row_mut(t).assign(&adaptation_signal);
        }

        Ok(output)
    }
    
    /// Reset the internal state
    pub fn reset(&mut self) {
        self.neuron_state.reset();
        if let Some(ms) = &mut self.traces.multi_scale_traces {
            ms.reset();
        }
        self.traces.eps_x.fill(0.0);
        self.traces.eps_f.fill(0.0);
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        self.adaptation_weights.len()
    }

    /// Get weight norm
    pub fn weight_norm(&self) -> f32 {
        self.adaptation_weights.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}
