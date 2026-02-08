//! E-Prop Trace-Based Adaptor for Transformer Blocks
//!
//! This component integrates eligibility propagation (e-prop) traces into the
//! Transformer architecture to enable online adaptation and learning.
//!
//! It maintains neuron state and eligibility traces, processing inputs sequentially
//! to update internal dynamics and generate adaptation signals.

use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use serde::{Deserialize, Serialize};

use crate::domain::eprop::{
    config::NeuronConfig,
    neuron::{NeuronDynamics, NeuronState, NeuronWorkspace},
    traces::EligibilityTraces,
};

/// Workspace for zero-allocation streaming inference in E-Prop Adaptor
#[derive(Debug, Default, Clone)]
pub struct EPropAdaptorStreamingWorkspace {
    pub neuron_workspace: crate::domain::eprop::neuron::NeuronWorkspace,
    pub eps_f_workspace: Array1<f32>,
}

/// Configuration for the E-Prop Adaptor
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Cached traces for gradient computation
    #[serde(skip)]
    cached_traces: Option<Array2<f32>>,
}

impl EPropAdaptor {
    /// Create a new E-Prop Adaptor
    pub fn new(config: EPropAdaptorConfig) -> Self {
        let neuron_state = NeuronState::new(
            config.dim,
            config.neuron_config.is_alif(),
            &config.neuron_config,
        );

        let mut traces =
            EligibilityTraces::new(config.dim, config.dim, config.neuron_config.is_alif());

        if config.use_multi_scale {
            // Initialize multi-scale traces if enabled
            traces.multi_scale_traces = Some(crate::domain::eprop::traces::MultiScaleTraces::new(
                config.dim,
                config.dim,
                [0.8, 0.95, 0.99],
            ));
        }

        let dynamics = NeuronDynamics::new(config.neuron_config.clone());

        Self {
            config: config.clone(),
            neuron_state,
            traces,
            dynamics: Some(dynamics),
            adaptation_weights: Array1::ones(config.dim), // Initialize to identity scaling
            cached_traces: None,
        }
    }

    /// Process a single step for streaming inference (Zero-Allocation)
    pub fn forward_step_into(
        &mut self,
        input: &ArrayView1<f32>,
        output: &mut ArrayViewMut1<f32>,
        workspace: &mut EPropAdaptorStreamingWorkspace,
    ) -> crate::common::errors::Result<()> {
        // Initialize dynamics if needed
        if self.dynamics.is_none() {
            self.dynamics = Some(NeuronDynamics::new(self.config.neuron_config.clone()));
        }
        let dynamics = self.dynamics.as_ref().unwrap();

        // Initialize state if needed
        if self.neuron_state.voltage.len() != self.config.dim {
             self.neuron_state = NeuronState::new(
                self.config.dim,
                self.config.neuron_config.is_alif(),
                &self.config.neuron_config,
            );
            
             self.traces = EligibilityTraces::new(
                self.config.dim,
                self.config.dim,
                self.config.neuron_config.is_alif(),
            );

            if self.config.use_multi_scale {
                self.traces.multi_scale_traces = Some(crate::domain::eprop::traces::MultiScaleTraces::new(
                    self.config.dim,
                    self.config.dim,
                    [0.8, 0.95, 0.99],
                ));
            }
        }

        // 1. Update neuron dynamics
        dynamics
            .update(&mut self.neuron_state, input, None, &mut workspace.neuron_workspace)
            .map_err(|e| crate::common::errors::ModelError::Generic(e.to_string()))?;

        // 2. Update eligibility traces
        if let Some(multi_scale) = &mut self.traces.multi_scale_traces {
            multi_scale
                .update_all_scales(&self.neuron_state, *input)
                .map_err(|e| crate::common::errors::ModelError::Generic(e.to_string()))?;
        }

        // 3. Compute adaptation signal
        use ndarray::Zip;

        if let Some(multi_scale) = &self.traces.multi_scale_traces {
             // Ensure workspace capacity
             if workspace.eps_f_workspace.len() != self.config.dim {
                 workspace.eps_f_workspace = Array1::zeros(self.config.dim);
             }

             // Compute weighted eps_f into workspace
             multi_scale.compute_weighted_traces_into(None, Some(&mut workspace.eps_f_workspace));
             
             // adaptation = eps_f * learned_weights
             Zip::from(output)
                .and(&workspace.eps_f_workspace)
                .and(&self.adaptation_weights)
                .for_each(|o, &e, &w| *o = e * w);
        } else {
             // Fallback: adaptation = spikes * learned_weights
             Zip::from(output)
                .and(&self.neuron_state.spikes) // Note: using spikes directly, not filtered_spikes, matching forward() logic
                .and(&self.adaptation_weights)
                .for_each(|o, &s, &w| *o = s * w);
        }

        Ok(())
    }

    /// Process a sequence of inputs and return the adaptation signal
    ///
    /// # Arguments
    /// * `input` - Input sequence of shape (seq_len, dim)
    ///
    /// # Returns
    /// Process a sequence of inputs and return the adaptation signal
    pub fn forward(&mut self, input: &Array2<f32>) -> crate::common::errors::Result<Array2<f32>> {
        let (seq_len, dim) = input.dim();

        if dim != self.config.dim {
            return Err(crate::common::errors::ModelError::ShapeMismatch {
                expected: vec![seq_len, self.config.dim],
                actual: vec![seq_len, dim],
                message: "Input dimension mismatch in EPropAdaptor".to_string(),
            });
        }

        // Initialize state if needed (e.g. after deserialization)
        if self.dynamics.is_none() {
            self.dynamics = Some(NeuronDynamics::new(self.config.neuron_config.clone()));
        }

        if self.neuron_state.voltage.len() != self.config.dim {
            self.neuron_state = NeuronState::new(
                self.config.dim,
                self.config.neuron_config.is_alif(),
                &self.config.neuron_config,
            );

            self.traces = EligibilityTraces::new(
                self.config.dim,
                self.config.dim,
                self.config.neuron_config.is_alif(),
            );

            if self.config.use_multi_scale {
                self.traces.multi_scale_traces = Some(crate::domain::eprop::traces::MultiScaleTraces::new(
                    self.config.dim,
                    self.config.dim,
                    [0.8, 0.95, 0.99],
                ));
            }
        }

        let mut output = Array2::zeros((seq_len, dim));
        // Allocate cache for traces
        let mut trace_cache = Array2::zeros((seq_len, dim));

        let dynamics = self.dynamics.as_ref().unwrap();
        let mut neuron_workspace = NeuronWorkspace::new(self.config.dim);

        // Process sequence step-by-step
        for t in 0..seq_len {
            let input_t = input.row(t).to_owned();

            // 1. Update neuron dynamics
            // We treat the input as the current injection
            dynamics
                .update(&mut self.neuron_state, &input_t.view(), None, &mut neuron_workspace)
                .map_err(|e| crate::common::errors::ModelError::Generic(e.to_string()))?;

            // 2. Update eligibility traces
            if let Some(multi_scale) = &mut self.traces.multi_scale_traces {
                multi_scale
                    .update_all_scales(&self.neuron_state, input_t.view())
                    .map_err(|e| crate::common::errors::ModelError::Generic(e.to_string()))?;
            }

            // 3. Compute adaptation signal
            let adaptation_signal = if let Some(multi_scale) = &self.traces.multi_scale_traces {
                let (_eps_x, eps_f) = multi_scale.compute_weighted_traces();
                // Store trace for gradient computation
                trace_cache.row_mut(t).assign(&eps_f);

                eps_f * &self.adaptation_weights
            } else {
                // Fallback: use spikes as simple adaptation
                // Store spikes as "trace"
                trace_cache.row_mut(t).assign(&self.neuron_state.spikes);

                &self.neuron_state.spikes * &self.adaptation_weights
            };

            // 4. Apply adaptation to generate output
            output.row_mut(t).assign(&adaptation_signal);
        }

        // Save traces for backward pass
        self.cached_traces = Some(trace_cache);

        Ok(output)
    }

    /// Compute gradients using cached traces and output gradients (e-prop rule)
    ///
    /// # Arguments
    /// * `output_grads` - Gradients w.r.t. the output of the adaptor
    ///
    /// # Returns
    /// * `input_grads` - Gradients w.r.t. the input (pass-through of output_grads)
    /// * `param_grads` - Gradients w.r.t. adaptation weights
    pub fn compute_gradients(&self, output_grads: &Array2<f32>) -> (Array2<f32>, Vec<Array2<f32>>) {
        // e-prop rule: dW = sum(dL/dy * trace)
        // input_grads = dL/dy (ignoring backprop through dynamics for now)

        let mut param_grads = Array1::zeros(self.config.dim);

        if let Some(traces) = &self.cached_traces {
            let (seq_len, _dim) = output_grads.dim();
            let (trace_seq, _trace_dim) = traces.dim();

            let len = seq_len.min(trace_seq);

            for t in 0..len {
                let grad_t = output_grads.row(t);
                let trace_t = traces.row(t);
                // Element-wise multiplication and accumulation
                param_grads = param_grads + (&grad_t * &trace_t);
            }
        }

        // Return gradients. Convert param_grads to Array2 (dim, 1) for compatibility
        let param_grads_2d = param_grads.insert_axis(ndarray::Axis(1));

        // Pass-through gradients for input
        (output_grads.clone(), vec![param_grads_2d])
    }

    /// Apply gradients to adaptation weights
    pub fn apply_gradients(&mut self, grads: &[Array2<f32>], lr: f32) -> crate::common::errors::Result<()> {
        if grads.is_empty() {
            return Ok(());
        }

        // We expect one gradient matrix of shape (dim, 1)
        let grad = &grads[0];
        let grad_1d = grad.column(0);

        // Simple SGD update: W = W - lr * grad
        self.adaptation_weights = &self.adaptation_weights - &(grad_1d.mapv(|x| x * lr));

        Ok(())
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
        self.adaptation_weights
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
    }
}
