//! Neuron dynamics implementation (LIF/ALIF)
//!
//! This module implements the core spiking neuron models used in e-prop:
//! - Leaky Integrate-and-Fire (LIF)
//! - Adaptive LIF (ALIF) with spike-frequency adaptation
//!
//! Both models support forward computation with surrogate gradients for
//! biologically plausible online learning.

use ndarray::{Array1, Zip};

// use crate::domain::eprop::adaptive_surrogate::{AdaptiveSurrogate, SurrogatePerformance,
// ActivityStats};
use crate::domain::eprop::adaptive_surrogate::SurrogatePerformance;
use crate::domain::eprop::config::{NeuronConfig, NeuronModel};

/// Workspace for zero-allocation neuron dynamics
#[derive(Debug, Clone, Default)]
pub struct NeuronWorkspace {
    /// Buffer for adaptive threshold computation
    pub threshold: Array1<f32>,
}

impl NeuronWorkspace {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            threshold: Array1::zeros(num_neurons),
        }
    }

    pub fn ensure_capacity(&mut self, num_neurons: usize) {
        if self.threshold.len() != num_neurons {
            self.threshold = Array1::zeros(num_neurons);
        }
    }
}

/// Neuron state for LIF/ALIF dynamics
///
/// Maintains all state variables needed for spiking neuron computation:
/// - Membrane potential (voltage)
/// - Spike outputs
/// - Filtered spikes (low-pass)
/// - Adaptation current (ALIF only)
/// - Surrogate derivatives for gradient computation
#[derive(Debug, Clone, Default)]
pub struct NeuronState {
    /// Membrane potential v_t
    pub voltage: Array1<f32>,

    /// Spike output z_t (binary: 0 or 1)
    pub spikes: Array1<f32>,

    /// Low-pass filtered spikes z̄_t = α * z̄_{t-1} + z_t
    pub filtered_spikes: Array1<f32>,

    /// Adaptation current a_t (ALIF only)
    pub adaptation: Option<Array1<f32>>,

    /// Surrogate derivative ψ_t for backprop approximation
    pub surrogate_deriv: Array1<f32>,

    /// Performance metrics for adaptive surrogates
    pub performance: Option<SurrogatePerformance>,
}

impl NeuronState {
    /// Create initial state (all zeros)
    ///
    /// # Arguments
    /// * `num_neurons` - Number of neurons in the layer
    /// * `use_adaptation` - Whether to allocate adaptation state (for ALIF)
    /// * `config` - Neuron configuration (for adaptive surrogate initialization)
    pub fn new(num_neurons: usize, use_adaptation: bool, config: &NeuronConfig) -> Self {
        let performance = if config.use_adaptive_surrogate {
            Some(SurrogatePerformance::new(
                config.surrogate_performance_window,
            ))
        } else {
            None
        };

        Self {
            voltage: Array1::zeros(num_neurons),
            spikes: Array1::zeros(num_neurons),
            filtered_spikes: Array1::zeros(num_neurons),
            adaptation: if use_adaptation {
                Some(Array1::zeros(num_neurons))
            } else {
                None
            },
            surrogate_deriv: Array1::zeros(num_neurons),
            performance,
        }
    }

    /// Reset state to initial values (all zeros)
    pub fn reset(&mut self) {
        self.voltage.fill(0.0);
        self.spikes.fill(0.0);
        self.filtered_spikes.fill(0.0);
        if let Some(ref mut adapt) = self.adaptation {
            adapt.fill(0.0);
        }
        self.surrogate_deriv.fill(0.0);
    }

    /// Get the number of neurons
    pub fn num_neurons(&self) -> usize {
        self.voltage.len()
    }

    /// Check if adaptation is enabled
    pub fn has_adaptation(&self) -> bool {
        self.adaptation.is_some()
    }
}

/// Neuron dynamics computation engine
///
/// Handles the forward pass for LIF and ALIF neurons, including:
/// - Membrane potential integration
/// - Spike generation with adaptive thresholds
/// - Surrogate gradient computation
/// - State updates
#[derive(Debug, Clone)]
pub struct NeuronDynamics {
    config: NeuronConfig,
}

impl NeuronDynamics {
    /// Create new dynamics engine with given configuration
    pub fn new(config: NeuronConfig) -> Self {
        Self { config }
    }

    /// Update neuron state based on input current
    ///
    /// Implements the LIF/ALIF dynamics with adaptive surrogate gradients.
    ///
    /// # Arguments
    /// * `state` - Current neuron state (will be modified)
    /// * `input_current` - Total input current I_t (recurrent + feedforward)
    /// * `loss_gradient` - Optional loss gradient for adaptive surrogate updates
    /// * `workspace` - Workspace for zero-allocation computation
    ///
    /// # Returns
    /// Ok(()) on success, Err if dimensions mismatch
    pub fn update(
        &self,
        state: &mut NeuronState,
        input_current: &ndarray::ArrayView1<f32>,
        loss_gradient: Option<&Array1<f32>>,
        workspace: &mut NeuronWorkspace,
    ) -> super::Result<()> {
        let n = state.num_neurons();

        if input_current.len() != n {
            return Err(super::EPropError::TraceDimensionMismatch {
                expected: n,
                actual: input_current.len(),
            });
        }
        
        workspace.ensure_capacity(n);

        // Compute adaptive threshold into workspace
        self.compute_threshold_into(state, &mut workspace.threshold)?;

        // Update membrane potential: v_{t+1} = α·v_t + I_t
        // In-place update: state.voltage = state.voltage * alpha + input
        use ndarray::Zip;
        Zip::from(&mut state.voltage)
            .and(input_current)
            .for_each(|v, &i| *v = *v * self.config.alpha + i);

        // Generate spikes and compute surrogate derivatives using adaptive system
        if self.config.use_adaptive_surrogate {
            self.compute_adaptive_spikes_into(&workspace.threshold, state)?;
        } else {
            self.compute_spikes_into(&workspace.threshold, state);
        }

        // Apply spike reset: v -= A_t for neurons that spiked
        // In-place update
        Zip::from(&mut state.voltage)
            .and(&state.spikes)
            .and(&workspace.threshold)
            .for_each(|v, &s, &th| {
                if s > 0.5 {
                    *v -= th;
                }
            });

        // Update filtered spikes: z̄_t = α·z̄_{t-1} + z_t
        // In-place update
        Zip::from(&mut state.filtered_spikes)
            .and(&state.spikes)
            .for_each(|z_bar, &z| *z_bar = *z_bar * self.config.alpha + z);

        // Update adaptation (ALIF only): a_{t+1} = ρ·a_t + z_t
        if let Some(ref mut adaptation) = state.adaptation {
            Zip::from(adaptation)
                .and(&state.spikes)
                .for_each(|a, &z| *a = *a * self.config.rho + z);
        }

        // Update adaptive surrogate performance if enabled
        if self.config.use_adaptive_surrogate {
             // Note: let-chains are not yet stable in all contexts, simplifying
             if let Some(ref mut performance) = state.performance {
                let current_loss = if let Some(loss_grad) = loss_gradient {
                    loss_grad.mapv(|x| x * x).sum().sqrt()
                } else {
                    state.spikes.mapv(|x| x * x).sum()
                };

                if let Some(loss_grad) = loss_gradient {
                    let _ = performance.update_with_gradient(loss_grad, &state.surrogate_deriv, current_loss);
                }

                if performance.should_adapt() {
                    performance.adapt();
                }
             }
        }

        Ok(())
    }

    /// Compute adaptive threshold A_t into workspace
    fn compute_threshold_into(&self, state: &NeuronState, threshold: &mut Array1<f32>) -> super::Result<()> {
        let n = state.num_neurons();
        if threshold.len() != n {
             // Should have been ensured by caller
             *threshold = Array1::from_elem(n, self.config.v_threshold);
        } else {
             threshold.fill(self.config.v_threshold);
        }

        if self.config.model == NeuronModel::ALIF {
            if let Some(ref adaptation) = state.adaptation {
                // threshold += adaptation * beta
                Zip::from(threshold)
                    .and(adaptation)
                    .for_each(|th, &a| *th += a * self.config.beta);
            } else {
                return Err(super::EPropError::InvalidDynamics(
                    "ALIF model requires adaptation state".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Compute spikes and surrogate derivatives using adaptive system
    fn compute_adaptive_spikes_into(
        &self,
        threshold: &Array1<f32>,
        state: &mut NeuronState,
    ) -> super::Result<()> {
        let n = state.voltage.len();
        
        // Get the adaptive surrogate instance
        // Need to extract performance first to avoid borrowing conflict if possible
        // But state.spikes is needed.
        // We can't hold `state.performance` mutable borrow while reading `state.spikes`
        // So we might need to be careful.
        
        let adaptive = if let Some(ref mut perf) = state.performance {
             // We need to clone adaptive to use it, or design this better.
             // For now, let's clone the surrogate which should be cheap (just params)
             perf.get_current_surrogate()
        } else {
             return Err(super::EPropError::InvalidDynamics(
                "Adaptive surrogate performance tracking not initialized".to_string(),
            ));
        };

        // Create activity statistics for adaptation
        let activity_stats = adaptive.create_activity_stats(&state.voltage, threshold, &state.spikes);

        // Update the adaptive surrogate with current activity
        if let Some(ref mut perf) = state.performance {
            perf.update_with_activity(adaptive.clone(), &activity_stats)?;
        }

        // Get the updated surrogate for computation
        let adaptive = if let Some(ref mut perf) = state.performance {
             perf.get_current_surrogate()
        } else {
             unreachable!()
        };

        // Compute spikes using Heaviside step function (binary output)
        for i in 0..n {
            let delta = state.voltage[i] - threshold[i];
            state.spikes[i] = if delta >= 0.0 { 1.0 } else { 0.0 };
        }

        // Compute surrogate derivatives using current adaptive function
        for i in 0..n {
            let delta = state.voltage[i] - threshold[i];
            state.surrogate_deriv[i] = adaptive.derivative(delta);
        }

        Ok(())
    }

    /// Compute spikes and surrogate derivatives (legacy static method)
    fn compute_spikes_into(
        &self,
        threshold: &Array1<f32>,
        state: &mut NeuronState,
    ) {
        let n = state.voltage.len();

        for i in 0..n {
            let delta = state.voltage[i] - threshold[i];

            // Heaviside step function
            state.spikes[i] = if delta >= 0.0 { 1.0 } else { 0.0 };

            // Surrogate derivative: piecewise linear approximation
            let abs_delta = delta.abs() / self.config.v_threshold;
            state.surrogate_deriv[i] = if abs_delta < 1.0 {
                (1.0 - abs_delta) / (self.config.gamma_pd * self.config.v_threshold)
            } else {
                0.0
            };
        }
    }

    // /// Update adaptive surrogate performance metrics
    // fn update_adaptive_performance(
    //     &self,
    //     performance: &mut SurrogatePerformance,
    //     spikes: &Array1<f32>,
    //     voltage: &Array1<f32>,
    //     loss_gradient: &Array1<f32>
    // ) -> super::Result<()> {
    //     // Update performance with current activity
    //     let activity_score = self.compute_activity_score(spikes, voltage);
    //     performance.update_with_gradient(loss_gradient, activity_score)?;
    //
    //     Ok(())
    // }

    /// Get current configuration
    pub fn config(&self) -> &NeuronConfig {
        &self.config
    }

    /// Compute firing rate from spike train
    ///
    /// Returns fraction of neurons that spiked (range: [0, 1])
    pub fn firing_rate(spikes: &Array1<f32>) -> f32 {
        spikes.sum() / spikes.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;
    use crate::domain::eprop::config::NeuronConfig;

    #[test]
    fn test_neuron_state_creation() {
        let config = NeuronConfig::default();
        let state = NeuronState::new(10, false, &config);
        assert_eq!(state.num_neurons(), 10);
        assert!(!state.has_adaptation());
    }

    #[test]
    fn test_neuron_state_with_adaptation() {
        let config = NeuronConfig::default();
        let state = NeuronState::new(10, true, &config);
        assert!(state.has_adaptation());
        assert_eq!(state.adaptation.as_ref().unwrap().len(), 10);
    }

    #[test]
    fn test_neuron_state_reset() {
        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, true, &config);

        // Modify state
        state.voltage.fill(1.0);
        state.spikes.fill(1.0);
        state.adaptation.as_mut().unwrap().fill(1.0);

        // Reset
        state.reset();

        // Check all zeros
        assert!(state.voltage.iter().all(|&x| x == 0.0));
        assert!(state.spikes.iter().all(|&x| x == 0.0));
        assert!(state.adaptation.as_ref().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_lif_dynamics_no_spike() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        let input = Array1::from_elem(5, 0.1); // Weak input
        let mut workspace = NeuronWorkspace::new(5);

        let result = dynamics.update(&mut state, &input.view(), None, &mut workspace);
        assert!(result.is_ok());

        // With weak input, should not spike
        assert!(state.spikes.iter().all(|&x| x == 0.0));

        // Voltage should increase
        assert!(state.voltage[0] > 0.0);
    }

    #[test]
    fn test_lif_dynamics_spike() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        let input = Array1::from_elem(5, 5.0); // Strong input
        let mut workspace = NeuronWorkspace::new(5);

        let result = dynamics.update(&mut state, &input.view(), None, &mut workspace);
        assert!(result.is_ok());

        // With strong input, should spike
        let spike_count: f32 = state.spikes.sum();
        assert!(spike_count > 0.0);
    }

    #[test]
    fn test_alif_adaptation() {
        let config = NeuronConfig::alif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, true, &config);
        let input = Array1::from_elem(5, 5.0); // Strong input to cause spikes
        let mut workspace = NeuronWorkspace::new(5);

        // First update
        let _ = dynamics.update(&mut state, &input.view(), None, &mut workspace);
        let first_spikes = state.spikes.clone();

        // If there were spikes, adaptation should increase
        if first_spikes.sum() > 0.0 {
            let adaptation_1 = state.adaptation.as_ref().unwrap().clone();

            // Second update with same input
            let _ = dynamics.update(&mut state, &input.view(), None, &mut workspace);
            let adaptation_2 = state.adaptation.as_ref().unwrap().clone();

            // Adaptation should have accumulated
            assert!(adaptation_2.sum() >= adaptation_1.sum());
        }
    }

    #[test]
    fn test_surrogate_derivative() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        let mut workspace = NeuronWorkspace::new(5);

        // Input near threshold should give non-zero surrogate derivative
        let input = Array1::from_elem(5, 0.9); // Just below threshold
        let _ = dynamics.update(&mut state, &input.view(), None, &mut workspace);

        // Surrogate derivative should be non-zero near threshold
        let surr_sum: f32 = state.surrogate_deriv.sum();
        assert!(surr_sum > 0.0);
    }

    #[test]
    fn test_spike_reset() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(1, false, &config);
        let mut workspace = NeuronWorkspace::new(1);

        // Strong input to cause spike
        let input = Array1::from_elem(1, 10.0);
        let _ = dynamics.update(&mut state, &input.view(), None, &mut workspace);

        // If spiked, voltage should have been reset
        if state.spikes[0] > 0.5 {
            assert!(state.voltage[0] < 10.0); // Should be reduced by threshold
        }
    }

    #[test]
    fn test_filtered_spikes() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        let mut workspace = NeuronWorkspace::new(5);

        // Generate some spikes
        let input = Array1::from_elem(5, 5.0);
        let _ = dynamics.update(&mut state, &input.view(), None, &mut workspace);

        let spikes_1 = state.spikes.clone();
        let filtered_1 = state.filtered_spikes.clone();

        // Filtered spikes should be similar to actual spikes initially
        if spikes_1.sum() > 0.0 {
            // At least some correlation
            assert!(filtered_1.sum() > 0.0);
        }
    }

    #[test]
    fn test_firing_rate() {
        let spikes = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0, 0.0]);
        let rate = NeuronDynamics::firing_rate(&spikes);
        assert_relative_eq!(rate, 0.4, epsilon = 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let config = NeuronConfig::lif();
        let dynamics = NeuronDynamics::new(config);

        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        let input = Array1::from_elem(10, 1.0); // Wrong size
        let mut workspace = NeuronWorkspace::new(5);

        let result = dynamics.update(&mut state, &input.view(), None, &mut workspace);
        assert!(result.is_err());
    }
}
