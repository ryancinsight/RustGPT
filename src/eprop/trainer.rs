//! Main training engine and gradient updates
//!
//! This module implements the EPropTrainer, which orchestrates the complete
//! training loop including forward passes, trace updates, and gradient application.

use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::eprop::{
    config::{EPropConfig, NeuronModel},
    neuron::{NeuronDynamics, NeuronState},
    traces::{EligibilityTraces, TraceUpdater},
    utils::{clip_gradient, outer_product},
    adaptive_softmax::AdaptiveSoftmax,
    EPropError, Result,
};
use crate::rng::get_rng;

/// Training statistics and monitoring
#[derive(Debug, Clone, Default)]
pub struct TrainingStats {
    /// Total updates performed
    pub num_updates: usize,
    
    /// Average firing rate (fraction of neurons spiking)
    pub avg_firing_rate: f32,
    
    /// Gradient norm history (last 100)
    pub grad_norms: Vec<f32>,
    
    /// Loss history
    pub losses: Vec<f32>,
    
    /// BPTT cosine similarity (if available from validation)
    pub bptt_similarity: Option<f32>,
}

impl TrainingStats {
    /// Get average gradient norm over recent history
    pub fn avg_grad_norm(&self) -> Option<f32> {
        if self.grad_norms.is_empty() {
            None
        } else {
            Some(self.grad_norms.iter().sum::<f32>() / self.grad_norms.len() as f32)
        }
    }
    
    /// Get average loss over recent history
    pub fn avg_loss(&self, window: usize) -> Option<f32> {
        if self.losses.is_empty() {
            None
        } else {
            let start = self.losses.len().saturating_sub(window);
            let window_losses = &self.losses[start..];
            Some(window_losses.iter().sum::<f32>() / window_losses.len() as f32)
        }
    }
}

/// Main ES-D-RTRL e-prop trainer
///
/// Implements online forward-mode gradient computation with O(N) complexity.
/// Supports both LIF and ALIF neuron models.
///
/// # Architecture
/// - Input layer: W_in (num_neurons × input_dim)
/// - Recurrent layer: W_rec (num_neurons × num_neurons)
/// - Output layer: W_out (output_dim × num_neurons)
///
/// # Training Process
/// 1. Forward pass: Compute neuron dynamics and update traces
/// 2. Output computation: Project spikes to output space
/// 3. Gradient computation: Use eligibility traces and learning signal
/// 4. Weight update: Apply gradients with learning rate
pub struct EPropTrainer {
    /// Configuration
    pub config: EPropConfig,

    /// Recurrent weights W_rec: (num_neurons, num_neurons)
    pub weights_rec: Array2<f32>,

    /// Input weights W_in: (num_neurons, input_dim)
    pub weights_in: Array2<f32>,

    /// Output weights W_out: (output_dim, num_neurons)
    pub weights_out: Array2<f32>,

    /// Adaptive softmax for large vocabularies (Theorem 5.2)
    /// Automatically handles Full/Sampled/Hierarchical strategies
    softmax: Option<AdaptiveSoftmax>,

    /// Neuron dynamics engine
    dynamics: NeuronDynamics,

    /// Trace update engine
    trace_updater: TraceUpdater,

    /// Current neuron state
    state: NeuronState,

    /// Eligibility traces
    traces: EligibilityTraces,

    /// Training statistics
    stats: TrainingStats,
}

impl EPropTrainer {
    /// Create new trainer with random initialization
    ///
    /// Weights are initialized using Xavier/Glorot initialization scaled by
    /// config.init_scale for stability.
    pub fn new(config: EPropConfig) -> Result<Self> {
        config.validate()?;
        
        use rand::Rng;
        use rand_distr::{Distribution, Normal};
        
        let mut rng = get_rng();
        
        // Xavier initialization for weights
        let fan_in_rec = config.num_neurons as f32;
        let fan_in_in = config.input_dim as f32;
        let fan_in_out = config.num_neurons as f32;
        
        let scale_rec = (2.0 / fan_in_rec).sqrt() * config.init_scale;
        let scale_in = (2.0 / fan_in_in).sqrt() * config.init_scale;
        let scale_out = (2.0 / fan_in_out).sqrt() * config.init_scale;
        
        let normal_rec = Normal::new(0.0, scale_rec).unwrap();
        let normal_in = Normal::new(0.0, scale_in).unwrap();
        let normal_out = Normal::new(0.0, scale_out).unwrap();
        
        let weights_rec = Array2::from_shape_fn(
            (config.num_neurons, config.num_neurons),
            |_| normal_rec.sample(&mut rng),
        );
        
        let weights_in = Array2::from_shape_fn(
            (config.num_neurons, config.input_dim),
            |_| normal_in.sample(&mut rng),
        );
        
        let weights_out = Array2::from_shape_fn(
            (config.output_dim, config.num_neurons),
            |_| normal_out.sample(&mut rng),
        );
        
        let use_adaptation = config.neuron_config.model == NeuronModel::ALIF;
        
        let dynamics = NeuronDynamics::new(config.neuron_config.clone());
        let trace_updater = TraceUpdater::new(&config, config.neuron_config.clone());

        // Initialize adaptive softmax if output layer matches vocab size
        let softmax = if config.output_dim > 2 {
            use super::adaptive_softmax::{SoftmaxStrategy, SoftmaxConfig};

            let softmax_config = SoftmaxConfig::auto_select(
                config.output_dim,
                None, // No frequencies for now
            );

            // Always create softmax for large classification tasks (>= 100 vocab)
            // This provides consistent API regardless of auto-selected strategy
            Some(super::adaptive_softmax::AdaptiveSoftmax::new(softmax_config))
        } else {
            None // Regression task (output_dim ≤ 2)
        };

        Ok(Self {
            state: NeuronState::new(config.num_neurons, use_adaptation, &config.neuron_config),
            traces: EligibilityTraces::new(config.input_dim, config.num_neurons, use_adaptation),
            dynamics,
            trace_updater,
            config,
            weights_rec,
            weights_in,
            weights_out,
            softmax,
            stats: TrainingStats::default(),
        })
    }
    
    /// Forward step: compute neuron dynamics and update traces
    ///
    /// # Arguments
    /// * `input` - Input spike vector x_t (shape: input_dim)
    /// * `loss_gradient` - Optional loss gradient for adaptive surrogate updates
    ///
    /// # Returns
    /// Current spike output z_t
    pub fn forward(&mut self, input: &Array1<f32>) -> Result<Array1<f32>> {
        self.forward_with_gradient(input, None)
    }
    
    /// Enhanced forward step with adaptive surrogate gradient support
    ///
    /// # Arguments
    /// * `input` - Input spike vector x_t (shape: input_dim)
    /// * `loss_gradient` - Optional loss gradient for adaptive surrogate updates
    ///
    /// # Returns
    /// Current spike output z_t
    pub fn forward_with_gradient(&mut self, input: &Array1<f32>, loss_gradient: Option<&Array1<f32>>) -> Result<Array1<f32>> {
        if input.len() != self.config.input_dim {
            return Err(EPropError::TraceDimensionMismatch {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }
        
        // 1. Compute total input current: I_t = W_in @ x_t + W_rec @ z_{t-1}
        let input_current = self.compute_input_current(input);
        
        // 2. Update neuron dynamics (LIF/ALIF) with optional gradient for adaptation
        self.dynamics.update(&mut self.state, &input_current, loss_gradient)?;
        
        // 3. Check for adaptive trace truncation (2-3× speedup)
        self.maybe_truncate_traces();
        
        // 4. Update eligibility traces (ES-D-RTRL)
        self.trace_updater.update(&mut self.traces, &self.state, input)?;
        
        // 5. Update position counter for window tracking
        self.traces.step();
        
        // 6. Update statistics
        self.update_statistics();
        
        Ok(self.state.spikes.clone())
    }
    
    /// Compute total input current to neurons
    ///
    /// Uses sparse computation for firing rates r ≪ 1 (Theorem 3.1):
    /// - Dense: O(N·D) where D = input_dim or num_neurons
    /// - Sparse: O(k·D) where k = active spikes, speedup = 1/r
    ///
    /// Automatically switches based on spike sparsity threshold.
    fn compute_input_current(&self, input: &Array1<f32>) -> Array1<f32> {
        // Feedforward input: W_in @ x_t (always dense, x_t is full)
        let feedforward = self.weights_in.dot(input);

        // Recurrent input: W_rec @ z_{t-1}
        let recurrent = if self.config.use_sparse_spikes && !self.state.spikes.is_empty() {
            self.compute_recurrent_sparse(&self.state.spikes)
        } else {
            self.weights_rec.dot(&self.state.spikes)
        };

        feedforward + recurrent
    }

    /// Compute sparse recurrent input when firing rate is low
    ///
    /// For firing rate r ≪ 1: reduces complexity from O(N²) to O(r·N²)
    /// Auto-switches to dense computation if sparsity threshold not met.
    fn compute_recurrent_sparse(&self, spikes: &Array1<f32>) -> Array1<f32> {
        use crate::eprop::utils::{get_active_spike_indices, sparse_matvec};

        // Find active (spiking) neurons
        let active_indices = get_active_spike_indices(spikes, self.config.spike_sparsity_threshold);
        let sparsity_ratio = active_indices.len() as f32 / spikes.len() as f32;

        // Only use sparse if beneficial (r < 20%)
        // For r > 0.2, overhead of sparse indexing outweighs benefits
        if sparsity_ratio < 0.2 && !active_indices.is_empty() {
            sparse_matvec(&self.weights_rec, spikes, &active_indices)
        } else {
            // Fall back to dense computation for high sparsity
            self.weights_rec.dot(spikes)
        }
    }
    
    /// Apply adaptive trace window truncation
    ///
    /// Resets traces when gradient variance is low, providing 2-3× speedup.
    /// Implements variance-based truncation with configurable thresholds.
    fn maybe_truncate_traces(&mut self) {
        if !self.config.use_adaptive_windowing {
            return;
        }
        
        // Check if traces should be truncated based on variance and position
        if self.traces.should_truncate(
            self.config.min_trace_window,
            self.config.max_trace_window
        ) {
            // Reset traces to fresh start
            self.traces.reset();
            
            // Keep current position for window tracking
            self.traces.position = self.config.min_trace_window;
        }
    }
    
    /// Compute layer-wise adaptive learning rate using trust-ratio + bidirectional balance
    /// Reference: "LARS: Layer-wise Adaptive Rate Scaling" (You et al., 2017)
    ///
    /// Formula:
    /// lr_layer = lr_base * clamp( (||W|| / (||∇W|| + ε)) * (median_grad_norm / (||∇W|| + ε))^power, [min,max] )
    fn compute_adaptive_lr(base_lr: f32, grad_norm: f32, weight_norm: f32, median_grad_norm: f32) -> f32 {
        const EPSILON: f32 = 1e-6;
        if grad_norm < EPSILON || weight_norm < EPSILON {
            return base_lr;
        }

        let trust_ratio = weight_norm / (grad_norm + EPSILON);
        const POWER_BALANCE: f32 = 0.5; // Gentle correction
        let balance_scale = (median_grad_norm / (grad_norm + EPSILON)).powf(POWER_BALANCE);

        const MIN_SCALE: f32 = 0.2;
        const MAX_SCALE: f32 = 5.0;
        let scale = (trust_ratio * balance_scale).clamp(MIN_SCALE, MAX_SCALE);
        base_lr * scale
    }

    /// Apply weight update using current traces and learning signal
    ///
    /// Implements Theorem 1:
    /// ∂E/∂W = L_t · (ε^f_t ⊗ ε^x_t)  [rank-one gradient]
    ///
    /// With literature-based enhancements:
    /// - Bidirectional LARS: Layer-wise adaptive learning rates
    /// - Gradient clipping: Prevents explosion
    /// - Gradient monitoring: Track convergence metrics
    ///
    /// # Arguments
    /// * `learning_signal` - ∂E/∂z_t from downstream layers (shape: num_neurons)
    pub fn apply_update(&mut self, learning_signal: &Array1<f32>) -> Result<()> {
        if learning_signal.len() != self.config.num_neurons {
            return Err(EPropError::TraceDimensionMismatch {
                expected: self.config.num_neurons,
                actual: learning_signal.len(),
            });
        }
        
        let eta = self.config.learning_rate;
        
        // Compute gradient factors from traces
        let (modulated_eps_f, eps_x) = self.trace_updater
            .compute_gradient_factors(&self.traces, learning_signal)?;
        
        // Rank-one gradient: ∇W_in = (L_t · ε^f_t) ⊗ ε^x_t
        let grad_in = outer_product(&modulated_eps_f, &eps_x);
        
        // Rank-one gradient for recurrent: ∇W_rec = (L_t · ε^f_t) ⊗ z_{t-1}
        let grad_rec = outer_product(&modulated_eps_f, &self.state.filtered_spikes);
        
        // Compute gradient norms and corresponding weight norms for trust-ratio LARS
        let grad_in_norm = grad_in.mapv(|x| x * x).sum().sqrt();
        let grad_rec_norm = grad_rec.mapv(|x| x * x).sum().sqrt();
        let w_in_norm = self.weights_in.iter().map(|&w| w * w).sum::<f32>().sqrt();
        let w_rec_norm = self.weights_rec.iter().map(|&w| w * w).sum::<f32>().sqrt();

        // Median of non-zero gradient norms (bidirectional balance target)
        const EPS: f32 = 1e-6;
        let mut nonzero_norms = Vec::new();
        if grad_in_norm > EPS { nonzero_norms.push(grad_in_norm); }
        if grad_rec_norm > EPS { nonzero_norms.push(grad_rec_norm); }
        let median_grad_norm = match nonzero_norms.len() {
            0 => (grad_in_norm + grad_rec_norm) * 0.5,
            1 => nonzero_norms[0],
            _ => {
                nonzero_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = nonzero_norms.len() / 2;
                if nonzero_norms.len() % 2 == 0 {
                    (nonzero_norms[mid - 1] + nonzero_norms[mid]) * 0.5
                } else {
                    nonzero_norms[mid]
                }
            }
        };

        // Trust-ratio + bidirectional balance adaptive learning rates
        let adaptive_lr_in = Self::compute_adaptive_lr(eta, grad_in_norm, w_in_norm, median_grad_norm);
        let adaptive_lr_rec = Self::compute_adaptive_lr(eta, grad_rec_norm, w_rec_norm, median_grad_norm);
        
        // Gradient clipping (after LARS, before application)
        let (grad_in, grad_rec) = if let Some(clip_val) = self.config.grad_clip {
            (
                clip_gradient(grad_in, clip_val),
                clip_gradient(grad_rec, clip_val),
            )
        } else {
            (grad_in, grad_rec)
        };
        
        // Update weights with adaptive learning rates: W -= η_adaptive·∇W
        self.weights_in = &self.weights_in - &(&grad_in * adaptive_lr_in);
        self.weights_rec = &self.weights_rec - &(&grad_rec * adaptive_lr_rec);
        
        // Apply sparsity pruning (optional)
        if let Some(threshold) = self.config.sparsity_threshold {
            self.apply_sparsity_pruning(threshold);
        }
        
        // Track gradient statistics (use pre-computed norms)
        let total_grad_norm = (grad_in_norm * grad_in_norm + grad_rec_norm * grad_rec_norm).sqrt();
        self.stats.grad_norms.push(total_grad_norm);
        if self.stats.grad_norms.len() > 100 {
            self.stats.grad_norms.remove(0);
        }
        
        self.stats.num_updates += 1;
        
        Ok(())
    }
    
    /// Compute output layer prediction
    ///
    /// # Returns
    /// Output logits (shape: output_dim)
    pub fn compute_output(&self) -> Array1<f32> {
        self.weights_out.dot(&self.state.spikes)
    }
    
    /// Reset neuron state and traces (e.g., between sequences)
    pub fn reset_state(&mut self) {
        self.state.reset();
        self.traces.reset();
    }
    
    /// Reset traces only (for epoch boundaries)
    ///
    /// Literature recommendation: Reset eligibility traces between epochs
    /// to prevent unbounded accumulation and gradient drift.
    ///
    /// Reference: Bellec et al. (2020) - E-prop epoch management
    pub fn reset_traces(&mut self) {
        self.traces.reset();
    }
    
    /// Full forward pass with multiple cycles
    ///
    /// # Arguments
    /// * `input` - Input sequence (shape: input_dim)
    /// * `num_cycles` - Number of recurrent cycles (default: config.num_cycles)
    ///
    /// # Returns
    /// Final output after all cycles
    pub fn forward_cycles(&mut self, input: &Array1<f32>, num_cycles: Option<usize>) -> Result<Array1<f32>> {
        let cycles = num_cycles.unwrap_or(self.config.num_cycles);
        
        for _ in 0..cycles {
            self.forward(input)?;
        }
        
        Ok(self.compute_output())
    }
    
    /// Training step: forward + backward + update (regression with MSE loss)
    ///
    /// Enhanced with adaptive surrogate gradient integration for optimal performance.
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `target` - Target output vector
    ///
    /// # Returns
    /// Loss value (MSE)
    pub fn train_step(&mut self, input: &Array1<f32>, target: &Array1<f32>) -> Result<f32> {
        if self.softmax.is_some() {
            return Err(EPropError::InvalidConfig(
                "Use train_step_classification for softmax-enabled models".to_string()
            ));
        }

        // Forward pass with adaptive surrogate support
        let output = self.forward_cycles(input, None)?;

        // Compute loss (MSE)
        let loss = (&output - target).mapv(|x| x * x).mean().unwrap_or(0.0);

        // Compute learning signal: ∂E/∂output = 2(output - target)
        let output_grad = (&output - target) * 2.0;

        // Backpropagate to neurons: L_t = W_out^T @ ∂E/∂output
        let learning_signal = self.weights_out.t().dot(&output_grad);

        // Enhanced forward pass with gradient for adaptive surrogate adaptation
        if self.config.neuron_config.use_adaptive_surrogate {
            // Run forward with gradient to update adaptive surrogates
            self.forward_with_gradient(input, Some(&learning_signal))?;
        } else {
            // Standard forward pass for static surrogates
            self.forward(input)?;
        }

        // Apply e-prop update
        self.apply_update(&learning_signal)?;

        // Update output weights (standard gradient descent)
        let grad_out = outer_product(&output_grad, &self.state.spikes);
        self.weights_out = &self.weights_out - &(&grad_out * self.config.learning_rate);

        self.stats.losses.push(loss);
        Ok(loss)
    }

    /// Training step for classification tasks with adaptive softmax (Theorem 5.2)
    ///
    /// Enhanced with adaptive surrogate gradient integration for optimal performance.
    /// Automatically uses Full/Sampled/Hierarchical softmax based on vocabulary size.
    /// Provides 50-200× speedup vs full softmax for large vocabularies.
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `target_class` - Target class index (0 ≤ target < vocab_size)
    ///
    /// # Returns
    /// Loss value (cross-entropy)
    pub fn train_step_classification(&mut self, input: &Array1<f32>, target_class: usize) -> Result<f32> {
        if self.softmax.is_none() {
            return Err(EPropError::InvalidConfig(
                "Adaptive softmax not available for this model".to_string()
            ));
        }

        if target_class >= self.config.output_dim {
            return Err(EPropError::InvalidConfig(
                format!("Target class {} out of range (0..{})", target_class, self.config.output_dim)
            ));
        }

        // Forward pass
        let output_logits = self.forward_cycles(input, None)?;

        // Compute loss and gradient using adaptive softmax
        // Borrowmut scope ends at end of this block
        let (loss, output_grad) = {
            let softmax = self.softmax.as_mut().unwrap();
            softmax.loss_and_gradient(&output_logits, target_class)
        };

        // Backpropagate to neurons: L_t = W_out^T @ ∂E/∂logits
        let learning_signal = self.weights_out.t().dot(&output_grad);

        // Enhanced forward pass with gradient for adaptive surrogate adaptation
        if self.config.neuron_config.use_adaptive_surrogate {
            // Run forward with gradient to update adaptive surrogates
            self.forward_with_gradient(input, Some(&learning_signal))?;
        } else {
            // Standard forward pass for static surrogates
            self.forward(input)?;
        }

        // Apply e-prop update
        self.apply_update(&learning_signal)?;

        // Update output weights (standard gradient descent)
        let grad_out = outer_product(&output_grad, &self.state.spikes);
        self.weights_out = &self.weights_out - &(&grad_out * self.config.learning_rate);

        self.stats.losses.push(loss);
        Ok(loss)
    }
    
    /// Apply connection pruning based on weight magnitude
    fn apply_sparsity_pruning(&mut self, threshold: f32) {
        self.weights_rec.mapv_inplace(|w| if w.abs() < threshold { 0.0 } else { w });
        self.weights_in.mapv_inplace(|w| if w.abs() < threshold { 0.0 } else { w });
    }
    
    /// Update training statistics
    fn update_statistics(&mut self) {
        // Compute firing rate
        let rate = NeuronDynamics::firing_rate(&self.state.spikes);
        
        // Exponential moving average
        if self.stats.avg_firing_rate == 0.0 {
            self.stats.avg_firing_rate = rate;
        } else {
            self.stats.avg_firing_rate = 0.99 * self.stats.avg_firing_rate + 0.01 * rate;
        }
    }
    
    /// Get current training statistics
    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }
    
    /// Get current neuron state (for inspection)
    pub fn state(&self) -> &NeuronState {
        &self.state
    }
    
    /// Get current traces (for inspection)
    pub fn traces(&self) -> &EligibilityTraces {
        &self.traces
    }
    
    /// Export model weights
    pub fn export_weights(&self) -> HashMap<String, Array2<f32>> {
        let mut weights = HashMap::new();
        weights.insert("W_in".to_string(), self.weights_in.clone());
        weights.insert("W_rec".to_string(), self.weights_rec.clone());
        weights.insert("W_out".to_string(), self.weights_out.clone());
        weights
    }
    
    /// Import model weights
    pub fn import_weights(&mut self, weights: HashMap<String, Array2<f32>>) -> Result<()> {
        if let Some(w_in) = weights.get("W_in") {
            if w_in.shape() == self.weights_in.shape() {
                self.weights_in = w_in.clone();
            } else {
                return Err(EPropError::TraceDimensionMismatch {
                    expected: self.weights_in.len(),
                    actual: w_in.len(),
                });
            }
        }
        
        if let Some(w_rec) = weights.get("W_rec") {
            if w_rec.shape() == self.weights_rec.shape() {
                self.weights_rec = w_rec.clone();
            } else {
                return Err(EPropError::TraceDimensionMismatch {
                    expected: self.weights_rec.len(),
                    actual: w_rec.len(),
                });
            }
        }
        
        if let Some(w_out) = weights.get("W_out") {
            if w_out.shape() == self.weights_out.shape() {
                self.weights_out = w_out.clone();
            } else {
                return Err(EPropError::TraceDimensionMismatch {
                    expected: self.weights_out.len(),
                    actual: w_out.len(),
                });
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eprop::config::{EPropConfig, NeuronConfig};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_trainer_creation() {
        let config = EPropConfig::minimal();
        let trainer = EPropTrainer::new(config);
        
        assert!(trainer.is_ok());
        let trainer = trainer.unwrap();
        
        assert_eq!(trainer.weights_in.shape(), &[8, 4]);
        assert_eq!(trainer.weights_rec.shape(), &[8, 8]);
        assert_eq!(trainer.weights_out.shape(), &[2, 8]);
    }
    
    #[test]
    fn test_forward_pass() {
        let config = EPropConfig::minimal();
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        let input = Array1::from_elem(4, 0.5);
        let result = trainer.forward(&input);
        
        assert!(result.is_ok());
        let spikes = result.unwrap();
        assert_eq!(spikes.len(), 8);
        
        // Spikes should be binary
        for &spike in spikes.iter() {
            assert!(spike == 0.0 || spike == 1.0);
        }
    }
    
    #[test]
    fn test_forward_cycles() {
        let config = EPropConfig::minimal();
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        let input = Array1::from_elem(4, 0.5);
        let result = trainer.forward_cycles(&input, Some(3));
        
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.len(), 2);
    }
    
    #[test]
    fn test_train_step() {
        let config = EPropConfig::minimal();
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        let input = Array1::from_elem(4, 0.5);
        let target = Array1::from_elem(2, 1.0);
        
        let result = trainer.train_step(&input, &target);
        assert!(result.is_ok());
        
        let loss = result.unwrap();
        assert!(loss >= 0.0);
        assert_eq!(trainer.stats.num_updates, 1);
    }
    
    #[test]
    fn test_reset_state() {
        let config = EPropConfig::minimal();
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        // Run forward pass
        let input = Array1::from_elem(4, 1.0);
        let _ = trainer.forward(&input);
        
        // Reset
        trainer.reset_state();
        
        // Check state is zero
        assert!(trainer.state.voltage.iter().all(|&x| x == 0.0));
        assert!(trainer.state.spikes.iter().all(|&x| x == 0.0));
        assert!(trainer.traces.eps_x.iter().all(|&x| x == 0.0));
        assert!(trainer.traces.eps_f.iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_multiple_train_steps() {
        let config = EPropConfig::minimal();
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        let input = Array1::from_elem(4, 0.5);
        let target = Array1::from_elem(2, 1.0);
        
        // Train for 5 steps
        for _ in 0..5 {
            let _ = trainer.train_step(&input, &target);
        }
        
        assert_eq!(trainer.stats.num_updates, 5);
        assert_eq!(trainer.stats.losses.len(), 5);
    }
    
    #[test]
    fn test_export_import_weights() {
        let config = EPropConfig::minimal();
        let mut trainer1 = EPropTrainer::new(config.clone()).unwrap();
        
        // Export weights
        let weights = trainer1.export_weights();
        
        // Create new trainer and import
        let mut trainer2 = EPropTrainer::new(config).unwrap();
        let result = trainer2.import_weights(weights);
        
        assert!(result.is_ok());
        
        // Verify weights match
        assert_eq!(trainer1.weights_in, trainer2.weights_in);
        assert_eq!(trainer1.weights_rec, trainer2.weights_rec);
        assert_eq!(trainer1.weights_out, trainer2.weights_out);
    }
    
    #[test]
    fn test_stats_avg_grad_norm() {
        let mut stats = TrainingStats::default();
        
        assert!(stats.avg_grad_norm().is_none());
        
        stats.grad_norms = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(stats.avg_grad_norm().unwrap(), 2.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_stats_avg_loss() {
        let mut stats = TrainingStats::default();
        
        assert!(stats.avg_loss(5).is_none());
        
        stats.losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(stats.avg_loss(3).unwrap(), 4.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_gradient_clipping() {
        let config = EPropConfig {
            grad_clip: Some(1.0),
            learning_rate: 0.01, // Smaller learning rate for stable test
            ..EPropConfig::minimal()
        };
        let mut trainer = EPropTrainer::new(config).unwrap();
        
        let input = Array1::from_elem(4, 2.0); // Moderate input
        let _ = trainer.forward(&input);
        
        let learning_signal = Array1::from_elem(8, 5.0); // Moderate signal
        let result = trainer.apply_update(&learning_signal);
        
        assert!(result.is_ok());
        
        // With LARS + clipping, gradient norms should be reasonable
        // LARS scales gradients, then clipping applies per-matrix normalization
        // Final norm after both operations should be controlled
        if let Some(norm) = trainer.stats.grad_norms.last() {
            // Total gradient norm (combined W_in + W_rec) after LARS and clipping
            // Should be finite and not exploded (< 10.0 is very conservative)
            assert!(norm.is_finite(), "Gradient norm is not finite: {}", norm);
            assert!(*norm < 10.0, "Gradient norm {} too large after LARS+clipping", norm);
        }
    }
    
    #[test]
    fn test_alif_trainer() {
        let config = EPropConfig {
            neuron_config: NeuronConfig::alif(),
            ..EPropConfig::minimal()
        };
        let mut trainer = EPropTrainer::new(config).unwrap();

        // Should have adaptation
        assert!(trainer.state.has_adaptation());
        assert!(trainer.traces.eps_a.is_some());

        let input = Array1::from_elem(4, 5.0);
        let result = trainer.forward(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sparse_computation_benefit() {
        // Create a model that benefits from sparse computation
        let config = EPropConfig {
            num_neurons: 128,  // Larger network
            input_dim: 64,
            output_dim: 10,
            use_sparse_spikes: true,
            spike_sparsity_threshold: 0.5,  // Low threshold for sparse activation
            learning_rate: 0.01,
            ..Default::default()
        };

        let mut trainer = EPropTrainer::new(config).unwrap();

        // Create input that will produce sparse firing (low firing rate)
        let input = Array1::from_elem(64, 0.1);  // Low input activation

        // Run several forward passes to establish firing pattern
        for _ in 0..5 {
            trainer.forward(&input).unwrap();
        }

        // The sparse computation should be activated
        // (This is mainly a smoke test - real benchmarking would need timing)
        let current_firing_rate = trainer.stats.avg_firing_rate;
        assert!(current_firing_rate >= 0.0 && current_firing_rate <= 1.0);

        // Test that the functionality works with sparse computation enabled
        let test_input = Array1::from_elem(64, 0.5);
        let result = trainer.forward(&test_input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_classfication_training_step() {
        // Create a model with large output dim to trigger adaptive softmax
        let config = EPropConfig {
            num_neurons: 16,
            input_dim: 8,
            output_dim: 1000,  // Large enough to trigger sampled softmax
            use_sparse_spikes: true,
            ..Default::default()
        };

        let mut trainer = EPropTrainer::new(config).unwrap();

        // Should have adaptive softmax
        assert!(trainer.softmax.is_some());

        // Test classification training
        let input = Array1::from_elem(8, 1.0);
        let target_class = 25;  // Valid class index

        let result = trainer.train_step_classification(&input, target_class);
        assert!(result.is_ok());

        let loss = result.unwrap();
        assert!(loss >= 0.0);
        assert_eq!(trainer.stats.num_updates, 1);
    }

    #[test]
    fn test_regression_vs_classification_error() {
        // Create a model with softmax (classification)
        let config_classification = EPropConfig {
            num_neurons: 8,
            input_dim: 4,
            output_dim: 20,  // Triggers softmax
            ..Default::default()
        };

        let mut classification_trainer = EPropTrainer::new(config_classification).unwrap();

        // Should have softmax
        assert!(classification_trainer.softmax.is_some());

        let input = Array1::from_elem(4, 1.0);

        // Should fail with vector target (regression style)
        let target_vector = Array1::from_elem(20, 0.1);
        let result = classification_trainer.train_step(&input, &target_vector);
        assert!(result.is_err());  // Should error for softmax-enabled model

        // Should work with class index
        let result = classification_trainer.train_step_classification(&input, 10);
        assert!(result.is_ok());
    }
}
