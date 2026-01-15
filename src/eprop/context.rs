//! Thread-local context management for persistent eligibility traces
//!
//! This module provides model-agnostic trace persistence across training sequences
//! using thread-local storage. Traces are automatically maintained across batches
//! within an epoch, enabling true temporal credit assignment across sequence boundaries.
//!
//! # Key Benefits
//! - **Persistent Memory**: Traces survive across sequences within epoch
//! - **Model Agnostic**: No coupling to specific layer implementations
//! - **Zero Overhead**: Thread-local with lazy initialization
//! - **Epoch Boundaries**: Clean reset between epochs
//!
//! # Usage
//!
//! ```rust
//! use eprop::context::EpropContext;
//!
//! // Initialize at epoch start
//! let layer_dims = vec![(128, 64), (64, 10)];
//! EpropContext::init_for_layers(layer_dims);
//!
//! // Training loop (traces persist across sequences)
//! for sequence in epoch {
//!     EpropContext::with_traces(|traces| {
//!         // Update traces with current sequence
//!         // Gradients benefit from accumulated temporal information
//!     });
//! }
//!
//! // Reset between epochs
//! EpropContext::reset();
//! ```

use std::cell::RefCell;
use super::{EligibilityTraces, EPropError, Result};

/// Thread-local storage for eligibility traces
///
/// Maintains one set of traces per layer, persisting across sequences
/// within a single epoch. Thread-local ensures no cross-thread interference.
thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static EPROP_TRACES: RefCell<Option<Vec<EligibilityTraces>>> = const { RefCell::new(None) };
}

/// Thread-local context for e-prop trace management
///
/// Provides a model-agnostic interface for maintaining persistent eligibility traces
/// across training sequences. Traces are stored in thread-local storage and survive
/// across batch boundaries within an epoch.
pub struct EpropContext;

impl EpropContext {
    pub fn init_for_layers_with_adaptation(layer_dims: Vec<(usize, usize, bool)>) {
        let traces: Vec<EligibilityTraces> = layer_dims
            .into_iter()
            .map(|(output_dim, input_dim, use_adaptation)| {
                EligibilityTraces::new(input_dim, output_dim, use_adaptation)
            })
            .collect();

        EPROP_TRACES.with(|cell| {
            *cell.borrow_mut() = Some(traces);
        });
    }

    /// Initialize context with traces for multiple layers
    ///
    /// Creates one `EligibilityTraces` per layer, dimensioned according to
    /// the provided (output_dim, input_dim) pairs.
    ///
    /// # Arguments
    /// * `layer_dims` - Vector of (output_dim, input_dim) for each layer
    ///
    /// # Example
    /// ```
    /// // Two layers: 128→64 and 64→10
    /// let dims = vec![(128, 64), (64, 10)];
    /// EpropContext::init_for_layers(dims);
    /// ```
    pub fn init_for_layers(layer_dims: Vec<(usize, usize)>) {
        Self::init_for_layers_with_adaptation(
            layer_dims
                .into_iter()
                .map(|(output_dim, input_dim)| (output_dim, input_dim, false))
                .collect(),
        );
    }

    /// Access traces with a closure (read-write)
    ///
    /// Provides mutable access to all layer traces. The closure receives
    /// `&mut Vec<EligibilityTraces>` for updating traces during training.
    ///
    /// # Arguments
    /// * `f` - Closure that operates on traces
    ///
    /// # Returns
    /// Result of closure execution, or error if context not initialized
    ///
    /// # Example
    /// ```
    /// EpropContext::with_traces(|traces| {
    ///     for (layer_idx, trace) in traces.iter_mut().enumerate() {
    ///         // Update traces for each layer
    ///     }
    /// });
    /// ```
    pub fn with_traces<F, R>(f: F) -> Result<R>
    where
        F: FnOnce(&mut Vec<EligibilityTraces>) -> R,
    {
        EPROP_TRACES.with(|cell| {
            let mut traces_opt = cell.borrow_mut();
            match traces_opt.as_mut() {
                Some(traces) => Ok(f(traces)),
                None => Err(EPropError::InvalidConfig(
                    "EpropContext not initialized. Call init_for_layers() first.".to_string()
                )),
            }
        })
    }

    /// Check if context is initialized
    pub fn is_initialized() -> bool {
        EPROP_TRACES.with(|cell| cell.borrow().is_some())
    }

    /// Get number of layers (trace sets)
    pub fn num_layers() -> Result<usize> {
        EPROP_TRACES.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|traces| traces.len())
                .ok_or_else(|| EPropError::InvalidConfig(
                    "EpropContext not initialized".to_string()
                ))
        })
    }

    /// Reset all traces to zero (keeps allocation)
    ///
    /// Call this between epochs to clear temporal memory while maintaining
    /// the trace structure. This is more efficient than `clear()` which
    /// deallocates everything.
    ///
    /// # Example
    /// ```
    /// // End of epoch
    /// EpropContext::reset();
    /// // Start new epoch (traces exist but are zeroed)
    /// ```
    pub fn reset() {
        EPROP_TRACES.with(|cell| {
            if let Some(ref mut traces) = *cell.borrow_mut() {
                for trace in traces.iter_mut() {
                    trace.reset();
                }
            }
        });
    }

    /// Clear context (deallocate all traces)
    ///
    /// Use this when completely shutting down e-prop training or switching
    /// to a different model architecture. Unlike `reset()`, this releases memory.
    pub fn clear() {
        EPROP_TRACES.with(|cell| {
            *cell.borrow_mut() = None;
        });
    }

    /// Update traces for a specific layer
    ///
    /// Helper method for single-layer trace updates. For multi-layer models,
    /// prefer `with_traces()` with explicit iteration.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of layer to update
    /// * `f` - Closure that updates the trace
    pub fn update_layer<F>(layer_idx: usize, f: F) -> Result<()>
    where
        F: FnOnce(&mut EligibilityTraces),
    {
        Self::with_traces(|traces| {
            if layer_idx < traces.len() {
                f(&mut traces[layer_idx]);
                Ok(())
            } else {
                Err(EPropError::TraceDimensionMismatch {
                    expected: traces.len(),
                    actual: layer_idx,
                })
            }
        })?
    }

    /// Compute gradients for a specific layer
    ///
    /// Helper method that extracts gradient factors (for rank-one outer product)
    /// from a layer's traces given a learning signal.
    ///
    /// # Arguments
    /// * `layer_idx` - Index of layer
    /// * `learning_signal` - Gradient signal from downstream
    ///
    /// # Returns
    /// Tuple of (modulated postsynaptic trace, presynaptic trace) ready for
    /// outer product: `∇W ≈ modulated_eps_f ⊗ eps_x`
    pub fn compute_layer_gradients(
        layer_idx: usize,
        learning_signal: &ndarray::Array1<f32>,
    ) -> Result<(ndarray::Array1<f32>, ndarray::Array1<f32>)> {
        Self::with_traces(|traces| {
            if layer_idx >= traces.len() {
                return Err(EPropError::TraceDimensionMismatch {
                    expected: traces.len(),
                    actual: layer_idx,
                });
            }

            let trace = &traces[layer_idx];

            if learning_signal.len() != trace.eps_f.len() {
                return Err(EPropError::TraceDimensionMismatch {
                    expected: trace.eps_f.len(),
                    actual: learning_signal.len(),
                });
            }

            // Modulate postsynaptic trace: L_t · ε^f_t
            let modulated_eps_f = learning_signal * &trace.eps_f;

            // Return both factors for outer product
            Ok((modulated_eps_f, trace.eps_x.clone()))
        })?
    }
}

/// Configuration presets for e-prop context initialization
///
/// These presets configure the exponential smoothing factor (α) which controls
/// the effective temporal horizon of eligibility traces.
#[derive(Debug, Clone, Copy)]
pub struct ContextPreset {
    /// Exponential smoothing factor α ∈ (0, 1)
    ///
    /// Larger values = longer memory, slower decay
    /// Effective horizon ≈ 1/(1-α) timesteps
    pub alpha: f32,

    /// Human-readable description
    pub description: &'static str,
}

impl ContextPreset {
    /// Default preset: α=0.9 (~10 timestep horizon)
    ///
    /// Balanced for sequences of 20-100 timesteps. Good for:
    /// - Speech recognition
    /// - Short-term sequence prediction
    /// - Online learning with moderate temporal dependencies
    pub const DEFAULT: Self = Self {
        alpha: 0.9,
        description: "Default: balanced memory (α=0.9, ~10 step horizon)",
    };

    /// Long-term memory: α=0.95 (~20 timestep horizon)
    ///
    /// Extended temporal credit assignment. Good for:
    /// - Long-sequence tasks (100-500 timesteps)
    /// - Reinforcement learning with delayed rewards
    /// - Complex temporal dependencies
    pub const LONG_MEMORY: Self = Self {
        alpha: 0.95,
        description: "Long memory: extended horizon (α=0.95, ~20 step horizon)",
    };

    /// Short-term memory: α=0.85 (~6.7 timestep horizon)
    ///
    /// Faster decay, more reactive. Good for:
    /// - Real-time control
    /// - Short sequences (5-30 timesteps)
    /// - Tasks requiring quick adaptation
    pub const SHORT_MEMORY: Self = Self {
        alpha: 0.85,
        description: "Short memory: quick decay (α=0.85, ~6.7 step horizon)",
    };

    /// Calculate effective temporal horizon
    ///
    /// Returns the number of timesteps at which traces decay to ~37% (1/e)
    /// of their original magnitude.
    pub fn effective_horizon(&self) -> f32 {
        1.0 / (1.0 - self.alpha)
    }
}

/// Simple configuration wrapper for thread-local e-prop
///
/// This is a lightweight alternative to the full `EPropConfig` for cases
/// where you just want to enable persistent traces without full e-prop training.
#[derive(Debug, Clone, Copy)]
pub struct ContextConfig {
    /// Enable thread-local trace persistence
    pub enabled: bool,

    /// Exponential smoothing factor (from preset or custom)
    pub alpha: f32,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alpha: ContextPreset::DEFAULT.alpha,
        }
    }
}

impl ContextConfig {
    /// Create from a preset
    pub fn from_preset(preset: ContextPreset, enabled: bool) -> Self {
        Self {
            enabled,
            alpha: preset.alpha,
        }
    }

    /// Calculate effective temporal horizon
    pub fn effective_horizon(&self) -> f32 {
        1.0 / (1.0 - self.alpha)
    }

    /// Compute optimal alpha for given sequence length (adaptive trace smoothing)
    ///
    /// Implementation of adaptive alpha smoothing based on sequence length.
    /// Formula: α_optimal(T) = 1 - 4/max(T, 20) ∈ [0.85, 0.98]
    ///
    /// This dynamically adjusts trace memory horizon based on task requirements:
    /// - Short sequences (T < 50): α = 0.85-0.90 → fast adaptation, short memory
    /// - Medium sequences (50-200): α = 0.90-0.95 → balanced adaptation
    /// - Long sequences (T > 200): α = 0.95-0.98 → long credit assignment
    ///
    /// # Arguments
    /// * `sequence_length` - Expected sequence length for the task
    ///
    /// # Returns
    /// Optimal alpha value clamped to [0.85, 0.98]
    pub fn adaptive_alpha(sequence_length: usize) -> f32 {
        // Mathematical foundation: Keep effective horizon at ~25% of sequence length
        // T_eff = 1/(1-α) = 0.25·T
        // α = 1 - 4/T  (derived by algebra)
        let alpha = 1.0 - 4.0 / sequence_length.max(20) as f32;
        alpha.clamp(0.85, 0.98)  // Safe operating range from literature
    }

    /// Create configuration with adaptive alpha for sequence length
    pub fn with_adaptive_alpha(sequence_length: usize) -> Self {
        Self {
            enabled: true,  // Enable when using adaptive alpha
            alpha: Self::adaptive_alpha(sequence_length),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_initialization() {
        let layer_dims = vec![(10, 5), (5, 3)];
        EpropContext::init_for_layers(layer_dims);

        assert!(EpropContext::is_initialized());
        assert_eq!(EpropContext::num_layers().unwrap(), 2);
    }

    #[test]
    fn test_context_not_initialized() {
        EpropContext::clear();
        assert!(!EpropContext::is_initialized());

        let result = EpropContext::with_traces(|_| ());
        assert!(result.is_err());
    }

    #[test]
    fn test_trace_access() {
        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        let result = EpropContext::with_traces(|traces| {
            assert_eq!(traces.len(), 1);
            assert_eq!(traces[0].eps_x.len(), 5);
            assert_eq!(traces[0].eps_f.len(), 10);
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_trace_reset() {
        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        // Modify traces
        EpropContext::with_traces(|traces| {
            traces[0].eps_x.fill(1.0);
            traces[0].eps_f.fill(2.0);
        }).unwrap();

        // Reset
        EpropContext::reset();

        // Check zeroed
        EpropContext::with_traces(|traces| {
            assert!(traces[0].eps_x.iter().all(|&x| x == 0.0));
            assert!(traces[0].eps_f.iter().all(|&x| x == 0.0));
        }).unwrap();

        // Still initialized
        assert!(EpropContext::is_initialized());
    }

    #[test]
    fn test_clear() {
        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        assert!(EpropContext::is_initialized());

        EpropContext::clear();

        assert!(!EpropContext::is_initialized());
    }

    #[test]
    fn test_update_layer() {
        let layer_dims = vec![(10, 5), (5, 3)];
        EpropContext::init_for_layers(layer_dims);

        let result = EpropContext::update_layer(0, |trace| {
            trace.eps_x.fill(1.0);
        });

        assert!(result.is_ok());

        EpropContext::with_traces(|traces| {
            assert!(traces[0].eps_x.iter().all(|&x| x == 1.0));
            assert!(traces[1].eps_x.iter().all(|&x| x == 0.0)); // Unchanged
        }).unwrap();
    }

    #[test]
    fn test_update_layer_out_of_bounds() {
        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        let result = EpropContext::update_layer(5, |_| {});
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_layer_gradients() {
        use ndarray::Array1;

        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        // Set up traces
        EpropContext::with_traces(|traces| {
            traces[0].eps_x.fill(0.5);
            traces[0].eps_f.fill(0.2);
        }).unwrap();

        let learning_signal = Array1::from_elem(10, 1.0);
        let result = EpropContext::compute_layer_gradients(0, &learning_signal);

        assert!(result.is_ok());
        let (mod_f, pre_x) = result.unwrap();
        assert_eq!(mod_f.len(), 10);
        assert_eq!(pre_x.len(), 5);
    }

    #[test]
    fn test_compute_gradients_dimension_mismatch() {
        use ndarray::Array1;

        let layer_dims = vec![(10, 5)];
        EpropContext::init_for_layers(layer_dims);

        let wrong_signal = Array1::from_elem(5, 1.0); // Should be 10
        let result = EpropContext::compute_layer_gradients(0, &wrong_signal);

        assert!(result.is_err());
    }

    #[test]
    fn test_preset_horizons() {
        use super::*;
        // Use approx comparison for floating point
        assert!((ContextPreset::DEFAULT.effective_horizon() - 10.0).abs() < 0.001);
        assert!((ContextPreset::LONG_MEMORY.effective_horizon() - 20.0).abs() < 0.001);
        assert!((ContextPreset::SHORT_MEMORY.effective_horizon() - 6.666667).abs() < 0.001);
    }

    #[test]
    fn test_context_config() {
        let config = ContextConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.alpha, ContextPreset::DEFAULT.alpha);
        assert!((config.effective_horizon() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_context_config_from_preset() {
        let config = ContextConfig::from_preset(ContextPreset::LONG_MEMORY, true);
        assert!(config.enabled);
        assert_eq!(config.alpha, 0.95);
        assert!((config.effective_horizon() - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_alpha_short_sequence() {
        // Short sequences should have lower alpha for faster adaptation
        let alpha = ContextConfig::adaptive_alpha(30);
        assert!(alpha >= 0.85 && alpha <= 0.90);
        // α = 1 - 4/30 = 0.866...
        assert!((alpha - 0.8667).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_alpha_medium_sequence() {
        // Medium sequences should have moderate alpha
        let alpha = ContextConfig::adaptive_alpha(100);
        assert!(alpha >= 0.90 && alpha <= 0.95);
        // α = 1 - 4/100 = 0.96, but clamped to 0.95
        assert!((alpha - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_alpha_long_sequence() {
        // Long sequences should have high alpha for long memory
        let alpha = ContextConfig::adaptive_alpha(500);
        assert!(alpha >= 0.95 && alpha <= 0.98);
        // α = 1 - 4/500 = 0.992, clamped to 0.98
        assert!((alpha - 0.98).abs() < 0.01);
    }

    #[test]
    fn test_adaptive_alpha_minimum_sequence() {
        // Very short sequences should be clamped to minimum
        let alpha = ContextConfig::adaptive_alpha(10);
        assert!(alpha >= 0.85 && alpha <= 0.98);
        // α = 1 - 4/20 = 0.80, clamped to 0.85
        assert!((alpha - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_with_adaptive_alpha() {
        let config = ContextConfig::with_adaptive_alpha(200);
        assert!(config.enabled);
        assert!(config.alpha >= 0.85 && config.alpha <= 0.98);
        // For sequence length 200: α = 1 - 4/200 = 0.98
        assert!((config.alpha - 0.98).abs() < 0.01);
    }
}
