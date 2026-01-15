use ndarray::{Array1, Array2};

use super::{EligibilityTraces, TraceUpdater};
use super::super::config::{EPropConfig, NeuronConfig};
use super::super::neuron::NeuronState;
use super::super::EPropError;
use serde::{Deserialize, Serialize};

/// Scale identifier for multi-scale traces
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TraceScale {
    /// Fast traces: α=0.8 (~5 timestep horizon)
    Fast = 0,
    /// Medium traces: α=0.95 (~20 timestep horizon)  
    Medium = 1,
    /// Slow traces: α=0.99 (~100 timestep horizon)
    Slow = 2,
}

/// Configuration for multi-scale trace weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleWeights {
    /// Weight for fast traces (α=0.8)
    pub fast: f32,
    /// Weight for medium traces (α=0.95)
    pub medium: f32,
    /// Weight for slow traces (α=0.99)
    pub slow: f32,
    /// Enable gradient-magnitude based weighting
    pub use_gradient_weighting: bool,
    /// Enable adaptive weight adjustment
    pub use_adaptive_weighting: bool,
    /// Exponential moving average factor for weight adaptation
    pub adaptation_alpha: f32,
}

impl Default for MultiScaleWeights {
    fn default() -> Self {
        Self {
            fast: 0.33,
            medium: 0.34,
            slow: 0.33,
            use_gradient_weighting: true,
            use_adaptive_weighting: true,
            adaptation_alpha: 0.9,
        }
    }
}

/// Multi-scale eligibility traces manager
///
/// Maintains three parallel trace sets with different temporal horizons:
/// - Fast traces (α=0.8): ~5 timestep horizon for immediate dependencies
/// - Medium traces (α=0.95): ~20 timestep horizon for sequential patterns  
/// - Slow traces (α=0.99): ~100 timestep horizon for long-range dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleTraces {
    /// Fast timescale traces
    pub fast: EligibilityTraces,
    /// Medium timescale traces
    pub medium: EligibilityTraces,
    /// Slow timescale traces
    pub slow: EligibilityTraces,
    
    /// Weight configuration
    pub weights: MultiScaleWeights,
    
    /// Exponential moving averages for adaptive weighting
    #[serde(skip)]
    pub fast_ema: f32,
    #[serde(skip)]
    pub medium_ema: f32,
    #[serde(skip)]
    pub slow_ema: f32,
}

impl MultiScaleTraces {
    /// Create new multi-scale traces
    pub fn new(input_dim: usize, num_neurons: usize, use_adaptation: bool) -> Self {
        let fast_config = create_scale_config(0.8, input_dim, num_neurons, use_adaptation);
        let medium_config = create_scale_config(0.95, input_dim, num_neurons, use_adaptation);
        let slow_config = create_scale_config(0.99, input_dim, num_neurons, use_adaptation);

        Self {
            fast: EligibilityTraces::new_with_config(fast_config, input_dim, num_neurons, use_adaptation),
            medium: EligibilityTraces::new_with_config(medium_config, input_dim, num_neurons, use_adaptation),
            slow: EligibilityTraces::new_with_config(slow_config, input_dim, num_neurons, use_adaptation),
            weights: MultiScaleWeights::default(),
            fast_ema: 0.0,
            medium_ema: 0.0,
            slow_ema: 0.0,
        }
    }

    /// Update all trace scales with current state and input
    pub fn update(
        &mut self,
        state: &NeuronState,
        input: &Array1<f32>,
        fast_updater: &TraceUpdater,
        medium_updater: &TraceUpdater,
        slow_updater: &TraceUpdater,
    ) -> super::Result<()> {
        // Update all three timescales simultaneously
        fast_updater.update(&mut self.fast, state, input)?;
        medium_updater.update(&mut self.medium, state, input)?;
        slow_updater.update(&mut self.slow, state, input)?;
        
        Ok(())
    }

    /// Compute weighted gradient factors from all scales
    pub fn compute_gradient_factors(
        &mut self,
        learning_signal: &Array1<f32>,
        fast_updater: &TraceUpdater,
        medium_updater: &TraceUpdater,
        slow_updater: &TraceUpdater,
    ) -> super::Result<(Array1<f32>, Array1<f32>)> {
        // Get gradient factors from each scale
        let (fast_grad, fast_input) = fast_updater.compute_gradient_factors(&self.fast, learning_signal)?;
        let (medium_grad, medium_input) = medium_updater.compute_gradient_factors(&self.medium, learning_signal)?;
        let (slow_grad, slow_input) = slow_updater.compute_gradient_factors(&self.slow, learning_signal)?;

        // Compute gradient magnitudes for weighting
        let fast_magnitude = fast_grad.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        let medium_magnitude = medium_grad.mapv(|x| x.abs()).mean().unwrap_or(0.0);
        let slow_magnitude = slow_grad.mapv(|x| x.abs()).mean().unwrap_or(0.0);

        // Update EMAs for adaptive weighting
        if self.weights.use_adaptive_weighting {
            let alpha = self.weights.adaptation_alpha;
            self.fast_ema = alpha * self.fast_ema + (1.0 - alpha) * fast_magnitude;
            self.medium_ema = alpha * self.medium_ema + (1.0 - alpha) * medium_magnitude;
            self.slow_ema = alpha * self.slow_ema + (1.0 - alpha) * slow_magnitude;
        }

        // Compute weights
        let weights = self.compute_weights(fast_magnitude, medium_magnitude, slow_magnitude);

        // Weighted combination
        let combined_grad = &fast_grad * weights.fast 
            + &medium_grad * weights.medium 
            + &slow_grad * weights.slow;
        
        let combined_input = &fast_input * weights.fast 
            + &medium_input * weights.medium 
            + &slow_input * weights.slow;

        Ok((combined_grad, combined_input))
    }

    /// Compute weights based on gradient magnitudes
    fn compute_weights(&self, fast_mag: f32, medium_mag: f32, slow_mag: f32) -> MultiScaleWeights {
        if self.weights.use_gradient_weighting {
            // Gradient magnitude based weighting (softmax)
            let max_mag = fast_mag.max(medium_mag.max(slow_mag));
            if max_mag > 0.0 {
                let exp_fast = (fast_mag / max_mag).exp();
                let exp_medium = (medium_mag / max_mag).exp();
                let exp_slow = (slow_mag / max_mag).exp();
                
                let sum_exp = exp_fast + exp_medium + exp_slow;
                
                MultiScaleWeights {
                    fast: exp_fast / sum_exp,
                    medium: exp_medium / sum_exp,
                    slow: exp_slow / sum_exp,
                    ..self.weights
                }
            } else {
                // Fallback to adaptive EMAs or uniform weights
                self.compute_adaptive_weights()
            }
        } else {
            // Use adaptive EMAs if available, otherwise uniform
            self.compute_adaptive_weights()
        }
    }

    /// Compute weights using adaptive EMAs
    fn compute_adaptive_weights(&self) -> MultiScaleWeights {
        if self.weights.use_adaptive_weighting && 
           (self.fast_ema != 0.0 || self.medium_ema != 0.0 || self.slow_ema != 0.0) {
            // Use EMA-based weighting
            let sum_ema = self.fast_ema + self.medium_ema + self.slow_ema;
            if sum_ema > 0.0 {
                MultiScaleWeights {
                    fast: self.fast_ema / sum_ema,
                    medium: self.medium_ema / sum_ema,
                    slow: self.slow_ema / sum_ema,
                    ..self.weights
                }
            } else {
                // Uniform fallback
                MultiScaleWeights {
                    fast: 1.0 / 3.0,
                    medium: 1.0 / 3.0,
                    slow: 1.0 / 3.0,
                    ..self.weights
                }
            }
        } else {
            // Default to current weights (can be custom configured)
            self.weights
        }
    }

    /// Reset all traces
    pub fn reset(&mut self) {
        self.fast.reset();
        self.medium.reset();
        self.slow.reset();
        self.fast_ema = 0.0;
        self.medium_ema = 0.0;
        self.slow_ema = 0.0;
    }

    /// Get effective horizon of each scale
    pub fn get_horizons(&self) -> (usize, usize, usize) {
        (
            (1.0 / (1.0 - 0.8)) as usize,  // Fast: ~5 steps
            (1.0 / (1.0 - 0.95)) as usize, // Medium: ~20 steps
            (1.0 / (1.0 - 0.99)) as usize, // Slow: ~100 steps
        )
    }

    /// Get current weight distribution
    pub fn get_current_weights(&self) -> MultiScaleWeights {
        if self.weights.use_gradient_weighting || self.weights.use_adaptive_weighting {
            // Compute current weights based on EMAs
            self.compute_adaptive_weights()
        } else {
            // Return configured weights
            self.weights
        }
    }
}

/// Create configuration for a specific trace scale
fn create_scale_config(alpha: f32, input_dim: usize, num_neurons: usize, use_adaptation: bool) -> ScaleTraceConfig {
    ScaleTraceConfig {
        alpha,
        input_dim,
        num_neurons,
        use_adaptation,
    }
}

/// Configuration for individual trace scales
#[derive(Debug, Clone)]
struct ScaleTraceConfig {
    alpha: f32,
    input_dim: usize,
    num_neurons: usize,
    use_adaptation: bool,
}

/// Extended eligibility traces that support custom alpha
impl EligibilityTraces {
    /// Create traces with custom configuration
    pub fn new_with_config(
        config: ScaleTraceConfig,
        input_dim: usize,
        num_neurons: usize,
        use_adaptation: bool,
    ) -> Self {
        Self {
            eps_x: Array1::zeros(input_dim),
            eps_f: Array1::zeros(num_neurons),
            eps_a: if use_adaptation {
                Some(Array1::zeros(num_neurons))
            } else {
                None
            },
            position: 0,
            gradient_variance_ema: 1.0,
            alpha_smooth: config.alpha,
        }
    }
}

/// Multi-scale trace updater
#[derive(Debug)]
pub struct MultiScaleUpdater {
    /// Update engine for fast traces
    pub fast_updater: TraceUpdater,
    /// Update engine for medium traces
    pub medium_updater: TraceUpdater,
    /// Update engine for slow traces
    pub slow_updater: TraceUpdater,
    /// Configuration
    pub config: EPropConfig,
}

impl MultiScaleUpdater {
    /// Create new multi-scale updater
    pub fn new(config: &EPropConfig, neuron_config: NeuronConfig) -> Self {
        let fast_alpha = 0.8;
        let medium_alpha = 0.95;
        let slow_alpha = 0.99;

        Self {
            fast_updater: create_scale_updater(fast_alpha, neuron_config.clone()),
            medium_updater: create_scale_updater(medium_alpha, neuron_config.clone()),
            slow_updater: create_scale_updater(slow_alpha, neuron_config),
            config: config.clone(),
        }
    }

    /// Create traces for this updater
    pub fn create_traces(&self, input_dim: usize, num_neurons: usize, use_adaptation: bool) -> MultiScaleTraces {
        MultiScaleTraces::new(input_dim, num_neurons, use_adaptation)
    }

    /// Update traces and compute gradient factors
    pub fn update_and_compute(
        &mut self,
        traces: &mut MultiScaleTraces,
        state: &NeuronState,
        input: &Array1<f32>,
        learning_signal: &Array1<f32>,
    ) -> super::Result<(Array1<f32>, Array1<f32>)> {
        // Update all traces
        traces.update(state, input, &self.fast_updater, &self.medium_updater, &self.slow_updater)?;

        // Compute weighted gradient factors
        traces.compute_gradient_factors(
            learning_signal,
            &self.fast_updater,
            &self.medium_updater,
            &self.slow_updater
        )
    }
}

/// Create trace updater for a specific scale
fn create_scale_updater(alpha: f32, neuron_config: NeuronConfig) -> TraceUpdater {
    TraceUpdater::from_alpha(alpha, neuron_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eprop::config::NeuronConfig;

    #[test]
    fn test_multi_scale_traces_creation() {
        let traces = MultiScaleTraces::new(4, 8, false);
        assert_eq!(traces.fast.eps_x.len(), 4);
        assert_eq!(traces.fast.eps_f.len(), 8);
        assert_eq!(traces.medium.eps_x.len(), 4);
        assert_eq!(traces.slow.eps_f.len(), 8);
        
        // Check horizons
        let (fast, medium, slow) = traces.get_horizons();
        assert_eq!(fast, 5);    // 1/(1-0.8) = 5
        assert_eq!(medium, 20); // 1/(1-0.95) = 20
        assert_eq!(slow, 100);  // 1/(1-0.99) = 100
    }

    #[test]
    fn test_multi_scale_updater_creation() {
        let config = EPropConfig::default();
        let neuron_config = NeuronConfig::lif();
        let updater = MultiScaleUpdater::new(&config, neuron_config);
        
        assert_eq!(updater.fast_updater.alpha(), 0.8);
        assert_eq!(updater.medium_updater.alpha(), 0.95);
        assert_eq!(updater.slow_updater.alpha(), 0.99);
    }

    #[test]
    fn test_adaptive_weights() {
        let mut traces = MultiScaleTraces::new(4, 8, false);
        
        // Set up different EMAs
        traces.fast_ema = 0.5;
        traces.medium_ema = 1.0;
        traces.slow_ema = 0.3;
        
        let weights = traces.compute_adaptive_weights();
        
        // Should sum to 1.0 and reflect EMA ratios
        assert!((weights.fast + weights.medium + weights.slow - 1.0).abs() < 1e-5);
        assert!(weights.medium > weights.fast); // 1.0 > 0.5
        assert!(weights.fast > weights.slow);   // 0.5 > 0.3
    }

    #[test]
    fn test_gradient_magnitude_weighting() {
        let mut traces = MultiScaleTraces::new(4, 8, false);
        
        let fast_grad = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let medium_grad = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let slow_grad = Array1::from_vec(vec![0.05, 0.1, 0.15, 0.2]);
        
        let fast_mag = fast_grad.mapv(|x| x.abs()).mean().unwrap();
        let medium_mag = medium_grad.mapv(|x| x.abs()).mean().unwrap();
        let slow_mag = slow_grad.mapv(|x| x.abs()).mean().unwrap();
        
        let weights = traces.compute_weights(fast_mag, medium_mag, slow_mag);
        
        // Fast should have highest weight due to largest gradients
        assert!(weights.fast > weights.medium);
        assert!(weights.medium > weights.slow);
    }
}
