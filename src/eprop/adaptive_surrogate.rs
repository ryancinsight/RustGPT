//! Adaptive Surrogate Gradients for Enhanced Learning
//!
//! This module implements dynamic surrogate gradient functions that adapt
//! based on training dynamics, neuron state, and task requirements.
//!
//! The system provides multiple surrogate functions with different properties
//! and adapts between them to optimize learning performance and stability.

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use crate::eprop::EPropError;

/// Types of surrogate gradient functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SurrogateFunction {
    /// Piecewise Linear (standard)
    PiecewiseLinear,
    /// Sigmoid approximation
    Sigmoid,
    /// Fast Sigmoid (optimized)
    FastSigmoid,
    /// Gaussian approximation
    Gaussian,
    /// Adaptive piecewise linear
    AdaptivePiecewise,
    /// Task-optimized hybrid
    Hybrid,
}

/// Adaptive surrogate gradient engine
#[derive(Debug, Clone)]
pub struct AdaptiveSurrogate {
    /// Current active function type
    current_function: SurrogateFunction,
    
    /// Performance metrics for adaptation
    performance_history: Vec<PerformanceMetrics>,
    
    /// Adaptation parameters
    adaptation_rate: f32,
    performance_window: usize,
    
    /// Function parameters (may vary by function type)
    function_params: FunctionParams,
    
    /// Neural activity tracking
    activity_stats: ActivityStats,
}

/// Performance metrics for surrogate function evaluation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Gradient correlation with true gradient
    pub gradient_correlation: f32,
    
    /// Learning stability (inverse of gradient variance)
    pub stability_score: f32,
    
    /// Training loss improvement rate
    pub loss_improvement_rate: f32,
    
    /// Spike generation efficiency
    pub spike_efficiency: f32,
    
    /// Overall performance score
    pub overall_score: f32,
}

/// Function-specific parameters
#[derive(Debug, Clone)]
struct FunctionParams {
    /// Sigmoid steepness parameter
    sigmoid_steepness: f32,
    
    /// Gaussian width parameter
    gaussian_width: f32,
    
    /// Adaptive window size
    adaptive_window: f32,
    
    /// Hybrid function weights
    hybrid_weights: [f32; 3],
}

/// Neural activity statistics for adaptation
#[derive(Debug, Clone)]
pub struct ActivityStats {
    /// Average firing rate
    avg_firing_rate: f32,
    
    /// Membrane potential variance
    voltage_variance: f32,
    
    /// Spike timing precision
    spike_precision: f32,
    
    /// Adaptation strength
    adaptation_strength: f32,
}

impl AdaptiveSurrogate {
    /// Create new adaptive surrogate gradient system
    pub fn new(initial_function: SurrogateFunction) -> Self {
        Self {
            current_function: initial_function,
            performance_history: Vec::with_capacity(100),
            adaptation_rate: 0.01,
            performance_window: 50,
            function_params: FunctionParams::default(),
            activity_stats: ActivityStats::default(),
        }
    }
    
    /// Create with optimized parameters for specific use case
    pub fn optimized_for_task(task_type: TaskType) -> Self {
        match task_type {
            TaskType::Classification => Self::new(SurrogateFunction::PiecewiseLinear),
            TaskType::Regression => Self::new(SurrogateFunction::Sigmoid),
            TaskType::Sequence => Self::new(SurrogateFunction::AdaptivePiecewise),
            TaskType::Memory => Self::new(SurrogateFunction::Hybrid),
        }
    }
    
    /// Compute surrogate gradient for voltage relative to threshold
    pub fn compute_surrogate_gradient(
        &mut self,
        voltage: &Array1<f32>,
        threshold: &Array1<f32>,
        neuron_state: &super::neuron::NeuronState,
        loss_gradient: Option<&Array1<f32>>,
    ) -> Array1<f32> {
        // Update activity statistics
        self.update_activity_stats(neuron_state);
        
        // Compute gradient using current function
        let gradient = self.compute_gradient_with_current_function(voltage, threshold);
        
        // Update performance metrics if loss gradient is available
        if let Some(loss_grad) = loss_gradient {
            self.update_performance_metrics(&gradient, loss_grad);
        }
        
        // Check if adaptation is needed
        if self.should_adapt() {
            self.adapt_function();
        }
        
        gradient
    }
    
    /// Compute gradient using the current surrogate function
    fn compute_gradient_with_current_function(
        &self,
        voltage: &Array1<f32>,
        threshold: &Array1<f32>,
    ) -> Array1<f32> {
        let n = voltage.len();
        let mut gradient = Array1::zeros(n);
        
        for i in 0..n {
            let delta = voltage[i] - threshold[i];
            gradient[i] = match self.current_function {
                SurrogateFunction::PiecewiseLinear => {
                    self.piecewise_linear_surrogate(delta, threshold[i])
                }
                SurrogateFunction::Sigmoid => {
                    self.sigmoid_surrogate(delta)
                }
                SurrogateFunction::FastSigmoid => {
                    self.fast_sigmoid_surrogate(delta)
                }
                SurrogateFunction::Gaussian => {
                    self.gaussian_surrogate(delta)
                }
                SurrogateFunction::AdaptivePiecewise => {
                    self.adaptive_piecewise_surrogate(delta, threshold[i])
                }
                SurrogateFunction::Hybrid => {
                    self.hybrid_surrogate(delta, threshold[i])
                }
            };
        }
        
        gradient
    }
    
    /// Piecewise linear surrogate gradient (original)
    fn piecewise_linear_surrogate(&self, delta: f32, threshold: f32) -> f32 {
        let abs_delta = delta.abs() / threshold;
        if abs_delta < 1.0 {
            (1.0 - abs_delta) / (0.3 * threshold) // gamma_pd = 0.3
        } else {
            0.0
        }
    }
    
    /// Sigmoid surrogate gradient
    fn sigmoid_surrogate(&self, delta: f32) -> f32 {
        let steepness = self.function_params.sigmoid_steepness;
        let sigmoid = 1.0 / (1.0 + (-steepness * delta).exp());
        sigmoid * (1.0 - sigmoid) * steepness
    }
    
    /// Fast sigmoid surrogate gradient (optimized approximation)
    fn fast_sigmoid_surrogate(&self, delta: f32) -> f32 {
        // Fast approximation: f(x) = x / (1 + |x|)
        let abs_delta = delta.abs();
        if abs_delta < 1.0 {
            (1.0 - abs_delta).max(0.0)
        } else {
            0.1 / abs_delta // Small gradient for far from threshold
        }
    }
    
    /// Gaussian surrogate gradient
    fn gaussian_surrogate(&self, delta: f32) -> f32 {
        let width = self.function_params.gaussian_width;
        let norm = 1.0 / (width * (2.0 * std::f32::consts::PI).sqrt());
        norm * (-0.5 * (delta / width).powi(2)).exp()
    }
    
    /// Adaptive piecewise linear surrogate
    fn adaptive_piecewise_surrogate(&self, delta: f32, threshold: f32) -> f32 {
        let window = self.function_params.adaptive_window;
        let normalized_delta = delta / threshold;
        
        // Adaptive window based on recent neuron activity
        let activity_factor = 1.0 + self.activity_stats.avg_firing_rate * 0.5;
        let adaptive_window = window * activity_factor;
        
        let abs_normalized = normalized_delta.abs();
        if abs_normalized < adaptive_window {
            (adaptive_window - abs_normalized) / (adaptive_window * threshold)
        } else {
            0.0
        }
    }
    
    /// Hybrid surrogate gradient (combination of multiple functions)
    fn hybrid_surrogate(&self, delta: f32, threshold: f32) -> f32 {
        let weights = self.function_params.hybrid_weights;
        
        let piecewise = self.piecewise_linear_surrogate(delta, threshold);
        let sigmoid = self.sigmoid_surrogate(delta);
        let gaussian = self.gaussian_surrogate(delta);
        
        weights[0] * piecewise + weights[1] * sigmoid + weights[2] * gaussian
    }
    
    /// Update activity statistics
    fn update_activity_stats(&mut self, neuron_state: &super::neuron::NeuronState) {
        let firing_rate = neuron_state.spikes.mean().unwrap_or(0.0);
        let voltage_var = neuron_state.voltage.var(0.0);
        
        // Update EMA
        let alpha = 0.1;
        self.activity_stats.avg_firing_rate = 
            alpha * firing_rate + (1.0 - alpha) * self.activity_stats.avg_firing_rate;
        self.activity_stats.voltage_variance = 
            alpha * voltage_var + (1.0 - alpha) * self.activity_stats.voltage_variance;
        
        // Update spike precision (coefficient of variation)
        if firing_rate > 0.0 {
            let spike_count = neuron_state.spikes.len() as f32 * firing_rate;
            let precision = if spike_count > 1.0 {
                1.0 / (1.0 + (spike_count - 1.0).sqrt())
            } else {
                1.0
            };
            self.activity_stats.spike_precision = 
                alpha * precision + (1.0 - alpha) * self.activity_stats.spike_precision;
        }
        
        // Update adaptation strength if available
        if let Some(adaptation) = &neuron_state.adaptation {
            let adapt_strength = adaptation.mean().unwrap_or(0.0);
            self.activity_stats.adaptation_strength = 
                alpha * adapt_strength + (1.0 - alpha) * self.activity_stats.adaptation_strength;
        }
    }
    
    /// Update performance metrics
    fn update_performance_metrics(&mut self, surrogate_grad: &Array1<f32>, true_grad: &Array1<f32>) {
        if surrogate_grad.len() != true_grad.len() {
            return; // Skip if dimensions don't match
        }
        
        // Compute gradient correlation
        let correlation = compute_correlation(surrogate_grad, true_grad);
        
        // Compute stability score (inverse of gradient variance)
        let surrogate_var = surrogate_grad.var(1.0);
        let stability = 1.0 / (1.0 + surrogate_var);
        
        // Compute spike efficiency
        let avg_surrogate = surrogate_grad.mean().unwrap_or(0.0);
        let spike_efficiency = if avg_surrogate > 0.0 {
            crate::richards::RichardsCurve::sigmoid(false).forward_scalar_f32(avg_surrogate)
        } else {
            0.0
        };
        
        // Compute overall score
        let overall_score = 0.4 * correlation + 0.3 * stability + 0.3 * spike_efficiency;
        
        let metrics = PerformanceMetrics {
            gradient_correlation: correlation,
            stability_score: stability,
            loss_improvement_rate: 0.0, // Would need loss history
            spike_efficiency,
            overall_score,
        };
        
        self.performance_history.push(metrics);
        
        // Keep history within window size
        if self.performance_history.len() > self.performance_window {
            self.performance_history.remove(0);
        }
    }
    
    /// Determine if function adaptation is needed
    fn should_adapt(&self) -> bool {
        if self.performance_history.len() < 10 {
            return false; // Need minimum history
        }
        
        let window = 10usize.min(self.performance_history.len() / 2).max(1);

        let recent_scores = self
            .performance_history
            .iter()
            .rev()
            .take(window)
            .map(|m| m.overall_score);
        let mut recent_sum = 0.0f32;
        let mut recent_n = 0usize;
        for s in recent_scores {
            recent_sum += s;
            recent_n += 1;
        }
        if recent_n == 0 {
            return false;
        }
        let recent_avg = recent_sum / recent_n as f32;

        let older_scores = self.performance_history.iter().take(window).map(|m| m.overall_score);
        let mut older_sum = 0.0f32;
        let mut older_n = 0usize;
        for s in older_scores {
            older_sum += s;
            older_n += 1;
        }
        if older_n == 0 {
            return false;
        }
        let older_avg = older_sum / older_n as f32;
            
        recent_avg < older_avg * 0.95 // 5% performance drop triggers adaptation
    }
    
    /// Adapt to better performing surrogate function
    fn adapt_function(&mut self) {
        if self.performance_history.len() < 5 {
            return;
        }
        
        // Evaluate all functions and select the best
        let mut best_function = self.current_function;
        let mut best_score = self.get_current_performance_score();
        
        for function in [
            SurrogateFunction::PiecewiseLinear,
            SurrogateFunction::Sigmoid,
            SurrogateFunction::FastSigmoid,
            SurrogateFunction::Gaussian,
            SurrogateFunction::AdaptivePiecewise,
            SurrogateFunction::Hybrid,
        ] {
            if function != self.current_function {
                let score = self.estimate_function_performance(function);
                if score > best_score {
                    best_score = score;
                    best_function = function;
                }
            }
        }
        
        if best_function != self.current_function {
            self.current_function = best_function;
            self.adapt_function_parameters();
        }
    }
    
    /// Get current performance score
    fn get_current_performance_score(&self) -> f32 {
        if self.performance_history.is_empty() {
            return 0.5;
        }

        let window = 10usize.min(self.performance_history.len()).max(1);
        let mut sum = 0.0f32;
        let mut n = 0usize;
        for s in self
            .performance_history
            .iter()
            .rev()
            .take(window)
            .map(|m| m.overall_score)
        {
            sum += s;
            n += 1;
        }
        if n == 0 { 0.5 } else { sum / n as f32 }
    }
    
    /// Estimate performance of a candidate function (simulation-based)
    fn estimate_function_performance(&self, function: SurrogateFunction) -> f32 {
        let mut candidate = self.clone();
        candidate.current_function = function;
        candidate.adapt_function_parameters();

        let var = candidate.activity_stats.voltage_variance;
        let mut sigma = if var.is_finite() && var >= 0.0 { var.sqrt() } else { 1.0 };
        if !sigma.is_finite() || sigma <= 0.0 {
            sigma = 1.0;
        }

        let grid = 33usize;
        let mut weights_sum = 0.0f64;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        let mut mean_abs = 0.0f64;

        for i in 0..grid {
            let z = -3.0f64 + (6.0f64 * (i as f64) / ((grid - 1) as f64));
            let w = (-0.5 * z * z).exp();
            let d = candidate.derivative((z as f32) * sigma) as f64;
            let v = if d.is_finite() { d } else { 0.0 };
            weights_sum += w;
            mean += w * v;
            mean_abs += w * v.abs();
            m2 += w * v * v;
        }

        if weights_sum <= 0.0 {
            return 0.5;
        }

        mean /= weights_sum;
        mean_abs /= weights_sum;
        m2 /= weights_sum;
        let var = (m2 - mean * mean).max(0.0);

        let stability = 1.0 / (1.0 + var);
        let responsiveness = mean_abs / (1.0 + mean_abs);
        let score = 0.6 * stability + 0.4 * responsiveness;
        (score as f32).clamp(0.0, 1.0)
    }

    /// Adapt function-specific parameters
    fn adapt_function_parameters(&mut self) {
        match self.current_function {
            SurrogateFunction::Sigmoid => {
                // Adapt steepness based on firing rate
                let target_steepness = match self.activity_stats.avg_firing_rate {
                    rate if rate < 0.1 => 2.0,  // Lower steepness for sparse activity
                    rate if rate > 0.5 => 8.0,  // Higher steepness for dense activity
                    _ => 4.0,                   // Default
                };
                self.function_params.sigmoid_steepness =
                    0.9 * self.function_params.sigmoid_steepness + 0.1 * target_steepness;
            }

            SurrogateFunction::Gaussian => {
                // Adapt width based on voltage variance
                let target_width = (self.activity_stats.voltage_variance.sqrt() * 2.0).max(0.1);
                self.function_params.gaussian_width =
                    0.9 * self.function_params.gaussian_width + 0.1 * target_width;
            }

            SurrogateFunction::AdaptivePiecewise => {
                // Adapt window based on spike precision
                let target_window = (self.activity_stats.spike_precision * 2.0).clamp(0.5, 2.0);
                self.function_params.adaptive_window =
                    0.9 * self.function_params.adaptive_window + 0.1 * target_window;
            }

            SurrogateFunction::Hybrid => {
                // Adapt weights based on overall activity
                let total_activity = self.activity_stats.avg_firing_rate +
                                    self.activity_stats.adaptation_strength;

                if total_activity < 0.3 {
                    // Low activity - emphasize fast sigmoid
                    self.function_params.hybrid_weights = [0.2, 0.6, 0.2];
                } else if total_activity > 0.7 {
                    // High activity - emphasize stable piecewise
                    self.function_params.hybrid_weights = [0.6, 0.2, 0.2];
                } else {
                    // Balanced - emphasize hybrid
                    self.function_params.hybrid_weights = [0.33, 0.33, 0.34];
                }
            }

            _ => {} // No parameter adaptation needed
        }
    }

    /// Get current function type
    pub fn current_function(&self) -> SurrogateFunction {
        self.current_function
    }

    /// Get performance history
    pub fn performance_history(&self) -> &[PerformanceMetrics] {
        &self.performance_history
    }

    /// Force switch to specific function (for debugging/testing)
    pub fn set_function(&mut self, function: SurrogateFunction) {
        self.current_function = function;
        self.adapt_function_parameters();
    }

    /// Reset adaptation state
    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.activity_stats = ActivityStats::default();
        self.function_params = FunctionParams::default();
    }
}

impl AdaptiveSurrogate {
    /// Create activity statistics from neuron state
    pub fn create_activity_stats(
        &self,
        voltage: &Array1<f32>,
        threshold: &Array1<f32>,
        spikes: &Array1<f32>,
    ) -> ActivityStats {
        let firing_rate = spikes.mean().unwrap_or(0.0);
        let voltage_var = voltage.var(0.0);

        // Compute spike timing precision (coefficient of variation)
        let spike_precision = if firing_rate > 0.0 && firing_rate < 1.0 {
            // Use voltage variance as proxy for timing precision
            // Lower variance = more precise timing
            let base_precision = 1.0 / (1.0 + voltage_var.sqrt());
            // Adjust based on firing rate (optimal around 0.1-0.2)
            let rate_factor = if firing_rate < 0.05 {
                firing_rate / 0.05 // Penalize very low rates
            } else if firing_rate > 0.3 {
                0.3 / firing_rate // Penalize very high rates
            } else {
                1.0 // Optimal range
            };
            base_precision * rate_factor
        } else {
            0.1 // Poor precision for extreme firing rates
        };

        // Estimate adaptation strength from threshold distribution
        let threshold_var = threshold.var(0.0);
        let adaptation_strength = if threshold.len() > 1 {
            // Higher variance suggests stronger adaptation
            (threshold_var / self.function_params.gaussian_width).min(1.0)
        } else {
            0.0
        };

        ActivityStats {
            avg_firing_rate: firing_rate,
            voltage_variance: voltage_var,
            spike_precision,
            adaptation_strength,
        }
    }

    /// Compute derivative for a single delta value
    pub fn derivative(&self, delta: f32) -> f32 {
        match self.current_function {
            SurrogateFunction::PiecewiseLinear => {
                self.piecewise_linear_surrogate(delta, 1.0) // threshold=1.0 as default
            }
            SurrogateFunction::Sigmoid => self.sigmoid_surrogate(delta),
            SurrogateFunction::FastSigmoid => self.fast_sigmoid_surrogate(delta),
            SurrogateFunction::Gaussian => self.gaussian_surrogate(delta),
            SurrogateFunction::AdaptivePiecewise => {
                self.adaptive_piecewise_surrogate(delta, 1.0) // threshold=1.0 as default
            }
            SurrogateFunction::Hybrid => {
                self.hybrid_surrogate(delta, 1.0) // threshold=1.0 as default
            }
        }
    }
}

/// Performance tracking for adaptive surrogate functions
#[derive(Debug, Clone)]
pub struct SurrogatePerformance {
    /// Current adaptive surrogate instance
    adaptive_surrogate: AdaptiveSurrogate,

    /// Performance history window size
    window_size: usize,

    /// Loss history for improvement rate calculation
    loss_history: Vec<f32>,

    /// Previous surrogate gradients for correlation analysis
    previous_surrogate_grads: Option<Array1<f32>>,
}

impl SurrogatePerformance {
    /// Create new performance tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            adaptive_surrogate: AdaptiveSurrogate::new(SurrogateFunction::PiecewiseLinear),
            window_size,
            loss_history: Vec::with_capacity(window_size * 2),
            previous_surrogate_grads: None,
        }
    }

    /// Get current adaptive surrogate instance
    pub fn get_current_surrogate(&self) -> AdaptiveSurrogate {
        self.adaptive_surrogate.clone()
    }

    /// Update performance with activity statistics
    pub fn update_with_activity(
        &mut self,
        adaptive: AdaptiveSurrogate,
        _activity_stats: &ActivityStats,
    ) -> Result<(), EPropError> {
        // Update the internal adaptive surrogate with the caller's updated version
        self.adaptive_surrogate = adaptive;
        Ok(())
    }

    /// Update performance with gradient and loss information
    pub fn update_with_gradient(
        &mut self,
        loss_gradient: &Array1<f32>,
        surrogate_gradient: &Array1<f32>,
        current_loss: f32,
    ) -> Result<(), EPropError> {
        // Compute gradient correlation with previous surrogate gradients
        let gradient_correlation = if let Some(ref prev_grads) = self.previous_surrogate_grads {
            if prev_grads.len() == surrogate_gradient.len() {
                compute_correlation(prev_grads, surrogate_gradient)
            } else {
                0.5 // Neutral correlation if dimensions changed
            }
        } else {
            0.5 // Neutral correlation for first update
        };

        // Update previous gradients for next correlation calculation
        self.previous_surrogate_grads = Some(surrogate_gradient.clone());

        // Compute stability score (inverse of gradient variance)
        let surrogate_var = surrogate_gradient.var(1.0);
        let stability_score = 1.0 / (1.0 + surrogate_var);

        // Compute loss improvement rate
        let loss_improvement_rate = if self.loss_history.len() >= 5 {
            let recent_avg = self.loss_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let older_avg = self.loss_history.iter().rev().skip(5).take(5).sum::<f32>() / 5.0;
            if older_avg > 0.0 {
                (older_avg - recent_avg) / older_avg // Positive = improvement
            } else {
                0.0
            }
        } else {
            0.0 // No improvement data yet
        };

        // Compute spike efficiency (how well surrogate gradients correlate with loss gradients)
        let spike_efficiency = compute_correlation(surrogate_gradient, loss_gradient).abs();

        // Compute overall performance score
        let overall_score = 0.3 * gradient_correlation +
                           0.2 * stability_score +
                           0.25 * loss_improvement_rate.clamp(0.0, 1.0) +
                           0.25 * spike_efficiency;

        // Create and store performance metrics
        let metrics = PerformanceMetrics {
            gradient_correlation,
            stability_score,
            loss_improvement_rate,
            spike_efficiency,
            overall_score,
        };

        // Update adaptive surrogate with these metrics
        self.adaptive_surrogate.performance_history.push(metrics);

        // Keep history within window size
        if self.adaptive_surrogate.performance_history.len() > self.window_size {
            self.adaptive_surrogate.performance_history.remove(0);
        }

        // Update loss history
        self.loss_history.push(current_loss);
        if self.loss_history.len() > self.window_size * 2 {
            self.loss_history.remove(0);
        }

        Ok(())
    }

    /// Get current performance score
    pub fn current_performance_score(&self) -> f32 {
        if self.adaptive_surrogate.performance_history.is_empty() {
            0.5
        } else {
            let recent_scores: Vec<f32> = self.adaptive_surrogate.performance_history
                .iter()
                .rev()
                .take(10.min(self.adaptive_surrogate.performance_history.len()))
                .map(|m| m.overall_score)
                .collect();

            if recent_scores.is_empty() {
                0.5
            } else {
                recent_scores.iter().sum::<f32>() / recent_scores.len() as f32
            }
        }
    }

    /// Check if adaptation should be triggered
    pub fn should_adapt(&self) -> bool {
        self.adaptive_surrogate.should_adapt()
    }

    /// Trigger adaptation to better performing surrogate
    pub fn adapt(&mut self) {
        self.adaptive_surrogate.adapt_function();
    }
}

/// Task types for optimization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TaskType {
    Classification,
    Regression,
    Sequence,
    Memory,
}

/// Compute correlation between two arrays
fn compute_correlation(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    
    let mean_a = a.iter().take(n).sum::<f32>() / n as f32;
    let mean_b = b.iter().take(n).sum::<f32>() / n as f32;
    
    let mut numerator = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    
    for i in 0..n {
        let diff_a = a[i] - mean_a;
        let diff_b = b[i] - mean_b;
        
        numerator += diff_a * diff_b;
        denom_a += diff_a * diff_a;
        denom_b += diff_b * diff_b;
    }
    
    let denominator = (denom_a * denom_b).sqrt();
    if denominator > 1e-8 {
        numerator / denominator
    } else {
        0.0
    }
}

impl Default for FunctionParams {
    fn default() -> Self {
        Self {
            sigmoid_steepness: 4.0,
            gaussian_width: 1.0,
            adaptive_window: 1.0,
            hybrid_weights: [0.33, 0.33, 0.34],
        }
    }
}

impl Default for ActivityStats {
    fn default() -> Self {
        Self {
            avg_firing_rate: 0.1,
            voltage_variance: 1.0,
            spike_precision: 0.5,
            adaptation_strength: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eprop::neuron::NeuronState;
    use crate::eprop::config::NeuronConfig;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_adaptive_surrogate_creation() {
        let adaptive = AdaptiveSurrogate::new(SurrogateFunction::PiecewiseLinear);
        assert_eq!(adaptive.current_function(), SurrogateFunction::PiecewiseLinear);
    }
    
    #[test]
    fn test_surrogate_functions() {
        let adaptive = AdaptiveSurrogate::new(SurrogateFunction::PiecewiseLinear);
        
        // Test at threshold
        let grad_linear = adaptive.piecewise_linear_surrogate(0.0, 1.0);
        let grad_sigmoid = adaptive.sigmoid_surrogate(0.0);
        let grad_gaussian = adaptive.gaussian_surrogate(0.0);
        
        // Should be positive at threshold
        assert!(grad_linear > 0.0);
        assert!(grad_sigmoid > 0.0);
        assert!(grad_gaussian > 0.0);
    }
    
    #[test]
    fn test_fast_sigmoid_properties() {
        let adaptive = AdaptiveSurrogate::new(SurrogateFunction::FastSigmoid);
        
        // At threshold
        let grad = adaptive.fast_sigmoid_surrogate(0.0);
        assert_relative_eq!(grad, 1.0, epsilon = 1e-6);
        
        // Far from threshold should approach 0
        let grad_far = adaptive.fast_sigmoid_surrogate(10.0);
        assert!(grad_far < 0.1);
    }
    
    #[test]
    fn test_activity_stats_update() {
        let mut adaptive = AdaptiveSurrogate::new(SurrogateFunction::PiecewiseLinear);
        
        let config = NeuronConfig::default();
        let mut state = NeuronState::new(5, false, &config);
        state.spikes.fill(0.5);
        state.voltage.fill(1.0);
        
        adaptive.update_activity_stats(&state);
        
        assert!(adaptive.activity_stats.avg_firing_rate > 0.0);
        assert!(adaptive.activity_stats.voltage_variance > 0.0);
    }
    
    #[test]
    fn test_function_switching() {
        let mut adaptive = AdaptiveSurrogate::new(SurrogateFunction::PiecewiseLinear);
        
        adaptive.set_function(SurrogateFunction::Sigmoid);
        assert_eq!(adaptive.current_function(), SurrogateFunction::Sigmoid);
    }
    
    #[test]
    fn test_correlation_computation() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0]); // Perfect correlation
        
        let corr = compute_correlation(&a, &b);
        assert!(corr > 0.99); // Should be nearly 1.0
    }
}
