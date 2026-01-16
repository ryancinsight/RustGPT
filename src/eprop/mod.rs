//! Optimized Eligibility Propagation (e-prop) via ES-D-RTRL for Scalable Spiking Neural Networks
//!
//! This module implements the unified e-prop framework enhanced with Exponentially Smoothed
//! Diagonal Approximated Real-Time Recurrent Learning (ES-D-RTRL), achieving **O(N) time and
//! memory complexity** while maintaining 90-99% gradient fidelity to full BPTT.
//!
//! # Architecture
//!
//! The implementation is split into focused modules:
//! - `config`: Configuration structures for neurons and training
//! - `neuron`: Neuron dynamics (LIF/ALIF) and state management
//! - `traces`: Eligibility trace computation and updates
//! - `trainer`: Main training loop and gradient updates
//! - `utils`: Utility functions for linear algebra operations
//! - `context`: Thread-local trace persistence across sequences
//!
//! # Key Features
//! - **Linear Complexity**: O(N) per timestep vs O(N²) for standard e-prop
//! - **Biological Plausibility**: Local eligibility traces + global learning signals
//! - **Online Learning**: Forward-only gradient computation (no backward pass)
//! - **SNN Optimized**: Leverages spike sparsity and signed-input properties
//! - **Scalable**: Supports brain-scale models (125k+ neurons)
//!
//! # Quick Start
//!
//! ```rust
//! use llm::eprop::{EPropConfig, EPropTrainer, NeuronConfig};
//! use ndarray::Array1;
//!
//! fn main() -> llm::eprop::Result<()> {
//!     let config = EPropConfig {
//!         num_neurons: 128,
//!         input_dim: 64,
//!         output_dim: 10,
//!         neuron_config: NeuronConfig::default(),
//!         ..Default::default()
//!     };
//!
//!     let mut trainer = EPropTrainer::new(config)?;
//!
//!     let dataset: Vec<(Array1<f32>, usize)> = vec![(Array1::zeros(64), 0)];
//!     for (input, target_class) in dataset {
//!         let _loss = trainer.train_step_classification(&input, target_class)?;
//!     }
//!
//!     Ok(())
//! }
//! ```

pub mod adaptive_softmax;
pub mod adaptive_surrogate;
pub mod checkpoint;
pub mod config;
pub mod context;
pub mod incremental_updates;
pub mod mixed_precision;
pub mod neuron;
pub mod traces;
pub mod trainer;
pub mod utils;

// Re-export main types for convenience
pub use adaptive_softmax::{AdaptiveSoftmax, SoftmaxConfig, SoftmaxStrategy};
pub use checkpoint::{CheckpointManager, CompressedTraceCheckpoint, TraceCheckpoint};
pub use config::{EPropConfig, NeuronConfig, NeuronModel};
pub use context::{ContextConfig, ContextPreset, EpropContext};
pub use incremental_updates::{
    IncrementalGradientResult, IncrementalGradientUpdater, IncrementalState,
};
pub use mixed_precision::QuantizedEligibilityTraces;
pub use neuron::{NeuronDynamics, NeuronState};
pub use traces::{EligibilityTraces, TraceUpdater};
pub use trainer::{EPropTrainer, TrainingStats};
pub use utils::{
    compute_sparsity_ratio, cosine_similarity, enhanced_sparse_matvec, outer_product,
    parallel_sparse_matvec, should_use_sparse_computation,
};

/// Errors specific to e-prop training
#[derive(thiserror::Error, Debug)]
pub enum EPropError {
    #[error("Invalid neuron dynamics parameters: {0}")]
    InvalidDynamics(String),

    #[error("Trace dimensionality mismatch: expected {expected}, got {actual}")]
    TraceDimensionMismatch { expected: usize, actual: usize },

    #[error("Learning signal not available at timestep {0}")]
    MissingLearningSignal(usize),

    #[error("Gradient anomaly detected: {0}")]
    GradientAnomaly(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Shape mismatch: {expected}, got {got}")]
    ShapeMismatch { expected: String, got: String },

    #[error("Compute error: {0}")]
    ComputeError(String),
}

pub type Result<T> = std::result::Result<T, EPropError>;
