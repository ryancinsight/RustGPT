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
//! use eprop::{EPropTrainer, EPropConfig, NeuronModel};
//!
//! let config = EPropConfig {
//!     num_neurons: 128,
//!     input_dim: 64,
//!     output_dim: 10,
//!     ..Default::default()
//! };
//!
//! let mut trainer = EPropTrainer::new(config);
//!
//! // Training loop
//! for (input, target) in dataset {
//!     let loss = trainer.train_step(&input.view(), &target.view())?;
//! }
//! ```

pub mod config;
pub mod context;
pub mod neuron;
pub mod adaptive_softmax;
pub mod checkpoint;
pub mod traces;
pub mod trainer;
pub mod utils;
pub mod incremental_updates;
pub mod adaptive_surrogate;
pub mod mixed_precision;

// Re-export main types for convenience
pub use config::{EPropConfig, NeuronConfig, NeuronModel};
pub use context::{EpropContext, ContextPreset, ContextConfig};
pub use neuron::{NeuronState, NeuronDynamics};
pub use adaptive_softmax::{AdaptiveSoftmax, SoftmaxConfig, SoftmaxStrategy};
pub use checkpoint::{CheckpointManager, TraceCheckpoint, CompressedTraceCheckpoint};
pub use traces::{EligibilityTraces, TraceUpdater};
pub use trainer::{EPropTrainer, TrainingStats};
pub use utils::{cosine_similarity, outer_product,
    should_use_sparse_computation, compute_sparsity_ratio,
    enhanced_sparse_matvec, parallel_sparse_matvec};
pub use mixed_precision::QuantizedEligibilityTraces;
pub use incremental_updates::{IncrementalGradientUpdater, IncrementalState, IncrementalGradientResult};

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
