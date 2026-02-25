//! Loss Functions Module
//!
//! This module provides loss functions for training neural networks.
//!
//! ## CPU Loss Functions (cpu_loss.rs)
//!
//! - Cross-entropy loss
//! - Symmetric cross-entropy loss
//! - Residual decorrelation loss
//! - InfoNCE loss
//! - Hard negative repulsion loss
//!
//! ## GPU Loss Functions (gpu_loss.rs)
//!
//! GPU-native implementations that keep all computations on GPU:
//! - `gpu_cross_entropy_loss` - Cross-entropy on GPU
//! - `gpu_symmetric_cross_entropy_loss` - Symmetric CE on GPU

mod cpu_loss;
mod gpu_loss;

// Re-export CPU loss functions
pub use cpu_loss::*;

// Re-export GPU loss functions
#[cfg(any(feature = "wgpu", feature = "gpu-cuda", feature = "gpu-metal"))]
pub use gpu_loss::{
    gpu_cross_entropy_loss, gpu_symmetric_cross_entropy_loss, GpuLossWorkspace,
    GpuSymmetricCEConfig,
};
