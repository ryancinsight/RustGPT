//! State space model (SSM) layers.
//!
//! This module provides state space model implementations including:
//! - Mamba: Full-featured selective SSM with attention mechanisms
//! - Mamba2: Optimized version of Mamba with larger convolution kernels  
//! - RG-LRU: Real-Gated Linear Recurrent Unit with diagonal recurrence
//!
//! The module also includes reusable components for building custom SSM architectures:
//! - StateManagement: Smart caching with automatic invalidation and memory optimization
//! - SelectiveScan: Optimized selective scanning with parallel processing support
//! - ProjectionLayers: Reusable linear projections and depthwise convolutions
//! - RichardsIntegration: Integration with the Richards adaptive activation system
//!
//! ## Usage Example
//! ```rust
//! use llm::layers::ssm::{SelectiveScanner, SsmRichardsActivation, StateManager};
//! use ndarray::Array2;
//!
//! // Create a state manager with memory limits
//! let mut state_manager = StateManager::new(512, 1024 * 1024); // 1MB limit
//!
//! // Create a selective scanner with parallel processing
//! let scanner = SelectiveScanner::new();
//!
//! // Create Richards-based activation for SSM
//! let activation = SsmRichardsActivation::sigmoid(true, true); // Learnable Swish-like
//!
//! // Use in your SSM implementation
//! let input = Array2::zeros((64, 512));
//! let cache = state_manager.cache(&input);
//! let output = activation.forward(&input);
//! ```

pub(crate) mod components;
pub(crate) mod mamba;
pub(crate) mod mamba2;
pub(crate) mod rg_lru;

pub use components::*;
pub use mamba::Mamba;
pub use mamba2::Mamba2;
pub use rg_lru::{MoHRgLru, RgLru};
