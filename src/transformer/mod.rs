//! Transformer architecture module
//!
//! This module provides reusable transformer block components that can be used
//! across different transformer architectures (standard, hierarchical, recurrent, etc.).
//!
//! The core components are:
//! - `TransformerBlock`: Standard autoregressive transformer block with:
//!   - Pre-attention normalization
//!   - Attention mechanism (PolyAttention with CoPE)
//!   - Pre-feedforward normalization
//!   - Feedforward network (RichardsGlu or MixtureOfExperts)
//!   - Residual connections
//! - `DiffusionBlock`: Diffusion-based transformer that replaces autoregressive prediction with
//!   denoising, featuring:
//!   - Time-conditioned attention for noise level awareness
//!   - Noise scheduling (linear, cosine, quadratic)
//!   - Forward/reverse diffusion processes
//!   - Denoising objective for generative modeling

pub mod adaptive_residuals;
pub mod diffusion_block;
pub mod transformer_block;
pub mod lrm;
pub mod hrm;
pub mod common;
pub mod speculative;
#[cfg(test)]
mod speculative_tests;

pub use diffusion_block::{
    DiffusionBlock, DiffusionBlockConfig, NoiseSchedule, NoiseScheduler, TimeEmbedding,
};
pub use transformer_block::{TransformerBlock, TransformerBlockConfig};
pub use adaptive_residuals::{AdaptiveResidualStrategy, UnifiedAdaptiveResiduals};
pub use lrm::{LRM, LRMConfig};
pub use hrm::{HRM, HRMConfig};
pub use common::FeedForwardVariant;

/// Re-export key types for convenience
pub use crate::attention::poly_attention::PolyAttention;
pub use crate::{
    mixtures::moe::MixtureOfExperts,
    richards::{RichardsGlu, RichardsNorm},
};
