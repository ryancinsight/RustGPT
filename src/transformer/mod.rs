//! Transformer architecture module
//!
//! This module provides reusable transformer block components that can be used
//! across different transformer architectures (standard, hierarchical, recurrent, etc.).
//!
//! The core component is the `TransformerBlock` which encapsulates:
//! - Pre-attention normalization
//! - Attention mechanism (PolyAttention with CoPE)
//! - Pre-feedforward normalization
//! - Feedforward network (RichardsGlu or MixtureOfExperts)
//! - Residual connections

pub mod transformer_block;

pub use transformer_block::{TransformerBlock, TransformerBlockConfig};

/// Re-export key types for convenience
pub use crate::attention::poly_attention::PolyAttention;
pub use crate::richards::{RichardsGlu, RichardsNorm};
pub use crate::mixtures::moe::MixtureOfExperts;
