//! Neural network layers (sequence modeling blocks).
//!
//! This module groups the model's major layer families (transformer,
//! diffusion-conditioned, and SSM) under a single namespace with clear internal boundaries.

pub mod components;
pub mod diffusion;
pub mod output;
pub mod ssm;
pub mod transformer;
