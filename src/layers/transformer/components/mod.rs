//! Transformer components module
//!
//! This module provides focused, modular components for transformer architecture.
//! Each component has a single responsibility and clear interface.

pub mod attention_context;
pub mod eprop_adaptor;
pub mod feedforward_processor;
pub mod normalization_layer;
pub mod residual_connection;
pub mod temporal_mixing_wrapper;
pub mod window_adaptation;
