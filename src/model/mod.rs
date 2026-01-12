// Model-related functionality grouped under a single namespace.
//
// This module intentionally re-exports the existing top-level modules to avoid
// breaking internal paths while providing a cohesive API surface:
// - llm::model::builder::{...}
// - llm::model::config::{...}
//
// Persistence is implemented as inherent methods on `LLM` and is kept internal.

pub use crate::{model_builder as builder, model_config as config};
