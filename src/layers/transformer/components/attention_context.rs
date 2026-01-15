//! Attention Context Component
//!
//! Manages attention context and similarity matrices for transformer blocks.
//! Handles incoming context application and similarity matrix updates.

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Attention context component
#[derive(Serialize, Deserialize, Debug)]
pub struct AttentionContext {
    /// Incoming similarity context from previous layer
    incoming_context: Option<Array2<f32>>,
    /// Current similarity context strength
    similarity_context_strength: Array2<f32>,
}

impl Default for AttentionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionContext {
    pub fn new() -> Self {
        Self {
            incoming_context: None,
            similarity_context_strength: Array2::zeros((1, 1)),
        }
    }

    /// Set incoming similarity context
    pub fn set_incoming_context(&mut self, context: Option<&Array2<f32>>) {
        if let Some(ctx) = context {
            // Validate context shape and set it
            self.incoming_context = Some(ctx.clone());
        } else {
            self.incoming_context = None;
        }
    }

    /// Get incoming similarity context
    pub fn get_incoming_context(&self) -> Option<&Array2<f32>> {
        self.incoming_context.as_ref()
    }

    /// Set similarity context strength
    pub fn set_strength(&mut self, strength: f32) {
        self.similarity_context_strength[[0, 0]] = strength;
    }

    /// Get similarity context strength
    pub fn get_strength(&self) -> f32 {
        self.similarity_context_strength[[0, 0]]
    }

    /// Apply similarity context to input
    pub fn apply_context(&self, input: &Array2<f32>) -> Array2<f32> {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            if strength == 0.0 || embed_dim == 0 {
                return input.clone();
            }

            let scale = strength / embed_dim as f32;

            if input.ncols() != context.nrows() || context.nrows() != context.ncols() {
                return input.clone();
            }

            let mut out = input.dot(context);
            out.zip_mut_with(input, |o, &x| {
                let ms = if o.is_finite() { *o } else { 0.0 };
                let xs = if x.is_finite() { x } else { 0.0 };
                *o = xs + scale * ms;
            });
            out
        } else {
            input.clone()
        }
    }

    /// Clear the incoming context
    pub fn clear_context(&mut self) {
        self.incoming_context = None;
    }

    /// Check if context is available
    pub fn has_context(&self) -> bool {
        self.incoming_context.is_some()
    }
}
