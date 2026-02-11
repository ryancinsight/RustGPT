//! Shared Attention Context Component
//!
//! This component provides attention context management that can be used
//! by multiple architectures (Transformer, Diffusion).
//! It encapsulates the logic for applying similarity-based context modulation.

use ndarray::{Array1, Array2, Zip};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// Shared attention context component
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SharedAttentionContext {
    /// Incoming similarity context from previous layer
    #[serde(skip)]
    pub incoming_context: Option<Array2<f32>>,
    /// Current similarity context strength
    pub similarity_context_strength: Array2<f32>,
}

impl Default for SharedAttentionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedAttentionContext {
    /// Create a new shared attention context component
    pub fn new() -> Self {
        Self {
            incoming_context: None,
            similarity_context_strength: Array2::zeros((1, 1)),
        }
    }

    /// Set incoming similarity context
    pub fn set_incoming_context(&mut self, context: Option<&Array2<f32>>) {
        if let Some(ctx) = context {
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
        if self.similarity_context_strength.len() != 1 {
            self.similarity_context_strength = Array2::zeros((1, 1));
        }
        self.similarity_context_strength[[0, 0]] = strength;
    }

    /// Get similarity context strength
    pub fn get_strength(&self) -> f32 {
        self.similarity_context_strength.get((0, 0)).copied().unwrap_or(0.0)
    }

    /// Check if context is available
    pub fn has_context(&self) -> bool {
        self.incoming_context.is_some()
    }

    /// Clear the incoming context
    pub fn clear_context(&mut self) {
        self.incoming_context = None;
    }

    /// Get parameter count (1 scalar for strength)
    pub fn parameters(&self) -> usize {
        1
    }

    /// Get L2 norm of parameters
    pub fn weight_norm(&self) -> f32 {
        self.get_strength().abs()
    }

    /// Apply similarity context to input (Batch/Sequence Mode)
    /// 
    /// Computes: Output = Input + (Strength / EmbedDim) * (Input · Context)
    /// Returns Cow::Borrowed if no context is applied, or Cow::Owned if transformed.
    pub fn apply_context<'a>(&self, input: &'a Array2<f32>) -> Cow<'a, Array2<f32>> {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            let embed_dim = input.ncols();

            if strength == 0.0 || embed_dim == 0 {
                return Cow::Borrowed(input);
            }

            // Expect embed_dim × embed_dim context.
            if input.ncols() != context.nrows() || context.nrows() != context.ncols() {
                return Cow::Borrowed(input);
            }

            let scale = strength / (embed_dim as f32).max(1.0);
            
            // Optimized matrix multiplication: Out = Input · Context
            let mut out = input.dot(context);
            
            // Mix: Out = Input + Scale * Out
            // Using Zip for efficient element-wise operation
            Zip::from(&mut out)
                .and(input)
                .for_each(|o, &i| {
                    let ms = if o.is_finite() { *o } else { 0.0 };
                    let xs = if i.is_finite() { i } else { 0.0 };
                    *o = xs + scale * ms;
                });
            Cow::Owned(out)
        } else {
            Cow::Borrowed(input)
        }
    }

    /// Compute gradients for similarity context
    /// 
    /// Returns (final_input_grads, similarity_strength_grad)
    pub fn compute_gradients(
        &self,
        input_original: &Array2<f32>,
        final_input_used_grads: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        let mut similarity_strength_grad = Array2::zeros((1, 1));
        let mut final_input_grads = final_input_used_grads.clone();

        if let Some(ctx) = &self.incoming_context {
            let embed_dim = input_original.ncols();
            if ctx.nrows() == embed_dim && ctx.ncols() == embed_dim {
                let d = (embed_dim.max(1)) as f32;
                
                // 1. Gradient for learnable similarity_context_strength.
                // dL/ds = (1/d) * sum(dX' ⊙ (X·S))
                let mixed = input_original.dot(ctx);
                let mut acc = 0.0f64;
                Zip::from(final_input_used_grads)
                    .and(&mixed)
                    .for_each(|&g, &m| {
                        let gs = if g.is_finite() { g as f64 } else { 0.0 };
                        let ms = if m.is_finite() { m as f64 } else { 0.0 };
                        acc += gs * ms;
                    });
                similarity_strength_grad[[0, 0]] = (acc as f32) / d;

                // 2. Backprop through similarity-context mixing for upstream gradient.
                // dX = dX' + k * dX'·S^T
                let s = self.get_strength();
                let s = if s.is_finite() { s } else { 0.0 };
                let k = s / d;
                
                if k != 0.0 {
                    let corr = final_input_grads.dot(&ctx.t());
                    Zip::from(&mut final_input_grads)
                        .and(&corr)
                        .for_each(|g, &c| {
                            let cs = if c.is_finite() { c } else { 0.0 };
                            *g += k * cs;
                        });
                }
            }
        }
        
        (final_input_grads, similarity_strength_grad)
    }

    /// Apply similarity context to input (Step Mode)
    ///
    /// Computes in-place: Output = Input + (Strength / EmbedDim) * (Context^T · Input)
    /// Note: For vector-matrix product, we use Context^T · Input equivalent to Input · Context
    pub fn apply_step_into(
        &self,
        input: &ndarray::ArrayView1<f32>,
        output: &mut ndarray::Array1<f32>,
    ) {
        if let Some(context) = &self.incoming_context {
            let strength = self.get_strength();
            
            if strength == 0.0 {
                output.assign(input);
                return;
            }

            let embed_dim = input.len();
            if embed_dim != context.nrows() || context.nrows() != context.ncols() {
                output.assign(input);
                return;
            }

            let scale = strength / (embed_dim as f32).max(1.0);

            // Step 1: y = scale * context^T * input
            // We use general_mat_vec_mul which computes y = alpha * A * x + beta * y
            // Here: output = scale * context^T * input
            ndarray::linalg::general_mat_vec_mul(scale, &context.t(), input, 0.0, output);
            
            // Step 2: output += input
            Zip::from(output)
                .and(input)
                .for_each(|o, &i| *o += i);
        } else {
            output.assign(input);
        }
    }
}
