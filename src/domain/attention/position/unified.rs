use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::config::{CoPEConfig, CoPEVariant};
use super::cope::CoPE;
use super::cope::CoPEGradients;
use super::factorized_cope::{FactorizedCoPE, FactorizedCoPEGradients};
use super::gated_cope::{GatedCoPE, GatedCoPEGradients};
use super::hierarchical_cope::{HierarchicalCoPE, HierarchicalCoPEGradients};
use super::optimized_cope::{OptimizedCoPE, OptimizedCoPEGradients};
use super::path_cope::{PathCoPE, PathCoPEGradients};
use super::traits::PositionEmbedding;
use super::window_aware_cope::WindowAwareCoPE;

/// Macro to reduce boilerplate by delegating method calls to CoPE variants.
/// This is a zero-cost abstraction that eliminates repetitive match statements.
macro_rules! delegate_to_cope_variant {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            UnifiedCoPE::Standard(c) => c.$method($($arg),*),
            UnifiedCoPE::Gated(c) => c.$method($($arg),*),
            UnifiedCoPE::Factorized(c) => c.$method($($arg),*),
            UnifiedCoPE::Hierarchical(c) => c.$method($($arg),*),
            UnifiedCoPE::Optimized(c) => c.$method($($arg),*),
            UnifiedCoPE::Path(c) => c.$method($($arg),*),
            UnifiedCoPE::WindowAware(c) => c.$method($($arg),*),
        }
    };
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum UnifiedCoPE {
    Standard(CoPE),
    Gated(GatedCoPE),
    Factorized(FactorizedCoPE),
    Hierarchical(HierarchicalCoPE),
    Optimized(OptimizedCoPE),
    Path(PathCoPE),
    WindowAware(Box<WindowAwareCoPE<UnifiedCoPE>>),
}

#[derive(Debug)]
pub enum UnifiedCoPEGradients {
    Standard(CoPEGradients),
    Gated(GatedCoPEGradients),
    Factorized(FactorizedCoPEGradients),
    Hierarchical(HierarchicalCoPEGradients),
    Optimized(OptimizedCoPEGradients),
    Path(PathCoPEGradients),
    WindowAware(Box<UnifiedCoPEGradients>),
}

impl UnifiedCoPEGradients {
    pub fn accumulate(&mut self, other: &Self) {
        match (self, other) {
            (UnifiedCoPEGradients::Standard(a), UnifiedCoPEGradients::Standard(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::Gated(a), UnifiedCoPEGradients::Gated(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::Factorized(a), UnifiedCoPEGradients::Factorized(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::Hierarchical(a), UnifiedCoPEGradients::Hierarchical(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::Optimized(a), UnifiedCoPEGradients::Optimized(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::Path(a), UnifiedCoPEGradients::Path(b)) => {
                a.accumulate(b);
            }
            (UnifiedCoPEGradients::WindowAware(a), UnifiedCoPEGradients::WindowAware(b)) => {
                a.accumulate(b);
            }
            _ => panic!("Gradient type mismatch in accumulate"),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            UnifiedCoPEGradients::Standard(g) => g.to_vec(),
            UnifiedCoPEGradients::Gated(g) => g.to_vec(),
            UnifiedCoPEGradients::Factorized(g) => g.to_vec(),
            UnifiedCoPEGradients::Hierarchical(g) => g.to_vec(),
            UnifiedCoPEGradients::Optimized(g) => g.to_vec(),
            UnifiedCoPEGradients::Path(g) => g.to_vec(),
            UnifiedCoPEGradients::WindowAware(g) => g.to_vec(),
        }
    }
}

impl UnifiedCoPE {
    pub fn from_config(config: CoPEConfig, embed_dim: usize) -> Self {
        let max_pos = config.max_pos;
        let inner = match config.variant {
            CoPEVariant::Standard => UnifiedCoPE::Standard(CoPE::new(max_pos, embed_dim)),
            CoPEVariant::Gated => UnifiedCoPE::Gated(GatedCoPE::new(max_pos, embed_dim)),
            CoPEVariant::Factorized { rank } => UnifiedCoPE::Factorized(FactorizedCoPE::new(max_pos, embed_dim, rank)),
            CoPEVariant::Hierarchical { num_chunks } => UnifiedCoPE::Hierarchical(HierarchicalCoPE::new(max_pos, embed_dim, num_chunks)),
            CoPEVariant::Optimized { rank } => UnifiedCoPE::Optimized(OptimizedCoPE::new(max_pos, embed_dim, rank)),
            CoPEVariant::Path => UnifiedCoPE::Path(PathCoPE::new(max_pos, embed_dim)),
        };

        if let Some(window_size) = config.window_size {
            UnifiedCoPE::WindowAware(Box::new(WindowAwareCoPE::new(inner, Some(window_size))))
        } else {
            inner
        }
    }

    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        // Default to Standard CoPE for backward compatibility in constructor
        UnifiedCoPE::Standard(CoPE::new(max_pos, embed_dim))
    }

    /// Helper for legacy code that expects direct access to embeddings
    /// Returns Some only if the underlying variant is Standard
    pub fn as_standard_embeddings(&self) -> Option<&ndarray::Array2<f32>> {
        match self {
            UnifiedCoPE::Standard(cope) => Some(&cope.pos_embeddings),
            _ => None,
        }
    }
}

impl PositionEmbedding for UnifiedCoPE {
    type Gradients = UnifiedCoPEGradients;

    fn max_pos(&self) -> usize {
        delegate_to_cope_variant!(self, max_pos)
    }

    fn contribution(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
    ) -> f32 {
        delegate_to_cope_variant!(self, contribution, q, k, query_pos, key_pos, inputs)
    }

    fn backward(
        &self,
        q: &ArrayView1<f32>,
        k: &ArrayView1<f32>,
        query_pos: usize,
        key_pos: usize,
        inputs: Option<&ArrayView2<f32>>,
        d_score: f32,
        grads: &mut Self::Gradients,
    ) -> (ndarray::Array1<f32>, ndarray::Array1<f32>) {
        match (self, grads) {
            (UnifiedCoPE::Standard(c), UnifiedCoPEGradients::Standard(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::Gated(c), UnifiedCoPEGradients::Gated(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::Factorized(c), UnifiedCoPEGradients::Factorized(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::Hierarchical(c), UnifiedCoPEGradients::Hierarchical(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::Optimized(c), UnifiedCoPEGradients::Optimized(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::Path(c), UnifiedCoPEGradients::Path(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            (UnifiedCoPE::WindowAware(c), UnifiedCoPEGradients::WindowAware(g)) => {
                c.backward(q, k, query_pos, key_pos, inputs, d_score, g)
            }
            _ => panic!("Gradient type mismatch in UnifiedCoPE::backward"),
        }
    }

    fn init_gradients(&self) -> Self::Gradients {
        match self {
            UnifiedCoPE::Standard(c) => UnifiedCoPEGradients::Standard(c.init_gradients()),
            UnifiedCoPE::Gated(c) => UnifiedCoPEGradients::Gated(c.init_gradients()),
            UnifiedCoPE::Factorized(c) => UnifiedCoPEGradients::Factorized(c.init_gradients()),
            UnifiedCoPE::Hierarchical(c) => UnifiedCoPEGradients::Hierarchical(c.init_gradients()),
            UnifiedCoPE::Optimized(c) => UnifiedCoPEGradients::Optimized(c.init_gradients()),
            UnifiedCoPE::Path(c) => UnifiedCoPEGradients::Path(c.init_gradients()),
            UnifiedCoPE::WindowAware(c) => UnifiedCoPEGradients::WindowAware(Box::new(c.init_gradients())),
        }
    }

    fn apply_gradients(&mut self, grads: &Self::Gradients, lr: f32) {
        match (self, grads) {
            (UnifiedCoPE::Standard(c), UnifiedCoPEGradients::Standard(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::Gated(c), UnifiedCoPEGradients::Gated(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::Factorized(c), UnifiedCoPEGradients::Factorized(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::Hierarchical(c), UnifiedCoPEGradients::Hierarchical(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::Optimized(c), UnifiedCoPEGradients::Optimized(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::Path(c), UnifiedCoPEGradients::Path(g)) => c.apply_gradients(g, lr),
            (UnifiedCoPE::WindowAware(c), UnifiedCoPEGradients::WindowAware(g)) => c.apply_gradients(g, lr),
            _ => panic!("Gradient type mismatch in UnifiedCoPE::apply_gradients"),
        }
    }

    fn embed_dim(&self) -> usize {
        delegate_to_cope_variant!(self, embed_dim)
    }

    fn parameters(&self) -> usize {
        delegate_to_cope_variant!(self, parameters)
    }

    fn weight_norm(&self) -> f32 {
        delegate_to_cope_variant!(self, weight_norm)
    }
}
