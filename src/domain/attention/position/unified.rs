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
                a.base_grads.accumulate(&b.base_grads);
                if let (Some(ga), Some(gb)) = (&mut a.w_gate_grads, &b.w_gate_grads) {
                    *ga += gb;
                }
                if let (Some(ba), Some(bb)) = (&mut a.b_gate_grads, &b.b_gate_grads) {
                    *ba += bb;
                }
            }
            (UnifiedCoPEGradients::Factorized(a), UnifiedCoPEGradients::Factorized(b)) => {
                if let (Some(qa), Some(qb)) = (&mut a.up_proj_grads, &b.up_proj_grads) {
                    *qa += qb;
                }
                if let (Some(ka), Some(kb)) = (&mut a.down_proj_grads, &b.down_proj_grads) {
                    *ka += kb;
                }
            }
            (UnifiedCoPEGradients::Hierarchical(a), UnifiedCoPEGradients::Hierarchical(b)) => {
                if let (Some(la), Some(lb)) = (&mut a.local_cope_grads, &b.local_cope_grads) {
                    *la += lb;
                }
                if let (Some(ga), Some(gb)) = (&mut a.global_cope_grads, &b.global_cope_grads) {
                    *ga += gb;
                }
            }
            (UnifiedCoPEGradients::Optimized(a), UnifiedCoPEGradients::Optimized(b)) => {
                if let (Some(ua), Some(ub)) = (&mut a.up_proj_grads, &b.up_proj_grads) {
                    *ua += ub;
                }
                if let (Some(da), Some(db)) = (&mut a.down_proj_grads, &b.down_proj_grads) {
                    *da += db;
                }
                if let (Some(wa), Some(wb)) = (&mut a.w_gate_grads, &b.w_gate_grads) {
                    *wa += wb;
                }
                if let (Some(ba), Some(bb)) = (&mut a.b_gate_grads, &b.b_gate_grads) {
                    *ba += bb;
                }
            }
            (UnifiedCoPEGradients::Path(a), UnifiedCoPEGradients::Path(b)) => {
                if let (Some(wa), Some(wb)) = (&mut a.w_householder_grads, &b.w_householder_grads) {
                    *wa += wb;
                }
                if let (Some(ua), Some(ub)) = (&mut a.u_beta_grads, &b.u_beta_grads) {
                    *ua += ub;
                }
                if let (Some(ba), Some(bb)) = (&mut a.b_beta_grads, &b.b_beta_grads) {
                    *ba += bb;
                }
                if let (Some(ca), Some(cb)) = (&mut a.base_cope_grads, &b.base_cope_grads) {
                    *ca += cb;
                }
                // These are f32, not Option<f32>
                a.alpha_path_grad += b.alpha_path_grad;
                a.alpha_cope_grad += b.alpha_cope_grad;
            }
            (UnifiedCoPEGradients::WindowAware(a), UnifiedCoPEGradients::WindowAware(b)) => {
                 a.accumulate(b);
            }
            _ => panic!("Gradient type mismatch in accumulate"),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            UnifiedCoPEGradients::Standard(g) => {
                g.pos_embeddings.as_ref().map(|x| x.iter().cloned().collect()).unwrap_or_default()
            }
            UnifiedCoPEGradients::Gated(g) => {
                let mut v = Vec::new();
                v.extend(g.base_grads.pos_embeddings.as_ref().map(|x| x.iter().cloned().collect::<Vec<_>>()).unwrap_or_default());
                if let Some(w) = &g.w_gate_grads { v.extend(w.iter()); }
                if let Some(b) = &g.b_gate_grads { v.extend(b.iter()); }
                v
            }
            UnifiedCoPEGradients::Factorized(g) => {
                let mut v = Vec::new();
                if let Some(q) = &g.up_proj_grads { v.extend(q.iter()); }
                if let Some(k) = &g.down_proj_grads { v.extend(k.iter()); }
                v
            }
            UnifiedCoPEGradients::Hierarchical(g) => {
                let mut v = Vec::new();
                if let Some(l) = &g.local_cope_grads { v.extend(l.iter()); }
                if let Some(gl) = &g.global_cope_grads { v.extend(gl.iter()); }
                if let Some(cw) = &g.chunk_predictor_w_grads { v.extend(cw.iter()); }
                if let Some(cb) = &g.chunk_predictor_b_grads { v.extend(cb.iter()); }
                v
            }
            UnifiedCoPEGradients::Optimized(g) => {
                let mut v = Vec::new();
                if let Some(u) = &g.up_proj_grads { v.extend(u.iter()); }
                if let Some(d) = &g.down_proj_grads { v.extend(d.iter()); }
                if let Some(w) = &g.w_gate_grads { v.extend(w.iter()); }
                if let Some(b) = &g.b_gate_grads { v.extend(b.iter()); }
                v
            }
            UnifiedCoPEGradients::Path(g) => {
                let mut v = Vec::new();
                if let Some(h) = &g.w_householder_grads { v.extend(h.iter()); }
                if let Some(u) = &g.u_beta_grads { v.extend(u.iter()); }
                if let Some(b) = &g.b_beta_grads { v.extend(b.iter()); }
                if let Some(base) = &g.base_cope_grads { v.extend(base.iter()); }
                v.push(g.alpha_path_grad);
                v.push(g.alpha_cope_grad);
                v
            }
            UnifiedCoPEGradients::WindowAware(g) => {
                 g.to_vec()
            }
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
