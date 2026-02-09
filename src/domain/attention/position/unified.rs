use crate::common::rng::get_rng;
use crate::infrastructure::optimizer::adam::Adam;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct UnifiedCoPE {
    pub max_pos: usize,
    pub embed_dim: usize,

    // Base CoPE (Legacy)
    pub pos_embeddings: Array2<f32>,
    pub optimizer: Adam,

    // PathCoPE params
    pub w_householder: Option<Array2<f32>>,
    pub opt_w_householder: Option<Adam>,
    pub u_beta: Option<Array2<f32>>,
    pub opt_u_beta: Option<Adam>,
    pub b_beta: Option<Array2<f32>>,
    pub opt_b_beta: Option<Adam>,

    // GatedCoPE params
    pub w_gate: Option<Array2<f32>>,
    pub opt_w_gate: Option<Adam>,
    pub b_gate: Option<Array2<f32>>,
    pub opt_b_gate: Option<Adam>,

    // Hierarchical params
    pub local_cope: Option<Array2<f32>>,
    pub opt_local_cope: Option<Adam>,
    pub global_cope: Option<Array2<f32>>,
    pub opt_global_cope: Option<Adam>,
    pub chunk_predictor_w: Option<Array2<f32>>,
    pub opt_chunk_predictor_w: Option<Adam>,
    pub chunk_predictor_b: Option<Array2<f32>>,
    pub opt_chunk_predictor_b: Option<Adam>,

    // Mixing weights
    pub alpha_local: f32,
    pub alpha_global: f32,
    pub alpha_path: f32,
    pub alpha_cope: f32,
}

pub struct UnifiedCoPEGradients {
    pub pos_embeddings: Option<Array2<f32>>,
    pub w_householder: Option<Array2<f32>>,
    pub u_beta: Option<Array2<f32>>,
    pub b_beta: Option<Array2<f32>>,
    pub w_gate: Option<Array2<f32>>,
    pub b_gate: Option<Array2<f32>>,
    pub local_cope: Option<Array2<f32>>,
    pub global_cope: Option<Array2<f32>>,
    pub chunk_predictor_w: Option<Array2<f32>>,
    pub chunk_predictor_b: Option<Array2<f32>>,
}

impl UnifiedCoPE {
    pub fn new(max_pos: usize, embed_dim: usize) -> Self {
        let mut rng = get_rng();
        let normal = Normal::new(0.0, 0.02).unwrap();

        let pos_embeddings =
            Array2::from_shape_fn((max_pos + 1, embed_dim), |_| normal.sample(&mut rng));
        let optimizer = Adam::new((max_pos + 1, embed_dim));

        Self {
            max_pos,
            embed_dim,
            pos_embeddings,
            optimizer,
            w_householder: None,
            opt_w_householder: None,
            u_beta: None,
            opt_u_beta: None,
            b_beta: None,
            opt_b_beta: None,
            w_gate: None,
            opt_w_gate: None,
            b_gate: None,
            opt_b_gate: None,
            local_cope: None,
            opt_local_cope: None,
            global_cope: None,
            opt_global_cope: None,
            chunk_predictor_w: None,
            opt_chunk_predictor_w: None,
            chunk_predictor_b: None,
            opt_chunk_predictor_b: None,
            alpha_local: 1.0,
            alpha_global: 0.0,
            alpha_path: 0.0,
            alpha_cope: 1.0,
        }
    }

    pub fn init_gradients(&self) -> UnifiedCoPEGradients {
        UnifiedCoPEGradients {
            pos_embeddings: Some(Array2::zeros(self.pos_embeddings.raw_dim())),
            w_householder: self
                .w_householder
                .as_ref()
                .map(|w| Array2::zeros(w.raw_dim())),
            u_beta: self.u_beta.as_ref().map(|w| Array2::zeros(w.raw_dim())),
            b_beta: self.b_beta.as_ref().map(|w| Array2::zeros(w.raw_dim())),
            w_gate: self.w_gate.as_ref().map(|w| Array2::zeros(w.raw_dim())),
            b_gate: self.b_gate.as_ref().map(|w| Array2::zeros(w.raw_dim())),
            local_cope: self.local_cope.as_ref().map(|w| Array2::zeros(w.raw_dim())),
            global_cope: self
                .global_cope
                .as_ref()
                .map(|w| Array2::zeros(w.raw_dim())),
            chunk_predictor_w: self
                .chunk_predictor_w
                .as_ref()
                .map(|w| Array2::zeros(w.raw_dim())),
            chunk_predictor_b: self
                .chunk_predictor_b
                .as_ref()
                .map(|w| Array2::zeros(w.raw_dim())),
        }
    }

    pub fn get_contribution(
        &self,
        q: &ArrayView1<'_, f32>,
        _k: &ArrayView1<'_, f32>,
        i: usize,
        j: usize,
        _inputs: Option<&ArrayView2<'_, f32>>,
    ) -> f32 {
        let pos = i.saturating_sub(j);
        if pos <= self.max_pos {
            q.dot(&self.pos_embeddings.row(pos))
        } else {
            0.0
        }
        // Note: PathCoPE and GatedCoPE logic would go here
    }

    pub fn backward(
        &self,
        q: &ArrayView1<'_, f32>,
        k: &ArrayView1<'_, f32>,
        i: usize,
        j: usize,
        _inputs: Option<&ArrayView2<'_, f32>>,
        d_s_ij: f32,
        grads: &mut UnifiedCoPEGradients,
    ) -> (Array1<f32>, Array1<f32>) {
        let pos = i.saturating_sub(j);
        let mut dq = Array1::zeros(q.dim());
        let dk = Array1::zeros(k.dim());

        if pos <= self.max_pos {
            // Legacy CoPE gradient: s += q dot P[pos]
            // dL/dq = d_s * P[pos]
            let p_emb = self.pos_embeddings.row(pos);
            Zip::from(&mut dq)
                .and(&p_emb)
                .for_each(|d, &p| *d += p * d_s_ij);

            // dL/dP[pos] = d_s * q
            if let Some(grad_pe) = &mut grads.pos_embeddings {
                let mut row = grad_pe.row_mut(pos);
                Zip::from(&mut row)
                    .and(q)
                    .for_each(|r, &q_val| *r += q_val * d_s_ij);
            }
        }

        // TODO: Add PathCoPE/GatedCoPE gradients

        (dq, dk)
    }

    pub fn apply_gradients(&mut self, grads: &Array2<f32>, lr: f32) {
        self.optimizer.step(&mut self.pos_embeddings, grads, lr);
    }

    pub fn apply_gradients_from_slice(&mut self, grads: &[Array2<f32>], lr: f32) {
        let mut idx = 0;

        // 1. Base
        if idx < grads.len() {
            self.optimizer
                .step(&mut self.pos_embeddings, &grads[idx], lr);
            idx += 1;
        }

        // 2. Path
        if let (Some(w), Some(opt)) = (&mut self.w_householder, &mut self.opt_w_householder) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.u_beta, &mut self.opt_u_beta) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.b_beta, &mut self.opt_b_beta) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }

        // 3. Gated
        if let (Some(w), Some(opt)) = (&mut self.w_gate, &mut self.opt_w_gate) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.b_gate, &mut self.opt_b_gate) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }

        // 4. Hierarchical
        if let (Some(w), Some(opt)) = (&mut self.local_cope, &mut self.opt_local_cope) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.global_cope, &mut self.opt_global_cope) {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.chunk_predictor_w, &mut self.opt_chunk_predictor_w)
        {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
        if let (Some(w), Some(opt)) = (&mut self.chunk_predictor_b, &mut self.opt_chunk_predictor_b)
        {
            if idx < grads.len() {
                opt.step(w, &grads[idx], lr);
                idx += 1;
            }
        }
    }

    /// Get the number of parameters in this UnifiedCoPE instance
    pub fn parameters(&self) -> usize {
        let mut count = self.pos_embeddings.len();
        if let Some(w) = &self.w_householder {
            count += w.len();
        }
        if let Some(w) = &self.u_beta {
            count += w.len();
        }
        if let Some(w) = &self.b_beta {
            count += w.len();
        }
        if let Some(w) = &self.w_gate {
            count += w.len();
        }
        if let Some(w) = &self.b_gate {
            count += w.len();
        }
        if let Some(w) = &self.local_cope {
            count += w.len();
        }
        if let Some(w) = &self.global_cope {
            count += w.len();
        }
        if let Some(w) = &self.chunk_predictor_w {
            count += w.len();
        }
        if let Some(w) = &self.chunk_predictor_b {
            count += w.len();
        }
        count
    }

    /// Get the weight norm (L2 norm) of all trainable parameters
    pub fn weight_norm(&self) -> f32 {
        let mut sum = self.pos_embeddings.iter().map(|&w| w * w).sum::<f32>();
        if let Some(w) = &self.w_householder {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.u_beta {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.b_beta {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.w_gate {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.b_gate {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.local_cope {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.global_cope {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.chunk_predictor_w {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        if let Some(w) = &self.chunk_predictor_b {
            sum += w.iter().map(|&w| w * w).sum::<f32>();
        }
        sum.sqrt()
    }
}

impl UnifiedCoPEGradients {
    pub fn accumulate(&mut self, other: &UnifiedCoPEGradients) {
        if let (Some(s), Some(o)) = (&mut self.pos_embeddings, &other.pos_embeddings) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.w_householder, &other.w_householder) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.u_beta, &other.u_beta) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.b_beta, &other.b_beta) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.w_gate, &other.w_gate) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.b_gate, &other.b_gate) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.local_cope, &other.local_cope) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.global_cope, &other.global_cope) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.chunk_predictor_w, &other.chunk_predictor_w) {
            *s += o;
        }
        if let (Some(s), Some(o)) = (&mut self.chunk_predictor_b, &other.chunk_predictor_b) {
            *s += o;
        }
    }

    pub fn to_vec(&self) -> Vec<Array2<f32>> {
        let mut v = Vec::new();
        if let Some(p) = &self.pos_embeddings {
            v.push(p.clone());
        }
        if let Some(p) = &self.w_householder {
            v.push(p.clone());
        }
        if let Some(p) = &self.u_beta {
            v.push(p.clone());
        }
        if let Some(p) = &self.b_beta {
            v.push(p.clone());
        }
        if let Some(p) = &self.w_gate {
            v.push(p.clone());
        }
        if let Some(p) = &self.b_gate {
            v.push(p.clone());
        }
        if let Some(p) = &self.local_cope {
            v.push(p.clone());
        }
        if let Some(p) = &self.global_cope {
            v.push(p.clone());
        }
        if let Some(p) = &self.chunk_predictor_w {
            v.push(p.clone());
        }
        if let Some(p) = &self.chunk_predictor_b {
            v.push(p.clone());
        }
        v
    }
}
