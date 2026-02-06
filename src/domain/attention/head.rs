use ndarray::Array2;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

use crate::{infrastructure::optimizer::adam::Adam, common::rng::get_rng};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PolyHead {
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,

    opt_w_q: Adam,
    opt_w_k: Adam,
    opt_w_v: Adam,
}

impl PolyHead {
    pub fn new(embed_dim: usize, head_dim: usize) -> Self {
        let std_qk = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();
        let std_v = (2.0f32 / (embed_dim as f32 + head_dim as f32)).sqrt();

        let mut rng = get_rng();
        let normal_qk = Normal::new(0.0, std_qk).unwrap();
        let normal_v = Normal::new(0.0, std_v).unwrap();

        let w_q =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_qk.sample(&mut rng));
        let w_k =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_qk.sample(&mut rng));
        let w_v =
            Array2::<f32>::from_shape_fn((embed_dim, head_dim), |_| normal_v.sample(&mut rng));

        let opt_w_q = Adam::new((embed_dim, head_dim));
        let opt_w_k = Adam::new((embed_dim, head_dim));
        let opt_w_v = Adam::new((embed_dim, head_dim));

        Self {
            w_q,
            w_k,
            w_v,
            opt_w_q,
            opt_w_k,
            opt_w_v,
        }
    }

    /// Get mutable reference to Q weight optimizer
    pub fn opt_w_q_mut(&mut self) -> &mut Adam {
        &mut self.opt_w_q
    }

    /// Get mutable reference to K weight optimizer
    pub fn opt_w_k_mut(&mut self) -> &mut Adam {
        &mut self.opt_w_k
    }

    /// Get mutable reference to V weight optimizer
    pub fn opt_w_v_mut(&mut self) -> &mut Adam {
        &mut self.opt_w_v
    }

    /// Apply gradient step to Q weights
    pub fn step_w_q(&mut self, grad: &Array2<f32>, lr: f32) {
        self.opt_w_q.step(&mut self.w_q, grad, lr);
    }

    /// Apply gradient step to K weights
    pub fn step_w_k(&mut self, grad: &Array2<f32>, lr: f32) {
        self.opt_w_k.step(&mut self.w_k, grad, lr);
    }

    /// Apply gradient step to V weights
    pub fn step_w_v(&mut self, grad: &Array2<f32>, lr: f32) {
        self.opt_w_v.step(&mut self.w_v, grad, lr);
    }
}
