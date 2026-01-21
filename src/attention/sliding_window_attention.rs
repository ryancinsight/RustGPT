use ndarray::{s, Array2, Axis};
use rand::distributions::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::network::Layer;

#[derive(Serialize, Deserialize, Debug)]
pub struct SlidingWindowAttention {
    pub embed_dim: usize,
    pub window_size: usize,
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
}

impl SlidingWindowAttention {
    pub fn new(embed_dim: usize, window_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-0.1, 0.1);

        let w_q = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));
        let w_k = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));
        let w_v = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));

        Self {
            embed_dim,
            window_size,
            w_q,
            w_k,
            w_v,
        }
    }
}

impl Layer for SlidingWindowAttention {
    fn layer_type(&self) -> &str {
        "SlidingWindowAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let seq_len = input.nrows();
        let mut output = Array2::<f32>::zeros((seq_len, self.embed_dim));

        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        for t in 0..seq_len {
            let start = t.saturating_sub(self.window_size - 1);
            let window_k = k.slice(s![start..=t, ..]);
            let window_v = v.slice(s![start..=t, ..]);

            let mut scores = q.row(t).dot(&window_k.t());
            scores.mapv_inplace(|x| (x / (self.embed_dim as f32).sqrt()).exp());
            let sum_scores = scores.sum();
            if sum_scores > 0.0 {
                scores.mapv_inplace(|x| x / sum_scores);
            }

            let weighted_v = scores.dot(&window_v);
            output.row_mut(t).assign(&weighted_v);
        }
        output
    }

    fn backward(&mut self, _grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        // TODO: Implement backpropagation
        Array2::zeros((_grads.nrows(), self.embed_dim))
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }
}
