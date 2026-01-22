use ndarray::{s, Array1, Array2, Axis};
use rand::distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};
use std::ops::AddAssign;

use crate::network::Layer;

#[derive(Debug, Clone)]
struct AttentionCache {
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    attention_scores: Vec<Array1<f32>>,
    input: Array2<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SlidingWindowAttention {
    pub embed_dim: usize,
    pub window_size: usize,
    pub w_q: Array2<f32>,
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    #[serde(skip)]
    cache: Option<AttentionCache>,
}

impl SlidingWindowAttention {
    pub fn new(embed_dim: usize, window_size: usize) -> Self {
        let mut rng = rand::rng();
        let uniform = Uniform::new(-0.1, 0.1).unwrap();

        let w_q = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));
        let w_k = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));
        let w_v = Array2::from_shape_fn((embed_dim, embed_dim), |_| uniform.sample(&mut rng));

        Self {
            embed_dim,
            window_size,
            w_q,
            w_k,
            w_v,
            cache: None,
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

        let mut attention_scores = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let start = t.saturating_sub(self.window_size - 1);
            let window_k = k.slice(s![start..=t, ..]);
            let window_v = v.slice(s![start..=t, ..]);

            let mut scores = q.row(t).dot(&window_k.t());
            let scale = (self.embed_dim as f32).sqrt();
            scores.mapv_inplace(|x| (x / scale).exp());
            let sum_scores = scores.sum();
            if sum_scores > 0.0 {
                scores.mapv_inplace(|x| x / sum_scores);
            }
            attention_scores.push(scores.clone());

            let weighted_v = scores.dot(&window_v);
            output.row_mut(t).assign(&weighted_v);
        }

        self.cache = Some(AttentionCache {
            q: q.clone(),
            k: k.clone(),
            v: v.clone(),
            attention_scores,
            input: input.clone(),
        });

        output
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let (input_grads, param_grads) = self.compute_gradients(&Array2::zeros((0, 0)), grads);
        self.apply_gradients(&param_grads, lr).unwrap();
        input_grads
    }

    fn parameters(&self) -> usize {
        self.w_q.len() + self.w_k.len() + self.w_v.len()
    }

    fn compute_gradients(
        &self,
        _input: &Array2<f32>,
        output_grads: &Array2<f32>,
    ) -> (Array2<f32>, Vec<Array2<f32>>) {
        let cache = self.cache.as_ref().expect("Cache should be present before backward pass");
        let seq_len = cache.input.nrows();
        let scale = (self.embed_dim as f32).sqrt();

        let mut grad_q = Array2::zeros(cache.q.raw_dim());
        let mut grad_k = Array2::zeros(cache.k.raw_dim());
        let mut grad_v = Array2::zeros(cache.v.raw_dim());

        for t in (0..seq_len).rev() {
            let start = t.saturating_sub(self.window_size - 1);
            let d_output_t = output_grads.row(t);

            let scores_t = &cache.attention_scores[t];
            let window_v_t = cache.v.slice(s![start..=t, ..]);
            let window_k_t = cache.k.slice(s![start..=t, ..]);
            let q_t = cache.q.row(t);

            // Backprop through weighted sum of V
            let d_scores_t = d_output_t.dot(&window_v_t.t());
            let d_window_v = scores_t.clone().insert_axis(Axis(1)).dot(&d_output_t.insert_axis(Axis(0)));
            grad_v.slice_mut(s![start..=t, ..]).add_assign(&d_window_v);

            // Backprop through softmax
            let d_s_dot_s = (&d_scores_t * scores_t).sum();
            let d_z_t = scores_t * (&d_scores_t - d_s_dot_s);
            let d_raw_scores_t = d_z_t / scale;

            // Backprop through QK dot product
            let d_q_t = d_raw_scores_t.dot(&window_k_t);
            let d_window_k = d_raw_scores_t.insert_axis(Axis(1)).dot(&q_t.insert_axis(Axis(0)));
            grad_q.row_mut(t).add_assign(&d_q_t);
            grad_k.slice_mut(s![start..=t, ..]).add_assign(&d_window_k);
        }

        // Gradients for weights
        let grad_w_q = cache.input.t().dot(&grad_q);
        let grad_w_k = cache.input.t().dot(&grad_k);
        let grad_w_v = cache.input.t().dot(&grad_v);

        // Gradients for input
        let d_input_from_q = grad_q.dot(&self.w_q.t());
        let d_input_from_k = grad_k.dot(&self.w_k.t());
        let d_input_from_v = grad_v.dot(&self.w_v.t());

        let input_grads = d_input_from_q + d_input_from_k + d_input_from_v;

        (input_grads, vec![grad_w_q, grad_w_k, grad_w_v])
    }

    fn apply_gradients(
        &mut self,
        gradients: &[Array2<f32>],
        learning_rate: f32,
    ) -> crate::errors::Result<()> {
        if gradients.len() != 3 {
             return Err(crate::errors::ModelError::GradientError {
                message: format!("Expected 3 gradients for SlidingWindowAttention, got {}", gradients.len()),
            });
        }

        self.w_q.scaled_add(-learning_rate, &gradients[0]);
        self.w_k.scaled_add(-learning_rate, &gradients[1]);
        self.w_v.scaled_add(-learning_rate, &gradients[2]);
        Ok(())
    }

    fn weight_norm(&self) -> f32 {
        let mut sum = 0.0;
        sum += self.w_q.iter().map(|x| x * x).sum::<f32>();
        sum += self.w_k.iter().map(|x| x * x).sum::<f32>();
        sum += self.w_v.iter().map(|x| x * x).sum::<f32>();
        sum.sqrt()
    }

    fn zero_gradients(&mut self) {
        // No stateful gradients to zero
    }
}
