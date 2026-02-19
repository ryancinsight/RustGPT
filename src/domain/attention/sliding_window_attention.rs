use std::ops::AddAssign;

use ndarray::{Array1, Array2, Axis, Zip, s};
use rand::distr::{Distribution, Uniform};
use serde::{Deserialize, Serialize};

use crate::{
    common::errors::Result,
    domain::layers::components::workspace_managed::{
        StreamingWorkspaceManaged, WorkspaceManaged, WorkspaceStats,
    },
    domain::network::Layer,
};

#[derive(Debug, Clone)]
struct AttentionCache {
    q: Array2<f32>,
    k: Array2<f32>,
    v: Array2<f32>,
    /// Stores attention scores for each time step.
    /// Shape: (seq_len, window_size)
    /// Valid elements for step t are in 0..current_window_len
    attention_scores: Array2<f32>,
    input: Array2<f32>,
}

/// Optimized sliding window cache with pre-sized buffers.
///
/// This implementation pre-allocates all buffers to their maximum expected dimensions,
/// eliminating allocation checks in the hot path of streaming inference.
#[derive(Debug, Clone)]
pub struct SlidingWindowCache {
    pub k_cache: Array2<f32>, // (window_size, embed_dim)
    pub v_cache: Array2<f32>, // (window_size, embed_dim)
    pub step: usize,
    pub titan_memory_state: Option<Array1<f32>>,
    /// Cached dimensions for fast validation
    cached_window_size: usize,
    cached_embed_dim: usize,
}

impl SlidingWindowCache {
    /// Create a new cache with pre-sized buffers.
    #[inline]
    pub fn new(window_size: usize, embed_dim: usize) -> Self {
        Self {
            k_cache: Array2::zeros((window_size, embed_dim)),
            v_cache: Array2::zeros((window_size, embed_dim)),
            step: 0,
            titan_memory_state: None,
            cached_window_size: window_size,
            cached_embed_dim: embed_dim,
        }
    }

    /// Reset the cache without deallocating buffers.
    /// Uses fill(0.0) for cache invalidation - faster than reallocation.
    #[inline]
    pub fn reset(&mut self) {
        self.k_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.step = 0;
        self.titan_memory_state = None;
    }

    /// Clear the cache (alias for reset for consistency with workspace trait)
    #[inline]
    pub fn clear(&mut self) {
        self.reset();
    }

    /// Check if cache dimensions match expected (for validation).
    #[inline]
    pub fn is_compatible(&self, window_size: usize, embed_dim: usize) -> bool {
        self.cached_window_size == window_size && self.cached_embed_dim == embed_dim
    }

    /// Get the current valid range for streaming access.
    /// Returns (start_index, valid_count) for the circular buffer.
    #[inline]
    pub fn valid_range(&self) -> (usize, usize) {
        let valid_count = self.step.min(self.cached_window_size);
        let start = if self.step >= self.cached_window_size {
            self.step % self.cached_window_size
        } else {
            0
        };
        (start, valid_count)
    }

    /// Get pre-computed capacity info.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cached_window_size
    }

    /// Get current fill level.
    #[inline]
    pub fn fill_level(&self) -> usize {
        self.step.min(self.cached_window_size)
    }
}

pub struct SlidingWindowStreamingWorkspace {
    pub q: Array1<f32>,
    pub k: Array1<f32>,
    pub v: Array1<f32>,
    pub scores: Array1<f32>,
    pub output: Array1<f32>,
}

impl SlidingWindowStreamingWorkspace {
    pub fn new(embed_dim: usize, window_size: usize) -> Self {
        Self {
            q: Array1::zeros(embed_dim),
            k: Array1::zeros(embed_dim),
            v: Array1::zeros(embed_dim),
            scores: Array1::zeros(window_size),
            output: Array1::zeros(embed_dim),
        }
    }
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
    #[serde(skip)]
    pub streaming_cache: Option<SlidingWindowCache>,
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
            streaming_cache: None,
        }
    }

    /// Process a single token step (Streaming/Rolling mode)
    pub fn forward_step(&mut self, input: &Array1<f32>) -> Array1<f32> {
        if self.streaming_cache.is_none() {
            self.streaming_cache = Some(SlidingWindowCache::new(self.window_size, self.embed_dim));
        }
        let cache = self.streaming_cache.as_mut().unwrap();

        // 1. Project
        // Note: Batch forward uses input.dot(&weights), which is x W.
        // We must match this behavior. input is treated as row vector.
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        // 2. Update Cache (Ring Buffer)
        let idx = cache.step % self.window_size;
        cache.k_cache.row_mut(idx).assign(&k);
        cache.v_cache.row_mut(idx).assign(&v);

        cache.step += 1;

        // 3. Compute Attention
        // Since there are no positional embeddings, order in the buffer doesn't matter
        // for the weighted sum (permutation invariant).
        // We just need to take the valid entries.

        let valid_rows = if cache.step <= self.window_size {
            // Cache not full yet
            s![0..cache.step, ..]
        } else {
            // Cache full, use all rows
            s![.., ..]
        };

        let k_window = cache.k_cache.slice(valid_rows);
        let v_window = cache.v_cache.slice(valid_rows);

        let scale = (self.embed_dim as f32).sqrt();
        let mut scores = k_window.dot(&q); // (current_len,)

        scores.mapv_inplace(|x| (x / scale).exp());
        let sum_scores = scores.sum();
        if sum_scores > 0.0 {
            scores.mapv_inplace(|x| x / sum_scores);
        }

        let output = scores.dot(&v_window);
        output
    }

    /// Process a single token step using a workspace to minimize allocations.
    /// Returns a view into the workspace output buffer - caller must clone if needed.
    #[inline]
    pub fn forward_step_with_workspace<'a>(
        &mut self,
        input: &Array1<f32>,
        ws: &'a mut SlidingWindowStreamingWorkspace,
    ) -> ndarray::ArrayView1<'a, f32> {
        if self.streaming_cache.is_none() {
            self.streaming_cache = Some(SlidingWindowCache::new(self.window_size, self.embed_dim));
        }
        let cache = self.streaming_cache.as_mut().unwrap();

        // 1. Project into workspace
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_q.t(), input, 0.0, &mut ws.q);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_k.t(), input, 0.0, &mut ws.k);
        ndarray::linalg::general_mat_vec_mul(1.0, &self.w_v.t(), input, 0.0, &mut ws.v);

        // 2. Update Cache
        let idx = cache.step % self.window_size;
        cache.k_cache.row_mut(idx).assign(&ws.k);
        cache.v_cache.row_mut(idx).assign(&ws.v);

        cache.step += 1;

        // 3. Compute Attention
        let valid_count = if cache.step <= self.window_size {
            cache.step
        } else {
            self.window_size
        };

        let mut scores_view = ws.scores.slice_mut(s![0..valid_count]);
        let k_window = cache.k_cache.slice(s![0..valid_count, ..]);

        // scores = k_window * q
        ndarray::linalg::general_mat_vec_mul(1.0, &k_window, &ws.q, 0.0, &mut scores_view);

        let scale = (self.embed_dim as f32).sqrt();
        scores_view.mapv_inplace(|x: f32| (x / scale).exp());
        let sum_scores = scores_view.sum();
        if sum_scores > 0.0 {
            scores_view.mapv_inplace(|x: f32| x / sum_scores);
        }

        // output = scores * v_window = v_window.t() * scores
        let v_window = cache.v_cache.slice(s![0..valid_count, ..]);
        ndarray::linalg::general_mat_vec_mul(1.0, &v_window.t(), &scores_view, 0.0, &mut ws.output);

        ws.output.view()
    }

    /// Process a single token step using a workspace, writing output into provided buffer.
    /// This is the true zero-allocation variant for hot paths.
    #[inline]
    pub fn forward_step_with_workspace_into(
        &mut self,
        input: &Array1<f32>,
        ws: &mut SlidingWindowStreamingWorkspace,
        output: &mut Array1<f32>,
    ) {
        let result = self.forward_step_with_workspace(input, ws);
        output.assign(&result);
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

        let mut attention_scores = Array2::zeros((seq_len, self.window_size));
        let window_size = self.window_size;
        let scale = (self.embed_dim as f32).sqrt();

        // Create views for parallel access
        let q_view = q.view();
        let k_view = k.view();
        let v_view = v.view();

        Zip::indexed(output.rows_mut())
            .and(attention_scores.rows_mut())
            .par_for_each(|t, mut out_row, mut score_row| {
                let start = t.saturating_sub(window_size - 1);
                let current_window_len = t - start + 1;

                let window_k = k_view.slice(s![start..=t, ..]);
                let window_v = v_view.slice(s![start..=t, ..]);

                let mut scores = q_view.row(t).dot(&window_k.t());

                scores.mapv_inplace(|x| (x / scale).exp());
                let sum_scores = scores.sum();
                if sum_scores > 0.0 {
                    scores.mapv_inplace(|x| x / sum_scores);
                }

                score_row
                    .slice_mut(s![..current_window_len])
                    .assign(&scores);

                let weighted_v = scores.dot(&window_v);
                out_row.assign(&weighted_v);
            });

        self.cache = Some(AttentionCache {
            q,
            k,
            v,
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
        let cache = self
            .cache
            .as_ref()
            .expect("Cache should be present before backward pass");
        let seq_len = cache.input.nrows();
        let scale = (self.embed_dim as f32).sqrt();

        let mut grad_q = Array2::zeros(cache.q.raw_dim());
        let mut grad_k = Array2::zeros(cache.k.raw_dim());
        let mut grad_v = Array2::zeros(cache.v.raw_dim());

        for t in (0..seq_len).rev() {
            let start = t.saturating_sub(self.window_size - 1);
            let current_window_len = t - start + 1;
            let d_output_t = output_grads.row(t);

            let row_view = cache.attention_scores.row(t);
            let scores_t = row_view.slice(s![..current_window_len]);

            let window_v_t = cache.v.slice(s![start..=t, ..]);
            let window_k_t = cache.k.slice(s![start..=t, ..]);
            let q_t = cache.q.row(t);

            // Backprop through weighted sum of V
            let d_scores_t = d_output_t.dot(&window_v_t.t());
            let d_window_v = scores_t
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&d_output_t.insert_axis(Axis(0)));
            grad_v.slice_mut(s![start..=t, ..]).add_assign(&d_window_v);

            // Backprop through softmax
            let d_s_dot_s = (&d_scores_t * &scores_t).sum();
            let d_z_t = &scores_t * (&d_scores_t - d_s_dot_s);
            let d_raw_scores_t = d_z_t / scale;

            // Backprop through QK dot product
            let d_q_t = d_raw_scores_t.dot(&window_k_t);
            let d_window_k = d_raw_scores_t
                .insert_axis(Axis(1))
                .dot(&q_t.insert_axis(Axis(0)));
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
    ) -> crate::common::errors::Result<()> {
        if gradients.len() != 3 {
            return Err(crate::common::errors::ModelError::GradientError {
                message: format!(
                    "Expected 3 gradients for SlidingWindowAttention, got {}",
                    gradients.len()
                ),
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

/// Workspace management for SlidingWindowAttention streaming inference
impl WorkspaceManaged for SlidingWindowAttention {
    /// Ensure workspace buffers have capacity for the given dimensions
    fn ensure_capacity(&mut self, _batch_size: usize, _seq_len: usize, embed_dim: usize) {
        if let Some(cache) = &mut self.streaming_cache {
            // Reallocate only if dimensions changed
            if !cache.is_compatible(self.window_size, embed_dim) {
                *cache = SlidingWindowCache::new(self.window_size, embed_dim);
            }
        }
    }

    /// Clear all workspace buffers
    fn clear_workspace(&mut self) {
        self.streaming_cache = None;
        self.cache = None;
    }

    /// Return memory statistics for streaming workspace
    fn workspace_stats(&self) -> WorkspaceStats {
        let mut buffer_count = 0;
        let mut total_bytes = 0;

        if let Some(cache) = &self.streaming_cache {
            // k_cache and v_cache
            buffer_count = 2;
            let cache_bytes =
                cache.cached_window_size * cache.cached_embed_dim * std::mem::size_of::<f32>();
            total_bytes = cache_bytes * 2; // k_cache and v_cache
        }

        WorkspaceStats {
            total_bytes,
            buffer_count,
            expected_shape: Some((self.window_size, self.embed_dim)),
        }
    }
}

/// Streaming state management for SlidingWindowAttention
impl StreamingWorkspaceManaged for SlidingWindowAttention {
    /// Initialize streaming state for inference with step-by-step processing
    fn init_streaming(&mut self, _batch_size: usize, embed_dim: usize) -> Result<()> {
        // Allocate streaming cache with proper dimensions
        self.streaming_cache = Some(SlidingWindowCache::new(self.window_size, embed_dim));
        Ok(())
    }

    /// Reset streaming state between sequences
    fn reset_streaming_state(&mut self) {
        if let Some(cache) = &mut self.streaming_cache {
            cache.reset();
        }
    }

    /// Check if streaming state is active
    fn is_streaming(&self) -> bool {
        self.streaming_cache.is_some()
    }
}
