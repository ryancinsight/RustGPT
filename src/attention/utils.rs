use ndarray::Array2;

/// Attention utility functions for common operations
/// Provides reusable helper functions for attention mechanisms

/// Apply causal mask in-place to an attention matrix
/// Sets all elements above the diagonal to 0 (causal masking)
#[inline]
pub fn apply_causal_mask_inplace(mat: &mut Array2<f32>) {
    let n = mat.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            mat[[i, j]] = 0.0;
        }
    }
}

/// Apply sliding window mask in-place to an attention matrix
/// Masks out attention beyond a specified window size
#[inline]
pub fn apply_sliding_window_mask_inplace(mat: &mut Array2<f32>, window: Option<usize>) {
    if let Some(w) = window {
        let n = mat.nrows();
        for i in 0..n {
            let j_min = i.saturating_sub(w - 1);
            for j in 0..j_min {
                mat[[i, j]] = 0.0;
            }
        }
    }
}

/// Compute dot-product attention scores between queries and keys
/// Returns attention matrix of shape (n_queries, n_keys)
#[inline]
pub fn compute_attention_scores(q: &Array2<f32>, k: &Array2<f32>, dk_scale: f32) -> Array2<f32> {
    let mut scores = q.dot(&k.t());
    scores.mapv_inplace(|x| x * dk_scale);
    scores
}

/// Apply softmax normalization to attention weights
/// Normalizes along the last dimension (key dimension)
#[inline]
pub fn apply_softmax_attention(weights: &mut Array2<f32>) {
    for mut row in weights.outer_iter_mut() {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
        for x in row.iter_mut() {
            *x = (*x - max_val).exp() / exp_sum;
        }
    }
}

/// Compute weighted sum of values using attention weights
/// Returns attended values of shape (n_queries, value_dim)
#[inline]
pub fn compute_weighted_sum(attention_weights: &Array2<f32>, values: &Array2<f32>) -> Array2<f32> {
    attention_weights.dot(values)
}

/// Combined attention computation: Q·K^T → softmax → weighted sum with V
/// Performs the complete attention operation in one function
#[inline]
pub fn compute_attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>, dk_scale: f32) -> Array2<f32> {
    let mut scores = compute_attention_scores(q, k, dk_scale);
    apply_softmax_attention(&mut scores);
    compute_weighted_sum(&scores, v)
}
