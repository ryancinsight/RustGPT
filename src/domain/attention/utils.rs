use ndarray::Array2;

use crate::domain::pade::PadeExp;

/// Smoothly saturate values to ±limit using tanh.
///
/// This is a drop-in replacement for hard clamping in stability-sensitive hot loops.
#[inline]
pub fn smooth_clip_tanh(x: f32, limit: f32) -> f32 {
    if !x.is_finite() || !limit.is_finite() || limit <= 0.0 {
        return 0.0;
    }
    let tanh = crate::domain::richards::RichardsCurve::tanh(false);
    limit * tanh.forward_scalar_f32(x / limit)
}

/// Smoothly saturate values to ±limit and return both the saturated value and its derivative.
///
/// If `x` is non-finite, returns (0, 0).
#[inline]
pub fn smooth_clip_tanh_with_grad(x: f32, limit: f32) -> (f32, f32) {
    if !x.is_finite() || !limit.is_finite() || limit <= 0.0 {
        return (0.0, 0.0);
    }
    let tanh = crate::domain::richards::RichardsCurve::tanh(false);
    let u = x / limit;
    let t = tanh.forward_scalar_f32(u);
    // d/dx [limit * tanh(x/limit)] = dtanh(x/limit)
    let dy_dx = tanh.derivative_scalar_f32(u);
    (limit * t, dy_dx)
}

/// Smoothly saturate a value into [0, 1] without hard clamping.
///
/// This is a smooth approximation of `x.clamp(0, 1)` that stays close to the
/// identity mapping for `x` in [0, 1], and only saturates smoothly outside.
#[inline]
pub fn smooth_saturate_01(x: f32) -> f32 {
    #[inline]
    fn softplus_beta(z: f32, beta: f32) -> f32 {
        if !z.is_finite() {
            return 0.0;
        }
        let t = (beta * z) as f64;
        // Stable softplus: (1/beta) * ln(1 + exp(beta*z))
        if t > 20.0 {
            z
        } else if t < -20.0 {
            (PadeExp::exp(t) as f32) / beta
        } else {
            let e = PadeExp::exp(t) as f32;
            e.ln_1p() / beta
        }
    }

    if !x.is_finite() {
        return 0.0;
    }

    // Smooth clamp via softplus: x - softplus(x-1) + softplus(-x)
    // With sufficiently large beta this becomes very close to hard clamping,
    // while remaining smooth.
    let beta = 10.0f32;
    x - softplus_beta(x - 1.0, beta) + softplus_beta(-x, beta)
}

/// Attention utility functions for common operations
/// Provides reusable helper functions for attention mechanisms
/// Apply causal mask in-place to an attention matrix
/// Sets all elements above the diagonal to -inf so softmax produces zero attention.
#[inline]
pub fn apply_causal_mask_inplace(mat: &mut Array2<f32>) {
    let n = mat.nrows();
    for i in 0..n {
        for j in (i + 1)..n {
            mat[[i, j]] = f32::NEG_INFINITY;
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
                mat[[i, j]] = f32::NEG_INFINITY;
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
        let mut argmax = 0usize;
        let mut max_val = f32::NEG_INFINITY;
        let mut any_finite = false;
        for (i, &v) in row.iter().enumerate() {
            if v.is_finite() {
                any_finite = true;
                if v > max_val {
                    max_val = v;
                    argmax = i;
                }
            }
        }

        // If the whole row is masked (all -inf) or non-finite, yield all zeros.
        if !any_finite {
            for x in row.iter_mut() {
                *x = 0.0;
            }
            continue;
        }

        let mut sum = 0.0f64;
        for &v in row.iter() {
            // Treat masked / non-finite entries as probability 0.
            if !v.is_finite() {
                continue;
            }
            sum += PadeExp::exp((v - max_val) as f64);
        }

        if !sum.is_finite() || sum <= 0.0 {
            // Fallback: deterministic one-hot at argmax.
            for (i, x) in row.iter_mut().enumerate() {
                *x = if i == argmax { 1.0 } else { 0.0 };
            }
            continue;
        }

        let inv_sum = (1.0 / sum) as f32;
        for x in row.iter_mut() {
            if !x.is_finite() {
                *x = 0.0;
                continue;
            }
            *x = (PadeExp::exp((*x - max_val) as f64) as f32) * inv_sum;
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
pub fn compute_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    dk_scale: f32,
) -> Array2<f32> {
    let mut scores = compute_attention_scores(q, k, dk_scale);
    apply_softmax_attention(&mut scores);
    compute_weighted_sum(&scores, v)
}
