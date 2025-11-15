//! Top-k selection and NIM computation utilities with optimized memory usage.
//!
//! This module provides zero-allocation, cache-efficient utilities for selecting
//! top-k elements from scores and computing Number of Important Mixture (NIM) components.

/// Select top-k items from scores and return their indices and normalized weights.
/// Uses in-place sorting and minimal allocations beyond output vectors.
///
/// # Arguments
/// * `scores` - Vector of scores for each item
/// * `k` - Number of top items to select (clamped to scores.len())
///
/// # Returns
/// A tuple of (indices, weights) where:
/// - indices: Vec<usize> of the selected item indices (sorted by score descending)
/// - weights: Vec<f32> of normalized weights summing to 1.0
pub fn select_top_k(scores: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = scores.len();
    let k_actual = k.min(n);

    if k_actual == 0 {
        return (Vec::new(), Vec::new());
    }

    // Create and sort indices in descending score order with direct comparison
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Single-pass: collect scores and compute sum simultaneously
    let mut top_indices = Vec::with_capacity(k_actual);
    let mut sum_top = 0.0f32;

    for &idx in &indices[..k_actual] {
        let score = scores[idx];
        top_indices.push(idx);
        sum_top += score;
    }

    // Normalize weights in-place during construction
    sum_top = sum_top.max(1e-12);
    let weights: Vec<f32> = indices[..k_actual]
        .iter()
        .map(|&idx| scores[idx] / sum_top)
        .collect();

    (top_indices, weights)
}

/// Compute NIM (Number of Important Mixture components) for a row of scores.
/// NIM measures the effective number of components: 1 / sum(p²) where p are normalized
/// probabilities.
///
/// Uses optimized single-pass computation with better numerical stability and cache-efficient
/// chunked processing.
///
/// # Arguments
/// * `scores` - Vector of scores for each component
///
/// # Returns
/// Float representing the effective number of important components
pub fn compute_nim(scores: &[f32]) -> f32 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    // Single pass: compute sum and sum of squares with manual unrolling for better performance
    let mut sum_all = 0.0f32;
    let mut sum_sq = 0.0f32;

    let (chunks_8, remainder) = scores.as_chunks::<8>();

    // Process 8 elements at a time for optimal cache usage
    for chunk in chunks_8 {
        let s0 = chunk[0];
        let s1 = chunk[1];
        let s2 = chunk[2];
        let s3 = chunk[3];
        let s4 = chunk[4];
        let s5 = chunk[5];
        let s6 = chunk[6];
        let s7 = chunk[7];

        sum_all += s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
        sum_sq += s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6 + s7 * s7;
    }

    // Handle remainder elements
    for &s in remainder {
        sum_all += s;
        sum_sq += s * s;
    }

    // Handle numerical edge cases with better stability bounds
    sum_all = sum_all.max(1e-12);
    sum_sq = sum_sq.max(1e-20);

    // Compute normalized sum of squares: sum((s/sum_all)²)
    let normalized_sum_sq = sum_sq / (sum_all * sum_all);

    // NIM = 1 / sum(p²) with numerical stability bounds
    1.0 / normalized_sum_sq.max(1e-12)
}

/// Fast path for computing NIM when you already have normalized probabilities.
/// This is a zero-copy optimization for cases where normalization has already been done.
///
/// # Arguments
/// * `normalized_probs` - Pre-normalized probability distribution (should sum to ~1.0)
///
/// # Returns
/// Float representing the effective number of important components
#[inline(always)]
pub fn compute_nim_from_normalized(normalized_probs: &[f32]) -> f32 {
    if normalized_probs.is_empty() {
        return 0.0;
    }

    // Single pass: compute sum of p² only (probabilities should already be normalized)
    let mut sum_p_sq = 0.0f32;

    // Manual unrolling for small arrays (common case)
    let (chunks_4, remainder) = normalized_probs.as_chunks::<4>();
    for chunk in chunks_4 {
        sum_p_sq +=
            chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2] + chunk[3] * chunk[3];
    }

    // Handle remainder
    for &p in remainder {
        sum_p_sq += p * p;
    }

    1.0 / sum_p_sq.max(1e-12)
}
