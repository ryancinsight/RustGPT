use ndarray::{Array1, Array2, ArrayView2};

use crate::pade::PadeExp;

/// Symmetric Cross Entropy (SCE) utilities
///
/// SCE combines the standard cross-entropy CE(y, p) with the reverse cross-entropy CE(p, y):
/// L_sce = alpha * CE(y, p) + beta * CE(p, y), where y is one-hot (stabilized) and p =
/// softmax(logits). Numerical stability is ensured by clamping y_i for non-target classes with
/// epsilon to avoid log(0).
pub struct SymmetricCEConfig {
    pub alpha: f32,
    pub beta: f32,
    pub epsilon: f32,
}

impl Default for SymmetricCEConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            epsilon: 1e-4,
        }
    }
}

pub fn cross_entropy(probs: &Array2<f32>, targets: &[usize]) -> f32 {
    let vocab = probs.ncols();
    let rows = probs.nrows().min(targets.len());
    let mut loss = 0.0f32;
    for i in 0..rows {
        let t = targets[i];
        if t >= vocab {
            continue;
        }
        let p = probs[[i, t]].max(f32::MIN_POSITIVE);
        loss -= p.ln();
    }
    if rows > 0 { loss / (rows as f32) } else { 0.0 }
}

/// Numerically-stable cross-entropy computed directly from logits.
///
/// This avoids taking `ln(p)` on probabilities that may underflow to 0.0 in `f32`.
/// Uses log-sum-exp with `ln_1p` for accuracy when the distribution is very peaky.
pub fn cross_entropy_from_logits(logits: &ArrayView2<f32>, targets: &[usize]) -> f32 {
    let vocab = logits.ncols();
    let rows = logits.nrows().min(targets.len());
    if rows == 0 || vocab == 0 {
        return 0.0;
    }

    let mut loss_f64 = 0.0f64;

    for i in 0..rows {
        let t = targets[i];
        if t >= vocab {
            continue;
        }

        let row = logits.row(i);
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        if !max_val.is_finite() {
            // Degenerate row; keep behavior defined
            continue;
        }

        // sum_j exp(logit_j - max)
        let mut sum = 0.0f64;
        for &x in row.iter() {
            // (x - max) <= 0, so exp is safe.
            sum += PadeExp::exp((x - max_val) as f64);
        }

        // sum >= 1 because it includes exp(0)=1 for the max element.
        let sum_minus_1 = (sum - 1.0).max(0.0);
        let lse = (max_val as f64) + sum_minus_1.ln_1p();
        let logp_t = (logits[[i, t]] as f64) - lse;
        loss_f64 -= logp_t;
    }

    (loss_f64 as f32) / (rows as f32)
}

pub fn cross_entropy_gradients(probs: &Array2<f32>, targets: &[usize]) -> Array2<f32> {
    let mut grads = probs.clone();
    let vocab = probs.ncols();
    let rows = probs.nrows().min(targets.len());
    for i in 0..rows {
        let t = targets[i];
        if t < vocab {
            grads[[i, t]] -= 1.0;
        }
    }
    if rows > 0 {
        let scale = 1.0 / (rows as f32);
        grads.mapv_inplace(|x| x * scale);
        grads
    } else {
        grads.fill(0.0);
        grads
    }
}

/// Residual decorrelation loss (Barlow Twins / VICReg-style redundancy reduction).
///
/// Given features `H` with shape (n_tokens, d_model), we center across tokens and
/// penalize squared off-diagonal covariance:
///
/// $$L = \sum_{i \ne j} \mathrm{cov}(H)_{ij}^2$$
///
/// This encourages residual channels to encode distinct information ("what it is")
/// and discourages confusable/entangled features ("what it is not").
pub fn residual_decorrelation_loss(features: &ArrayView2<f32>) -> f32 {
    let n = features.nrows();
    let d = features.ncols();
    if n < 2 || d < 2 {
        return 0.0;
    }

    // Compute per-dimension mean using ndarray operations for better performance
    let mut mean = Array1::<f64>::zeros(d);
    for row in features.outer_iter() {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += if v.is_finite() { v as f64 } else { 0.0 };
        }
    }
    mean.mapv_inplace(|x| x / (n as f64));

    // Center features
    let mut centered = Array2::<f64>::zeros((n, d));
    for (i, row) in features.outer_iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let val = (v as f64) - mean[j];
            centered[[i, j]] = if val.is_finite() { val } else { 0.0 };
        }
    }

    // Compute covariance matrix: C = X^T X / n
    let inv_n = 1.0f64 / (n as f64);
    let cov = centered.t().dot(&centered) * inv_n;

    // Sum squared off-diagonal elements
    let mut loss = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if i != j {
                let cij = cov[[i, j]];
                loss += cij * cij;
            }
        }
    }

    // Normalize by number of off-diagonal entries for scale stability
    let denom = (d * (d - 1)) as f64;
    (loss / denom.max(1.0)) as f32
}

/// Gradients of `residual_decorrelation_loss` w.r.t. the input features.
///
/// Let X be centered features (n x d), C = X^T X / n.
/// L = sum_{i!=j} C_ij^2.
/// dL/dC = G where G_ij = 2*C_ij for i!=j, else 0.
/// dL/dX = (2/n) * X * G (since G is symmetric).
/// Then project back through centering: dL/dH = dL/dX - mean_token(dL/dX).
pub fn residual_decorrelation_gradients(features: &ArrayView2<f32>) -> Array2<f32> {
    let n = features.nrows();
    let d = features.ncols();
    let mut grad = Array2::<f32>::zeros((n, d));
    if n < 2 || d < 2 {
        return grad;
    }

    // Compute per-dimension mean
    let mut mean = Array1::<f64>::zeros(d);
    for row in features.outer_iter() {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += if v.is_finite() { v as f64 } else { 0.0 };
        }
    }
    mean.mapv_inplace(|x| x / (n as f64));

    // Center features
    let mut centered = Array2::<f64>::zeros((n, d));
    for (i, row) in features.outer_iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let val = (v as f64) - mean[j];
            centered[[i, j]] = if val.is_finite() { val } else { 0.0 };
        }
    }

    // Compute covariance C = X^T X / n
    let inv_n = 1.0f64 / (n as f64);
    let cov = centered.t().dot(&centered) * inv_n;

    // G = dL/dC: zero diagonal, 2*C_ij off-diagonal
    let mut g = cov.mapv(|x| 2.0 * x);
    for i in 0..d {
        g[[i, i]] = 0.0;
    }

    // dL/dX = (2/n) * X * G
    let scale = 2.0f64 * inv_n;
    let dx = centered.dot(&g) * scale;

    // Project through centering: subtract token-mean gradient per dimension
    let dx_mean = dx.mean_axis(ndarray::Axis(0)).unwrap();
    let mut dx_centered = dx;
    for mut row in dx_centered.outer_iter_mut() {
        row -= &dx_mean;
    }

    // Convert to f32 and normalize
    let denom = (d * (d - 1)) as f32;
    for (i, row) in dx_centered.outer_iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let val = if v.is_finite() { v as f32 } else { 0.0 };
            grad[[i, j]] = if denom > 0.0 { val / denom } else { val };
        }
    }

    grad
}

pub fn info_nce_loss_and_grads(
    anchor: &[f32],
    positive: &[f32],
    negatives: &[Vec<f32>],
    k: usize,
    temperature: f32,
) -> (f32, Vec<f32>, Vec<f32>) {
    let d = anchor.len();
    if d == 0 || positive.len() != d || k == 0 || negatives.is_empty() {
        return (0.0, vec![0.0; d], vec![0.0; d]);
    }

    let tau = temperature.max(1e-6) as f64;
    let mut na2 = 0.0f64;
    let mut np2 = 0.0f64;
    let mut dot_ap = 0.0f64;
    for j in 0..d {
        let a = anchor[j];
        let p = positive[j];
        let aa = if a.is_finite() { a as f64 } else { 0.0 };
        let pp = if p.is_finite() { p as f64 } else { 0.0 };
        na2 += aa * aa;
        np2 += pp * pp;
        dot_ap += aa * pp;
    }
    let na = na2.sqrt().max(1e-12);
    let np = np2.sqrt().max(1e-12);
    let sim_pos = dot_ap / (na * np);

    let mut sims: Vec<(f64, usize, f64, f64)> = Vec::with_capacity(negatives.len());
    for (idx, neg) in negatives.iter().enumerate() {
        if neg.len() != d {
            continue;
        }
        let mut dot = 0.0f64;
        let mut nb2 = 0.0f64;
        for j in 0..d {
            let a = anchor[j];
            let b = neg[j];
            let aa = if a.is_finite() { a as f64 } else { 0.0 };
            let bb = if b.is_finite() { b as f64 } else { 0.0 };
            dot += aa * bb;
            nb2 += bb * bb;
        }
        let nb = nb2.sqrt().max(1e-12);
        let sim = dot / (na * nb);
        sims.push((sim, idx, dot, nb));
    }
    if sims.is_empty() {
        return (0.0, vec![0.0; d], vec![0.0; d]);
    }

    sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let take = k.min(sims.len());
    let mut logits: Vec<f64> = Vec::with_capacity(1 + take);
    logits.push(sim_pos / tau);
    for (sim, _, _, _) in sims.iter().take(take) {
        logits.push(*sim / tau);
    }

    let max_logit = logits
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut exp_sum = 0.0f64;
    let mut exp_logits: Vec<f64> = Vec::with_capacity(logits.len());
    for &l in &logits {
        let e = (l - max_logit).exp();
        exp_sum += e;
        exp_logits.push(e);
    }
    let inv_sum = if exp_sum > 0.0 { 1.0 / exp_sum } else { 0.0 };
    let p_pos = exp_logits[0] * inv_sum;
    let loss = -(p_pos.max(1e-12)).ln();

    let mut grad_anchor = vec![0.0f32; d];
    let mut grad_positive = vec![0.0f32; d];
    let inv_tau = 1.0f64 / tau;
    let dlogit_pos = p_pos - 1.0;

    for j in 0..d {
        let a = if anchor[j].is_finite() {
            anchor[j] as f64
        } else {
            0.0
        };
        let p = if positive[j].is_finite() {
            positive[j] as f64
        } else {
            0.0
        };
        let grad_a = (p / (na * np)) - (sim_pos / (na * na)) * a;
        let grad_p = (a / (na * np)) - (sim_pos / (np * np)) * p;
        grad_anchor[j] = (dlogit_pos * inv_tau * grad_a) as f32;
        grad_positive[j] = (dlogit_pos * inv_tau * grad_p) as f32;
    }

    for (i, (_, idx, dot, nb)) in sims.iter().take(take).enumerate() {
        let p_i = exp_logits[1 + i] * inv_sum;
        let dlogit = p_i;
        let neg = &negatives[*idx];
        for j in 0..d {
            let a = if anchor[j].is_finite() {
                anchor[j] as f64
            } else {
                0.0
            };
            let b = if neg[j].is_finite() { neg[j] as f64 } else { 0.0 };
            let grad_a = (b / (na * nb)) - ((*dot / (na * nb)) / (na * na)) * a;
            grad_anchor[j] += (dlogit * inv_tau * grad_a) as f32;
        }
    }

    (loss as f32, grad_anchor, grad_positive)
}

/// Hard-negative repulsion loss over a pooled representation.
///
/// This implements a lightweight "learn what it is not" objective without requiring a
/// second positive view/augmentation. Given an anchor vector `a` and a set of negative
/// vectors `negatives`, we select the top-k most similar negatives (hard negatives) by
/// cosine similarity and penalize any similarity above a margin:
///
/// $$L = \frac{1}{k} \sum_{n \in \mathrm{TopK}} \mathrm{softplus}((\cos(a,n) - m)/\tau)$$
///
/// Returns (loss, grad_wrt_anchor).
pub fn hard_negative_repulsion_loss_and_grad(
    anchor: &[f32],
    negatives: &[Vec<f32>],
    k: usize,
    margin: f32,
    temperature: f32,
) -> (f32, Vec<f32>) {
    let d = anchor.len();
    if d == 0 || negatives.is_empty() || k == 0 {
        return (0.0, vec![0.0; d]);
    }

    let tau = temperature.max(1e-6);
    let m = margin;

    // Norm of anchor.
    let mut na2 = 0.0f64;
    for &v in anchor {
        let x = if v.is_finite() { v as f64 } else { 0.0 };
        na2 += x * x;
    }
    let na = na2.sqrt().max(1e-12);

    // Compute similarities for all negatives.
    let mut sims: Vec<(f32, usize, f64)> = Vec::with_capacity(negatives.len());
    for (idx, neg) in negatives.iter().enumerate() {
        if neg.len() != d {
            continue;
        }
        let mut dot = 0.0f64;
        let mut nb2 = 0.0f64;
        for j in 0..d {
            let a = anchor[j];
            let b = neg[j];
            let af = if a.is_finite() { a as f64 } else { 0.0 };
            let bf = if b.is_finite() { b as f64 } else { 0.0 };
            dot += af * bf;
            nb2 += bf * bf;
        }
        let nb = nb2.sqrt().max(1e-12);
        let cos = (dot / (na * nb)).clamp(-1.0, 1.0) as f32;
        sims.push((cos, idx, nb));
    }
    if sims.is_empty() {
        return (0.0, vec![0.0; d]);
    }

    // Select top-k by similarity.
    sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let k_eff = k.min(sims.len());
    let top = &sims[..k_eff];

    let mut loss = 0.0f64;
    let mut grad = vec![0.0f64; d];

    // d cos / d a = b/(||a|| ||b||) - cos * a/(||a||^2)
    for &(cos_f32, neg_idx, nb) in top {
        let cos = cos_f32 as f64;

        // softplus(x) where x=(cos - m)/tau
        let x = ((cos_f32 - m) / tau) as f64;
        // stable softplus
        let sp = if x > 30.0 {
            x
        } else if x < -30.0 {
            0.0
        } else {
            (1.0 + x.exp()).ln()
        };
        loss += sp;

        // d softplus / d x = sigmoid(x)
        let sig = 1.0 / (1.0 + (-x).exp());
        // dL/dcos = sigmoid(x) * (1/tau)
        let dldcos = sig * (1.0 / (tau as f64));

        let neg = &negatives[neg_idx];
        for j in 0..d {
            let a = anchor[j];
            let b = neg[j];
            let af = if a.is_finite() { a as f64 } else { 0.0 };
            let bf = if b.is_finite() { b as f64 } else { 0.0 };

            let dcos_da = (bf / (na * nb)) - (cos * af / (na * na));
            grad[j] += dldcos * dcos_da;
        }
    }

    // Average over k.
    let inv_k = 1.0f64 / (k_eff as f64);
    loss *= inv_k;
    for g in &mut grad {
        *g *= inv_k;
    }

    let grad_f32: Vec<f32> = grad
        .into_iter()
        .map(|v| if v.is_finite() { v as f32 } else { 0.0 })
        .collect();
    (loss as f32, grad_f32)
}

pub fn symmetric_cross_entropy(
    probs: &Array2<f32>,
    targets: &[usize],
    alpha: f32,
    beta: f32,
    epsilon: f32,
) -> f32 {
    let vocab = probs.ncols();
    let rows = probs.nrows().min(targets.len());
    if rows == 0 {
        return 0.0;
    }

    let ce = cross_entropy(probs, targets);

    // Reverse CE: sum_k p_k * (-ln y_k), where y is stabilized one-hot.
    // With y_t = 1 => -ln y_t = 0, and y_{k!=t} = eps => -ln eps is constant.
    let c_other = -(epsilon.max(f32::MIN_POSITIVE)).ln();
    let mut rce = 0.0f32;
    for i in 0..rows {
        let t = targets[i];
        if t < vocab {
            let p_t = probs[[i, t]];
            rce += c_other * (1.0 - p_t);
        } else {
            // If the target is invalid, treat all classes as non-target (matches original loop).
            rce += c_other;
        }
    }
    rce /= rows as f32;

    alpha * ce + beta * rce
}

/// Symmetric Cross Entropy where the CE term is computed from logits via log-sum-exp.
///
/// This reduces loss spikes caused by `f32` softmax underflow making `p_target == 0.0`.
pub fn symmetric_cross_entropy_from_logits(
    logits: &ArrayView2<f32>,
    probs: &ArrayView2<f32>,
    targets: &[usize],
    alpha: f32,
    beta: f32,
    epsilon: f32,
) -> f32 {
    let vocab = probs.ncols();
    let rows = probs.nrows().min(targets.len()).min(logits.nrows());
    if rows == 0 || vocab == 0 {
        return 0.0;
    }

    let ce = cross_entropy_from_logits(&logits.slice(ndarray::s![0..rows, ..]), &targets[..rows]);

    let c_other = -(epsilon.max(f32::MIN_POSITIVE)).ln();
    let mut rce = 0.0f32;
    for i in 0..rows {
        let t = targets[i];
        if t < vocab {
            let p_t = probs[[i, t]];
            rce += c_other * (1.0 - p_t);
        } else {
            rce += c_other;
        }
    }
    rce /= rows as f32;

    alpha * ce + beta * rce
}

pub fn symmetric_cross_entropy_gradients(
    probs: &Array2<f32>,
    targets: &[usize],
    alpha: f32,
    beta: f32,
    epsilon: f32,
) -> Array2<f32> {
    let vocab = probs.ncols();
    let rows = probs.nrows().min(targets.len());
    let mut grad = Array2::<f32>::zeros(probs.raw_dim());
    if rows == 0 {
        return grad;
    }

    let ce_grad = cross_entropy_gradients(probs, targets);

    // Reverse CE gradient w.r.t logits: p ∘ (c - E_p[c]) where
    // c_t = 0, c_{k!=t} = -ln(eps).
    // IMPORTANT: loss is averaged over rows, so gradients must also be scaled by 1/rows.
    let c_other = -(epsilon.max(f32::MIN_POSITIVE)).ln();
    let rce_scale = beta / (rows as f32);

    for i in 0..rows {
        let t = targets[i];
        if t >= vocab {
            continue;
        }

        let p_t = probs[[i, t]];
        // E_p[c] = sum_k p_k c_k = (1 - p_t) * c_other
        let e_c = (1.0 - p_t) * c_other;
        for k in 0..vocab {
            let pk = probs[[i, k]];
            let ck = if k == t { 0.0 } else { c_other };
            grad[[i, k]] = rce_scale * pk * (ck - e_c);
        }
    }

    // Combine and normalize
    for (g, &gc) in grad.iter_mut().zip(ce_grad.iter()) {
        *g += alpha * gc;
    }
    grad
}

/// Mean Squared Error loss for epsilon prediction in diffusion models
/// L_eps = E[||epsilon - epsilon_pred||^2]
pub fn epsilon_mse(eps_pred: &Array2<f32>, eps_true: &Array2<f32>) -> f32 {
    assert_eq!(
        eps_pred.shape(),
        eps_true.shape(),
        "epsilon_mse: shapes must match"
    );
    let n = (eps_pred.nrows() * eps_pred.ncols()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (a, b) in eps_pred.iter().zip(eps_true.iter()) {
        let d = *a - *b;
        sum += d * d;
    }
    sum / n
}

/// Gradients of epsilon MSE loss w.r.t eps_pred
/// d/d(eps_pred) = 2/N * (eps_pred - eps_true)
pub fn epsilon_mse_gradients(eps_pred: &Array2<f32>, eps_true: &Array2<f32>) -> Array2<f32> {
    assert_eq!(
        eps_pred.shape(),
        eps_true.shape(),
        "epsilon_mse_gradients: shapes must match"
    );
    let n = (eps_pred.nrows() * eps_pred.ncols()) as f32;
    let scale = if n > 0.0 { 2.0 / n } else { 0.0 };
    let mut grad = Array2::<f32>::zeros(eps_pred.raw_dim());
    for ((g, &p), &t) in grad.iter_mut().zip(eps_pred.iter()).zip(eps_true.iter()) {
        *g = scale * (p - t);
    }
    grad
}

/// Mean Squared Error loss for v-prediction parameterization in diffusion
/// v = sqrt(alpha_bar) * epsilon − sqrt(1 − alpha_bar) * x0
pub fn v_mse(v_pred: &Array2<f32>, v_true: &Array2<f32>) -> f32 {
    assert_eq!(v_pred.shape(), v_true.shape(), "v_mse: shapes must match");
    let n = (v_pred.nrows() * v_pred.ncols()) as f32;
    if n == 0.0 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for (a, b) in v_pred.iter().zip(v_true.iter()) {
        let d = *a - *b;
        sum += d * d;
    }
    sum / n
}

/// Gradients of v MSE loss w.r.t v_pred
/// d/d(v_pred) = 2/N * (v_pred − v_true)
pub fn v_mse_gradients(v_pred: &Array2<f32>, v_true: &Array2<f32>) -> Array2<f32> {
    assert_eq!(
        v_pred.shape(),
        v_true.shape(),
        "v_mse_gradients: shapes must match"
    );
    let n = (v_pred.nrows() * v_pred.ncols()) as f32;
    let scale = if n > 0.0 { 2.0 / n } else { 0.0 };
    let mut grad = Array2::<f32>::zeros(v_pred.raw_dim());
    for ((g, &p), &t) in grad.iter_mut().zip(v_pred.iter()).zip(v_true.iter()) {
        *g = scale * (p - t);
    }
    grad
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn test_sce_numerical_stability() {
        let probs: Array2<f32> = array![[0.999999f32, 0.000001f32]];
        let targets = [0usize];
        let s = symmetric_cross_entropy(&probs, &targets, 1.0, 1.0, 1e-6);
        assert!(s.is_finite());
    }

    #[test]
    fn test_sce_gradient_matches_finite_difference() {
        let logits: Array2<f32> = array![[2.0f32, -1.0f32]];
        let softmax = crate::soft::Softmax::new();
        let probs = softmax.forward_immutable(&logits.view());
        let targets = [0usize];
        let alpha = 1.0;
        let beta = 0.1;
        let eps = 1e-4;
        let grad = symmetric_cross_entropy_gradients(&probs, &targets, alpha, beta, eps);

        // Finite difference
        let h = 1e-3;
        for k in 0..logits.ncols() {
            let mut logits_pos = logits.clone();
            logits_pos[[0, k]] += h;
            let probs_pos = softmax.forward_immutable(&logits_pos.view());
            let l_pos = symmetric_cross_entropy(&probs_pos, &targets, alpha, beta, eps);

            let mut logits_neg = logits.clone();
            logits_neg[[0, k]] -= h;
            let probs_neg = softmax.forward_immutable(&logits_neg.view());
            let l_neg = symmetric_cross_entropy(&probs_neg, &targets, alpha, beta, eps);

            let fd = (l_pos - l_neg) / (2.0 * h);
            let gk = grad[[0, k]];
            assert!((fd - gk).abs() < 5e-3, "fd {} vs grad {}", fd, gk);
        }
    }

    #[test]
    fn test_sce_gradient_multirow_matches_finite_difference() {
        let logits: Array2<f32> = array![[2.0f32, -1.0f32], [-0.5f32, 0.25f32]];
        let softmax = crate::soft::Softmax::new();
        let probs = softmax.forward_immutable(&logits.view());
        let targets = [0usize, 1usize];
        let alpha = 1.0;
        let beta = 0.2;
        let eps = 1e-4;
        let grad = symmetric_cross_entropy_gradients(&probs, &targets, alpha, beta, eps);

        // Finite difference on logits
        let h = 1e-3;
        for i in 0..logits.nrows() {
            for k in 0..logits.ncols() {
                let mut logits_pos = logits.clone();
                logits_pos[[i, k]] += h;
                let probs_pos = softmax.forward_immutable(&logits_pos.view());
                let l_pos = symmetric_cross_entropy(&probs_pos, &targets, alpha, beta, eps);

                let mut logits_neg = logits.clone();
                logits_neg[[i, k]] -= h;
                let probs_neg = softmax.forward_immutable(&logits_neg.view());
                let l_neg = symmetric_cross_entropy(&probs_neg, &targets, alpha, beta, eps);

                let fd = (l_pos - l_neg) / (2.0 * h);
                let gk = grad[[i, k]];
                assert!(
                    (fd - gk).abs() < 5e-3,
                    "fd {} vs grad {} at ({},{})",
                    fd,
                    gk,
                    i,
                    k
                );
            }
        }
    }

    #[test]
    fn test_sce_decomposes_into_ce_and_rce() {
        let probs: Array2<f32> = array![[0.7f32, 0.3f32]];
        let targets = [0usize];
        let alpha = 1.0;
        let beta = 0.2;
        let eps = 1e-4;
        let s = symmetric_cross_entropy(&probs, &targets, alpha, beta, eps);
        let ce = cross_entropy(&probs, &targets);
        // Compute RCE explicitly
        let rce = {
            let mut r = 0.0;
            for (k, &p) in probs.row(0).iter().enumerate() {
                let y = if k == targets[0] { 1.0 } else { eps };
                r += p * (-y.ln());
            }
            r
        };
        assert!((s - (alpha * ce + beta * rce)).abs() < 1e-6);
    }

    #[test]
    fn test_ce_gradients_basic() {
        let probs: Array2<f32> = array![[0.6f32, 0.4f32]];
        let targets = [1usize];
        let g = cross_entropy_gradients(&probs, &targets);
        assert!(g.iter().all(|&x| x.is_finite()));
        // Sum of gradients per row equals zero for softmax CE
        let s: f32 = g.row(0).sum();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_mse_and_gradients_fd() {
        let eps_true: Array2<f32> = array![[0.1f32, -0.2f32], [0.3f32, 0.4f32]];
        let mut eps_pred: Array2<f32> = array![[0.0f32, 0.0f32], [0.0f32, 0.0f32]];
        let grad = epsilon_mse_gradients(&eps_pred, &eps_true);
        assert!(grad.iter().all(|&x| x.is_finite()));

        // Finite difference check on a single coordinate
        let h = 1e-3;
        eps_pred[[0, 1]] += h;
        let l_pos = epsilon_mse(&eps_pred, &eps_true);
        eps_pred[[0, 1]] -= 2.0 * h;
        let l_neg = epsilon_mse(&eps_pred, &eps_true);
        let fd = (l_pos - l_neg) / (2.0 * h);
        let g = grad[[0, 1]];
        assert!((fd - g).abs() < 1e-3, "fd {} vs grad {}", fd, g);
    }

    #[test]
    fn test_info_nce_loss_prefers_positive() {
        let anchor = [1.0f32, 0.0f32];
        let positive = [1.0f32, 0.0f32];
        let negatives = vec![vec![0.0f32, 1.0f32], vec![-1.0f32, 0.0f32]];
        let (loss, grad_a, grad_p) = info_nce_loss_and_grads(&anchor, &positive, &negatives, 2, 0.1);
        assert!(loss < 0.01);
        assert!(grad_a.iter().all(|&v| v.is_finite()));
        assert!(grad_p.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_info_nce_gradients_finite() {
        let anchor = [1.0f32, 2.0f32];
        let positive = [2.0f32, 1.0f32];
        let negatives = vec![vec![-1.0f32, 0.5f32], vec![0.1f32, -0.3f32]];
        let (_loss, grad_a, grad_p) = info_nce_loss_and_grads(&anchor, &positive, &negatives, 2, 0.5);
        assert!(grad_a.iter().all(|&v| v.is_finite()));
        assert!(grad_p.iter().all(|&v| v.is_finite()));
        let nonzero_a = grad_a.iter().any(|&v| v.abs() > 1e-8);
        let nonzero_p = grad_p.iter().any(|&v| v.abs() > 1e-8);
        assert!(nonzero_a || nonzero_p);
    }
}
