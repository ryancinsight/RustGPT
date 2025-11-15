use ndarray::{Array1, Array2};

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

fn one_hot_row(vocab: usize, target: usize) -> Array1<f32> {
    let mut y = Array1::<f32>::zeros(vocab);
    if target < vocab {
        y[target] = 1.0;
    }
    y
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
        grads.mapv(|x| x / (rows as f32))
    } else {
        grads.fill(0.0);
        grads
    }
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

    let mut rce = 0.0f32;
    for i in 0..rows {
        let t = targets[i];
        let mut c_sum = 0.0f32;
        for k in 0..vocab {
            let y = if k == t { 1.0 } else { epsilon }; // stabilized one-hot
            c_sum += probs[[i, k]] * (-y.ln());
        }
        rce += c_sum;
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

    for i in 0..rows {
        let t = targets[i];
        // c_i = -log(y_i)
        let mut c = Array1::<f32>::zeros(vocab);
        for k in 0..vocab {
            c[k] = if k == t {
                0.0
            } else {
                -(epsilon.max(f32::MIN_POSITIVE)).ln()
            };
        }
        // dot(c, p)
        let c_dot_p: f32 = c.iter().zip(probs.row(i)).map(|(ci, &pi)| ci * pi).sum();
        // grad_rce_row = p ∘ (c - c_dot_p)
        for k in 0..vocab {
            let pk = probs[[i, k]];
            grad[[i, k]] = beta * pk * (c[k] - c_dot_p);
        }
    }

    // Combine and normalize
    for ((g, &gc), &p) in grad.iter_mut().zip(ce_grad.iter()).zip(probs.iter()) {
        *g = alpha * gc + *g;
        let _ = p; // keep iteration structure
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
    assert_eq!(v_pred.shape(), v_true.shape(), "v_mse_gradients: shapes must match");
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
        let softmax = crate::softmax::Softmax::new();
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
}
