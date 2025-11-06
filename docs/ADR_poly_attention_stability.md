# ADR: PolyAttention Stability and Gradient Bounds

Status: Accepted

## Overview

This ADR formalizes stability properties of Polynomial Attention with s-clipping, gating via Richards curves, learned threshold prediction (AutoDeco-inspired two-layer network), and CoPE integration.

We establish boundedness of logits and gradient magnitudes under the introduced s-clipping and characterize sufficient conditions that preclude NaNs. We also identify redundancy removed in the backward pass and document ordering for predictor gradients.

## Formal Modeling

Let `q_i, k_j, v_j ∈ R^d` and define the pre-logit score

$$ s_{ij} = \langle q_i, k_j \rangle \cdot \gamma + \delta_{ij} $$

where `γ` subsumes scalar scaling and `δ_{ij}` aggregates additive terms (e.g., CoPE positional component if present). Define the clipped score

$$ \bar s_{ij} = \mathrm{clip}(s_{ij}, -L, L) $$

and the polynomial logit

$$ \ell_{ij} = (\bar s_{ij})^{p},\quad p \in \mathbb{N}_{\ge 1}. $$

Per-head gating uses a differentiable Richards curve `g_h(x)` applied to a per-head projection; the output per token i is

$$ y_i = \sum_j g_h(x_i)\, \ell_{ij} \, v_j + \text{residual}. $$

The learned threshold predictor outputs per-token thresholds `m_i \in (0,1)` via

$$ m = \sigma\big(W_2\,\phi(\mathrm{RN}(X W_1 + b_1)) + b_2\big), $$

with `RN` a Richards-based normalization (tanh-equivalent scaling), `φ` the ReLU, and `σ` the sigmoid.

## Theorem 1 (Logit and Local Gradient Bounds)

Assume `L > 0` and `p \ge 1`. Then for any `s \in \mathbb{R}`:

1. Bounded logits: `|\ell| = |\bar s|^{p} \le L^{p}`.
2. Local gradient bound: `\left|\partial \ell/\partial s\right| = p\,|\bar s|^{p-1} \cdot \mathbf{1}_{|s| \le L} \le p\,L^{p-1}`.

Proof.

Clipping enforces `|\bar s| \le L`. The derivative exists where the clamp is active (i.e., inside interval) and is zero outside. Direct computation yields `\partial \ell/\partial \bar s = p\,\bar s^{p-1}` and `\partial \bar s/\partial s = 1` for `|s|\le L`, else `0`. The stated bounds follow. ∎

Corollary. The gradient w.r.t. `q` and `k` is bounded by `p\,L^{p-1}` times the scale and the norms of `k` and `q` respectively, within the active clipping interval.

## Theorem 2 (Predictor Gradient Boundedness)

Let `z = W_2\,\phi(\mathrm{RN}(X W_1 + b_1)) + b_2`, `m = \sigma(z)`. Assume `RN` implements exact `tanh(α·x)` scaling with bounded `α`, ReLU derivative in `{0,1}`, and that `\|W_1\|,\|W_2\|` are finite. Then

$$ \|\nabla_X m\| \le \|W_2\|\,\|W_1\|\,\sup_x |\sigma'(z)|\,\sup_x |\mathrm{RN}'(x)|. $$

Since `\sigma'(z) = \sigma(z)(1-\sigma(z)) \in (0,1/4]` and `\mathrm{RN}'(x) = α\,\mathrm{sech}^2(αx) \le α`, we obtain

$$ \|\nabla_X m\| \le \frac{α}{4}\,\|W_2\|\,\|W_1\|. $$

Thus, predictor gradients are bounded provided parameter norms and `α` remain bounded.

## NaN Exclusion Conditions

- Optimizer safety: Adam updates use `\sqrt{v} + \varepsilon`, `\varepsilon = 10^{-8}` ⇒ no division by zero.
- No log/sqrt of model activations in attention path ⇒ no domain violations.
- Clamp zeros gradient outside `[-L,L]`, preventing runaway growth from extreme `s`.
- Richards parameters in normalization and GLU are clamped during `step()` (e.g., `ν, k` lower-bounded), preventing invalid states.

Under these conditions, NaNs can only arise from externally-injected non-finite values or unchecked operations outside this formulation.

## Practical Stability Considerations

- Choose `L` such that `p\,L^{p-1}` is moderate (default `L=10`, `p=3` ⇒ bound `≤ 300`).
- Monitor gate polynomial coefficients; keep `l2_reg` > 0 to prevent blow-up.
- CoPE contributions can be large pre-clamp; clipping ensures `\bar s` remains bounded and shuts off gradients outside `[-L,L]`.
- Sigmoid saturation can reduce predictor learning; this is benign for stability and mitigated by RN scaling.

## Redundancy and Ordering

- Removed obsolete local gradient placeholders (`grad_w_tau`, `grad_alpha_tau`, `grad_beta_tau`) from `PolyAttention::compute_gradients`.
- Predictor gradient append order is: `W1`, `b1`, `W2`, `b2`. `apply_gradients` steps in this exact order.

## Conclusion

The s-clipping coupled with bounded-derivative non-linearities yields provable bounds on logits and gradient magnitudes. Together with Adam’s `\varepsilon` and Richards parameter constraints, the architecture avoids typical NaN sources. Remaining instabilities are constrained by hyperparameters (`L`, `p`, regularization) and parameter norms.

