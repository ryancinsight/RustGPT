## Scope
- Replace all MSE denoising paths with cross-entropy-only training consistent with diffusion-language modeling.
- Extract CE into a dedicated `src/loss.rs` module and implement Symmetric Cross Entropy (SCE).
- Preserve training pipeline compatibility, computational efficiency, and gradient flow.

## Remove MSE Denoising
- Delete pure-MSE diffusion training in `src/llm.rs:1311` and its inner MSE computation `src/llm.rs:1370-1376`.
- Remove denoising helper `src/transformer/diffusion_block.rs:546-554` and its unit test `src/transformer/diffusion_block.rs:860-889`.
- Drop CLI flag and usages for `diffusion_mse_weight` in `src/main.rs:30-32`, `src/main.rs:446-450`, `src/main.rs:497-501`, and perf check `src/main.rs:510-515` calling `denoising_loss`.
- Eliminate the MSE term and gradient mixing in diffusion CE training `src/llm.rs:1563-1569`, keeping the CE chain-rule path via scheduler `(-sqrt(1-α)/sqrt(α))`.

## New `loss.rs` Module
- Create `src/loss.rs` with:
  - `pub fn cross_entropy(probs: &Array2<f32>, targets: &[usize]) -> f32` (batch-average).
  - `pub fn cross_entropy_gradients(probs: &Array2<f32>, targets: &[usize]) -> Array2<f32>` returning `probs - one_hot` scaled by batch.
  - `pub fn symmetric_cross_entropy(probs: &Array2<f32>, targets: &[usize], alpha: f32, beta: f32, epsilon: f32) -> f32` where SCE = `alpha*CE(y,p) + beta*CE(p,y)`; use `y_i = 1 for target, epsilon otherwise` to stabilize `log(y)`.
  - `pub fn symmetric_cross_entropy_gradients(logits: &Array2<f32>, probs: &Array2<f32>, targets: &[usize], alpha: f32, beta: f32, epsilon: f32) -> Array2<f32>`; CE grad as above; reverse-CE grad per row `p ∘ (c - p·c)` with `c_i = -log(y_i)`; total grad `alpha*grad_ce + beta*grad_rce`.
- Add module-level rustdoc detailing math, stability, and expected inputs.

## Refactor CE Out of `llm.rs`
- Remove duplicated CE helpers `softmax`, `cross_entropy_loss_step`, `compute_gradients_step` (`src/llm.rs:1204-1227`) and `compute_cross_entropy_*` (`src/llm.rs:1605-1648`).
- Import and use `loss::{cross_entropy, cross_entropy_gradients, symmetric_cross_entropy, symmetric_cross_entropy_gradients}` in:
  - Transformer training paths at `src/llm.rs:882-891` and `src/llm.rs:1529-1537`.
  - Diffusion CE training `src/llm.rs:1529-1584` replacing CE pieces and removing MSE mixing.

## Training Pipeline Updates
- Keep function name `train_diffusion_ce` for compatibility; change signature to drop `mse_weight` (call sites `src/main.rs:449-450`, `src/main.rs:500-501`).
- Within `train_diffusion_ce`:
  - Continuous path chain-rule to ε remains (`src/llm.rs:1559-1562`), minus MSE addition.
  - Compute loss as `sce = symmetric_cross_entropy(...)` and backprop with `symmetric_cross_entropy_gradients(...)` applied to logits, then propagate to ε via scheduler mapping.
  - Discrete-masked path still uses CE/SCE over masked tokens (no scheduler factor).
- Preserve `crate::softmax::Softmax` for probability computation `src/llm.rs:1204` to avoid duplicating softmax.

## Mathematical Correctness
- SCE definition follows `SCE(y,p) = α·CE(y,p) + β·CE(p,y)`; reverse term uses `y_i = 1 for target, ε for others` to avoid `log(0)`.
- Reverse-CE gradient per row derives from softmax Jacobian: `∂/∂z [∑ p_i c_i] = p ∘ (c - p·c)`; CE gradient remains `p - one_hot`.
- CE-only Transformer training unaffected; Diffusion training relies on `x̂₀` reconstruction with CE/SCE pushing probabilities, aligning with language-model likelihood lower bound.

## Efficiency & Compatibility
- Vectorize all ops with `ndarray` (row-wise broadcasting), reuse existing softmax to maintain performance.
- Keep existing types and shapes; avoid extra allocations via in-place updates where safe.
- No API changes beyond removing `mse_weight` and CE helpers in `llm.rs` (call sites updated).

## Unit Tests (`loss.rs`)
- Numerical stability: extreme logits produce finite SCE; verify no NaN/Inf across batches.
- Gradient correctness: finite-difference check on small logits vs `symmetric_cross_entropy_gradients` within tolerance (e.g., 1e-4).
- Symmetry property: `SCE == α·CE(y,p) + β·CE(p,y)` equality test using both functions.
- Edge cases: empty targets, out-of-range token ids, tiny `epsilon`; ensure well-defined outputs and zero gradients where rows are skipped, with `debug_asserts` in dev builds.

## Documentation Updates
- Add rustdoc at `src/loss.rs` describing SCE math, assumptions, and stability.
- Function-level docs in `loss.rs` for each API, with examples.
- Update LLM training rustdoc sections to state diffusion now uses SCE-only; remove MSE references; adjust docstrings near `src/llm.rs:1305-1310` and the diffusion CE section logs `src/llm.rs:1594-1599`.

## Verification
- Run unit tests for `loss.rs` and existing training tests; ensure gradients flow and training logs reflect CE/SCE only.
- Confirm removal points compile: no remaining references to `denoising_loss`, `train_diffusion`, or `diffusion_mse_weight`.
- Sanity-train a tiny batch to ensure throughput comparable to prior CE-only steps.

## Notes
- Paper alignment: LLaDA uses a principled likelihood-based objective; unifying both AR and diffusion under cross-entropy is consistent with maximum likelihood training and avoids MSE denoising objectives.
- Default SCE weights set to `α=1.0, β=0.1, ε=1e-4`; exposed for future tuning if desired, but not required for this refactor.