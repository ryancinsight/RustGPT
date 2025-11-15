## Goals
- Achieve transformer-comparable next-token performance with diffusion-based training.
- Fix loss/gradient formulation, align LR scheduling, and strengthen timestep conditioning.
- Maintain architectural parity with `TransformerBlock` while preserving diffusion semantics.

## Key Findings (Code References)
- Training loop uses only CE on recovered `x0_hat` and re-weights grads again by `ce_weight` (double-scaling) in `src/llm.rs:1414–1417`.
- No epsilon MSE loss; diffusion forward predicts noise epsilon and CE alone under-trains denoiser (`src/llm.rs:1353–1368`, `src/transformer/diffusion_block.rs:492–543`).
- Timestep conditioning is weak (fixed sinusoidal + tiny offsets to gating) (`src/transformer/diffusion_block.rs:503–515`).
- Diffusion training uses constant LR and no warmup/cosine/LARS unlike transformer (`src/llm.rs:1261–1466` vs `src/llm.rs` warmup methods).
- Discrete masked path masks first k tokens rather than random positions (`src/llm.rs:1328–1336`).

## Planned Changes
### 1) Correct Objective and Gradient Mapping
- Add epsilon-prediction loss: L_eps = E[||ε − ε_θ(x_t,t)||²] with optional v-pred parameterization; compute per batch step (DDPM-consistent MSE). 
- Keep CE on logits from `x0_hat` for language supervision; mix losses with schedule λ_ce(t), λ_eps(t). Default: stronger CE at low-noise, stronger MSE mid/high-noise.
- Remove extra CE re-scaling in chain rule: use dL/dε = −√(1−ᾱ_t)/√(ᾱ_t) · dL/dx̂0 without multiplying by `ce_weight` again.
- Implement importance sampling of t or weight normalization so gradient magnitudes are balanced across timesteps.

### 2) Align Optimizer and LR Scheduling
- Use same warmup + cosine annealing as transformer, optionally LARS trust-ratio per layer. Integrate `train_with_warmup` schedule into diffusion CE/MSE training.
- Add gradient clipping by global norm before `apply_gradients` to stabilize denoiser.

### 3) Strengthen Timestep Conditioning
- Replace ephemeral gating offsets with FiLM-style scale/shift derived from learnable time embedding MLP: γ(t), β(t) modulate Norm and FFN activations.
- Make `TimeEmbedding` learnable: parameters + optimizer; optionally small 2-layer MLP.

### 4) Improve Discrete-Masked Variant
- Randomly mask positions across the sequence (uniform or Poisson) instead of prefix masking. Ensure absorbing-state semantics via configured `mask_token_id`.
- Add schedule for masking ratio correlated with t to approximate forward noise level.

### 5) Training Loop Integration (High-Level)
- In `train_diffusion_ce`, for each sample:
  - Sample t; compute `x_t = q_sample(x0, t)`.
  - Predict ε via stacked diffusion blocks.
  - Recover `x0_hat` using scheduler and compute CE on logits → targets.
  - Compute MSE(ε_pred, ε) and mixed loss L = λ_ce(t)·CE + λ_eps(t)·MSE.
  - Backprop: map CE grads to ε via correct chain rule; add ε MSE grads; accumulate and apply with LR schedule and clipping.

### 6) Evaluation and Parity Metrics
- Log per-epoch: CE, MSE, mixed loss, grad norms, effective LR, active heads/experts, routing entropy (existing metrics).
- Compute perplexity on a validation split for direct transformer parity comparison.
- Track loss curves by t to verify balanced training across noise levels.

### 7) Hyperparameters (Initial)
- λ_eps(t) = 1.0; λ_ce(t) = sigmoid((t0 − t)/σ) with t0 ≈ 0.25·T, σ ≈ 0.1·T.
- LR: match transformer default; warmup 5% of steps; cosine to 10% of max LR.
- Clip grad norm to 1.0–2.0.
- Timesteps: keep 1000 for scheduler; training sample steps ≈ 100–200.

## Mathematical Guarantees
- DDPM objective equivalence: minimizing MSE on ε under Gaussian forward q(x_t|x_0) provides a valid lower bound surrogate to NLL; the CE on x̂0 adds supervised signal consistent with sequence modeling.
- Chain rule scaling: dL/dε = −√(1−ᾱ_t)/√(ᾱ_t) · dL/dx̂0 is exact for x̂0 reconstruction from ε, removing incorrect re-weighting.
- FiLM modulation preserves residual block invariants and bounded gradients when γ,β are constrained (e.g., via small init).

## Implementation Touchpoints
- `src/llm.rs`:
  - Extend `train_diffusion_ce` with ε MSE path, loss mixing, corrected gradient mapping, warmup+cosine LR, gradient clipping.
  - Random mask positions for discrete mode.
- `src/transformer/diffusion_block.rs`:
  - Add learnable `TimeEmbedding` or `TimeMLP` and FiLM hooks into `RichardsNorm`/FFN.
- `src/loss.rs`:
  - Add `epsilon_mse` and gradients.
- Logging: ensure `tracing::info` emits CE/MSE/mixed, LR, t-stats.

## Verification Plan
- Unit tests: epsilon loss gradients (finite-difference), chain-rule correctness for dL/dε scaling, FiLM modulation does not change shapes/params counts.
- Integration tests: diffusion block forward/backward parity remains; loss decreases on synthetic data; perplexity converges on small corpus.
- Metrics: compare validation perplexity vs transformer baseline on same data and epochs.

## Risk & Rollback
- If instability at high t, reduce λ_eps(t) in extreme noise, increase clipping, or lower LR.
- Keep discrete-masked path behind a feature flag; default off.

Confirm, and I will implement these changes with precise code edits and tests.