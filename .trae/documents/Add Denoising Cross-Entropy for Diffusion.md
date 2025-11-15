## Goals
- Implement a denoising cross-entropy (DCE) training variant for diffusion to match CE-style logs/metrics, enabling apples-to-apples comparison with Transformer/TRM.
- Combine denoising MSE with CE over the output projection of recovered x0 (configurable weights), or run CE-only.

## CLI Additions
- `--diffusion_ce` (bool): use DCE pipeline for diffusion pretraining.
- `--diffusion_ce_weight <f32>` (default: 0.5): CE loss weight.
- `--diffusion_mse_weight <f32>` (default: 0.5): MSE loss weight. If `--diffusion_ce` and `diffusion_mse_weight=0`, runs CE-only.

## Training Pipeline (LLM::train_diffusion_ce)
1) Tokenize batch sequences; slice `input_ids = seq[..len-1]` and `target_ids = seq[1..]`.
2) Embed: `x0 = TokenEmbeddings.forward([input_ids])` → shape `[seq_len, embed_dim]`.
3) Sample noise ε and timestep t; compute `x_t = NoiseScheduler.q_sample(x0, t, ε)`.
4) Predict noise: forward `x_t` through all DiffusionBlocks with `set_timestep(t)` to get `ε_θ`.
5) Recover x0_hat: `x0_hat = (x_t - sqrt(1-ᾱ_t) * ε_θ) / sqrt(ᾱ_t)` using scheduler’s `sqrt_alpha_cumprod(t)` and `sqrt_one_minus_alpha_cumprod(t)`.
6) Logits: pass `x0_hat` through final `DynamicTanhNorm` (if present) and `OutputProjection` to get `[seq_len, vocab_size]`.
7) Loss:
- MSE: `mse = mean((ε_θ - ε)^2)`.
- CE: standard token-level CE on logits vs `target_ids`.
- Total: `loss = mse_weight*mse + ce_weight*ce`.
8) Gradients:
- CE grads: `dL/dlogits` → OutputProjection.backward → `grad_hidden` (shape `[seq_len, embed_dim]`), then through final norm (if present) to get `grad_x0_hat`.
- Chain rule to predicted noise: `grad_eps = grad_x0_hat * (-sqrt(1-ᾱ_t)/sqrt(ᾱ_t))` (broadcast scalar).
- MSE grads: `grad_eps += 2*(ε_θ - ε)/N`.
- Backprop `grad_eps` through DiffusionBlocks with `compute_gradients(input=x_t, grads=grad_eps)` and `apply_gradients`.
- Optionally backprop into TokenEmbeddings via `grad_x0` if desired; default: leave embeddings updated via CE path only when explicitly enabled (keep simple: no embedding update for DCE unless requested; can add `--diffusion_update_embeddings` flag).
9) Logging: print per-epoch `loss`, `mse`, `ce`, and `grad_norm` formatted like `train_with_warmup` for consistency.

## Integration
- In `main.rs`, if `--diffusion_ce` present during diffusion pretraining, call `train_diffusion_ce(pretraining_examples, epochs, lr, batch_size, ce_weight, mse_weight)` instead of `train_diffusion`.
- Instruction tuning stays with CE (`train_with_warmup`).

## Tests
- Unit: verify `x0_hat` recovery formula correctness by round-trip (`q_sample` then recover) on synthetic data.
- Integration: small dataset run prints CE and MSE (when both enabled), and losses decrease.
- Gradient shapes: ensure DiffusionBlock `compute_gradients` receives correct shapes and param gradients non-empty.

## Notes
- Keeps backward compatibility; no changes to existing CE training for Transformer/TRM.
- Defaults provide balanced MSE+CE; adjust via flags for experiments.