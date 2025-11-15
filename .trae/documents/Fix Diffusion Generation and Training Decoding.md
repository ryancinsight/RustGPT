## Goals
- Make diffusion generation use the full network stack and decode via the trained output projection instead of nearest-embedding heuristics.
- Ensure predicted noise and reverse steps leverage all DiffusionBlocks.
- Improve prompt conditioning during diffusion generation.
- Remove deprecated RNG usage and minor warnings.

## Changes
- LLM::sample_diffusion
  - Use all DiffusionBlocks: for each reverse step t, set timestep for all and compute predicted noise by forwarding x_t through all DiffusionBlocks sequentially.
  - After finishing reverse steps, pass the denoised embeddings through `DynamicTanhNorm` (if present) and `OutputProjection` to obtain logits.
  - Decode greedily per position from logits; stop at EOS when encountered.
  - If a prompt is provided, initialize the first `prompt_tokens.len()` rows of `x_t` with their token embeddings to condition the process; keep noise for the remaining positions.
- Train Diffusion (minor clean-up)
  - Keep denoising MSE training; no change to objective.
  - Remove deprecated `thread_rng` usage and `r#gen` calls; use `rng.random::<f32>()`.
- Tests
  - Add a unit test to verify that `sample_diffusion_with_prompt` uses `OutputProjection` by checking that logits are produced and decoding does not use nearest embedding.

## Verification
- Run `cargo run --release --bin main -- --diffusion` and `--trm --diffusion` to confirm diffusion outputs change with prompt and differ from Transformer-only runs.
- Observe improved outputs beyond minimal punctuation and echoes.

## Scope
- Targeted edits in `src/llm.rs` only for diffusion sampling.
- No public API changes; backward compatible.
