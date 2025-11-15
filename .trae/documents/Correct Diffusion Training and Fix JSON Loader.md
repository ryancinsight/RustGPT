## Goals
- Implement proper diffusion denoising training (MSE on predicted noise) in `LLM::train_diffusion`.
- Fix JSON loader to accept array-of-objects with `text` strings.
- Add minimal epoch-level diffusion MSE logging.

## Changes
- `src/llm.rs`
  - Replace `train_diffusion` with a dedicated denoising loop:
    - Use `TokenEmbeddings.forward` to get `x0`.
    - Sample `t` and noise; compute `x_t` via `NoiseScheduler.q_sample` from first `DiffusionBlock`.
    - Forward through all `DiffusionBlock`s using `forward_with_timestep`.
    - Compute MSE loss and output grads; backprop per-block via `compute_gradients` and `apply_gradients`.
    - Backprop final input grads to `TokenEmbeddings` and apply gradients.
    - Aggregate loss per epoch and log.
  - Use `rand_distr::Normal` for noise; avoid deprecated RNG APIs.

- `src/dataset_loader.rs`
  - In `get_data_from_json`, try parsing `Vec<String>`; if it fails, try `Vec<{text: String}>` and extract `text`.
  - Retain relaxed fallback only if necessary.

## Verification
- Build and run both architectures; check diffusion training logs report MSE values.
- Ensure loader accepts object-based JSON and yields non-empty sequences.
- Confirm no changes to transformer CE training path.

## Scope
- Targeted edits only in `llm.rs` and `dataset_loader.rs`. No API or trait changes.
