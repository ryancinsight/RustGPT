## Root Cause
- Warnings show latent gradient shape mismatch: expected `[(1, embed_dim=128)]` vs got `[(1, seq_len=7)]`.
- In TRM, `apply_gradients` assumes the last entry in `param_grads` is the latent init gradient. Under certain training paths, the last gradient can instead reflect sequence-shaped tensors, causing mismatch.
- Gradient propagation ordering and shape contracts within `compute_gradients_trm` and `apply_gradients` are insufficiently explicit; relying on positional last-element convention is fragile.

## Fixes
- Explicit latent gradient slot:
  - Change TRM `compute_gradients_trm` to return `(input_grads, GradPack)` where `GradPack` contains separate vectors: `attn_params`, `ffn_params`, and `latent_init_grad` (optional). Avoid positional assumptions.
  - Update TRM `apply_gradients` to consume `GradPack` and apply each category explicitly; validate latent gradient shape against `latent_init` and skip with warning if mismatched.
- Shape consistency enforcement:
  - Ensure `final_input_grads` computed against `answer_input` has shape `(batch, embed_dim)`.
  - If any intermediate produces `(batch, seq_len)`, identify and correct: inputs to transformer subcomponents must be embedding-shaped not token-id shaped.
- Training pipeline mapping:
  - Confirm LLM training loop calls `TRM.compute_gradients` layer-wise only, and that returned param gradients belong to TRM (no leakage from other layers).
- Robust latent gradient derivation:
  - Compute latent init gradient from `final_input_grads` by aggregating across batch to `(1, embed_dim)` (e.g., mean over batch) rather than a placeholder. This guarantees matching shape and meaningful updates.
  - Explicitly set `latent_init_grad` last in `GradPack` to eliminate ambiguity.

## Instrumentation
- Add structured logs in TRM `compute_gradients_trm` and `apply_gradients`:
  - Shapes of `final_input_grads`, counts of attn/ffn param grads, and latent grad shape.
  - When a mismatch occurs, log expected vs actual, layer index, and skip application.
- In diffusion training, log embedding parameter norm deltas per epoch to confirm actual updates.

## Tests
- Unit tests for TRM:
  - Verify `compute_gradients_trm` returns latent grad shape `(1, embed_dim)` and that `apply_gradients` updates `latent_init` without warnings.
  - Shape mismatch test: intentionally pass a `(1, seq_len)` latent grad and assert it is detected and skipped.
- Integration test in LLM:
  - Diffusion + TRM training on small synthetic dataset; assert embedding norms change across epochs and loss decreases.

## Performance & Safety
- No significant performance impact; separating gradient categories improves clarity without extra heavy compute.
- Maintain existing gradient thresholds; keep clamps and numerical stability measures.

## Deliverables
- Code refactor introducing `GradPack` for TRM gradients.
- Updated `apply_gradients` to explicitly consume latent gradient.
- Logs and tests covering the shape contracts and training updates.

## Verification
- Run full test suite; ensure no gradient mismatch warnings during TRM diffusion training.
- Observe non-zero embedding norm deltas and monotonic decrease in SCE loss over epochs for a small dataset.
