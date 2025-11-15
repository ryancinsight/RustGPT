## Problem
- Diffusion path prints "Generating response using trained diffusion model..." regardless of whether any training occurred. Users can see identical outputs after removing training, making the message misleading.

## Fixes
- Add a `trained_flag` to `main.rs` that toggles true only when any train function is invoked with `epochs > 0` and returns successfully.
- Capture parameter norms before and after training to produce an objective training delta:
  - Implement `LLM::total_weight_norm()` that sums `layer.weight_norm()` across all layers.
  - Log `param_norm_before`, `param_norm_after`, and `delta` to verify training effect.
- Update After-Training section:
  - If `trained_flag` is false, print "Generating response using current diffusion model (no training run)."
  - If `trained_flag` is true, print "Generating response using trained diffusion model (Δ‖θ‖={delta:.4})."
  - Apply similar messaging to TRM/Transformer modes for consistency.
- Guard instruction tuning and pretraining blocks with `epochs > 0` and set `trained_flag` accordingly. Skip printing misleading pretraining/tuning headings if epochs are zero.

## Instrumentation
- Log per-epoch SCE loss and grad norm across modes; already present.
- Add seed and steps in generation logs for reproducibility.

## Tests
- Add integration test that sets epochs to zero and confirms messaging indicates no training and param norm delta equals zero.
- Add test with non-zero epochs on a tiny dataset and assert delta != 0 and message reflects trained model.

## Verification
- Run full suite; confirm After-Training block is truthful under both trained and untrained runs.
- Validate diffusion+TRM path behaves consistently with message gating.
