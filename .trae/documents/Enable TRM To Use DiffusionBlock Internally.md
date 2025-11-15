## Goals
- Update TRM to operate directly over a DiffusionBlock, analogous to its existing TransformerBlock path.
- Keep TRM as a single recursive layer that can switch processing core between TransformerBlock and DiffusionBlock.
- Preserve external TRM Layer API and compatibility.

## Architectural Changes
- TRM core selection:
  - Add optional DiffusionBlock inside TRM; retain existing TransformerBlock.
  - Method `set_timestep(usize)` on TRM to control diffusion timestep; propagate to internal DiffusionBlock during forward.
- Forward recursion:
  - In latent and answer updates, branch on processing core:
    - Transformer path: current manual Pre-LN + attention + FFN + residuals.
    - Diffusion path: call `diffusion.forward_with_timestep(&combined_input, t)`; residuals remain identical.
- Gradients:
  - If diffusion core active: use `DiffusionBlock::compute_gradients` and `apply_gradients` for param updates.
  - If transformer core active: maintain current placeholder gradient behavior for transformer parameters (or minimal alignment with transformer’s gradients if available).
- Parameters and metrics:
  - `parameter_count()` and `weight_norm()` include diffusion when active.

## Config & Builder Updates
- `ModelConfig`: use existing `trm_use_diffusion` flag.
- `TRM::from_model_config`: when `trm_use_diffusion=true`, initialize internal DiffusionBlock.
- `model_builder::build_trm_layers`: always construct TRM; remove earlier layer-stack swapping to pure diffusion when `trm_use_diffusion=true`.

## Tests
- Add TRM tests:
  - Construction with/without diffusion core.
  - Forward shape parity.
  - Diffusion core gradients path exercised (shapes and param grad vector non-empty).

## Logging
- Optional: print a brief note in architecture summary when TRM uses diffusion core.

## Deliverables
- Modified `src/trm.rs` implementing diffusion core support.
- Updated `src/model_builder.rs` to always build TRM.
- Unit tests validating TRM diffusion behavior.
- No external API breaks; TRM remains a `LayerEnum::TRM`.