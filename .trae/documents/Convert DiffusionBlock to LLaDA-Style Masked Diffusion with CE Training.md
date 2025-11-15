## Goals
- Make DiffusionBlock a true LLaDA-like masked diffusion denoiser: bidirectional attention, timestep conditioning, masked-token reconstruction.
- Unify pretraining and chat-tuning with transformer/TRM (cross-entropy over next tokens), the only difference being the timestep component.
- Keep DiffusionBlock a drop-in layer with the same Layer trait; preserve final OutputProjection for logits.

## Key Changes
- Replace continuous DDPM internals with discrete masked diffusion behavior (absorbing-state `<mask>`), optional continuous path retained behind a feature flag.
- Timestep conditioning: sinusoidal time embedding injected into attention (already present) and used consistently across forward and gradients.
- Attention: always non-causal in diffusion; gradients updated to honor non-causal masking (already fixed).
- Output: DiffusionBlock continues to output embeddings; logits computed by the existing final OutputProjection layer, enabling CE training identical to transformer/TRM.

## Training Flow (Unified CE)
- Pretraining (Diffusion):
  - Same loop as transformer/TRM: minimize next-token cross-entropy.
  - For diffusion blocks, sample a global mask ratio t ∼ U[0,1], mask K tokens with `<mask>`, set `set_timestep(t)` on each DiffusionBlock, forward → logits → CE.
  - No MSE denoising term by default; optional denoising (continuous) can be toggled via flags.
- Chat-tuning (Diffusion):
  - Same CE loop as transformer/TRM; mask only response tokens per sequence; set timestep and forward.

## Implementation Steps
- Config and Interfaces:
  - DiffusionBlockConfig: set `discrete_masked=true` by default for diffusion architecture and require `mask_token_id` (derived from vocab). Keep continuous schedule fields optional.
  - Ensure `DiffusionBlock::from_model_config` sets `discrete_masked=true` and `mask_token_id=vocab.encode("<mask>")` when building diffusion networks (currently defaults to false; update builder to pass a block config or set via a constructor with mask id).
- Forward path:
  - Keep current norm→attention→ffn→residual structure.
  - Use time embedding t throughout the forward pass; remove duplicated calls (already removed) and ensure gating offsets are the only modulation unless we add more conditioning.
- Training API:
  - Pretraining: route diffusion architecture to `train_diffusion_ce` always; default `ce_weight=1.0, mse_weight=0.0`.
  - Chat-tuning: also use `train_diffusion_ce` with masking only over response tokens and `t∼U[0,1]` per sequence.
  - Ensure `train_diffusion_ce` in discrete mode constructs masked ids, embeds them, sets timesteps, and computes CE-only gradients (already partially implemented); add branch to mask only response tokens for SFT.
- Sampling (masked diffusion):
  - Implement masked iterative unmasking: start from all masked (or masked prompt response region), run denoiser for S steps, each step remask low-confidence token positions with a threshold, and update embeddings/logits until convergence; use existing OutputProjection for token selection.

## Code Touches (by file)
- src/transformer/diffusion_block.rs:
  - Default `discrete_masked=true` in config when building diffusion networks; keep `mask_token_id`.
  - Keep `set_timestep(t)` and forward with bidirectional attention.
- src/model_builder.rs:
  - When `ArchitectureType::Diffusion`, create DiffusionBlocks with `discrete_masked=true` and `mask_token_id=vocab.encode("<mask>")`.
- src/llm.rs:
  - Route diffusion pretraining and chat-tuning to `train_diffusion_ce`, using `ce_weight=1.0, mse_weight=0.0` by default.
  - In `train_diffusion_ce`, add an SFT branch to mask only response tokens, and ensure every diffusion block receives `set_timestep(t)` before forward.
- src/encoding/vocabulary.rs:
  - `<mask>` token included (already added).

## Tests and Benchmarks
- Unit tests:
  - Verify masked scheduler masks ratio K≈t·seq_len and preserves absorbing state.
  - Validate that transformer vs diffusion (discrete) forward shapes/logits are identical for the same inputs when `t=0` (no masking), confirming parity.
  - Ensure gradients remain finite and correct under non-causal attention.
- Training checks:
  - Pretraining CE loss decreases over synthetic data for both transformer and diffusion with same hyperparameters.
  - Chat-tuning masks only response tokens; verify CE gradients only touch response positions.
- Performance:
  - Compare forward latency of TransformerBlock vs DiffusionBlock under seq_len {64, 512} to ensure similar throughput; confirm no NxN materialization.

## Backward Compatibility
- Keep continuous DDPM paths (NoiseScheduler, denoising_loss, sample()) behind feature flags, disabled by default.
- Layer trait unchanged; DiffusionBlock remains drop-in.

## Acceptance Criteria
- Diffusion architecture trains with CE in both pretraining and chat-tuning, identical loops to transformer/TRM, with only timestep conditioning as a difference.
- Diffusion forward uses non-causal attention; gradients correct across masking modes.
- Masked diffusion sampling implemented for generation.
- All tests pass; performance parity within ±10% of transformer on forward throughput.

## Next Actions
- Implement builder and main routing changes for default CE in diffusion, pass mask id, and enable discrete masked diffusion by default.
- Update `train_diffusion_ce` with SFT masking behavior and consistent timestep handling.
- Add masked sampling utility for diffusion generation and minimal tests to validate its behavior.