## Research Summary (Updated for LLaDA)
- LLaDA defines a masked discrete diffusion process over tokens: forward masking with ratio t∼U[0,1], reverse denoising predicts masked tokens with a Transformer; optimized via a likelihood lower bound (ELBO), competitive with ARMs at 8B scale [LLaDA 2502.09992v3: https://arxiv.org/html/2502.09992v3, PDF: https://arxiv.org/pdf/2502.09992, Demo: https://ml-gsai.github.io/LLaDA-demo/].
- Relation to D3PM: absorbing-state masking (special [MASK]) yields stable training and parallel sampling; discrete transition matrices Q_t govern corruption; auxiliary CE losses improve performance [D3PM overview: https://www.emergentmind.com/topics/masked-discrete-diffusion-models].
- Attention and masks: denoiser uses bidirectional attention; masking ratio controls corruption level per step; positional encodings remain applicable; sampling can leverage flexible remasking.

## Current Codebase Fit and Gaps
- Our `DiffusionBlock` implements continuous DDPM-style over embeddings with non-causal attention and denoising objectives (src/transformer/diffusion_block.rs:336–575). This differs from LLaDA’s discrete masked diffusion.
- Training functions (`train_diffusion`, `train_diffusion_ce`) already bridge noise prediction to CE, but remain continuous (src/llm.rs:1305–1572).
- PolyAttention respects causal vs non-causal in forward but not in gradients (src/attention/poly_attention.rs:404–408), which must be fixed for bidirectional denoising.

## Implementation Plan (Drop‑in Replacement, LLaDA‑style)
- Discrete Masked Diffusion Process
  - Add `DiscreteMaskScheduler` with absorbing-state [MASK] and ratio schedule t∼U[0,1]; implement structured Q_t with `Q_t = (1−β_t)I + β_t · 1·e_mask^T` and efficient sampling without materializing full matrices (windowed masking by token index) (new module under `src/diffusion/discrete.rs`).
  - Integrate into `DiffusionBlock` with new config `discrete_masked: bool` and `mask_token_id`; when enabled, operate over token indices/logits rather than raw embeddings.

- Denoiser Architecture and Interfaces
  - Keep `DiffusionBlock` interface intact (`Layer` trait); internally switch between continuous and discrete paths via config.
  - For discrete mode: input is masked token embeddings; forward predicts x̂₀ tokens via output projection + CE head; enable bidirectional attention (`causal_attention=false`).
  - Implement flexible remasking per LLaDA: at each step, allow a subset of predicted tokens to remain masked based on confidence thresholds; expose strategy in config.

- Training Pipeline (Pretraining + SFT Alignment)
  - Update `train_diffusion_ce` to LLaDA regime: random global mask ratio per sequence (U[0,1]) in pretraining; during SFT, mask only response tokens; optimize ELBO proxy + CE auxiliary loss (retain existing CE path, add ELBO term from discrete scheduler).
  - Add optional classifier-free guidance hooks compatible with SMDM/LLaDA guidance (configurable guidance weight); keep default off.

- Attention and Gradient Corrections (Required regardless of mode)
  - Fix PolyAttention gradient masking: cache `last_causal` set in `forward_impl` and use it in `compute_gradients` to set `j_end` correctly (src/attention/poly_attention.rs:404–408, 676–687).

- Continuous DDPM Corrections (Parity and mathematical soundness)
  - Correct cosine schedule to derive per‑step β_t from ᾱ_t (src/transformer/diffusion_block.rs:116–130).
  - Correct posterior mean to use per‑step α_t (src/transformer/diffusion_block.rs:188–202).
  - Use Gaussian noise in sampling (src/transformer/diffusion_block.rs:539–566).
  - Parameter accounting: exclude non‑learnable time embedding from `parameters()` (src/transformer/diffusion_block.rs:598–605).
  - Remove duplicate time embedding call (src/transformer/diffusion_block.rs:468, 475).

## Tests and Benchmarks
- Discrete mask scheduler
  - Unit: absorbing behavior, mask ratios, ELBO term numerics.
- Denoiser parity
  - Shapes: identical to transformer_block; masks: causal vs non‑causal behaviors.
  - Gradients: finite and consistent across masking modes; compare discrete vs continuous.
- Performance
  - Forward latency: `TransformerBlock` vs `DiffusionBlock` (discrete + continuous) for seq {64, 512}; ensure ±10% parity in block throughput.
- Sampling
  - Parallel masked sampling: validate token unmasking progression; Gaussian noise moments (continuous path).

## Documentation
- Describe LLaDA‑style masked diffusion usage, configuration flags, training phases, and guidance.
- Clarify differences vs AR transformers; note bidirectional attention and remasking strategy.
- Document PolyAttention gradient masking fix and DDPM math corrections.

## Acceptance Criteria
- `DiffusionBlock` remains drop‑in (Layer trait), now supporting discrete masked diffusion compatible with LLaDA.
- All math corrections validated by tests; gradients finite; no NxN materialization.
- Benchmarks show acceptable parity; sampling and training conform to LLaDA principles.
- Docs updated with interface changes and usage guidance.