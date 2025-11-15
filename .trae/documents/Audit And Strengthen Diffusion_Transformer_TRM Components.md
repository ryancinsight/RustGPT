## Audit Report
### DiffusionBlock
- Forward implementation: Pre-norm → PolyAttention (time-conditioned gating) → residual → pre-FFN norm → FFN → residual with cached states is complete (`src/transformer/diffusion_block.rs:491-543`). Noise scheduler math (cosine/linear/quadratic) clamps ᾱ and β for stability (`src/transformer/diffusion_block.rs:116-146, 148-176, 183-192, 194-208`).
- Backward: Delegates to FFN/attention/Norm, returns input grads (`src/transformer/diffusion_block.rs:602-609`). Grad-graph cached and used in `compute_gradients` (`src/transformer/diffusion_block.rs:640-677`).
- Issue: Residual gradient split uses 0.5 scaling for the two branches (`src/transformer/diffusion_block.rs:651-652`). Mathematically, for `residual = input + attn_out`, both branches should receive full gradient (not halved). This under-trains both paths and biases learning.
- Issue: Time-conditioning via mean scalar `time_bias` modulates `alpha_g/beta_g` ephemeral values (`src/transformer/diffusion_block.rs:503-515`) but carries no learnable path and no gradient to time embedding (by design). Recommend switching to per-head FiLM-style conditioning with learnable scales and keeping restoration to avoid drift.
- Stability: Posterior mean derivation follows DDPM; ᾱ clamping prevents log/√ domain errors. Sampling loops are correct but CPU-only and per-element RNG, which is slow.
- Performance: Multiple clones in forward and gradient paths; per-head gating modification in Rust loops; no parallelization; attention and FFN compute could benefit from SIMD or GPU.

### TransformerBlock
- Forward path matches canonical structure with cached intermediates (`src/transformer/transformer_block.rs:186-212`).
- Backward is complete and consistent (`src/transformer/transformer_block.rs:214-228`).
- Issue: Same residual gradient halving in `compute_gradients` (`src/transformer/transformer_block.rs:265-266`). Should pass full gradient to both branches.
- Observability/metrics are adequate; parameter counting and weight norm exposed.

### TRM
- Forward recursion implements latent updates and answer refinement, with optional diffusion conditioning (`src/trm.rs:331-373, 404-430`). Uses in-place scaled add for latent stability (`src/trm.rs:393-394`). Early-stopping heuristic present (`src/trm.rs:456-466`).
- Issue (critical): `compute_gradients_trm` returns zero-shaped parameter gradients rather than true transformer grads when diffusion is None (`src/trm.rs:532-539`). This breaks learning and contradicts documented gradient flow. Apply path then attempts to consume mismatched shapes (`src/trm.rs:556-574`).
- Loss and training helpers use MSE locally; acceptable for autoencoding but not unified with the CE-based language objective.
- Stability fallbacks return input unchanged on anomalies (`src/trm.rs:470-476, 479-485`). Good for robustness but may hide defects.

## Mathematical Correctness
- Residual gradient propagation must not dampen by 0.5 factors; each branch in sum receives full upstream gradient.
- Diffusion posterior mean uses `α_t` and `ᾱ_t` with correct coefficients; confirm Jacobians in tests.
- TRM gradients must flow through TransformerBlock sub-components using cached states, matching theoretical derivatives.

## Performance Opportunities
- Replace per-element RNG loops in diffusion sampling with vectorized generation and batched operations; optionally parallelize via `rayon`.
- Reduce clones by using views/slices; reuse buffers for temporaries.
- Consider SIMD via `std::simd` in FFN and norm paths; consider `wgpu` kernels for attention and FFN for large sequences.
- Make time-conditioning per-head vectorized and remove inner loops.

## Test Suite Additions
- Numerical gradient checks (finite differences) for:
  - TransformerBlock `compute_gradients` on small inputs.
  - DiffusionBlock `compute_gradients` including scheduler chain when used by higher-level code.
- Edge/stability tests:
  - Extreme timesteps and clamped ᾱ/β values.
  - Long sequences and windowed attention boundaries.
- TRM gradient flow:
  - Verify non-zero, finite parameter gradients; shape alignment with underlying transformer.
- Benchmarks (criterion):
  - Forward/backward latency and throughput for each component across sequence lengths.
  - Memory footprint snapshots during recursion.

## Documentation Updates
- Module-level docs for diffusion_block detailing forward/reverse processes and conditioning.
- Clarify residual gradient rules in both blocks; document cached intermediates and usage.
- TRM docs: add explicit gradient computation section reflecting actual implementation.

## Implementation Plan
1. Fix residual gradient splitting to full propagation in both components.
2. Implement proper TRM parameter gradient computation by delegating to TransformerBlock `compute_gradients` with correct cached states; remove zero tensors.
3. Optimize diffusion sampling and forward by reducing clones, vectorizing RNG, and introducing optional `rayon` parallelism.
4. Add `criterion` benchmarks for forward/backward paths and TRM recursion.
5. Add numerical gradient tests and stability edge-case tests for all three components.
6. Introduce learnable per-head FiLM-style time conditioning (feature-gated) and document.
7. Update module-level docs and function docstrings to reflect math and IO.

## Deliverables
- Code fixes (residual grads, TRM gradients, sampling optimizations).
- New tests covering gradients and stability; benchmarks.
- Documentation updates with math references and IO specs.

## Verification
- Run unit tests and gradient checks; ensure no numerical instability.
- Compare performance before/after on representative sequence lengths.
- Validate TRM learning with non-trivial targets producing non-zero updates.
