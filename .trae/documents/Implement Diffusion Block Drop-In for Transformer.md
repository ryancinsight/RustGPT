## Goals
- Implement `diffusion_block` as a strict drop-in for `transformer_block` with identical Layer-level I/O and compatible construction.
- Preserve all functionality currently provided by `transformer_block` (norms, residuals, feedforward variants, head-selection, windowing) while enabling diffusion-specific conditioning.
- Provide unit tests for interface and behavior compatibility; add perf logging to compare throughput, memory and accuracy via `cargo run --release --bin main`.
- Analyze and, if needed, redesign attention for diffusion; ensure backward compatibility.

## Current State (Verified)
- `TransformerBlock` API and behavior: `src/transformer/transformer_block.rs:181` implements `Layer` with Pre-LN, `PolyAttention`, residuals, FFN, caching, gradients.
- `DiffusionBlock` exists: `src/transformer/diffusion_block.rs:691` implements `Layer` but uses a custom `DiffusionAttention` with simplified gradients and separate time/noise machinery.
- Attention: `PolyAttention` is advanced and configurable; supports a non-causal path via `forward_impl(input, causal)`: `src/attention/poly_attention.rs:239`.
- Builder: `model_builder.rs` selects Transformer vs Diffusion stacks; `main.rs` defaults to Diffusion.

## Attention Analysis & Decision
- Suitability: `PolyAttention` already provides stable training, gating, CoPE, head selection, and windowing; it is preferable to re-use it for diffusion with bi-directional masks.
- Modifications:
  - Use `PolyAttention::forward_impl(input, causal=false)` inside diffusion to enable bi-directional attention.
  - Introduce time conditioning minimally-invasive: add a per-token, per-head bias term derived from `TimeEmbedding` that modulates the gating path (`alpha_g`, `beta_g`) without breaking existing invariants.
  - Preserve backward compatibility: a `causal_attention: bool` flag in `DiffusionBlockConfig` will switch between `causal=true` (AR-compatible) and `false` (diffusion-optimized).

## Architectural Changes
- Replace `DiffusionAttention` with a wrapper that delegates to `PolyAttention` and injects time-conditioning into gating only (no K/V shape changes) to retain API and gradients.
- Align configs:
  - Implement `impl From<TransformerBlockConfig> for DiffusionBlockConfig` and `DiffusionBlock::from_model_config` to mirror transformer's parameter derivation; add diffusion-only fields with sane defaults.
  - Maintain the same Layer trait semantics (`forward(&mut, &Array2<f32>) -> Array2<f32>`) and cache layout.
- Timestep handling:
  - Keep `current_timestep` field and `set_timestep(t)`; `LLM` sets timestep before diffusion forwards (`llm.rs:1432`).

## Implementation Steps
1) Unify DiffusionBlock internals with PolyAttention
- In `src/transformer/diffusion_block.rs`:
  - Replace `attention: DiffusionAttention` with `attention: PolyAttention` (`lines ~476-486`).
  - In `forward_with_timestep`, call `self.attention.forward_impl(&norm1_out, self.config.causal_attention)`; if `false`, bi-directional; retain residual structure (`lines ~598-631`).
  - Inject time-conditioning: compute a small vector from `time_embedding.forward(t, num_timesteps)` and modulate gating via lightweight offsets to `alpha_g`/`beta_g` (applied per-head) before `forward_impl`; reset after forward to avoid state drift.
  - Gradient path: delegate `compute_gradients` and `apply_gradients` directly to `PolyAttention` + FFN, identical to `TransformerBlock` (`transformer_block.rs:238-321`).

2) Config compatibility & constructors
- Add `impl From<TransformerBlockConfig> for DiffusionBlockConfig` ensuring identical shared fields; provide defaults for diffusion-only fields (`time_embed_dim=embed_dim`, `num_timesteps=1000`, `noise_schedule=Cosine { s: 0.008 }`, `causal_attention=false`).
- Ensure `DiffusionBlock::from_model_config` mirrors transformer's logic (`transformer_block.rs:140-162`).

3) Fix tests and add compatibility tests
- Update `diffusion_block.rs` tests to use `set_timestep` + `forward` or `forward_with_timestep`; remove incorrect `forward(&input, 500)` call (`diffusion_block.rs:927-932`).
- Add new tests:
  - Interface parity: construct blocks from the same `ModelConfig` and verify `forward` I/O shapes, parameter counts, and `LayerEnum` behavior.
  - Gradient compatibility: run `compute_gradients` on both blocks and assert param-grad vector length equality; shape checks on input grads.
  - Diffusion conditioning: verify outputs change with timestep; denoising loss decreases when training steps are applied.

4) Performance logging & comparison
- Add lightweight perf logging in `main.rs` around training and generation paths:
  - Throughput: tokens/sec and samples/sec via `std::time::Instant`.
  - Memory: parameter count (already printed) + optional process RSS using `sysinfo` (optional if allowed) or omit external dep and report param-derived memory estimate.
  - Accuracy: cross-entropy for Transformer, denoising MSE for Diffusion (already computable via `DiffusionBlock::denoising_loss`).
- Preserve the exact run command: `cargo run --release --bin main`; print a summary block with both architectures when toggled.

## Backward Compatibility
- No breaking changes to `TransformerBlock` or `LayerEnum`.
- `DiffusionBlock` now uses the same attention foundation; if `causal_attention=true`, it behaves AR-compatible, enabling drop-in usage where desired.
- `model_builder` remains unchanged; optional follow-up: add a `ModelConfig` flag to build Transformer stacks with `DiffusionBlock` for A/B without touching downstream code.

## Unit Tests (Scope)
- Location: co-located in `transformer_block.rs` and `diffusion_block.rs` modules.
- Cases:
  - Creation/from_model_config parity (embed_dim/heads/window/head_selection).
  - Forward/backward shape parity on random inputs.
  - Gradients: count and basic numeric sanity (finite, non-NaN).
  - Diffusion processes: noise scheduler properties, q_sample/posterior_mean invariants.
  - Property-based tests (proptest) for stability under random inputs/timesteps.

## Benchmark Plan
- Single command run: `cargo run --release --bin main`.
- Inside `main.rs`, print:
  - Throughput: training examples/sec and generation tokens/sec.
  - Memory: parameters and estimated bytes.
  - Accuracy: final losses (Transformer CE, Diffusion MSE).
- Execute twice by switching `ArchitectureType::{Transformer, Diffusion}`; combine logs for comparison.

## Documentation
- Add rustdoc module docs for `diffusion_block` explaining time conditioning, scheduler math, and attention changes.
- Inline doc comments on config and public methods; integration guidelines: how to set `causal_attention` and use as drop-in.

## Deliverables & Acceptance
- Fully functional `diffusion_block` implemented on top of `PolyAttention`, matching `TransformerBlock` behavior at the Layer level.
- Unit tests for compatibility and diffusion functions passing.
- Perf comparison printed by `main.rs` under the given command.
- Documentation providing integration guidelines and attention changes.
- Backward compatibility preserved; no API breaks.
