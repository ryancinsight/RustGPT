# Diffusion Block

This module implements a diffusion-conditioned transformer block that replaces autoregressive prediction with denoising, conditioning the block via a learnable time embedding projected into FiLM-style modulation of the normalized activations.

## Overview

- Pre-attention normalization `LN₁`, FiLM modulation from timestep conditioning
- PolyAttention (bi-directional by default; causal optional) with residual
- Pre-FFN normalization `LN₂`, FiLM modulation
- Feedforward (RichardsGlu or Mixture-of-Experts) with residual
- Output interpreted as noise prediction `ε̂(x_t, t)` or `v̂(x_t, t)`; training target mapped accordingly

## Architecture

```
x_t ─► LN₁ ─► FiLM(γ_attn, β_attn) ─► Attention ─┐
                                                 │
                                                 ├─► Residual ─► LN₂ ─► FiLM(γ_ffn, β_ffn) ─► FFN ─┐
                                                 │                                                │
                                                 └─────────────────────────────────────────────────┼─► Residual ─► ε̂(x_t, t)
```

Time conditioning path:

```
t ─► TimeEmbedding e_t ∈ ℝ^{d_t}
      ├─► h = tanh(e_t W₁ + b₁)
      └─► [γ_attn, β_attn, γ_ffn, β_ffn] = h W₂ + b₂ ∈ ℝ^{4d}
```

## Mathematical Specification

Let `x_t ∈ ℝ^{T×d}` be the noisy input at timestep `t`, `e_t ∈ ℝ^{d_t}` be the sinusoidal time embedding. Define

- `h = tanh(e_t W₁ + b₁)` with `W₁ ∈ ℝ^{d_t×h_t}`, `b₁ ∈ ℝ^{h_t}`
- `[γ_attn, β_attn, γ_ffn, β_ffn] = h W₂ + b₂` where each `γ_*, β_* ∈ ℝ^{d}`
- FiLM modulation `FiLM(Activ, γ, β): Activ' = γ ⊙ Activ + β`

Block equations:

- `Z₁ = LN₁(x_t)`; `Z₁' = FiLM(Z₁, γ_attn, β_attn)`
- `A = Attn(Z₁')`, then `R₁ = x_t + A`
- `Z₂ = LN₂(R₁)`; `Z₂' = FiLM(Z₂, γ_ffn, β_ffn)`
- `F = FFN(Z₂')`, then `Y = R₁ + F`

Prediction mapping:

- Epsilon-prediction: `ε̂ = Y`
- V-prediction: `ε̂ = √ᾱ_t · Y + √(1-ᾱ_t) · x_t`

Training targets:

- If predicting `ε`: target `= ε`
- If predicting `v`: target `= √ᾱ_t · ε − √(1-ᾱ_t) · x₀`

## Gradient Invariants

With upstream gradient `∂L/∂Y`:
- Split at output residual: `∂L/∂R₁ = ∂L/∂Y`, `∂L/∂F = ∂L/∂Y`
- Through FFN: `(∂L/∂Z₂', ∂L/∂θ_ffn)` from `FFN.backward(Z₂', ∂L/∂F)`
- FiLM (FFN path): `Z₂' = γ_ffn ⊙ Z₂ + β_ffn` ⇒
  - `∂L/∂Z₂ = γ_ffn ⊙ ∂L/∂Z₂'`
  - `∂L/∂γ_ffn = Σ_t Σ_d (Z₂[t, d] · ∂L/∂Z₂'[t, d])`
  - `∂L/∂β_ffn = Σ_t Σ_d ∂L/∂Z₂'[t, d]`
- Through `LN₂`: `(∂L/∂R₁)_from_ffn, ∂L/∂θ_ln2)`
- Combine residual-1: `G₁ = ∂L/∂R₁ + (∂L/∂R₁)_from_ffn`
- Split residual-1: `∂L/∂x_t_direct = G₁`, `∂L/∂A = G₁`
- Attention path: `(∂L/∂Z₁', ∂L/∂θ_attn)`
- FiLM (Attn path): `Z₁' = γ_attn ⊙ Z₁ + β_attn` ⇒
  - `∂L/∂Z₁ = γ_attn ⊙ ∂L/∂Z₁'`
  - `∂L/∂γ_attn = Σ_t Σ_d (Z₁[t, d] · ∂L/∂Z₁'[t, d])`
  - `∂L/∂β_attn = Σ_t Σ_d ∂L/∂Z₁'[t, d]`
- Through `LN₁`: `(∂L/∂x_t_norm, ∂L/∂θ_ln1)`; final `∂L/∂x_t = ∂L/∂x_t_direct + ∂L/∂x_t_norm`

Time conditioning gradients:
- Backpropagate `∂L/∂γ_*`, `∂L/∂β_*` through `W₂, b₂` to `h`, then through `tanh` to `e_t`, then to `W₁, b₁`

V-prediction scaling:
- When mapping `Y → ε̂`, gradients w.r.t. block output scale by `√ᾱ_t`; an extra additive path to `∂L/∂x_t` scales by `√(1-ᾱ_t)`

## Complexity Analysis

Parameters: `N` (seq len), `D` (embed dim), `H` (heads), `d_h = D/H` (head dim), `W` (attention window, `W ≤ N`), `p` (poly degree), `d_t` (time embedding dim), `h_t` (time hidden), `h_ffn` (FFN hidden).

- QKV projections: `3 · (N · D · d_h) · H = 3 · N · D · D`
- Attention inner loop per head: `N · W · d_h` for streaming `φ(s)·V` accumulation
- Output projection per head: `N · d_h · D`; all heads: `N · D · D`
- Mixture-of-heads gating per head: `N · D` for `X·w_g`; all heads: `N · D · H`
- Threshold predictor (learned): `N · (D · h_t + h_t · H)` when enabled
- FFN: `2 · N · D · h_ffn + N · h_ffn · D`

Total forward FLOPs (dominant terms):
`≈ N · (4 · D · D) + N · W · D + N · D · H + N · (D · h_t + h_t · H) + 3 · N · D · h_ffn`

Backward FLOPs (dominant terms):
- Per-head gradients mirror forward plus Q/K/V backprop terms: `≈ 3 · N · D · D + 2 · N · W · D` (derivatives wrt `Q,K,V` and `φ`)
- Gating and threshold predictor gradients add `≈ N · D · H + N · (D · h_t + h_t · H)`
- FFN backprop: `≈ 3 · N · D · h_ffn`

Memory footprint (activations, `f32`):
- Cached input: `N · D` (`poly_attention.rs`: `src/attention/poly_attention.rs:474`)
- Optional soft top-p mask: `N · H` (`src/attention/poly_attention.rs:256`)
- Gate values for metrics: `N · H` (`src/attention/forward.rs:296`)
- FFN hidden: `N · h_ffn`
- Parameter gradients (peak): dominated by `W_out ∈ ℝ^{D×D}` and per-head `W_q/W_k/W_v ∈ ℝ^{D×d_h}`

Asymptotics:
- Forward time: `O(N · D² + N · W · D + N · D · H)`
- Backward time: `O(N · D² + N · W · D + N · D · H)`
- Activations: `O(N · (D + h_ffn + H))`

Evidence in code:
- Attention inner loop over tokens and window per head: `src/attention/forward.rs:326–399`
- Per-head backward loops over `i,j` pairs: `src/attention/poly_attention.rs:649–688` and `1483–1560`
- Output projection per head: `src/attention/forward.rs:398–399`
- Gating `X·w_g` per head: `src/attention/forward.rs:175–189` and `97–111`
- Threshold predictor forward: `src/attention/forward.rs:59–88`

## Sampling

- Posterior mean `μ_t(x_t, ε̂)` and variance `σ_t²` follow DDPM; deterministic step for `t=0`
- DDIM sampling reduces steps via non-Markovian updates; `v`-prediction requires mapping to `ε̂` before step

## Notes

- `causal_attention=false` uses bi-directional masks suitable for diffusion; can be toggled if needed
- MoE changes constant factors in FFN; dropout is optional and inverted for expectation preservation

---

## Mixture-of-Heads: Theoretical Foundations and Implementation

- Effective head contribution per token: `eff_i^h = g_i^h · m_i^h`, where `g_i^h = \mathrm{Richards}(α_h · (X_i·w_g^h) + β_h)` and `m_i^h ∈ [0,1]` is the learned/soft top-p selector.
- Final head output scales as `Y_i^h ← eff_i^h · (φ_p(S_i,: )·V)`; see gating and selection: `src/attention/forward.rs:386–388`, `src/attention/poly_attention.rs:689–707`.
- Stability safeguards: score clamping `s ∈ [-8,8]` before polynomial evaluation, ensuring bounded derivatives in `s^p`; forward: `src/attention/forward.rs:354–375`, backward: `src/attention/poly_attention.rs:1494–1513`.

Polynomial attention mapping:
- `φ_p(s) = scale · (a · s^p + b)`; forward: `src/attention/forward.rs:374–375`; backward derivatives accumulate w.r.t `a,b,scale`: `src/attention/poly_attention.rs:1518–1541`.

Gradient routing through gating:
- `∂L/∂g_i^h = (∑_h g_yh_gated[i,h] · y_pre[i,h]) · m_i^h · ∂\mathrm{Richards}/∂z` with `z = α_h · (X·w_g^h) + β_h`; implementation: `src/attention/poly_attention.rs:700–717`.

Head selection predictor (optional):
- Two-layer network on token features with conditional modulation; forward: `src/attention/forward.rs:59–88`, gradients: `src/attention/poly_attention.rs:556–582` and `1786–1804`.

---

## Bottlenecks and Inefficiencies

- Dense QKV and output projections dominate FLOPs: `O(N · D²)` twice per layer (`QKV` and `W_out`).
- When `W ≈ N` (bi-directional), attention accumulation is `O(N² · D)`; windowing reduces to `O(N · W · D)`.
- Per-token, per-head gating performs `X·w_g^h` (`O(N · D · H)`) and scales output; gradients add similar cost.
- Backward recomputes `Q,K,V` for memory efficiency but increases compute; heavy nested loops per head and token pair.
- Temporary buffers `row_buffers` allocate `N` small arrays per head in forward (`src/attention/forward.rs:326–391`), which can be fused into direct writes to reduce allocator pressure.
- Metrics construction `gate_values ∈ ℝ^{N×H}` (`src/attention/forward.rs:296–314`) adds transient memory that can be streamed.

---

## Comparative Analysis vs. Alternatives

- FlashAttention (exact, IO-aware tiling): replaces materialization with tiled streaming, reducing HBM IO and memory from quadratic to linear in `N` while preserving accuracy [Dao et al., 2022; 2024].
  - References: arXiv:2205.14135; FlashAttention-2 (ICLR 2024).
- Multi-Query / Grouped-Query Attention (share `K,V` across heads): reduces memory and bandwidth by eliminating per-head `K,V`; minimal accuracy impact [Shazeer, 2019; Ainslie et al., 2023].
  - Reference: arXiv:1911.02150; GQA PDF (2023).
- Talking-Heads Attention (inter-head mixing on logits/weights): improves accuracy but increases computation on head-head dimension and requires attention matrix materialization [Shazeer et al., 2020].
  - Reference: arXiv:2003.02436.

Implications for this codebase:
- Flash-style tiling can be adapted to polynomial attention by tiling `(Q,K,V)` and streaming `φ_p(s)` accumulation without building `N×N` intermediates; compatible with clamped scores and windowing.
- MQA/GQA directly reduce parameter and activation sizes for `K,V`, and shrink output projection bandwidth.
- Talking-Heads mixing conflicts with the current streaming implementation because it operates across the `N×N×H` attention weights; not recommended for this code path.

---

## Optimization Proposals and Expected Impact

1. Early head skipping at token granularity
- Mechanism: if `eff_i^h < τ_skip`, skip computing `y_pre_row` for `(i,h)` before entering `j` loop.
- Integration points: compute `eff_i` early (`src/attention/forward.rs:386–388`) and branch the row path.
- Expected: if average active heads per token `H_active ≪ H`, attention FLOPs drop from `N · W · D` to `N · W · (H_active/H) · D`. Inference speed improves proportionally; accuracy preserved via learned gating.

Implementation status:
- Early per-token head skipping implemented with configurable `eff_skip_threshold` in `PolyAttention` and enforced in `compute_poly_attention_forward`.

2. Adopt Multi-Query or Grouped-Query Attention
- Change: share `W_k, W_v` across heads (or groups) while keeping per-head `W_q`.
- Complexity: QKV projections shrink from `3 · N · D · D` to `N · D · D (Q) + N · D · d_h (shared K) + N · D · d_h (shared V)`; KV-cache and memory reduce by `≈ H×`.
- Accuracy: literature shows minor quality degradation; for diffusion-conditioned blocks, gating can compensate. No MOE involvement.

3. Flash-style tiling for polynomial attention
- Implement tiled blocks over `K,V` and rows of `Q`, compute `φ_p(s)` on-the-fly; avoid `row_buffers` and large temporary `gate_values` materialization.
- Expected: memory becomes linear in `N`; wall-clock speed gains on long sequences; training stability unchanged due to identical math [arXiv:2205.14135].

4. Fuse output projections
- Replace per-head block GEMMs with a single batched GEMM on concatenated `Y_head` using BLAS batched interfaces; reduces kernel launches and improves cache locality.
- Integration: instead of iterating heads to multiply `y_head · W_block`, build `Y ∈ ℝ^{N×D}` by concatenation and multiply once.

5. Stream metrics without `N×H` buffers
- Compute RMS gate statistics and tau min/max incrementally per head/token, removing `gate_values` matrix (`src/attention/forward.rs:403–407`).
- Expected: transient memory reduction `O(N·H)`; negligible compute change.

6. Numerical improvements
- Maintain clamping `s ∈ [-8,8]`; for `p>3`, use stable Horner-like evaluation to replace iterative multiplication in `src/attention/forward.rs:362–373` and `src/attention/poly_attention.rs:769–781, 803–816, 1503–1513`.
- Expected: reduced overflow checks, improved backward stability; accuracy unchanged.

7. Window scheduling
- Default bi-directional attention (`causal=false`) makes `W≈N`; enforce adaptive window `W≪N` for diffusion tasks with local dependencies, guided by `tau` and gate RMS metrics.
- Expected: attention term reduces from `O(N²·D)` to `O(N·W·D)`; training stability maintained via FiLM conditioning.

Implementation status:
- Adaptive window scheduling enabled by default in DiffusionBlock; bounds configurable and updated after forward.
- Min-SNR weighted loss API provided for training (`compute_weighted_loss`), ensuring stable gradients across timesteps.

---

## Current Metrics and Targets

- FLOPs (symbolic): see totals above; dominant `N·D²` terms (QKV + `W_out`) and `N·W·D` (attention accumulation).
- Memory (symbolic): activations `≈ N·(D+h_ffn+H)`; parameters dominated by `W_out ∈ ℝ^{D×D}` and per-head QKV.
- Training stability: safeguarded via score clamping and gradient anomaly checks (`src/attention/poly_attention.rs:819–833, 1155–1181`); FiLM conditioning preserves residual paths.

Targets (non-MOE):
- Maintain accuracy: use head-skipping only when `eff_i^h` is learned small; adopt MQA/GQA with fine-tuning to preserve denoising quality.
- Improve inference speed: reduce effective heads and apply tiling; aim for `≥1.5×` speedup on long `N` via Flash-style tiling.
- Reduce memory footprint: eliminate `N×H` transient buffers; adopt MQA/GQA for KV sharing.

---

## Literature Benchmarks and References

- FlashAttention: Tri Dao et al., 2022, arXiv:2205.14135; FlashAttention-2, 2024 (ICLR). Memory linear in `N`, IO-aware tiling.
- Multi-Query Attention: Noam Shazeer, 2019, arXiv:1911.02150. Keys/Values shared across heads; faster decoding, minimal quality loss.
- Grouped-Query Attention: Ainslie et al., 2023 (PDF 2305.13245). KV shared among groups; balances quality and efficiency.
- Talking-Heads Attention: Shazeer et al., 2020, arXiv:2003.02436. Inter-head mixing; higher compute; attention matrix materialization.

These approaches inform proposals 2–4 and align with the current mixture-of-heads gating without involving MOE.