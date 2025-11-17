# Attention Module Documentation

## Overview
- Module: `attention` implements polynomial multi-head attention with adaptive gating and optional dynamic head selection.
- Core files:
  - `src/attention/poly_attention.rs`: Layer definition, parameters, gradients, adaptive degree, gating, and integration points.
  - `src/attention/forward.rs`: High-performance forward path with per-head projections, gating, windowed interactions, and output projection.
  - `src/attention/config.rs`: Initialization utilities for heads, output projection, gating params, CoPE, and head selection configuration.
  - `src/attention/memory.rs`: Thread-local scratch buffers intended to reduce transient allocations.
  - `src/attention/position/cope.rs`: CoPE positional embeddings and their gradient application.

## Mathematical Framework
- Polynomial Attention replaces softmax with a learnable polynomial transform `φ(s) = scale·(a·s^p + b)` applied to attention scores `s = q·k + cope(i−j)`.
- Multi-Head Aggregation: per head output `y_h[i] = Σ_j φ(s_ij)·V[j]`, then projected via block `W_out[h]` and summed across heads.
- Gating: per-head gate `g_h = Richards(α_h·(X·W_g_h) + β_h)`, optionally modulated by learned thresholds (`HeadSelectionStrategy::Learned`) or soft top-p.
- Stability: scores are clamped to `[-8, 8]` before polynomial evaluation; gradients are globally clipped.
- Theoretical properties (implemented in `poly_attention.rs` doc block): bounded gradients, numerical stability, and convergence under standard assumptions.

## Computational Complexity
- Let `n` be sequence length, `d` embedding dim, `H` heads, `d_h = d/H`, and window size `w`.
- Per head per token:
  - Score calculation: `O(w·d_h)` for sliding window, or `O(n·d_h)` for full attention.
  - Polynomial evaluation: `O(w)`; degree `p` handled efficiently for small `p` or via iterative multiply.
  - Value aggregation `φ·V`: GEMM-like `O(w·d_h)`.
- Per head total: `O(n·w·d_h)`; across heads: `O(n·w·d)`.
- Output projection: `Y_h·W_out[h]` → `O(n·d_h·d)` per head; when concatenated and summed, total `O(n·d·d)` but implemented block-wise.
- Optional optimized path (`forward.rs`) precomputes `Q·Kᵀ` for `n ≤ 1024` to reduce repeated dot products, improving constant factors at small-to-medium `n`.

## Performance Metrics (Release)
- Attention comparison (`src/bin/bench_attention_compare.rs`):
  - Example: `baseline_tps≈38994`, `optimized_tps≈44129`, `speedup≈13%`.
  - Another config: `baseline_tps≈37372`, `optimized_tps≈39358`, `speedup≈5.3%`.
- Throughput depends on `n`, window size, head count, and polynomial degree. Optimized path benefits small-to-mid `n` via precomputed `Q·Kᵀ` and reduced allocations.

## Integration
- `TransformerBlock` (`src/transformer/transformer_block.rs`): Uses attention in pre-norm residual block; caches intermediates for precise analytical gradients.
- `DiffusionBlock` (`src/transformer/diffusion_block.rs`): Non-autoregressive; calls `forward_impl` with `causal=false`; now vectorized FiLM modulation and time-conditioning.
- Head selection strategies (`src/mixtures/moh.rs` and `src/attention/config.rs`): configure gating (Fixed, SoftTopP, Learned predictor). Soft top-p implemented in `forward.rs` with Richards sigmoid and PadeExp stabilization.
- Positional encoding CoPE: Adds `q·pos` for distance-aware bias; gradients applied via local accumulation.

## Memory Efficiency
- Heads store `W_q/W_k/W_v (D×d_h)` and gating params `W_g (D×H)`, `α/β (1×H)`.
- Transients per head: `Q/K/V (n×d_h)`, windowed `φ_row (w)`, `y_head (n×d_h)`, and projected `out_block (n×d)`.
- Optimizations:
  - Zero-initialized output accumulation (removed implicit residual add in attention forward).
  - Optional precomputed score matrix `Q·Kᵀ (n×n)` for small `n`.
  - TLS buffers available for scores/intermediates (`memory.rs`) to cut allocator churn.

## Gradient Stability and Loss
- Clamped scores before polynomial evaluation, bounded gating via Richards curves.
- Global gradient sanitization and clipping (LARS-style scaling per subcomponent) in both transformer and attention layers.
- Tests validate analytical vs backward gradients and numerical finiteness; property tests ensure bounded norms.

## Current Hot Paths
- `forward.rs: compute_poly_attention_forward`:
  - Per-row loop with parallel map; `φ_row` computed and applied via GEMM to `V_slice`.
  - Output projection per head via GEMM.
  - Soft top-p gating and metrics updates.
- `poly_attention.rs: compute_gradients_parallel`:
  - Parallel per-head gradient computation; accumulates CoPE, projections, gating, and predictor gradients.

## Next Optimizations
- Parallel Row Assignment:
  - Use `Zip::from(y_head.rows_mut()).par_apply` to fill head outputs without temporary row buffers.
- TLS Buffer Integration:
  - Wire `attention::memory` buffers into forward to reuse `scores_full`, per-head `y_head`, and temp matrices.
- Mixed Precision:
  - Feature-flagged storage for `W_out`, `W_g`, and predictor weights (e.g., f16/bf16) to reduce bandwidth and memory.
- Kernel Fusion:
  - Fuse score clamp + polynomial evaluation into a single pass on `scores_row` to cut memory traffic.
- Degree Adaptation Smoothing:
  - Use EMA with hysteresis thresholds to minimize flips and retraining churn across epochs.
- Sparse Windowing:
  - For large `n`, adaptively prune low-φ indices within the window and switch to sparse GEMV for `φ·V`.
- SIMD/BLAS:
  - Explore `std::simd` for per-row operations; consider vendor BLAS or `ndarray`’s BLAS feature for GEMM hot paths.
- Predictor Path Vectorization:
  - Batch predictor forward/backward across tokens/heads to reduce overhead and leverage matrix ops for gradient accumulation.

## How To Benchmark
- Criterion: `cargo bench -q` (release by default).
- Attention comparison: `cargo run --release --bin bench_attention_compare`.
- Transformer throughput probe: `cargo run --release --bin bench_transformer`.

## Safety & Invariants
- `embed_dim % num_heads == 0` enforced.
- Score clamps and gradient clipping prevent numerical blow-ups.
- Parameter partitions tracked to route gradients precisely.

## References
- See `src/attention/poly_attention.rs:54` for mathematical analysis and theorems.
- Forward implementation details: `src/attention/forward.rs:45`.
- Integration points: `src/transformer/transformer_block.rs:246`, `src/transformer/diffusion_block.rs:971`.