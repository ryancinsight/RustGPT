# Transformer Block Audit and Optimization Report

## Scope
- Components: `TransformerBlock` (pre-attn norm → attention → residual → pre-ffn norm → FFN → residual)
- Targets: performance, memory efficiency, gradient stability/loss behavior

## Baselines (Release)
- Transformer forward probe (`bench_transformer`): throughput ≈ 17.7K–23.2K tokens/s for `n=256, d=256, heads=8`
- Attention baseline vs optimized (`bench_attention_compare`): speedup 5.3–13.2% depending on config

## Changes Implemented
- Attention forward
  - Accumulator starts at zeros (removed implicit residual add)
  - Parallel per-row compute; optional precomputed full score matrix for `n ≤ 1024`
  - Fewer temporary allocations; selective reuse via local matrices; GEMM-based scoring
- Transformer block
  - Reduced `cached_intermediates` footprint (removed unused elements)
  - Adaptive window logic maintained; invariant gradient flow preserved
- Bench & tests
  - Criterion benches for forward
  - Release-mode comparison harness
  - Property tests: finite gradients, bounded norms

## Performance Analysis
- Time Complexity
  - Per-head per-token: `O(window * head_dim)` for banded attention
  - With precomputed scores: reduces inner dot cost; preserves overall `O(n * window)` scaling
- Matrix Multiplication
  - GEMM (`general_mat_mul`) used for `phi·V` and `Y·W_out`
  - Precompute `Q·Kᵀ` when beneficial, then polynomial map

## Memory Profiles
- Core tensors per head
  - `Q,K,V`: `(n × d_h)` each; total `3·n·d_h` per head
  - `phi_row`: `(window)` per row, ephemeral
  - `y_head`: `(n × d_h)`
- Block outputs
  - `out_block`: `(n × d_model)` per head projected
  - Reduced intermediates in `TransformerBlock` cache: dropped `attn_out`, `ffn_out` from cache
- Example (`n=256, d=256, heads=8 ⇒ d_h=32`)
  - Per head Q/K/V ≈ `3·256·32·4B ≈ 96KB`
  - `y_head` ≈ `256·32·4B ≈ 32KB`
  - All heads (Q/K/V/y_head) ≈ ~1MB transient
  - Removed cached `attn_out` and `ffn_out` saves ≈ `2·(n·d)·4B ≈ 0.5MB`

## Gradient Stability
- Analytical checks
  - Residual gradient splits preserved; norms combined correctly
  - Clamps on score `s` to `[-8,8]` stabilize polynomial evaluation for `p ≥ 1`
  - Global gradient clipping in `apply_gradients` prevents exploding updates
- Tests
  - RMSE analytical vs backward threshold maintained in existing tests
  - New property tests ensure non-finite gradients are rejected and norms bounded

## Metrics (Before/After)
- Attention compare (release)
  - Baseline ≈ 37.3K–40.8K tokens/s
  - Optimized ≈ 39.3K–43.3K tokens/s
  - Speedup: 5–13% depending on sequence/head settings
- Transformer throughput (release)
  - Representative: ~17.7K–23.2K tokens/s (variance with warnings clean-up and environment)

## Conclusions
- Throughput improvements achieved; further gains available via chunked parallel row writes using ndarray `Zip::par_apply` and TLS buffer integration for scores/y_head
- Memory footprint reduced via cache trimming; windowed attention constrains `phi` and partial V access
- Gradient stability consolidated with clamps and tests; training curves expected more stable under typical configs

## Next Steps
- Integrate TLS buffers (`attention::memory`) for `scores_full` and `y_head` to avoid per-iteration allocations with safe parallel chunking
- Add optional mixed-precision parameter storage (feature flag) for W_out and gating vectors
- Extend criterion benches to sweep sequence lengths and window sizes; export CSV for dashboards