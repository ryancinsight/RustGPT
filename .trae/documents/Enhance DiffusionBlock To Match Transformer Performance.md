## Research Summary
- Architectures
  - `src/transformer/transformer_block.rs`: Pre-LN norms → `PolyAttention` → Pre-LN → `RichardsGlu`/MoE → residuals (forward: 226–241; gradients: 268–347). Configuration and LARS-like scaling in `apply_gradients` (349–482).
  - `src/transformer/diffusion_block.rs`: Adds `TimeEmbedding` MLP → FiLM modulation (gamma/beta) on both norms → `PolyAttention.forward_impl(causal=false)` → FFN → residuals; noise scheduler (41–111) and discrete masked diffusion support (292–477; 820–923). EMA for time conditioner; optional dropout; custom gradient routing (1068–1197; 1208–1388).
- Attention
  - `src/attention/poly_attention.rs`: Unified attention with CoPE, head gating via Richards curves, adaptive polynomial degree, threshold predictor; supports causal/non-causal windows (430–468; 1291–1318). Gradient paths include per-head Wq/Wk/Wv, output projection, gating, CoPE, and optional threshold predictor (470–1141, 1244–1269, 1281–1646).
- Normalization/FFN
  - `src/richards/richards_norm.rs`: Dynamic Richards-based normalization with learnable parameters and per-feature affine, full gradient support (42–104; 196–290; 292–330).
  - `src/richards/richards_glu.rs`: GLU with learnable Richards activation and sigmoid gate; Xavier init, full analytic gradients, LARS-like trust-ratio scaling (31–58; 66–93; 110–239; 241–309).
- Training & Metrics
  - Diffusion mixed objective trains CE on next-token plus ε-MSE or v-MSE with curriculum `lambda_ce_schedule(t)`; logs `loss`, `grad_norm`, `epoch_ms`, `tokens_per_sec`, `tau_range`, `pred_norm_rms`, validation CE/MSE (LLM training: 1680–2255; epoch logs: 882–898, 2214–2225).
  - Discrete masked diffusion integrates `DiscreteMaskScheduler` for absorbing `[MASK]` (diffusion/discrete.rs; usage in `llm.rs`: 1680–1767, 2079–2211).

## Root Cause Analysis
- Gradient routing fallback divergence
  - Transformer: on missing partitions, falls back to routing all arrays to attention (`apply_gradients`, 360–373) so parameters still update.
  - Diffusion: on missing partitions, sets all partition counts to 0 (1243–1248), risking silent no-op updates if metadata is ever missing or miscounted.
- Mismatch in clipping/scaling
  - Transformer clips global param gradient at `clip=5.0` and applies LARS-like scaling per submodule (374–447).
  - Diffusion uses `clip=2.5` and per-submodule scaling but differs for time-conditioner; mismatch can slow learning and cause under-updates (1214–1250; 1334–1388).
- FiLM modulation magnitude/bias
  - Current mapping `gamma=1+0.1*x`, `beta=0.1*x` (865–878) may inject large bias early depending on time-MLP init, causing activation sanitization (789–808) and extra dropout (888–901) to frequently trigger, reducing effective signal.
- Excess sanitization/clamping and dropout usage
  - Multiple sanitize clamps at ±50 (789–808) plus optional dropout after both attention and FFN (888–901) can damp gradients and slow convergence compared to TransformerBlock where sanitization is lighter.
- Optimizer configuration for time-conditioner
  - Time-conditioning optimizers default to Adam with AMSGrad but no decoupled weight decay; the time MLP can overfit modulation without mild WD.

## Enhancement Plan
- Robust gradient routing
  - Align `DiffusionBlock::apply_gradients` fallback with Transformer: if partition metadata missing, route all arrays to attention or assert mismatch; never default to zeros. Add strict count checks and warnings with corrective routing.
- Unify clipping/scaling
  - Set diffusion `clip` to 5.0; keep per-submodule LARS-style trust-ratio scaling consistent with Transformer for attention, FFN, norms, and time-conditioner.
- FiLM reparameterization
  - Replace fixed `0.1` scaling with bounded nonlinearity: `gamma = 1 + s_g * tanh(x)`, `beta = s_b * tanh(x)` with small learnable scales (`s_g≈0.01`, `s_b≈0.01`) to reduce early bias and stabilize gradients. Backward paths accumulate scale factors.
- Min-SNR loss weighting
  - Use `DiffusionBlock::min_snr_weight(t, γ)` to weight ε-MSE or v-MSE per timestep; couple with CE mixing: `λ_ce(t) = f(t)` and `λ_mse(t) = min_snr_weight(t, γ)`. Improves stability and convergence speed.
- Optimizer upgrades for time-conditioner
  - Switch `opt_time_*` to AdamW (decoupled weight decay `wd≈0.01`) or set via `set_weight_decay(wd, true)`. Retain AMSGrad.
- Initialization tuning
  - Reduce stddev for `time_w*` via smaller fan-in scaling or use uniform Kaiming; optionally initialize `b*` with zeros; ensure EMA starts from copies and `use_ema_for_sampling` toggles remain.
- Regularization
  - Keep dropout disabled by default; cap sanitize clipping to ±20 to reduce hard clamps; optionally enable mild dropout (≤0.1) only after FFN if needed.

## Hyperparameter Optimization
- Search ranges
  - `dropout_rate`: 0.0–0.1; `ema_decay`: 0.995–0.9995; `wd_time`: 0.001–0.02; FiLM scales `s_g,s_b`: 0.005–0.02; `clip_norm_pred` (LLM backward): 1.5–3.0; `γ` for Min-SNR: 1–5.
- Procedure
  - Grid or random search on synthetic task; track `loss`, `grad_norm`, `epoch_ms`, `tokens/s`, validation CE/MSE.

## Validation Suite
- Unit tests
  - FiLM forward/backward shape and gradient correctness (DiffusionBlock::film_backward).
  - Gradient partitions: ensure non-empty metadata routes exact counts; mismatch triggers corrective routing.
  - Min-SNR weighting monotonicity and boundedness.
  - Time-conditioning AdamW update equivalence and WD behavior.
- Integration
  - End-to-end training on small corpus (shared config) for Transformer vs Diffusion: report `avg_loss`, `grad_norm`, `tokens/s`, validation metrics after N epochs.
- Benchmarks
  - Micro-benchmarks on forward/compute_gradients for both blocks to compare throughput; perf tests exist in diffusion tests (`#[ignore] perf_*`, 1609–1778); add analogous transformer perf where useful.
- Ablation Studies
  - Toggle FiLM, Min-SNR, dropout, EMA sampling, ε vs v parameterization; measure impacts on convergence and validation.

## Deliverables
- Code updates implementing routing, clipping/scaling, FiLM reparam, Min-SNR weighting, AdamW for time-conditioner, and init tuning.
- Comprehensive unit and integration tests; perf/ablation scripts.
- Performance report: original vs revised DiffusionBlock vs TransformerBlock across accuracy/loss/convergence speed and computational efficiency.

## Key References
- TransformerBlock residual/gradients: `src/transformer/transformer_block.rs:226–241, 268–347, 374–447`.
- DiffusionBlock forward/time-conditioning/gradients: `src/transformer/diffusion_block.rs:820–923, 1068–1197, 1208–1388`.
- PolyAttention mechanics: `src/attention/poly_attention.rs:430–468, 1281–1646`.
- Norm/FFN implementations: `src/richards/richards_norm.rs:196–330`; `src/richards/richards_glu.rs:241–309`.
- Discrete scheduler: `src/diffusion/discrete.rs` and its training use in `src/llm.rs:1680–1767, 2079–2211`.

If you approve, I will implement the changes, add tests/benchmarks, and produce the comparison report.