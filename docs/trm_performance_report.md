# TRM Performance Analysis and Improvement Plan

## Performance Audit
- Datasets: identical tokenized sequences from `Dataset` loader (`src/main.rs:444`).
- Training loops compared:
  - Transformer: `LLM::train_with_warmup` (`src/llm.rs:508`).
  - Diffusion: `LLM::train_diffusion_ce` (`src/llm.rs:1544`).
  - TRM Autoencoding: `LLM::train_trm_autoencoding` (`src/llm.rs:909`).
- Metrics observed per epoch:
  - Loss, gradient norm, tokens/sec, attention `tau_range`, predictor norm RMS (`src/llm.rs:700-746`, `src/llm.rs:2126-2146`).
- Inference speed benchmarks:
  - TransformerBlock: `benches/transformer_block_bench.rs`.
  - DiffusionBlock: `benches/diffusion_block_bench.rs`.
  - TRM: `benches/trm_benchmark.rs`.
- Computational efficiency estimates (FLOPs/bytes): `metrics::perf` (`src/metrics/perf.rs`).

## Architectural Investigation
- TRM forward recursion and caches: `TRM::forward_recursive` (`src/trm.rs:429-593`).
- Attention, normalization, residuals inside TRM:
  - Pre-attn norm → Attention → Residual → Pre-FFN norm → FFN (`src/trm.rs:488-496`, `src/trm.rs:533-546`).
- Gradient flow through TRM recursion and answer path: `compute_training_gradients` (`src/trm.rs:603-671`) and `backward_through_transformer` (`src/trm.rs:673-703`).
- Initialization/hyperparameters:
  - TRMConfig (`src/trm.rs:268-285`), learnable latent `latent_init` (`src/trm.rs:208-211`).
  - Prior issue: hardcoded transformer settings in TRM new; fixed to use `TransformerBlock::from_model_config` (`src/trm.rs:352-367`).
- TransformerBlock reference behavior: `forward` and gradients (`src/transformer/transformer_block.rs:226-252`, `src/transformer/transformer_block.rs:268-347`).
- DiffusionBlock conditioning and residuals: `forward_with_timestep` (`src/transformer/diffusion_block.rs:835-944`).
- Gradient stability checks: `GRADIENT_ANOMALY_THRESHOLD` usage (`src/llm.rs:1444-1481`, `src/trm.rs:1139-1155`).

## Research Phase
- Token relation mechanisms: recursive refinement and shared weights align with fixed-point iterative methods; TRM implements contraction via residual blending (`latent_update_alpha`) and pre-norms.
- Successful variants:
  - Pre-norm transformers improve stability (used across blocks).
  - EMA-conditioned FiLM in diffusion improves training stability (`src/transformer/diffusion_block.rs:1413-1441`).
  - Adaptive attention degree via metrics (`DegreeAdaptationMetrics` in `src/llm.rs:2138-2154`).
- Key differences TRM vs Transformer/Diffusion:
  - TRM previously ignored `ModelConfig` for attention/head settings; fixed.
  - TRM uses recursive latent blending (`src/trm.rs:511-517`), adding extra compute per supervision step.
  - Diffusion adds timestep FiLM and noise scheduling; Transformer is single-pass per layer.

## Enhancement Plan
- Architectural modifications:
  - Use `TransformerBlock::from_model_config` for TRM (implemented) to align attention, windowing, heads, MoE options.
  - Expose `latent_update_alpha` via `ModelConfig.trm_latent_update_alpha` (already read at `src/trm.rs:357-360`).
  - Optional: enable adaptive head selection consistent with Transformer (`ModelConfig.head_selection`).
- Training protocol adjustments:
  - Apply LR warmup + cosine annealing for TRM phases via `LLM::train_trm_complete` pipeline; maintain gradient clipping in TRM apply (`src/trm.rs:779-807`).
  - Regularize latent state via tightened clamp (`TRM_STATE_CLIP`) if instability observed (`src/trm.rs:287`, `src/trm.rs:397-417`).
- Evaluation metrics and benchmarking procedures:
  - Use `metrics::perf` to estimate FLOPs/bytes across architectures for given `(seq_len, embed_dim, hidden_dim, heads, degree)`.
  - Run criterion benches for wall-clock throughput; compare `transformer_block_forward`, `diffusion_block_forward`, `TRM Forward Pass`.
  - Track attention metrics (`tau_range`, predictor RMS) per epoch (already logged in `LLM`).
- Implementation roadmap:
  - Phase 1: Config alignment (done).
  - Phase 2: Add perf estimators (done) and capture benchmark HTML reports.
  - Phase 3: Hyperparameter sweep for `latent_update_alpha`, `num_recursions`, supervision steps via CLI (`src/main.rs:126-142`).
  - Phase 4: Optional enabling of MoE and adaptive heads for TRM via `ModelConfig`.

## Validation Protocol
- A/B testing framework:
  - Train three variants on identical pretraining/chat datasets selected via `Dataset` (`src/main.rs:441-451`).
  - Architectures: `ArchitectureType::Transformer`, `ArchitectureType::Diffusion`, `ArchitectureType::TRM` (`src/main.rs:177-183`).
  - TRM variants: sweep `num_recursions`, `latent_update_alpha`, supervision/inference steps (`src/main.rs:390-395`).
- Success criteria:
  - Loss reductions comparable to transformer baseline in instruction epochs; no gradient anomalies; tokens/sec within 1.5× of transformer for same config and seq length; attention `tau_range` stable.
- Fallback options:
  - Reduce `num_recursions` and increase `latent_update_alpha` for stronger contraction.
  - Disable diffusion coupling in TRM if instability; revert to transformer-only TRM.

## Experimental Evidence
- Bench harnesses compiled and executed for forward speed.
- Estimation functions available to quantify compute budgets (`src/metrics/perf.rs`).
- Logging already reports per-epoch metrics; enable `RUST_LOG=info`.

## Version Control of Variants
- Use CLI flags to persist variant configurations in saved model metadata (`src/main.rs:572-576`).
- Save separate files per variant with descriptive names (e.g., `models/trm_r2_a005.bin`).

## Summary
- Root cause of misalignment: TRM hardcoded transformer settings; fixed.
- Added FLOPs/bytes estimators and benches to quantify performance.
- Defined A/B protocol and success thresholds; outlined hyperparameter sweep and stabilization steps.