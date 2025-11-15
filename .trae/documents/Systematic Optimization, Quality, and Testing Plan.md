## Scope & Objectives
- Optimize performance in critical training/evaluation paths
- Improve code quality, documentation, and consistency
- Complete missing diffusion features and harden error handling
- Expand testing (unit, integration, regression) with measurable benchmarks
- Refine CLI UX (no GUI) for clarity, accessibility, and i18n readiness

## Performance Optimization
- Baseline & Metrics
  - Capture per-epoch timing, throughput (examples/sec), grad norm distribution, and memory estimates in `main.rs` and `LLM` logs
  - Add micro-benchmarks using Criterion for:
    - TokenEmbeddings forward (`src/embeddings.rs:64-70`)
    - PolyAttention forward/backward hot paths (`src/attention/poly_attention.rs:239-274`, `276-847`)
    - Diffusion denoising loop (new `LLM::train_diffusion`, `src/llm.rs:1298-1369`)
- Targeted Improvements (examples)
  - Pre-allocate buffers in PolyAttention gradient computation to avoid per-iteration allocations; reuse workspace arrays
  - Replace nested scalar loops with batched matmuls where possible; leverage `ndarray::linalg::general_mat_mul` already used (`poly_attention.rs:401-433`)
  - Reduce cloning in training loop (`src/llm.rs:881-904`, `1068-1077`) via views and in-place ops when safe
  - Parallelize diffusion batch processing with `rayon::par_chunks` where independence allows (care with RNG seeding)
  - Cache timestep-conditioned gating transforms once per timestep, not per head per token
- Benchmarks
  - Add Criterion benches under `benches/` (no functional change): attention_forward, attention_backward, embeddings_forward, train_batch_step
  - Define success thresholds (≥15% latency reduction in attention backward, ≤10% allocs per step)

## Code Quality Improvements
- Modularization & Refactors
  - Split `LLM::train_batch` into smaller helpers: compute_logits, compute_loss_grads, accumulate_param_grads, apply_layer_grads (`src/llm.rs:836-1082`)
  - Extract diffusion training helpers from `LLM::train_diffusion` into `llm::diffusion_train.rs`-like module to isolate logic
  - Encapsulate JSON loader parsing variants in `dataset_loader.rs` with typed enums and unified parse function (`src/dataset_loader.rs:61-88`)
- Documentation & Comments
  - Add rustdoc to public structs/methods lacking docs (e.g., `LLM`, dataset loader, training functions)
  - Inline comments for non-obvious math (LARS scaling `src/llm.rs:1048-1152`, diffusion scheduler math `src/transformer/diffusion_block.rs:175-220`)
- Coding Standards
  - Enforce consistent error naming (`ModelError::*`), iterator-based patterns, avoid redundant clones, prefer views and mapv_inplace
  - Run `clippy` and apply recommended lints for performance and readability

## Feature Completeness
- Diffusion
  - Implement timesteps sampling schedule (uniform or cosine-weighted) instead of deterministic `(epoch+count)%T`
  - Add option to predict velocity (`v-prediction`) for improved stability; configurable via CLI flag
  - Support classifier-free guidance style conditioning hooks (placeholder interface only; no external deps)
- Transformer
  - Allow toggling between `PolyAttention` and `SelfAttention` via CLI to match previous baselines
- CLI Enhancements
  - `--diffusion` (already added), plus:
    - `--epochs-pretrain`, `--epochs-tune`, `--batch-size`, `--lr`
    - `--attention {poly,self}`

## Error Handling & Robustness
- Expand Coverage
  - Validate dataset contents (non-empty, reasonable length) in loader and log samples
  - Harden diffusion pipeline when no diffusion blocks present (currently checked; improve message)
  - Guard against NaN/Inf in gradients and scheduler math; early abort with actionable messages
- Logging Improvements
  - Add structured logs for per-layer adaptive LR scales, max layer grad norms, and anomaly detections
  - Log diffusion MSE per epoch and per timestep bucket summary
- Defensive Checks
  - Ensure consistent shapes between logits and targets (already logged) and return typed error instead of zero grads
  - RNG seeding options for reproducibility via CLI without global state

## Testing & Validation
- Unit Tests
  - Dataset loader: arrays of strings, arrays of objects with `text`, malformed inputs
  - Diffusion MSE pipeline: fixed shapes, non-NaN losses, monotonic decrease on synthetic small data
  - PolyAttention math: parameter gradient sizes and finiteness
- Integration Tests
  - Two-stage pipeline runs for both architectures end-to-end with tiny datasets; verify logs and saved model
  - A/B tests for attention types and window sizes
- Regression Tests
  - Snapshot logs for key metrics; compare ranges across runs to detect regressions
- Automation
  - Add CI job: build, clippy, test, criterion (optional quick mode) with artifacts for perf summaries

## User Experience (CLI)
- Refine Flags & Help
  - Descriptive `--help` with examples for diffusion vs transformer training
  - Group flags by stage (pretrain vs tune), architecture, performance tuning
- Interaction Flow
  - Clear stage boundaries printed with active block types; success/failure summaries
- Accessibility & i18n Readiness
  - Consistent, concise messages; avoid jargon; centralize strings for potential localization later

## Deliverables & Acceptance
- Performance: Benchmarks added and initial improvements with measurable gains in attention backward path
- Quality: Refactored training functions, improved docs, clippy-clean
- Features: Diffusion training options (timestep schedule, v-pred), attention toggle
- Robustness: Better error messages, additional checks, structured logs
- Tests: Expanded unit/integration/regression coverage; CI pipeline defined

## Timeline (Phases)
1. Benchmarks & Profiling (Criterion, logging baselines)
2. Hot-path optimizations (PolyAttention/LLM training refactors)
3. Loader & CLI hardening (data validation, toggles)
4. Diffusion feature expansion (timestep schedule, v-pred)
5. Testing expansion & CI setup
6. Final performance validation and documentation update