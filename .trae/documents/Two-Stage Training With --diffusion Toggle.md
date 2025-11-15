## Goals
- Implement a robust two-stage training pipeline (Pretraining → Instruction Tuning) for both Transformer and Diffusion architectures.
- Add a CLI flag `--diffusion` to select Diffusion; default to Transformer when absent.
- Validate input JSON datasets, preserve model state across stages, log active block, and maintain clean separation of phases.

## Changes
### CLI Flag & Architecture Selection
- In `src/main.rs` (Args at lines ~10–17): add `#[arg(long)] diffusion: bool`.
- Set `architecture = if args.diffusion { ArchitectureType::Diffusion } else { ArchitectureType::Transformer }` (near current hardcoded architecture line ~47–54).
- Keep all other config derivations unchanged.

### Data Validation
- Before building `Dataset`, validate both files using `serde_json`:
  - Ensure files exist, are parseable JSON arrays of strings (or objects containing `text` string if that’s the current schema).
  - Return a clear error via `anyhow/thiserror` with a message indicating which file failed and why.
- Implement lightweight validators directly in `main.rs` to avoid broad refactors:
  - `fn validate_json_lines(path: &str) -> Result<()>` checks type and non-empty content.

### Two-Stage Pipeline
- Stage 1 — Pretraining:
  - Transformer: call `llm.train_with_batch_size(pretraining_examples, epochs, lr, batch_size)` with existing hyperparams.
  - Diffusion: call `llm.train_diffusion(pretraining_examples, epochs, lr, batch_size)` (already implemented to delegate to batch training).
- Stage 2 — Instruction Tuning:
  - Both architectures: call `llm.train_with_warmup(chat_training_examples, instruction_epochs, instruction_lr, batch_size, warmup_epochs)`.
  - Preserve model state: use the same `LLM` instance created before Stage 1; do not recreate the network.
- Save model after Stage 2 to `models/rustgpt.bin` (Transformer) or `models/rustgpt-diffusion.bin` (Diffusion).

### Logging & Separation
- Use `tracing` to log:
  - Active architecture and block type before each stage (e.g., "[Train] Architecture=Transformer Block=TransformerBlock" / "Architecture=Diffusion Block=DiffusionBlock").
  - Stage boundaries: "=== PRETRAINING (Transformer|Diffusion) ===" and "=== INSTRUCTION TUNING (Transformer|Diffusion) ===".
  - Throughput and simple accuracy proxies (already added) remain.

### Model State Transitions
- Ensure any architecture-specific toggles are set before stages:
  - For TRM only: mode switches; not needed here.
  - For Diffusion: continue using timestep-agnostic training flows; generation uses sampling afterward.
- Keep cached intermediates untouched across stages; rely on `LLM` methods for training.

### Backward Compatibility
- Default behavior without `--diffusion` remains Transformer.
- No changes to `Layer` trait or builders; `model_builder` stays intact.

## Tests
- Add or adapt tests to verify:
  - CLI toggling activates correct architecture and logs block type.
  - Validation rejects malformed JSON and accepts correct format.
  - Two-stage execution preserves weights (loss decreases in Stage 2; parameter count constant).
  - Parity checks: `LayerEnum` stacks match expectations for each architecture.

## Delivery
- Code updates in `main.rs` only for CLI, validation, logging, and phase orchestration.
- No changes to training methods; reuse existing `LLM` APIs.
- Verified by running `cargo run --release --bin main` with/without `--diffusion` to produce perf metrics and logs indicating selected block and stages.