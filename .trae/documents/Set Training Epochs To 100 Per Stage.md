## Goal
- Ensure both training stages (Pretraining and Instruction Tuning) run for 100 epochs for both Transformer and Diffusion architectures.

## Changes
- In `src/main.rs`, update Diffusion Pretraining call:
  - Replace `llm.train_diffusion(pretraining_examples, 1, 0.001, 4)?;` with `llm.train_diffusion(pretraining_examples, 100, 0.001, 4)?;`
  - Update adjacent log line to reflect 100 epochs.
- Transformer Pretraining:
  - Already set to 100 epochs via `train_with_batch_size(..., 100, ...)` — leave unchanged.
- Instruction Tuning:
  - Both paths already use `instruction_epochs = 100;` — leave unchanged.
- Keep `--diffusion` flag behavior and logging intact.

## Verification
- Run:
  - Diffusion: `cargo run --release --bin main -- --diffusion` → observe 100 epochs in pretraining logs.
  - Transformer: `cargo run --release --bin main` → pretraining logs remain 100 epochs.
- Confirm stage separation and active block logging remain correct.

## Scope
- Single-line change in `main.rs` for diffusion pretraining epochs; no API or trait changes, no test modifications needed.