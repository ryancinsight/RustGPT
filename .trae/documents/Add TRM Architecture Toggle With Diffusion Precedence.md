## Goal
- Restore explicit TRM architecture selection in CLI.
- Ensure three architectures are available: Transformer, Diffusion, TRM.
- When both `--trm` and `--diffusion` are set, select Diffusion (TRM can use either; diffusion takes precedence when requested).
- Keep existing training flows intact; TRM uses standard training, Diffusion uses denoising training.

## Changes
- `src/main.rs`:
  - Add `--trm` flag in `Args`.
  - Set `architecture` as:
    - If `--diffusion`: `ArchitectureType::Diffusion`
    - Else if `--trm`: `ArchitectureType::TRM`
    - Else: `ArchitectureType::Transformer`
  - Logging for TRM stages analogous to others.
- No changes to `model_builder.rs` (already supports TRM).
- No changes to `LLM` training logic; TRM paths are handled by `train_with_warmup` which toggles TRM training mode internally.

## Verification
- Build and run:
  - `cargo run --release --bin main -- --trm` → TRM architecture logs and training/tuning.
  - `cargo run --release --bin main -- --trm --diffusion` → Diffusion selected (precedence), denoising training.
  - `cargo run --release --bin main` → Transformer.

## Scope
- Minimal CLI and selection updates only; no structural refactors required for TRM support.