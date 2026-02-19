# AGENTS.md - RustGPT Development Guide

## Build & Test Commands

- **Build**: `cargo build --release`
- **Build with GPU (WGPU)**: `cargo build --release --features gpu-wgpu`
- **Build with GPU (CUDA)**: `cargo build --release --features gpu-cuda`
- **Build with all GPU backends**: `cargo build --release --features gpu-all`
- **Test all**: `cargo test --lib`
- **Test single**: `cargo test --lib [test_name] -- --exact`
- **Test integration**: `cargo test --test [test_file]` (e.g., `cargo test --test transformer_block_verification`)
- **Test GPU components**: `cargo test --test gpu_shared_components_phase56` (Phase 5.6)
- **Lint**: `cargo clippy --all-targets`
- **Format check**: `cargo fmt -- --check`
- **Format fix**: `cargo fmt`
- **Benchmark**: `cargo bench --bench [bench_name]`
- **Check compilation**: `cargo check --lib`

## Architecture & Codebase

**Language**: Rust 1.85+, Edition 2024 with `ndarray` for matrix operations

**Structure** (Clean Architecture):
- `src/domain/` - Core LLM architectures (Transformers, TRM, Diffusion, Mamba, RG-LRU)
- `src/application/` - Training & inference logic
- `src/infrastructure/` - Serialization, data loading, file I/O
- `src/presentation/` - CLI and Web UI
- `src/common/` - Shared utilities & error handling
- `tests/` - 183+ integration tests
- `benches/` - Performance benchmarks

**Key APIs**:
- `TransformerBlock`, `TemporalMixingLayer` - Attention and recurrent variants
- `LLMModel` - Training & inference entrypoint
- `ModelConfig`, `TemporalMixingType` - Configuration enums
- `AttentionContext`, `FeedforwardProcessor` - Modular components

## Code Style & Conventions

**Formatting**: Rustfmt with `edition = "2024"`, `tab_spaces = 4`, auto-import reordering

**Naming**:
- Types: PascalCase (`TransformerBlock`, `TemporalMixingType`)
- Functions: snake_case (`compute_attention`, `forward_pass`)
- Constants: UPPER_SNAKE_CASE
- Generics: Use descriptive names (prefer `T` only for simple cases)

**Imports**: Group in order: (1) std, (2) external crates, (3) local modules

**Error Handling**: Use `Result<T, Box<dyn Error>>` with `thiserror` for custom errors; no `panic!()` calls

**Testing**: Integration tests in `tests/` directory; use `proptest` for property-based testing

**Documentation**: Doc comments for public APIs; examples in test files mirror expected usage patterns
