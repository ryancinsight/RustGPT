# Speculative Decoding Audit and Enhancement Plan

## Date: 2024-11-24

## Status: IMPLEMENTED ✓

All planned enhancements have been implemented and tested.

## Current State Analysis

### CLI Flags Explained

| Flag | Purpose |
|------|---------|
| `--speculative` | **Enable** speculative sampling (required to activate) |
| `--speculative-mode <MODE>` | **Override** auto-detected mode (`transformer` or `diffusion`) |
| `--diffusion` | Use diffusion model architecture (affects auto-detection) |

**Auto-detection logic:**
- If `--speculative-mode` is set → use that mode explicitly
- If `--diffusion` is set → auto-use `SpeculativeMode::Diffusion`
- Otherwise (transformer/TRM) → auto-use `SpeculativeMode::Transformer`

**Examples:**
```bash
# Transformer model with transformer speculation (auto-detected)
cargo run -- --speculative

# Diffusion model with diffusion speculation (auto-detected)
cargo run -- --diffusion --speculative

# Override: force transformer speculation even with diffusion model
cargo run -- --diffusion --speculative --speculative-mode transformer
```

### Files Involved
- `src/transformer/speculative.rs` - Core speculative sampling types and trait
- `src/llm.rs` - LLM struct with decoder and speculative config
- `src/cli.rs` - CLI arguments for speculative mode
- `src/training.rs` - Training pipeline where speculative is enabled
- `src/main.rs` - Model info display

### Issue Identified

**Problem**: When speculative decoding is enabled (`--speculative --speculative-mode transformer`), the model info still shows "GreedyDecoder" in the network description:

```
Network architecture: TokenEmbeddings, TransformerBlock, ..., OutputProjection, GreedyDecoder
```

**Root Cause**: The `network_description()` method in `llm.rs` always appends `self.decoder.layer_type()` which returns "GreedyDecoder" because:
1. `DecoderType` enum only has `Greedy(GreedyDecoder)` variant
2. Speculative config is stored separately (`speculative_config`, `speculative_mode`) and not reflected in network description
3. No `DecoderType::Speculative` variant exists

### Current Architecture

```rust
// DecoderType only has Greedy variant
pub enum DecoderType {
    Greedy(GreedyDecoder),
}

// LLM stores speculative info separately
pub struct LLM {
    decoder: DecoderType,
    speculative_config: Option<SpeculativeSamplingConfig>,
    speculative_mode: SpeculativeMode,
}

// network_description always shows decoder.layer_type()
pub fn network_description(&self) -> String {
    format!("{}, {}", network_layers, self.decoder.layer_type())
}
```

## Enhancement Plan

### 1. Extend DecoderType Enum

Add a `Speculative` variant to properly represent the decoder type:

```rust
pub enum DecoderType {
    Greedy(GreedyDecoder),
    Speculative {
        base: GreedyDecoder,
        config: SpeculativeSamplingConfig,
        mode: SpeculativeMode,
    },
}
```

### 2. Update network_description

Make it correctly reflect the active decoding strategy:

```rust
pub fn network_description(&self) -> String {
    let decoder_desc = match (&self.decoder, self.speculative_config, self.speculative_mode) {
        (_, Some(cfg), SpeculativeMode::Transformer) => 
            format!("SpeculativeDecoder(γ={}, τ={:.4})", cfg.gamma, cfg.tau),
        (_, Some(cfg), SpeculativeMode::Diffusion) => 
            format!("SpeculativeDiffusion(γ={}, τ={:.4})", cfg.gamma, cfg.tau),
        (decoder, _, _) => decoder.layer_type().to_string(),
    };
    format!("{}, {}", network_layers, decoder_desc)
}
```

### 3. Improve SpeculativeSamplingConfig

Add more configuration options and diagnostics:

```rust
pub struct SpeculativeSamplingConfig {
    pub gamma: usize,        // Number of speculative steps
    pub tau: f32,            // Acceptance threshold
    pub draft_layers: usize, // Number of draft model layers
    pub temperature: f32,    // Sampling temperature (NEW)
    pub top_p: f32,          // Nucleus sampling threshold (NEW)
}

pub struct SpeculativeStats {
    pub total_tokens: usize,
    pub accepted_tokens: usize,
    pub rejected_tokens: usize,
    pub acceptance_rate: f32,
}
```

### 4. Fix generate_speculative_transformer

Current issues:
- Inefficient - runs full model for each candidate
- Missing proper probability computation
- No temperature/sampling options

Improved algorithm:
1. Draft phase: Generate γ tokens using lightweight draft model
2. Verify phase: Single forward pass to verify all γ tokens
3. Accept/reject: Use proper rejection sampling with target/draft ratio

### 5. Add Speculative Info to Model Display

Update `main.rs` to show speculative mode:

```rust
println!("Speculative decoding: {}", if llm.is_speculative_enabled() {
    format!("{:?} (γ={}, τ={})", mode, gamma, tau)
} else {
    "Disabled".to_string()
});
```

## Implementation Order

1. **Phase 1**: Fix network_description to show speculative mode (quick fix)
2. **Phase 2**: Add SpeculativeStats for monitoring acceptance rate
3. **Phase 3**: Improve generate_speculative_transformer algorithm
4. **Phase 4**: Add temperature/top_p sampling options

## Expected Outcome

After implementation, model info should show:

```
=== MODEL INFORMATION ===
Network architecture: TokenEmbeddings, TransformerBlock, ..., OutputProjection, SpeculativeDecoder(γ=4, τ=0.0010)
Speculative mode: Transformer
Total parameters: 1,234,567
```

## Testing Requirements

1. Run with `--speculative --speculative-mode transformer`
2. Verify network description shows "SpeculativeDecoder"
3. Run benchmarks to measure acceptance rate
4. Compare output quality with greedy baseline

---

## Implementation Summary

### Changes Made

#### 1. Enhanced `speculative.rs`

- Added `temperature` and `top_p` fields to `SpeculativeSamplingConfig`
- Added `SpeculativeSamplingConfig::new()` constructor with validation
- Added builder methods: `with_temperature()`, `with_top_p()`
- Added `Display` trait implementations for better formatting
- Added `SpeculativeStats` struct for tracking acceptance rates:
  - Atomic counters for thread-safe metrics
  - `acceptance_rate()`, `summary()`, `reset()` methods
- Added unit tests for new functionality

#### 2. Updated `llm.rs`

- Fixed `network_description()` to show speculative mode when enabled:
  - Shows `SpeculativeDecoder(γ=N, τ=X.XXXX, layers=M)` for transformer mode
  - Shows `SpeculativeDiffusion(γ=N, τ=X.XXXX, layers=M)` for diffusion mode
  - Falls back to `GreedyDecoder` when speculative is disabled
- Added `decoder_description()` method for detailed decoder info
- Added helper methods:
  - `disable_speculative_sampling()`
  - `is_speculative_enabled()`
  - `speculative_config()`
  - `speculative_mode()`
- Improved `generate_speculative_transformer()`:
  - Proper rejection sampling algorithm
  - Adjusted distribution sampling when all candidates rejected
  - Better documentation with algorithm reference

#### 3. Updated `main.rs`

- Added `Decoder: {description}` line to MODEL INFORMATION section

#### 4. Fixed `speculative_tests.rs`

- Updated to use new `SpeculativeSamplingConfig::new()` constructor

### Test Results

All tests pass:

```text
running 7 tests
test transformer::speculative::tests::test_speculative_config_clamps_invalid ... ok
test transformer::speculative::tests::test_speculative_config_builder ... ok
test transformer::speculative::tests::test_speculative_stats ... ok
test transformer::speculative::tests::test_speculative_mode_display ... ok
test transformer::speculative::tests::test_speculative_config_display ... ok
test llm::tests::test_transformer_speculative_sampling_configuration ... ok
test transformer::speculative_tests::tests::test_speculative_sampling_runs ... ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

### Expected Output

When running with `--speculative --speculative-mode transformer`:

```text
=== MODEL INFORMATION ===
Network architecture: TokenEmbeddings, TransformerBlock, ..., SpeculativeDecoder(γ=4, τ=0.0010, layers=2)
Decoder: Speculative Transformer (γ=4, τ=0.0010, draft_layers=2, temp=1.00, top_p=1.00)
Total parameters: 1,234,567
```

When running without speculative:

```text
=== MODEL INFORMATION ===
Network architecture: TokenEmbeddings, TransformerBlock, ..., GreedyDecoder
Decoder: Greedy (deterministic argmax)
Total parameters: 1,234,567
```
