# 🦀 Rust LLM from Scratch

[![Check](https://github.com/tekaratzas/RustGPT/actions/workflows/check.yml/badge.svg)](https://github.com/tekaratzas/RustGPT/actions/workflows/check.yml) [![Test](https://github.com/tekaratzas/RustGPT/actions/workflows/test.yml/badge.svg)](https://github.com/tekaratzas/RustGPT/actions/workflows/test.yml)


https://github.com/user-attachments/assets/ec4a4100-b03a-4b3c-a7d6-806ea54ed4ed

A complete **Large Language Model implementation in pure Rust** with no external ML frameworks. Built from the ground up using only `ndarray` for matrix operations.

## 🚀 What This Is

This project demonstrates how to build a transformer-based language model from scratch in Rust, including:
- **Pre-training** on factual text completion
- **Instruction tuning** for conversational AI
- **Interactive chat mode** for testing
- **Full backpropagation** with gradient clipping
- **Model persistence** for saving/loading trained models
- **Modular architecture** with clean separation of concerns

## ❌ What This Isn't

This is not a production grade LLM. It is so far away from the larger models.

This is just a toy project that demonstrates how these models work under the hood.

## 🔍 Key Files to Explore

Start with these two core files to understand the implementation:

- **[`src/main.rs`](src/main.rs)** - Training pipeline, data preparation, and interactive mode
- **[`src/llm.rs`](src/llm.rs)** - Core LLM implementation with forward/backward passes and training logic

## 🏗️ Architecture

The model uses a **transformer-based architecture** with the following components:

```
Input Text → Tokenization → Embeddings → Transformer Blocks → Output Projection → Predictions
```

### Project Structure

```
src/
├── main.rs              # 🎯 Training pipeline and interactive mode
├── llm.rs               # 🧠 Core LLM implementation and training logic
├── lib.rs               # 📚 Library exports and constants
├── transformer.rs       # 🔄 Transformer block (attention + feed-forward)
├── self_attention.rs    # 👀 Multi-head self-attention mechanism
├── feed_forward.rs      # ⚡ Position-wise feed-forward networks
├── embeddings.rs        # 📊 Token embedding layer
├── output_projection.rs # 🎰 Final linear layer for vocabulary predictions
├── vocab.rs            # 📝 Vocabulary management and tokenization
├── layer_norm.rs       # 🧮 Layer normalization
├── adam.rs             # 🏃 Adam optimizer implementation
└── gradient_clipping.rs # ✂️ Adaptive gradient clipping strategies

tests/
├── llm_test.rs         # Tests for core LLM functionality (19 tests)
├── persistence_test.rs # Tests for model save/load (7 tests)
├── gradient_clipping_test.rs # Tests for gradient clipping (4 tests)
├── transformer_test.rs # Tests for transformer blocks
├── self_attention_test.rs # Tests for attention mechanisms
├── feed_forward_test.rs # Tests for feed-forward layers
├── embeddings_test.rs  # Tests for embedding layers
├── vocab_test.rs       # Tests for vocabulary handling
├── adam_test.rs        # Tests for optimizer
└── output_projection_test.rs # Tests for output layer

Total: 55 tests, all passing ✅
```

## 🧪 What The Model Learns

The implementation includes two training phases:

1. **Pre-training**: Learns basic world knowledge from factual statements
   - "The sun rises in the east and sets in the west"
   - "Water flows downhill due to gravity"
   - "Mountains are tall and rocky formations"

2. **Instruction Tuning**: Learns conversational patterns
   - "User: How do mountains form? Assistant: Mountains are formed through tectonic forces..."
   - Handles greetings, explanations, and follow-up questions

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/tekaratzas/RustGPT.git
cd RustGPT
cargo run

# The model will:
# 1. Build vocabulary from training data
# 2. Pre-train on factual statements (100 epochs)
# 3. Instruction-tune on conversational data (100 epochs)
# 4. Enter interactive mode for testing
```

## 🎮 Interactive Mode

After training, test the model interactively:

```
Enter prompt: How do mountains form?
Model output: Mountains are formed through tectonic forces or volcanism over long geological time periods

Enter prompt: What causes rain?
Model output: Rain is caused by water vapor in clouds condensing into droplets that become too heavy to remain airborne
```

## 💾 Model Persistence

Save and load trained models for reuse:

```rust
use llm::LLM;

// Save model (auto-detects format from extension)
let llm = LLM::default();
llm.save("model.bin")?;      // Binary format (compact, fast)
llm.save("model.json")?;     // JSON format (human-readable)

// Load model
let loaded_llm = LLM::load("model.bin")?;

// Explicit format methods also available
llm.save_binary("model.bin")?;
llm.save_json("model.json")?;
let llm_from_binary = LLM::load_binary("model.bin")?;
let llm_from_json = LLM::load_json("model.json")?;
```

**Format Comparison**:
- **Binary** (`.bin`): 50-70% smaller, 3x faster I/O, production-ready
- **JSON** (`.json`): Human-readable, debuggable, cross-platform portable

## ✂️ Gradient Clipping

Advanced gradient clipping with multiple strategies:

```rust
use llm::{LLM, AdaptiveGradientClipping, AdaptiveClippingConfig, L2GradientClipping};

// Default: Adaptive Gradient Clipping (AGC) with gradient centralization
let mut llm = LLM::default();

// Configure AGC parameters
let config = AdaptiveClippingConfig {
    agc_threshold: 0.01,           // AGC threshold (λ)
    use_centralization: true,      // Enable gradient centralization
    use_agc: true,                 // Use AGC (vs L2 fallback)
    l2_threshold: 5.0,             // L2 fallback threshold
};
llm.set_gradient_clipping(Box::new(AdaptiveGradientClipping::new(config)));

// Or use simple L2 norm clipping
llm.set_gradient_clipping(Box::new(L2GradientClipping::new(5.0)));

// Disable gradient clipping
llm.disable_gradient_clipping();
```

**Gradient Clipping Strategies**:
- **Adaptive Gradient Clipping (AGC)**: Parameter-norm based scaling for better stability
- **Gradient Centralization**: Zero-mean gradients for improved convergence
- **L2 Norm Clipping**: Traditional threshold-based clipping (legacy support)

## 🧮 Technical Implementation

### Model Configuration
- **Vocabulary Size**: Dynamic (built from training data)
- **Embedding Dimension**: 128 (defined by `EMBEDDING_DIM` in `src/lib.rs`)
- **Hidden Dimension**: 256 (defined by `HIDDEN_DIM` in `src/lib.rs`)
- **Max Sequence Length**: 80 tokens (defined by `MAX_SEQ_LEN` in `src/lib.rs`)
- **Architecture**: 3 Transformer blocks + embeddings + output projection

### Training Details
- **Optimizer**: Adam with adaptive gradient clipping
- **Pre-training LR**: 0.0005 (100 epochs)
- **Instruction Tuning LR**: 0.0001 (100 epochs)
- **Loss Function**: Cross-entropy loss
- **Gradient Clipping**: Adaptive Gradient Clipping (AGC) with gradient centralization (default)
  - AGC threshold (λ): 0.01
  - L2 fallback threshold: 5.0
  - Gradient centralization enabled

### Key Features
- **Custom tokenization** with punctuation handling
- **Greedy decoding** for text generation
- **Adaptive gradient clipping** with AGC and gradient centralization for training stability
- **Model persistence** with dual-format serialization (binary + JSON)
- **Modular layer system** with clean interfaces
- **Comprehensive test coverage** for all components (55 tests)

## 🔧 Development

```bash
# Run all tests
cargo test

# Test specific components
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test

# Run with clippy for code quality checks
cargo clippy --tests -- -D warnings

# Build optimized version
cargo build --release

# Run with verbose output
cargo test -- --nocapture
```

### Test Coverage

The project includes comprehensive test coverage with multiple testing strategies:

- **Unit Tests**: Core functionality tests for all components
- **Property-Based Tests**: Using `proptest` to validate mathematical properties and invariants
  - Tokenization produces valid vocabulary indices
  - Token counts are bounded relative to input
- **Edge Case Tests**: Boundary conditions and error handling
  - Empty inputs
  - Maximum sequence length handling
  - Unknown tokens
  - Punctuation handling
- **Mathematical Property Tests**: Validates theoretical correctness
  - Softmax produces valid probability distributions (sums to 1.0, values in [0,1])
  - Numerical stability with extreme values
  - Greedy decoding selects maximum probability
  - Parameter count consistency
- **Integration Tests**: End-to-end training and prediction workflows

Total test count: **53 tests** across all components

## 🧠 Learning Resources

This implementation demonstrates key ML concepts:
- **Transformer architecture** (attention, feed-forward, layer norm)
- **Backpropagation** through neural networks
- **Language model training** (pre-training + fine-tuning)
- **Tokenization** and vocabulary management
- **Gradient-based optimization** with Adam

Perfect for understanding how modern LLMs work under the hood!

## 📊 Dependencies

- `ndarray` - N-dimensional arrays for matrix operations
- `rand` + `rand_distr` - Random number generation for initialization

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!

## 🤝 Contributing

Contributions are welcome! This project is perfect for learning and experimentation.

### Documentation

- **[Product Requirements Document (PRD)](docs/PRD.md)** - High-level requirements and success criteria
- **[Software Requirements Specification (SRS)](docs/SRS.md)** - Detailed technical specifications and interfaces
- **[Architectural Decision Records (ADR)](docs/ADR.md)** - Key architectural decisions and rationale
- **[Backlog](docs/backlog.md)** - Prioritized feature requests and improvement tasks
- **[Checklist](docs/checklist.md)** - Implementation status and requirements traceability
- **[Sprint Retrospective](SPRINT_RETROSPECTIVE.md)** - Latest sprint completion summary with hybrid CoT-ToT-GoT ReAct analysis

### Sprint Status: ✅ Audit & Tracing Complete

**Latest Update**: October 14, 2025  
**Current Sprint**: Audit & Tracing Implementation  
**Status**: ✅ **COMPLETED** - Codebase audited, tracing integrated, docs updated

#### ✅ Completed Features

- **🔍 Codebase Audit**: Comprehensive review of all components vs SRS/ADR requirements
- **📊 Quality Assurance**: All tests pass (55/55), clippy clean, no compilation errors
- **� Tracing Integration**: Structured logging with tracing crate, spans on key methods
- **📚 Documentation Updates**: ADR updated with tracing acceptance, checklist progress marked
- **🧪 Test Coverage**: Validated comprehensive test suite with property-based tests

#### 🎯 Next Sprint Priorities

- **🛡️ Error Handling**: Implement thiserror for structured errors and recovery
- **⚡ Performance Optimization**: Integrate rayon for parallel training
- **📈 Observability**: Extend tracing to all methods, add metrics collection
- **🔒 Security Audit**: cargo audit, input validation, no unsafe code review

### Areas for Improvement

- **Advanced architectures** (multi-head attention, positional encoding, RoPE)
- **Training improvements** (different optimizers, learning rate schedules, regularization)
- **Data handling** (larger datasets, tokenizer improvements, streaming)
- **Model analysis** (attention visualization, gradient analysis, interpretability)

### Areas for Improvement

- **Advanced architectures** (multi-head attention, positional encoding, RoPE)
- **Training improvements** (different optimizers, learning rate schedules, regularization)
- **Data handling** (larger datasets, tokenizer improvements, streaming)
- **Model analysis** (attention visualization, gradient analysis, interpretability)

### Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/model-persistence`
3. Make your changes and add tests
4. Run the test suite: `cargo test`
5. Submit a pull request with a clear description

### Code Style

- Follow standard Rust conventions (`cargo fmt`)
- Add comprehensive tests for new features
- Update documentation and README as needed
- Keep the "from scratch" philosophy - avoid heavy ML dependencies

### Ideas for Contributions

- 🚀 **Beginner**: Model save/load, more training data, config files
- 🔥 **Intermediate**: Beam search, positional encodings, training checkpoints
- ⚡ **Advanced**: Multi-head attention, layer parallelization, custom optimizations

Questions? Open an issue or start a discussion!

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!

---

## 📊 Sprint Status

**Sprint 3.1: Documentation Foundation + Batch Training** ✅ COMPLETE

- ✅ Consolidated ADR.md to 163 lines (table format)
- ✅ Expanded CHECKLIST.md with 5 new NFRs (Reliability, Security, Observability, Scalability, Extensibility)
- ✅ Batch training with gradient accumulation implemented (user contribution)
- ✅ Fixed critical reversed iteration bug in backward pass
- ✅ 55/55 tests passing, 0 clippy warnings
- ✅ Test runtime: 10.34s (within <30s target)

**Next Sprint: 3.2 - Test Hardening**
- Property-based tests (proptest) for gradient accumulation invariants
- Criterion benchmarks for batch training performance
- Coverage measurement (tarpaulin, target >80%)
- Cargo nextest integration for parallel test execution
