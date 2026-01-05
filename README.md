# 🦀 RustGPT: Advanced LLM Implementation in Pure Rust

[![Check](https://github.com/tekaratzas/RustGPT/actions/workflows/check.yml/badge.svg)](https://github.com/tekaratzas/RustGPT/actions/workflows/check.yml) [![Test](https://github.com/tekaratzas/RustGPT/actions/workflows/test.yml/badge.svg)](https://github.com/tekaratzas/RustGPT/actions/workflows/test.yml)

A **complete Large Language Model implementation in pure Rust** with advanced architectures including Transformers, TRM (Transformer-Recurrent Mixtures), Diffusion models, Mamba, and RG-LRU. Built from scratch using only `ndarray` for matrix operations.

## 🚀 What This Is

RustGPT is an educational and experimental platform demonstrating modern LLM architectures:

- **Multiple Architecture Support**: Transformers, TRM, Diffusion models, Mamba, RG-LRU
- **Advanced Features**: Speculative sampling, Mixture of Experts, Adaptive residuals
- **Comprehensive Training**: Pre-training + instruction tuning pipelines
- **Robust Error Handling**: Proper Result types, no panic!() calls
- **Production-grade Serialization**: Versioned model persistence with integrity checks
- **Extensive Testing**: 183+ unit tests with property-based testing

## 🏗️ Current Architecture

The project now supports multiple advanced architectures:

### 1. **Transformer Architecture**
```
Input → Tokenization → Embeddings → Transformer Blocks → Output Projection → Predictions
```

### 2. **TRM (Transformer-Recurrent Mixture)**
Hybrid architecture combining transformer attention with recurrent components for improved efficiency.

### 3. **Diffusion Models**
Denoising diffusion probabilistic models for text generation with progressive refinement.

### 4. **Mamba**
State-space models with selective scan mechanisms for linear-time sequence processing.

### 5. **RG-LRU (Real-Gated Linear Recurrent Units)**
Trainable temporal-mixing layers with diagonal, stable recurrence for efficient sequence processing.

### 6. **MoH-RG-LRU (Multi-head RG-LRU with Mixture-of-Heads)**
Combines multiple RG-LRU heads with learned gating for improved capacity and efficiency.

### Key Components

- **Polynomial Attention**: Multi-head attention with polynomial logit transformations
- **Richards GLU**: Advanced gating mechanisms with Richards curve activation
- **Adaptive Residuals**: Dynamic residual scaling for stable training
- **Mixture of Experts**: Sparse expert routing for improved capacity
- **Speculative Sampling**: Accelerated decoding with draft-verify mechanisms
- **Modular Transformer Components**: AttentionContext, FeedforwardProcessor, NormalizationLayer, and ResidualConnection for flexible architecture composition
- **Temporal Mixing**: Supports both attention and RG-LRU as temporal mixing mechanisms

## 🔍 Project Structure

```
src/
├── main.rs                  # 🎯 Training pipeline and CLI
├── llm.rs                   # 🧠 Core LLM implementation
├── lib.rs                   # 📚 Library exports and constants
├── attention/               # 👀 Advanced attention mechanisms
├── layers/                  # 🏗️ Layer implementations
│   ├── transformer/         # Transformer blocks
│   ├── recurrence/          # Recurrent components
│   ├── ssm/                 # State-space models (Mamba, RG-LRU)
│   ├── diffusion/           # Diffusion model components
│   └── components/          # Shared components
├── mixtures/                # 🧪 Mixture of Experts
├── decoding/                # 🎰 Decoding strategies
├── encoding/                # 📝 Tokenization and vocabulary
├── richards/                # 📈 Richards curve utilities
├── eprop/                   # 🔄 Training and optimization
└── ... (20+ modules)

tests/
├── attention_parallel.rs   # Attention mechanism tests
├── model_persistence_roundtrip.rs # Serialization tests
├── transformer_block_stability.rs # Stability tests
└── ... (183+ unit tests)
```

## 🧪 Training Pipeline

The model supports a sophisticated training process:

### 1. **Pre-training Phase**
- Learns basic language patterns and world knowledge
- Uses factual statements and general text data
- Configurable epochs and learning rates

### 2. **Instruction Tuning Phase**
- Fine-tunes for conversational AI capabilities
- Uses question-answer pairs and dialogue data
- Lower learning rate for refinement

### 3. **Advanced Features**
- **Speculative Sampling**: `--speculative` flag enables draft-verify decoding
- **Diffusion Training**: `--diffusion` flag enables diffusion-based training
- **Mixture of Experts**: Configurable expert routing strategies
- **Adaptive Windowing**: Dynamic attention window adaptation

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/tekaratzas/RustGPT.git
cd RustGPT
cargo run --release

# Basic training (default transformer)
cargo run --release

# With speculative sampling (transformer mode)
cargo run --release -- --speculative --speculative-mode transformer

# With speculative sampling (diffusion mode)
cargo run --release -- --speculative --speculative-mode diffusion

# With Mamba architecture
cargo run --release -- --architecture mamba

# With RG-LRU architecture
cargo run --release -- --architecture rg-lru

# With deterministic training (fixed seed)
cargo run --release -- --seed 42

# Continue training from saved model
cargo run --release -- --continue-from models/rustgpt.bin
```

## 🎮 Interactive Mode

After training, test the model interactively:

```bash
# Run with interactive flag
cargo run --release -- --interactive

# Example conversation
Enter prompt: How do mountains form?
Model: Mountains form through tectonic forces or volcanism over geological time

Enter prompt: What causes rain?
Model: Rain occurs when water vapor condenses into droplets that become too heavy to remain airborne

# Interactive mode with specific architecture
cargo run --release -- --architecture mamba --interactive
```

## 💾 Model Persistence

### Versioned Serialization with Integrity Checks

```rust
use llm::{LLM, ModelPersistence};

// Save with versioning, checksums, and metadata
let llm = LLM::default();
llm.save_versioned("model.rgpt", Some("Trained RustGPT model".to_string()))?;

// Load with automatic validation
let loaded_llm = LLM::load_versioned("model.rgpt")?;
// ✅ Validates SHA256 checksum
// ✅ Checks version compatibility  
// ✅ Includes comprehensive metadata

// Save different architectures
let mamba_llm = LLM::new_mamba(vocab.clone(), config);
mamba_llm.save_versioned("mamba_model.rgpt", Some("Mamba architecture".to_string()))?;

let rg_lru_llm = LLM::new_rg_lru(vocab.clone(), config);
rg_lru_llm.save_versioned("rg_lru_model.rgpt", Some("RG-LRU architecture".to_string()))?;
```

### Format Options

- **Binary** (`.bin`, `.rgpt`): Compact, fast I/O, production-ready
- **JSON** (`.json`): Human-readable, debuggable
- **MessagePack**: Efficient binary format with schema support

## 🧮 Technical Implementation

### Current Configuration
- **Vocabulary Size**: Dynamic (up to 50,000 tokens)
- **Embedding Dimension**: 128 (configurable)
- **Hidden Dimension**: 256 (configurable)
- **Max Sequence Length**: 256 tokens
- **Architecture Options**: Transformer, TRM, Diffusion, Mamba, RG-LRU, MoH-RG-LRU
- **Normalization**: Richards-based Dynamic Tanh Normalization
- **Positional Encoding**: CoPE (Context-aware Positional Encoding)
- **Activation**: Richards GLU and SwiGLU
- **Temporal Mixing**: Attention or RG-LRU (configurable per transformer block)
- **Speculative Sampling**: Transformer and Diffusion modes with configurable gamma and tau

### Training Details
- **Optimizer**: Adam with gradient clipping
- **Learning Rates**: Configurable per phase
- **Loss Function**: Cross-entropy with label smoothing
- **Regularization**: L2 regularization, gradient norm monitoring
- **Batch Processing**: Gradient accumulation for large batches

### Advanced Features

#### Speculative Sampling
- **Draft Model**: Fast approximation model
- **Verification Model**: Full model for validation
- **Gamma Parameter**: Controls speculation aggressiveness
- **Tau Parameter**: Controls acceptance threshold
- **Transformer Support**: New speculative sampling implementation for transformer models
- **Diffusion Support**: Existing speculative sampling for diffusion models

#### Mamba Architecture
- **Selective SSM**: State-space models with input-dependent parameters
- **Causal Convolution**: Depthwise convolution for sequence processing
- **Selective Scan**: Efficient sequence processing with selective state updates

#### RG-LRU Architecture
- **Real-Gated Recurrence**: Trainable temporal mixing with gated updates
- **Diagonal Recurrence**: Stable recurrence with diagonal parameterization
- **Multi-head Support**: MoH-RG-LRU combines multiple heads with learned gating

#### Diffusion Models
- **Karras Schedule**: Noise scheduling for diffusion
- **SNR Weighting**: Signal-to-noise ratio based training
- **Latent Diffusion**: Efficient latent space processing

#### Mixture of Experts
- **Expert Routing**: Top-k gating with load balancing
- **Adaptive Depth**: Dynamic layer selection
- **Threshold Prediction**: Learned routing thresholds

## 🔧 Development & Testing

### Running Tests

```bash
# Run all tests (183+ unit tests)
cargo test --lib

# Run integration tests
cargo test --test transformer_block_stability
cargo test --test model_persistence_roundtrip

# Run attention tests
cargo test --test attention_parallel

# Run with clippy for code quality
cargo clippy --tests -- -D warnings

# Build optimized version
cargo build --release

# Run with verbose output
cargo test -- --nocapture

# Test specific architectures
cargo test --lib -- --test-threads=1  # For deterministic test ordering
```

### Test Coverage

- **183+ Unit Tests**: Core functionality validation
- **Property-Based Tests**: Mathematical invariants using `proptest`
- **Edge Case Testing**: Boundary conditions and error handling
- **Stability Tests**: Gradient boundedness and numerical stability
- **Integration Tests**: End-to-end workflow validation

### Observability

Structured logging via `tracing` crate:

```bash
# Set log level
RUST_LOG=debug cargo run
RUST_LOG=info cargo run   # Default
RUST_LOG=warn cargo run   # Warnings only
RUST_LOG=error cargo run   # Errors only
```

Example training output:
```
INFO  llm::training: Starting pre-training phase
INFO  llm::training: Epoch 1/100 - loss: 2.3456, grad_norm: 0.1234
INFO  llm::training: Epoch 2/100 - loss: 2.1234, grad_norm: 0.0987
INFO  llm::training: Transitioning to instruction tuning phase
```

## 📊 Dependencies

Minimal dependency footprint:

- `ndarray` - N-dimensional arrays for matrix operations
- `rand` + `rand_distr` - Random number generation
- `serde` + `serde_json` - Serialization
- `tracing` - Structured logging
- `rayon` - Parallel processing
- `sha2` - Cryptographic hashing for integrity checks

**No PyTorch, TensorFlow, or Candle** - pure Rust implementation!

## 🤝 Contributing

RustGPT welcomes contributions for learning and experimentation!

### Current Architecture Options

- **Transformer**: Standard transformer blocks
- **TRM**: Transformer-Recurrent Mixture
- **Diffusion**: Denoising diffusion models
- **Mamba**: State-space models with selective scan
- **RG-LRU**: Real-Gated Linear Recurrent Units

### Areas for Contribution

- **🚀 Beginner**: Documentation, examples, test cases
- **🔥 Intermediate**: New layer types, decoding strategies
- **⚡ Advanced**: Architecture improvements, training optimizations

### Getting Started

```bash
# Fork the repository
# Create a feature branch
git checkout -b feature/new-architecture

# Make changes and add tests
# Run the test suite
cargo test

# Submit a pull request
```

### Code Quality Standards

- Follow Rust conventions (`cargo fmt`)
- Comprehensive test coverage for new features
- Proper error handling (no panic!() calls)
- Documentation updates for new functionality

## 📈 Project Status

### Current Capabilities

- ✅ **Multiple Architectures**: Transformer, TRM, Diffusion, Mamba, RG-LRU, MoH-RG-LRU
- ✅ **Advanced Training**: Speculative sampling (Transformer & Diffusion), MoE, adaptive residuals
- ✅ **Robust Serialization**: Versioned persistence with integrity checks
- ✅ **Comprehensive Testing**: 183+ unit tests, property-based testing
- ✅ **Production Error Handling**: Proper Result types throughout
- ✅ **Configurable Pipeline**: CLI-driven training with multiple options
- ✅ **Modular Components**: AttentionContext, FeedforwardProcessor, NormalizationLayer, ResidualConnection
- ✅ **Temporal Mixing**: Configurable attention or RG-LRU per transformer block

### Recent Improvements

- **Latest**: Added modular transformer components for flexible architecture composition
- **Latest**: Implemented speculative sampling for transformer models
- **Latest**: Added Mamba and RG-LRU state-space model implementations
- **Sprint 5.2**: Systematic error handling (eliminated all panic!() calls)
- **Sprint 5.1**: Code quality improvements (removed placeholder comments)
- **Sprint 4.3**: Serialization integrity (SHA256 checksums, versioning)
- **Sprint 4.2**: Training reliability (divergence detection, observability)

### Roadmap

- **Next Sprint**: Convert remaining unwrap() calls in hot paths
- **Future**: Beam search, advanced positional encodings, mixed-precision training
- **Long-term**: Multi-modal capabilities, larger scale training, architecture auto-selection

## 📚 Learning Resources

RustGPT demonstrates modern LLM concepts:

- **Architecture Design**: Multiple neural network architectures
- **Training Techniques**: Speculative sampling, diffusion models
- **Optimization**: Mixture of Experts, adaptive residuals
- **Error Handling**: Production-grade Rust error management
- **Testing**: Comprehensive test strategies for ML systems

Perfect for understanding how state-of-the-art LLMs work under the hood!

---

**No external ML frameworks** - just pure Rust, linear algebra, and careful engineering!