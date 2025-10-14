# Software Requirements Specification (SRS)

## Project: RustGPT - Educational Transformer LLM Implementation

### Version: 0.1.0
### Status: Active Development
### Last Updated: 2025-10-14

---

## 1. Introduction

### 1.1 Purpose
This Software Requirements Specification (SRS) defines the detailed technical requirements for RustGPT, an educational transformer-based language model implementation in Rust. This document serves as the single source of truth for implementation, testing, and validation.

### 1.2 Scope
RustGPT provides a complete, from-scratch implementation of a transformer architecture including:
- Token and positional embeddings
- Self-attention mechanisms with causal masking
- Feed-forward networks
- Layer normalization
- Backpropagation and gradient-based optimization
- Training and inference pipelines

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|------------|
| LLM | Large Language Model |
| FFN | Feed-Forward Network |
| MHA | Multi-Head Attention |
| ReLU | Rectified Linear Unit activation function |
| Adam | Adaptive Moment Estimation optimizer |
| EOS | End-of-Sequence token |
| SSOT | Single Source of Truth |
| DDD | Domain-Driven Design |
| TDD | Test-Driven Development |

### 1.4 References
- Vaswani et al., "Attention Is All You Need" (2017)
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
- Ba et al., "Layer Normalization" (2016)
- Rust Programming Language Documentation (2024 edition)

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    RustGPT System                        │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Training   │  │  Inference   │  │ Interactive  │  │
│  │   Pipeline   │  │   Engine     │  │     Mode     │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                  │                  │          │
│         └──────────────────┼──────────────────┘          │
│                            │                             │
│                    ┌───────▼────────┐                    │
│                    │   LLM Core     │                    │
│                    │  (Layer Trait) │                    │
│                    └───────┬────────┘                    │
│         ┌──────────────────┼──────────────────┐          │
│         │                  │                  │          │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼──────┐  │
│  │ Embeddings  │  │  Transformer    │  │   Output   │  │
│  │   Layer     │  │     Blocks      │  │ Projection │  │
│  └─────────────┘  └────────┬────────┘  └────────────┘  │
│                    ┌────────▼────────┐                   │
│                    │ Self-Attention  │                   │
│                    │  Feed-Forward   │                   │
│                    │  Layer Norm     │                   │
│                    └─────────────────┘                   │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Vocabulary   │  │    Adam      │  │   Dataset    │  │
│  │  Manager     │  │  Optimizer   │  │    Loader    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Module Responsibilities

#### 2.2.1 Core Modules

| Module | Responsibility | Public API |
|--------|---------------|------------|
| `llm` | Orchestrates training/inference, implements Layer trait | `LLM::new()`, `train()`, `predict()`, `tokenize()` |
| `embeddings` | Token and positional embeddings | `Embeddings::new()`, implements `Layer` |
| `transformer` | Transformer block composition | `TransformerBlock::new()`, implements `Layer` |
| `self_attention` | Scaled dot-product attention with causal masking | `SelfAttention::new()`, implements `Layer` |
| `feed_forward` | Position-wise FFN with ReLU | `FeedForward::new()`, implements `Layer` |
| `layer_norm` | Layer normalization | `LayerNorm::new()`, `normalize()`, `backward()` |
| `output_projection` | Vocabulary projection layer | `OutputProjection::new()`, implements `Layer` |
| `vocab` | Tokenization and vocabulary management | `Vocab::new()`, `encode()`, `decode()` |
| `adam` | Adam optimizer implementation | `Adam::new()`, `step()` |
| `dataset_loader` | Dataset loading and validation | `Dataset::new()`, `load_json()`, `load_csv()` |

---

## 3. Detailed Requirements

### 3.1 Layer Trait (SRS-LAYER)

**SRS-LAYER-001**: All neural network components MUST implement the `Layer` trait.

```rust
pub trait Layer {
    fn layer_type(&self) -> &str;
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
    fn parameters(&self) -> usize;
}
```

**Acceptance Criteria**:
- ✅ `layer_type()` returns unique identifier string
- ✅ `forward()` performs forward pass, returns output of same batch size
- ✅ `backward()` performs backward pass, returns gradients w.r.t. input
- ✅ `parameters()` returns total trainable parameter count

**Test Coverage**:
- Unit tests for each layer implementation
- Integration tests for layer composition
- Property tests for gradient flow

---

### 3.2 Embeddings Layer (SRS-EMB)

**SRS-EMB-001**: Token embeddings MUST map vocabulary indices to dense vectors.

**Mathematical Specification**:
$$
E_{\text{token}}[i] = W_{\text{embed}}[i, :] \quad \text{where } i \in [0, \text{vocab\_size})
$$

**SRS-EMB-002**: Positional embeddings MUST encode sequence position.

**Mathematical Specification**:
$$
E_{\text{pos}}[j] = W_{\text{pos}}[j, :] \quad \text{where } j \in [0, \text{max\_seq\_len})
$$

**SRS-EMB-003**: Final embedding MUST be sum of token and positional embeddings.

$$
E_{\text{final}}[i, j] = E_{\text{token}}[i] + E_{\text{pos}}[j]
$$

**Acceptance Criteria**:
- ✅ Token embedding matrix shape: `(vocab_size, embedding_dim)`
- ✅ Positional embedding matrix shape: `(max_seq_len, embedding_dim)`
- ✅ Output shape: `(batch_size, seq_len, embedding_dim)`
- ✅ Gradients flow correctly through both embedding matrices

**Test Coverage**:
- `test_embeddings_creation`: Validates initialization
- `test_embed_tokens`: Validates token embedding lookup
- `test_positional_embeddings`: Validates positional encoding
- `test_max_sequence_length`: Validates sequence length constraints
- `test_embedding_backwards`: Validates gradient computation

---

### 3.3 Self-Attention (SRS-ATT)

**SRS-ATT-001**: Self-attention MUST compute scaled dot-product attention.

**Mathematical Specification**:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where:
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$
- $d_k$ = embedding dimension
- $M$ = causal mask ($M_{ij} = -\infty$ for $j > i$, else $0$)

**SRS-ATT-002**: Causal masking MUST prevent attention to future tokens.

**Acceptance Criteria**:
- ✅ Query, Key, Value weight matrices shape: `(embedding_dim, embedding_dim)`
- ✅ Attention scores scaled by $\sqrt{d_k}$
- ✅ Causal mask applied before softmax
- ✅ Residual connection: `output = attention(input) + input`
- ✅ Gradients computed via chain rule

**Test Coverage**:
- `test_self_attention_forward`: Validates forward pass
- `test_self_attention_with_different_sequence_lengths`: Validates variable lengths
- Property tests for attention weight properties (sum to 1, non-negative)

---

### 3.4 Feed-Forward Network (SRS-FFN)

**SRS-FFN-001**: FFN MUST implement two-layer network with ReLU activation.

**Mathematical Specification**:
$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

Where:
- $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
- $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
- $\text{ReLU}(x) = \max(0, x)$

**SRS-FFN-002**: Residual connection MUST be applied.

$$
\text{output} = \text{FFN}(x) + x
$$

**Acceptance Criteria**:
- ✅ Weight matrix $W_1$ shape: `(embedding_dim, hidden_dim)`
- ✅ Weight matrix $W_2$ shape: `(hidden_dim, embedding_dim)`
- ✅ Bias vectors initialized to zero
- ✅ ReLU activation applied element-wise
- ✅ Residual connection preserves input shape
- ✅ Gradients computed via chain rule with ReLU derivative

**Test Coverage**:
- `test_feed_forward_forward`: Validates forward pass
- `test_feed_forward_and_backward`: Validates gradient computation
- `test_feed_forward_with_different_sequence_lengths`: Validates variable lengths

---

### 3.5 Layer Normalization (SRS-NORM)

**SRS-NORM-001**: Layer normalization MUST normalize across feature dimension.

**Mathematical Specification**:
$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:
- $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ (mean)
- $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2$ (variance)
- $\epsilon = 10^{-5}$ (numerical stability)
- $\gamma, \beta$ are learnable parameters

**Acceptance Criteria**:
- ✅ Normalization applied per-sample, across features
- ✅ Epsilon prevents division by zero
- ✅ Learnable scale ($\gamma$) and shift ($\beta$) parameters
- ✅ Gradients computed w.r.t. input, $\gamma$, and $\beta$

**Test Coverage**:
- Unit tests for normalization properties (mean ≈ 0, variance ≈ 1)
- Property tests for numerical stability

---

### 3.6 Transformer Block (SRS-TRANS)

**SRS-TRANS-001**: Transformer block MUST compose attention, FFN, and layer norms.

**Architecture**:
```
input
  ↓
self_attention (with residual)
  ↓
layer_norm_1
  ↓
feed_forward (with residual)
  ↓
layer_norm_2
  ↓
output
```

**Acceptance Criteria**:
- ✅ Self-attention includes residual connection
- ✅ Feed-forward includes residual connection
- ✅ Layer norms applied after residual connections
- ✅ Backward pass correctly chains gradients

**Test Coverage**:
- `test_transformer_block`: Validates forward/backward passes

---

### 3.7 Output Projection (SRS-OUT)

**SRS-OUT-001**: Output projection MUST map embeddings to vocabulary logits.

**Mathematical Specification**:
$$
\text{logits} = xW_{\text{out}} + b_{\text{out}}
$$

Where:
- $W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times \text{vocab\_size}}$
- $b_{\text{out}} \in \mathbb{R}^{\text{vocab\_size}}$

**Acceptance Criteria**:
- ✅ Weight matrix shape: `(embedding_dim, vocab_size)`
- ✅ Bias vector shape: `(vocab_size,)`
- ✅ Output shape: `(batch_size, seq_len, vocab_size)`

**Test Coverage**:
- `test_output_projection_forward`: Validates forward pass
- `test_output_projection_backward`: Validates gradient computation

---

### 3.8 Training Pipeline (SRS-TRAIN)

**SRS-TRAIN-001**: Training MUST implement teacher-forcing with next-token prediction.

**Algorithm**:
```
for epoch in 1..epochs:
    for sequence in dataset:
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        
        logits = forward(input_ids)
        probs = softmax(logits)
        loss = cross_entropy(probs, target_ids)
        
        grads = compute_gradients(probs, target_ids)
        clip_gradients(grads, max_norm=5.0)
        backward(grads, learning_rate)
```

**SRS-TRAIN-002**: Cross-entropy loss MUST be computed per-token.

**Mathematical Specification**:
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p(y_i | x_i)
$$

**SRS-TRAIN-003**: Gradient clipping MUST prevent exploding gradients.

**Mathematical Specification**:
$$
\text{if } \|\nabla\| > \text{max\_norm}: \quad \nabla \leftarrow \nabla \frac{\text{max\_norm}}{\|\nabla\|}
$$

**Acceptance Criteria**:
- ✅ Loss decreases over epochs
- ✅ Gradients clipped to L2 norm ≤ 5.0
- ✅ Adam optimizer updates all parameters
- ✅ Training completes without NaN/Inf

**Test Coverage**:
- `test_llm_train`: Validates training loop
- `test_llm_integration`: End-to-end training test
- Property tests for loss monotonicity

---

### 3.9 Inference Pipeline (SRS-INFER)

**SRS-INFER-001**: Inference MUST use autoregressive generation.

**Algorithm**:
```
tokens = tokenize(prompt)
for _ in range(max_new_tokens):
    logits = forward(tokens)
    probs = softmax(logits[-1])  # Last token
    next_token = greedy_decode(probs)
    tokens.append(next_token)
    if next_token == EOS:
        break
return detokenize(tokens)
```

**SRS-INFER-002**: Greedy decoding MUST select highest probability token.

**Mathematical Specification**:
$$
\text{next\_token} = \arg\max_i p_i
$$

**Acceptance Criteria**:
- ✅ Generation stops at EOS token or max length
- ✅ Output tokens are valid vocabulary indices
- ✅ Detokenization produces readable text

**Test Coverage**:
- `test_llm_predict`: Validates prediction
- `test_llm_predict_empty_input`: Edge case handling
- `test_llm_predict_max_seq_len`: Boundary condition

---

### 3.10 Tokenization (SRS-TOK)

**SRS-TOK-001**: Tokenization MUST split on whitespace and punctuation.

**Algorithm**:
```
for word in text.split_whitespace():
    for char in word:
        if char.is_punctuation():
            if current_word:
                tokens.append(encode(current_word))
            tokens.append(encode(char))
            current_word = ""
        else:
            current_word += char
    if current_word:
        tokens.append(encode(current_word))
```

**SRS-TOK-002**: Unknown tokens MUST be skipped (not added to output).

**Acceptance Criteria**:
- ✅ Punctuation separated into individual tokens
- ✅ Whitespace used as word boundary
- ✅ Unknown words filtered out
- ✅ EOS token handled specially

**Test Coverage**:
- `test_llm_tokenize`: Basic tokenization
- `test_llm_tokenize_punctuation`: Punctuation handling
- `test_llm_tokenize_unknown_words`: Unknown token filtering
- Property tests for tokenization invariants

---

### 3.11 Model Persistence (SRS-PERSIST)

**SRS-PERSIST-001**: LLM MUST support saving trained models to disk.

**API Specification**:

```rust
impl LLM {
    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>>;
}
```

**SRS-PERSIST-002**: Save/load MUST preserve all model parameters and vocabulary.

**Acceptance Criteria**:

- ✅ Round-trip serialization: `load(save(model)) == model`
- ✅ Vocabulary encoding/decoding preserved
- ✅ All layer parameters preserved
- ✅ Network architecture preserved

**SRS-PERSIST-003**: Serialization MUST support both JSON and binary formats.

**Format Detection**:

- `.json` extension → JSON format (human-readable)
- Other extensions → Binary format (compact)

**Acceptance Criteria**:

- ✅ JSON format produces readable text
- ✅ Binary format smaller than JSON
- ✅ Both formats preserve model identically

**SRS-PERSIST-004**: Layer polymorphism MUST be serializable via LayerEnum.

**LayerEnum Definition**:

```rust
#[derive(Serialize, Deserialize)]
pub enum LayerEnum {
    Embeddings(Embeddings),
    SelfAttention(Box<SelfAttention>),
    FeedForward(Box<FeedForward>),
    LayerNorm(LayerNorm),
    OutputProjection(OutputProjection),
}
```

**Acceptance Criteria**:

- ✅ All layer types represented in enum
- ✅ Large variants boxed to reduce enum size
- ✅ Enum implements Layer trait
- ✅ Serialization/deserialization works

**Test Coverage**:

- `test_llm_save_load_json`: JSON round-trip
- `test_llm_save_load_binary`: Binary round-trip
- `test_llm_save_load_auto_detect`: Format auto-detection
- `test_binary_smaller_than_json`: Size comparison
- `test_json_is_human_readable`: Format validation
- `test_save_load_preserves_vocab`: Vocabulary preservation
- `test_load_nonexistent_file`: Error handling

---

## 4. Quality Attributes

### 4.1 Testability (SRS-TEST)

**SRS-TEST-001**: All modules MUST have ≥80% line coverage.

**SRS-TEST-002**: Property-based tests MUST validate mathematical invariants:

- Softmax outputs sum to 1.0
- Attention weights are non-negative
- Gradients have finite values
- Parameter counts are deterministic

**SRS-TEST-003**: Edge cases MUST be explicitly tested:

- Empty inputs
- Maximum sequence length
- Single-token sequences
- Unknown tokens

---

### 4.2 Maintainability (SRS-MAINT)

**SRS-MAINT-001**: Files MUST be ≤500 lines.

**SRS-MAINT-002**: Public APIs MUST have rustdoc comments with examples.

**SRS-MAINT-003**: Mathematical algorithms MUST include inline LaTeX notation.

---

### 4.3 Performance (SRS-PERF)

**SRS-PERF-001**: Training MUST complete 100 epochs in ≤10 minutes on modern CPU.

**SRS-PERF-002**: Inference latency MUST be ≤100ms per token.

**SRS-PERF-003**: Memory usage MUST be ≤2GB for default configuration.

---

## 5. Traceability Matrix

| Requirement | Test(s) | Status |
|-------------|---------|--------|
| SRS-LAYER-001 | All layer tests | ✅ Pass |
| SRS-EMB-001 | `test_embed_tokens` | ✅ Pass |
| SRS-EMB-002 | `test_positional_embeddings` | ✅ Pass |
| SRS-EMB-003 | `test_embeddings_creation` | ✅ Pass |
| SRS-ATT-001 | `test_self_attention_forward` | ✅ Pass |
| SRS-ATT-002 | Manual inspection | ✅ Pass |
| SRS-FFN-001 | `test_feed_forward_forward` | ✅ Pass |
| SRS-FFN-002 | `test_feed_forward_and_backward` | ✅ Pass |
| SRS-TRANS-001 | `test_transformer_block` | ✅ Pass |
| SRS-OUT-001 | `test_output_projection_forward` | ✅ Pass |
| SRS-TRAIN-001 | `test_llm_train` | ✅ Pass |
| SRS-TRAIN-002 | `test_llm_integration` | ✅ Pass |
| SRS-TRAIN-003 | Manual inspection | ✅ Pass |
| SRS-INFER-001 | `test_llm_predict` | ✅ Pass |
| SRS-INFER-002 | `test_greedy_decode_properties` | ✅ Pass |
| SRS-TOK-001 | `test_llm_tokenize` | ✅ Pass |
| SRS-TOK-002 | `test_llm_tokenize_unknown_words` | ✅ Pass |
| SRS-PERSIST-001 | `test_llm_save_load_*` | ✅ Pass |
| SRS-PERSIST-002 | `test_save_load_preserves_vocab` | ✅ Pass |
| SRS-PERSIST-003 | `test_binary_smaller_than_json`, `test_json_is_human_readable` | ✅ Pass |
| SRS-PERSIST-004 | `test_llm_save_load_auto_detect` | ✅ Pass |
| SRS-TEST-002 | `test_softmax_properties`, `prop_*` | ✅ Pass |
| SRS-TEST-003 | `test_*_empty_input`, `test_*_max_seq_len` | ✅ Pass |

---

## 6. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-10-14 | System | Initial SRS creation |

