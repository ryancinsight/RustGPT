# Beam Search Implementation (Phase 4 - Secondary Objective)

## Overview

Beam Search is an advanced text generation algorithm that explores multiple hypotheses in parallel, producing higher quality output than greedy decoding. This implementation includes **adaptive beam width**, which dynamically adjusts the number of hypotheses based on prediction confidence.

## What is Beam Search?

### Greedy Decoding (Baseline)

Traditional greedy decoding selects the most likely token at each step:

```
At each step:
  1. Compute probabilities for all tokens
  2. Select token with highest probability
  3. Append to sequence
  4. Repeat
```

**Problem**: Greedy decoding can miss better sequences because it never reconsiders earlier choices.

### Beam Search

Beam search maintains multiple hypotheses (beams) and explores them in parallel:

```
At each step:
  1. For each active beam:
     - Compute probabilities for all tokens
     - Generate top-k candidates
  2. Score all candidates
  3. Keep top beam_width candidates
  4. Repeat until all beams complete
```

**Benefit**: Explores multiple paths, finding better sequences than greedy decoding.

## Key Features

### 1. Fixed Beam Width

Maintain a constant number of hypotheses throughout generation:

```rust
let config = BeamSearchConfig::new()
    .with_beam_width(4);  // Keep 4 hypotheses
```

### 2. Adaptive Beam Width

Dynamically adjust beam width based on prediction confidence:

```rust
let config = BeamSearchConfig::new()
    .with_beam_width(4)
    .with_adaptive_beam(true)
    .with_beam_range(1, 8);
```

**Strategy**:
- **High entropy** (uncertain predictions) → increase beam width
- **Low entropy** (confident predictions) → decrease beam width

**Formula**:
```
entropy = -Σ p(x) * log(p(x))
normalized_entropy = entropy / 7.0  // Normalize to [0, 1]

if normalized_entropy > threshold:
    beam_width = min(beam_width + 1, max_beam_width)
else:
    beam_width = max(beam_width - 1, min_beam_width)
```

### 3. Beam Scoring

Hypotheses are scored using cumulative log probabilities:

```
score = Σ log(p(token_i))
normalized_score = score / length  // Prevent bias towards shorter sequences
```

### 4. Temperature Sampling

Control the randomness of predictions:

```rust
let config = BeamSearchConfig::new()
    .with_temperature(0.8);  // More confident (sharper distribution)
```

- `temperature < 1.0`: More confident, less diverse
- `temperature = 1.0`: No change
- `temperature > 1.0`: More diverse, less confident

## Configuration

### BeamSearchConfig

```rust
pub struct BeamSearchConfig {
    /// Initial beam width (number of hypotheses to maintain)
    pub beam_width: usize,
    
    /// Enable adaptive beam width based on prediction confidence
    pub use_adaptive_beam: bool,
    
    /// Minimum beam width for adaptive beam search
    pub min_beam_width: usize,
    
    /// Maximum beam width for adaptive beam search
    pub max_beam_width: usize,
    
    /// Softmax entropy threshold for beam width adaptation
    pub adaptation_threshold: f32,
    
    /// Maximum generation length
    pub max_length: usize,
    
    /// Sampling temperature
    pub temperature: f32,
}
```

### Default Values

```rust
BeamSearchConfig::default() = {
    beam_width: 4,
    use_adaptive_beam: false,
    min_beam_width: 1,
    max_beam_width: 8,
    adaptation_threshold: 0.5,
    max_length: 100,
    temperature: 1.0,
}
```

## Usage Examples

### Basic Beam Search

```rust
use llm::{LLM, BeamSearchConfig};

let mut llm = LLM::load("model.bin")?;

let config = BeamSearchConfig::new()
    .with_beam_width(4)
    .with_max_length(50);

let output = llm.generate_with_beam_search("User: Hello", &config);
println!("Output: {}", output);
```

### Adaptive Beam Search

```rust
let config = BeamSearchConfig::new()
    .with_beam_width(4)
    .with_adaptive_beam(true)
    .with_beam_range(2, 8)
    .with_adaptation_threshold(0.6)
    .with_max_length(50);

let output = llm.generate_with_beam_search("User: Explain quantum physics", &config);
```

### High-Quality Generation

```rust
// Larger beam width + lower temperature = higher quality
let config = BeamSearchConfig::new()
    .with_beam_width(8)
    .with_temperature(0.7)
    .with_max_length(100);

let output = llm.generate_with_beam_search("User: Write a poem", &config);
```

### Fast Generation

```rust
// Smaller beam width = faster
let config = BeamSearchConfig::new()
    .with_beam_width(2)
    .with_max_length(50);

let output = llm.generate_with_beam_search("User: Quick answer", &config);
```

## Performance Characteristics

### Computational Complexity

| Method | Complexity | Notes |
|--------|-----------|-------|
| Greedy Decoding | O(N × V) | N = length, V = vocab size |
| Beam Search (width=B) | O(N × B × V) | B times slower than greedy |
| Adaptive Beam | O(N × B_avg × V) | B_avg < B_max |

### Memory Usage

| Method | Memory | Notes |
|--------|--------|-------|
| Greedy Decoding | O(N) | Single sequence |
| Beam Search (width=B) | O(B × N) | B sequences |
| Adaptive Beam | O(B_max × N) | Worst case |

### Quality vs. Speed Trade-off

| Beam Width | Quality | Speed | Use Case |
|------------|---------|-------|----------|
| 1 | Baseline | Fastest | Greedy decoding |
| 2-4 | Good | Fast | General purpose |
| 4-8 | Better | Moderate | High quality |
| 8+ | Best | Slow | Maximum quality |

### Adaptive Beam Benefits

- **Saves computation**: Reduces beam width when model is confident
- **Improves quality**: Increases beam width when model is uncertain
- **Automatic tuning**: No need to manually tune beam width

## Implementation Details

### BeamHypothesis

Represents a single hypothesis during beam search:

```rust
pub struct BeamHypothesis {
    pub tokens: Vec<usize>,      // Token sequence
    pub score: f32,               // Cumulative log probability
    pub is_complete: bool,        // Hit end token or max length
}
```

### BeamSearchState

Manages the beam search process:

```rust
pub struct BeamSearchState {
    pub beams: Vec<BeamHypothesis>,           // Active beams
    pub current_beam_width: usize,            // Current beam width (adaptive)
    pub completed: Vec<BeamHypothesis>,       // Completed beams
}
```

**Key Methods**:
- `expand()`: Expand beams with new predictions
- `compute_entropy()`: Compute softmax entropy for adaptation
- `adapt_beam_width()`: Adjust beam width based on entropy
- `mark_complete()`: Mark beams that hit end token or max length
- `get_best()`: Get hypothesis with highest normalized score

### Generation Algorithm

```rust
1. Initialize beam search state with input tokens
2. For each generation step:
   a. For each active beam:
      - Run forward pass through model
      - Get probability distribution
   b. Compute entropy (if adaptive beam enabled)
   c. Adapt beam width based on entropy
   d. Expand beams with top-k candidates
   e. Score and rank all candidates
   f. Keep top beam_width candidates
   g. Mark complete beams (end token or max length)
3. Return best hypothesis (highest normalized score)
```

## Comparison with Greedy Decoding

### Greedy Decoding

**Pros**:
- Fast (O(N × V))
- Simple
- Low memory usage

**Cons**:
- Can miss better sequences
- No exploration
- Locally optimal, not globally optimal

### Beam Search

**Pros**:
- Better quality (explores multiple paths)
- Finds better sequences
- Configurable quality/speed trade-off

**Cons**:
- Slower (B times slower)
- Higher memory usage
- More complex

### When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| Real-time chat | Greedy decoding or beam_width=2 |
| General text generation | Beam search (beam_width=4) |
| High-quality content | Beam search (beam_width=8) |
| Creative writing | Beam search + higher temperature |
| Factual answers | Beam search + lower temperature |

## Testing

Comprehensive tests in `tests/beam_search_test.rs`:

- ✅ Configuration creation and builder pattern
- ✅ Beam hypothesis creation and scoring
- ✅ Beam search state initialization
- ✅ Beam expansion and pruning
- ✅ Completion marking
- ✅ Best hypothesis selection
- ✅ Entropy computation
- ✅ Adaptive beam width adjustment
- ✅ Beam width bounds enforcement
- ✅ Integration with LLM
- ✅ Temperature sampling

Run tests:
```bash
cargo test --test beam_search_test
```

## Future Enhancements

1. **Diverse Beam Search**: Penalize similar beams to increase diversity
2. **Length Normalization**: Better scoring for different length sequences
3. **Coverage Penalty**: Penalize repetition
4. **Constrained Beam Search**: Force certain tokens to appear
5. **Nucleus Sampling**: Combine with top-p sampling
6. **Beam Groups**: Multiple independent beam groups

## References

1. **Beam Search**: Graves (2012), "Sequence Transduction with Recurrent Neural Networks"
2. **Diverse Beam Search**: Vijayakumar et al. (2016), "Diverse Beam Search"
3. **Length Normalization**: Wu et al. (2016), "Google's Neural Machine Translation System"

---

**Phase 4 Secondary Objective Complete**: RustGPT now has adaptive beam search for high-quality text generation! 🎉

