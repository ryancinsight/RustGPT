# Shared Hypernetwork Implementation Guide

## Overview

This guide shows how to implement hypernetwork sharing across layers if you want to experiment with it.

## Current vs Shared Architecture

### Current (No Sharing)
```rust
struct HyperMixerBlock {
    token_mixing: TokenMixingMLP,  // Contains its own Hypernetwork
    channel_mixing: ChannelMixingMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

// Each block has independent hypernetwork
// 3 blocks = 3 hypernetworks = 1,040,400 params
```

### Shared Architecture
```rust
struct HyperMixerBlock {
    token_mixing: TokenMixingMLP,  // References shared Hypernetwork
    channel_mixing: ChannelMixingMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

// All blocks share one hypernetwork
// 3 blocks = 1 hypernetwork = 346,800 params
```

---

## Implementation Approach

### Option 1: Rc<RefCell<>> (Simplest)

Use Rust's reference counting to share the hypernetwork.

**Pros**: 
- Simple to implement
- No architectural changes needed
- Easy to serialize/deserialize

**Cons**:
- Runtime overhead (RefCell borrow checking)
- Not thread-safe (but we're not using threads)

### Option 2: Arc<Mutex<>> (Thread-safe)

Use atomic reference counting with mutex.

**Pros**:
- Thread-safe
- Can parallelize if needed

**Cons**:
- More overhead
- Overkill for single-threaded training

### Option 3: External Storage (Most Flexible)

Store hypernetwork outside the blocks, pass references.

**Pros**:
- Most flexible
- Best performance
- Clear ownership

**Cons**:
- Requires more architectural changes
- More complex to implement

---

## Recommended Implementation (Option 1)

### Step 1: Modify TokenMixingMLP

**File**: `src/token_mixing.rs`

```rust
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct TokenMixingMLP {
    /// Shared hypernetwork (wrapped in Rc<RefCell<>>)
    hypernetwork: Rc<RefCell<Hypernetwork>>,
    
    hidden_dim: usize,
    max_seq_len: usize,
    embedding_dim: usize,
    
    // ... rest of fields
}

impl TokenMixingMLP {
    /// Create with a shared hypernetwork
    pub fn new_shared(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        shared_hypernetwork: Rc<RefCell<Hypernetwork>>,
    ) -> Self {
        Self {
            hypernetwork: shared_hypernetwork,
            hidden_dim,
            max_seq_len,
            embedding_dim,
            cached_input: None,
            cached_mean_pooled: None,
            cached_generated_weights: None,
            cached_transposed_input: None,
        }
    }
    
    /// Keep existing constructor for backward compatibility
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
    ) -> Self {
        let output_size = (max_seq_len * hidden_dim) + hidden_dim + (hidden_dim * max_seq_len) + max_seq_len;
        let hypernetwork = Hypernetwork::new(embedding_dim, hypernetwork_hidden_dim, output_size);
        
        Self {
            hypernetwork: Rc::new(RefCell::new(hypernetwork)),
            hidden_dim,
            max_seq_len,
            embedding_dim,
            cached_input: None,
            cached_mean_pooled: None,
            cached_generated_weights: None,
            cached_transposed_input: None,
        }
    }
}

impl Layer for TokenMixingMLP {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // ... existing code, but change:
        // let generated_weights = self.hypernetwork.forward(&mean_pooled);
        // to:
        let generated_weights = self.hypernetwork.borrow_mut().forward(&mean_pooled);
        
        // ... rest of forward pass
    }
    
    fn compute_gradients(&self, _input: &Array2<f32>, output_grads: &Array2<f32>) 
        -> (Array2<f32>, Vec<Array2<f32>>) {
        // ... existing code, but change:
        // let (grad_mean_pooled, hyper_param_grads) = 
        //     self.hypernetwork.compute_gradients(&grad_generated_weights);
        // to:
        let (grad_mean_pooled, hyper_param_grads) = 
            self.hypernetwork.borrow().compute_gradients(&grad_generated_weights);
        
        // ... rest of compute_gradients
    }
    
    fn apply_gradients(&mut self, param_grads: &[Array2<f32>], lr: f32) {
        if param_grads.len() >= 4 {
            self.hypernetwork.borrow_mut().apply_gradients(param_grads, lr);
        }
    }
    
    fn parameters(&self) -> usize {
        // When shared, only count once (handled at model level)
        self.hypernetwork.borrow().parameters()
    }
}
```

### Step 2: Modify HyperMixerBlock

**File**: `src/hypermixer.rs`

```rust
use std::rc::Rc;
use std::cell::RefCell;

impl HyperMixerBlock {
    /// Create with shared hypernetwork
    pub fn new_shared(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        shared_hypernetwork: Rc<RefCell<Hypernetwork>>,
    ) -> Self {
        let token_mixing_hidden_dim = embedding_dim / 2;
        
        Self {
            token_mixing: TokenMixingMLP::new_shared(
                embedding_dim,
                token_mixing_hidden_dim,
                max_seq_len,
                shared_hypernetwork,
            ),
            channel_mixing: ChannelMixingMLP::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
    
    // Keep existing constructor for backward compatibility
    pub fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: usize,
    ) -> Self {
        // ... existing code
    }
}
```

### Step 3: Modify ModelBuilder

**File**: `src/model_builder.rs`

```rust
use std::rc::Rc;
use std::cell::RefCell;

fn build_hypermixer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    let hypernetwork_hidden_dim = config.get_hypernetwork_hidden_dim();
    
    // Create ONE shared hypernetwork
    let token_mixing_hidden_dim = config.embedding_dim / 2;
    let output_size = (config.max_seq_len * token_mixing_hidden_dim) 
        + token_mixing_hidden_dim 
        + (token_mixing_hidden_dim * config.max_seq_len) 
        + config.max_seq_len;
    
    let shared_hypernetwork = Rc::new(RefCell::new(
        Hypernetwork::new(config.embedding_dim, hypernetwork_hidden_dim, output_size)
    ));
    
    // Create all layers with the shared hypernetwork
    for _ in 0..config.num_layers {
        layers.push(LayerEnum::HyperMixerBlock(Box::new(
            HyperMixerBlock::new_shared(
                config.embedding_dim,
                config.hidden_dim,
                config.max_seq_len,
                Rc::clone(&shared_hypernetwork),  // Share the same hypernetwork
            ),
        )));
    }
}
```

### Step 4: Handle Serialization

**Challenge**: `Rc<RefCell<>>` doesn't implement `Serialize` by default.

**Solution**: Custom serialization or use a different approach.

```rust
// Option A: Don't serialize shared state (reconstruct on load)
// Option B: Serialize hypernetwork separately and reconstruct sharing
// Option C: Use a different sharing mechanism
```

---

## Simpler Alternative: Configuration Flag

Add a configuration option to enable/disable sharing without changing the core architecture.

**File**: `src/model_config.rs`

```rust
pub struct ModelConfig {
    // ... existing fields
    pub share_hypernetwork: bool,  // NEW
}

impl ModelConfig {
    pub fn hypermixer(
        embedding_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        max_seq_len: usize,
        hypernetwork_hidden_dim: Option<usize>,
        share_hypernetwork: bool,  // NEW parameter
    ) -> Self {
        Self {
            architecture: ArchitectureType::HyperMixer,
            embedding_dim,
            hidden_dim,
            num_layers,
            max_seq_len,
            hypernetwork_hidden_dim,
            share_hypernetwork,  // NEW
        }
    }
}
```

Then in `model_builder.rs`:
```rust
fn build_hypermixer_layers(layers: &mut Vec<LayerEnum>, config: &ModelConfig) {
    if config.share_hypernetwork {
        // Use shared implementation
        build_hypermixer_layers_shared(layers, config);
    } else {
        // Use independent implementation (current)
        build_hypermixer_layers_independent(layers, config);
    }
}
```

---

## Testing Strategy

### 1. Verify Parameter Count
```rust
#[test]
fn test_shared_hypernetwork_params() {
    let config = ModelConfig::hypermixer(128, 256, 3, 80, Some(32), true);
    let network = build_network(&config, 533);
    
    // Should be ~640K instead of ~1.39M
    assert!(network.parameters() < 700_000);
}
```

### 2. Verify Gradient Flow
```rust
#[test]
fn test_shared_hypernetwork_gradients() {
    // Create model with shared hypernetwork
    // Run forward pass
    // Run backward pass
    // Verify hypernetwork parameters are updated
    // Verify all layers see the updates
}
```

### 3. Compare Training
```rust
// Train two models:
// 1. With shared hypernetwork
// 2. Without shared hypernetwork
// Compare final loss and convergence speed
```

---

## Expected Results

### Parameter Count
```
Before: 1,386,917 params
After:  ~640,000 params (54% reduction)
```

### Training Impact
- **Best case**: Similar performance (regularization helps)
- **Expected**: 5-10% worse performance
- **Worst case**: 15-20% worse performance

### When to Use
- ✅ Simple tasks (classification, short sequences)
- ✅ Limited memory/compute
- ✅ Want regularization
- ❌ Complex tasks (long-form generation, reasoning)
- ❌ Need maximum expressiveness

---

## Recommendation

**Don't implement sharing yet.** Instead:

1. **First**: Try reducing `max_seq_len` and `token_mixing_hidden_dim` (2 lines of code)
2. **Measure**: Check if parameter count is acceptable
3. **If still too large**: Then consider implementing sharing
4. **Test**: Compare performance on your specific task

The simple parameter reductions give you 54% savings with zero risk. Only implement sharing if you need even more reduction and are willing to accept potential performance degradation.

